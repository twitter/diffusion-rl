# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

import torch.distributions as td

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_dim=256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def sample(self, state):
        return self.decode(state)


class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.randn_like(std_a)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) + \
            std_a.unsqueeze(1) * torch.FloatTensor(
            np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(self.device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)
        return log_pis

    def sample(self, state):
        return self.forward(state)


class BC_KL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 num_samples_match=10,
                 kl_type='backward'
                 ):

        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.actor = RegularActor(state_dim, action_dim, max_action, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

        self.num_samples_match = num_samples_match
        self.kl_type = kl_type

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def kl_loss(self, action, state):
        """ action in shape: (batch_size, num_samples_match, action_dim) """
        """ Backward KL: KL(behavior_policy(a|s) | current_policy(a|s)) """
        state_rep = state.unsqueeze(1).repeat(1, action.size(1), 1).view(-1, state.size(-1))
        action_reshape = action.view(-1, action.size(-1))
        action_log_pis = self.actor.log_pis(state=state_rep, raw_action=action_reshape)
        action_log_prob = action_log_pis.view(state.size(0), action.size(1))
        return (-action_log_prob).mean(1)

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            num_samples = self.num_samples_match
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            # actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_sample=num_samples)  # num)

            kl_loss = self.kl_loss(raw_sampled_actions, state).mean()
            self.actor_optimizer.zero_grad()
            kl_loss.backward()
            self.actor_optimizer.step()

        logger.record_tabular('VAE Loss', vae_loss.cpu().data.numpy())
        logger.record_tabular('KL Loss', kl_loss.cpu().data.numpy())

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
