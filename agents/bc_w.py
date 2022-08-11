# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from torch.distributions import Distribution, Normal

EPS = 1e-8

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, device, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value+self.epsilon) + self.epsilon
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Sampling without reparameterization.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size(), device=self.device),
                torch.ones(self.normal_std.size(), device=self.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


# Implicit Policy
class Actor(nn.Module):
    """
    Gaussian Policy
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 hidden_sizes=[256, 256],
                 layer_norm=False):
        super(Actor, self).__init__()

        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim
        for next_size in hidden_sizes:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc_mean = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

        self.max_action = max_action
        self.device = device

    def forward(self, state):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        tanh_normal = TanhNormal(mean, std, self.device)
        action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.max_action

        return action, log_prob

    def log_prob(self, state, action):
        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        tanh_normal = TanhNormal(mean, std, self.device)
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def sample(self,
               state,
               reparameterize=False,
               deterministic=False):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            tanh_normal = TanhNormal(mean, std, self.device)
            if reparameterize:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
            action = action * self.max_action

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=300):
        super(Critic, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


class BC_W(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 hidden_sizes=[256,256],
                 w_gamma=5.0,
                 c_iter=3,
                 ):

        self.actor = Actor(state_dim, action_dim, max_action,
                           device=device,
                           hidden_sizes=hidden_sizes,
                           layer_norm=False).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

        self.w_gamma = w_gamma
        self.c_iter = c_iter

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state, deterministic=True)
        return action.cpu().data.numpy().flatten()

    def optimize_c(self, state, action_b):
        action_pi = self.actor.sample(state).detach()

        batch_size = state.shape[0]
        alpha = torch.rand((batch_size, 1)).to(self.device)
        a_intpl = (action_pi + alpha * (action_b - action_pi)).requires_grad_(True)
        grads = torch.autograd.grad(outputs=self.critic(state, a_intpl).mean(), inputs=a_intpl, create_graph=True,
                                    only_inputs=True)[0]
        slope = (grads.square().sum(dim=-1) + EPS).sqrt()
        gradient_penalty = torch.max(slope - 1.0, torch.zeros_like(slope)).square().mean()

        logits_p = self.critic(state, action_pi)
        logits_b = self.critic(state, action_b)
        logits_diff = logits_p - logits_b
        critic_loss = - logits_diff.mean() + gradient_penalty * self.w_gamma

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def optimize_p(self, state, action_b):
        action_pi = self.actor.sample(state)
        logits_p = self.critic(state, action_pi)
        logits_b = self.critic(state, action_b)
        logits_diff = logits_p - logits_b

        # Actor Training
        actor_loss = logits_diff.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()


    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, _, _, _ = replay_buffer.sample(batch_size)

            critic_loss = self.optimize_c(state, action)
            actor_loss = self.optimize_p(state, action)

            for _ in range(self.c_iter - 1):
                state, action, _, _, _ = replay_buffer.sample(batch_size)
                critic_loss = self.optimize_c(state, action)

        logger.record_tabular('Actor Loss', actor_loss)
        logger.record_tabular('Critic Loss', critic_loss)

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
