# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
NEGATIVE_SLOPE = 1. / 10.


class NormalNoise(object):
    def __init__(self, device, mean=0., std=1.):
        self.mean = mean
        self.std = std
        self.device = device

    def sample_noise(self, shape, dtype=None, requires_grad=False):
        return torch.randn(size=shape, dtype=dtype, device=self.device, requires_grad=requires_grad) * self.std + self.mean


class ImplicitPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, noise, noise_dim, device):
        # noise_dim : dimension of noise for "concat" method
        super(ImplicitPolicy, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + int(noise_dim), self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action

        self.noise = noise
        self.noise_dim = int(noise_dim)

        self.device = device

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        # state.shape = (batch_size, state_dim)

        epsilon = self.noise.sample_noise(shape=(state.shape[0], self.noise_dim)).clamp(-3, 3)
        state = torch.cat([state, epsilon], 1)      # dim = (state.shape[0], state_dim + noise_dim)

        a = F.leaky_relu(self.l1(state), negative_slope=NEGATIVE_SLOPE)
        a = F.leaky_relu(self.l2(a), negative_slope=NEGATIVE_SLOPE)

        return self.l3(a)

    def sample_multiple_actions(self, state, num_action=10, std=-1.):
        # num_action : number of actions to sample from policy for each state

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        batch_size = state.shape[0]
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        if std <= 0:
            state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        else:   # std > 0
            if num_action == 1:
                noises = torch.normal(torch.zeros_like(state), torch.ones_like(state))  # B * state_dim
                state = (state + (std * noises).clamp(-0.05, 0.05)).to(self.device)  # B x state_dim
            else:   # num_action > 1
                state_noise = state.unsqueeze(1).repeat(1, num_action, 1)   # B * num_action * state_dim
                noises = torch.normal(torch.zeros_like(state_noise), torch.ones_like(state_noise))  # B * num_q_samples * state_dim
                state_noise = state_noise + (std * noises).clamp(-0.05, 0.05)  # N x num_action x state_dim
                state = torch.cat((state_noise, state.unsqueeze(1)), dim=1).view((batch_size * (num_action+1)), -1).to(self.device)  # (B * num_action) x state_dim
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        return state, self.forward(state)

    def sample(self, state):
        return self.forward(state)


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class BC_GAN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 ):
        noise_dim = min(action_dim, 10)
        self.noise = NormalNoise(device=device, mean=0.0, std=1.0)
        self.actor = ImplicitPolicy(state_dim, action_dim, max_action, self.noise, noise_dim, device).to(
            device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.4, 0.999))

        self.discriminator = Discriminator(state_dim=state_dim, action_dim=action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.4, 0.999))

        self.adversarial_loss = torch.nn.BCELoss()

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.g_iter = 2

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            state_repeat, action_samples = self.actor.sample_multiple_actions(state, num_action=5, std=3e-4)
            true_samples = torch.cat([state, action], 1)
            fake_samples = torch.cat([state_repeat, action_samples], 1)

            fake_labels = torch.zeros(fake_samples.size(0), 1, device=self.device)
            real_labels = torch.rand(size=(true_samples.size(0), 1), device=self.device) * (1.0 - 0.80) + 0.80

            real_loss = self.adversarial_loss(self.discriminator(true_samples), real_labels)
            fake_loss = self.adversarial_loss(self.discriminator(fake_samples.detach()), fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if it % self.g_iter == 0:
                generator_loss = self.adversarial_loss(self.discriminator(fake_samples),
                                                       torch.ones(fake_samples.size(0), 1, device=self.device))
                self.actor_optimizer.zero_grad()
                generator_loss.backward()
                self.actor_optimizer.step()

        logger.record_tabular('Generator Loss', generator_loss.cpu().data.numpy())
        logger.record_tabular('Discriminator Loss', discriminator_loss.cpu().data.numpy())

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
