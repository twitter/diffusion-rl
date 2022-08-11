import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

LOG_SIG_MAX = 2.
LOG_SIG_MIN = -20.


class Generator(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 z_dim=64,
                 w_dim=256,
                 num_layers=2):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.device = device
        self.max_action = max_action

        self.network = nn.Sequential(nn.Linear(state_dim + action_dim, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, action_dim),
                                     nn.Tanh())

    def forward(self, x):

        z = torch.randn((x.shape[0], self.z_dim), device=self.device)
        w = torch.cat([x, z], dim=-1)
        a = self.network(w) * self.max_action

        return a

    def sample(self, x):
        return self.forward(x)


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, w_dim=256, num_layers=2):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(nn.Linear(state_dim + action_dim, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, 256),
                                     nn.LeakyReLU(0.1),
                                     nn.Linear(256, 1))

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)


class BC_GAN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4):

        self.actor = self.generator = Generator(state_dim,
                                                action_dim,
                                                max_action,
                                                device,
                                                z_dim=min(action_dim, 10)).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=2e-4)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)

        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def sample_action(self, state):
        if self.actor.training:
            self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        self.actor.train()

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """
            Generator Training
            """
            new_action = self.actor(state)
            gen_logits = self.discriminator(state, new_action)
            generator_loss = nn.functional.softplus(-gen_logits).mean()

            self.gen_optim.zero_grad()
            generator_loss.backward()
            self.gen_optim.step()

            """
            Discriminator Training
            """
            fake_labels = torch.zeros(state.shape[0], 1, device=self.device)
            real_labels = torch.ones(state.shape[0], 1, device=self.device)

            real_loss = self.adversarial_loss(self.discriminator(state, action), real_labels)
            fake_loss = self.adversarial_loss(self.discriminator(state, new_action.detach()), fake_labels)
            discriminator_loss = real_loss + fake_loss

            self.disc_optim.zero_grad()
            discriminator_loss.backward()
            self.disc_optim.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

        # Logging
        logger.record_tabular('Generator Loss', generator_loss.item())
        logger.record_tabular('Real Loss', real_loss.item())
        logger.record_tabular('Fake Loss', fake_loss.item())
        logger.record_tabular('Discriminator Loss', discriminator_loss.item())
