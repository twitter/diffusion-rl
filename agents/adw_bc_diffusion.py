import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP_Unet, MLP, Tanh_MLP


def expectile_reg_loss(diff, quantile=0.7):
    weight = torch.where(diff > 0, quantile, (1 - quantile))
    return weight * (diff ** 2)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        hidden_dim = 256
        self.model = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state):
        return self.model(state)


class ADW_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 model_type='MLP',
                 beta_schedule='linear',
                 n_timesteps=100,
                 quantile=0.7,
                 temp=3.0,
                 lr=3e-4,
                 ):

        if model_type == 'MLP':
            self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'MLP_Unet':
            self.model = MLP_Unet(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'Tanh_MLP':
            self.model = Tanh_MLP(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.value_fun = Value(state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_fun.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

        self.quantile = quantile
        self.temp = temp

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Value Function Training
            with torch.no_grad():
                q1, q2 = self.critic_target(state, action)
                q = torch.min(q1, q2)  # Clipped Double Q-learning
            v = self.value_fun(state)
            value_loss = expectile_reg_loss(q - v, self.quantile).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Critic Training
            current_q1, current_q2 = self.critic(state, action)
            target_q = (reward + not_done * self.discount * self.value_fun(next_state)).detach()
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Policy Training
            v = self.value_fun(state)
            weight = torch.exp((q - v) / q.abs().mean() * self.temp).clamp_max(100.0).detach()
            loss = self.actor.loss(action, state, weight)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        logger.record_tabular('ADW_BC Loss', loss.item())
        logger.record_tabular('Critic Loss', critic_loss.item())

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
