import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP_Unet, MLP, Tanh_MLP


class BC(object):
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
                 lr=2e-4,
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

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

        # Logging
        logger.record_tabular('Diffusion BC Loss', loss.item())

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
