# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA
from utils import pytorch_util as ptu


def identity(x):
    return x


class ParallelizedLayerMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            input_dim,
            output_dim,
            w_std_value=1.0,
            b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity

        self.layer_norm = layer_norm

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                ptu.orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]

        dim = len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs)

        return torch.min(preds, dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError


class ED_PCQ(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=0.1,
                 model_type='MLP',
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 num_qs=50,
                 num_q_layers=3,
                 q_eta=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.num_qs = num_qs
        self.q_eta = q_eta
        self.critic = ParallelizedEnsembleFlattenMLP(ensemble_size=num_qs,
                                                     hidden_sizes=[256] * num_q_layers,
                                                     input_size=state_dim + action_dim,
                                                     output_size=1,
                                                     layer_norm=None,
                                                     ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, replay_buffer, iterations, batch_size=100):

        for step in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_qs = self.critic(state, action)

            if not self.max_q_backup:
                next_action = self.ema_model(next_state)
                target_q = self.critic_target.sample(next_state, next_action)
            else:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q = self.critic_target.sample(next_state_rpt, next_action_rpt)
                target_q = target_q.view(batch_size, 10).max(dim=1, keepdim=True)[0]

            target_q = (reward + not_done * self.discount * target_q).detach().unsqueeze(0)

            critic_loss = F.mse_loss(current_qs, target_q, reduction='none')
            critic_loss = critic_loss.mean(dim=(1, 2)).sum()

            if self.q_eta > 0:
                state_tile = state.unsqueeze(0).repeat(self.num_qs, 1, 1)
                action_tile = action.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
                qs_preds_tile = self.critic(state_tile, action_tile)
                qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), action_tile, retain_graph=True,
                                                     create_graph=True)
                qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
                qs_pred_grads = qs_pred_grads.transpose(0, 1)

                qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
                masks = torch.eye(self.num_qs, device=self.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
                qs_pred_grads = (1 - masks) * qs_pred_grads
                grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)

                critic_loss += self.q_eta * grad_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)

            new_action = self.actor(state)
            q_new_action = self.critic.sample(state, new_action)
            lmbda = self.eta / q_new_action.abs().mean().detach()
            q_loss = - lmbda * q_new_action.mean()

            self.actor_optimizer.zero_grad()
            bc_loss.backward()
            q_loss.backward()
            self.actor_optimizer.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

        # Logging
        logger.record_tabular('BC Loss', bc_loss.item())
        logger.record_tabular('QL Loss', q_loss.item())
        logger.record_tabular('Critic Loss', critic_loss.item())
        logger.record_tabular('ED Loss', grad_loss.item())
        logger.record_tabular('Target_Q Mean', target_q.mean().item())

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.sample(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
