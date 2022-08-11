# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb


class EasyBlock(nn.Module):
    def __init__(self, input_dim, t_embed_dim, s_embed_dim):
        super(EasyBlock, self).__init__()
        self.block_1 = nn.Linear(input_dim, input_dim)

        self.block_2 = nn.Sequential(nn.Mish(),
                                     nn.Linear(input_dim, input_dim),
                                     nn.Mish())

        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, input_dim),
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(s_embed_dim, input_dim),
        )

    def forward(self, x, t, s):
        out = self.block_1(x) + self.time_mlp(t) + self.state_mlp(s)
        out = self.block_2(out)
        return out


class MLP_Unet(nn.Module):
    """
    MLP-Unet Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 h_dim=256,
                 t_dim=16):

        super(MLP_Unet, self).__init__()
        self.device = device

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, h_dim),
            nn.Mish())
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.Mish(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Mish(),
        )

        self.down_1 = EasyBlock(h_dim, t_dim, state_dim)
        self.down_2 = nn.Sequential(nn.Linear(h_dim, h_dim // 2), nn.Mish())
        self.down_3 = EasyBlock(h_dim // 2, t_dim, state_dim)
        self.down_4 = nn.Sequential(nn.Linear(h_dim // 2, h_dim // 4), nn.Mish())

        self.mid_block = EasyBlock(h_dim // 4, t_dim, state_dim)

        self.up_1 = EasyBlock(h_dim // 4, t_dim, state_dim)
        self.up_2 = nn.Sequential(nn.Linear(h_dim // 4, h_dim // 2), nn.Mish())
        self.up_3 = EasyBlock(h_dim // 2, t_dim, state_dim)
        self.up_4 = nn.Sequential(nn.Linear(h_dim // 2, h_dim), nn.Mish())

        self.final_layer = nn.Linear(h_dim, action_dim)

    def forward(self, x, time, state):

        x = self.action_mlp(x)
        t = self.time_mlp(time)
        s = self.state_mlp(state)

        x = self.down_1(x, t, s)
        x = self.down_2(x)
        x = self.down_3(x, t, s)
        x = self.down_4(x)

        x = self.mid_block(x, t, s)

        x = self.up_1(x, t, s)
        x = self.up_2(x)
        x = self.up_3(x, t, s)
        x = self.up_4(x)

        return self.final_layer(x)


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class Tanh_MLP(nn.Module):
    """
    MLP + Tanh Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 t_dim=16):

        super(Tanh_MLP, self).__init__()
        self.device = device
        self.max_action = max_action

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Sequential(nn.Linear(256, action_dim), nn.Tanh())

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x) * self.max_action

