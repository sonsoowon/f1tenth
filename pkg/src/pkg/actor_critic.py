import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GaussianActor(nn.Module):
    def __init__(self, lidar_dim, action_dim, act_const):
        super().__init__()
        self.mu_net = nn.Sequential(
            layer_init(nn.Linear(lidar_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, action_dim)),
            nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(0.5 * torch.ones(action_dim))
        self.act_const = act_const

    def forward(self, lidar):
        pi = self._distribution(lidar)
        action = self._scale_action_range(pi.sample())
        return action, pi

    def _distribution(self, lidar):
        mu = self.mu_net(lidar)
        std = torch.exp(self.log_std) # std transformation: soft plus
        return Normal(mu, std)

    def _scale_action_range(self, act):
        # (-1, 1) 인 범위의 action 값 확장
        scale = torch.tensor([[self.act_const.s_max - self.act_const.s_min, self.act_const.v_max - self.act_const.v_min]])
        bias = torch.tensor([[self.act_const.s_max + self.act_const.s_min, self.act_const.v_max + self.act_const.v_min]])

        return 0.5 * (scale * act + bias)


class Critic(nn.Module):
    def __init__(self, lidar_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            layer_init(nn.Linear(lidar_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
            nn.Tanh()
        )

    def forward(self, lidar):
        return torch.squeeze(self.v_net(lidar), -1) # Critical to ensure v has right shape
