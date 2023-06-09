import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import ReplayBuffer
from torch.distributions import Normal

from utils import make_env, save_model, Tracker
from core import mlp, preprocess_lidar
from config import ac_config, exp_config


class GaussianActor(nn.Module):
    def __init__(self, lidar_dim, act_dim, hidden_dims, act_const):
        super().__init__()
        self.mu_net = mlp(lidar_dim, act_dim, hidden_dims, output_activation=nn.Tanh)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act_const = act_const

    def forward(self, lidar):
        pi = self._distribution(lidar)
        return torch.round(pi.sample(), decimals=1)

    def _distribution(self, lidar):
        mu = self.mu_net(lidar)
        mu = self._scale_action_range(mu)
        std = torch.exp(self.log_std)

        return Normal(mu, std)

    def _scale_action_range(self, act):
        # (-1, 1) 인 범위의 action 값 확장
        scale = torch.tensor([[self.act_const.s_max - self.act_const.s_min, self.act_const.v_max - self.act_const.v_min]])
        bias = torch.tensor([[self.act_const.s_max + self.act_const.s_min, self.act_const.v_max + self.act_const.v_min]])

        return 0.5 * (scale * act + bias)


class Critic(nn.Module):
    def __init__(self, lidar_dim, act_dim, hidden_dims):
        super().__init__()
        self.v_net = mlp(lidar_dim + act_dim, 1, hidden_dims)

    def forward(self, x):
        return self.v_net(x)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to(config.device)
        self.gamma = config.gamma

        self.memory = ReplayBuffer(config.buffer_limit)
        self.batch_size = config.batch_size

        lidar_dim, act_dim, hidden_dims = config.lidar_dim, config.act_dim, config.hidden_dims
        self.actor = GaussianActor(lidar_dim, act_dim, hidden_dims, config.act_const)
        self.critic = Critic(lidar_dim, act_dim, hidden_dims)
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # parameter for soft update
        self.tau = config.tau

    def update(self):
        # randomly sample from replay buffer
        lidars, actions, rewards, next_lidars, dones = self.memory.sample(self.batch_size)

        # compute critic loss & update
        target_q_values = rewards + self.gamma * \
                          self.critic_target(torch.cat([next_lidars, self.actor_target(next_lidars)], dim=-1)) * dones
        current_q_values = self.critic(torch.cat([lidars, actions], dim=-1))
        critic_loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # compute pg loss & update
        actor_loss = -self.critic(torch.cat([lidars, self.actor(lidars)], dim=-1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

        return critic_loss, actor_loss

    def _soft_update(self, net, net_target, tau):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            # mix the target and current parameters with the ratio of (1 - tau) : (tau)
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train(map_name, tracker, ac_config, exp_config):
    """
    exp_config : {epochs, model_save_epoch, update_start_size, update_iters, poses, fast_lap}
    """
    env = make_env(map_name)
    agent = ActorCritic(ac_config)

    epochs = exp_config['epochs']
    max_epi_len = exp_config['max_epi_len']
    model_save_epoch = exp_config['model_save_epoch']
    update_start_size = exp_config['update_start_size']
    update_iters = exp_config['update_iters']
    poses = exp_config['poses']
    fast_lap = exp_config['fast_lap']

    total_step = 0
    for epoch in range(exp_config.epochs):
        obs, reward, done, _ = env.reset(poses=np.array(poses))
        lidar = preprocess_lidar(obs)
        done = False

        if ac_config.anneal_lr:
            frac = 1.0 - (epoch - 1.0) / epochs
            lr_ac_now, lr_c_now = frac * ac_config.lr_actor, frac * ac_config.lr_critic
            agent.actor_opt.param_groups[0]["lr"] = lr_ac_now
            agent.critic_opt.param_groups[0]["lr"] = lr_c_now

        epi_return, epi_len = 0.0, 0
        while not done:
            total_step += 1
            epi_len += 1

            action = agent.actor(lidar).detach().numpy()
            obs, reward, done, _ = env.step(action)

            next_lidar = preprocess_lidar(obs)
            done_mask = 0.0 if done else 1.0
            agent.memory.put((lidar, np.squeeze(action), reward, next_lidar, done_mask))

            # state transition
            lidar = next_lidar

            # record reward
            epi_return += reward
            env.render(mode='human_fast')

            if epi_len == max_epi_len:
                epi_return += agent.critic(torch.cat([lidar, agent.actor(lidar)]))
                print("time out")
                break


        lap_time = round(obs['lap_times'][0], 3)
        if int(obs['lap_counts'][0]) == 2 and fast_lap > lap_time:
            save_model(map_name, agent, epoch, fast=True)
            fast_lap = lap_time
        tracker.record_train(total_step=total_step,
                             epi_info={'epi_return': epi_return, 'epi_time': lap_time, 'epi_len': epi_len})

        if epoch % model_save_epoch == 0 and epoch > 0:
            save_model(map_name, agent, epoch)

        if epoch % 10 == 0:
            print("epoch :{}, return : {:.1f}, time: {:.1f}, epi_len: {}".format(epoch, epi_return, lap_time, epi_len))

        # enough memory
        if agent.memory.size() > update_start_size:
            # off-line training
            for i in range(update_iters):
                v_loss, pg_loss = agent.update()
                tracker.record_train(total_step=total_step, losses={'v_loss': v_loss, 'pg_loss': pg_loss})

    env.close()


def test(map_name, tracker, ac_config, exp_config, param_path):
    env = make_env(map_name)
    agent = ActorCritic(ac_config)
    agent.load_state_dict(torch.load(param_path))

    for t in range(exp_config['eval_epochs']):
        obs, reward, done, _ = env.reset(poses=np.array(exp_config['poses']))
        env.render()
        lidar = preprocess_lidar(obs)
        done = False

        epi_return = 0.0
        while not done:
            action = agent.actor(lidar).detach().numpy()
            obs, reward, done, _ = env.step(action)
            lidar = preprocess_lidar(obs)
            epi_return += reward
            env.render(mode='human_fast')

        tracker.record_test(epi_return, round(obs['lap_times'][0], 3))
    env.close()


if __name__ == "__main__":
    map_name = "map_easy3"
    tracker = Tracker(map_name)
    train(map_name, tracker, ac_config, exp_config)

    param_path = "./checkpoints/map_easy3/ActorCritic-1000.pt"
    test(map_name, tracker, ac_config, exp_config, param_path)

