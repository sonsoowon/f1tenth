import numpy as np
import torch
import scipy.signal
import collections
import random


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting with the environment
    """

    def __init__(self, size, num_envs, device, dim_info, gamma=0.99, lam=0.95):
        self.lidar_dim, self.action_dim = dim_info.lidar_dim, dim_info.action_dim
        self.device = device

        self.lidars = torch.zeros((size, num_envs) + (self.lidar_dim,)).to(device)
        self.actions = torch.zeros((size, num_envs) + (self.action_dim,)).to(device)
        self.logprobs = torch.zeros((size, num_envs)).to(device)
        self.rewards = torch.zeros((size, num_envs)).to(device)
        self.dones = torch.zeros((size, num_envs)).to(device)
        self.values = torch.zeros((size, num_envs)).to(device)
        self.advantages = torch.zeros((size, num_envs)).to(device)
        self.returns = torch.zeros((size, num_envs)).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, lidar, act, rew, val, logp, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.lidars[self.ptr] = lidar
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.dones[self.ptr] = done
        self.ptr += 1

    def get(self):
        """
        Call this at the end of an epoch to get all data from the buffer.
        Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr = 0

        return dict(
            lidars=self.lidars.reshape((-1,) + (self.lidar_dim,)),
            actions=self.actions.reshape((-1,) + (self.action_dim,)),
            values=self.values.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=self.advantages.reshape(-1),
            logprobs=self.logprobs.reshape(-1),
            rewards=self.rewards.reshape(-1),
            dones=self.dones.reshape(-1)
        )

    def calculate_adv_ret(self, last_value):
        next_values = torch.cat([self.values, last_value], dim=0)
        deltas = self.rewards + self.gamma * next_values[1:] * self.dones - self.values
        advantages = np.transpose([discount_cumsum(delta.numpy(), self.gamma * self.lam) for delta in torch.transpose(deltas, 0, 1)])
        self.advantages = torch.Tensor(advantages).to(self.device)
        self.returns = self.advantages + self.values


# replay buffer
class ReplayBuffer:
    """
    A buffer for DQN, DDQN, DDPG, SAC ... off-policy algorithm
    """
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, next_s_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, next_s, done = transition
            s_lst.append(s.tolist())
            a_lst.append(a.tolist())
            r_lst.append([r])
            next_s_lst.append(next_s.tolist())
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.Tensor(s_lst), torch.Tensor(a_lst), torch.Tensor(r_lst), torch.Tensor(next_s_lst), torch.Tensor(
            done_mask_lst)

    def size(self):
        return len(self.buffer)