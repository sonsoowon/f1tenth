from pathlib import Path
import os
import sys
import gym
from dataclasses import dataclass
from typing import List
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

@dataclass
class GlobalLogger:
    total_steps: List
    episodic_return_steps: List
    train_episodic_return: List
    train_episodic_time: List
    test_episodic_return: List
    test_episodic_time: List
    policy_loss: List
    value_loss: List
    entropy_loss: List

def get_tracker(map_name):
    log_path = str(Path("./runs") / Path(map_name))
    global_logger = GlobalLogger(
        total_steps=[],
        episodic_return_steps=[],
        train_episodic_return=[],
        train_episodic_time=[],
        test_episodic_return=[],
        test_episodic_time=[],
        policy_loss=[],
        value_loss=[],
        entropy_loss=[]
    )
    global_logger.log_path = log_path
    return global_logger, SummaryWriter(log_path)


class VectorEnv:
    def __init__(self, config, n):
        self.envs = tuple(make_env(config.map_name) for _ in range(n))
        self.poses = np.array(config.poses)

    # Call this only once at the beginning of training:
    def reset(self):
        # obs, reward, done, _
        return tuple(np.transpose([env.reset(poses=self.poses) for env in self.envs]))

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        return_values = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(np.expand_dims(a, axis=0))
            if done:
                observation, _, _, _ = env.reset(poses=self.poses)
            return_values.append([observation, reward, done, info])
        return tuple(np.transpose(return_values))

    def render(self):
        self.envs[0].render(mode='human_fast')
    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()

def make_env(map_name):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    return gym.make('f110_gym:f110-v0',
                    map="{}/maps/{}".format(current_dir, map_name),
                    map_ext=".png", num_agents=1)

def save_model(map_name, actor_critic, update, fast=False):
    ckpt_path = Path("./checkpoints") / Path(map_name)
    if not ckpt_path.exists():
        os.makedirs(ckpt_path)
    model_name = f"{type(actor_critic).__name__}-{update}-fast" if fast else f"{type(actor_critic).__name__}-{update}"
    model_path = ckpt_path / Path(f"{model_name}.pt")
    torch.save(actor_critic.state_dict(), str(model_path))
    print(f"model saved to {model_path}")


def save_graph(global_logger, map_name, mode='train'):
    x = np.array(global_logger.episodic_return_steps)
    epi_return = np.array(global_logger.train_episodic_return).squeeze(1)
    epi_time = np.array(global_logger.train_episodic_time).squeeze(1)
    if mode == 'test':
        epi_return = np.array(global_logger.test_episodic_return).squeeze(1)
        epi_time = np.array(global_logger.test_episodic_time).squeeze(1)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x, epi_return)
    plt.title(f'{map_name}: {mode} episodic result')
    plt.ylabel('epi_return')
    plt.xticks(visible=False)

    plt.subplot(2, 1, 2)
    plt.plot(x, epi_time)
    plt.xlabel('global_step')
    plt.ylabel('epi_time')
    plt.tight_layout()

    save_path = Path(global_logger.log_path)
    file_name = Path(f"{map_name}_{mode}_episodic_result.jpg")
    file_path = save_path / file_name
    print(f"figure is saved to {str(file_path)}")
    plt.savefig(str(file_path), dpi=200)
