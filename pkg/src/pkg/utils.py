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
import matplotlib.ticker as ticker
import seaborn as sns

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

class Tracker:
    def __init__(self, map_name):
        log_path = str(Path("./runs") / Path(map_name))
        self.global_logger = GlobalLogger(
            total_steps=[],
            episodic_return_steps=[],
            train_episodic_return=[],
            train_episodic_time=[],
            test_episodic_return=[],
            test_episodic_time=[],
            policy_loss=[],
            value_loss=[]
        )
        self.global_logger.log_path = log_path
        self.writer = SummaryWriter(log_path)

    def record_train(self, total_step, epi_info=None, losses=None):
        if epi_info is not None:
            epi_return, epi_time, epi_len = epi_info['epi_return'], epi_info['epi_time'], epi_info['epi_len']
            self.writer.add_scalar("charts/episodic_return", epi_return, total_step)
            self.writer.add_scalar("charts/episodic_time", epi_time, total_step)
            self.writer.add_scalar("charts/episodic_length", epi_len, total_step)
            self.global_logger.train_episodic_return.append(epi_return)
            self.global_logger.train_episodic_time.append(epi_time)
            self.global_logger.episodic_return_steps.append(epi_len)

        if losses is not None:
            value_loss, policy_loss = losses['v_loss'], losses['pg_loss']
            self.writer.add_scalar("losses/value_loss", value_loss.item(), total_step)
            self.writer.add_scalar("losses/policy_loss", policy_loss.item(), total_step)
            self.global_logger.value_loss.append(value_loss.item())
            self.global_logger.policy_loss.append(policy_loss.item())

    def record_test(self, epi_return, epi_time):
        self.global_logger.test_episodic_return.append(epi_return)
        self.global_logger.test_episodic_time.append(epi_time)

def save_model(map_name, actor_critic, update, fast=False):
    ckpt_path = Path("./checkpoints") / Path(map_name)
    if not ckpt_path.exists():
        os.makedirs(ckpt_path)
    model_name = f"{type(actor_critic).__name__}-{update}-fast" if fast else f"{type(actor_critic).__name__}-{update}"
    model_path = ckpt_path / Path(f"{model_name}.pt")
    torch.save(actor_critic.state_dict(), str(model_path))
    print(f"model saved to {model_path}")


def make_env(race_track):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    return gym.make('f110_gym:f110-v0',
                    map="{}/maps/{}".format(current_dir, race_track),
                    map_ext=".png", num_agents=1)


def save_graph(tracker, map_name, mode='train'):
    global_logger = tracker.global_logger
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
