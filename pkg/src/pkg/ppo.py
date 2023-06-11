import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os, sys

from utils import make_env, get_tracker, save_model, save_graph, VectorEnv
from buffer import PPOBuffer, discount_cumsum
from core import preprocess_lidar
from actor_critic import GaussianActor, Critic
from config import map_easy3_config, exp_config, ac_config, ppo_config

class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        lidar_dim, act_dim = config.lidar_dim, config.action_dim
        self.actor = GaussianActor(lidar_dim, act_dim, config.act_const)
        self.critic = Critic(lidar_dim)
        
    def get_value(self, lidar):
        return self.critic(lidar)

    def get_action_and_value(self, lidar, action=None):
        sampled_action, probs = self.actor(lidar)
        if action is None:
            action = sampled_action
        
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(lidar)


class PPO:
    def __init__(self, ac_config, exp_config, ppo_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = ActorCritic(ac_config).to(self.device)

        """
        PPO Config
        """
        self.clip_ratio = ppo_config.clip_ratio
        self.vf_coef = ppo_config.vf_coef
        self.ent_coef = ppo_config.ent_coef
        self.norm_adv = ppo_config.norm_adv
        self.max_grad_norm = ppo_config.max_grad_norm
        self.lr = ppo_config.lr
        self.anneal_lr = ppo_config.anneal_lr
        self.minibatch_size = ppo_config.minibatch_size
        self.gamma, self.gae_lambda = ppo_config.gamma, ppo_config.gae_lambda

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        """
        Experiment config
        """
        self.num_envs = exp_config.num_envs
        self.num_rollout_steps = exp_config.num_rollout_steps
        self.num_updates = exp_config.total_timesteps // (self.num_rollout_steps * self.num_envs)
        self.save_positions = np.arange(0, self.num_updates // 10 + self.num_updates, self.num_updates // 10)
        self.update_iters = exp_config.update_iters
        self.print_interval = exp_config.print_interval

        self.buffer = PPOBuffer(self.num_rollout_steps, self.num_envs, self.device,
                                ac_config, gamma=self.gamma, lam=self.gae_lambda)

    def train(self, config, writer, global_logger):
        envs = VectorEnv(config, self.num_envs)
        global_step, start_time = 0, time.time()

        next_obs, _, _, _ = envs.reset()
        next_done = torch.zeros(self.num_envs).to(self.device)

        # envs.render()

        fast_lap = config.fast_lap
        for update in range(self.num_updates + 1):
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.num_updates
                lr_now = frac * self.lr
                self.optimizer.param_groups[0]["lr"] = lr_now

            epi_return, epi_len = torch.zeros(self.num_envs), torch.zeros(self.num_envs)
            for step in range(self.num_rollout_steps):
                global_step += 1 * self.num_envs
                next_lidar = preprocess_lidar(next_obs)

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_lidar)

                next_obs, reward, done, _ = envs.step(action.cpu().numpy())
                reward = torch.tensor(reward.astype('float32')).to(self.device)
                done = torch.tensor(done.astype('bool')).to(self.device)

                self.buffer.store(next_lidar, action, reward, value, logprob, next_done)

                next_done = done
                epi_return += reward
                epi_len += 1

                for i in range(self.num_envs):
                    # Skip the envs that are not done
                    if next_done[i]:
                        epi_return[i], epi_len[i] = 0.0, 0

            lap_time = round(next_obs[0]['lap_times'][0], 3)
            print("update :{}, return : {:.1f}, time: {:.1f}, epi_len: {}".format(update, epi_return[0], lap_time, epi_len[0]))
            if int(next_obs[0]['lap_counts'][0]) == 2 and fast_lap > lap_time:
                save_model(config.map_name, self.agent, update, fast=True)
                fast_lap = lap_time

            writer.add_scalar("charts/episodic_return", epi_return[0], global_step)
            writer.add_scalar("charts/episodic_length", epi_len[0], global_step)
            writer.add_scalar("charts/episodic_time", lap_time, global_step)
            global_logger.train_episodic_return.append(epi_return[0])
            global_logger.episodic_return_steps.append(epi_len[0])
            global_logger.train_episodic_time.append(lap_time)

            # Save model
            if update in self.save_positions:
                save_model(config.map_name, self.agent, update)

            with torch.no_grad():
                next_lidar = preprocess_lidar(next_obs)
                next_value = self.agent.get_value(next_lidar).reshape(1, -1)
                self.buffer.calculate_adv_ret(next_value)

            if update in self.save_positions:
                save_model(config.map_name, self.agent, update)


            data = self.buffer.get()
            self.update(data, writer, global_logger, global_step, start_time)

        envs.close()
        writer.close()
        print("Training is finished")

    def recompute_adv(self, values, rewards, dones):
        deltas = rewards + self.gamma * values * dones - values
        advantages = np.transpose([discount_cumsum(delta.detach().numpy(), self.gamma * self.gae_lambda)
                                   for delta in deltas.reshape(self.num_envs, -1)])
        return torch.Tensor(advantages).to(self.device).reshape(-1)

    def update(self,data, writer, global_logger, global_step, start_time):
        b_lidars = data['lidars']
        b_logprobs = data['logprobs']
        b_actions = data['actions']
        b_advantages = data['advantages']
        b_returns = data['returns']
        b_values = data['values']
        b_rewards = data['rewards']
        b_dones = data['dones']

        # ! Optimizing the policy and value network
        self.agent.train()
        batch_size = self.num_rollout_steps * self.num_envs
        b_inds = np.arange(batch_size)  # batch_size = num_rollout_steps
        mb_size = self.minibatch_size
        for _ in range(self.update_iters):
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = self.agent.get_action_and_value(b_lidars[mb_inds], b_actions[mb_inds])
                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = ratio * mb_advantages
                pg_loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                v_loss = ((new_value - b_returns[mb_inds]) ** 2).mean()
                b_values[mb_inds] = new_value # update value

                entropy_loss = -entropy.mean()

                total_loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
            #b_advantages = self.recompute_adv(b_values, b_rewards, b_dones)

        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        global_logger.value_loss.append(v_loss.item())
        global_logger.policy_loss.append(pg_loss.item())
        global_logger.entropy_loss.append(entropy_loss.item())

def test(ac_config, env_config, param_path):
    env = make_env(env_config.map_name)
    agent = ActorCritic(ac_config)
    agent.load_state_dict(torch.load(param_path))

    for t in range(10):
        obs, reward, done, _ = env.reset(poses=np.array(env_config.poses))
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

    env.close()


if __name__ == "__main__":

    #global_logger, writer = get_tracker(map_name)

    #ppo = PPO(ac_config, exp_config, ppo_config)
    #ppo.train(map_easy3_config, writer, global_logger)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    param = "{}\checkpoints\map_easy3\ActorCritic-50.pt".format(current_dir)
    test(ac_config, map_easy3_config, param)

