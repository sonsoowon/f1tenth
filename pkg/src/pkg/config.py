from omegaconf import OmegaConf
import numpy as np

ac_config = OmegaConf.create({
    'lidar_dim': 405,
    'action_dim': 2,
    'act_const': {'v_min': 3.0, 'v_max': 20.0, 's_min': -0.4, 's_max': 0.4}
})

ppo_config = OmegaConf.create({
    "gamma": 0.99,
    "gae_lambda": 0.9,

    "clip_ratio": 0.25,
    "vf_coef": 0.5,
    "ent_coef": 0.01,

    "lr": 0.0001,
    "anneal_lr": True,

    "max_grad_norm": 2.0,
    "norm_adv": True,

    "minibatch_size": 128
})

map_easy3_config = OmegaConf.create({
    "map_name": "map_easy3",
    "poses": [[0.60070, -0.2753, 1.5707]],
    "fast_lap": 5000
})

exp_config = OmegaConf.create({
    "seed": 0, # environment seed
    "num_envs": 2, # the number of environments for parallel training
    "num_eval": 10, # the number of evaluations
    "num_rollout_steps": 5000, # the number of policy rollout steps
    "total_timesteps": 5000000, # total number of frames,
    "update_iters": 10,
    "print_interval": 10
})
