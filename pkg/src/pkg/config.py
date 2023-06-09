from omegaconf import OmegaConf
import numpy as np

ac_config = OmegaConf.create({
    'device': 'cpu',
    'gamma': 0.98,

    'buffer_limit': 50000,
    'batch_size': 128,

    'lidar_dim': 405,
    'act_dim': 2,
    'hidden_dims': [512, 256, 256, 64, 32],
    'act_const': {'v_min': 1.0, 'v_max': 20.0, 's_min': -0.4, 's_max': 0.4},

    'lr_actor': 0.0005,
    'lr_critic': 0.001,
    'anneal_lr': True,

    'tau': 0.005
})

exp_config = OmegaConf.create({
    'epochs': 10000,
    'max_epi_len': 10000,
    'model_save_epoch': 100,
    'update_start_size': 20000,
    'update_iters': 10,
    'poses': [[0.60070, -0.2753, 1.5707]],
    'fast_lap': 10000,
    'eval_epochs': 10
})
