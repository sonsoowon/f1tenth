import numpy as np
import torch
import torch.nn as nn


def mlp(state_dim, action_dim, hidden_dims, activation=nn.ReLU, output_activation=nn.Identity):
    layers = [nn.Linear(state_dim, hidden_dims[0]), activation()]
    for cur_h_dim, next_h_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
        layers += [nn.Linear(cur_h_dim, next_h_dim), activation()]
    layers += [nn.Linear(hidden_dims[-1], action_dim), output_activation()]
    return nn.Sequential(*layers)


def preprocess_lidar(obs, bubble_masked=False, bubble_radius=30):
    """ Preprocess the LiDAR scan array. Expert implementation includes:
        1.Setting each value to the mean over some window
        2.Rejecting high values (eg. > 3m)
    """
    ranges = obs['scans'][0]

    # we won't use the LiDAR data from directly behind us
    proc_ranges = np.array(ranges[135:-135:2])

    # sets each value to the mean over a given window
    proc_ranges = np.convolve(proc_ranges, np.ones(3), 'same') / 3
    proc_ranges = np.clip(proc_ranges, 0, 3000000)

    if bubble_masked:
        closest = np.argmin(proc_ranges)
        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = np.max([0, closest - bubble_radius])
        max_index = np.min([closest + bubble_radius, len(proc_ranges) - 1])
        proc_ranges[min_index:max_index] = 0

    return torch.as_tensor(proc_ranges, dtype=torch.float32)
