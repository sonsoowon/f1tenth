import numpy as np
import torch


def preprocess_lidar(obs, clip=False):
    """ Preprocess the LiDAR scan array. Expert implementation includes:
        1.Setting each value to the mean over some window
        2.Rejecting high values (eg. > 3m)
    """

    lidar_list = [one_obs['scans'][0] for one_obs in obs]
    # we won't use the LiDAR data from directly behind us

    proc_ranges_list = []
    for lidar in lidar_list:
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(np.array(lidar[135:-135:2]), np.ones(3), 'same') / 3
        proc_ranges = np.clip(proc_ranges, 0, 3000000)
        proc_ranges_list.append(proc_ranges)

    return torch.as_tensor(np.array(proc_ranges_list), dtype=torch.float32)