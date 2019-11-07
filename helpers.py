import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import math

import matplotlib.pyplot as plt

def generate_random_simulation(min_x, max_x, min_y, max_y, n_steps, momentum=0.9):
    # generate starting position
    start_x = min_x + (max_x - min_x) * torch.rand(1)
    start_y = min_y + (max_y - min_y) * torch.rand(1)
    
    # generate random directions
    random_angles = torch.rand(n_steps) * 2 * math.pi
    random_dir_xs = torch.cos(random_angles)
    random_dir_ys = torch.sin(random_angles)
    random_dirs = torch.cat((random_dir_xs[..., None], random_dir_ys[..., None]), 1)
    
    # generate resulting positions
    positions = []
    pos_x = start_x
    pos_y = start_y
    dir_prev_x, dir_prev_y = random_dirs[0]
    for dir_x, dir_y in random_dirs:
        # weighted average between the new and previous direction (to have some smoothness)
        newDir_x = (1 - momentum) * dir_x + momentum * dir_prev_x
        newDir_y = (1 - momentum) * dir_y + momentum * dir_prev_y
        dir_prev_x = newDir_x
        dir_prev_y = newDir_y

        # bound the obtained position
        pos_x = torch.clamp(pos_x + newDir_x, min_x, max_x)
        pos_y = torch.clamp(pos_y + newDir_y, min_y, max_y)

        positions.append([pos_x, pos_y])

    positions = torch.FloatTensor(positions)
    
    return positions

def noise(size, std):
    return torch.Tensor(size).normal_(std=std)

RSSI_T = 60
RSSI_N = 2
RSSI_STD = 5
def RSSI(dist):
    return RSSI_T - 10 * RSSI_N * torch.log10(dist) + noise(dist.size(), RSSI_STD)

RTT_A = 10
RTT_B = 10
RTT_STD = 4
def RTT(dist):
    return RTT_A * dist + RTT_B + noise(dist.size(), RTT_STD)

def generate_features(dist, n_samples):
    rssis = torch.stack([RSSI(dist) for i in range(n_samples)])
    rtts = RTT(dist)
    
    features = torch.cat((
        rssis.mean(0),
        rssis.std(0),
        rtts
    ), -1)
    
    return features

class SimulationData:
    def __init__(self, simulations, anchors_pos, momentum=0.9, n_rssi_measurements=10):
        self.simulations = simulations
        self.anchors_pos = anchors_pos
        self.n_simulations, self.n_steps, _ = self.simulations.size()
        
        # generate signals from the anchors distances
        self.distances = self.simulations[..., None, :].sub(self.anchors_pos).pow(2).sum(dim=3).sqrt() + 1e-2
        self.features = generate_features(self.distances, n_rssi_measurements)
        
    def __len__(self):
        return self.simulations.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.simulations[idx]
        
    def plot_simulation(self, positions, ax=plt):
        ax.grid(True)
        ax.axis('equal')
        ax.plot(*positions.T)

        for x, y in self.anchors_pos:
            ax.plot(x, y, '.', c='r', markersize=15)

        ax.plot(*positions[0], '*', c='green', markersize=15)
        ax.plot(*positions[-1], '*', c='orange', markersize=15);
    
    def get_random_loaders(self, split_ratio, batch_size):
        indices = torch.randperm(self.n_simulations)
        split = int(split_ratio * self.n_simulations)
        train_indices = indices[:split]
        test_indices = indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        
        return train_loader, test_loader