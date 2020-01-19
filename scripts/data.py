"""
Local module for data generation and manipulation
"""

# Pytorch
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Manipulation
import math
import numpy as np
from os.path import join, exists
from os import mkdir

# Visualization
import matplotlib.pyplot as plt


def generate_gaussian_simulations(min_x, max_x, min_y, max_y, n_steps, step_size=1, std=0.1):
    """Generate a natural path where the next direction angle is sampled from a gaussian distribution
    
    Arguments:
        min_x {float} -- x minimum boundary
        max_x {float} -- x maximum boundary
        min_y {float} -- y minimum boundary
        max_y {float} -- y maximum boundary
        n_steps {int} -- the number of simulation steps
    
    Keyword Arguments:
        step_size {int} -- the distance between two successive positions (default: {1})
        std {float} -- the std from the gaussian distribution (default: {0.1})
    
    Returns:
        tensor -- n_steps x 2 tensor corresponding to the positions of the new path
    """
    assert min_x <= max_x
    assert min_y <= max_y
    
    # generate starting position
    start_x = min_x + (max_x - min_x) * torch.rand(1)
    start_y = min_y + (max_y - min_y) * torch.rand(1)
    
    # generate the angle changes
    cur_angle = 2 * np.pi * torch.rand(1) - np.pi
    deviations = 2 * np.pi * torch.empty(n_steps - 1).normal_(mean=0, std=std)
    
    positions = [[start_x, start_y]]
    cur_x, cur_y = start_x, start_y
    
    for dev in deviations:
        cur_angle = cur_angle + dev
        
        # put angle into (-pi, pi] range
        if cur_angle > np.pi:
            cur_angle.sub_(2 * np.pi)
        elif cur_angle <= -np.pi:
            cur_angle.add_(2 * np.pi)
            
        cur_x = cur_x + step_size * torch.cos(cur_angle)
        cur_y = cur_y + step_size * torch.sin(cur_angle)
        
            
        # handle positions crossing the boundaries (reflected like a mirror)
        if cur_x < min_x:
            cur_angle = torch.sign(cur_angle) * np.pi - cur_angle
            cur_x = min_x + (min_x - cur_x)
        elif cur_x > max_x:
            cur_angle = torch.sign(cur_angle) * np.pi - cur_angle
            cur_x = max_x + (max_x - cur_x)
            
        if cur_y < min_y:
            cur_angle = -cur_angle
            cur_y = min_y + (min_y - cur_y)
        elif cur_y > max_y:
            cur_angle = -cur_angle
            cur_y = max_y + (max_y - cur_y)
        
        # add the new position
        positions.append([cur_x, cur_y])
        
    return torch.FloatTensor(positions)

def noise(size, std):
    """generate a tensor with values sampled from a normal distribution
    
    Arguments:
        size {int or tuple} -- the size of the tensor
        std {float} -- the standard deviation of the distribution
    
    Returns:
        tensor -- the tensor with the sampled values
    """
    return torch.Tensor(size).normal_(std=std)

def rssi(dist, rssi_t=60, rssi_n=2, rssi_std=5, **kwargs):
    """Construct RSSI features from a distance tensor
    
    Arguments:
        dist {tensor} -- a tensor of distances
    
    Keyword Arguments:
        rssi_t {int} -- the RSSI at a distance of 1 from the anchor (default: {60})
        rssi_n {int} -- the path loss exponent (default: {2})
        rssi_std {float} -- the standard deviation of the noise (default: {5})
    
    Returns:
        tensor -- tensor matching the size of the input tensor with RSSI measurements
    """
    return rssi_t - 10 * rssi_n * torch.log10(dist) + noise(dist.size(), rssi_std)

def rtt(dist, rtt_a=1.1, rtt_b=10, rtt_std=4, **kwargs):
    """Construct RTT features from a distance tensor
    
    Arguments:
        dist {tensor} -- a tensor of distances
    
    Keyword Arguments:
        rtt_a {float} -- the slope (default: {1.1})
        rtt_b {float} -- the offset (default: {10})
        rtt_std {float} -- the standard deviation of the noise (default: {4})
    
    Returns:
        tensor -- tensor matching the size of the input tensor with RTT measurements
    """
    return rtt_a * dist + rtt_b + noise(dist.size(), rtt_std)

def generate_features(dist, n_rssi_measurements=10, **kwargs):
    """Generate RSSI (mean and std) and RTT features from a tensor of distances. Parameters
        specific to the RSSI and RTT signals can be passed in as keyword arguments (kwargs) 
    
    Arguments:
        dist {tensor} -- a tensor of distances
    
    Keyword Arguments:
        n_rssi_measurements {int} -- the number of rssi measurements to sampled before reporting
            the mean and the std (default: {10})
    
    Returns:
        tensor -- tensor with the concatenation of mean(rssi), std(rssi) and rtt in the last dimension
    """
    rssis = torch.stack([rssi(dist, **kwargs) for i in range(n_rssi_measurements)])
    rtts = rtt(dist, **kwargs)

    features = torch.cat((
        rssis.mean(0), # mean(rssi)
        rssis.std(0), # std(rssi)
        rtts # rtt
    ), -1)

    return features

class SimulationData(Dataset):
    """A dataset of simulations"""

    def __init__(self, simulations, anchors_pos, **kwargs):
        """Constructs a dataset of simulations and generate the features. 
        
        Arguments:
            simulations {tensor} -- Ns x S x 2 tensor corresponding to Ns simulations of S steps each
            anchors_pos {[type]} -- Na x 2 tensor corresponding to the positions of the anchors
        """
        self.simulations = simulations
        self.anchors_pos = anchors_pos
        self.n_simulations, self.n_steps, _ = self.simulations.size()
        
        # generate signals from the anchors distances
        self.distances = self.simulations[..., None, :].sub(self.anchors_pos).pow(2).sum(dim=3).sqrt() + 1e-2
        self.features = generate_features(self.distances, **kwargs)
        self.feature_dim = self.features.size(-1)
        
    def __len__(self):
        return self.simulations.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.simulations[idx]
        
    def plot_simulation(self, positions, ax=plt):
        """Plot positions in the simulation environment
        
        Arguments:
            positions {tensor} -- a tensor of positions
        
        Keyword Arguments:
            ax -- the pyplot canvas in which we plot the positions (default: {plt})
        """
        ax.grid(True)

        # plot positions
        ax.plot(*positions.T)

        # highlight anchor positions
        for x, y in self.anchors_pos:
            ax.plot(x, y, '.', c='r', markersize=15)

        # highlight starting and ending positions
        ax.plot(*positions[0], '.', c='green', markersize=15)
        ax.plot(*positions[-1], '.', c='orange', markersize=15)
    
    def get_random_loaders(self, split_ratio, batch_size):
        """Split the data into train and test sets and return iterators of each
        
        Arguments:
            split_ratio {float} -- the ratio of training simulations
            batch_size {int} -- the number of simulations per iteration
        
        Returns:
            (iterator, iterator) -- the train and test iterators
        """
        # generate a random permutation of indices
        indices = torch.randperm(self.n_simulations)

        # split the data into train and test sets
        split = int(split_ratio * self.n_simulations)
        train_indices = indices[:split]
        test_indices = indices[split:]

        # generate iterators
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        
        return train_loader, test_loader

def rand_uniform(*size, min_=0, max_=1):
    """Create a tensor of the given size with values sampled from a uniform distribution
    
    Keyword Arguments:
        min_ {int} -- the maximum possible value (exclusive) (default: {0})
        max_ {int} -- the minimum possible value (default: {1})
    
    Returns:
        [type] -- [description]
    """
    return torch.rand(size) * (max_ - min_) + min_

def generate_calibration_data(n_sims, n_samples, x_bounds, y_bounds):
    """Generate horizontal only and vertical only simulations 
    
    Arguments:
        n_sims {int} -- the number of simulations per axis
        n_samples {int} -- the number of steps per simulation
        x_bounds {(int, int)} -- the x boundaries
        y_bounds {(int, int)} -- the y boundaries
    
    Returns:
        (tensor, tensor) -- two n_sims x n_samples x 2 tensors corresponding to the horizontal
                            and vertical simulations
    """
    min_x, max_x = x_bounds
    min_y, max_y = y_bounds

    # generate horizontal calibration data
    ys = rand_uniform(n_sims, min_=min_y, max_=max_y).unsqueeze(1).repeat(1, n_samples)
    xs = torch.linspace(min_x, max_x, n_samples).unsqueeze(0).repeat(n_sims, 1)
    horizontal_sims = torch.stack((xs, ys), 2)
    
    # generate vertical calibration data
    xs = rand_uniform(n_sims, min_=min_x, max_=max_x).unsqueeze(1).repeat(1, n_samples)
    ys = torch.linspace(min_y, max_y, n_samples).unsqueeze(0).repeat(n_sims, 1)
    vertical_sims = torch.stack((xs, ys), 2)
    
    return horizontal_sims, vertical_sims


##### DEPRECATED #####

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
        new_dir_x = (1 - momentum) * dir_x + momentum * dir_prev_x
        new_dir_y = (1 - momentum) * dir_y + momentum * dir_prev_y
        dir_prev_x = new_dir_x
        dir_prev_y = new_dir_y

        # bound the obtained position
        pos_x = torch.clamp(pos_x + new_dir_x, min_x, max_x)
        pos_y = torch.clamp(pos_y + new_dir_y, min_y, max_y)

        positions.append([pos_x, pos_y])

    positions = torch.FloatTensor(positions)

    return positions

RESULTS_PATH = 'results/'
MODEL_FILE_NAME = 'model.pt'
TRAIN_LOSS_FILE_NAME = 'train_loss.pt'
TEST_LOSS_FILE_NAME = 'test_loss.pt'

def save_result(result, results_name):
    if not exists(RESULTS_PATH):
        mkdir(RESULTS_PATH)
        
    
    path = join(RESULTS_PATH, results_name)
    if not exists(path):
        mkdir(path)
        
    torch.save(result['train_loss'], join(path, TRAIN_LOSS_FILE_NAME))
    torch.save(result['test_loss'], join(path, TEST_LOSS_FILE_NAME))
    torch.save(result['model'].state_dict(), join(path, MODEL_FILE_NAME))

def load_result(results_name, model):
    if not exists(RESULTS_PATH):
        raise FileNotFoundError('There is no model directory')
        
    path = join(RESULTS_PATH, results_name)
    if not exists(path):
        raise FileNotFoundError(f'The file {results_name} doesn\'t exist')
        
    train_loss = torch.load(join(path, TRAIN_LOSS_FILE_NAME))
    test_loss = torch.load(join(path, TEST_LOSS_FILE_NAME))
    model_state_dict = torch.load(join(path, MODEL_FILE_NAME))
    
    model.load_state_dict(model_state_dict)

    result = {
        'model': model,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

    return result