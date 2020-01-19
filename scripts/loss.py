"""
Local module for personalized losses
"""

import torch
from torch.nn.functional import mse_loss
import numpy as np


def recovery_loss(source, decoded, **kwargs):
    """The MSE loss between the two endpoints of the autoencoder
    
    Arguments:
        source {tensor} -- the input (features)
        decoded {tensor} -- the output of the decoder (input recovery)
    
    Returns:
        [torch.float] -- the loss
    """
    return mse_loss(decoded, source)

def fixed_points_loss(target, encoded, **kwargs):
    """The MSE loss between the first points of the embeddings and the first
       point of the simulation
    
    Arguments:
        target {tensor} -- the target (simulation points)
        encoded {tensor} -- the output of the encoder (embeddings)
    
    Returns:
        [torch.float] -- the loss
    """
    mask = [0]
    return mse_loss(encoded[mask], target[mask])

MAX_DISTANCE_THRESHOLD = 1
def subsequent_distances_loss(encoded, **kwargs):
    """Loss that penalizes big successive distances
    
    Arguments:
        encoded {tensor} -- the output of the encoder (embeddings)
    
    Returns:
        [torch.float] -- the loss
    """
    subsequent_distances = encoded[:-1].sub(encoded[1:]).pow(2).sum(dim=1)
    mask = subsequent_distances.gt(MAX_DISTANCE_THRESHOLD).float()
    return (subsequent_distances * mask).mean()

def imu_loss(target, encoded, **kwargs):
    """The MSE loss between the first order finite differences of the embeddings and
       simulations
    
    Arguments:
        target {tensor} -- the target (simulation points)
        encoded {tensor} -- the output of the encoder (embeddings)
    
    Returns:
        [torch.float] -- the loss
    """
    dirs_target = target[:-1].sub(target[1:])
    dirs_encoded = encoded[:-1].sub(encoded[1:])
    return mse_loss(dirs_target, dirs_encoded)

def second_order_loss(source, target, encoded, decoded, **kwargs):
    """The MSE loss between the second order central finite differences of the embeddings
       and simulations
    
    Arguments:
        target {tensor} -- the target (simulation points)
        encoded {tensor} -- the output of the encoder (embeddings)
    
    Returns:
        [torch.float] -- the loss
    """
    target_so = target[:-2] - 2 * target[1:-1] + target[2:]
    encoded_so = encoded[:-2] - 2 * encoded[1:-1] + encoded[2:]
    return mse_loss(target_so, encoded_so)

def calibration_loss(source, target, encoded, decoded, horizontal=True, **kwargs):
    """The MSE loss between the successive points in the coordinate that is not supposed
       to change
    
    Arguments:
        encoded {tensor} -- the output of the encoder (embeddings)
    
    Keyword Arguments:
        horizontal {bool} -- if the simulation is horizontal (default: {True})
    
    Returns:
        [torch.float] -- the loss
    """
    x = encoded[:, 1] if horizontal else encoded[:, 0]
    return mse_loss(x[1:], x[:-1])