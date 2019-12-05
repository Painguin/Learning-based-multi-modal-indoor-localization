"""
Local module for personalized losses
"""

import torch
from torch.nn.functional import mse_loss


def recovery_loss(source, target, encoded, decoded):
    return mse_loss(decoded, source)

def fixed_points_loss(source, target, encoded, decoded):
    mask = [0]
    return mse_loss(encoded[mask], target[mask])

MAX_DISTANCE_THRESHOLD = 1
def subsequent_distances_loss(source, target, encoded, decoded):
    subsequent_distances = encoded[:-1].sub(encoded[1:]).pow(2).sum(dim=1)
    mask = subsequent_distances.gt(MAX_DISTANCE_THRESHOLD).float()
    return (subsequent_distances * mask).mean()

def angle_differences_loss(source, target, encoded, decoded):
    directions = encoded[1:].sub(encoded[:-1])
    angles = torch.atan2(directions[:, 1], directions[:, 0] + 1e-5)
    angle_differences = angles[:-1].sub(angles[1:])
    return angle_differences.pow(2).mean()