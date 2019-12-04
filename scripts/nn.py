"""
Local module for neural network functions
"""

import torch
from torch import nn
from torch.optim import Adam

import numpy as np

def create_autoencoder_model(input_dim, hid_dim, bot_dim):
    encoder = nn.Sequential(
        nn.Linear(input_dim, hid_dim), nn.ReLU(),
        nn.Linear(hid_dim, bot_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(bot_dim, hid_dim), nn.ReLU(),
        nn.Linear(hid_dim, input_dim)
    )

    model = nn.Sequential(
        encoder,
        decoder
    )
    
    return model

def train_encoder(model, train_loader, test_loader, nb_epochs, loss_function=nn.MSELoss()):
    """Train a model"""

    optimizer = Adam(model.parameters())
    train_losses = []
    test_losses = []

    encoder, decoder = model

    for _ in range(nb_epochs):
        losses = []
        for batch_input, batch_target in train_loader:
            source = batch_input.view(-1, batch_input.size(-1))
            # target = batch_target.view(-1, batch_target.size(-1))
            encoded = encoder(source)
            decoded = decoder(encoded)

            loss = loss_function(decoded, source)

            optimizer.zero_grad() # set gradients to zero
            loss.backward() # compute gradient
            optimizer.step() # update weights
            losses.append(loss.item())
        train_losses.append(torch.Tensor(losses).mean().item())

        losses = []
        for batch_input, batch_target in test_loader:
            source = batch_input.view(-1, batch_input.size(-1))
            # target = batch_target.view(-1, batch_target.size(-1))
            encoded = encoder(source)
            decoded = decoder(encoded)

            loss = loss_function(decoded, source)
            losses.append(loss.item())
        test_losses.append(torch.Tensor(losses).mean().item())

    return train_losses, test_losses

def train_encoder_with_constraints(model, train_loader, test_loader, nb_epochs, loss_functions):
    """Train a model with constraints"""
    
    optimizer = Adam(model.parameters())
    
    train_losses = {}
    test_losses = {}
    for loss_function in loss_functions:
        train_losses[loss_function['function'].__name__] = []
        test_losses[loss_function['function'].__name__] = []
    train_losses['total_loss'] = []
    test_losses['total_loss'] = []
    
    encoder, decoder = model

    for _ in range(nb_epochs):
        losses = {}
        for loss_function in loss_functions:
            losses[loss_function['function'].__name__] = []
        losses['total_loss'] = []

        for batch_input, batch_target in train_loader:
            source = batch_input.view(-1, batch_input.size(-1))
            target = batch_target.view(-1, batch_target.size(-1))
            encoded = encoder(source)
            decoded = decoder(encoded)

            loss = 0
            for loss_function in loss_functions:
                f_loss = loss_function['weight'] * loss_function['function'](source, target, encoded, decoded)
                loss = loss + f_loss
                losses[loss_function['function'].__name__].append(f_loss.item())
            losses['total_loss'].append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for loss_function in loss_functions:
            train_losses[loss_function['function'].__name__].append(np.mean(losses[loss_function['function'].__name__]))
        train_losses['total_loss'].append(np.mean(losses['total_loss']))

        losses = {}
        for loss_function in loss_functions:
            losses[loss_function['function'].__name__] = []
        losses['total_loss'] = []

        for batch_input, batch_target in test_loader:
            source = batch_input.view(-1, batch_input.size(-1))
            target = batch_target.view(-1, batch_target.size(-1))
            encoded = encoder(source)
            decoded = decoder(encoded)

            loss = 0
            for loss_function in loss_functions:
                f_loss = loss_function['weight'] * loss_function['function'](source, target, encoded, decoded)
                loss = loss + f_loss
                losses[loss_function['function'].__name__].append(f_loss.item())
            losses['total_loss'].append(loss.item())

        for loss_function in loss_functions:
            test_losses[loss_function['function'].__name__].append(np.mean(losses[loss_function['function'].__name__]))
        test_losses['total_loss'].append(np.mean(losses['total_loss']))

    return train_losses, test_losses
