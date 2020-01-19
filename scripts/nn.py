"""
Local module for neural network functions
"""

# Pytorch
import torch
from torch import nn
from torch import optim

# Manipulation
import numpy as np
import os

# Visualization
import tqdm
import seaborn
import matplotlib.pyplot as plt

def create_autoencoder_model(input_dim, hid_dim, bot_dim, activation=nn.ReLU()):
    """Creates a simple autoencoder with two layers and the specified dimensions and activation.
    
    Arguments:
        input_dim {int} -- the dimension of the input (and the output)
        hid_dim {int} -- the dimension of the hidden layer in both the encoder and decoder part
        bot_dim {int} -- the dimension of the bottleneck (middle of the autoencoder)
    
    Keyword Arguments:
        activation {function} -- the activation function to be used in the hidden layer (default: {nn.ReLU()})
    
    Returns:
        nn.Sequential -- the autoencoder
    """ 
    encoder = nn.Sequential(
        nn.Linear(input_dim, hid_dim), activation,
        nn.Linear(hid_dim, bot_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(bot_dim, hid_dim), activation,
        nn.Linear(hid_dim, input_dim)
    )

    model = nn.Sequential(
        encoder,
        decoder
    )

    return model

def create_conv_autoencoder_model(input_dim, hid_dim, bot_dim, kernel_size=9, activation=nn.ReLU()):
    """Creates an autoencoder with an outer convolution layer and a inner fully connected layer with the specified dimensions and activation.
    
    Arguments:
        input_dim {int} -- the dimension of the input (and the output)
        hid_dim {int} -- the dimension of the hidden layer in both the encoder and decoder part
        bot_dim {int} -- the dimension of the bottleneck (middle of the autoencoder)
    
    Keyword Arguments:
        kernel_size {int} -- the size of the filters. With a kernel_size of k, the filter will look at the (k - 1) / 2 previous and next rows to compute the current one (default: {9})
        activation {function} -- the activation function to be used in the hidden layer (default: {nn.ReLU()})
    
    Returns:
        nn.Sequential -- the autoencoder
    """    
    encoder = nn.Sequential(
        Conv(input_dim, hid_dim, kernel_size), activation,
        nn.Linear(hid_dim, bot_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(bot_dim, hid_dim), activation,
        ConvInv(hid_dim, input_dim, kernel_size)
    )

    model = nn.Sequential(
        encoder,
        decoder
    )

    return model

def create_autoencoder_model_2hls(input_dim, hid_dim1, hid_dim2, bot_dim, activation=nn.ReLU()):
    """Creates an autoencoder with three layers and the specified dimensions and activation.
    
    Arguments:
        input_dim {int} -- the dimension of the input (and the output)
        hid_dim1 {int} -- the dimension of the first hidden layer
        hid_dim2 {int} -- the dimension of the second hidden layer
        bot_dim {int} -- the dimension of the bottleneck (middle of the autoencoder)
    
    Keyword Arguments:
        activation {function} -- the activation function to be used in the hidden layer (default: {nn.ReLU()})
    
    Returns:
        nn.Sequential -- the autoencoder
    """ 

    encoder = nn.Sequential(
        nn.Linear(input_dim, hid_dim1), activation,
        nn.Linear(hid_dim1, hid_dim2), activation,
        nn.Linear(hid_dim2, bot_dim)
    )

    decoder = nn.Sequential(
        nn.Linear(bot_dim, hid_dim2), activation,
        nn.Linear(hid_dim2, hid_dim1), activation,
        nn.Linear(hid_dim1, input_dim)
    )

    model = nn.Sequential(
        encoder,
        decoder
    )

    return model

def train_encoder(model, train_loader, test_loader, nb_epochs, optimizer, loss_function=nn.MSELoss()):
    """Train a model"""

    train_losses = []
    test_losses = []

    encoder, decoder = model

    for _ in tqdm.tqdm(range(nb_epochs)):
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

def train_encoder_with_constraints(model, train_loader, test_loader, nb_epochs, optimizer, loss_functions):
    """Train a model with constraints"""

    crea

    train_losses = {}
    test_losses = {}
    for loss_function in loss_functions:
        train_losses[loss_function['function'].__name__] = []
        test_losses[loss_function['function'].__name__] = []
    train_losses['total_loss'] = []
    test_losses['total_loss'] = []

    encoder, decoder = model

    for e_idx in tqdm.tqdm(range(nb_epochs)):
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
                weight = loss_function['weight']
                if hasattr(weight, '__len__'):
                    weight = weight[e_idx]
                f_loss = weight * loss_function['function'](source, target, encoded, decoded)
                loss = loss + f_loss
                losses[loss_function['function'].__name__].append(f_loss.item())
            losses['total_loss'].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for loss_function in loss_functions:
            train_losses[loss_function['function'].__name__].append(
                np.mean(losses[loss_function['function'].__name__])
            )
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
                weight = loss_function['weight']
                if hasattr(weight, '__len__'):
                    weight = weight[e_idx]
                f_loss = weight * loss_function['function'](source, target, encoded, decoded)
                loss = loss + f_loss
                losses[loss_function['function'].__name__].append(f_loss.item())
            losses['total_loss'].append(loss.item())

        for loss_function in loss_functions:
            test_losses[loss_function['function'].__name__].append(
                np.mean(losses[loss_function['function'].__name__])
            )
        test_losses['total_loss'].append(np.mean(losses['total_loss']))

    Model()

    return train_losses, test_losses

class Model(object):
    """Encapsulates an autoencoder with training procedures and history"""

    def __init__(self, model):
        """Constructs an autoencoder with training procedures and history
        
        Arguments:
            model {Model} -- the torch model to be used
        """
        self.model = model
        self.encoder, self.decoder = model
        self.criterion = nn.MSELoss()

        self.train_loss = []
        self.test_loss = []

    def save(self, fpath):
        """Save a model to a file. Use "Model.load" to recover it.
        
        Arguments:
            fpath {string} -- the destination path
        """
        to_save = {
            'model': self.model,
            'state_dict': self.model.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
        }

        torch.save(to_save, fpath)

    @classmethod
    def load(cls, fpath):
        """Load a model from a file
        
        Arguments:
            fpath {string} -- the path of the file
        
        Returns:
            Model -- the loaded model
        """
        loaded = torch.load(fpath)

        model = loaded['model']
        model.load_state_dict(loaded['state_dict'])

        instance = cls(model)
        instance.train_loss = loaded['train_loss']
        instance.test_loss = loaded['test_loss']

        return instance

    def _update(self, loss, optimizer):
        """Update the weights corresponding to some loss using an optimizer
        
        Arguments:
            loss {torch.float32} -- the loss
            optimizer {[type]} -- the optimizer
        """
        optimizer.zero_grad() # set gradients to zero
        loss.backward() # compute gradient
        optimizer.step() # update weights

    def run_epoch(self, iterator, optimizer):
        """Run one epoch of the iterator. If optimizer is not None, then update the weights after each iteration
        
        Arguments:
            iterator -- the data through which we iterate
            optimizer {nn.optmizer} -- The optimizer 
        
        Returns:
            float -- the average loss
        """
        total_loss = 0

        for batch_input, batch_target in iterator:
            source = batch_input.view(-1, batch_input.size(-1))
            output = self.model(source)

            loss = self.criterion(output, source)
            if optimizer is not None:
                self._update(loss, optimizer)

            total_loss += loss.item()
        return total_loss / len(iterator)

    def train(self, iterator, optimizer):
        """Train the model through one epoch
        
        Arguments:
            iterator -- the data through which we iterate
            optimizer {nn.optimizer} -- The optimizer 
        
        Returns:
            float -- the average loss
        """
        return self.run_epoch(iterator, optimizer)

    def evaluate(self, iterator):
        """Evaluate the model through one epoch
        
        Arguments:
            iterator -- the data through which we iterate
        
        Returns:
            float -- the average loss
        """
        return self.run_epoch(iterator, None)

    def train_eval(self, train_iterator, test_iterator, optimizer, n_epochs):
        """Train the model through multiple epochs. Also evaluate with the test iterator after each epoch.
        
        Arguments:
            train_iterator -- the data used to train the model
            test_iterator -- the data used to evaluate the model
            optimizer {nn.optimizer} -- The optimizer
            n_epochs {int} -- The number of epochs
        """
        for _ in tqdm.tqdm(range(n_epochs)):
            epoch_train_loss = self.train(train_iterator, optimizer)
            self.train_loss.append(epoch_train_loss)

            epoch_test_loss = self.evaluate(test_iterator)
            self.test_loss.append(epoch_test_loss)

    def plot_loss(self, ylim=None):
        """Plot the loss from the training procedure
        
        Keyword Arguments:
            ylim {[int, int]} -- the boundaries of the visualization across the y axis (default: {None})
        """
        plt.plot(self.train_loss, c='blue')
        plt.plot(self.test_loss, c='orange')
        plt.legend(['train loss', 'test loss'])
        plt.ylim(ylim)
        plt.grid(True)

    def plot_embedding(self, data, indices):
        """Plot the embeddings of some given simulation data next to the real paths indexed by some given indices
        
        Arguments:
            data {data.SimulationData} -- The simulation data
            indices {int or range or list[int]} -- The indices of the data from which we want embeddings
        """
        if isinstance(indices, int):
            indices = [indices]

        source, target = data[indices]

        embeddings = self.encoder(source).detach()

        fig, axes = plt.subplots(nrows=len(indices), ncols=2, figsize=(8, 4 * len(indices)))
        for idx, (embedding, simulation) in enumerate(zip(embeddings, target)):
            if len(indices) > 1:
                ax1 = axes[idx, 0]
                ax2 = axes[idx, 1]
            else:
                # handle the case when we have only 1 row
                ax1 = axes[0]
                ax2 = axes[1]
            
            ax1.grid(True)
            ax1.axis('equal')
            ax1.plot(*embedding.T)
            ax1.plot(*embedding[0], '.', c='green', markersize=15)
            ax1.plot(*embedding[-1], '.', c='orange', markersize=15)
        
            data.plot_simulation(simulation, ax2)
        
        for ax, col in zip(axes[0] if len(indices) > 1 else axes, ['Encoder embedding', 'Simulation']):
            ax.set_title(col)
        
    def plot_embedding_on_simulation(self, data, indices):
        """Plot the embeddings of some given simulation data together with the real paths indexed by some given indices
        
        Arguments:
            data {data.SimulationData} -- The simulation data
            indices {int or range or list[int]} -- The indices of the data from which we want embeddings
        """
        if isinstance(indices, int):
            indices = [indices]

        source, target = data[indices]

        embeddings = self.encoder(source).detach()

        n_cols = 4
        n_rows = (len(indices) - 1) // n_cols + 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
        for idx, (embedding, simulation) in enumerate(zip(embeddings, target)):
            if len(indices) > n_cols:
                ax = axes[idx // n_cols, idx % n_cols]
            else:
                # handle the case when we have only 1 row
                ax = axes[idx % n_cols]
            
            data.plot_simulation(simulation, ax)
            data.plot_simulation(embedding, ax)


class ModelMultiLoss(Model):
    """Encapsulates an autoencoder with multiple loss functions together with training procedures and history"""

    def __init__(self, model, loss_functions):
        """Constructor an autoencoder with multiple custom loss functions together with training procedures and history
        
        Arguments:
            model {Model} -- the torch model to be used
            loss_functions {list[dict[function, name, weight]]} -- list of dictionaries each containing information for a loss function,
                each of them should look like {"function": function, "name": function_name, "weight": w}, the weight can be a list that
                specifies the weight for each epoch
        """
        self.model = model
        self.encoder, self.decoder = model
        self.loss_functions = loss_functions
        self.current_epoch = 0

        self.train_loss = {}
        self.test_loss = {}
        for func in loss_functions:
            self.train_loss[func['name']] = []
            self.test_loss[func['name']] = []

    def save(self, fpath):
        """Save a model to a file. Use "Model.load" to recover it.
        
        Arguments:
            fpath {string} -- the destination path
        """
        to_save = {
            'model': self.model,
            'state_dict': self.model.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'loss_functions': self.loss_functions,
            'epoch': self.current_epoch
        }

        torch.save(to_save, fpath)

    @classmethod
    def load(cls, fpath):
        """Load a model from a file
        
        Arguments:
            fpath {string} -- the path of the file
        
        Returns:
            Model -- the loaded model
        """
        loaded = torch.load(fpath)

        model = loaded['model']
        model.load_state_dict(loaded['state_dict'])
        loss_functions = loaded['loss_functions']

        instance = cls(model, loss_functions)
        instance.train_loss = loaded['train_loss']
        instance.test_loss = loaded['test_loss']
        instance.current_epoch = loaded['epoch']

        return instance

    def change_loss_functions(self, new_loss_functions):
        """Change the loss functions of an already trained model
        
        Arguments:
            new_loss_functions {list[dict[function, name, weight]]} -- the new loss functions that will replace the previous ones
        """
        previous_funcs = set(map(lambda x: x['name'], self.loss_functions))
        for new_func in new_loss_functions:
            if new_func['name'] not in previous_funcs:
                self.train_loss[new_func['name']] = [0] * self.current_epoch
                self.test_loss[new_func['name']] = [0] * self.current_epoch
        self.loss_functions = new_loss_functions

    def run_epoch(self, iterator, optimizer, e_idx=0):

        """Run one epoch of the iterator. If optimizer is not None, then update the weights after each iteration
        
        Arguments:
            iterator -- the data through which we iterate
            optimizer {nn.optmizer} -- The optimizer 
        
        Keyword Arguments:
            e_idx {int} -- the epoch index (to support loss weights that change with time) (default: {0})
        
        Returns:
            dict[name -> float] -- the average loss for each loss function
        """

        total_loss = {}
        for func in self.loss_functions:
            total_loss[func['name']] = 0

        for batch_input, batch_target in iterator:
            source = batch_input.view(-1, batch_input.size(-1))
            target = batch_target.view(-1, batch_target.size(-1))
            encoded = self.model[0](source)
            decoded = self.model[1](encoded)

            loss = 0
            for func in self.loss_functions:
                weight = func['weight']

                # if the weight value is a list, then get weight at the corresponding epoch
                if hasattr(weight, '__len__'):
                    weight = weight[e_idx]

                f_loss = weight * func['function'](source=source, target=target, encoded=encoded, decoded=decoded)
                loss = loss + f_loss
                total_loss[func['name']] += f_loss.item()

            if optimizer is not None:
                self._update(loss, optimizer)

        for func in self.loss_functions:
            total_loss[func['name']] /= len(iterator)

        return total_loss

    def train(self, iterator, optimizer, e_idx=0):
        """Train the model through one epoch
        
        Arguments:
            iterator -- the data through which we iterate
            optimizer {nn.optimizer} -- The optimizer 
        
        Keyword Arguments:
            e_idx {int} -- the epoch index (to support loss weights that change with time) (default: {0})
        
        Returns:
            dict[name -> float] -- the average loss for each loss function
        """
        return self.run_epoch(iterator, optimizer, e_idx)

    def evaluate(self, iterator, e_idx=0):
        """Evaluate the model through one epoch
        
        Arguments:
            iterator -- the data through which we iterate
        
        Keyword Arguments:
            e_idx {int} -- the epoch index (to support loss weights that change with time) (default: {0})
        
        Returns:
            dict[name -> float] -- the average loss for each loss function
        """

        return self.run_epoch(iterator, None, e_idx)

    def train_eval(self, train_iterator, test_iterator, optimizer, n_epochs):
        """Train the model through multiple epochs. Also evaluate with the test iterator after each epoch.
        
        Arguments:
            train_iterator -- the data used to train the model
            test_iterator -- the data used to evaluate the model
            optimizer {nn.optimizer} -- The optimizer
            n_epochs {int} -- The number of epochs
        """
        for e_idx in tqdm.tqdm(range(n_epochs)):
            epoch_train_loss = self.train(train_iterator, optimizer, e_idx)
            for func in self.loss_functions:
                self.train_loss[func['name']].append(epoch_train_loss[func['name']])

            epoch_test_loss = self.evaluate(test_iterator, e_idx)
            for func in self.loss_functions:
                self.test_loss[func['name']].append(epoch_test_loss[func['name']])
            self.current_epoch += 1
    
    def plot_loss(self, ylim=None):
        """Plot the losses from the training procedure
        
        Keyword Arguments:
            ylim {[int, int]} -- the boundaries of the visualization across the y axis (default: {None})
        """
        colors = seaborn.color_palette('Paired')
        loss_names = []
        
        plt.figure(figsize=(12,6))
        for idx, func in enumerate(self.loss_functions):
            func_name = func['name']
            plt.plot(self.train_loss[func_name], c=colors[2 * idx])
            plt.plot(self.test_loss[func_name], c=colors[2 * idx + 1])
            loss_names.append('train_' + func_name)
            loss_names.append('test_' + func_name)
        
        plt.legend(loss_names)
        plt.ylim(ylim)
        plt.grid(True)

class ModelWithCalibration(ModelMultiLoss):
    """Encapsulates an autoencoder with calibration, multiple loss functions together with training procedures and history"""

    def __init__(self, model, loss_functions, loss_functions_cal):
        """Constructor an autoencoder with calibration, multiple custom loss functions together with training procedures and history
        
        Arguments:
            model {Model} -- the torch model to be used
            loss_functions {list[dict[function, name, weight]]} -- list of dictionaries each containing information for a loss function,
                each of them should look like {"function": function, "name": function_name, "weight": w}, the weight can be a list that
                specifies the weight for each epoch
            loss_functions_cal {list[dict[function, name, weight]]} -- loss functions for the calibration phase. Should follow the same
                structure as loss_functions
        """
        super().__init__(model, loss_functions)
        self.loss_functions_cal = loss_functions_cal

        for func in loss_functions_cal:
            self.train_loss[func['name']] = []
            self.test_loss[func['name']] = []

        self.current_epoch = 0

    def save(self, fpath):
        """Save a model to a file. Use "Model.load" to recover it.
        
        Arguments:
            fpath {string} -- the destination path
        """
        to_save = {
            'model': self.model,
            'state_dict': self.model.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'loss_functions': self.loss_functions,
            'loss_functions_cal': self.loss_functions_cal,
            'epoch': self.current_epoch
        }

        torch.save(to_save, fpath)

    @classmethod
    def load(cls, fpath):
        """Load a model from a file
        
        Arguments:
            fpath {string} -- the path of the file
        
        Returns:
            Model -- the loaded model
        """
        loaded = torch.load(fpath)

        model = loaded['model']
        model.load_state_dict(loaded['state_dict'])
        loss_functions = loaded['loss_functions']
        loss_functions_cal = loaded['loss_functions_cal']

        instance = cls(model, loss_functions, loss_functions_cal)
        instance.train_loss = loaded['train_loss']
        instance.test_loss = loaded['test_loss']
        instance.current_epoch = loaded['epoch']

        return instance

    @classmethod
    def from_ModelMultiLoss(cls, instance, loss_functions, loss_functions_cal):
        """Transforms a model with multiple losses (ModelMultiLoss) into one with calibration (ModelWithCalibration)
        
        Arguments:
            instance {ModelMultiLoss} -- the ModelMultiLoss instance
            loss_functions {list[dict[function, name, weight]]} -- the loss functions for training the model
            loss_functions_cal {list[dict[function, name, weight]]} -- the loss functions for the calibration
        
        Returns:
            [type] -- [description]
        """
        new_instance = cls(instance.model, instance.loss_functions, [])
        new_instance.train_loss = instance.train_loss
        new_instance.test_loss = instance.test_loss
        new_instance.current_epoch = instance.current_epoch
        new_instance.change_loss_functions(loss_functions, loss_functions_cal)
        return new_instance

    def change_loss_functions(self, new_loss_functions, new_loss_functions_cal):
        """Change the loss functions of an already trained model
        
        Arguments:
            new_loss_functions {list[dict[function, name, weight]]} -- the new loss functions that will replace the previous ones
        """
        previous_funcs = set(map(lambda x: x['name'], self.loss_functions))
        for new_func in new_loss_functions:
            if new_func['name'] not in previous_funcs:
                self.train_loss[new_func['name']] = [0] * self.current_epoch
                self.test_loss[new_func['name']] = [0] * self.current_epoch

        previous_funcs_cal = set(map(lambda x: x['name'], self.loss_functions_cal))
        for new_func in new_loss_functions_cal:
            if new_func['name'] not in previous_funcs_cal:
                self.train_loss[new_func['name']] = [0] * self.current_epoch

        self.loss_functions = new_loss_functions
        self.loss_functions_cal = new_loss_functions_cal

    def calibrate(self, horizontal_cal, vertical_cal, optimizer_cal, e_idx=0):
        """Train the model with the calibration data.
        
        Arguments:
            horizontal_cal -- iterator over horizontal calibration data
            vertical_cal -- iterator over vertical calibration data
            optimizer_cal {nn.optimizer} -- optimizer used to update the weights
        
        Keyword Arguments:
            e_idx {int} -- the epoch index (to support loss weights that change with time) (default: {0})
        
        Returns:
            dict[name -> float] -- the average loss for each calibration loss function
        """
        total_loss = {}
        for func in self.loss_functions_cal:
            total_loss[func['name']] = 0

        for batch_input, batch_target in horizontal_cal:
            source = batch_input.view(-1, batch_input.size(-1))
            target = batch_target.view(-1, batch_target.size(-1))
            encoded = self.model[0](source)
            decoded = self.model[1](encoded)

            loss = 0
            for func in self.loss_functions_cal:
                weight = func['weight']
                if hasattr(weight, '__len__'):
                    weight = weight[e_idx]
                f_loss = weight * func['function'](source=source, target=target, encoded=encoded, decoded=decoded, horizontal=True)
                loss = loss + f_loss
                total_loss[func['name']] += f_loss.item()

            self._update(loss, optimizer_cal)

        for batch_input, batch_target in vertical_cal:
            source = batch_input.view(-1, batch_input.size(-1))
            target = batch_target.view(-1, batch_target.size(-1))
            encoded = self.model[0](source)
            decoded = self.model[1](encoded)

            loss = 0
            for func in self.loss_functions_cal:
                weight = func['weight']
                if hasattr(weight, '__len__'):
                    weight = weight[e_idx]
                f_loss = weight * func['function'](source=source, target=target, encoded=encoded, decoded=decoded, horizontal=False)
                loss = loss + f_loss
                total_loss[func['name']] += f_loss.item()

            self._update(loss, optimizer_cal)

        for func in self.loss_functions_cal:
            total_loss[func['name']] /= (len(horizontal_cal) + len(vertical_cal))

        return total_loss

    def train(self, iterator, optimizer, horizontal_cal, vertical_cal, optimizer_cal, e_idx=0):
        """Calibrate and then train the model.
        
        Arguments:
            iterator -- the data through which we iterate
            optimizer {nn.optimizer} -- The optimizer used to update the weights
            horizontal_cal -- iterator over horizontal calibration data
            vertical_cal -- iterator over vertical calibration data
            optimizer_cal {nn.optimizer} -- optimizer used to update the weights for calibration
        
        Keyword Arguments:
            e_idx {int} -- the epoch index (to support loss weights that change with time) (default: {0})
        
        Returns:
            dict[name -> float] -- the average loss for each regular loss function and calibration loss function
        """
        # calibrate
        loss = self.calibrate(horizontal_cal, vertical_cal, optimizer_cal, e_idx)

        # train
        loss.update(self.run_epoch(iterator, optimizer, e_idx))
        return loss

    def train_eval(self, train_iterator, test_iterator, optimizer, horizontal_cal, vertical_cal, optimizer_cal, n_epochs):
        """Train the model through multiple epochs. Also evaluate with the test iterator after each epoch.
        
        Arguments:
            train_iterator -- the data used to train the model
            test_iterator -- the data used to evaluate the model
            optimizer {nn.optimizer} -- The optimizer used to update the weights
            horizontal_cal -- iterator over horizontal calibration data
            vertical_cal -- iterator over vertical calibration data
            optimizer_cal {nn.optimizer} -- optimizer used to update the weights for calibration
            n_epochs {int} -- The number of epochs
        """
        
        for e_idx in tqdm.tqdm(range(n_epochs)):
            epoch_train_loss = self.train(train_iterator, optimizer, horizontal_cal, vertical_cal, optimizer_cal, e_idx)
            for func in self.loss_functions:
                self.train_loss[func['name']].append(epoch_train_loss[func['name']])
            for func in self.loss_functions_cal:
                self.train_loss[func['name']].append(epoch_train_loss[func['name']])

            epoch_test_loss = self.evaluate(test_iterator, e_idx)
            for func in self.loss_functions:
                self.test_loss[func['name']].append(epoch_test_loss[func['name']])
            self.current_epoch += 1
    
    def plot_loss(self, ylim=None):
        """Plot the losses from the training procedure
        
        Keyword Arguments:
            ylim {[int, int]} -- the boundaries of the visualization across the y axis (default: {None})
        """
        colors = seaborn.color_palette('Paired')
        loss_names = []
        
        # train and test losses
        plt.figure(figsize=(12,6))
        for idx, func in enumerate(self.loss_functions):
            func_name = func['name']
            plt.plot(self.train_loss[func_name], c=colors[(2 * idx) % 12])
            plt.plot(self.test_loss[func_name], c=colors[(2 * idx + 1) % 12])
            loss_names.append('train_' + func_name)
            loss_names.append('test_' + func_name)

        # calibration loss
        for idx, func in enumerate(self.loss_functions_cal, start=len(self.loss_functions)):
            func_name = func['name']
            plt.plot(self.train_loss[func_name], c=colors[(2 * idx + 1) % 12])
            loss_names.append(func_name)
        
        plt.legend(loss_names)
        plt.ylim(ylim)
        plt.grid(True)

class Conv(nn.Module):
    """Represents a convolution layer"""
    def __init__(self, input_dim, output_dim, kernel_size):
        """Construct a convolution layer with the specified sizes
        
        Arguments:
            input_dim {int} -- the dimension of the input
            output_dim {int} -- the dimension of the ouput
            kernel_size {[type]} -- the size of the filters. With a kernel_size of k, the filter will look at the (k - 1) / 2
                previous and next rows to compute the current one
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # the size of the kernel must be odd
        assert kernel_size % 2 == 1
        
        self.padder = nn.ReplicationPad1d(kernel_size // 2)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size)
        
    def forward(self, x):
        """applies the forward pass of the convolution
        
        Arguments:
            x {torch tensor} -- the features of one or multiple simulations
        
        Returns:
            torch tensor -- the result of applying the convolutions over the input
        """

        # if the input corresponds to a single simulation, then we will need to add a dimension
        single = x.dim() == 2
        
        # transpose so that we consider the features as the convolution channels
        x = x.transpose(-1, -2)
        
        if single:
            x = x.unsqueeze(0) # add dimension
            
        src = self.padder(x) # pad to allow first and last values to be computed
        out = self.conv(src) # apply convolution
            
        if single:
            out = out.squeeze(0) # remove the dimension back
            
        # transpose back
        out = out.transpose(-1, -2)
        return out

class ConvInv(nn.Module):
    """Represents a transposed convolution layer"""
    
    def __init__(self, input_dim, output_dim, kernel_size):
        """Construct a convolution layer with the specified sizes
        
        Arguments:
            input_dim {int} -- the dimension of the input
            output_dim {int} -- the dimension of the ouput
            kernel_size {[type]} -- the size of the filters. With a kernel_size of k, the filter will generate values for the (k - 1) / 2
                previous and next rows as well as the current one.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        
        # the size of the kernel must be odd
        assert kernel_size % 2 == 1
        
        self.padder = nn.ReplicationPad1d(kernel_size // 2)
        self.conv = nn.ConvTranspose1d(input_dim, output_dim, kernel_size)
        
    def forward(self, x):
        """applies the forward pass of the transposed convolution
        
        Arguments:
            x {torch tensor} -- the features of one or multiple simulations
        
        Returns:
            torch tensor -- the result of applying the transposed convolutions over the input
        """


        # if the input corresponds to a single simulation, then we will need to add a dimension
        single = x.dim() == 2
        
        # transpose so that we consider the features as the convolution channels
        x = x.transpose(-1, -2)

        if single:
            x = x.unsqueeze(0) # add dimension
            
        src = self.padder(x) # pad to respect the inverse procedure of the convolution
        out = self.conv(src) # apply transposed convolution
            
        if single:
            out = out.squeeze(0) # remove the dimension back
            
        # remove the padding values (crop)
        out = out[..., self.kernel_size - 1: -(self.kernel_size - 1)]

        # transpose back
        out = out.transpose(-1, -2)
            
        return out
