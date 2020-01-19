"""
Local module for visualization functions
"""

# Pytorch
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn

def plot_loss_evolution(train_losses, test_losses):
    """Plot the losses from the training procedure
    
    Arguments:
        train_losses {list[float]} -- [description]
        test_losses {list[float]} -- [description]
    """
    plt.plot(train_losses, c='blue')
    plt.plot(test_losses, c='orange')
    plt.legend(['train loss', 'test loss'])
    plt.ylim(0, 200)
    plt.grid(True)

def plot_embedding(encoder, data, indices):
    """Plot the embeddings of some given simulation data next to the real paths indexed by some given indices
    
    Arguments:
        encoder {nn.Module} -- The encoder part of the autoencoder
        data {data.SimulationData} -- The simulation data
        indices {int or range or list[int]} -- The indices of the data from which we want embeddings
    """
    if isinstance(indices, int):
        indices = [indices]

    encoded = encoder(data[indices][0]).detach()

    ground_truth = data[indices][1]
    
    fig, axes = plt.subplots(nrows=len(indices), ncols=2, figsize=(8, 4 * len(indices)))
    for idx, (embedding, simulation) in enumerate(zip(encoded, ground_truth)):
        if len(indices) > 1:
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
        
        ax1.grid(True)
        ax1.axis('equal')
        ax1.plot(*embedding.T)
        ax1.plot(*embedding[0], '*', c='green', markersize=15)
        ax1.plot(*embedding[-1], '*', c='orange', markersize=15);
    
        data.plot_simulation(simulation, ax2)
    
    
    for ax, col in zip(axes[0] if len(indices) > 1 else axes, ['Encoder embedding', 'Simulation']):
        ax.set_title(col)

def plot_embedding_on_simulation(encoder, data, indices):
    """Plot the embeddings of some given simulation data together with the real paths indexed by some given indices
    
    Arguments:
        encoder {nn.Module} -- The encoder part of the autoencoder
        data {data.SimulationData} -- The simulation data
        indices {int or range or list[int]} -- The indices of the data from which we want embeddings
    """

    if isinstance(indices, int):
        indices = [indices]
    
    encoded = encoder(data[indices][0]).detach()

    ground_truth = data[indices][1]

    n_cols = 4
    n_rows = (len(indices) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for idx, (embedding, simulation) in enumerate(zip(encoded, ground_truth)):
        if len(indices) > n_cols:
            ax = axes[idx // n_cols, idx % n_cols]
        else:
            # handle the case when we have only 1 row
            ax = axes[idx % n_cols]
        
        data.plot_simulation(simulation, ax)
        data.plot_simulation(embedding, ax)

def plot_embedding_pca(encoder, data, pca, indices):
    """Plot the embeddings of some given simulation data next to the real paths and the pca embeddings
       indexed by some given indices
    
    Arguments:
        encoder {nn.Module} -- The encoder part of the autoencoder
        data {data.SimulationData} -- The simulation data
        pca {array} -- the pca embeddings
        indices {int or range or list[int]} -- The indices of the data from which we want embeddings
    """
    if isinstance(indices, int):
        indices = [indices]
    encoded = encoder(data[indices][0]).detach()
    pca_emb = pca[indices]
    ground_truth = data[indices][1]
    
    fig, axes = plt.subplots(nrows=len(indices), ncols=3, figsize=(12, 4 * len(indices)))
    for idx, (embedding, pca_embedding, simulation) in enumerate(zip(encoded, pca_emb, ground_truth)):
        if len(indices) > 1:
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            ax3 = axes[idx, 2]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]
        
        ax1.grid(True)
        ax1.plot(*pca_embedding.T)
        ax1.plot(*pca_embedding[0], '*', c='green', markersize=15)
        ax1.plot(*pca_embedding[-1], '*', c='orange', markersize=15)
        ax1.axis('equal')

        ax2.grid(True)
        ax2.plot(*embedding.T)
        ax2.plot(*embedding[0], '*', c='green', markersize=15)
        ax2.plot(*embedding[-1], '*', c='orange', markersize=15)
        ax2.axis('equal')

        data.plot_simulation(simulation, ax3)
    
    for ax, col in zip(axes[0] if len(indices) > 1 else axes, ['PCA embedding', 'Encoder embedding', 'Simulation']):
        ax.set_title(col)

def plot_multiloss_evolution(train_losses, test_losses, ylim=[0,200]):
    """Plot the losses from the training procedure of a model with multiple losses
    
    Keyword Arguments:
        ylim {[int, int]} -- the boundaries of the visualization across the y axis (default: {[0, 200]})
        """
    assert len(train_losses) == len(test_losses) and len(train_losses) <= 6
    colors = seaborn.color_palette('Paired')
    loss_names = []
    
    plt.figure(figsize=(12,6))
    for idx, loss_name in enumerate(train_losses):
        plt.plot(train_losses[loss_name], c=colors[2 * idx])
        plt.plot(test_losses[loss_name], c=colors[2 * idx + 1])
        loss_names.append('train_' + loss_name)
        loss_names.append('test_' + loss_name)
    
    plt.legend(loss_names)
    plt.ylim(ylim)
    plt.grid(True)

def merge_losses(losses):
    """Given the losses of multiple training procedures that don't use the same loss functions,
       merge them (concatenate) into a single loss history
    
    Arguments:
        losses {list[dict[name -> list[float]]]} -- the list of the losses resulted from multiple individual runs
    
    Returns:
        dict[name -> list[float]] -- the multiple dictionary losses merged into a single one
    """
    # get all the unique loss names
    keys = set.union(*[set(loss.keys()) for loss in losses])
    
    merged_loss = {}
    for key in keys:
        merged_loss[key] = []

    for loss in losses:
        # length (number of epochs) of the training procedre
        length = len(loss[list(loss.keys())[0]])

        for key in keys:
            if key in loss:
                # if the loss is present then simply append the values
                merged_loss[key] += loss[key]
            else:
                # if the loss is not present then append only zeros
                merged_loss[key] += [0.] * length

    return merged_loss
