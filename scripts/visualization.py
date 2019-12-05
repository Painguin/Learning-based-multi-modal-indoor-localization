"""
Local module for visualization functions
"""

import matplotlib.pyplot as plt

import seaborn

def plot_loss_evolution(train_losses, test_losses):
    plt.plot(train_losses, c='blue')
    plt.plot(test_losses, c='orange')
    plt.legend(['train loss', 'test loss'])
    plt.ylim(0, 200)
    plt.grid(True)

def plot_embedding(encoder, data, indices):
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
    if isinstance(indices, int):
        indices = [indices]
    encoded = encoder(data[indices][0]).detach()
    ground_truth = data[indices][1]

    n_cols = 1
    n_rows = (len(indices) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for idx, (embedding, simulation) in enumerate(zip(encoded, ground_truth)):
        if len(indices) > n_cols:
            ax = axes[idx // n_cols]
        else:
            ax = axes[idx % n_cols]
        
        data.plot_simulation(simulation, ax)
        data.plot_simulation(embedding, ax)

def plot_embedding_pca(encoder, data, pca, indices):
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
