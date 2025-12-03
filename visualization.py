"""
Visualization utilities for ELA-ViT
Generates heatmaps for operational mode k̄ and LII (Figure 2 in paper)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_k_bar_heatmap(model, lii_tracker, save_path='./outputs'):
    """
    Plot operational mode k̄ heatmap (Figure 2 visualization).
    
    Shows the current operational mode for each head in each layer.
    Layer 0 is displayed at the bottom (inverted Y-axis).
    
    Args:
        model: ViT model with attention tracking
        lii_tracker: LII tracker with k̄ history
        save_path: Directory to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Build k̄ matrix: [num_layers, num_heads]
    k_bar_matrix = np.zeros((model.num_layers, model.num_heads))
    
    for layer_idx in range(model.num_layers):
        if lii_tracker.k_bar_head_buffers[layer_idx]:
            # Use most recent k̄ values
            k_bar_matrix[layer_idx, :] = lii_tracker.k_bar_head_buffers[layer_idx][-1]
        else:
            k_bar_matrix[layer_idx, :] = np.nan
    
    # Create figure with large fonts
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    ax = sns.heatmap(
        k_bar_matrix,
        annot=False,  # No numbers on cells
        cmap='viridis_r',
        cbar_kws={'shrink': 0.8}
    )
    
    # Invert Y-axis so Layer 0 is at bottom
    ax.invert_yaxis()
    
    # Set titles and labels with large fonts
    ax.set_title(
        r'Current Operational Mode ($\bar{k}$) per Head',
        fontsize=24,
        pad=20,
        fontweight='bold'
    )
    ax.set_ylabel('Layer Index', fontsize=20, labelpad=15)
    ax.set_xlabel('Head Index', fontsize=20, labelpad=15)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Update colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'Operational Mode ($\bar{k}$)', fontsize=20, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'k_bar_heatmap.png'), dpi=300)
    plt.close()


def plot_lii_heatmap(model, lii_tracker, save_path='./outputs'):
    """
    Plot Layer Instability Index (LII) heatmap (Figure 2 visualization).
    
    Shows the LII value for each head in each layer.
    Layer 0 is displayed at the bottom (inverted Y-axis).
    
    Args:
        model: ViT model with attention tracking
        lii_tracker: LII tracker with computed LII values
        save_path: Directory to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Build LII matrix: [num_layers, num_heads]
    lii_matrix = np.zeros((model.num_layers, model.num_heads))
    
    for layer_idx in range(model.num_layers):
        lii_per_head = lii_tracker.compute_head_lii(layer_idx)
        
        if lii_per_head is not None:
            lii_matrix[layer_idx, :] = lii_per_head
        else:
            lii_matrix[layer_idx, :] = np.nan
    
    # Create figure with large fonts
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    ax = sns.heatmap(
        lii_matrix,
        annot=False,  # No numbers on cells
        cmap='coolwarm',
        cbar_kws={'shrink': 0.8}
    )
    
    # Invert Y-axis so Layer 0 is at bottom
    ax.invert_yaxis()
    
    # Set titles and labels with large fonts
    ax.set_ylabel('Layer Index', fontsize=20, labelpad=15)
    ax.set_xlabel('Head Index', fontsize=20, labelpad=15)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Update colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Layer Instability Index (LII)', fontsize=20, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lii_heatmap.png'), dpi=300)
    plt.close()
