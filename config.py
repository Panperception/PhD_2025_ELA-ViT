"""
Configuration file for Energy Landscape-Aware Vision Transformers (ELA-ViT)
Based on: "Energy Landscape-Aware Vision Transformers" (NeurIPS 2025)
"""

from dataclasses import dataclass


@dataclass
class ELAViTConfig:
    """Configuration for ELA-ViT training framework."""
    
    # Model configuration
    model_name: str = 'google/vit-base-patch16-224'
    num_classes: int = 10
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    batch_size: int = 8
    num_epochs: int = 1
    
    # LII (Layer Instability Index) parameters
    window_size: int = 20  # Δ in paper: sliding window length
    rho: float = 0.9  # Attention mass concentration threshold
    
    # Adaptive freezing parameters
    warmup_steps: int = 60  # T in paper: warm-up phase duration
    freeze_threshold_multiplier: float = 1.0  # Controls τ_freeze
    
    # Visualization
    plot_batches: int = 20  # Generate heatmaps after this many batches
    output_dir: str = './outputs'
    
    # Device
    device: str = 'cuda'
    num_workers: int = 0
    
    # Random seed
    seed: int = 42
