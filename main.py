"""
Main entry point for Energy Landscape-Aware Vision Transformer (ELA-ViT)
Based on: "Energy Landscape-Aware Vision Transformers" (NeurIPS 2025)

Usage:
    python main.py
"""

import torch
import numpy as np
import random

from config import ELAViTConfig
from vit_model import ViTWithAttentionTracking
from trainer import ELAViTTrainer
from data_utils import get_cifar10_loaders


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """
    Main training function implementing the ELA-ViT framework.
    
    Three-stage adaptive fine-tuning (Algorithm 1):
    1. Warm-up Phase: Collect LII statistics
    2. Freeze Decision: Freeze layers with low LII
    3. Consolidation Phase: Fine-tune remaining layers
    """
    # Load configuration
    config = ELAViTConfig()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    print("=" * 60)
    print("Energy Landscape-Aware Vision Transformers (ELA-ViT)")
    print("NeurIPS 2025")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Number of classes: {config.num_classes}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    print(f"\nLII Parameters:")
    print(f"  Window size (Δ): {config.window_size}")
    print(f"  Attention threshold (ρ): {config.rho}")
    print(f"  Warm-up steps (T): {config.warmup_steps}")
    print(f"  Freeze threshold multiplier: {config.freeze_threshold_multiplier}")
    print("=" * 60 + "\n")
    
    # Load data (CIFAR-10 with small subset for demonstration)
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_subset_size=100  # Small subset for quick demonstration
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    # Initialize model
    print("Initializing ViT model with attention tracking...")
    model = ViTWithAttentionTracking(
        model_name=config.model_name,
        num_classes=config.num_classes
    )
    print(f"Model: {config.model_name}")
    print(f"Number of layers: {model.num_layers}")
    print(f"Number of heads: {model.num_heads}")
    print(f"Hidden size: {model.hidden_size}\n")
    
    # Initialize trainer
    print("Initializing ELA-ViT trainer...")
    trainer = ELAViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        freeze_threshold_multiplier=config.freeze_threshold_multiplier,
        window_size=config.window_size,
        plot_batches=config.plot_batches,
        output_dir=config.output_dir
    )
    
    # Train model
    print("Starting ELA-ViT training...\n")
    trainer.train(num_epochs=config.num_epochs)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Frozen layers: {len(trainer.frozen_layers)}/{model.num_layers}")
    print(f"Visualizations saved to: {config.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
