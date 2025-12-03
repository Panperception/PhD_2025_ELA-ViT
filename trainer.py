"""
Energy Landscape-Aware ViT Trainer
Implements the adaptive fine-tuning framework (Section 3.4 and Algorithm 1)

Three-stage training:
1. Warm-up Phase (0 ≤ t < T): Collect LII statistics
2. Freeze Decision (t = T): Freeze stable layers
3. Consolidation Phase (t > T): Train unfrozen layers
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Set

from lii_tracker import LIITracker
from vit_model import ViTWithAttentionTracking
from visualization import plot_k_bar_heatmap, plot_lii_heatmap


class ELAViTTrainer:
    """
    Energy Landscape-Aware Vision Transformer Trainer.
    
    Implements Algorithm 1 from the paper with three phases:
    - Warm-up: Estimate Layer Instability Index (LII)
    - Decision: Freeze layers with LII < τ_freeze
    - Consolidation: Fine-tune remaining adaptive layers
    """
    
    def __init__(
        self,
        model: ViTWithAttentionTracking,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 60,
        freeze_threshold_multiplier: float = 1.0,
        window_size: int = 20,
        plot_batches: int = 20,
        output_dir: str = './outputs'
    ):
        """
        Initialize ELA-ViT trainer.
        
        Args:
            model: ViT model with attention tracking
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            learning_rate: Learning rate η
            weight_decay: Weight decay λ_wd
            warmup_steps: Warm-up duration T (Section 3.4)
            freeze_threshold_multiplier: Multiplier for computing τ_freeze
            window_size: Sliding window size Δ for LII
            plot_batches: Generate visualizations after this many batches
            output_dir: Directory for saving outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        
        # Optimizer (AdamW as specified in Section 4)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # LII tracker for computing Layer Instability Index
        self.lii_tracker = LIITracker(
            model.num_layers,
            model.num_heads,
            window_size
        )
        
        # Training configuration
        self.warmup_steps = warmup_steps  # T in Algorithm 1
        self.freeze_threshold_multiplier = freeze_threshold_multiplier
        self.plot_batches = plot_batches
        
        # Training state
        self.step_count = 0
        self.frozen_layers: Set[int] = set()
        self.freeze_decision_made = False
    
    def train_step(self, batch):
        """
        Execute one training step.
        
        Args:
            batch: (images, labels) tuple
            
        Returns:
            Loss value
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass (capture attention during warm-up or for visualization)
        capture_attention = (
            self.step_count < max(self.warmup_steps, self.plot_batches)
        )
        outputs = self.model(
            pixel_values=images,
            labels=labels,
            compute_attention=capture_attention
        )
        
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        
        # Update LII tracker during warm-up
        if capture_attention:
            self._update_lii_tracker()
        
        self.step_count += 1
        return loss.item()
    
    def _update_lii_tracker(self):
        """
        Update LII tracker with current attention weights.
        Computes operational mode k̄ (Eq. 2) and updates circular buffers.
        """
        attention_weights = self.model.attention_weights
        
        for layer_idx in range(self.model.num_layers):
            if layer_idx in attention_weights:
                attn = attention_weights[layer_idx]
                
                # Compute operational modes (Eq. 2)
                k_bar_scalar = self.lii_tracker.compute_operational_mode_scalar(attn)
                k_bar_vector = self.lii_tracker.compute_operational_mode_per_head(attn)
                
                # Update circular buffers
                self.lii_tracker.update(layer_idx, k_bar_scalar, k_bar_vector)
    
    def make_freeze_decision(self):
        """
        Make layer freezing decision (Stage II in Section 3.4).
        
        Freezes layers where LII_ℓ < τ_freeze.
        τ_freeze is computed as: median(LII_ℓ) × freeze_threshold_multiplier
        """
        if self.freeze_decision_made:
            return
        
        print("\n" + "=" * 60)
        print("FREEZE DECISION (Stage II)")
        print("=" * 60)
        
        # Get LII values for all layers
        lii_values = self.lii_tracker.get_all_layer_liis()
        
        # Filter valid LII values
        valid_liis = [
            v for v in lii_values.values()
            if v != float('inf') and not np.isnan(v)
        ]
        
        if not valid_liis:
            print("Insufficient data for freeze decision")
            return
        
        # Compute dynamic threshold τ_freeze
        tau_freeze = np.median(valid_liis) * self.freeze_threshold_multiplier
        print(f"Freeze threshold τ_freeze: {tau_freeze:.4f}")
        print(f"(median LII: {np.median(valid_liis):.4f}, "
              f"multiplier: {self.freeze_threshold_multiplier})")
        print()
        
        # Freeze layers with LII < τ_freeze
        for layer_idx, lii in lii_values.items():
            if lii < tau_freeze:
                self.model.freeze_layer(layer_idx)
                self.frozen_layers.add(layer_idx)
                print(f"Layer {layer_idx:2d} FROZEN   (LII: {lii:.4f})")
            else:
                print(f"Layer {layer_idx:2d} ACTIVE   (LII: {lii:.4f})")
        
        print()
        print(f"Frozen: {len(self.frozen_layers)}/{self.model.num_layers} layers")
        print("=" * 60 + "\n")
        
        self.freeze_decision_made = True
    
    def generate_visualizations(self):
        """
        Generate heatmap visualizations (Figure 2 in paper).
        - Operational mode k̄ per head
        - Layer Instability Index (LII) per head
        """
        print(f"\n[VISUALIZATION] Generating heatmaps...")
        plot_k_bar_heatmap(self.model, self.lii_tracker, self.output_dir)
        plot_lii_heatmap(self.model, self.lii_tracker, self.output_dir)
        print(f"[VISUALIZATION] Saved to {self.output_dir}/\n")
    
    def train(self, num_epochs: int):
        """
        Main training loop implementing Algorithm 1.
        
        Args:
            num_epochs: Number of training epochs
        """
        print("=" * 60)
        print("ELA-ViT TRAINING")
        print("=" * 60)
        print(f"Warm-up steps (T): {self.warmup_steps}")
        print(f"Window size (Δ): {self.lii_tracker.window_size}")
        print(f"Attention threshold (ρ): {self.lii_tracker.rho}")
        print("=" * 60 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            
            # Training progress bar
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
            
            for batch in pbar:
                loss = self.train_step(batch)
                
                # Stage I -> Visualization
                if self.step_count == self.plot_batches:
                    self.generate_visualizations()
                
                # Stage I -> Stage II: Make freeze decision
                if self.step_count == self.warmup_steps:
                    self.make_freeze_decision()
                
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Validation
            val_accuracy = self.validate()
            print(f"Epoch {epoch} | Validation Accuracy: {val_accuracy:.2f}%\n")
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Top-1 accuracy (%)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(pixel_values=images, compute_attention=False)
            predictions = outputs.logits.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return 100.0 * correct / total
