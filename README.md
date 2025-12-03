# Energy Landscape-Aware Vision Transformers (ELA-ViT)

Implementation of "Energy Landscape-Aware Vision Transformers: Layerwise Dynamics and Adaptive Task-Specific Training via Hopfield States" (NeurIPS 2025).

## Overview

ELA-ViT introduces a novel adaptive fine-tuning framework that dynamically freezes stable transformer layers based on their energy landscape behavior. The method uses the **Layer Instability Index (LII)** to identify layers that have converged to attractor states and can be safely frozen, reducing computational cost while maintaining or improving accuracy.

## Key Concepts

### Layer Instability Index (LII)

The LII quantifies the metastability of each layer by measuring the variability of the **operational mode** k̄ across inputs:

- **Operational mode k̄** (Eq. 2): The median minimum number of tokens required to accumulate ρ=0.9 of attention mass
- **LII** (Eq. 3): The Median Absolute Deviation (MAD) of k̄ over a sliding window

### Three-Stage Training (Algorithm 1)

1. **Warm-up Phase** (0 ≤ t < T): Collect LII statistics while training all layers
2. **Freeze Decision** (t = T): Freeze layers where LII < τ_freeze
3. **Consolidation Phase** (t > T): Continue training only unfrozen layers

## Project Structure

```
.
├── config.py           # Configuration dataclass
├── lii_tracker.py      # Layer Instability Index computation (Section 3.2)
├── vit_model.py        # ViT with attention tracking
├── trainer.py          # ELA-ViT training framework (Algorithm 1)
├── visualization.py    # Heatmap generation (Figure 2)
├── data_utils.py       # Dataset loaders
└── main.py            # Main entry point
```

## Code Organization

### Core Components

1. **`lii_tracker.py`** - `LIITracker` class
   - Computes operational mode k̄ (Eq. 2)
   - Tracks k̄ history in circular buffers
   - Computes LII as MAD (Eq. 3)

2. **`vit_model.py`** - Model components
   - `AttentionWrapper`: Extracts attention weights
   - `ViTWithAttentionTracking`: ViT with attention tracking capability

3. **`trainer.py`** - `ELAViTTrainer` class
   - Implements three-stage training (Algorithm 1)
   - Stage I: Warm-up with LII estimation
   - Stage II: Freeze decision based on τ_freeze
   - Stage III: Consolidation training

4. **`visualization.py`** - Plotting utilities
   - `plot_k_bar_heatmap()`: Visualize operational modes
   - `plot_lii_heatmap()`: Visualize LII values

## Variable Naming (Aligned with Paper)

| Paper Notation | Code Variable | Description |
|---------------|--------------|-------------|
| k̄_ℓ | `k_bar_scalar` | Operational mode (scalar) |
| k̄_ℓ | `k_bar_vector` | Operational mode per head |
| LII_ℓ | `lii` | Layer Instability Index |
| ρ | `rho` | Attention mass threshold (0.9) |
| Δ | `window_size` | Sliding window size |
| T | `warmup_steps` | Warm-up duration |
| τ_freeze | `tau_freeze` | Freeze threshold |
| η | `learning_rate` | Learning rate |
| λ_wd | `weight_decay` | Weight decay |

## Usage

### Training

```python
python main.py
```

### Custom Configuration

```python
from config import ELAViTConfig
from vit_model import ViTWithAttentionTracking
from trainer import ELAViTTrainer
from data_utils import get_cifar10_loaders

# Customize configuration
config = ELAViTConfig(
    learning_rate=1e-5,
    warmup_steps=60,
    window_size=20,
    freeze_threshold_multiplier=1.0
)

# Load data
train_loader, val_loader = get_cifar10_loaders(
    batch_size=config.batch_size
)

# Initialize model and trainer
model = ViTWithAttentionTracking(
    model_name='google/vit-base-patch16-224',
    num_classes=10
)

trainer = ELAViTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **config.__dict__
)

# Train
trainer.train(num_epochs=10)
```

## Key Parameters

### LII Configuration
- `window_size` (Δ): Size of sliding window for computing LII (default: 20)
- `rho`: Attention mass concentration threshold (default: 0.9)

### Training Configuration
- `warmup_steps` (T): Duration of warm-up phase (default: 60)
- `freeze_threshold_multiplier`: Multiplier for computing τ_freeze (default: 1.0)
- `learning_rate`: AdamW learning rate (default: 1e-5)
- `weight_decay`: AdamW weight decay (default: 0.1)

## Citation

```bibtex
@inproceedings{xia2025energy,
  title={Energy Landscape-Aware Vision Transformers: Layerwise Dynamics and Adaptive Task-Specific Training via Hopfield States},
  author={Xia, Runze and Jiang, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Dependencies

- PyTorch
- transformers (HuggingFace)
- torchvision
- numpy
- matplotlib
- seaborn
- tqdm
