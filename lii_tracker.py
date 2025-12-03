"""
Layer Instability Index (LII) Tracker
Implements Section 3.2 of the paper: "Layer Instability Index (LII)"

The LII quantifies the metastability of each layer by measuring the variability
of the operational mode k̄_ℓ across inputs.
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Optional


class LIITracker:
    """
    Tracks Layer Instability Index (LII) for each transformer layer.
    
    The operational mode k̄ is defined as the median minimum number of tokens
    required to accumulate ρ=0.9 of the attention mass (Eq. 2 in paper).
    
    LII is computed as the Median Absolute Deviation (MAD) of k̄ (Eq. 3).
    """
    
    def __init__(self, num_layers: int, num_heads: int, window_size: int = 20, rho: float = 0.9):
        """
        Initialize LII tracker.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            window_size: Sliding window size Δ for computing LII
            rho: Attention mass concentration threshold (default: 0.9)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.rho = rho
        
        # Circular buffers for operational modes (Eq. 2)
        self.k_bar_layer_buffers = {
            i: deque(maxlen=window_size) for i in range(num_layers)
        }
        self.k_bar_head_buffers = {
            i: deque(maxlen=window_size) for i in range(num_layers)
        }
    
    def compute_operational_mode_scalar(self, attention_weights: torch.Tensor) -> float:
        """
        Compute scalar operational mode k̄ for a layer (Eq. 2).
        
        Returns the median minimum number of tokens to accumulate ρ of attention mass.
        
        Args:
            attention_weights: [batch, num_heads, num_queries, num_keys]
            
        Returns:
            Scalar k̄ value
        """
        # Average over queries
        attn = attention_weights.mean(dim=2)  # [batch, num_heads, num_keys]
        
        # Sort attention scores in descending order
        sorted_attn, _ = torch.sort(attn, dim=-1, descending=True)
        
        # Cumulative sum to find k where mass >= ρ
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        k_values = (cumsum_attn >= self.rho).float().argmax(dim=-1) + 1
        
        return k_values.float().median().item()
    
    def compute_operational_mode_per_head(self, attention_weights: torch.Tensor) -> np.ndarray:
        """
        Compute operational mode k̄ per head (vector form).
        
        Args:
            attention_weights: [batch, num_heads, num_queries, num_keys]
            
        Returns:
            Array of shape [num_heads] with k̄ for each head
        """
        # Average over queries
        attn_avg_query = attention_weights.mean(dim=2)
        
        # Sort and find k per head
        sorted_attn, _ = torch.sort(attn_avg_query, dim=-1, descending=True)
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        k_raw = (cumsum_attn >= self.rho).float().argmax(dim=-1) + 1
        
        # Median over batch dimension
        k_per_head = k_raw.float().median(dim=0).values
        
        return k_per_head.cpu().numpy()
    
    def update(self, layer_idx: int, k_bar_scalar: float, k_bar_vector: np.ndarray):
        """
        Update circular buffers with new operational mode values.
        
        Args:
            layer_idx: Layer index
            k_bar_scalar: Scalar operational mode for the layer
            k_bar_vector: Per-head operational modes [num_heads]
        """
        self.k_bar_layer_buffers[layer_idx].append(k_bar_scalar)
        self.k_bar_head_buffers[layer_idx].append(k_bar_vector)
    
    def compute_layer_lii(self, layer_idx: int) -> float:
        """
        Compute Layer Instability Index (LII) for a layer (Eq. 3).
        
        LII_ℓ = median_t |k̄_ℓ^t - median_{t'}(k̄_ℓ^{t'-Δ:t'})|
        
        Args:
            layer_idx: Layer index
            
        Returns:
            LII value (MAD of operational mode)
        """
        if len(self.k_bar_layer_buffers[layer_idx]) < 2:
            return float('inf')
        
        values = np.array(list(self.k_bar_layer_buffers[layer_idx]))
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        return mad
    
    def compute_head_lii(self, layer_idx: int) -> Optional[np.ndarray]:
        """
        Compute LII per head for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Array of shape [num_heads] with LII for each head, or None if insufficient data
        """
        if len(self.k_bar_head_buffers[layer_idx]) < 2:
            return None
        
        # Stack history: [time_steps, num_heads]
        history = np.stack(list(self.k_bar_head_buffers[layer_idx]))
        
        # Compute MAD per head
        medians = np.median(history, axis=0)
        deviations = np.abs(history - medians)
        lii_per_head = np.median(deviations, axis=0)
        
        return lii_per_head
    
    def get_all_layer_liis(self) -> Dict[int, float]:
        """
        Get LII values for all layers.
        
        Returns:
            Dictionary mapping layer_idx to LII value
        """
        return {i: self.compute_layer_lii(i) for i in range(self.num_layers)}
