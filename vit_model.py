"""
Vision Transformer with Attention Tracking
Implements attention weight extraction for computing LII (Section 3.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
import numpy as np


# Global storage for attention weights
_attention_storage = {}


class AttentionWrapper(nn.Module):
    """
    Wrapper for attention modules to capture attention weights.
    
    This wrapper extracts attention probabilities (softmax scores) from
    self-attention layers for LII computation.
    """
    
    def __init__(self, original_attention: nn.Module, num_heads: int, 
                 head_dim: int, layer_idx: int):
        """
        Initialize attention wrapper.
        
        Args:
            original_attention: Original attention module from ViT
            num_heads: Number of attention heads
            head_dim: Dimension per head
            layer_idx: Index of the layer in the network
        """
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Forward pass with attention weight extraction.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output from original attention module
        """
        # Call original attention
        output = self.original_attention(hidden_states, *args, **kwargs)
        
        # Extract attention weights if requested
        if _attention_storage.get('compute_flag', False):
            with torch.no_grad():
                try:
                    self._extract_attention_weights(hidden_states)
                except Exception:
                    # Silently fail if extraction unsuccessful
                    pass
        
        return output
    
    def _extract_attention_weights(self, hidden_states: torch.Tensor):
        """
        Extract attention probability matrix from self-attention computation.
        
        Computes: softmax(Q K^T / sqrt(d_k))
        """
        if not hasattr(self.original_attention, 'attention'):
            return
        
        self_attn = self.original_attention.attention
        B, N, _ = hidden_states.shape
        
        # Extract Q and K matrices
        if hasattr(self_attn, 'query'):
            Q = self_attn.query(hidden_states)
            K = self_attn.key(hidden_states)
        elif hasattr(self_attn, 'qkv'):
            # Some implementations use fused QKV
            qkv = self_attn.qkv(hidden_states).reshape(
                B, N, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            Q, K = qkv[0], qkv[1]
        else:
            return
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=-1)
        
        # Store for LII computation
        if 'weights' not in _attention_storage:
            _attention_storage['weights'] = {}
        _attention_storage['weights'][self.layer_idx] = attention_probs.detach().cpu()


class ViTWithAttentionTracking(nn.Module):
    """
    Vision Transformer with attention tracking capability.
    
    Wraps a pretrained ViT model and patches attention modules to enable
    extraction of attention weights for LII computation.
    """
    
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', 
                 num_classes: int = 100):
        """
        Initialize ViT with attention tracking.
        
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Extract model configuration
        self.config = self.model.config
        self.num_layers = len(self.model.vit.encoder.layer)
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Patch attention modules with wrappers
        self._patch_attention_modules()
    
    def _patch_attention_modules(self):
        """
        Replace attention modules with tracking wrappers.
        """
        for layer_idx, layer in enumerate(self.model.vit.encoder.layer):
            if hasattr(layer, 'attention'):
                layer.attention = AttentionWrapper(
                    layer.attention,
                    self.num_heads,
                    self.head_dim,
                    layer_idx
                )
    
    @property
    def attention_weights(self):
        """
        Get captured attention weights.
        
        Returns:
            Dictionary mapping layer_idx to attention tensor
            Shape: [batch, num_heads, num_queries, num_keys]
        """
        return _attention_storage.get('weights', {})
    
    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None,
                compute_attention: bool = True):
        """
        Forward pass with optional attention tracking.
        
        Args:
            pixel_values: Input images [batch, channels, height, width]
            labels: Target labels [batch]
            compute_attention: Whether to extract attention weights
            
        Returns:
            Model outputs (logits, loss if labels provided)
        """
        # Reset attention storage
        _attention_storage['weights'] = {}
        _attention_storage['compute_flag'] = compute_attention
        
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def freeze_layer(self, layer_idx: int):
        """
        Freeze a specific transformer layer (Section 3.4).
        
        Args:
            layer_idx: Index of layer to freeze
        """
        for param in self.model.vit.encoder.layer[layer_idx].parameters():
            param.requires_grad = False
