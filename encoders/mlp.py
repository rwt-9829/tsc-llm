"""MLP-based encoder for time series embedding.

Provides a simple multi-layer perceptron architecture that directly
maps time series values to LLM embedding dimension.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Multi-Layer Perceptron Encoder for time series.
    
    Architecture: Stacked linear layers (input→512→1024→2048→4096)
    with ReLU activation between layers and LayerNorm at output.
    
    Note: This encoder treats the entire time series as a single vector,
    producing only 1 token output (k=1 effectively).
    
    Attributes:
        mlp: Sequential stack of linear layers with ReLU.
        norm: LayerNorm for output stabilization.
    """
    
    def __init__(self, input_dim=150, hidden_dims=[512, 1024, 2048], output_dim=4096):
        """Initialize the MLP encoder.
        
        Args:
            input_dim: Length of input time series (default: 150).
            hidden_dims: List of hidden layer dimensions (default: [512, 1024, 2048]).
            output_dim: Output embedding dimension (default: 4096 for LLaMA).
        """
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """Forward pass through the MLP encoder.
        
        Args:
            x: Input tensor of shape [B, T] where B is batch size and T is sequence length.
            
        Returns:
            Tensor of shape [B, 1, output_dim] suitable for LLM input.
        """
        x = x.unsqueeze(1)  # [B, 1, T] - add sequence dimension
        x = self.mlp(x)     # [B, 1, output_dim]
        x = self.norm(x)
        return x