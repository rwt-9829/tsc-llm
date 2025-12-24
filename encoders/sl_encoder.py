"""Inception-based encoder for time series embedding (SL Encoder).

Provides a multi-scale convolutional architecture inspired by Inception networks,
using parallel convolutions with different kernel sizes to capture patterns
at multiple temporal scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):
    """Inception Block with multi-scale 1D convolutions.
    
    Applies multiple parallel convolutions with different kernel sizes
    (1, 3, 5, 7, ..., 2*num_kernels-1) and averages their outputs.
    This captures patterns at multiple temporal scales simultaneously.
    
    Attributes:
        kernels: ModuleList of Conv1d layers with varying kernel sizes.
        dropout: Dropout layer for regularization.
    """
    
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """Initialize the Inception block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per kernel.
            num_kernels: Number of parallel convolutions (default: 6).
            init_weight: If True, apply Kaiming initialization (default: True).
        """
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # Create convolutions with kernel sizes: 1, 3, 5, 7, 9, 11, ...
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()
        self.dropout = nn.Dropout(p=0.2)

    def _initialize_weights(self):
        """Initialize convolution weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through parallel convolutions.
        
        Args:
            x: Input tensor of shape [B, in_channels, T].
            
        Returns:
            Tensor of shape [B, out_channels, T] (mean of all kernel outputs).
        """
        res_list = []
        for i in range(self.num_kernels):
            tmp = self.dropout(self.kernels[i](x))
            res_list.append(tmp)
        # Stack along new dim and average across kernels
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Encoder(nn.Module):
    """Stacked Inception Encoder for time series (SL Encoder).
    
    Architecture: 4 Inception blocks with LayerNorm, followed by
    a linear projection to LLM embedding dimension. Preserves
    temporal resolution throughout the network.
    
    Attributes:
        incept_1-4: Inception blocks for multi-scale feature extraction.
        layernorm1-4: LayerNorm layers for each block output.
        projection: Linear layer mapping to LLM embedding dimension.
    """
    
    def __init__(self, window_size, num_kernels, k):
        """Initialize the SL Encoder.
        
        Args:
            window_size: Length of input time series.
            num_kernels: Number of parallel convolutions in each Inception block.
            k: Number of output channels/tokens.
        """
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.k = k
        
        # Stack of Inception blocks: 1 → k → k → k → k channels
        self.incept_1 = Inception_Block_V1(1, self.k, num_kernels)
        self.layernorm1 = nn.LayerNorm(window_size)
        self.incept_2 = Inception_Block_V1(self.k, self.k, num_kernels)
        self.layernorm2 = nn.LayerNorm(window_size)
        self.incept_3 = Inception_Block_V1(self.k, self.k, num_kernels)
        self.layernorm3 = nn.LayerNorm(window_size)
        self.incept_4 = Inception_Block_V1(self.k, self.k, num_kernels)
        self.layernorm4 = nn.LayerNorm(window_size)
        
        # Project from window_size to LLM hidden dim
        self.projection = nn.Linear(window_size, 4096)

    def forward(self, x):
        """Forward pass through stacked Inception blocks.
        
        Args:
            x: Input tensor of shape [B, T] where B is batch size and T is sequence length.
            
        Returns:
            Tensor of shape [B, k, 4096] suitable for LLM input.
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # [B, 1, T] - add channel dimension
            
        # Apply Inception blocks with LayerNorm
        x = self.layernorm1(self.incept_1(x))  # [B, k, T]
        x = self.layernorm2(self.incept_2(x))  # [B, k, T]
        x = self.layernorm3(self.incept_3(x))  # [B, k, T]
        x = self.layernorm4(self.incept_4(x))  # [B, k, T]
        
        # Project to LLM embedding dimension
        x = self.projection(x)  # [B, k, 4096]
        return x 

