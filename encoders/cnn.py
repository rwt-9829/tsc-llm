"""CNN-based encoder for time series embedding.

Provides a simple stacked convolutional architecture that progressively
increases channel dimensions to match LLM embedding size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Stacked 1D Convolutional Encoder for time series.
    
    Architecture: 4 conv layers (1→512→1024→2048→4096) with ReLU activation,
    followed by adaptive pooling, batch normalization, and sigmoid.
    
    Attributes:
        conv1-4: Convolutional layers with increasing channel dimensions.
        pool: Adaptive average pooling to fixed output length k.
        norm: Batch normalization for output stabilization.
        act: ReLU activation function.
    """
    
    def __init__(self, input_channels=1, output_dim=4096, window_size=3, stride=2, k=16):
        """Initialize the CNN encoder.
        
        Args:
            input_channels: Number of input channels (default: 1 for univariate).
            output_dim: Output embedding dimension (default: 4096 for LLaMA).
            window_size: Convolutional kernel size (default: 3).
            stride: Convolution stride for downsampling (default: 2).
            k: Number of output tokens/time steps (default: 16).
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=512, kernel_size=window_size, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=window_size, stride=stride, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=window_size, stride=stride, padding=1)
        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=output_dim, kernel_size=window_size, stride=stride, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=k)
        self.norm = nn.BatchNorm1d(output_dim) 
        self.act = nn.ReLU() 
    
    def forward(self, x):
        """Forward pass through the CNN encoder.
        
        Args:
            x: Input tensor of shape [B, T] where B is batch size and T is sequence length.
            
        Returns:
            Tensor of shape [B, k, output_dim] suitable for LLM input.
        """
        x = x.unsqueeze(1)  # [B, 1, seq_len] - add channel dimension
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)  # [B, output_dim, k]
        x = self.norm(x)
        x = torch.sigmoid(x)  # Normalize to [0, 1] range
        x = x.permute(0, 2, 1)  # [B, k, output_dim]
        return x