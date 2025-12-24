"""ResNet-based encoder for time series embedding.

Provides a residual network architecture with skip connections
for more effective gradient flow in deep time series encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual Block with optional downsampling.
    
    Implements a standard ResNet block: conv→ReLU→conv + shortcut,
    where the shortcut handles dimension/stride mismatches.
    
    Attributes:
        conv1: First convolution with optional stride for downsampling.
        conv2: Second convolution maintaining spatial dimensions.
        bn1, bn2: Batch normalization layers.
        shortcut: Identity or 1x1 conv for dimension matching.
    """
    
    def __init__(self, in_channels, out_channels, downsample=False):
        """Initialize the residual block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            downsample: If True, reduces spatial dimension by half (default: False).
        """
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection: identity or projection
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape [B, in_channels, T].
            
        Returns:
            Tensor of shape [B, out_channels, T'] where T' = T/2 if downsample else T.
        """
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)  # Residual connection
        return F.relu(out)


class Encoder(nn.Module):
    """ResNet-style Encoder for time series.
    
    Architecture: 4 residual blocks (1→512→1024→2048→4096) with
    downsampling at each stage, followed by adaptive pooling.
    
    Attributes:
        layer1-4: Residual blocks with increasing channel dimensions.
        pool: Adaptive average pooling to fixed output length k.
        norm: LayerNorm for output (unused in forward).
    """
    
    def __init__(self, in_channels=1, output_dim=4096, k=16):
        """Initialize the ResNet encoder.
        
        Args:
            in_channels: Number of input channels (default: 1 for univariate).
            output_dim: Output embedding dimension (default: 4096 for LLaMA).
            k: Number of output tokens/time steps (default: 16).
        """
        super().__init__()
        self.layer1 = ResidualBlock1D(in_channels, 512, downsample=True)
        self.layer2 = ResidualBlock1D(512, 1024, downsample=True)
        self.layer3 = ResidualBlock1D(1024, 2048, downsample=True)
        self.layer4 = ResidualBlock1D(2048, output_dim, downsample=True)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(output_size=k)

    def forward(self, x):
        """Forward pass through the ResNet encoder.
        
        Args:
            x: Input tensor of shape [B, T] where B is batch size and T is sequence length.
            
        Returns:
            Tensor of shape [B, k, output_dim] suitable for LLM input.
        """
        x = x.unsqueeze(1)  # [B, 1, seq_len] - add channel dimension
        x = self.layer1(x)  # [B, 512, T/2]
        x = self.layer2(x)  # [B, 1024, T/4]
        x = self.layer3(x)  # [B, 2048, T/8]
        x = self.layer4(x)  # [B, 4096, T/16]
        x = self.pool(x)    # [B, output_dim, k]
        x = x.permute(0, 2, 1)  # [B, k, output_dim]
        return x
