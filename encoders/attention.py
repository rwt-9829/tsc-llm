"""Attention-augmented CNN encoder for time series embedding.

Provides a convolutional encoder with channel-wise attention mechanism
to selectively weight temporal features before pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN Encoder with Channel-wise Attention.
    
    Architecture: 4 conv layers with LayerNorm, PReLU, and dropout,
    followed by a channel-wise attention mechanism that weights
    temporal positions before adaptive pooling.
    
    Attributes:
        conv1-4: Convolutional layers with increasing channel dimensions.
        norm1-4: LayerNorm for each conv output.
        attention_query: 1x1 conv for attention query projection.
        attention_softmax: 1x1 conv for attention weight computation.
        pool: Adaptive average pooling to fixed output length k.
    """
    
    def __init__(self, input_channels=1, output_dim=4096, window_size=3, stride=2, k=16):
        """Initialize the attention-augmented CNN encoder.
        
        Args:
            input_channels: Number of input channels (default: 1 for univariate).
            output_dim: Output embedding dimension (default: 4096 for LLaMA).
            window_size: Convolutional kernel size (default: 3).
            stride: Convolution stride for downsampling (default: 2).
            k: Number of output tokens/time steps (default: 16).
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 512, kernel_size=window_size, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=window_size, stride=stride, padding=1)
        self.conv3 = nn.Conv1d(1024, 2048, kernel_size=window_size, stride=stride, padding=1)
        self.conv4 = nn.Conv1d(2048, output_dim, kernel_size=window_size, stride=stride, padding=1)

        # LayerNorm for each convolutional block
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(1024)
        self.norm3 = nn.LayerNorm(2048)
        self.norm4 = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(0.2)
        self.act = nn.PReLU()

        # Channel-wise attention: computes attention weights over time dimension
        self.attention_query = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.attention_softmax = nn.Conv1d(output_dim, output_dim, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(output_size=k)

    def forward(self, x):
        """Forward pass through the attention-augmented CNN encoder.
        
        Args:
            x: Input tensor of shape [B, T] or [B, 1, T].
            
        Returns:
            Tensor of shape [B, k, output_dim] suitable for LLM input.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, seq_len] - add channel dimension

        # Progressive feature extraction with norm, activation, dropout
        x = self.conv1(x)
        x = self._norm_act_drop(x, self.norm1)

        x = self.conv2(x)
        x = self._norm_act_drop(x, self.norm2)

        x = self.conv3(x)
        x = self._norm_act_drop(x, self.norm3)

        x = self.conv4(x)
        x = self._norm_act_drop(x, self.norm4)  # [B, output_dim, T_out]

        # Apply channel-wise attention: weight temporal positions
        query = self.attention_query(x)          # [B, output_dim, T_out]
        attn_weights = self.attention_softmax(x) # [B, output_dim, T_out]
        attn_weights = F.softmax(attn_weights, dim=2)  # Softmax over time dimension

        x = query * attn_weights  # Element-wise attention weighting

        x = self.pool(x)  # [B, output_dim, k]
        
        x = x.permute(0, 2, 1)  # [B, k, output_dim]

        return x

    def _norm_act_drop(self, x, norm_layer):
        """Apply LayerNorm, activation, and dropout.
        
        Args:
            x: Input tensor of shape [B, C, T].
            norm_layer: LayerNorm layer to apply.
            
        Returns:
            Normalized, activated, and dropout-applied tensor.
        """
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = norm_layer(x)
        x = x.transpose(1, 2)  # back to [B, C, T]
        x = self.act(x)
        x = self.dropout(x)
        return x