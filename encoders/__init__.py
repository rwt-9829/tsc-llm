"""Encoder modules for time series to LLM embedding transformation.

This package provides various encoder architectures that transform
raw time series data into embeddings compatible with LLM hidden dimensions (4096).

Available encoders:
    - attention: CNN with channel-wise attention mechanism
    - cnn: Simple stacked convolutional encoder
    - mlp: Multi-layer perceptron encoder
    - resnet: ResNet-style encoder with residual connections
    - sl_encoder: Inception-based encoder with multi-scale kernels
"""

from . import attention
from . import cnn
from . import mlp
from . import resnet
from . import sl_encoder