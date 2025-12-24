# Time Series Classification with LLM-based Models (TSC-LLM)

A deep learning framework for time series classification that combines various encoder architectures with Large Language Model (LLM) embeddings. This project leverages pretrained LLMs (e.g., LLaMA) to enhance time series classification performance on UCR benchmark datasets.

## Overview

This project provides three model variants for time series classification:

1. **TS_CLS (LLM)**: Full model that encodes time series data into LLM embedding space and uses the LLM's hidden states for classification
2. **TS_CLS_NoLLM**: Encoder-only model that maps time series to embedding space without LLM inference (faster training)
3. **TS_CLS_NoEncoder**: LLM-only model using text prompts for few-shot classification

### Supported Encoders

- **sl_encoder**: Inception-based encoder with multi-scale convolutional kernels (default)
- **cnn**: Simple stacked convolutional encoder
- **attention**: CNN with channel-wise attention mechanism
- **mlp**: Multi-layer perceptron encoder
- **resnet**: ResNet-style encoder with residual connections

## Environment Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **VRAM**: At least 16GB GPU memory for LLM-based models (LLaMA-3.1-8B)
- **RAM**: 32GB+ recommended

### Software Requirements

- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
pandas
matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tsc-llm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas matplotlib
```

## Dataset

This project uses the **UCR Time Series Archive** datasets. Download the UCR dataset from:
- [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

Expected data format:
- CSV files where the first column contains labels
- Remaining columns contain time series values
- Directory structure: `UCR_TS_Archive_2015/<DatasetName>/<DatasetName>_TRAIN` and `<DatasetName>_TEST`

## Usage

### Basic Training

Run the training script with default parameters:

```bash
python main.py --data_path /path/to/UCR_TS_Archive_2015/ --model_path /path/to/Llama-3.1-8B
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | `/home/shuheng/data/training/UCR_TS_Archive_2015/` | Path to UCR time series dataset directory |
| `--model_path` | str | `/shared/models/hf/Llama-3.1-8B` | Path to pretrained LLM |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `0.01` | Learning rate |
| `--batch_size` | int | `32` | Batch size for training |
| `--num_kernels` | int | `6` | Number of convolutional kernels in encoder |
| `--kernel_size` | int | `16` | Kernel size / number of output tokens |
| `--opt` | str | `adam` | Optimizer choice: `sgd` or `adam` |
| `--model` | str | `no_llm` | Model type: `llm`, `no_llm`, or `no_encoder` |
| `--encoder` | str | `sl_encoder` | Encoder type: `cnn`, `attention`, `mlp`, `resnet`, `sl_encoder` |
| `--device` | str | `cuda:0` | Device for training (auto-detects CUDA) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--log_file` | str | `log.txt` | Output log file name |

### Example Commands

**Train with full LLM model:**
```bash
python main.py --model llm --encoder sl_encoder --epochs 100 --lr 0.01
```

**Train encoder-only model (faster, no LLM inference):**
```bash
python main.py --model no_llm --encoder resnet --epochs 50 --lr 0.001
```

**Train with SGD optimizer:**
```bash
python main.py --opt sgd --lr 0.1 --epochs 100
```

**Train on specific GPU:**
```bash
python main.py --device cuda:1
```

## Project Structure

```
tsc-llm/
├── main.py              # Main training script with data loading and training loop
├── model.py             # Model architectures (TS_CLS, TS_CLS_NoLLM, TS_CLS_NoEncoder)
├── README.md            # This file
└── encoders/            # Encoder modules
    ├── __init__.py      # Package initialization
    ├── attention.py     # CNN with channel-wise attention
    ├── cnn.py           # Simple convolutional encoder
    ├── mlp.py           # Multi-layer perceptron encoder
    ├── resnet.py        # ResNet-style encoder
    └── sl_encoder.py    # Inception-based multi-scale encoder
```

## Output

Training logs are saved to the `logs/` directory with auto-incrementing filenames (e.g., `log.txt`, `log1.txt`, `log2.txt`).

Each log contains:
- Model configuration
- Per-dataset training progress
- Best accuracy and accuracy at minimum training loss
- Total running time
- Average metrics across all datasets

## Notes

- The LLM weights are frozen during training; only the encoder and prediction head are trainable
- Early stopping is triggered when training accuracy reaches 99.5%
- Data is normalized using min-max normalization before training
- GPU memory is automatically cleared between datasets to prevent OOM errors