"""Time Series Classification with LLM-based Models.

This module provides training and evaluation pipelines for time series
classification using various encoder architectures combined with LLM embeddings.
It supports multiple UCR datasets and provides configurable training options.
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import TS_CLS, TS_CLS_NoLLM, TS_CLS_NoEncoder
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import logging
import re
import gc

from transformers import AutoModelForCausalLM

def parse_args():
    """Parse command-line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments including data paths, model settings,
            training hyperparameters, and device configuration.
    """
    parser = argparse.ArgumentParser(description='Time Series Classification Training')
    parser.add_argument('--data_path', type=str, default='/home/shuheng/data/training/UCR_TS_Archive_2015/',
                        help='Path to UCR time series dataset directory')
    parser.add_argument('--model_path', type=str, default='/shared/models/hf/Llama-3.1-8B')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--opt', type=str, default = 'adam', choices=['sgd', 'adam'])
    parser.add_argument('--model', type=str, default = 'no_llm', choices=['llm', 'no_llm', 'no_encoder'])
    parser.add_argument('--encoder', type=str, default = 'sl_encoder', choices=['cnn', 'attention', 'mlp', 'resnet', 'sl_encoder'])
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--log_file', type=str, default='log.txt', help='Path to output log file')
    return parser.parse_args()

def setup_logging(base_name="log.txt"):
    """Configure logging with auto-incrementing log file names.
    
    Creates a new log file in the 'logs' directory with an incrementing
    index to avoid overwriting previous logs (e.g., log.txt, log1.txt, log2.txt).
    
    Args:
        base_name: Base name for log files (default: 'log.txt').
        
    Returns:
        str: Name of the created log file.
    """
    os.makedirs("logs", exist_ok=True)
    
    # Extract name and extension for pattern matching
    name, ext = os.path.splitext(base_name)
    existing_files = os.listdir("logs")
    
    pattern = re.compile(rf"{re.escape(name)}(\d*){re.escape(ext)}")
    max_index = -1
    for f in existing_files:
        match = pattern.fullmatch(f)
        if match:
            index_str = match.group(1)
            index = int(index_str) if index_str else 0
            max_index = max(max_index, index)
    
    next_index = max_index + 1
    new_log_file = f"{name}{next_index if next_index > 0 else ''}{ext}"
    full_path = os.path.join("logs", new_log_file)

    open(full_path, 'w').close()
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(full_path),
            logging.StreamHandler()
        ]
    )

    return new_log_file

def load_data(data_path):
    """Load and preprocess UCR time series data from CSV file.
    
    UCR format: first column contains labels, remaining columns contain
    time series values. Labels are normalized to consecutive integers
    starting from 0.
    
    Args:
        data_path: Path to the CSV file containing time series data.
        
    Returns:
        tuple: (features, labels) as torch tensors.
            - features: FloatTensor of shape [N, T] where N is number of samples
              and T is time series length.
            - labels: LongTensor of shape [N] with normalized class labels.
    """
    data = pd.read_csv(data_path, header=None)
    labels = data.iloc[:, 0].values
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    normalized_labels = [label_mapping[label] for label in labels]
    features = data.iloc[:, 1:].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(normalized_labels, dtype=torch.long)

def train(model, train_dataloader, test_dataloader, args):
    """Train the model and evaluate on test set after each epoch.
    
    Supports SGD with step LR scheduler or AdamW optimizer. Implements
    early stopping when training accuracy reaches 99.5%.
    
    Args:
        model: The time series classification model to train.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test/validation data.
        args: Training arguments containing optimizer type, learning rate, epochs, etc.
        
    Returns:
        tuple: (best_test_accuracy, test_accuracy_at_min_train_loss)
            - best_test_accuracy: Highest test accuracy achieved during training.
            - test_accuracy_at_min_train_loss: Test accuracy when training loss was lowest.
    """
    # Configure optimizer based on args.opt
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            list(model.encoder.parameters()) + list(model.pred_head.parameters()), lr=args.lr, weight_decay = 1e-5, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    if args.opt == 'adam':
        optimizer = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(model.pred_head.parameters()), lr=args.lr, weight_decay = 1e-5)
   
    criterion = nn.CrossEntropyLoss()


    # Track best metrics across epochs
    best_acc = 0
    best_loss = float('inf')
    test_acc_at_min_train_loss = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        itr = 0
        
        for time_series, labels in train_dataloader:
            itr += 1
            time_series, labels = time_series.to(args.device), labels.to(args.device)
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            # Forward pass
            result = model(time_series)
            loss = criterion(result, labels)

            # Backward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            _, predicted = torch.max(result, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        if args.opt == 'sgd':
            scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {total_loss / len(train_dataloader):.4f}, Training Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        test_acc, test_loss = test(model, test_dataloader, args)

        # Track test accuracy at minimum training loss
        if total_loss / len(train_dataloader) < best_loss:
            best_loss = total_loss / len(train_dataloader)
            test_acc_at_min_train_loss = test_acc

        best_acc = max(best_acc, test_acc)
        
        # Early stopping if training accuracy is high enough
        if accuracy >= 0.995:
            break

    return best_acc, test_acc_at_min_train_loss

def test(model, dataloader, args):
    """Evaluate the model on a dataset.
    
    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader containing test/validation data.
        args: Arguments containing device configuration.
        
    Returns:
        tuple: (accuracy, average_loss) on the evaluation dataset.
    """
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for time_series, labels in dataloader:
            time_series, labels = time_series.to(args.device), labels.to(args.device)

            result = model(time_series)
            loss = criterion(result, labels)
            total_loss += loss.item()

            _, predicted = torch.max(result, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    test_loss = total_loss / len(dataloader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    return accuracy, test_loss

def main():
    """Main entry point for training time series classification models.
    
    Iterates through all UCR datasets in the specified directory,
    trains a model on each, and logs the results.
    """
    args = parse_args()
    args.log_file = setup_logging()

    # Clear GPU memory and set random seeds for reproducibility
    torch.cuda.empty_cache()
    gc.collect()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Model configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("-" * 30)

    
    subfolders = [f.path for f in os.scandir(args.data_path) if f.is_dir()]

    best_accs = []
    best_losses = []

    ovr_start = time.time()

    subfolders.sort()

    for subfolder in subfolders:
        dataset_name = os.path.basename(subfolder)

        print(f"dataset_name: {dataset_name}")

        train_data_path = os.path.join(subfolder, f"{dataset_name}_TRAIN")
        test_data_path = os.path.join(subfolder, f"{dataset_name}_TEST")

        train_features, train_labels = load_data(train_data_path)
        test_features, test_labels = load_data(test_data_path)

        # Min-max normalization using global min/max from both train and test
        feature_min = min(train_features.min(), test_features.min())
        feature_max = max(train_features.max(), test_features.max())
        train_features = (train_features - feature_min) / (feature_max - feature_min)
        test_features = (test_features - feature_min) / (feature_max - feature_min)

        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Determine number of unique classes
        classes = len(torch.unique(torch.cat((train_labels, test_labels))))

        # Initialize model based on configuration
        if args.model == 'llm':
            # Full model with encoder + LLM
            model = TS_CLS(args.encoder, args.model_path, classes, train_features.size(1), args.device, args.num_kernels, args.kernel_size).to(args.device)
        elif args.model == 'no_llm':
            # Encoder only, no LLM inference
            model = TS_CLS_NoLLM(args.encoder, args.model_path, classes, train_features.size(1), args.device, args.num_kernels, args.kernel_size).to(args.device)
        elif args.model == 'no_encoder':
            # LLM-only with text prompts
            model = TS_CLS_NoEncoder(args.model_path, classes, args.device).to(args.device)

        train_start = time.time()
        best_acc, acc_min_train_loss = train(model, train_dataloader, test_dataloader, args)
        best_accs.append(best_acc)
        best_losses.append(acc_min_train_loss)

        logging.info(f"{dataset_name}: {best_acc}, {acc_min_train_loss}")
        logging.info(f"{dataset_name} time: {time.time() - train_start}")

        # Clean up to free GPU memory before next dataset
        del model
        del train_features, train_labels, test_features, test_labels
        del train_dataset, test_dataset
        del train_dataloader, test_dataloader
        torch.cuda.empty_cache()
        gc.collect()

    for i in best_accs:
        logging.info(i)
    
    logging.info("-----------------------------")

    for i in best_losses:
        logging.info(i)

    logging.info(f"total running time: {time.time() - ovr_start}")
    avg_best_acc = sum(best_accs) / len(best_accs) if best_accs else 0
    avg_best_loss = sum(best_losses) / len(best_losses) if best_losses else 0
    logging.info(f"Average best accuracy: {avg_best_acc}")
    logging.info(f"Average accuracy at min train loss: {avg_best_loss}")

if __name__ == "__main__":
    main()