"""Training entry point for neural models on UrbanSound8K."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data.urban_sound import UrbanSoundDataset, build_samples_for_folds, collate_batch, create_dataloaders
from src.features.extractor import FeatureExtractor
from src.models.cnn import CNNClassifier
from src.models.cnn1d import CNN1DClassifier
from src.models.dnn import DNNClassifier
from src.models.lstm import LSTMClassifier
from src.utils.metrics import (
    compute_classification_metrics,
    save_classification_report,
    save_confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UrbanSound8K neural models.")
    parser.add_argument("--model", choices=["dnn", "cnn", "cnn1d", "lstm"], default="cnn")
    parser.add_argument("--clear-cache", action="store_true", help="Clear feature cache before training")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test/validation split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train_test_split (default: 42)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(model_name: str, sample_bundle) -> nn.Module:
    if model_name == "dnn":
        input_dim = sample_bundle.stats.numel()
        return DNNClassifier(input_dim=input_dim, num_classes=len(config.CLASS_NAMES))
    if model_name == "cnn":
        return CNNClassifier(in_channels=sample_bundle.sequence.shape[0], num_classes=len(config.CLASS_NAMES))
    if model_name == "cnn1d":
        return CNN1DClassifier(num_classes=len(config.CLASS_NAMES))
    if model_name == "lstm":
        # sequence shape: (channels, n_mfcc, time) = (1, 128, time)
        # LSTM input_size should be n_mfcc (number of features per time step)
        _, n_mfcc, _ = sample_bundle.sequence.shape
        return LSTMClassifier(input_size=n_mfcc, num_classes=len(config.CLASS_NAMES))
    raise ValueError(f"Unsupported model {model_name}")


def print_model_info(model: nn.Module, model_name: str, device: torch.device) -> None:
    """Print model architecture and parameter information."""
    print("\n" + "=" * 60)
    print(f"Model: {model_name.upper()}")
    print("=" * 60)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Device info
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    print("=" * 60 + "\n")


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    """Train one epoch matching debug.ipynb approach.
    
    Matches debug.ipynb training:
    - Loss: sum loss.item() for each batch, then divide by number of batches
    """
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for batch, labels in tqdm(loader, desc="Train", leave=False):
        stats = batch["stats"].to(device)
        sequence = batch["sequence"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if isinstance(model, DNNClassifier):
            outputs = model(stats)
        elif isinstance(model, (CNNClassifier, CNN1DClassifier, LSTMClassifier)):
            outputs = model(sequence)
        else:
            outputs = model(sequence)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Match debug.ipynb: accumulate loss.item() (not multiplied by batch size)
        train_loss += loss.item()
        num_batches += 1
    
    # Calculate average loss: divide by number of batches (matching debug.ipynb)
    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
    return avg_train_loss


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluate model matching debug.ipynb approach.
    
    Matches debug.ipynb evaluation:
    - Loss: sum loss.item() for each batch, then divide by number of batches
    - Accuracy: correct / total (0-1 range)
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    num_batches = 0
    with torch.no_grad():
        for batch, labels in tqdm(loader, desc="Eval", leave=False):
            stats = batch["stats"].to(device)
            sequence = batch["sequence"].to(device)
            labels = labels.to(device)
            
            if isinstance(model, DNNClassifier):
                outputs = model(stats)
            elif isinstance(model, (CNNClassifier, CNN1DClassifier, LSTMClassifier)):
                outputs = model(sequence)
            else:
                outputs = model(sequence)
            
            loss = criterion(outputs, labels)
            
            # Match debug.ipynb: accumulate loss.item() (not multiplied by batch size)
            test_loss += loss.item()
            num_batches += 1
            
            # Get predictions (match debug.ipynb: torch.max(outputs.data, 1))
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate average loss: divide by number of batches (matching debug.ipynb)
    avg_loss = test_loss / num_batches if num_batches > 0 else 0.0
    
    # Calculate accuracy (0-1 range, matching debug.ipynb)
    accuracy = test_correct / test_total if test_total > 0 else 0.0
    
    # Compute additional metrics
    metrics = compute_classification_metrics(all_labels, all_preds)
    # Override accuracy with the calculated one (to match debug.ipynb format)
    metrics["accuracy"] = accuracy
    
    return avg_loss, metrics, all_labels, all_preds


def plot_training_history(history: list, results_dir: Path, model_name: str, suffix: str) -> None:
    """Plot and save accuracy and loss graphs from training history.
    
    Args:
        history: Training history list
        results_dir: Directory to save plots
        model_name: Name of the model
        suffix: Custom suffix used in artifact filenames (e.g., "random_rs42")
    """
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_accs = [h["accuracy"] for h in history]
    
    # Extract train accuracy (may not exist for all epochs)
    train_acc_epochs = []
    train_accs = []
    for h in history:
        if "train_accuracy" in h and h["train_accuracy"] is not None:
            train_acc_epochs.append(h["epoch"])
            train_accs.append(h["train_accuracy"])
    
    # Create figure with 2 subplots (similar to debug.ipynb)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    axes[0].plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.upper()} - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    if train_acc_epochs:  # Only plot train accuracy if available
        axes[1].plot(train_acc_epochs, train_accs, label='Train Accuracy', marker='o', markersize=4)
    axes[1].plot(epochs, val_accs, label='Validation Accuracy', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name.upper()} - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = results_dir / f"{model_name}_{suffix}_training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    device = torch.device(args.device)

    # Clear cache if requested (useful when feature extraction logic changes)
    if args.clear_cache and config.CACHE_DIR.exists():
        import shutil
        print(f"Clearing feature cache at {config.CACHE_DIR}...")
        shutil.rmtree(config.CACHE_DIR)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    train_loader, val_loader = create_dataloaders(
        extractor=extractor,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    feature_bundle, _ = train_loader.dataset[0]
    model = build_model(args.model, feature_bundle).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Print model information
    print_model_info(model, args.model, device)
    
    # Print training configuration
    print("Training Configuration:")
    print(f"  Test size: {args.test_size}")
    print(f"  Random state: {args.random_state}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: Adam")
    print(f"  Loss function: CrossEntropyLoss")
    print("=" * 60)
    print(f"\nStarting training for {args.epochs} epochs...\n")

    best_val_f1 = 0.0
    history = []
    checkpoint_dir = config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_loss, metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        
        # Calculate train accuracy (only every 5 epochs or last epoch to save time)
        train_acc = None
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for batch, labels in train_loader:
                    stats = batch["stats"].to(device)
                    sequence = batch["sequence"].to(device)
                    labels = labels.to(device)
                    if isinstance(model, DNNClassifier):
                        outputs = model(stats)
                    else:
                        outputs = model(sequence)
                    preds = outputs.argmax(dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            model.train()
        
        metrics["train_loss"] = train_loss
        metrics["val_loss"] = val_loss
        if train_acc is not None:
            metrics["train_accuracy"] = train_acc
        metrics["epoch"] = epoch
        history.append(metrics)
        
        # Log every 5 epochs or at the last epoch
        if epoch % 20 == 0 or epoch == args.epochs:
            train_acc_str = f", train_acc={train_acc:.4f}" if train_acc is not None else ""
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{train_acc_str}, "
                  f"val_acc={metrics['accuracy']:.4f}, val_f1={metrics['f1']:.4f}")

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            suffix = f"random_rs{args.random_state}"
            ckpt_path = checkpoint_dir / f"{args.model}_{suffix}.pt"
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
            save_confusion_matrix(
                y_true,
                y_pred,
                config.CLASS_NAMES,
                results_dir / f"{args.model}_{suffix}_cm.png",
            )
            save_classification_report(
                y_true,
                y_pred,
                config.CLASS_NAMES,
                results_dir / f"{args.model}_{suffix}_report.txt",
            )

    # Save training history
    suffix = f"random_rs{args.random_state}"
    history_path = results_dir / f"{args.model}_{suffix}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    
    # Plot and save accuracy and loss graphs
    print("\nGenerating training plots...")
    plot_training_history(history, results_dir, args.model, suffix=suffix)


if __name__ == "__main__":
    main()
