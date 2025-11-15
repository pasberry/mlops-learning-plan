"""
Config-driven MNIST training script - SOLUTION
Demonstrates best practices for ML experiment management with configuration files.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from pathlib import Path
import argparse
import json
from datetime import datetime

from config_loader import load_config, save_config, set_seed, get_device, validate_config


class ConfigurableCNN(nn.Module):
    """Configurable CNN based on config parameters."""

    def __init__(self, config):
        """
        Initialize model with architecture from config.

        Args:
            config: Config object with model.architecture parameters
        """
        super(ConfigurableCNN, self).__init__()
        arch = config.model.architecture

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, arch.conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(arch.conv1_channels, arch.conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(arch.dropout_conv)

        # Fully connected layers
        # After 2 pooling layers: 28->14->7
        fc_input_size = arch.conv2_channels * 7 * 7
        self.fc1 = nn.Linear(fc_input_size, arch.fc1_size)
        self.dropout2 = nn.Dropout(arch.dropout_fc)
        self.fc2 = nn.Linear(arch.fc1_size, 10)

    def forward(self, x):
        """Forward pass."""
        # Conv block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def get_dataloaders(config):
    """
    Create dataloaders from config.

    Args:
        config: Config object

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load datasets
    train_dataset = datasets.MNIST(
        config.data.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        config.data.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, writer):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Config object
        writer: TensorBoard writer

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Logging
        if batch_idx % config.logging.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), step)
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]  '
                  f'Loss: {loss.item():.6f}')

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """
    Validate the model.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def save_checkpoint(model, optimizer, epoch, metrics, config, filepath):
    """
    Save checkpoint with metrics and config.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        config: Config object
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def save_experiment_summary(config, metrics, output_dir):
    """
    Save experiment summary as JSON.

    Args:
        config: Config object
        metrics: Final metrics dictionary
        output_dir: Directory to save summary
    """
    summary = {
        'experiment_name': config.experiment.name,
        'description': config.experiment.description,
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'final_metrics': metrics,
    }

    summary_path = Path(output_dir) / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Experiment summary saved: {summary_path}")


def get_optimizer(model, config):
    """
    Create optimizer from config.

    Args:
        model: PyTorch model
        config: Config object

    Returns:
        Optimizer instance
    """
    optimizer_name = config.training.optimizer.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer instance
        config: Config object

    Returns:
        Scheduler instance or None
    """
    if not config.training.scheduler.enabled:
        return None

    if config.training.scheduler.type == "ReduceLROnPlateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler.factor,
            patience=config.training.scheduler.patience,
            min_lr=config.training.scheduler.min_lr,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.training.scheduler.type}")


def main(config_path):
    """
    Main training function.

    Args:
        config_path: Path to configuration file
    """
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"{'='*60}\n")

    # Set seed for reproducibility
    set_seed(config.device.seed)
    print(f"Random seed: {config.device.seed}")

    # Get device
    device = get_device(config)

    # Create output directory
    output_dir = Path(config.logging.checkpoint_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config to output directory
    save_config(config, output_dir / 'config.yaml')

    # Load data
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = ConfigurableCNN(config).to(device)
    print(f"\nModel: {config.model.name}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, config)
    if scheduler:
        print(f"Scheduler: {config.training.scheduler.type}")

    # TensorBoard writer
    tensorboard_dir = Path(config.logging.tensorboard_dir) / config.experiment.name
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")

    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    print(f"{'='*60}\n")

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.training.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, writer
        )

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if scheduler:
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nTraining: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }

        if not config.logging.save_best_only:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, metrics, config, checkpoint_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = output_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, metrics, config, best_path)
            print(f"✓ New best accuracy: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping.enabled:
            if patience_counter >= config.training.early_stopping.patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"No improvement for {patience_counter} epochs")
                break

    # Save experiment summary
    final_metrics = {
        'best_val_acc': best_acc,
        'final_epoch': epoch,
        'total_epochs_trained': epoch,
    }
    save_experiment_summary(config, final_metrics, output_dir)

    writer.close()

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"View TensorBoard: tensorboard --logdir={config.logging.tensorboard_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST classifier with config')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    main(args.config)
