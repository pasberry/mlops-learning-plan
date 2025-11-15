# Lab 1.3: Config-Driven Training

**Goal**: Separate configuration from code for reproducible experiments

**Estimated Time**: 45-60 minutes

**Prerequisites**:
- Lab 1.2 completed
- Understanding of YAML format

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Understand why configuration management matters in ML
- ✅ Create YAML configuration files
- ✅ Load and validate configs in Python
- ✅ Run experiments with different configurations
- ✅ Track experiment parameters and results

---

## Background: Why Config-Driven Development?

### The Problem

```python
# BAD: Hardcoded parameters scattered throughout code
model = SimpleCNN(hidden_size=128)
optimizer = Adam(model.parameters(), lr=0.001)
train(model, epochs=10, batch_size=64)
```

**Issues**:
- Hard to reproduce experiments
- Parameters scattered across files
- Can't easily compare runs
- No record of what worked

### The Solution

```yaml
# config.yaml
model:
  name: SimpleCNN
  hidden_size: 128

training:
  learning_rate: 0.001
  epochs: 10
  batch_size: 64
```

```python
# GOOD: Load config
config = load_config('config.yaml')
model = SimpleCNN(hidden_size=config['model']['hidden_size'])
```

**Benefits**:
- ✅ Single source of truth
- ✅ Easy to reproduce
- ✅ Version control friendly
- ✅ Compare experiments by comparing configs

---

## Lab Instructions

### Part 1: Create Configuration File

Create `config/mnist_config.yaml`:

```yaml
# MNIST Training Configuration

# Experiment metadata
experiment:
  name: "mnist_baseline"
  description: "Baseline MNIST classifier with CNN"
  tags:
    - mnist
    - cnn
    - baseline

# Data configuration
data:
  dataset: "MNIST"
  data_dir: "./data"
  train_split: 0.9
  val_split: 0.1
  num_workers: 2

# Model configuration
model:
  name: "SimpleCNN"
  architecture:
    conv1_channels: 32
    conv2_channels: 64
    fc1_size: 128
    dropout_conv: 0.25
    dropout_fc: 0.5

# Training configuration
training:
  batch_size: 64
  num_epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0001

  # Learning rate scheduler
  scheduler:
    enabled: true
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 2
    min_lr: 0.00001

  # Early stopping
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.001

# Logging and checkpointing
logging:
  log_interval: 100  # Log every N batches
  checkpoint_dir: "./models/staging/mnist"
  save_best_only: false
  tensorboard_dir: "./runs"

# Hardware
device:
  use_cuda: true  # Use GPU if available
  seed: 42  # Random seed for reproducibility
```

### Part 2: Create Config Loader

Create `ml/config_loader.py`:

```python
"""Configuration management for ML experiments."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import torch


class Config:
    """Configuration class with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __repr__(self):
        """String representation."""
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with dot notation access
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: For full reproducibility, also set:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device(config: Config) -> torch.device:
    """
    Get device from config.

    Args:
        config: Config object

    Returns:
        torch.device
    """
    if config.device.use_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config('config/mnist_config.yaml')

    # Access with dot notation
    print(f"Experiment: {config.experiment.name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Access with dictionary notation
    print(f"Model: {config['model']['name']}")

    # Convert to dict
    print(f"\nFull config:\n{config.to_dict()}")
```

### Part 3: Update Training Script

Create `ml/train_mnist_config.py`:

```python
"""
Config-driven MNIST training script.
Demonstrates best practices for ML experiment management.
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

from config_loader import load_config, save_config, set_seed, get_device


class ConfigurableCNN(nn.Module):
    """Configurable CNN based on config parameters."""

    def __init__(self, config):
        super(ConfigurableCNN, self).__init__()
        arch = config.model.architecture

        self.conv1 = nn.Conv2d(1, arch.conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(arch.conv1_channels, arch.conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(arch.dropout_conv)

        self.fc1 = nn.Linear(arch.conv2_channels * 7 * 7, arch.fc1_size)
        self.dropout2 = nn.Dropout(arch.dropout_fc)
        self.fc2 = nn.Linear(arch.fc1_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = x.view(-1, self.fc2.in_features)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def get_dataloaders(config):
    """Create dataloaders from config."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

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
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

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

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate the model."""
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
    """Save checkpoint with metrics."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def save_experiment_summary(config, metrics, output_dir):
    """Save experiment summary as JSON."""
    summary = {
        'experiment_name': config.experiment.name,
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'final_metrics': metrics,
    }

    summary_path = Path(output_dir) / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Experiment summary saved: {summary_path}")


def main(config_path):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print(f"Loaded config: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")

    # Set seed for reproducibility
    set_seed(config.device.seed)

    # Get device
    device = get_device(config)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config.logging.checkpoint_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    save_config(config, output_dir / 'config.yaml')

    # Load data
    print("Loading data...")
    train_loader, test_loader = get_dataloaders(config)

    # Initialize model
    model = ConfigurableCNN(config).to(device)
    print(f"Model initialized: {config.model.name}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if config.training.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

    # Learning rate scheduler
    scheduler = None
    if config.training.scheduler.enabled:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler.factor,
            patience=config.training.scheduler.patience,
            min_lr=config.training.scheduler.min_lr,
            verbose=True
        )

    # TensorBoard writer
    writer = SummaryWriter(
        Path(config.logging.tensorboard_dir) / config.experiment.name
    )

    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
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

        print(f"Training: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
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
            print(f"New best accuracy: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping.enabled:
            if patience_counter >= config.training.early_stopping.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    # Save experiment summary
    final_metrics = {
        'best_val_acc': best_acc,
        'final_epoch': epoch,
    }
    save_experiment_summary(config, final_metrics, output_dir)

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST classifier with config')
    parser.add_argument(
        '--config',
        type=str,
        default='config/mnist_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    main(args.config)
```

### Part 4: Run Experiments

```bash
# Run with default config
python ml/train_mnist_config.py --config config/mnist_config.yaml

# Create a new config for experiment 2
cp config/mnist_config.yaml config/mnist_large.yaml

# Edit mnist_large.yaml to change parameters:
# - experiment.name: "mnist_large"
# - model.architecture.conv2_channels: 128
# - model.architecture.fc1_size: 256

# Run experiment 2
python ml/train_mnist_config.py --config config/mnist_large.yaml

# View in TensorBoard
tensorboard --logdir=runs
```

---

## Exercise 1: Create Config Variants

Create configs for different experiments:

**config/mnist_small.yaml**: Smaller, faster model
```yaml
model:
  architecture:
    conv1_channels: 16
    conv2_channels: 32
    fc1_size: 64
```

**config/mnist_sgd.yaml**: Use SGD instead of Adam
```yaml
training:
  optimizer: "sgd"
  learning_rate: 0.01
```

**config/mnist_overfit.yaml**: Test overfitting
```yaml
training:
  num_epochs: 50
  early_stopping:
    enabled: false
```

Run all configs and compare results in TensorBoard.

---

## Exercise 2: Add Config Validation

Create `ml/config_validator.py`:

```python
"""Validate configuration files."""

def validate_config(config):
    """
    Validate configuration.

    Args:
        config: Config object

    Raises:
        ValueError: If config is invalid
    """
    # Check required fields
    required_fields = ['experiment', 'data', 'model', 'training']
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required field: {field}")

    # Validate ranges
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    print("✅ Config validation passed")
    return True
```

---

## Exercise 3: Command-Line Config Overrides

Add ability to override config values from command line:

```python
def override_config(config, overrides):
    """
    Override config values from command line.

    Args:
        config: Config object
        overrides: List of "key.nested.key=value" strings
    """
    for override in overrides:
        keys, value = override.split('=')
        keys = keys.split('.')

        # Navigate to nested key
        obj = config
        for key in keys[:-1]:
            obj = getattr(obj, key)

        # Set value (with type conversion)
        setattr(obj, keys[-1], type(getattr(obj, keys[-1]))(value))

# Usage
parser.add_argument('--override', nargs='+', help='Override config values')
args = parser.parse_args()

config = load_config(args.config)
if args.override:
    override_config(config, args.override)

# Now you can run:
# python train.py --config config.yaml --override training.batch_size=128 training.learning_rate=0.0001
```

---

## Exercise 4: Experiment Tracking Table

Create `ml/compare_experiments.py`:

```python
"""Compare multiple experiments."""

import json
from pathlib import Path
import pandas as pd

def load_experiment_results(base_dir='./models/staging/mnist'):
    """Load all experiment results."""
    base_path = Path(base_dir)
    results = []

    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / 'experiment_summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    results.append({
                        'name': data['experiment_name'],
                        'best_acc': data['final_metrics']['best_val_acc'],
                        'epochs': data['final_metrics']['final_epoch'],
                        'lr': data['config']['training']['learning_rate'],
                        'batch_size': data['config']['training']['batch_size'],
                        'optimizer': data['config']['training']['optimizer'],
                    })

    df = pd.DataFrame(results)
    df = df.sort_values('best_acc', ascending=False)
    return df

if __name__ == "__main__":
    results = load_experiment_results()
    print("\nExperiment Comparison:")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
```

---

## Challenge: Hyperparameter Sweep

Create a script that runs multiple experiments with different hyperparameters:

```python
"""Run hyperparameter sweep."""

import itertools
from ml.train_mnist_config import main
from ml.config_loader import load_config, save_config
from pathlib import Path

def run_sweep():
    """Run hyperparameter sweep."""
    # Define hyperparameter grid
    learning_rates = [0.001, 0.0001]
    batch_sizes = [32, 64, 128]

    # Load base config
    base_config = load_config('config/mnist_config.yaml')

    # Run experiments
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        # Update config
        base_config.experiment.name = f"sweep_lr{lr}_bs{bs}"
        base_config.training.learning_rate = lr
        base_config.training.batch_size = bs

        # Save temp config
        temp_config_path = f'config/temp_sweep_{lr}_{bs}.yaml'
        save_config(base_config, temp_config_path)

        # Run training
        print(f"\n{'='*80}")
        print(f"Running: LR={lr}, BS={bs}")
        print(f"{'='*80}")
        main(temp_config_path)

if __name__ == "__main__":
    run_sweep()
```

---

## Key Takeaways

### Benefits of Config-Driven Development

✅ **Reproducibility**: Save config with every experiment
✅ **Experimentation**: Easy to try new parameters
✅ **Comparison**: Compare experiments by comparing configs
✅ **Version Control**: Configs are text files, easy to diff
✅ **Collaboration**: Share configs with team
✅ **Automation**: Easy to sweep hyperparameters

### Best Practices

✅ **One config file per experiment**
✅ **Save config with model checkpoints**
✅ **Use meaningful experiment names**
✅ **Validate configs before training**
✅ **Track configs in version control**
✅ **Document config schema**

---

## Project Structure

After this lab, your project should look like:

```
mlops-learning-plan/
├── config/
│   ├── mnist_config.yaml
│   ├── mnist_large.yaml
│   └── mnist_small.yaml
├── ml/
│   ├── train_mnist.py
│   ├── train_mnist_config.py
│   ├── config_loader.py
│   └── compare_experiments.py
├── models/
│   └── staging/
│       └── mnist/
│           ├── mnist_baseline/
│           │   ├── config.yaml
│           │   ├── best_model.pt
│           │   └── experiment_summary.json
│           └── mnist_large/
│               └── ...
└── runs/  # TensorBoard logs
```

---

## Submission Checklist

- ✅ Config file created and loads successfully
- ✅ Training runs with config
- ✅ Can run multiple experiments with different configs
- ✅ TensorBoard shows multiple runs
- ✅ Experiment summaries are saved
- ✅ At least 2 config variants created

---

## Phase 1 Complete! 🎉

You've now mastered:
- ✅ Airflow DAGs and orchestration
- ✅ PyTorch model training
- ✅ Configuration-driven experiments

**Next up: Phase 2 - Data & Pipelines with Airflow**

---

**Next**: [Phase 2 Overview →](../phase2/README.md)
