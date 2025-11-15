# Lab 3.1: Build Tabular Classifier with PyTorch

**Objective**: Build a production-ready tabular classification model with proper code structure.

**Time**: 2-3 hours

**Prerequisites**:
- Phase 2 completed (feature data available)
- PyTorch installed
- Understanding of neural networks basics

---

## What You'll Build

A click-through rate (CTR) prediction model that:
- Loads tabular features from Phase 2
- Trains a multi-layer perceptron (MLP)
- Evaluates on validation set
- Saves model checkpoints
- Uses proper code organization

**Use Case**: Predict whether a user will click on an ad given features like user demographics, ad type, time of day, etc.

---

## Architecture Overview

```
Input Features (e.g., 50 dims)
       ↓
   Linear(50 → 256)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(256 → 128)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(128 → 64)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(64 → 1)
       ↓
   Sigmoid (output probability)
```

---

## Step 1: Project Structure Setup

Create the directory structure:

```bash
cd /home/user/mlops-learning-plan

# Create directories
mkdir -p ml/{data,models,training,registry}
mkdir -p config
mkdir -p models/staging/ctr_model
mkdir -p experiments/runs

# Create __init__ files
touch ml/__init__.py
touch ml/data/__init__.py
touch ml/models/__init__.py
touch ml/training/__init__.py
touch ml/registry/__init__.py
```

Your structure should look like:
```
ml/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── datasets.py      # ← We'll create this
│   └── loaders.py       # ← We'll create this
├── models/
│   ├── __init__.py
│   ├── tabular.py       # ← We'll create this
│   └── base.py          # ← We'll create this
└── training/
    ├── __init__.py
    ├── train.py         # ← We'll create this
    └── trainer.py       # ← We'll create this
```

---

## Step 2: Create Sample Feature Data

For this lab, we'll create sample CTR prediction data (simulating Phase 2 output).

**Create**: `scripts/generate_sample_features.py`

```python
"""Generate sample CTR prediction features for training."""
import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_ctr_features(n_samples=10000, n_features=20):
    """Generate synthetic CTR prediction data.

    Features:
    - User features: age, gender, location (one-hot)
    - Ad features: category, size, position
    - Context features: hour, day_of_week, device_type
    - Interaction features: user-ad affinity scores

    Label: 1 if clicked, 0 if not
    """
    np.random.seed(42)

    # User features
    user_age = np.random.randint(18, 65, n_samples)
    user_gender = np.random.randint(0, 2, n_samples)  # 0: F, 1: M

    # Ad features
    ad_category = np.random.randint(0, 10, n_samples)  # 10 categories
    ad_size = np.random.choice(['small', 'medium', 'large'], n_samples)
    ad_position = np.random.randint(1, 6, n_samples)  # 1-5

    # Context features
    hour = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples)

    # Interaction features (synthetic affinity)
    user_ad_affinity = np.random.random(n_samples)
    recency_score = np.random.random(n_samples)

    # One-hot encode categorical features
    ad_size_small = (ad_size == 'small').astype(int)
    ad_size_medium = (ad_size == 'medium').astype(int)
    ad_size_large = (ad_size == 'large').astype(int)

    device_mobile = (device_type == 'mobile').astype(int)
    device_desktop = (device_type == 'desktop').astype(int)
    device_tablet = (device_type == 'tablet').astype(int)

    # Additional engineered features
    is_weekend = (day_of_week >= 5).astype(int)
    is_evening = ((hour >= 18) & (hour <= 22)).astype(int)

    # Create label (CTR) with some logic
    # Higher probability if:
    # - Good user-ad affinity
    # - Premium position
    # - Evening hours
    # - Mobile device
    click_prob = (
        0.1 +  # Base rate
        0.3 * user_ad_affinity +
        0.2 * (6 - ad_position) / 5 +  # Better positions
        0.15 * is_evening +
        0.1 * device_mobile +
        0.05 * (user_age > 30) +
        np.random.normal(0, 0.1, n_samples)  # Noise
    )
    click_prob = np.clip(click_prob, 0, 1)
    label = (np.random.random(n_samples) < click_prob).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'user_age': user_age,
        'user_gender': user_gender,
        'ad_category': ad_category,
        'ad_size_small': ad_size_small,
        'ad_size_medium': ad_size_medium,
        'ad_size_large': ad_size_large,
        'ad_position': ad_position,
        'hour': hour,
        'day_of_week': day_of_week,
        'device_mobile': device_mobile,
        'device_desktop': device_desktop,
        'device_tablet': device_tablet,
        'is_weekend': is_weekend,
        'is_evening': is_evening,
        'user_ad_affinity': user_ad_affinity,
        'recency_score': recency_score,
        'label': label
    })

    # Normalize numerical features
    numerical_cols = ['user_age', 'ad_position', 'hour', 'user_ad_affinity', 'recency_score']
    for col in numerical_cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data


def save_train_val_test_splits(data, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split and save data to parquet files."""
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Save
    for split, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_data.to_parquet(split_dir / 'data.parquet', index=False)
        print(f"Saved {len(split_data)} samples to {split_dir}/data.parquet")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    print(f"  Positive rate (train): {train_data['label'].mean():.3f}")
    print(f"  Features: {len(data.columns) - 1}")


if __name__ == '__main__':
    # Generate features
    print("Generating sample CTR prediction features...")
    data = generate_ctr_features(n_samples=10000)

    # Save splits
    output_dir = 'data/features/v1'
    save_train_val_test_splits(data, output_dir)

    print(f"\n✅ Feature data ready at {output_dir}/")
    print("   Next: Build the PyTorch model!")
```

**Run it**:
```bash
python scripts/generate_sample_features.py
```

---

## Step 3: Create Base Model Interface

**Create**: `ml/models/base.py`

```python
"""Base model interface for all PyTorch models."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models.

    All models should inherit from this class and implement:
    - forward()
    - get_config()
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization.

        Returns:
            Dictionary of model configuration
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model weights and config.

        Args:
            path: Path to save model (without extension)
        """
        import os
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), f"{path}.pt")

        # Save config
        with open(f"{path}_config.json", 'w') as f:
            json.dump(self.get_config(), f, indent=2)

        print(f"Model saved to {path}.pt")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from weights and config.

        Args:
            path: Path to model file (without extension)
            **kwargs: Additional arguments to override config

        Returns:
            Loaded model instance
        """
        import json

        # Load config
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)

        # Override with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(torch.load(f"{path}.pt"))

        print(f"Model loaded from {path}.pt")
        return model
```

---

## Step 4: Create Tabular Model

**Create**: `ml/models/tabular.py`

```python
"""Tabular classification models."""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from ml.models.base import BaseModel


class TabularClassifier(BaseModel):
    """Multi-layer perceptron for tabular classification.

    Architecture:
        Input → Linear → ReLU → Dropout →
        [Linear → ReLU → Dropout] × (n_layers - 1) →
        Linear → Sigmoid

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions (e.g., [256, 128, 64])
        dropout: Dropout probability (default: 0.3)
        output_dim: Output dimension (default: 1 for binary classification)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.output_dim = output_dim

        # Build layers
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        logits = self.layers(x)
        probs = self.sigmoid(logits)
        return probs

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'output_dim': self.output_dim
        }

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"TabularClassifier(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  dropout={self.dropout},\n"
            f"  output_dim={self.output_dim},\n"
            f"  params={n_params:,}\n"
            f")"
        )


# Example usage and testing
if __name__ == '__main__':
    # Create model
    model = TabularClassifier(
        input_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )

    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 16)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test save/load
    model.save('test_model')
    loaded_model = TabularClassifier.load('test_model')
    print("\n✅ Save/load test passed")
```

---

## Step 5: Create Dataset Class

**Create**: `ml/data/datasets.py`

```python
"""PyTorch dataset classes for tabular data."""
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional


class TabularDataset(Dataset):
    """Dataset for tabular data with features and labels.

    Expects data in parquet format with:
    - All columns except 'label' are features
    - 'label' column contains targets

    Args:
        data_path: Path to parquet file or directory containing data.parquet
        label_col: Name of label column (default: 'label')
        feature_cols: List of feature columns (if None, uses all except label_col)
    """

    def __init__(
        self,
        data_path: str,
        label_col: str = 'label',
        feature_cols: Optional[list] = None
    ):
        self.data_path = Path(data_path)
        self.label_col = label_col

        # Load data
        if self.data_path.is_dir():
            data_file = self.data_path / 'data.parquet'
        else:
            data_file = self.data_path

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = pd.read_parquet(data_file)

        # Determine feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in self.data.columns if col != label_col]
        else:
            self.feature_cols = feature_cols

        # Extract features and labels
        self.features = self.data[self.feature_cols].values.astype('float32')
        self.labels = self.data[label_col].values.astype('float32')

        print(f"Loaded {len(self.data)} samples from {data_file}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Positive rate: {self.labels.mean():.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample.

        Returns:
            Dictionary with 'features' and 'label' tensors
        """
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }

    def get_feature_dim(self):
        """Return number of features."""
        return len(self.feature_cols)

    def get_statistics(self):
        """Return dataset statistics."""
        return {
            'n_samples': len(self.data),
            'n_features': len(self.feature_cols),
            'positive_rate': float(self.labels.mean()),
            'feature_cols': self.feature_cols
        }


# Test
if __name__ == '__main__':
    # Load dataset
    dataset = TabularDataset('data/features/v1/train')

    print(f"\nDataset info:")
    print(dataset.get_statistics())

    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Label: {sample['label'].item()}")

    print("\n✅ Dataset loaded successfully")
```

---

## Step 6: Create DataLoader Utilities

**Create**: `ml/data/loaders.py`

```python
"""DataLoader utilities for creating train/val/test loaders."""
import torch
from torch.utils.data import DataLoader
from typing import Tuple
from ml.data.datasets import TabularDataset


def create_dataloaders(
    features_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        features_dir: Base directory containing train/, val/, test/ subdirectories
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from pathlib import Path

    base_dir = Path(features_dir)

    # Create datasets
    train_dataset = TabularDataset(base_dir / 'train')
    val_dataset = TabularDataset(base_dir / 'val')
    test_dataset = TabularDataset(base_dir / 'test')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


# Test
if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders(
        features_dir='data/features/v1',
        batch_size=128
    )

    # Test iteration
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Features: {batch['features'].shape}")
    print(f"  Labels: {batch['label'].shape}")

    print("\n✅ DataLoaders working correctly")
```

---

## Step 7: Create Training Script

**Create**: `ml/training/trainer.py`

```python
"""Training loop logic."""
import torch
import torch.nn as nn
from typing import Dict, Callable
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary of metrics (loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * features.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Dictionary of metrics (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

    # Calculate AUC
    all_outputs = torch.cat(all_outputs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_outputs)

    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'auc': auc
    }

    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    save_path: str,
    early_stopping_patience: int = 5
) -> Dict[str, list]:
    """Complete training loop with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        save_path: Path to save best model
        early_stopping_patience: Patience for early stopping

    Returns:
        Dictionary of training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹ Early stopping triggered after {epoch} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))
    print(f"\n✅ Training complete. Best model loaded.")

    return history
```

**Continue in next file...**

---

## Step 8: Create Training CLI Script

**Create**: `ml/training/train.py`

```python
"""Command-line training script."""
import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.tabular import TabularClassifier
from ml.data.loaders import create_dataloaders
from ml.training.trainer import train_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_training_artifacts(output_dir: Path, model, config, history, metrics):
    """Save model, config, and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'model_best.pt'
    config_path = output_dir / 'config.yaml'
    metrics_path = output_dir / 'metrics.json'
    history_path = output_dir / 'history.json'

    # Model weights
    torch.save(model.state_dict(), model_path)

    # Config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # History
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n📁 Artifacts saved to {output_dir}/")
    print(f"   - model_best.pt")
    print(f"   - config.yaml")
    print(f"   - metrics.json")
    print(f"   - history.json")


def main(config_path: str, output_dir: str = None):
    """Main training function.

    Args:
        config_path: Path to config YAML file
        output_dir: Output directory for artifacts (optional)
    """
    # Load config
    config = load_config(config_path)
    print("📋 Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")

    # Create dataloaders
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        features_dir=config['data']['features_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0)
    )

    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[1]
    print(f"   Input dimension: {input_dim}")

    # Create model
    print("\n🏗 Creating model...")
    model = TabularClassifier(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/runs/ctr_model_{timestamp}"

    output_dir = Path(output_dir)
    model_save_path = output_dir / 'model_best.pt'

    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Early stopping patience: {config['training']['early_stopping_patience']}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config['training']['epochs'],
        save_path=str(model_save_path),
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    from ml.training.trainer import validate_epoch
    test_metrics = validate_epoch(model, test_loader, criterion, device, epoch=0)

    print(f"\n🎯 Test Results:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   AUC: {test_metrics['auc']:.4f}")

    # Save all artifacts
    save_training_artifacts(
        output_dir=output_dir,
        model=model,
        config=config,
        history=history,
        metrics={
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'best_val_loss': min(history['val_loss']),
            'best_val_auc': max(history['val_auc'])
        }
    )

    print(f"\n✅ Training complete!")
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tabular classifier')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for artifacts')

    args = parser.parse_args()

    main(config_path=args.config, output_dir=args.output_dir)
```

---

## Step 9: Create Configuration File

**Create**: `config/model_config.yaml`

```yaml
# Model configuration
model:
  type: "tabular_classifier"
  hidden_dims: [256, 128, 64]
  dropout: 0.3

# Training configuration
training:
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5
  num_workers: 0

# Data configuration
data:
  features_path: "data/features/v1"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Output configuration
output:
  model_dir: "models/staging/ctr_model"
  experiment_dir: "experiments/runs"
```

---

## Step 10: Run Training

Now run the complete training pipeline:

```bash
cd /home/user/mlops-learning-plan

# 1. Generate sample data (if not already done)
python scripts/generate_sample_features.py

# 2. Run training
python ml/training/train.py --config config/model_config.yaml

# You should see:
# - Training progress with loss/accuracy
# - Validation metrics each epoch
# - Early stopping when no improvement
# - Test set evaluation
# - Saved artifacts
```

Expected output:
```
📋 Configuration loaded:
...

🔧 Using device: cpu

📊 Loading data...
Loaded 7000 samples from data/features/v1/train/data.parquet
Loaded 1500 samples from data/features/v1/val/data.parquet
Loaded 1500 samples from data/features/v1/test/data.parquet

🏗 Creating model...
TabularClassifier(
  input_dim=16,
  hidden_dims=[256, 128, 64],
  dropout=0.3,
  output_dim=1,
  params=54,785
)

🚀 Starting training...

Epoch 1/30 [Train]: 100%|███| 55/55 [00:02<00:00]
Epoch 1/30 [Val]: 100%|███| 12/12 [00:00<00:00]

Epoch 1/30:
  Train Loss: 0.5234, Acc: 0.7123
  Val Loss: 0.4987, Acc: 0.7234, AUC: 0.7654
  ✅ Best model saved (val_loss: 0.4987)

...

✅ Training complete!
📁 Artifacts saved to experiments/runs/ctr_model_20241114_120530/
```

---

## Step 11: Verify and Test Model

**Create**: `scripts/test_model.py`

```python
"""Test trained model."""
import torch
import yaml
import json
from pathlib import Path
from ml.models.tabular import TabularClassifier
from ml.data.loaders import create_dataloaders


def load_trained_model(run_dir: str):
    """Load model from experiment run directory."""
    run_dir = Path(run_dir)

    # Load config
    with open(run_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load metrics
    with open(run_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    # Get input dim from data
    _, val_loader, _ = create_dataloaders(
        features_dir=config['data']['features_path'],
        batch_size=128
    )
    sample_batch = next(iter(val_loader))
    input_dim = sample_batch['features'].shape[1]

    # Create model
    model = TabularClassifier(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )

    # Load weights
    model.load_state_dict(torch.load(run_dir / 'model_best.pt'))
    model.eval()

    print(f"✅ Model loaded from {run_dir}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   AUC: {metrics['test_auc']:.4f}")

    return model, config


def predict_sample(model, features):
    """Make prediction on a single sample."""
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        output = model(features_tensor)
        prob = output.item()
        prediction = 1 if prob > 0.5 else 0

    return prediction, prob


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/test_model.py <run_directory>")
        print("Example: python scripts/test_model.py experiments/runs/ctr_model_20241114_120530")
        sys.exit(1)

    run_dir = sys.argv[1]

    # Load model
    model, config = load_trained_model(run_dir)

    # Test on some examples
    print("\n🧪 Testing predictions...")
    _, val_loader, _ = create_dataloaders(
        features_dir=config['data']['features_path'],
        batch_size=1
    )

    for i, batch in enumerate(val_loader):
        if i >= 5:  # Test 5 samples
            break

        features = batch['features'][0]
        true_label = batch['label'][0].item()

        pred, prob = predict_sample(model, features)

        print(f"\nSample {i+1}:")
        print(f"  True label: {int(true_label)}")
        print(f"  Predicted: {pred} (prob: {prob:.3f})")
        print(f"  {'✅' if pred == int(true_label) else '❌'}")
```

**Run it**:
```bash
# Replace with your actual run directory
python scripts/test_model.py experiments/runs/ctr_model_20241114_120530
```

---

## Step 12: Deliverables Checklist

Verify you have:

- [ ] **Code structure**:
  - `ml/models/base.py` - Base model class
  - `ml/models/tabular.py` - Tabular classifier
  - `ml/data/datasets.py` - Dataset class
  - `ml/data/loaders.py` - DataLoader utilities
  - `ml/training/trainer.py` - Training loops
  - `ml/training/train.py` - CLI script

- [ ] **Configuration**:
  - `config/model_config.yaml` - All hyperparameters

- [ ] **Data**:
  - `data/features/v1/train/data.parquet`
  - `data/features/v1/val/data.parquet`
  - `data/features/v1/test/data.parquet`

- [ ] **Trained model**:
  - `experiments/runs/{timestamp}/model_best.pt`
  - `experiments/runs/{timestamp}/config.yaml`
  - `experiments/runs/{timestamp}/metrics.json`
  - `experiments/runs/{timestamp}/history.json`

- [ ] **Test results**:
  - Test AUC > 0.70 (should be ~0.75-0.80 with sample data)

---

## Key Takeaways

1. **Separation of Concerns**: Data loading, model definition, and training are separate modules
2. **Config-Driven**: All hyperparameters come from config file
3. **Reproducibility**: Save config + weights + metrics together
4. **Early Stopping**: Prevent overfitting, save best model
5. **Proper Evaluation**: Use separate val/test sets, track AUC for imbalanced data

---

## Next Steps

You've built a complete tabular classifier! Next:

1. **Lab 3.2**: Integrate this training into Airflow
2. **Lab 3.3**: Add experiment tracking to compare runs
3. **Lab 3.4**: Build a two-tower ranking model

---

## Troubleshooting

**Issue**: Import errors
- Solution: Make sure you're running from project root and `ml/` has `__init__.py` files

**Issue**: Data not found
- Solution: Run `scripts/generate_sample_features.py` first

**Issue**: Poor model performance (AUC < 0.60)
- Check data loading (labels correct?)
- Try different learning rate (0.0001 - 0.01)
- Increase model capacity (more layers/units)

**Issue**: Training too slow
- Reduce batch size
- Use fewer epochs for testing
- Simplify model architecture

---

**Great job! You've built your first production-ready tabular model! 🎉**

Next: **Lab 3.2 - Training DAG** to orchestrate this with Airflow.
