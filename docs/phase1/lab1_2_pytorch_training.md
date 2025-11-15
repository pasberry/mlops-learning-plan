# Lab 1.2: PyTorch Training Script

**Goal**: Build a complete training script from scratch for image classification

**Estimated Time**: 60-90 minutes

**Prerequisites**:
- Lab 1.1 completed
- PyTorch installed

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Understand PyTorch Dataset and DataLoader
- ✅ Build a neural network with `nn.Module`
- ✅ Implement a complete training loop
- ✅ Add validation and model checkpointing
- ✅ Structure ML code professionally

---

## Background: PyTorch Training Components

### 1. Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create DataLoader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2. Model Definition

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Lab Instructions

### Part 1: Create the Training Script

Create `ml/train_mnist.py`:

```python
"""
MNIST Image Classification with PyTorch
A complete training script demonstrating best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def get_dataloaders(batch_size=64, data_dir='./data'):
    """Create train and test dataloaders for MNIST."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
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

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]  '
                  f'Loss: {loss.item():.6f}')

    # Calculate epoch metrics
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

    print(f'\nValidation: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')

    return test_loss, test_acc


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    """Main training function."""
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    data_dir = './data'
    checkpoint_dir = './models/staging/mnist'

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders(batch_size, data_dir)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = SimpleCNN().to(device)
    print(f"\nModel architecture:\n{model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Training: Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)
            print(f"New best accuracy: {best_acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

### Part 2: Run the Training Script

```bash
# Activate virtual environment
source venv/bin/activate

# Create ml directory if it doesn't exist
mkdir -p ml

# Run training
python ml/train_mnist.py
```

Expected output:
```
Using device: cpu
Loading MNIST dataset...
Downloading MNIST...
Train samples: 60000
Test samples: 10000

Model architecture:
SimpleCNN(...)

Starting training for 5 epochs...
Epoch 1/5
Loss: 0.234...
...
Training: Loss: 0.1234, Accuracy: 96.12%
Validation: Average loss: 0.0987, Accuracy: 97.34%
Checkpoint saved: ./models/staging/mnist/checkpoint_epoch_1.pt
New best accuracy: 97.34%
...
Training completed!
Best validation accuracy: 98.45%
```

---

## Exercise 1: Add Learning Rate Scheduling

Add a learning rate scheduler that reduces LR when validation loss plateaus:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# After creating optimizer
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# After each epoch validation
scheduler.step(val_loss)
```

---

## Exercise 2: Implement Early Stopping

Add logic to stop training if validation loss doesn't improve:

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=3)
for epoch in range(num_epochs):
    # ... training code ...
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

---

## Exercise 3: Add TensorBoard Logging

Track metrics with TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/mnist_experiment')

# Log during training
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# Close writer
writer.close()

# View with: tensorboard --logdir=runs
```

---

## Exercise 4: Load and Test Saved Model

Create `ml/test_mnist.py`:

```python
"""Test a saved MNIST model."""
import torch
from train_mnist import SimpleCNN, get_dataloaders

def test_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SimpleCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    _, test_loader = get_dataloaders(batch_size=64)

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    test_model('./models/staging/mnist/best_model.pt')
```

---

## Challenge: Build a Custom Dataset

Create a custom dataset for tabular data:

```python
import pandas as pd
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """Custom dataset for tabular data."""

    def __init__(self, csv_file, features, target):
        """
        Args:
            csv_file: Path to CSV file
            features: List of feature column names
            target: Target column name
        """
        self.data = pd.read_csv(csv_file)
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get features and target
        x = self.data.iloc[idx][self.features].values
        y = self.data.iloc[idx][self.target]

        # Convert to tensors
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y])

        return x, y
```

---

## Key Takeaways

### PyTorch Best Practices

✅ **Separate model definition from training logic**
✅ **Use DataLoader for efficient batching**
✅ **Always set model.train() and model.eval() modes**
✅ **Zero gradients before backward pass**
✅ **Save checkpoints regularly**
✅ **Track both training and validation metrics**
✅ **Use device-agnostic code (CPU/GPU)**

### Common Pitfalls

❌ **Forgetting optimizer.zero_grad()**: Gradients accumulate!
❌ **Not using model.eval()**: Dropout and BatchNorm behave differently
❌ **Hardcoding device**: Use `device = torch.device(...)`
❌ **Not shuffling training data**: Can hurt convergence
❌ **Overfitting to training data**: Always validate on held-out data

---

## Code Structure Best Practices

```
ml/
├── models/
│   └── cnn.py              # Model definitions
├── data/
│   └── datasets.py         # Custom datasets
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── utils.py                # Helper functions
```

We'll adopt this structure in Phase 3!

---

## Testing Your Code

```bash
# Test model architecture
python -c "from ml.train_mnist import SimpleCNN; print(SimpleCNN())"

# Test data loading
python -c "from ml.train_mnist import get_dataloaders; train, test = get_dataloaders(); print(f'Batches: {len(train)}')"

# Run full training
python ml/train_mnist.py

# Test saved model
python ml/test_mnist.py
```

---

## Submission Checklist

Before moving to the next lab:

- ✅ Training script runs successfully
- ✅ Model achieves >95% test accuracy
- ✅ Checkpoints are saved properly
- ✅ You can load and test a saved model
- ✅ At least one exercise completed
- ✅ You understand the training loop

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)

---

**Congratulations! You've built a complete PyTorch training pipeline!** 🎉

**Next**: [Lab 1.3 - Config-Driven Training →](./lab1_3_config_driven_training.md)
