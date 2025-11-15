"""
MNIST Image Classification with PyTorch - SOLUTION
A complete training script demonstrating best practices for PyTorch model training.
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
    """
    Create train and test dataloaders for MNIST.

    Args:
        batch_size: Batch size for training and testing
        data_dir: Directory to store/load MNIST data

    Returns:
        tuple: (train_loader, test_loader)
    """
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
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda)
        epoch: Current epoch number

    Returns:
        tuple: (average_loss, accuracy)
    """
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
    """
    Validate the model.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to validate on (cpu/cuda)

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

    print(f'\nValidation: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')

    return test_loss, test_acc


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
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

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")

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
