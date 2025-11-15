"""Training loop logic."""
import torch
import torch.nn as nn
from typing import Dict
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
        Dictionary of metrics (loss, accuracy, auc)
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
