"""Training script for two-tower ranking model."""
import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.two_tower import TwoTowerModel
from ml.data.ranking_dataset import TwoTowerDataset


def create_dataloaders(data_dir: str, batch_size: int):
    """Create train/val/test dataloaders."""
    train_dataset = TwoTowerDataset(data_dir, split='train')
    val_dataset = TwoTowerDataset(data_dir, split='val')
    test_dataset = TwoTowerDataset(data_dir, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader, train_dataset


def train_epoch_two_tower(model, dataloader, criterion, optimizer, device, epoch):
    """Training epoch for two-tower model."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        user_feats, item_feats, labels = batch
        user_feats = user_feats.to(device)
        item_feats = item_feats.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        scores = model(user_feats, item_feats)
        loss = criterion(scores, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * user_feats.size(0)
        predicted = (scores > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def validate_epoch_two_tower(model, dataloader, criterion, device, epoch):
    """Validation epoch for two-tower model."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            user_feats, item_feats, labels = batch
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            labels = labels.to(device)

            scores = model(user_feats, item_feats)
            loss = criterion(scores, labels)

            total_loss += loss.item() * user_feats.size(0)
            predicted = (scores > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

    # Calculate AUC
    all_scores = torch.cat(all_scores).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    auc = roc_auc_score(all_labels, all_scores)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'auc': auc
    }


def main(config_path: str):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("📋 Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")

    # Load data
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        data_dir=config['data']['data_path'],
        batch_size=config['training']['batch_size']
    )

    # Create model
    print("\n🏗 Creating two-tower model...")
    model = TwoTowerModel(
        user_input_dim=train_dataset.get_user_dim(),
        item_input_dim=train_dataset.get_movie_dim(),
        embedding_dim=config['model']['embedding_dim'],
        user_hidden_dims=config['model']['user_hidden_dims'],
        item_hidden_dims=config['model']['item_hidden_dims'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    print(model)

    # Optimizer and loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"experiments/runs/ranking_model_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'model_best.pt'

    # Training loop
    print(f"\n🚀 Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping_patience']

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch_two_tower(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate_epoch_two_tower(
            model, val_loader, criterion, device, epoch
        )

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Best model saved")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping")
                break

    # Load best model
    model.load_state_dict(torch.load(model_path))

    # Test
    print("\n📊 Testing...")
    test_metrics = validate_epoch_two_tower(model, test_loader, criterion, device, 0)

    print(f"\n🎯 Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    # Save artifacts
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'best_val_loss': best_val_loss,
            'best_val_auc': max(history['val_auc'])
        }, f, indent=2)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Artifacts in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ranking_config.yaml')
    args = parser.parse_args()

    main(args.config)
