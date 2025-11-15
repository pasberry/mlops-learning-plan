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
from ml.training.trainer import train_model, validate_epoch


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
    output_dir.mkdir(parents=True, exist_ok=True)

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
