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
    import os
    if os.path.exists('data/features/v1'):
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
    else:
        print("⚠️  Data not found. Run: python scripts/generate_sample_features.py")
