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
    # This test requires data to be generated first
    # Run: python scripts/generate_sample_features.py
    import os
    if os.path.exists('data/features/v1/train'):
        dataset = TabularDataset('data/features/v1/train')

        print(f"\nDataset info:")
        print(dataset.get_statistics())

        # Get a sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Label: {sample['label'].item()}")

        print("\n✅ Dataset loaded successfully")
    else:
        print("⚠️  Data not found. Run: python scripts/generate_sample_features.py")
