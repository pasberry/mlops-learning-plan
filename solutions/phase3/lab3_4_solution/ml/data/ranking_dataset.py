"""Dataset for two-tower ranking model."""
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple


class TwoTowerDataset(Dataset):
    """Dataset for two-tower recommendation model.

    Returns user features, item features, and labels.
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: Base directory containing users.parquet, movies.parquet
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load user and movie features
        self.users_df = pd.read_parquet(self.data_dir / 'users.parquet')
        self.movies_df = pd.read_parquet(self.data_dir / 'movies.parquet')

        # Load interactions
        interactions_path = self.data_dir / split / 'interactions.parquet'
        self.interactions_df = pd.read_parquet(interactions_path)

        # Extract feature columns
        self.user_feature_cols = [c for c in self.users_df.columns if c != 'user_id']
        self.movie_feature_cols = [c for c in self.movies_df.columns if c != 'movie_id']

        # Convert to numpy for faster indexing
        self.user_features = self.users_df[self.user_feature_cols].values.astype('float32')
        self.movie_features = self.movies_df[self.movie_feature_cols].values.astype('float32')

        # Create lookup dictionaries
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df['user_id'])}
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movies_df['movie_id'])}

        print(f"Loaded {split} dataset:")
        print(f"  Interactions: {len(self.interactions_df)}")
        print(f"  User features: {len(self.user_feature_cols)}")
        print(f"  Movie features: {len(self.movie_feature_cols)}")
        print(f"  Positive rate: {self.interactions_df['label'].mean():.3f}")

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return user features, movie features, and label."""
        interaction = self.interactions_df.iloc[idx]

        user_id = interaction['user_id']
        movie_id = interaction['movie_id']
        label = interaction['label']

        # Get feature indices
        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        # Get features
        user_feats = self.user_features[user_idx]
        movie_feats = self.movie_features[movie_idx]

        return (
            torch.FloatTensor(user_feats),
            torch.FloatTensor(movie_feats),
            torch.FloatTensor([label])
        )

    def get_user_dim(self):
        """Return user feature dimension."""
        return len(self.user_feature_cols)

    def get_movie_dim(self):
        """Return movie feature dimension."""
        return len(self.movie_feature_cols)


# Test
if __name__ == '__main__':
    import os
    if os.path.exists('data/recommendations/v1'):
        dataset = TwoTowerDataset('data/recommendations/v1', split='train')

        # Get a sample
        user_feats, movie_feats, label = dataset[0]

        print(f"\nSample:")
        print(f"  User features: {user_feats.shape}")
        print(f"  Movie features: {movie_feats.shape}")
        print(f"  Label: {label.item()}")

        print(f"\n✅ Dataset working correctly")
    else:
        print("⚠️  Data not found. Run: python scripts/generate_recommendation_data.py")
