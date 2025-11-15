# Lab 3.4: Two-Tower Ranking Model

**Objective**: Build a two-tower neural network architecture for ranking and recommendation tasks.

**Time**: 3-4 hours

**Prerequisites**:
- Lab 3.1 completed (tabular model basics)
- Understanding of embeddings
- Familiarity with recommendation systems (helpful but not required)

---

## What You'll Build

A two-tower (dual encoder) model that:
- Learns separate embeddings for users and items
- Computes similarity scores for ranking
- Uses contrastive learning (e.g., triplet loss)
- Can be used for recommendations, search, or matching tasks

**Use Case**: Movie recommendation system (predict user-movie affinity)

---

## Two-Tower Architecture Explained

### The Concept

```
User Features          Item Features
     │                      │
     ▼                      ▼
┌─────────┐          ┌─────────┐
│  User   │          │  Item   │
│  Tower  │          │  Tower  │
│         │          │         │
│ [Dense  │          │ [Dense  │
│  Layers]│          │  Layers]│
└─────────┘          └─────────┘
     │                      │
     ▼                      ▼
[User Embedding]    [Item Embedding]
     (128-dim)            (128-dim)
          │                  │
          └─────────┬────────┘
                    ▼
              Dot Product
                    │
                    ▼
             Affinity Score
```

### Key Ideas

1. **Two Separate Networks**: One for users, one for items
2. **Embedding Space**: Both networks produce embeddings in the same space
3. **Similarity Metric**: Dot product (or cosine) measures affinity
4. **Contrastive Learning**: Positive pairs scored higher than negative pairs

### Why Two-Tower?

**Advantages**:
- ✅ Scalable: Pre-compute item embeddings, fast lookup at inference
- ✅ Flexible: Easy to add new items without retraining user tower
- ✅ Interpretable: Embeddings can be visualized and understood
- ✅ General: Works for recommendations, search, matching

**Use Cases**:
- Netflix: User → Movie recommendations
- Spotify: User → Song recommendations
- LinkedIn: User → Job matching
- E-commerce: User → Product ranking

---

## Step 1: Generate Sample Recommendation Data

**Create**: `scripts/generate_recommendation_data.py`

```python
"""Generate sample movie recommendation data."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_recommendation_data(
    n_users=1000,
    n_movies=500,
    n_interactions=10000
):
    """Generate synthetic user-movie interaction data.

    Features:
    - User features: age, gender, favorite_genre, activity_level
    - Movie features: genre, year, duration, avg_rating
    - Interaction: rating (1-5 stars)

    We'll create:
    - users.parquet: User features
    - movies.parquet: Movie features
    - interactions.parquet: User-movie ratings
    """
    np.random.seed(42)

    # === User Features ===
    user_ids = np.arange(n_users)
    user_ages = np.random.randint(18, 70, n_users)
    user_genders = np.random.choice(['M', 'F'], n_users)

    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
    user_fav_genres = np.random.choice(genres, n_users)

    # Activity level (how often they watch)
    user_activity = np.random.choice(['low', 'medium', 'high'], n_users, p=[0.3, 0.5, 0.2])

    users_df = pd.DataFrame({
        'user_id': user_ids,
        'age': user_ages,
        'gender': user_genders,
        'favorite_genre': user_fav_genres,
        'activity_level': user_activity
    })

    # One-hot encode categorical features
    users_df = pd.get_dummies(users_df, columns=['gender', 'favorite_genre', 'activity_level'])

    # Normalize age
    users_df['age'] = (users_df['age'] - users_df['age'].mean()) / users_df['age'].std()

    # === Movie Features ===
    movie_ids = np.arange(n_movies)
    movie_genres = np.random.choice(genres, n_movies)
    movie_years = np.random.randint(1990, 2024, n_movies)
    movie_durations = np.random.randint(80, 180, n_movies)  # minutes
    movie_avg_ratings = np.random.uniform(2.0, 5.0, n_movies)

    movies_df = pd.DataFrame({
        'movie_id': movie_ids,
        'genre': movie_genres,
        'year': movie_years,
        'duration': movie_durations,
        'avg_rating': movie_avg_ratings
    })

    # One-hot encode genre
    movies_df = pd.get_dummies(movies_df, columns=['genre'])

    # Normalize numerical features
    for col in ['year', 'duration', 'avg_rating']:
        movies_df[col] = (movies_df[col] - movies_df[col].mean()) / movies_df[col].std()

    # === Interactions (User-Movie Ratings) ===
    # Generate interactions with some logic:
    # - Users tend to like movies in their favorite genre
    # - Higher activity users have more interactions
    # - Ratings influenced by movie avg_rating and genre match

    interaction_user_ids = np.random.choice(user_ids, n_interactions)
    interaction_movie_ids = np.random.choice(movie_ids, n_interactions)

    # Create ratings with some logic
    ratings = []
    for user_id, movie_id in zip(interaction_user_ids, interaction_movie_ids):
        user = users_df[users_df['user_id'] == user_id].iloc[0]
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]

        # Base rating from movie avg_rating (denormalize)
        base_rating = 3.5

        # Boost if genre matches user preference
        # (Check if user has favorite_genre_X and movie has genre_X both = 1)
        genre_match = 0
        for genre in genres:
            user_col = f'favorite_genre_{genre}'
            movie_col = f'genre_{genre}'
            if user_col in user and movie_col in movie:
                if user[user_col] == 1 and movie[movie_col] == 1:
                    genre_match = 1
                    break

        if genre_match:
            base_rating += np.random.uniform(0.5, 1.5)
        else:
            base_rating += np.random.uniform(-1.0, 0.5)

        # Add noise
        rating = base_rating + np.random.normal(0, 0.5)

        # Clip to 1-5
        rating = np.clip(rating, 1, 5)

        ratings.append(rating)

    interactions_df = pd.DataFrame({
        'user_id': interaction_user_ids,
        'movie_id': interaction_movie_ids,
        'rating': ratings
    })

    # Create binary label: 1 if rating >= 4 (liked), 0 otherwise
    interactions_df['label'] = (interactions_df['rating'] >= 4.0).astype(int)

    # Remove duplicates (keep last interaction)
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

    print(f"Generated data:")
    print(f"  Users: {len(users_df)}")
    print(f"  Movies: {len(movies_df)}")
    print(f"  Interactions: {len(interactions_df)}")
    print(f"  Positive rate: {interactions_df['label'].mean():.3f}")

    return users_df, movies_df, interactions_df


def create_train_val_test_splits(users_df, movies_df, interactions_df, output_dir):
    """Create train/val/test splits for ranking.

    Strategy:
    - Split interactions by user (chronological would be better in production)
    - Keep all users and movies in features
    """
    n = len(interactions_df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    # Shuffle
    interactions_df = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_interactions = interactions_df[:n_train]
    val_interactions = interactions_df[n_train:n_train + n_val]
    test_interactions = interactions_df[n_train + n_val:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save user and movie features (same for all splits)
    users_df.to_parquet(output_dir / 'users.parquet', index=False)
    movies_df.to_parquet(output_dir / 'movies.parquet', index=False)

    # Save interaction splits
    for split, split_df in [('train', train_interactions),
                            ('val', val_interactions),
                            ('test', test_interactions)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        split_df.to_parquet(split_dir / 'interactions.parquet', index=False)
        print(f"Saved {len(split_df)} interactions to {split_dir}/")

    print(f"\n✅ Data saved to {output_dir}/")
    print(f"   - users.parquet ({len(users_df)} users)")
    print(f"   - movies.parquet ({len(movies_df)} movies)")
    print(f"   - train/interactions.parquet ({len(train_interactions)} interactions)")
    print(f"   - val/interactions.parquet ({len(val_interactions)} interactions)")
    print(f"   - test/interactions.parquet ({len(test_interactions)} interactions)")


if __name__ == '__main__':
    # Generate data
    users_df, movies_df, interactions_df = generate_recommendation_data(
        n_users=1000,
        n_movies=500,
        n_interactions=10000
    )

    # Save splits
    create_train_val_test_splits(
        users_df,
        movies_df,
        interactions_df,
        output_dir='data/recommendations/v1'
    )
```

**Run it**:
```bash
python scripts/generate_recommendation_data.py
```

---

## Step 2: Create Two-Tower Dataset

**Create**: `ml/data/ranking_dataset.py`

```python
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
    dataset = TwoTowerDataset('data/recommendations/v1', split='train')

    # Get a sample
    user_feats, movie_feats, label = dataset[0]

    print(f"\nSample:")
    print(f"  User features: {user_feats.shape}")
    print(f"  Movie features: {movie_feats.shape}")
    print(f"  Label: {label.item()}")

    print(f"\n✅ Dataset working correctly")
```

---

## Step 3: Create Two-Tower Model

**Create**: `ml/models/ranking.py`

```python
"""Two-tower models for ranking and recommendations."""
import torch
import torch.nn as nn
from typing import List
from ml.models.base import BaseModel


class TwoTowerModel(BaseModel):
    """Two-tower architecture for ranking.

    Separate encoders for user and item, with dot product similarity.

    Architecture:
        User Features → User Tower → User Embedding (embedding_dim)
        Item Features → Item Tower → Item Embedding (embedding_dim)

        Similarity = dot(User Embedding, Item Embedding)
        Score = sigmoid(Similarity)

    Args:
        user_input_dim: Dimension of user features
        item_input_dim: Dimension of item features
        embedding_dim: Dimension of final embeddings (default: 128)
        user_hidden_dims: Hidden layer sizes for user tower (default: [256, 128])
        item_hidden_dims: Hidden layer sizes for item tower (default: [256, 128])
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int = 128,
        user_hidden_dims: List[int] = None,
        item_hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()

        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim
        self.user_hidden_dims = user_hidden_dims or [256, 128]
        self.item_hidden_dims = item_hidden_dims or [256, 128]
        self.dropout = dropout

        # Build user tower
        self.user_tower = self._build_tower(
            user_input_dim,
            self.user_hidden_dims,
            embedding_dim,
            dropout
        )

        # Build item tower
        self.item_tower = self._build_tower(
            item_input_dim,
            self.item_hidden_dims,
            embedding_dim,
            dropout
        )

        # Initialize weights
        self._init_weights()

    def _build_tower(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Module:
        """Build a tower (encoder) network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout probability

        Returns:
            Sequential module
        """
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

        # Output layer (embedding)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # L2 normalize embeddings
        layers.append(nn.functional.normalize)

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features to embedding.

        Args:
            user_features: (batch_size, user_input_dim)

        Returns:
            User embeddings: (batch_size, embedding_dim)
        """
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features to embedding.

        Args:
            item_features: (batch_size, item_input_dim)

        Returns:
            Item embeddings: (batch_size, embedding_dim)
        """
        return self.item_tower(item_features)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute affinity scores.

        Args:
            user_features: (batch_size, user_input_dim)
            item_features: (batch_size, item_input_dim)

        Returns:
            Affinity scores: (batch_size, 1)
        """
        # Get embeddings
        user_embeddings = self.encode_user(user_features)
        item_embeddings = self.encode_item(item_features)

        # Compute dot product (element-wise multiply + sum)
        # (batch_size, embedding_dim) * (batch_size, embedding_dim)
        # -> (batch_size, embedding_dim) -> (batch_size)
        similarity = (user_embeddings * item_embeddings).sum(dim=1, keepdim=True)

        # Apply sigmoid to get probability
        score = torch.sigmoid(similarity)

        return score

    def get_config(self):
        """Return model configuration."""
        return {
            'user_input_dim': self.user_input_dim,
            'item_input_dim': self.item_input_dim,
            'embedding_dim': self.embedding_dim,
            'user_hidden_dims': self.user_hidden_dims,
            'item_hidden_dims': self.item_hidden_dims,
            'dropout': self.dropout
        }

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"TwoTowerModel(\n"
            f"  user_input_dim={self.user_input_dim},\n"
            f"  item_input_dim={self.item_input_dim},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  user_hidden_dims={self.user_hidden_dims},\n"
            f"  item_hidden_dims={self.item_hidden_dims},\n"
            f"  dropout={self.dropout},\n"
            f"  params={n_params:,}\n"
            f")"
        )


# Test
if __name__ == '__main__':
    # Create model
    model = TwoTowerModel(
        user_input_dim=20,
        item_input_dim=15,
        embedding_dim=128,
        user_hidden_dims=[256, 128],
        item_hidden_dims=[256, 128],
        dropout=0.3
    )

    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    user_features = torch.randn(batch_size, 20)
    item_features = torch.randn(batch_size, 15)

    scores = model(user_features, item_features)
    print(f"\nInput shapes:")
    print(f"  User: {user_features.shape}")
    print(f"  Item: {item_features.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test encoding separately
    user_emb = model.encode_user(user_features)
    item_emb = model.encode_item(item_features)
    print(f"\nEmbedding shapes:")
    print(f"  User: {user_emb.shape}")
    print(f"  Item: {item_emb.shape}")

    # Check normalization
    user_norms = torch.norm(user_emb, dim=1)
    print(f"User embedding norms: {user_norms.mean():.3f} (should be ~1.0)")

    print("\n✅ Two-tower model working correctly")
```

---

## Step 4: Create Training Script for Two-Tower

**Create**: `ml/training/train_ranking.py`

```python
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.ranking import TwoTowerModel
from ml.data.ranking_dataset import TwoTowerDataset
from ml.training.trainer import train_epoch, validate_epoch


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

    return train_loader, val_loader, test_loader, train_dataset


def train_epoch_two_tower(model, dataloader, criterion, optimizer, device, epoch):
    """Training epoch for two-tower model."""
    from tqdm import tqdm

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
    from tqdm import tqdm
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

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

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
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")

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
            'test_auc': test_metrics['auc']
        }, f, indent=2)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Artifacts in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ranking_config.yaml')
    args = parser.parse_args()

    main(args.config)
```

---

## Step 5: Create Config

**Create**: `config/ranking_config.yaml`

```yaml
# Two-tower ranking model configuration

model:
  type: "two_tower"
  embedding_dim: 128
  user_hidden_dims: [256, 128]
  item_hidden_dims: [256, 128]
  dropout: 0.3

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5

data:
  data_path: "data/recommendations/v1"
```

---

## Step 6: Run Training

```bash
# Generate data
python scripts/generate_recommendation_data.py

# Train two-tower model
python ml/training/train_ranking.py --config config/ranking_config.yaml
```

---

## Step 7: Inference - Get Recommendations

**Create**: `scripts/get_recommendations.py`

```python
"""Get recommendations using trained two-tower model."""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ml.models.ranking import TwoTowerModel
import yaml


def load_model(run_dir: str):
    """Load trained two-tower model."""
    run_dir = Path(run_dir)

    # Load config
    with open(run_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data to get dimensions
    data_path = Path(config['data']['data_path'])
    users_df = pd.read_parquet(data_path / 'users.parquet')
    movies_df = pd.read_parquet(data_path / 'movies.parquet')

    user_feature_cols = [c for c in users_df.columns if c != 'user_id']
    movie_feature_cols = [c for c in movies_df.columns if c != 'movie_id']

    # Create model
    model = TwoTowerModel(
        user_input_dim=len(user_feature_cols),
        item_input_dim=len(movie_feature_cols),
        embedding_dim=config['model']['embedding_dim'],
        user_hidden_dims=config['model']['user_hidden_dims'],
        item_hidden_dims=config['model']['item_hidden_dims'],
        dropout=config['model']['dropout']
    )

    # Load weights
    model.load_state_dict(torch.load(run_dir / 'model_best.pt'))
    model.eval()

    return model, users_df, movies_df, user_feature_cols, movie_feature_cols


def get_recommendations(user_id: int, model, users_df, movies_df, user_feature_cols, movie_feature_cols, top_k=10):
    """Get top-K movie recommendations for a user."""
    # Get user features
    user_row = users_df[users_df['user_id'] == user_id]
    if user_row.empty:
        print(f"User {user_id} not found")
        return []

    user_features = torch.FloatTensor(user_row[user_feature_cols].values)

    # Get all movie features
    movie_features = torch.FloatTensor(movies_df[movie_feature_cols].values)

    # Compute scores for all movies
    with torch.no_grad():
        # Repeat user features for all movies
        user_features_repeated = user_features.repeat(len(movies_df), 1)

        # Get scores
        scores = model(user_features_repeated, movie_features).squeeze().numpy()

    # Get top-K
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_movies = movies_df.iloc[top_indices].copy()
    top_movies['score'] = scores[top_indices]

    return top_movies[['movie_id', 'score']]


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python scripts/get_recommendations.py <run_dir> <user_id> [top_k]")
        print("Example: python scripts/get_recommendations.py experiments/runs/ranking_model_20241114_120000 42 10")
        sys.exit(1)

    run_dir = sys.argv[1]
    user_id = int(sys.argv[2])
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Load model
    model, users_df, movies_df, user_feature_cols, movie_feature_cols = load_model(run_dir)

    # Get recommendations
    recommendations = get_recommendations(
        user_id, model, users_df, movies_df, user_feature_cols, movie_feature_cols, top_k
    )

    print(f"\nTop {top_k} recommendations for User {user_id}:")
    print("=" * 40)
    for idx, row in recommendations.iterrows():
        print(f"Movie ID: {row['movie_id']}, Score: {row['score']:.4f}")
```

**Usage**:
```bash
python scripts/get_recommendations.py experiments/runs/ranking_model_20241114_120000 42 10
```

---

## Deliverables Checklist

- [ ] **Recommendation data generated**
- [ ] **Two-tower dataset**: `ml/data/ranking_dataset.py`
- [ ] **Two-tower model**: `ml/models/ranking.py`
- [ ] **Training script**: `ml/training/train_ranking.py`
- [ ] **Config file**: `config/ranking_config.yaml`
- [ ] **Model trained** with AUC > 0.70
- [ ] **Inference script** working

---

## Advanced: Triplet Loss (Optional)

For better embedding quality, use triplet loss:

```python
class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
```

Modify dataset to return triplets (user, liked_movie, disliked_movie).

---

## Key Takeaways

1. **Two-tower = Scalable**: Pre-compute item embeddings, fast retrieval
2. **Embeddings matter**: They capture latent features
3. **Separate encoders**: User and item towers can evolve independently
4. **General pattern**: Works for many matching problems
5. **Production-ready**: This architecture powers real recommendation systems

---

## Next Steps

Phase 3 complete! 🎉

**Phase 4 Preview**:
- Deploy models with FastAPI
- Batch inference pipelines
- Model monitoring
- Automated retraining

---

**Congratulations! You've built advanced ML models with PyTorch! 🚀**
