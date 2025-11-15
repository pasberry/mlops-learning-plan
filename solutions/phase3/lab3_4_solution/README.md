# Lab 3.4 Solution: Two-Tower Ranking Model

Complete solution for building a two-tower (dual encoder) neural network for ranking and recommendation tasks.

## Overview

This solution implements:
- Two-tower architecture with separate user and item encoders
- Embedding-based similarity matching
- Movie recommendation system (user-movie affinity prediction)
- Scalable architecture for production recommendations

## Two-Tower Architecture

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

## Why Two-Tower?

### Advantages
- **Scalable**: Pre-compute item embeddings, fast lookup at inference
- **Flexible**: Easy to add new items without retraining user tower
- **Interpretable**: Embeddings can be visualized and understood
- **General**: Works for recommendations, search, matching

### Use Cases
- Netflix: User → Movie recommendations
- Spotify: User → Song recommendations
- LinkedIn: User → Job matching
- E-commerce: User → Product ranking

## Project Structure

```
lab3_4_solution/
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Base model interface
│   │   └── two_tower.py         # TwoTowerModel implementation
│   ├── data/
│   │   ├── __init__.py
│   │   └── ranking_dataset.py   # TwoTowerDataset class
│   └── training/
│       ├── __init__.py
│       └── train_ranking.py     # Training script
├── config/
│   └── ranking_config.yaml      # Model configuration
├── scripts/
│   └── generate_recommendation_data.py  # Data generation
└── README.md
```

## Quick Start

### Step 1: Install Dependencies

```bash
pip install torch pandas pyarrow pyyaml scikit-learn tqdm
```

### Step 2: Generate Movie Recommendation Data

```bash
cd /home/user/mlops-learning-plan/solutions/phase3/lab3_4_solution
python scripts/generate_recommendation_data.py
```

This creates:
- `data/recommendations/v1/users.parquet` (1,000 users)
- `data/recommendations/v1/movies.parquet` (500 movies)
- `data/recommendations/v1/train/interactions.parquet` (7,000 ratings)
- `data/recommendations/v1/val/interactions.parquet` (1,500 ratings)
- `data/recommendations/v1/test/interactions.parquet` (1,500 ratings)

### Step 3: Train Two-Tower Model

```bash
python ml/training/train_ranking.py --config config/ranking_config.yaml
```

Expected output:
```
📋 Configuration:
model:
  embedding_dim: 128
  user_hidden_dims: [256, 128]
  item_hidden_dims: [256, 128]
  dropout: 0.3
...

🔧 Device: cpu

📊 Loading data...
Loaded train dataset:
  Interactions: 7000
  User features: 13
  Movie features: 12
  Positive rate: 0.543

🏗 Creating two-tower model...
TwoTowerModel(
  user_input_dim=13,
  item_input_dim=12,
  embedding_dim=128,
  user_hidden_dims=[256, 128],
  item_hidden_dims=[256, 128],
  dropout=0.3,
  params=89,985
)

🚀 Training...

Epoch 1/30 [Train]: 100%|████████| 28/28 [00:03<00:00]
Epoch 1/30 [Val]: 100%|██████████| 6/6 [00:00<00:00]

Epoch 1:
  Train Loss: 0.5123, Acc: 0.7234
  Val Loss: 0.4987, Acc: 0.7456, AUC: 0.7845
  ✅ Best model saved

...

🎯 Test Results:
  Loss: 0.4856
  Accuracy: 0.7589
  AUC: 0.8023

✅ Training complete! Artifacts in experiments/runs/ranking_model_20241115_120530/
```

### Step 4: Use Model for Recommendations

```python
import torch
import pandas as pd
from ml.models.two_tower import TwoTowerModel

# Load trained model
model = TwoTowerModel(
    user_input_dim=13,
    item_input_dim=12,
    embedding_dim=128,
    user_hidden_dims=[256, 128],
    item_hidden_dims=[256, 128],
    dropout=0.3
)
model.load_state_dict(torch.load('experiments/runs/ranking_model_.../model_best.pt'))
model.eval()

# Load user and movie data
users_df = pd.read_parquet('data/recommendations/v1/users.parquet')
movies_df = pd.read_parquet('data/recommendations/v1/movies.parquet')

# Get recommendations for user 42
user_id = 42
user_row = users_df[users_df['user_id'] == user_id]
user_feature_cols = [c for c in users_df.columns if c != 'user_id']
movie_feature_cols = [c for c in movies_df.columns if c != 'movie_id']

user_features = torch.FloatTensor(user_row[user_feature_cols].values)
movie_features = torch.FloatTensor(movies_df[movie_feature_cols].values)

# Compute scores for all movies
with torch.no_grad():
    # Repeat user features for all movies
    user_features_repeated = user_features.repeat(len(movies_df), 1)

    # Get scores
    scores = model(user_features_repeated, movie_features).squeeze().numpy()

# Get top-10 recommendations
import numpy as np
top_10_indices = np.argsort(scores)[-10:][::-1]
recommended_movies = movies_df.iloc[top_10_indices]

print(f"Top 10 recommendations for User {user_id}:")
for idx, (movie_idx, score) in enumerate(zip(top_10_indices, scores[top_10_indices]), 1):
    movie_id = movies_df.iloc[movie_idx]['movie_id']
    print(f"{idx}. Movie ID {movie_id}: Score {score:.4f}")
```

## Model Architecture Details

### User Tower
```
User Features (13 dims)
       ↓
   Linear(13 → 256)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(256 → 128)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(128 → 128)  # Embedding layer
       ↓
   L2 Normalize
```

### Item Tower
```
Item Features (12 dims)
       ↓
   Linear(12 → 256)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(256 → 128)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(128 → 128)  # Embedding layer
       ↓
   L2 Normalize
```

### Similarity Computation
```python
# Normalized embeddings
user_emb = L2_normalize(user_tower(user_features))  # (batch, 128)
item_emb = L2_normalize(item_tower(item_features))  # (batch, 128)

# Dot product similarity
similarity = (user_emb * item_emb).sum(dim=1)  # (batch,)

# Probability
score = sigmoid(similarity)  # (batch,)
```

Total parameters: ~90,000

## Data Features

### User Features (13 total)
- age (normalized)
- gender (M/F → one-hot)
- favorite_genre (Action/Comedy/Drama/Horror/Sci-Fi/Romance → one-hot, 6 dims)
- activity_level (low/medium/high → one-hot, 3 dims)

### Movie Features (12 total)
- genre (Action/Comedy/Drama/Horror/Sci-Fi/Romance → one-hot, 6 dims)
- year (normalized)
- duration (normalized)
- avg_rating (normalized)

### Interaction Label
- Binary: 1 (liked, rating ≥ 4), 0 (not liked, rating < 4)
- Positive rate: ~54%

## Configuration

Edit `config/ranking_config.yaml`:

```yaml
model:
  embedding_dim: 128         # Size of user/item embeddings
  user_hidden_dims: [256, 128]  # User tower architecture
  item_hidden_dims: [256, 128]  # Item tower architecture
  dropout: 0.3

training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 30
  early_stopping_patience: 5
```

## Production Deployment

### Offline: Pre-compute Item Embeddings

```python
# One-time: Encode all items
model.eval()
all_movie_features = torch.FloatTensor(movies_df[movie_feature_cols].values)

with torch.no_grad():
    item_embeddings = model.encode_item(all_movie_features)  # (N_items, 128)

# Save embeddings
torch.save(item_embeddings, 'item_embeddings.pt')
```

### Online: Fast User-to-Item Retrieval

```python
# At inference time
def get_recommendations_fast(user_id, k=10):
    # Encode user
    user_features = get_user_features(user_id)  # (1, 13)
    with torch.no_grad():
        user_emb = model.encode_user(user_features)  # (1, 128)

    # Load pre-computed item embeddings
    item_embeddings = torch.load('item_embeddings.pt')  # (N_items, 128)

    # Compute similarities (fast dot product)
    scores = (user_emb @ item_embeddings.T).squeeze()  # (N_items,)

    # Get top-k
    top_k_indices = torch.topk(scores, k).indices
    return top_k_indices.numpy()

# Very fast! No need to run item tower at inference time
```

## Advanced: Approximate Nearest Neighbor Search

For millions of items, use ANN libraries:

```python
import faiss

# Build FAISS index
item_embeddings_np = item_embeddings.numpy()
index = faiss.IndexFlatIP(128)  # Inner product (dot product)
index.add(item_embeddings_np)

# Search
user_emb_np = user_emb.numpy()
scores, indices = index.search(user_emb_np, k=10)
```

## Testing

```bash
# Test model
python ml/models/two_tower.py

# Test dataset
python ml/data/ranking_dataset.py
```

## Expected Performance

With default configuration:
- **Training Accuracy**: ~75-77%
- **Validation AUC**: ~0.78-0.80
- **Test AUC**: ~0.77-0.82
- **Training Time**: ~3-5 minutes on CPU (10,000 interactions)

## Comparison: Two-Tower vs Single Tower

| Aspect | Two-Tower | Single Tower |
|--------|-----------|--------------|
| Inference Speed | Fast (pre-compute items) | Slow (full forward pass) |
| Scalability | Excellent | Poor |
| New Items | Easy (only encode new item) | Retrain needed |
| Accuracy | Good | Slightly better |
| Use Case | Production systems | Offline ranking |

## Key Takeaways

1. **Separate Encoders**: User and item towers independent
2. **Embedding Space**: Both produce vectors in same space
3. **Scalability**: Pre-compute item embeddings once
4. **Production-Ready**: This architecture powers real systems
5. **Flexibility**: Easy to add new users/items

## Extensions

1. **Triplet Loss**: Better embeddings via contrastive learning
2. **Hard Negative Mining**: Sample difficult negatives
3. **Multi-Task Learning**: Predict rating + genre simultaneously
4. **Attention**: Add attention layers in towers
5. **Cross Features**: Combine user-item interactions

## Next Steps

1. **Phase 4**: Deploy model with FastAPI for serving
2. **Batch Inference**: Generate recommendations for all users
3. **A/B Testing**: Compare against baseline model

## Troubleshooting

**Issue**: Poor model performance (AUC < 0.70)
- Increase embedding dimension: 256 or 512
- Deeper towers: [512, 256, 128, 64]
- More training data: increase n_interactions

**Issue**: Embeddings not normalized
- Check `encode_user` and `encode_item` apply L2 normalization
- Verify with: `torch.norm(embeddings, dim=1)` should be ~1.0

**Issue**: Training too slow
- Increase batch size: 512
- Reduce number of interactions
- Simplify tower architecture

## References

- Two-Tower Neural Networks (YouTube DNN paper)
- Collaborative Filtering for Implicit Feedback
- Large-Scale Item Recommendations (Google)
- Neural Collaborative Filtering
