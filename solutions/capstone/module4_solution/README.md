# Capstone Module 4: Two-Tower Model Training

## Overview

Implements and trains the two-tower neural network architecture for personalized feed ranking.

## Two-Tower Architecture

### Model Structure

```python
class TwoTowerRankingModel(nn.Module):
    """
    Dual-encoder architecture for user-item matching.
    """

    def __init__(self, user_features_dim, item_features_dim, embedding_dim=64):
        super().__init__()

        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Item Tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, user_features, item_features):
        """
        Compute relevance score.
        """
        # Get embeddings
        user_emb = self.user_tower(user_features)  # [batch, 64]
        item_emb = self.item_tower(item_features)  # [batch, 64]

        # Dot product + sigmoid
        score = (user_emb * item_emb).sum(dim=1, keepdim=True)
        return torch.sigmoid(score)

    def get_user_embedding(self, user_features):
        """Get user embedding for caching."""
        return self.user_tower(user_features)

    def get_item_embedding(self, item_features):
        """Get item embedding for caching."""
        return self.item_tower(item_features)
```

### Why Two-Tower?

**Advantages:**
```
✓ Scalable: Compute embeddings independently
✓ Fast Inference: Pre-compute item embeddings
✓ Cacheable: User embeddings can be cached
✓ Flexible: Easy to update towers separately
```

**Use Cases:**
- Google YouTube recommendations
- Meta Facebook feed
- Pinterest recommendations
- Spotify Discover

## Training Pipeline

### Dataset Preparation

```python
class RankingDataset(Dataset):
    """PyTorch dataset for ranking."""

    def __init__(self, user_features, item_features, labels):
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_features[idx],
            self.item_features[idx],
            self.labels[idx]
        )
```

### Training Loop

```python
def train_model(model, train_loader, val_loader, num_epochs=50):
    """Train two-tower model."""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_auc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for user_feat, item_feat, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(user_feat, item_feat)
            loss = criterion(predictions, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for user_feat, item_feat, labels in val_loader:
                preds = model(user_feat, item_feat)
                val_predictions.append(preds)
                val_labels.append(labels)

        # Calculate metrics
        val_predictions = torch.cat(val_predictions).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_auc = roc_auc_score(val_labels, val_predictions)

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model, f'best_model_{val_auc:.4f}.pt')

        # Logging
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"  Val AUC: {val_auc:.4f}")
```

### Loss Functions

```python
# Binary Cross-Entropy (default)
def bce_loss(predictions, labels):
    return -labels * log(predictions) - (1-labels) * log(1-predictions)

# Weighted BCE (for imbalanced data)
def weighted_bce_loss(predictions, labels, pos_weight=2.0):
    return -pos_weight * labels * log(predictions) - (1-labels) * log(1-predictions)

# Pairwise Ranking Loss
def pairwise_loss(pos_score, neg_score, margin=0.5):
    return max(0, margin - (pos_score - neg_score))
```

## Advanced Techniques

### Hard Negative Mining

```python
def sample_hard_negatives(user_id, positive_items, all_items, model, k=10):
    """
    Sample items user didn't interact with but model ranks highly.
    These are "hard" negatives - model thinks they're good but user skipped.
    """
    # Get user embedding
    user_emb = model.get_user_embedding(user_features[user_id])

    # Get candidate items (excluding positives)
    candidates = all_items - set(positive_items)

    # Score all candidates
    candidate_embeddings = model.get_item_embedding(item_features[candidates])
    scores = user_emb @ candidate_embeddings.T

    # Sample top-k as hard negatives
    hard_negatives = candidates[argsort(scores)[-k:]]

    return hard_negatives
```

### Batch Hard Mining

```python
def batch_hard_triplet_loss(user_emb, item_emb, labels, margin=0.3):
    """
    For each anchor (user-positive_item):
    - Find hardest positive in batch
    - Find hardest negative in batch
    - Compute triplet loss
    """
    # Distance matrix
    distances = cdist(user_emb, item_emb)

    # For each positive pair, find hardest negative
    loss = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            # Hardest negative: closest item with label=0
            neg_dists = distances[i][labels == 0]
            hardest_neg = neg_dists.min()

            # Positive distance
            pos_dist = distances[i][i]

            # Triplet loss
            loss += max(0, pos_dist - hardest_neg + margin)

    return loss / (labels == 1).sum()
```

### Multi-Task Learning

```python
class MultiTaskTwoTower(nn.Module):
    """
    Learn multiple objectives simultaneously:
    - Click prediction
    - Like prediction
    - Share prediction
    """

    def __init__(self, user_dim, item_dim, embedding_dim=64):
        super().__init__()

        # Shared towers
        self.user_tower = UserTower(user_dim, embedding_dim)
        self.item_tower = ItemTower(item_dim, embedding_dim)

        # Task-specific heads
        self.click_head = nn.Linear(embedding_dim * 2, 1)
        self.like_head = nn.Linear(embedding_dim * 2, 1)
        self.share_head = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_feat, item_feat):
        user_emb = self.user_tower(user_feat)
        item_emb = self.item_tower(item_feat)

        # Concatenate embeddings
        combined = torch.cat([user_emb, item_emb], dim=1)

        # Multiple predictions
        click_pred = torch.sigmoid(self.click_head(combined))
        like_pred = torch.sigmoid(self.like_head(combined))
        share_pred = torch.sigmoid(self.share_head(combined))

        return click_pred, like_pred, share_pred
```

## Model Evaluation

### Offline Metrics

```python
def evaluate_model(model, test_loader):
    """Comprehensive offline evaluation."""

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for user_feat, item_feat, labels in test_loader:
            preds = model(user_feat, item_feat)
            all_predictions.append(preds.numpy())
            all_labels.append(labels.numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    # Metrics
    metrics = {
        'auc': roc_auc_score(labels, predictions),
        'logloss': log_loss(labels, predictions),
        'accuracy': accuracy_score(labels, predictions > 0.5),
        'precision': precision_score(labels, predictions > 0.5),
        'recall': recall_score(labels, predictions > 0.5)
    }

    return metrics
```

### Ranking Metrics

```python
def evaluate_ranking(model, user_items_dict, k=50):
    """Evaluate ranking quality."""

    ndcg_scores = []
    precision_scores = []

    for user_id, ground_truth_items in user_items_dict.items():
        # Get rankings
        rankings = get_top_k_items(model, user_id, k=100)

        # NDCG@k
        ndcg = ndcg_score([ground_truth_items], [rankings], k=k)
        ndcg_scores.append(ndcg)

        # Precision@k
        top_k = rankings[:k]
        precision = len(set(top_k) & set(ground_truth_items)) / k
        precision_scores.append(precision)

    return {
        'ndcg@50': np.mean(ndcg_scores),
        'precision@50': np.mean(precision_scores)
    }
```

## Deployment

### Model Export

```python
def export_model(model, filepath):
    """Export for production serving."""

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'user_tower_state': model.user_tower.state_dict(),
        'item_tower_state': model.item_tower.state_dict(),
        'embedding_dim': model.embedding_dim,
        'user_features_dim': model.user_features_dim,
        'item_features_dim': model.item_features_dim,
        'version': datetime.now().isoformat(),
        'metrics': evaluate_model(model, test_loader)
    }

    torch.save(checkpoint, filepath)
```

### Embedding Pre-Computation

```python
def precompute_item_embeddings(model, all_items):
    """
    Pre-compute embeddings for all items.
    Enables fast retrieval at serving time.
    """
    model.eval()

    item_embeddings = {}

    with torch.no_grad():
        for item_id, item_features in all_items.items():
            embedding = model.get_item_embedding(item_features)
            item_embeddings[item_id] = embedding.numpy()

    # Save to cache
    save_to_redis(item_embeddings, ttl=86400)  # 24h

    return item_embeddings
```

## Integration Points

```
Module 3 (Features) → Training data
Module 5 (Serving) → Model deployment
Module 6 (Monitoring) → Performance tracking
Module 7 (System) → Full pipeline
```

## Learning Outcomes

✅ Two-tower architecture
✅ Efficient training at scale
✅ Advanced sampling techniques
✅ Multi-task learning
✅ Production model export
