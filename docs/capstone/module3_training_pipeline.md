# Module 3: Model Training DAG

**Estimated Time**: 2-3 days
**Difficulty**: Medium-Hard

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Implement PyTorch ranking models (two-tower and deep MLP architectures)
- ✅ Build production-grade training loops with proper logging
- ✅ Integrate MLflow for experiment tracking and model registry
- ✅ Evaluate ranking models with appropriate metrics (AUC, log loss, NDCG)
- ✅ Orchestrate training with Airflow DAGs
- ✅ Version and register models for production

## Overview

This module builds the **model training pipeline** that:
1. Loads engineered features
2. Trains a PyTorch neural ranking model
3. Evaluates model performance
4. Logs experiments to MLflow
5. Registers the best model
6. Orchestrates everything with Airflow

**Two Model Options**:
- **Two-Tower Model**: Separate embeddings for users and items (scalable)
- **Deep MLP**: Concatenates all features (simpler, good starting point)

## Background

### Neural Ranking Models

**Why Neural Networks for Ranking?**
- Capture non-linear relationships
- Learn embeddings for users and items
- Handle high-dimensional sparse features
- Generalize better than linear models

**Common Architectures**:

1. **Two-Tower** (YouTube, Pinterest):
   - Separate neural networks for user and item
   - Learn embeddings independently
   - Score = dot product of embeddings
   - Advantage: Can pre-compute item embeddings

2. **Deep MLP** (Wide & Deep):
   - Concatenate all features
   - Deep neural network
   - Direct prediction
   - Advantage: Simpler, can learn any interaction

3. **Transformer-based** (BERT4Rec):
   - Sequential modeling of user behavior
   - Attention mechanisms
   - State-of-the-art but complex

**We'll implement options 1 and 2** (two-tower and deep MLP).

### Ranking Metrics

**Binary Classification Metrics**:
- **AUC-ROC**: Measures ability to distinguish clicks from non-clicks
- **Log Loss**: Penalizes confident wrong predictions
- **Precision/Recall**: At various thresholds

**Ranking-Specific Metrics**:
- **Precision@K**: Of top K predictions, how many are relevant?
- **Recall@K**: Of all relevant items, how many in top K?
- **NDCG@K**: Normalized Discounted Cumulative Gain (position-aware)

## Step 1: Implement PyTorch Models

### File Structure
```python
src/models/
├── __init__.py
├── ranker.py       # Model architectures
├── trainer.py      # Training loop
└── evaluator.py    # Evaluation metrics
```

### Model Architectures

**File**: `src/models/ranker.py`

```python
"""
PyTorch ranking model architectures.

Implements:
1. Two-Tower Model
2. Deep MLP Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class TwoTowerModel(nn.Module):
    """
    Two-tower neural ranking model.

    Separate towers for user and item features, combined via dot product.
    """

    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            user_feature_dim: Number of user features
            item_feature_dim: Number of item features
            embedding_dim: Final embedding dimension
            hidden_dims: Hidden layer dimensions for each tower
            dropout: Dropout probability
        """
        super(TwoTowerModel, self).__init__()

        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim

        # User tower
        user_layers = []
        prev_dim = user_feature_dim

        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection to embedding
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)

        # Item tower
        item_layers = []
        prev_dim = item_feature_dim

        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection to embedding
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_features: [batch_size, user_feature_dim]
            item_features: [batch_size, item_feature_dim]

        Returns:
            scores: [batch_size] prediction scores
        """
        # Get embeddings
        user_emb = self.user_tower(user_features)  # [batch_size, embedding_dim]
        item_emb = self.item_tower(item_features)  # [batch_size, embedding_dim]

        # Normalize embeddings (for stable dot product)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Dot product
        scores = (user_emb * item_emb).sum(dim=1)  # [batch_size]

        # Sigmoid for probability
        scores = torch.sigmoid(scores)

        return scores

    def get_user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
        """Get user embedding (for serving)."""
        return F.normalize(self.user_tower(user_features), p=2, dim=1)

    def get_item_embedding(self, item_features: torch.Tensor) -> torch.Tensor:
        """Get item embedding (for serving)."""
        return F.normalize(self.item_tower(item_features), p=2, dim=1)


class DeepMLPModel(nn.Module):
    """
    Deep MLP ranking model.

    Concatenates all features and passes through deep network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Total number of input features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super(DeepMLPModel, self).__init__()

        self.input_dim = input_dim

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [batch_size, input_dim]

        Returns:
            scores: [batch_size] prediction scores
        """
        logits = self.network(features).squeeze(1)  # [batch_size]
        scores = torch.sigmoid(logits)
        return scores


def create_model(config: Dict, feature_dims: Dict) -> nn.Module:
    """
    Factory function to create model based on config.

    Args:
        config: Model configuration dict
        feature_dims: Dict with feature dimensions

    Returns:
        PyTorch model
    """
    model_type = config['model']['type']

    if model_type == 'two_tower':
        model = TwoTowerModel(
            user_feature_dim=feature_dims['user'],
            item_feature_dim=feature_dims['item'],
            embedding_dim=config['model']['embedding_dim'],
            hidden_dims=config['model']['user_tower_dims'],
            dropout=config['model']['dropout']
        )
    elif model_type == 'deep_mlp':
        model = DeepMLPModel(
            input_dim=feature_dims['total'],
            hidden_dims=config['model'].get('hidden_dims', [512, 256, 128]),
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
```

### Training Loop

**File**: `src/models/trainer.py`

```python
"""
Model training logic.

Handles:
- Data loading
- Training loop
- Validation
- Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import yaml


class RankingDataset(Dataset):
    """PyTorch Dataset for ranking data."""

    def __init__(self, features_df: pd.DataFrame, model_type: str = 'deep_mlp'):
        """
        Args:
            features_df: DataFrame with features and target
            model_type: 'two_tower' or 'deep_mlp'
        """
        self.model_type = model_type

        # Extract target
        self.targets = torch.FloatTensor(features_df['click'].values)

        # Extract features
        feature_cols = [
            col for col in features_df.columns
            if col not in ['click', 'timestamp', 'user_id', 'item_id', 'interaction_id']
        ]

        if model_type == 'two_tower':
            # Separate user and item features
            user_feature_cols = [col for col in feature_cols if col.startswith('user_')]
            item_feature_cols = [col for col in feature_cols if col.startswith('item_')]

            self.user_features = torch.FloatTensor(
                features_df[user_feature_cols].values
            )
            self.item_features = torch.FloatTensor(
                features_df[item_feature_cols].values
            )
        else:
            # All features concatenated
            self.features = torch.FloatTensor(features_df[feature_cols].values)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.model_type == 'two_tower':
            return {
                'user_features': self.user_features[idx],
                'item_features': self.item_features[idx],
                'target': self.targets[idx]
            }
        else:
            return {
                'features': self.features[idx],
                'target': self.targets[idx]
            }


class RankingTrainer:
    """Trainer for ranking models."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Load configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str = "feed_ranking"
    ) -> Dict:
        """
        Train model.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            experiment_name: MLflow experiment name

        Returns:
            Dict with training history
        """
        # Setup MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log config
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])

            # Move model to device
            model = model.to(self.device)

            # Setup training
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )

            # Training loop
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_auc': []
            }

            best_val_auc = 0.0
            patience_counter = 0
            patience = self.config['training']['early_stopping']['patience']

            epochs = self.config['training']['epochs']

            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")

                # Train
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
                history['train_loss'].append(train_loss)

                # Validate
                val_loss, val_auc = self._validate(model, val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)

                # Log to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_auc", val_auc, step=epoch)

                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

                # Early stopping
                if val_auc > best_val_auc + self.config['training']['early_stopping']['min_delta']:
                    best_val_auc = val_auc
                    patience_counter = 0

                    # Save best model
                    torch.save(model.state_dict(), "models/training/best_model.pt")
                    print(f"  ✅ New best model! AUC: {val_auc:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print(f"  Early stopping triggered!")
                        break

            # Load best model
            model.load_state_dict(torch.load("models/training/best_model.pt"))

            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model")

            # Log final metrics
            mlflow.log_metric("best_val_auc", best_val_auc)

            return history

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            if 'features' in batch:
                # Deep MLP
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)

                # Forward
                outputs = model(features)
            else:
                # Two-tower
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                targets = batch['target'].to(self.device)

                # Forward
                outputs = model(user_features, item_features)

            # Loss
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move to device
                if 'features' in batch:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    outputs = model(features)
                else:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    outputs = model(user_features, item_features)

                # Loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Store for AUC calculation
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets, all_predictions)

        return avg_loss, auc


def main():
    """Train model."""
    print("Loading data...")
    train_df = pd.read_parquet("data/features/train_features.parquet")
    val_df = pd.read_parquet("data/features/val_features.parquet")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']

    # Create datasets
    train_dataset = RankingDataset(train_df, model_type=model_type)
    val_dataset = RankingDataset(val_df, model_type=model_type)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    from src.models.ranker import create_model

    # Determine feature dimensions
    if model_type == 'two_tower':
        feature_dims = {
            'user': train_dataset.user_features.shape[1],
            'item': train_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {
            'total': train_dataset.features.shape[1]
        }

    model = create_model(config, feature_dims)
    print(f"Created {model_type} model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = RankingTrainer()
    history = trainer.train(model, train_loader, val_loader)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
```

### Evaluation Metrics

**File**: `src/models/evaluator.py`

```python
"""
Model evaluation metrics.

Implements:
- AUC-ROC
- Log Loss
- Precision@K
- Recall@K
- NDCG@K
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class RankingEvaluator:
    """Evaluate ranking models."""

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            model: Trained PyTorch model
            test_loader: Test data loader

        Returns:
            Dict with metrics
        """
        model.eval()
        model = model.to(self.device)

        all_predictions = []
        all_targets = []

        print("Evaluating model...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if 'features' in batch:
                    features = batch['features'].to(self.device)
                    targets = batch['target']
                    outputs = model(features)
                else:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    targets = batch['target']
                    outputs = model(user_features, item_features)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # Calculate metrics
        metrics = {}

        # AUC
        metrics['auc'] = roc_auc_score(targets, predictions)

        # Log Loss
        metrics['log_loss'] = log_loss(targets, predictions)

        # Precision@K, Recall@K
        for k in [5, 10, 20]:
            precision, recall = self._precision_recall_at_k(
                targets, predictions, k
            )
            metrics[f'precision_at_{k}'] = precision
            metrics[f'recall_at_{k}'] = recall

        # NDCG@K (optional, more complex)
        # metrics['ndcg_at_10'] = self._ndcg_at_k(targets, predictions, 10)

        return metrics

    def _precision_recall_at_k(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        k: int
    ) -> tuple:
        """
        Calculate Precision@K and Recall@K.

        Args:
            targets: Ground truth binary labels
            predictions: Predicted scores
            k: Top K items

        Returns:
            (precision@k, recall@k)
        """
        # Get top K predictions
        top_k_indices = np.argsort(predictions)[-k:]
        top_k_targets = targets[top_k_indices]

        # Precision@K: fraction of top K that are relevant
        precision = top_k_targets.sum() / k

        # Recall@K: fraction of all relevant items in top K
        total_relevant = targets.sum()
        recall = top_k_targets.sum() / total_relevant if total_relevant > 0 else 0.0

        return precision, recall


def main():
    """Evaluate trained model."""
    import pandas as pd
    from src.models.ranker import create_model
    from src.models.trainer import RankingDataset
    import yaml

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load test data
    test_df = pd.read_parquet("data/features/test_features.parquet")

    model_type = config['model']['type']
    test_dataset = RankingDataset(test_df, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Load model
    if model_type == 'two_tower':
        feature_dims = {
            'user': test_dataset.user_features.shape[1],
            'item': test_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {
            'total': test_dataset.features.shape[1]
        }

    model = create_model(config, feature_dims)
    model.load_state_dict(torch.load("models/training/best_model.pt"))

    # Evaluate
    evaluator = RankingEvaluator()
    metrics = evaluator.evaluate(model, test_loader)

    print("\n📊 Test Set Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save metrics
    import json
    with open("models/training/test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
```

## Step 2: Build Training DAG

**File**: `dags/training_dag.py`

```python
"""
Model Training DAG.

Orchestrates:
1. Load features
2. Train model
3. Evaluate model
4. Log to MLflow
5. Register model
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
import mlflow

from src.models.ranker import create_model
from src.models.trainer import RankingDataset, RankingTrainer
from src.models.evaluator import RankingEvaluator


default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def train_model():
    """Train ranking model."""
    print("Training model...")

    # Load data
    train_df = pd.read_parquet("data/features/train_features.parquet")
    val_df = pd.read_parquet("data/features/val_features.parquet")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']

    # Create datasets
    train_dataset = RankingDataset(train_df, model_type=model_type)
    val_dataset = RankingDataset(val_df, model_type=model_type)

    # Create loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    if model_type == 'two_tower':
        feature_dims = {
            'user': train_dataset.user_features.shape[1],
            'item': train_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {
            'total': train_dataset.features.shape[1]
        }

    model = create_model(config, feature_dims)

    # Train
    trainer = RankingTrainer()
    history = trainer.train(model, train_loader, val_loader)

    print("Training complete!")


def evaluate_model():
    """Evaluate on test set."""
    print("Evaluating model...")

    # Load test data
    test_df = pd.read_parquet("data/features/test_features.parquet")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']
    test_dataset = RankingDataset(test_df, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Load model
    if model_type == 'two_tower':
        feature_dims = {
            'user': test_dataset.user_features.shape[1],
            'item': test_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {
            'total': test_dataset.features.shape[1]
        }

    model = create_model(config, feature_dims)
    model.load_state_dict(torch.load("models/training/best_model.pt"))

    # Evaluate
    evaluator = RankingEvaluator()
    metrics = evaluator.evaluate(model, test_loader)

    print(f"Test AUC: {metrics['auc']:.4f}")

    # Save metrics
    import json
    with open("models/training/test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def register_model():
    """Register model to MLflow."""
    print("Registering model...")

    # Load metrics
    import json
    with open("models/training/test_metrics.json", 'r') as f:
        metrics = json.load(f)

    # Register to MLflow model registry
    mlflow.set_experiment("feed_ranking")

    # Get latest run
    experiment = mlflow.get_experiment_by_name("feed_ranking")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run_id = runs.iloc[0]['run_id']

    # Register model
    model_uri = f"runs:/{latest_run_id}/model"
    model_details = mlflow.register_model(model_uri, "feed_ranker")

    print(f"Registered model version: {model_details.version}")


with DAG(
    'model_training',
    default_args=default_args,
    description='Train and evaluate ranking model',
    schedule_interval=None,  # Triggered by ETL DAG
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=['training', 'mlflow'],
) as dag:

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    register_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    train_task >> evaluate_task >> register_task
```

## Testing

```bash
# Test model training
cd capstone_project

# 1. Test model creation
python -c "
from src.models.ranker import create_model
import yaml

with open('config/model_config.yaml') as f:
    config = yaml.safe_load(f)

model = create_model(config, {'total': 20})
print(model)
"

# 2. Test training
python src/models/trainer.py

# 3. Test evaluation
python src/models/evaluator.py

# 4. Test DAG
airflow dags test model_training 2024-11-15

# 5. Check MLflow
mlflow ui --port 5000
# Open http://localhost:5000
```

## Review Checklist

- [ ] Model architectures implemented correctly
- [ ] Training loop works end-to-end
- [ ] Validation AUC > 0.70
- [ ] Test AUC > 0.70
- [ ] MLflow logging works
- [ ] Model registered to MLflow
- [ ] DAG runs successfully
- [ ] All metrics calculated correctly

## What to Submit

1. **Model Code**:
   - `src/models/ranker.py`
   - `src/models/trainer.py`
   - `src/models/evaluator.py`

2. **DAG**: `dags/training_dag.py`

3. **Results**:
   - Training curves (loss, AUC)
   - Test set metrics
   - MLflow screenshot

4. **Model Comparison**: Compare two-tower vs deep MLP

5. **Reflection**:
   - Which model performed better?
   - How did training time compare?
   - What would you improve?

## Next Steps

With a trained model, proceed to [Module 4: Model Serving](module4_model_serving.md) to deploy it via API!
