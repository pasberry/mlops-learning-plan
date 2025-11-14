# Phase 3: Modeling & Training with PyTorch

**Goal**: Integrate PyTorch training into Airflow, build production ML code, and implement experiment tracking.

---

## Overview

In Phase 2, you built data pipelines that ingest, validate, and engineer features. Now it's time to **train models** on that data and orchestrate the entire process with Airflow.

By the end of Phase 3, you'll have:
- ✅ Structured ML codebase (separation of concerns)
- ✅ Tabular classification model in PyTorch
- ✅ Two-tower ranking model
- ✅ Training DAG integrated with data pipeline
- ✅ Experiment tracking system
- ✅ Model registry for versioning

---

## The Big Picture: Where Training Fits

```
┌─────────────────────────────────────────────────────────────┐
│                     ML PIPELINE FLOW                        │
│                                                             │
│  Phase 2: DATA PREP          Phase 3: TRAINING             │
│  ┌──────────────┐           ┌──────────────┐              │
│  │ ETL Pipeline │──────────▶│   Training   │              │
│  │              │           │   Pipeline   │              │
│  │ - Ingest     │           │              │              │
│  │ - Validate   │           │ - Load data  │              │
│  │ - Features   │           │ - Train      │              │
│  └──────────────┘           │ - Evaluate   │              │
│                             │ - Register   │              │
│                             └──────────────┘              │
│                                     │                      │
│                                     ▼                      │
│                             ┌──────────────┐              │
│                             │   Model      │              │
│                             │  Registry    │              │
│                             └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Training is just another pipeline. It consumes feature data from Phase 2 and produces model artifacts.

---

## Phase 3 Learning Objectives

After completing Phase 3, you'll be able to:

1. **Structure ML code professionally**
   - Separate data loading, model definition, and training logic
   - Write modular, testable PyTorch code
   - Use configuration files to drive experiments

2. **Build production PyTorch models**
   - Tabular classifiers (e.g., CTR prediction)
   - Two-tower ranking models (e.g., recommendations)
   - Custom loss functions and metrics

3. **Integrate training with Airflow**
   - Create training DAGs that depend on data pipelines
   - Pass artifacts between tasks
   - Handle model evaluation and selection

4. **Track experiments systematically**
   - Log hyperparameters, metrics, and artifacts
   - Compare multiple runs
   - Select and register best models

5. **Version and register models**
   - Save model checkpoints with metadata
   - Implement basic model registry
   - Promote models from staging to production

---

## PyTorch Project Structure

We'll organize ML code into clear modules:

```
ml/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── datasets.py          # PyTorch Dataset classes
│   └── loaders.py           # DataLoader utilities
├── models/
│   ├── __init__.py
│   ├── tabular.py           # Tabular models
│   ├── ranking.py           # Two-tower models
│   └── base.py              # Base model interface
├── training/
│   ├── __init__.py
│   ├── train.py             # Training script (CLI entry point)
│   ├── trainer.py           # Training loop logic
│   └── evaluate.py          # Evaluation logic
└── registry/
    ├── __init__.py
    └── model_registry.py    # Model versioning
```

### Why This Structure?

**Separation of Concerns**:
- **data/**: Data loading is independent of model architecture
- **models/**: Models are defined separately from training logic
- **training/**: Training logic can work with any model
- **registry/**: Model management is a separate concern

**Benefits**:
- Easy to test each component independently
- Swap models without changing training code
- Reuse data loaders across experiments
- Clear interfaces between components

---

## Key Concepts

### 1. Config-Driven Training

All hyperparameters and paths come from config files:

```yaml
# config/model_config.yaml
model:
  type: "tabular_classifier"
  hidden_dims: [256, 128, 64]
  dropout: 0.3

training:
  batch_size: 512
  learning_rate: 0.001
  epochs: 20
  early_stopping_patience: 3

data:
  features_path: "data/features/v1/train"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

**Why?**
- Reproducibility: Exact config is saved with each run
- Easy experimentation: Change config, run again
- No hardcoded values in code

### 2. PyTorch Dataset Pattern

```python
class TabularDataset(torch.utils.data.Dataset):
    """Load features and labels for training."""

    def __init__(self, features_path, split='train'):
        self.data = pd.read_parquet(f"{features_path}/{split}/data.parquet")
        self.features = self.data.drop('label', axis=1).values
        self.labels = self.data['label'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }
```

**Pattern**: Dataset encapsulates data loading logic. DataLoader handles batching.

### 3. Training Loop Anatomy

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Single training epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
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

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

**Key steps**: forward → loss → backward → optimize

### 4. Experiment Tracking

Every training run logs:
- **Config**: Hyperparameters, data paths
- **Metrics**: Loss, accuracy, AUC per epoch
- **Artifacts**: Model checkpoints, plots

```python
# experiments/runs/{run_id}/
#   ├── config.yaml           # Exact config used
#   ├── metrics.json          # Training metrics
#   ├── model_best.pt         # Best checkpoint
#   └── plots/
#       └── loss_curve.png
```

### 5. Model Registry Pattern

```python
# models/staging/{model_name}/{version}/
#   ├── model.pt              # PyTorch weights
#   ├── config.yaml           # Model config
#   └── metrics.json          # Evaluation metrics

# Promote to production
# models/production/{model_name}/
#   ├── current -> v3/        # Symlink to current version
#   ├── v1/
#   ├── v2/
#   └── v3/
```

**Pattern**: Staging for experiments, production for deployed models.

---

## Airflow + PyTorch Integration

### Training DAG Structure

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('training_pipeline', schedule_interval='@weekly') as dag:

    # Task 1: Prepare training data
    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_training_data
    )

    # Task 2: Train model
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_training
    )

    # Task 3: Evaluate model
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_on_test_set
    )

    # Task 4: Register best model
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_to_staging
    )

    prepare_data >> train_model >> evaluate_model >> register_model
```

### Passing Data Between Tasks

**Option 1: XComs (small data)**
```python
def train_model(**context):
    # Training logic...
    metrics = {'loss': 0.23, 'auc': 0.89}
    context['ti'].xcom_push(key='metrics', value=metrics)

def evaluate_model(**context):
    metrics = context['ti'].xcom_pull(key='metrics')
    # Use metrics...
```

**Option 2: File paths (large data)**
```python
def train_model(**context):
    model_path = f"models/staging/ctr_model/v{version}/model.pt"
    torch.save(model.state_dict(), model_path)
    context['ti'].xcom_push(key='model_path', value=model_path)
```

---

## Lab Overview

### Lab 3.1: Build Tabular Classifier
- Create PyTorch dataset for tabular data
- Build multi-layer perceptron (MLP) model
- Implement training script
- Structure code properly (data.py, model.py, train.py)

**Output**: Standalone training script that works with Phase 2 features

### Lab 3.2: Training DAG
- Create Airflow DAG for training orchestration
- Integrate with feature engineering pipeline
- Implement model evaluation
- Register models with metadata

**Output**: End-to-end pipeline from raw data to registered model

### Lab 3.3: Experiment Tracking
- Implement experiment logging (local or MLflow)
- Track hyperparameters, metrics, artifacts
- Compare multiple runs
- Select best model based on metrics

**Output**: Experiment tracking system with run comparison

### Lab 3.4: Two-Tower Model
- Build two-tower architecture for ranking
- Implement embedding layers
- Custom loss functions (e.g., triplet loss)
- Train and evaluate

**Output**: Advanced model architecture for recommendations

---

## Prerequisites from Phase 2

Before starting Phase 3, ensure you have:

1. **Feature data**: Output from Phase 2 feature engineering
   - Path: `data/features/v1/{train,val,test}/`
   - Format: Parquet files with features + labels

2. **Data pipeline DAG**: Working ETL → feature engineering DAG

3. **Directory structure**:
   ```bash
   mkdir -p ml/{data,models,training,registry}
   mkdir -p models/{staging,production}
   mkdir -p experiments/runs
   mkdir -p config
   ```

---

## Success Criteria

You've completed Phase 3 when you can:

✅ Explain the separation between data, model, and training code
✅ Build and train a PyTorch tabular model from scratch
✅ Build a two-tower ranking model
✅ Create an Airflow DAG that orchestrates training
✅ Track experiments and compare runs
✅ Register models with versioning
✅ Load and evaluate saved models
✅ Explain when to retrain (we'll automate in Phase 4)

---

## Common Patterns & Best Practices

### 1. Always Use Config Files
❌ **Bad**: Hardcoded hyperparameters
```python
model = TabularModel(hidden_dims=[128, 64], dropout=0.3)
```

✅ **Good**: Config-driven
```python
config = yaml.safe_load(open('config/model_config.yaml'))
model = TabularModel(**config['model'])
```

### 2. Separate Model Definition from Training
❌ **Bad**: Training logic inside model
```python
class MyModel(nn.Module):
    def train_model(self, data):
        # Training logic in model class
```

✅ **Good**: Model is just architecture
```python
class MyModel(nn.Module):
    def forward(self, x):
        return self.layers(x)

# Training in separate trainer.py
```

### 3. Always Log Experiments
❌ **Bad**: Print to console
```python
print(f"Epoch {epoch}, Loss: {loss}")
```

✅ **Good**: Log to tracking system
```python
experiment_tracker.log_metrics({
    'epoch': epoch,
    'train_loss': train_loss,
    'val_loss': val_loss
})
```

### 4. Save Model + Config Together
❌ **Bad**: Only save weights
```python
torch.save(model.state_dict(), 'model.pt')
```

✅ **Good**: Save weights + config + metrics
```python
torch.save(model.state_dict(), f'{run_dir}/model.pt')
yaml.dump(config, open(f'{run_dir}/config.yaml', 'w'))
json.dump(metrics, open(f'{run_dir}/metrics.json', 'w'))
```

### 5. Use Early Stopping
```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss = validate_epoch(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

---

## Troubleshooting

### "Out of memory" during training
- Reduce batch size
- Use gradient accumulation
- Clear cache: `torch.cuda.empty_cache()`

### "NaN loss" during training
- Check learning rate (try 1e-4 to 1e-3)
- Check data normalization
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_()`

### "Overfitting quickly"
- Add dropout
- Use L2 regularization (weight decay)
- Get more data or augment
- Reduce model capacity

### "Model not improving"
- Check data loading (labels correct?)
- Verify loss function matches problem type
- Try different learning rate
- Check for data leakage

---

## Next Steps

1. **Start with Lab 3.1**: Build your first tabular model
2. **Progress to Lab 3.2**: Integrate into Airflow
3. **Add tracking in Lab 3.3**: Never lose an experiment
4. **Build advanced model in Lab 3.4**: Two-tower architecture

After Phase 3, you'll move to **Phase 4: Deployment & Monitoring**, where you'll:
- Serve models with FastAPI
- Build batch inference pipelines
- Monitor model performance
- Implement automated retraining

---

**Ready to train some models? Let's start with Lab 3.1! 🚀**
