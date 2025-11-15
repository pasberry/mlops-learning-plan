# Lab 3.1 Solution: Tabular Classifier with PyTorch

Complete solution for building a production-ready tabular classification model for CTR (Click-Through Rate) prediction.

## Overview

This solution implements:
- Multi-layer perceptron (MLP) for binary classification
- Modular code structure with separate data, model, and training modules
- Configuration-driven training
- Automatic checkpointing and early stopping
- Comprehensive evaluation metrics (accuracy, AUC)

## Project Structure

```
lab3_1_solution/
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Base model interface
│   │   └── tabular.py       # TabularClassifier implementation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py      # TabularDataset class
│   │   └── loaders.py       # DataLoader utilities
│   └── training/
│       ├── __init__.py
│       ├── trainer.py       # Training loops
│       └── train.py         # CLI training script
├── config/
│   └── model_config.yaml    # Model configuration
├── scripts/
│   └── generate_sample_features.py  # Data generation
└── README.md
```

## Quick Start

### Step 1: Install Dependencies

```bash
pip install torch pandas pyarrow pyyaml scikit-learn tqdm
```

### Step 2: Generate Sample Data

```bash
cd /home/user/mlops-learning-plan/solutions/phase3/lab3_1_solution
python scripts/generate_sample_features.py
```

This creates:
- `data/features/v1/train/data.parquet` (7,000 samples)
- `data/features/v1/val/data.parquet` (1,500 samples)
- `data/features/v1/test/data.parquet` (1,500 samples)

### Step 3: Train the Model

```bash
python ml/training/train.py --config config/model_config.yaml
```

Expected output:
```
📋 Configuration loaded:
...

🔧 Using device: cpu

📊 Loading data...
Loaded 7000 samples from data/features/v1/train/data.parquet
  Features: 16
  Positive rate: 0.423

🏗 Creating model...
TabularClassifier(
  input_dim=16,
  hidden_dims=[256, 128, 64],
  dropout=0.3,
  output_dim=1,
  params=54,785
)

🚀 Starting training...

Epoch 1/30 [Train]: 100%|████████| 55/55 [00:02<00:00]
Epoch 1/30 [Val]: 100%|██████████| 12/12 [00:00<00:00]

Epoch 1/30:
  Train Loss: 0.5234, Acc: 0.7123
  Val Loss: 0.4987, Acc: 0.7234, AUC: 0.7654
  ✅ Best model saved (val_loss: 0.4987)

...

✅ Training complete!
📁 Artifacts saved to experiments/runs/ctr_model_20241115_120530/
   - model_best.pt
   - config.yaml
   - metrics.json
   - history.json
```

### Step 4: Review Results

Check the saved artifacts:

```bash
# View metrics
cat experiments/runs/ctr_model_20241115_120530/metrics.json

# Output:
{
  "test_loss": 0.4856,
  "test_accuracy": 0.7453,
  "test_auc": 0.7891,
  "best_val_loss": 0.4812,
  "best_val_auc": 0.7923
}

# View training history
cat experiments/runs/ctr_model_20241115_120530/history.json
```

## Model Architecture

```
Input Features (16 dims)
       ↓
   Linear(16 → 256)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(256 → 128)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(128 → 64)
       ↓
   ReLU + Dropout(0.3)
       ↓
   Linear(64 → 1)
       ↓
   Sigmoid (output probability)
```

Total parameters: ~54,785

## Configuration

Edit `config/model_config.yaml` to customize:

```yaml
model:
  hidden_dims: [256, 128, 64]  # Layer sizes
  dropout: 0.3                 # Dropout rate

training:
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5
```

## Features

### Data Features (16 total)
- User: age (normalized), gender
- Ad: category, size (3 classes), position (1-5)
- Context: hour, day_of_week, device (3 classes)
- Derived: is_weekend, is_evening, user_ad_affinity, recency_score

### Target
- Binary label: 1 (clicked), 0 (not clicked)
- Positive rate: ~42%

## Training Details

- **Loss**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **Early Stopping**: Patience of 5 epochs on validation loss
- **Metrics**: Loss, Accuracy, AUC-ROC

## Usage Examples

### Custom Training

```bash
# Train with different config
python ml/training/train.py \
    --config config/my_config.yaml \
    --output-dir experiments/my_experiment
```

### Load Trained Model

```python
import torch
from ml.models.tabular import TabularClassifier

# Load model
model = TabularClassifier(
    input_dim=16,
    hidden_dims=[256, 128, 64],
    dropout=0.3
)
model.load_state_dict(torch.load('experiments/runs/.../model_best.pt'))
model.eval()

# Make predictions
import numpy as np
features = np.random.randn(1, 16)  # Example features
with torch.no_grad():
    prob = model(torch.FloatTensor(features))
    print(f"Click probability: {prob.item():.3f}")
```

## Testing Individual Components

```bash
# Test model
python ml/models/tabular.py

# Test dataset
python ml/data/datasets.py

# Test dataloaders
python ml/data/loaders.py
```

## Expected Performance

With default configuration:
- **Training Accuracy**: ~74-76%
- **Validation AUC**: ~0.78-0.80
- **Test AUC**: ~0.77-0.79
- **Training Time**: ~2-3 minutes on CPU (10,000 samples)

## Key Takeaways

1. **Modular Design**: Separate concerns (data, model, training)
2. **Configuration-Driven**: All hyperparameters in YAML file
3. **Reproducibility**: Save config + weights + metrics together
4. **Early Stopping**: Prevent overfitting automatically
5. **Proper Evaluation**: Separate validation and test sets, use AUC for imbalanced data

## Next Steps

1. **Lab 3.2**: Integrate this training into Airflow DAG
2. **Lab 3.3**: Add experiment tracking to compare multiple runs
3. **Lab 3.4**: Build two-tower ranking model

## Troubleshooting

**Issue**: Import errors
- Make sure you're in the lab3_1_solution directory
- Check all `__init__.py` files exist

**Issue**: Data not found
- Run `python scripts/generate_sample_features.py` first
- Check `data/features/v1/` directories exist

**Issue**: Poor model performance (AUC < 0.70)
- Increase epochs or reduce early_stopping_patience
- Try different learning rates: 0.0001, 0.001, 0.01
- Increase model capacity: `[512, 256, 128, 64]`

**Issue**: Training too slow
- Reduce batch size
- Reduce number of samples in generate_sample_features.py
- Simplify model architecture

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Binary Classification Best Practices
- AUC-ROC for Imbalanced Classification
