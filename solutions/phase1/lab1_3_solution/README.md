# Lab 1.3 Solution: Config-Driven Training

Complete, working solution for Lab 1.3 - Configuration-driven ML experiment management.

## Files Included

1. **config.yaml** - Complete YAML configuration file with all training parameters
2. **config_loader.py** - Configuration management utilities
3. **train_mnist_config.py** - Full config-driven training script with TensorBoard
4. **compare_experiments.py** - Experiment comparison and analysis tool

## Why Config-Driven Development?

**The Problem:**
```python
# BAD: Hardcoded parameters scattered in code
model = CNN(32, 64, 128)  # What do these numbers mean?
optimizer = Adam(lr=0.001)  # Hard to reproduce
train(epochs=10)  # Can't compare experiments easily
```

**The Solution:**
```yaml
# GOOD: All parameters in config.yaml
model:
  architecture:
    conv1_channels: 32
    conv2_channels: 64
    fc1_size: 128
training:
  learning_rate: 0.001
  num_epochs: 10
```

**Benefits:**
- ✅ Single source of truth for all parameters
- ✅ Easy to reproduce experiments
- ✅ Version control friendly
- ✅ Compare experiments by comparing configs
- ✅ No code changes needed to try new parameters

## Prerequisites

```bash
# Install dependencies
pip install torch torchvision pyyaml pandas tensorboard

# Verify installations
python -c "import torch, yaml, pandas; print('All packages installed')"
```

## Quick Start

### 1. Navigate to Solution Directory

```bash
cd /home/user/mlops-learning-plan/solutions/phase1/lab1_3_solution
```

### 2. Train with Default Configuration

```bash
# Train with default config
python train_mnist_config.py --config config.yaml
```

**Expected output:**
```
============================================================
Experiment: mnist_baseline
Description: Baseline MNIST classifier with CNN
============================================================

Random seed: 42
Using device: cpu
Output directory: ./models/staging/mnist/mnist_baseline
Config saved to: ./models/staging/mnist/mnist_baseline/config.yaml

Loading data...
Train samples: 60000
Test samples: 10000

Model: ConfigurableCNN
Trainable parameters: 421,226
Optimizer: adam
Learning rate: 0.001
Scheduler: ReduceLROnPlateau
TensorBoard logs: ./runs/mnist_baseline

Starting training for 10 epochs...
...
```

### 3. View Training in TensorBoard

```bash
# Open TensorBoard (in a separate terminal)
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

### 4. Compare Multiple Experiments

```bash
# After training several experiments
python compare_experiments.py --base-dir ./models/staging/mnist --stats
```

## Detailed Usage

### Training Script (train_mnist_config.py)

**Basic usage:**
```bash
python train_mnist_config.py --config config.yaml
```

**Features:**
- ✅ Loads all parameters from YAML config
- ✅ Validates configuration before training
- ✅ Sets random seed for reproducibility
- ✅ Saves config with model checkpoints
- ✅ TensorBoard logging
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Experiment summary in JSON

**Output structure:**
```
./models/staging/mnist/mnist_baseline/
├── config.yaml                  # Copy of config used
├── best_model.pt               # Best model checkpoint
├── checkpoint_epoch_1.pt       # Epoch checkpoints (if enabled)
├── checkpoint_epoch_2.pt
└── experiment_summary.json     # Final results
```

### Configuration File (config.yaml)

**Key sections:**

```yaml
# Experiment metadata
experiment:
  name: "mnist_baseline"
  description: "Baseline MNIST classifier"
  tags: [mnist, cnn, baseline]

# Model architecture
model:
  architecture:
    conv1_channels: 32
    conv2_channels: 64
    fc1_size: 128
    dropout_conv: 0.25
    dropout_fc: 0.5

# Training parameters
training:
  batch_size: 64
  num_epochs: 10
  learning_rate: 0.001
  optimizer: "adam"

  # Learning rate scheduler
  scheduler:
    enabled: true
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 2

  # Early stopping
  early_stopping:
    enabled: true
    patience: 5

# Logging
logging:
  checkpoint_dir: "./models/staging/mnist"
  tensorboard_dir: "./runs"
```

### Configuration Loader (config_loader.py)

**Usage examples:**

```python
from config_loader import load_config, save_config, set_seed, get_device

# Load config
config = load_config('config.yaml')

# Access with dot notation
print(config.training.learning_rate)  # 0.001
print(config.model.name)  # ConfigurableCNN

# Access with dictionary notation
print(config['training']['batch_size'])  # 64

# Set random seed
set_seed(config.device.seed)

# Get device
device = get_device(config)

# Save config
save_config(config, 'output/config_backup.yaml')
```

**Functions:**
- `load_config(path)` - Load YAML config into Config object
- `save_config(config, path)` - Save Config object to YAML
- `set_seed(seed)` - Set random seed for reproducibility
- `get_device(config)` - Get device (CPU/GPU) from config
- `validate_config(config)` - Validate config parameters

### Experiment Comparison (compare_experiments.py)

**Basic usage:**
```bash
# Compare all experiments
python compare_experiments.py

# Show detailed statistics
python compare_experiments.py --stats

# Show all columns
python compare_experiments.py --show-all

# Compare by hyperparameter
python compare_experiments.py --compare-by lr

# Export to CSV
python compare_experiments.py --export results.csv
```

**Command-line arguments:**
- `--base-dir`: Directory containing experiments (default: `./models/staging/mnist`)
- `--show-all`: Show all columns in table
- `--stats`: Show detailed statistics
- `--compare-by`: Group by hyperparameter (lr, batch_size, optimizer)
- `--export`: Export results to CSV file

**Example output:**
```
====================================================================================================
EXPERIMENT COMPARISON
====================================================================================================

name              best_acc  epochs  lr      batch_size  optimizer
mnist_large         98.45      8   0.0010          64  adam
mnist_baseline      98.23     10   0.0010          64  adam
mnist_small         97.89      9   0.0010          64  adam
mnist_sgd           97.65     10   0.0100          64  sgd

====================================================================================================
Total experiments: 4
Best accuracy: 98.45% (mnist_large)
====================================================================================================

====================================================================================================
BEST CONFIGURATION (by best_acc)
====================================================================================================

  name: mnist_large
  best_acc: 98.45
  lr: 0.001
  batch_size: 64
  optimizer: adam
  conv1_ch: 32
  conv2_ch: 128
  fc1_size: 256
  ...
====================================================================================================
```

## Running Multiple Experiments

### Experiment 1: Baseline (Default)

```bash
# Use default config
python train_mnist_config.py --config config.yaml
```

### Experiment 2: Larger Model

Create `config_large.yaml`:
```yaml
experiment:
  name: "mnist_large"
  description: "Larger model with more channels"

model:
  architecture:
    conv1_channels: 32
    conv2_channels: 128  # Increased from 64
    fc1_size: 256        # Increased from 128

# ... rest same as config.yaml
```

Run:
```bash
python train_mnist_config.py --config config_large.yaml
```

### Experiment 3: Different Optimizer

Create `config_sgd.yaml`:
```yaml
experiment:
  name: "mnist_sgd"
  description: "Using SGD instead of Adam"

training:
  optimizer: "sgd"
  learning_rate: 0.01  # Higher LR for SGD

# ... rest same as config.yaml
```

Run:
```bash
python train_mnist_config.py --config config_sgd.yaml
```

### Experiment 4: Smaller Model (Faster Training)

Create `config_small.yaml`:
```yaml
experiment:
  name: "mnist_small"
  description: "Smaller model for faster training"

model:
  architecture:
    conv1_channels: 16   # Reduced from 32
    conv2_channels: 32   # Reduced from 64
    fc1_size: 64         # Reduced from 128

training:
  batch_size: 128  # Larger batches for speed

# ... rest same as config.yaml
```

Run:
```bash
python train_mnist_config.py --config config_small.yaml
```

### Compare All Experiments

```bash
python compare_experiments.py --stats --compare-by optimizer
```

## Advanced Features

### 1. Learning Rate Scheduling

The config includes a learning rate scheduler that reduces LR when validation loss plateaus:

```yaml
training:
  scheduler:
    enabled: true
    type: "ReduceLROnPlateau"
    factor: 0.5       # Reduce LR by 50%
    patience: 2       # Wait 2 epochs before reducing
    min_lr: 0.00001   # Minimum learning rate
```

Watch for scheduler messages during training:
```
Epoch 00003: reducing learning rate of group 0 to 5.0000e-04.
```

### 2. Early Stopping

Prevents overfitting by stopping when validation loss stops improving:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 5      # Stop if no improvement for 5 epochs
    min_delta: 0.001 # Minimum improvement threshold
```

### 3. TensorBoard Visualization

View training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir=./runs

# Open http://localhost:6006
```

**What you'll see:**
- Training and validation loss curves
- Training and validation accuracy
- Learning rate changes over time
- Compare multiple experiments side-by-side

### 4. Experiment Reproducibility

Every experiment saves:
1. **Config file** - Exact parameters used
2. **Random seed** - For reproducible results
3. **Model checkpoint** - Model weights and optimizer state
4. **Experiment summary** - Final metrics and config

**To reproduce an experiment:**
```bash
# Use the saved config
python train_mnist_config.py --config ./models/staging/mnist/mnist_baseline/config.yaml
```

### 5. Checkpoint Management

**Save all checkpoints:**
```yaml
logging:
  save_best_only: false  # Save every epoch
```

**Save only best model:**
```yaml
logging:
  save_best_only: true  # Save only when accuracy improves
```

## Example Workflows

### Workflow 1: Hyperparameter Tuning

```bash
# 1. Create configs with different learning rates
for lr in 0.001 0.0001 0.0005; do
  cat > config_lr${lr}.yaml <<EOF
experiment:
  name: "mnist_lr${lr}"
training:
  learning_rate: ${lr}
# ... rest of config
EOF
  python train_mnist_config.py --config config_lr${lr}.yaml
done

# 2. Compare results
python compare_experiments.py --compare-by lr
```

### Workflow 2: Architecture Search

```bash
# Test different model sizes
python train_mnist_config.py --config config_small.yaml
python train_mnist_config.py --config config.yaml
python train_mnist_config.py --config config_large.yaml

# Compare
python compare_experiments.py --stats
```

### Workflow 3: Optimizer Comparison

```bash
# Test different optimizers
python train_mnist_config.py --config config_adam.yaml
python train_mnist_config.py --config config_sgd.yaml
python train_mnist_config.py --config config_adamw.yaml

# Compare
python compare_experiments.py --compare-by optimizer
```

## Common Issues and Troubleshooting

### Issue 1: "Config file not found"

**Solution:**
```bash
# Check file exists
ls -lh config.yaml

# Use absolute path
python train_mnist_config.py --config /full/path/to/config.yaml
```

### Issue 2: "Missing required field in config"

**Solution:**
```bash
# Validate config
python -c "from config_loader import load_config, validate_config; validate_config(load_config('config.yaml'))"
```

### Issue 3: TensorBoard shows no data

**Solution:**
```bash
# Check TensorBoard directory
ls -lh ./runs/

# Make sure training has started
# TensorBoard updates every 30 seconds

# Specify exact directory
tensorboard --logdir=./runs/mnist_baseline
```

### Issue 4: Early stopping too aggressive

**Solution:**
```yaml
# Increase patience
training:
  early_stopping:
    patience: 10  # Wait longer before stopping
```

### Issue 5: Out of memory

**Solution:**
```yaml
# Reduce batch size
training:
  batch_size: 32  # Or even 16

# Or reduce model size
model:
  architecture:
    conv2_channels: 32  # Smaller model
    fc1_size: 64
```

## Performance Benchmarks

**Expected Results (10 epochs):**

| Configuration | Accuracy | Training Time (CPU) | Parameters |
|---------------|----------|---------------------|------------|
| Small         | 97.5-98% | 3-5 min            | ~100K      |
| Baseline      | 98-99%   | 5-8 min            | ~400K      |
| Large         | 98.5-99% | 8-12 min           | ~1M        |

## Best Practices

### Configuration Management

✅ **One config per experiment**
```bash
experiments/
├── baseline.yaml
├── large_model.yaml
├── sgd_optimizer.yaml
└── high_dropout.yaml
```

✅ **Descriptive experiment names**
```yaml
experiment:
  name: "mnist_large_dropout05_lr0001"  # Describes key changes
```

✅ **Version control configs**
```bash
git add config/*.yaml
git commit -m "Add SGD optimizer experiment config"
```

✅ **Document experiments**
```yaml
experiment:
  description: "Testing higher dropout to prevent overfitting"
  tags: [regularization, dropout, baseline_comparison]
```

### Experiment Organization

✅ **Consistent naming**
- Use prefixes: `mnist_`, `cifar_`, etc.
- Include key parameters: `_lr0001`, `_bs128`

✅ **Track everything**
- Save configs with checkpoints
- Log to TensorBoard
- Export summaries to JSON

✅ **Compare systematically**
- Change one thing at a time
- Run multiple seeds for important experiments
- Document what worked and what didn't

## Key Concepts Demonstrated

### Configuration Management
- ✅ YAML configuration files
- ✅ Nested configuration structure
- ✅ Config validation
- ✅ Dot notation access to config values

### Experiment Tracking
- ✅ Saving experiment metadata
- ✅ TensorBoard integration
- ✅ JSON experiment summaries
- ✅ Checkpoint management

### Reproducibility
- ✅ Random seed setting
- ✅ Config versioning
- ✅ Saving all hyperparameters
- ✅ Deterministic training (when possible)

### Automation
- ✅ Scheduler management from config
- ✅ Early stopping from config
- ✅ Optimizer selection from config
- ✅ Model architecture from config

## Success Criteria

You've successfully completed Lab 1.3 if:

- ✅ Training runs with config file
- ✅ Can modify parameters without changing code
- ✅ Can run multiple experiments with different configs
- ✅ TensorBoard shows training curves
- ✅ Can compare experiments programmatically
- ✅ Experiment summaries are saved
- ✅ Can reproduce experiments from saved configs

## What's Next?

After completing this lab, you have:
- ✅ Mastered Airflow DAG orchestration (Lab 1.1)
- ✅ Built PyTorch training pipelines (Lab 1.2)
- ✅ Implemented config-driven experiments (Lab 1.3)

**Phase 1 Complete!** 🎉

**Next:** Phase 2 - Data Pipelines & Feature Engineering

---

**Congratulations on mastering config-driven ML development!** 🎉

You now have a solid foundation in MLOps fundamentals. The skills you've learned in Phase 1 will be the building blocks for the more advanced topics in Phases 2-4.
