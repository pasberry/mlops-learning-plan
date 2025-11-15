# Lab 3.3 Solution: Experiment Tracking & Comparison

Complete solution for systematic experiment tracking to compare training runs and select the best models.

## Overview

This solution implements:
- Local JSON-based experiment tracking system
- Automatic logging of hyperparameters, metrics, and artifacts
- Run comparison and analysis tools
- Best model selection based on metrics
- Visualization of training curves

## Key Features

1. **Never Lose Experiments**: Every run automatically logged
2. **Structured Comparison**: Easy to see which config works best
3. **Reproducibility**: Config + metrics + artifacts saved together
4. **No External Dependencies**: Pure Python, no MLflow/Weights&Biases needed
5. **Simple API**: Easy to integrate with existing training code

## Project Structure

```
lab3_3_solution/
├── ml/
│   ├── __init__.py
│   └── tracking/
│       ├── __init__.py
│       └── experiment_tracker.py   # ExperimentTracker class
├── scripts/
│   └── compare_experiments.py      # Comparison utilities
└── README.md
```

## Storage Structure

After running experiments:

```
experiments/
├── metadata.json                    # All experiments index
└── runs/
    ├── run_001/
    │   ├── params.json             # Hyperparameters
    │   ├── metrics.json            # Training metrics (per epoch)
    │   └── artifacts/              # Saved files
    │       ├── model_best.pt
    │       └── plots/
    ├── run_002/
    │   ├── params.json
    │   ├── metrics.json
    │   └── artifacts/
    └── run_003/
        └── ...
```

## Quick Start

### Step 1: Basic Usage

```python
from ml.tracking.experiment_tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(base_dir='experiments')

# Create experiment run
run_id = tracker.create_run(
    experiment_name='ctr_model',
    tags={'version': 'v1', 'team': 'ml'},
    description='Testing higher learning rate'
)

# Log hyperparameters
tracker.log_params(run_id, {
    'learning_rate': 0.001,
    'batch_size': 128,
    'hidden_dims': [256, 128, 64],
    'dropout': 0.3
})

# Log metrics during training
for epoch in range(30):
    # ... training code ...
    tracker.log_metrics(run_id, {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_auc': val_auc
    }, step=epoch)

# Log final test metrics
tracker.log_metrics(run_id, {
    'test_accuracy': test_acc,
    'test_auc': test_auc,
    'test_loss': test_loss
})

# Log artifacts
tracker.log_artifact(run_id, 'path/to/model_best.pt')
tracker.log_artifact(run_id, 'path/to/loss_curve.png')

# Mark as completed
tracker.finish_run(run_id, status='completed')
```

### Step 2: Compare Experiments

```bash
# Compare all runs for an experiment
python scripts/compare_experiments.py ctr_model

# Output:
# Found 5 runs for 'ctr_model'
#
# ================================================================================
# EXPERIMENT COMPARISON: ctr_model
# ================================================================================
# run_id   status     learning_rate  batch_size  test_accuracy  test_auc
# run_005  completed  0.001          128         0.7589         0.8234
# run_003  completed  0.001          128         0.7523         0.8201
# run_001  completed  0.001          128         0.7489         0.8145
# run_002  completed  0.0001         128         0.7401         0.8023
# run_004  completed  0.01           128         0.7234         0.7856
#
# ================================================================================
# BEST RUN:
# ================================================================================
# Run ID: run_005
# Best test_auc: 0.8234
#
# Parameters:
#   learning_rate: 0.001
#   batch_size: 128
#   hidden_dims: [512, 256, 128, 64]
#   dropout: 0.4
#
# Final Metrics:
#   test_accuracy: 0.7589
#   test_auc: 0.8234
#   test_loss: 0.4123
```

### Step 3: Visualize Training Curves

```bash
# Plot validation AUC for multiple runs
python scripts/compare_experiments.py ctr_model plot val_auc run_001 run_002 run_003

# Creates: experiments/metric_comparison_val_auc.png
```

### Step 4: List All Experiments

```bash
python scripts/compare_experiments.py --list

# Output:
# ================================================================================
# ALL EXPERIMENTS
# ================================================================================
#
# ctr_model:
#   Total runs: 5
#   Completed: 4
#   Run IDs: run_001, run_002, run_003, run_004, run_005
#
# ranking_model:
#   Total runs: 3
#   Completed: 3
#   Run IDs: run_006, run_007, run_008
```

## Integration with Training Code

### Update Lab 3.1 Training Script

Add experiment tracking to `ml/training/train.py`:

```python
from ml.tracking.experiment_tracker import ExperimentTracker

def main(config_path: str, output_dir: str = None, experiment_name: str = "ctr_model"):
    """Main training function with experiment tracking."""

    # Initialize experiment tracker
    tracker = ExperimentTracker()

    # Create experiment run
    run_id = tracker.create_run(
        experiment_name=experiment_name,
        tags={'config': config_path},
        description=f"Training run with config {config_path}"
    )

    # Load config
    config = load_config(config_path)

    # Log all parameters
    tracker.log_params(run_id, {
        'model_type': config['model']['type'],
        'hidden_dims': config['model']['hidden_dims'],
        'dropout': config['model']['dropout'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training'].get('weight_decay', 0.0),
        'epochs': config['training']['epochs'],
        'early_stopping_patience': config['training']['early_stopping_patience']
    })

    # ... existing training setup code ...

    # In the training loop, log metrics per epoch
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(...)
        val_metrics = validate_epoch(...)

        # Log epoch metrics
        tracker.log_metrics(run_id, {
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc']
        }, step=epoch)

        # ... rest of training loop ...

    # After training, log final test metrics
    test_metrics = validate_epoch(model, test_loader, criterion, device, epoch=0)

    tracker.log_metrics(run_id, {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc']
    })

    # Log model artifact
    tracker.log_artifact(run_id, str(model_save_path), 'model_best.pt')

    # Finish run
    tracker.finish_run(run_id, status='completed')

    print(f"\n✅ Experiment tracked: {run_id}")

    return output_dir, run_id
```

## Programmatic Access

### Get Best Run

```python
from ml.tracking.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Get best run by AUC
best_run = tracker.get_best_run(
    metric='test_auc',
    experiment_name='ctr_model',
    mode='max'  # or 'min' for loss
)

print(f"Best run: {best_run['run_id']}")
print(f"Best AUC: {best_run['metrics']['test_auc'][-1]['value']:.4f}")

# Load best model
import torch
from ml.models.tabular import TabularClassifier

model = TabularClassifier(**best_run['params'])
model_path = f"{best_run['run_dir']}/artifacts/model_best.pt"
model.load_state_dict(torch.load(model_path))
```

### Compare Runs Programmatically

```python
tracker = ExperimentTracker()

# Get all runs for experiment
runs = tracker.list_runs('ctr_model')
run_ids = [r['run_id'] for r in runs]

# Compare as DataFrame
import pandas as pd
comparison_df = tracker.compare_runs(run_ids)

# Sort by metric
comparison_df = comparison_df.sort_values('test_auc', ascending=False)

# Get top 3 runs
top_3 = comparison_df.head(3)
print(top_3[['run_id', 'learning_rate', 'test_auc']])
```

### Get Specific Run Details

```python
tracker = ExperimentTracker()

# Get run details
run = tracker.get_run('run_001')

print(f"Run ID: {run['run_id']}")
print(f"Status: {run['status']}")
print(f"Created: {run['created_at']}")

print("\nParameters:")
for k, v in run['params'].items():
    print(f"  {k}: {v}")

print("\nMetrics:")
for metric_name, history in run['metrics'].items():
    if history:
        final_value = history[-1]['value']
        print(f"  {metric_name}: {final_value:.4f}")

# Access metric history
val_auc_history = run['metrics']['val_auc']
for entry in val_auc_history:
    print(f"Epoch {entry['step']}: {entry['value']:.4f}")
```

## Hyperparameter Sweep Example

Run multiple experiments with different configs:

```python
import itertools
import yaml
from ml.tracking.experiment_tracker import ExperimentTracker

# Define search space
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [64, 128, 256]
hidden_configs = [
    [256, 128, 64],
    [512, 256, 128],
    [128, 64, 32]
]

tracker = ExperimentTracker()

# Generate all combinations
configs = list(itertools.product(learning_rates, batch_sizes, hidden_configs))

for i, (lr, bs, hidden) in enumerate(configs, 1):
    print(f"\nExperiment {i}/{len(configs)}")
    print(f"LR={lr}, BS={bs}, Hidden={hidden}")

    # Create experiment run
    run_id = tracker.create_run(
        experiment_name='ctr_model_sweep',
        tags={'sweep': 'hyperparams'},
        description=f'Sweep {i}: lr={lr}, bs={bs}'
    )

    # Log params
    tracker.log_params(run_id, {
        'learning_rate': lr,
        'batch_size': bs,
        'hidden_dims': hidden
    })

    # Train model (your training code)
    # ...

    # Log results
    # tracker.log_metrics(run_id, {...})
    # tracker.finish_run(run_id)

# Compare all runs
best = tracker.get_best_run('test_auc', experiment_name='ctr_model_sweep')
print(f"\nBest config: {best['params']}")
```

## API Reference

### ExperimentTracker

#### `__init__(base_dir='experiments')`
Initialize tracker with storage directory.

#### `create_run(experiment_name, tags=None, description='') -> str`
Create a new experiment run. Returns run_id.

#### `log_params(run_id, params: dict)`
Log hyperparameters for a run.

#### `log_metrics(run_id, metrics: dict, step=None)`
Log metrics. Can be called multiple times (e.g., per epoch).

#### `log_artifact(run_id, artifact_path, artifact_name=None)`
Copy and log a file (model, plot, etc.).

#### `finish_run(run_id, status='completed')`
Mark run as finished. Status: 'completed', 'failed', or 'killed'.

#### `get_run(run_id) -> dict`
Get all information for a run.

#### `list_runs(experiment_name=None) -> list`
List all runs, optionally filtered by experiment.

#### `compare_runs(run_ids: list) -> DataFrame`
Compare multiple runs as a DataFrame.

#### `get_best_run(metric, experiment_name=None, mode='max') -> dict`
Get best run based on a metric.

## Data Format

### params.json
```json
{
  "learning_rate": 0.001,
  "batch_size": 128,
  "hidden_dims": [256, 128, 64],
  "dropout": 0.3
}
```

### metrics.json
```json
{
  "train_loss": [
    {"step": 0, "value": 0.6234, "timestamp": "2024-11-15T10:30:00"},
    {"step": 1, "value": 0.5987, "timestamp": "2024-11-15T10:32:00"}
  ],
  "val_auc": [
    {"step": 0, "value": 0.7123, "timestamp": "2024-11-15T10:31:00"},
    {"step": 1, "value": 0.7345, "timestamp": "2024-11-15T10:33:00"}
  ]
}
```

### metadata.json
```json
[
  {
    "run_id": "run_001",
    "experiment_name": "ctr_model",
    "created_at": "2024-11-15T10:30:00",
    "finished_at": "2024-11-15T11:00:00",
    "status": "completed",
    "tags": {"version": "v1"},
    "description": "Baseline model",
    "run_dir": "experiments/runs/run_001"
  }
]
```

## Best Practices

1. **Always Log Params First**: Before training starts
2. **Log Metrics Per Epoch**: Use `step` parameter
3. **Log Final Metrics Separately**: Without step
4. **Descriptive Names**: Use clear experiment names
5. **Add Tags**: For filtering and organization
6. **Finish Runs**: Always call `finish_run()`

## Comparison with MLflow

| Feature | ExperimentTracker | MLflow |
|---------|------------------|--------|
| Setup | Zero config | Requires setup |
| Storage | Local JSON | Database + files |
| UI | CLI/Python | Web UI |
| Scale | Hundreds of runs | Thousands of runs |
| Features | Basic tracking | Advanced features |
| Dependencies | None | mlflow package |

**When to use this**: Small-medium projects, quick experiments, learning
**When to use MLflow**: Large teams, production systems, many users

## Troubleshooting

**Issue**: Metrics not showing in comparison
```python
# Check metrics file exists
import os
print(os.path.exists('experiments/runs/run_001/metrics.json'))

# Check metrics were logged
tracker = ExperimentTracker()
run = tracker.get_run('run_001')
print(run['metrics'])
```

**Issue**: Run not found
```python
# List all runs
tracker = ExperimentTracker()
runs = tracker.list_runs()
print([r['run_id'] for r in runs])
```

**Issue**: Best run returns None
```python
# Check metric name matches exactly
tracker = ExperimentTracker()
run = tracker.get_run('run_001')
print(run['metrics'].keys())  # Check exact metric names
```

## Extensions

Add these features for production:

1. **Search**: Query runs by params/metrics
2. **Cleanup**: Delete old runs
3. **Export**: Export to CSV/MLflow
4. **Alerts**: Email when best model found
5. **Visualization**: More advanced plots
6. **Tagging**: Better organization system

## Next Steps

1. **Lab 3.4**: Build two-tower ranking model
2. **Phase 4**: Deploy best model for serving
3. **Production**: Migrate to MLflow for scale

## References

- Experiment Tracking Best Practices
- MLflow: https://mlflow.org/
- Weights & Biases: https://wandb.ai/
