# Lab 3.3: Experiment Tracking & Comparison

**Objective**: Implement systematic experiment tracking to compare training runs and select the best models.

**Time**: 2-3 hours

**Prerequisites**:
- Lab 3.1 completed (tabular model)
- Lab 3.2 completed (training DAG)
- Understanding of hyperparameter tuning

---

## What You'll Build

An experiment tracking system that:
- Logs all training runs with hyperparameters, metrics, and artifacts
- Enables comparison across multiple experiments
- Tracks which config produced which results
- Helps select the best model systematically

**Two Approaches**:
1. **Local JSON-based tracking** (simple, no dependencies)
2. **MLflow** (production-grade, feature-rich)

We'll implement both so you understand the patterns.

---

## Why Experiment Tracking Matters

### The Problem

```python
# Day 1
python train.py --lr 0.001 --hidden 256
# AUC: 0.85

# Day 2
python train.py --lr 0.0001 --hidden 512
# AUC: 0.87

# Day 3
"Wait, which config gave 0.87 again? 🤔"
```

### The Solution

```python
# All experiments logged
experiment_tracker.log_run(
    params={'lr': 0.001, 'hidden': 256},
    metrics={'auc': 0.85},
    artifacts=['model.pt']
)

# Later: compare all runs
best_run = tracker.get_best_run(metric='auc')
```

**Never lose an experiment again!**

---

## Part A: Local JSON-Based Tracking

### Step 1: Create Experiment Tracker

**Create**: `ml/tracking/experiment_tracker.py`

```python
"""Simple experiment tracking using local JSON files."""
import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class ExperimentTracker:
    """Track ML experiments locally using JSON files.

    Structure:
        experiments/
            metadata.json                 # All experiments metadata
            runs/
                run_001/
                    config.yaml           # Run configuration
                    metrics.json          # Training metrics
                    artifacts/            # Model checkpoints, plots
                        model_best.pt
                        loss_curve.png
                run_002/
                ...
    """

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata([])

    def _load_metadata(self) -> List[Dict]:
        """Load experiments metadata."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: List[Dict]):
        """Save experiments metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_run(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: str = ""
    ) -> str:
        """Create a new experiment run.

        Args:
            experiment_name: Name of the experiment (e.g., 'ctr_model')
            tags: Optional tags (e.g., {'team': 'ml', 'priority': 'high'})
            description: Human-readable description

        Returns:
            run_id: Unique run identifier
        """
        # Generate run ID
        metadata = self._load_metadata()
        run_id = f"run_{len(metadata) + 1:03d}"

        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        # Create run metadata
        run_metadata = {
            'run_id': run_id,
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'tags': tags or {},
            'description': description,
            'run_dir': str(run_dir)
        }

        # Add to metadata
        metadata.append(run_metadata)
        self._save_metadata(metadata)

        print(f"✅ Created experiment run: {run_id}")
        print(f"   Directory: {run_dir}")

        return run_id

    def log_params(self, run_id: str, params: Dict[str, Any]):
        """Log hyperparameters for a run.

        Args:
            run_id: Run identifier
            params: Dictionary of parameters
        """
        run_dir = self.runs_dir / run_id
        params_file = run_dir / "params.json"

        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"✅ Logged {len(params)} parameters to {run_id}")

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for a run.

        Args:
            run_id: Run identifier
            metrics: Dictionary of metric name -> value
            step: Optional step/epoch number
        """
        run_dir = self.runs_dir / run_id
        metrics_file = run_dir / "metrics.json"

        # Load existing metrics
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        # Add new metrics
        timestamp = datetime.now().isoformat()
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            **metrics
        }

        # Append to each metric's history
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append({'step': step, 'value': value, 'timestamp': timestamp})

        # Save
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"✅ Logged metrics to {run_id}: {metrics}")

    def log_artifact(self, run_id: str, artifact_path: str, artifact_name: Optional[str] = None):
        """Log an artifact (file) for a run.

        Args:
            run_id: Run identifier
            artifact_path: Path to artifact file
            artifact_name: Name to save as (default: same as source)
        """
        run_dir = self.runs_dir / run_id
        artifacts_dir = run_dir / "artifacts"

        # Determine target name
        if artifact_name is None:
            artifact_name = Path(artifact_path).name

        # Copy artifact
        target_path = artifacts_dir / artifact_name
        shutil.copy(artifact_path, target_path)

        print(f"✅ Logged artifact to {run_id}: {artifact_name}")

        return str(target_path)

    def finish_run(self, run_id: str, status: str = 'completed'):
        """Mark a run as finished.

        Args:
            run_id: Run identifier
            status: 'completed', 'failed', or 'killed'
        """
        metadata = self._load_metadata()

        for run in metadata:
            if run['run_id'] == run_id:
                run['status'] = status
                run['finished_at'] = datetime.now().isoformat()
                break

        self._save_metadata(metadata)
        print(f"✅ Run {run_id} marked as {status}")

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run information.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with run metadata, params, and metrics
        """
        metadata = self._load_metadata()
        run_meta = next((r for r in metadata if r['run_id'] == run_id), None)

        if not run_meta:
            return None

        run_dir = self.runs_dir / run_id

        # Load params
        params_file = run_dir / "params.json"
        params = {}
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)

        # Load metrics
        metrics_file = run_dir / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

        return {
            **run_meta,
            'params': params,
            'metrics': metrics
        }

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Dict]:
        """List all runs, optionally filtered by experiment name.

        Args:
            experiment_name: Filter by experiment name

        Returns:
            List of run metadata dictionaries
        """
        metadata = self._load_metadata()

        if experiment_name:
            metadata = [r for r in metadata if r['experiment_name'] == experiment_name]

        return metadata

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison
        """
        rows = []

        for run_id in run_ids:
            run = self.get_run(run_id)
            if not run:
                continue

            # Extract final metrics (last value in each metric's history)
            final_metrics = {}
            for metric_name, metric_history in run['metrics'].items():
                if metric_history:
                    final_metrics[metric_name] = metric_history[-1]['value']

            row = {
                'run_id': run_id,
                'experiment_name': run['experiment_name'],
                'status': run['status'],
                'created_at': run['created_at'],
                **run['params'],
                **final_metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_run(self, metric: str, experiment_name: Optional[str] = None, mode: str = 'max') -> Optional[Dict]:
        """Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            experiment_name: Filter by experiment name
            mode: 'max' or 'min'

        Returns:
            Best run dictionary
        """
        runs = self.list_runs(experiment_name)

        if not runs:
            return None

        best_run = None
        best_value = float('-inf') if mode == 'max' else float('inf')

        for run_meta in runs:
            run = self.get_run(run_meta['run_id'])
            if metric not in run['metrics']:
                continue

            # Get final metric value
            metric_history = run['metrics'][metric]
            if not metric_history:
                continue

            metric_value = metric_history[-1]['value']

            # Check if better
            is_better = (
                (mode == 'max' and metric_value > best_value) or
                (mode == 'min' and metric_value < best_value)
            )

            if is_better:
                best_value = metric_value
                best_run = run

        return best_run


# Example usage
if __name__ == '__main__':
    tracker = ExperimentTracker()

    # Create a run
    run_id = tracker.create_run(
        experiment_name='ctr_model',
        tags={'version': 'v1', 'team': 'ml'},
        description='Baseline model with default hyperparameters'
    )

    # Log params
    tracker.log_params(run_id, {
        'learning_rate': 0.001,
        'batch_size': 128,
        'hidden_dims': [256, 128, 64]
    })

    # Simulate training and log metrics
    for epoch in range(10):
        tracker.log_metrics(run_id, {
            'train_loss': 0.5 - epoch * 0.03,
            'val_auc': 0.7 + epoch * 0.02
        }, step=epoch)

    # Finish run
    tracker.finish_run(run_id)

    # Get best run
    best = tracker.get_best_run('val_auc', experiment_name='ctr_model')
    print(f"\nBest run: {best['run_id']}")
    print(f"Best val_auc: {best['metrics']['val_auc'][-1]['value']:.4f}")
```

---

### Step 2: Integrate with Training Script

**Update**: `ml/training/train.py` (add experiment tracking)

```python
# At the top, add import
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

    # ... existing training code ...

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

---

### Step 3: Create Experiment Comparison Script

**Create**: `scripts/compare_experiments.py`

```python
"""Compare experiment runs."""
import sys
from ml.tracking.experiment_tracker import ExperimentTracker
import pandas as pd


def compare_experiments(experiment_name: str):
    """Compare all runs for an experiment."""
    tracker = ExperimentTracker()

    # Get all runs
    runs = tracker.list_runs(experiment_name)

    if not runs:
        print(f"No runs found for experiment: {experiment_name}")
        return

    print(f"Found {len(runs)} runs for '{experiment_name}'")

    # Get run IDs
    run_ids = [r['run_id'] for r in runs]

    # Compare
    comparison_df = tracker.compare_runs(run_ids)

    # Sort by test_auc (descending)
    if 'test_auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('test_auc', ascending=False)

    # Display
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPARISON: {experiment_name}")
    print("=" * 80)

    # Select columns to display
    display_cols = [
        'run_id', 'status', 'learning_rate', 'batch_size',
        'test_accuracy', 'test_auc', 'test_loss'
    ]
    display_cols = [c for c in display_cols if c in comparison_df.columns]

    print(comparison_df[display_cols].to_string(index=False))

    # Show best run
    print("\n" + "=" * 80)
    print("BEST RUN (by test_auc):")
    print("=" * 80)

    best_run = tracker.get_best_run('test_auc', experiment_name)
    if best_run:
        print(f"Run ID: {best_run['run_id']}")
        print(f"\nParameters:")
        for key, value in best_run['params'].items():
            print(f"  {key}: {value}")

        print(f"\nFinal Metrics:")
        for metric_name, metric_history in best_run['metrics'].items():
            if metric_history and metric_name.startswith('test_'):
                final_value = metric_history[-1]['value']
                print(f"  {metric_name}: {final_value:.4f}")

    return comparison_df


def plot_metric_comparison(run_ids: list, metric: str):
    """Plot metric across epochs for multiple runs."""
    import matplotlib.pyplot as plt

    tracker = ExperimentTracker()

    plt.figure(figsize=(10, 6))

    for run_id in run_ids:
        run = tracker.get_run(run_id)
        if not run or metric not in run['metrics']:
            continue

        metric_history = run['metrics'][metric]
        steps = [m['step'] for m in metric_history if m['step'] is not None]
        values = [m['value'] for m in metric_history if m['step'] is not None]

        # Get learning rate for label
        lr = run['params'].get('learning_rate', 'unknown')
        label = f"{run_id} (lr={lr})"

        plt.plot(steps, values, marker='o', label=label)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison Across Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = f'experiments/metric_comparison_{metric}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to {plot_path}")

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/compare_experiments.py <experiment_name>")
        print("  python scripts/compare_experiments.py ctr_model")
        print("\nOr to plot a specific metric:")
        print("  python scripts/compare_experiments.py ctr_model plot val_auc run_001 run_002")
        sys.exit(1)

    experiment_name = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == 'plot':
        metric = sys.argv[3]
        run_ids = sys.argv[4:]
        plot_metric_comparison(run_ids, metric)
    else:
        compare_experiments(experiment_name)
```

---

### Step 4: Run Multiple Experiments

Create different configs to compare:

**Create**: `config/model_config_lr001.yaml`
```yaml
model:
  type: "tabular_classifier"
  hidden_dims: [256, 128, 64]
  dropout: 0.3

training:
  batch_size: 128
  learning_rate: 0.001  # Higher LR
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5
  num_workers: 0

data:
  features_path: "data/features/v1"
```

**Create**: `config/model_config_lr0001.yaml`
```yaml
model:
  type: "tabular_classifier"
  hidden_dims: [256, 128, 64]
  dropout: 0.3

training:
  batch_size: 128
  learning_rate: 0.0001  # Lower LR
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5
  num_workers: 0

data:
  features_path: "data/features/v1"
```

**Create**: `config/model_config_deep.yaml`
```yaml
model:
  type: "tabular_classifier"
  hidden_dims: [512, 256, 128, 64]  # Deeper network
  dropout: 0.4

training:
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30
  early_stopping_patience: 5
  num_workers: 0

data:
  features_path: "data/features/v1"
```

**Run experiments**:
```bash
# Experiment 1: LR = 0.001
python ml/training/train.py --config config/model_config_lr001.yaml

# Experiment 2: LR = 0.0001
python ml/training/train.py --config config/model_config_lr0001.yaml

# Experiment 3: Deeper network
python ml/training/train.py --config config/model_config_deep.yaml
```

**Compare**:
```bash
python scripts/compare_experiments.py ctr_model
```

Expected output:
```
Found 3 runs for 'ctr_model'

================================================================================
EXPERIMENT COMPARISON: ctr_model
================================================================================
run_id   status     learning_rate  batch_size  test_accuracy  test_auc  test_loss
run_003  completed  0.001          128         0.7523         0.8234    0.4321
run_001  completed  0.001          128         0.7489         0.8201    0.4356
run_002  completed  0.0001         128         0.7401         0.8123    0.4512

================================================================================
BEST RUN (by test_auc):
================================================================================
Run ID: run_003

Parameters:
  learning_rate: 0.001
  batch_size: 128
  hidden_dims: [512, 256, 128, 64]
  dropout: 0.4

Final Metrics:
  test_accuracy: 0.7523
  test_auc: 0.8234
  test_loss: 0.4321
```

---

## Part B: MLflow Integration (Optional)

MLflow is a production-grade experiment tracking system.

### Step 1: Install MLflow

```bash
pip install mlflow
```

### Step 2: Create MLflow Wrapper

**Create**: `ml/tracking/mlflow_tracker.py`

```python
"""MLflow experiment tracking wrapper."""
import mlflow
import yaml
from pathlib import Path
from typing import Dict, Any


class MLflowTracker:
    """Wrapper around MLflow for experiment tracking."""

    def __init__(self, tracking_uri: str = "file:./experiments/mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        print(f"MLflow tracking URI: {tracking_uri}")

    def start_run(self, experiment_name: str, run_name: str = None, tags: Dict = None):
        """Start an MLflow run.

        Args:
            experiment_name: Name of experiment
            run_name: Name for this run
            tags: Dictionary of tags
        """
        # Set or create experiment
        mlflow.set_experiment(experiment_name)

        # Start run
        mlflow.start_run(run_name=run_name, tags=tags)
        print(f"✅ Started MLflow run: {mlflow.active_run().info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, artifact_path: str):
        """Log an artifact file."""
        mlflow.log_artifact(artifact_path)

    def log_model(self, model_path: str, artifact_path: str = "model"):
        """Log a model."""
        mlflow.log_artifact(model_path, artifact_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        mlflow.end_run(status=status)

    def get_best_run(self, experiment_name: str, metric: str, mode: str = "max"):
        """Get best run for an experiment.

        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Best run info
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
            max_results=1
        )

        if runs.empty:
            return None

        return runs.iloc[0].to_dict()


# Integration example
def train_with_mlflow(config_path: str):
    """Training with MLflow tracking."""
    tracker = MLflowTracker()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Start run
    tracker.start_run(
        experiment_name='ctr_model_mlflow',
        run_name=f"lr_{config['training']['learning_rate']}",
        tags={'config': config_path}
    )

    try:
        # Log all params
        tracker.log_params({
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'hidden_dims': str(config['model']['hidden_dims']),
            'dropout': config['model']['dropout']
        })

        # Training loop (simplified)
        for epoch in range(10):
            # Train...
            train_loss = 0.5 - epoch * 0.03
            val_auc = 0.7 + epoch * 0.02

            # Log metrics
            tracker.log_metrics({
                'train_loss': train_loss,
                'val_auc': val_auc
            }, step=epoch)

        # Log final metrics
        tracker.log_metrics({
            'test_auc': 0.85,
            'test_accuracy': 0.78
        })

        # Log artifacts
        tracker.log_artifact('config/model_config.yaml')

        # End run
        tracker.end_run(status="FINISHED")

    except Exception as e:
        tracker.end_run(status="FAILED")
        raise e


if __name__ == '__main__':
    # Test MLflow tracking
    train_with_mlflow('config/model_config.yaml')

    # View MLflow UI:
    # mlflow ui --backend-store-uri file:./experiments/mlruns
```

### Step 3: View MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./experiments/mlruns

# Open browser to http://localhost:5000
```

---

## Step 5: Hyperparameter Sweep

**Create**: `scripts/hyperparam_sweep.py`

```python
"""Run hyperparameter sweep across multiple configs."""
import subprocess
import itertools
import yaml
from pathlib import Path


def generate_config(base_config_path: str, overrides: dict, output_path: str):
    """Generate a config file with overrides."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        d = config
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    # Save
    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    return output_path


def run_sweep():
    """Run hyperparameter sweep."""
    # Define search space
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [64, 128, 256]
    hidden_configs = [
        [256, 128, 64],
        [512, 256, 128],
        [128, 64, 32]
    ]

    # Base config
    base_config = 'config/model_config.yaml'

    # Generate all combinations
    configs = list(itertools.product(learning_rates, batch_sizes, hidden_configs))

    print(f"🔍 Running hyperparameter sweep with {len(configs)} configurations...")

    for i, (lr, bs, hidden) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"{'='*60}")
        print(f"Learning Rate: {lr}")
        print(f"Batch Size: {bs}")
        print(f"Hidden Dims: {hidden}")

        # Generate config
        overrides = {
            'training.learning_rate': lr,
            'training.batch_size': bs,
            'model.hidden_dims': hidden
        }

        config_path = f'config/sweep_config_{i}.yaml'
        generate_config(base_config, overrides, config_path)

        # Run training
        cmd = [
            'python', 'ml/training/train.py',
            '--config', config_path
        ]

        result = subprocess.run(cmd, cwd='/home/user/mlops-learning-plan')

        if result.returncode != 0:
            print(f"⚠️  Configuration {i} failed")
        else:
            print(f"✅ Configuration {i} completed")

    print(f"\n{'='*60}")
    print("🎉 Sweep complete! Compare results:")
    print("python scripts/compare_experiments.py ctr_model")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_sweep()
```

**Run sweep** (warning: takes time):
```bash
python scripts/hyperparam_sweep.py
```

---

## Deliverables Checklist

- [ ] **Experiment tracker**: `ml/tracking/experiment_tracker.py`
- [ ] **Updated training script** with tracking
- [ ] **Comparison script**: `scripts/compare_experiments.py`
- [ ] **Multiple configs** created and tested
- [ ] **3+ experiment runs** logged
- [ ] **Best run identified** via comparison
- [ ] **Optional**: MLflow integration working

---

## Key Takeaways

1. **Never lose experiments**: Every run is logged automatically
2. **Structured comparison**: Easy to see which config works best
3. **Reproducibility**: Config + metrics + artifacts saved together
4. **Systematic tuning**: No more "I think this worked before..."
5. **Production patterns**: Same approach scales to 1000s of experiments

---

## Next Steps

1. **Lab 3.4**: Build two-tower ranking model
2. **Phase 4**: Deploy best model for serving

---

## Troubleshooting

**Issue**: Metrics not showing in comparison
- Check that metrics are logged with correct names
- Verify metrics.json exists in run directory

**Issue**: Can't find best run
- Ensure runs have 'completed' status
- Check metric name matches exactly

**Issue**: MLflow UI not starting
- Check tracking URI path exists
- Ensure port 5000 is available
- Try different port: `mlflow ui --port 5001`

---

**Excellent! You now have systematic experiment tracking! 🎉**

Next: **Lab 3.4 - Two-Tower Model** for ranking/recommendations.
