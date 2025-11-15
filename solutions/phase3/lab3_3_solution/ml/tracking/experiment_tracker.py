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

        # Append to each metric's history
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append({'step': step, 'value': value, 'timestamp': timestamp})

        # Save
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        if step is not None:
            print(f"✅ Logged metrics to {run_id} (step {step}): {metrics}")
        else:
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
    if best:
        print(f"\nBest run: {best['run_id']}")
        print(f"Best val_auc: {best['metrics']['val_auc'][-1]['value']:.4f}")
