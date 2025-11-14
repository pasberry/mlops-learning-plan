# Lab 3.2: Training DAG - Integrate PyTorch with Airflow

**Objective**: Build an Airflow DAG that orchestrates the complete training pipeline.

**Time**: 2-3 hours

**Prerequisites**:
- Lab 3.1 completed (tabular model built)
- Understanding of Airflow DAGs
- Phase 2 feature engineering DAG (optional but recommended)

---

## What You'll Build

An end-to-end training pipeline DAG that:
1. **Prepares training data** (validates features exist)
2. **Trains the model** (calls PyTorch training script)
3. **Evaluates the model** (tests on held-out set)
4. **Registers the model** (saves to staging with metadata)

```
┌─────────────────────────────────────────────────────┐
│              TRAINING PIPELINE DAG                  │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐             │
│  │   Validate   │───▶│    Train     │             │
│  │    Data      │    │    Model     │             │
│  └──────────────┘    └──────────────┘             │
│                             │                      │
│                             ▼                      │
│                      ┌──────────────┐             │
│                      │   Evaluate   │             │
│                      │    Model     │             │
│                      └──────────────┘             │
│                             │                      │
│                             ▼                      │
│                      ┌──────────────┐             │
│                      │   Register   │             │
│                      │    Model     │             │
│                      └──────────────┘             │
└─────────────────────────────────────────────────────┘
```

---

## Architecture: Airflow ↔ PyTorch Integration

### Key Principle: Thin Orchestration Layer

**Airflow DAG**: Orchestration only (what to run, when, dependencies)
**Python Scripts**: Actual ML logic (how to train)

```python
# ❌ Bad: ML logic in DAG
def train_in_dag():
    model = TabularClassifier(...)
    for epoch in range(100):
        train_epoch(...)  # Too much logic in Airflow!

# ✅ Good: DAG calls external script
def train_in_dag():
    import subprocess
    subprocess.run(['python', 'ml/training/train.py', '--config', 'config/model_config.yaml'])
```

**Why?**
- ML code can be tested independently
- Easy to run outside Airflow (debugging)
- DAG stays clean and readable
- Version control for ML code separate from orchestration

---

## Step 1: Create Model Registry Module

First, we need a way to version and register trained models.

**Create**: `ml/registry/model_registry.py`

```python
"""Model registry for versioning and managing models."""
import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ModelRegistry:
    """Simple file-based model registry.

    Structure:
        models/staging/{model_name}/
            v1/
                model.pt
                config.yaml
                metrics.json
                metadata.json
            v2/
            ...

        models/production/{model_name}/
            current -> v2/  (symlink)
            v1/
            v2/
            ...
    """

    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.staging_dir = self.base_dir / "staging"
        self.production_dir = self.base_dir / "production"

        # Create directories
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        stage: str = "staging",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a trained model.

        Args:
            model_name: Name of the model (e.g., 'ctr_model')
            model_path: Path to model.pt file
            config: Model configuration dict
            metrics: Evaluation metrics dict
            stage: 'staging' or 'production'
            version: Version string (auto-generated if None)
            metadata: Additional metadata

        Returns:
            Path to registered model directory
        """
        # Determine version
        if version is None:
            version = self._get_next_version(model_name, stage)

        # Create model directory
        if stage == "staging":
            model_dir = self.staging_dir / model_name / version
        elif stage == "production":
            model_dir = self.production_dir / model_name / version
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'staging' or 'production'")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        shutil.copy(model_path, model_dir / "model.pt")

        # Save config
        with open(model_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)

        # Save metrics
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'model_name': model_name,
            'version': version,
            'stage': stage,
            'registered_at': datetime.now().isoformat(),
            'model_path': str(model_dir / "model.pt")
        })

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model registered: {model_name} v{version} ({stage})")
        print(f"   Path: {model_dir}")
        print(f"   Metrics: {metrics}")

        return str(model_dir)

    def _get_next_version(self, model_name: str, stage: str) -> str:
        """Get next version number for a model."""
        if stage == "staging":
            model_base_dir = self.staging_dir / model_name
        else:
            model_base_dir = self.production_dir / model_name

        if not model_base_dir.exists():
            return "v1"

        # Find existing versions
        existing_versions = [
            d.name for d in model_base_dir.iterdir()
            if d.is_dir() and d.name.startswith('v')
        ]

        if not existing_versions:
            return "v1"

        # Extract version numbers
        version_numbers = []
        for v in existing_versions:
            try:
                version_numbers.append(int(v[1:]))  # Remove 'v' prefix
            except ValueError:
                continue

        next_version = max(version_numbers) + 1 if version_numbers else 1
        return f"v{next_version}"

    def get_model_path(self, model_name: str, version: str, stage: str = "staging") -> Path:
        """Get path to a specific model version."""
        if stage == "staging":
            return self.staging_dir / model_name / version / "model.pt"
        else:
            return self.production_dir / model_name / version / "model.pt"

    def get_latest_version(self, model_name: str, stage: str = "staging") -> Optional[str]:
        """Get the latest version of a model."""
        if stage == "staging":
            model_base_dir = self.staging_dir / model_name
        else:
            model_base_dir = self.production_dir / model_name

        if not model_base_dir.exists():
            return None

        versions = [
            d.name for d in model_base_dir.iterdir()
            if d.is_dir() and d.name.startswith('v')
        ]

        if not versions:
            return None

        # Sort by version number
        versions.sort(key=lambda v: int(v[1:]))
        return versions[-1]

    def list_models(self, stage: str = "staging") -> Dict[str, list]:
        """List all models in a stage."""
        if stage == "staging":
            base_dir = self.staging_dir
        else:
            base_dir = self.production_dir

        models = {}
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                versions = [
                    v.name for v in model_dir.iterdir()
                    if v.is_dir() and v.name.startswith('v')
                ]
                if versions:
                    models[model_dir.name] = sorted(versions, key=lambda v: int(v[1:]))

        return models

    def promote_to_production(self, model_name: str, version: str) -> str:
        """Promote a staging model to production.

        Args:
            model_name: Name of the model
            version: Version to promote

        Returns:
            Path to production model
        """
        staging_model_dir = self.staging_dir / model_name / version

        if not staging_model_dir.exists():
            raise FileNotFoundError(f"Model not found: {staging_model_dir}")

        # Copy to production
        production_model_dir = self.production_dir / model_name / version
        shutil.copytree(staging_model_dir, production_model_dir, dirs_exist_ok=True)

        # Update metadata
        metadata_path = production_model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['stage'] = 'production'
        metadata['promoted_at'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create 'current' symlink
        current_link = self.production_dir / model_name / "current"
        if current_link.exists():
            current_link.unlink()

        current_link.symlink_to(version, target_is_directory=True)

        print(f"✅ Model promoted to production: {model_name} {version}")
        print(f"   Current production model: {current_link} -> {version}")

        return str(production_model_dir)


# Test
if __name__ == '__main__':
    registry = ModelRegistry()

    # Test registration
    print("Testing model registry...")

    # Simulate registering a model
    test_config = {
        'model': {'hidden_dims': [256, 128, 64]},
        'training': {'batch_size': 128}
    }

    test_metrics = {
        'test_accuracy': 0.85,
        'test_auc': 0.90
    }

    # List models
    print("\nStaging models:", registry.list_models('staging'))
    print("Production models:", registry.list_models('production'))
```

---

## Step 2: Create DAG Helper Functions

**Create**: `dags/training_pipeline_tasks.py`

```python
"""Task functions for training pipeline DAG."""
import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/user/mlops-learning-plan')

from ml.registry.model_registry import ModelRegistry


def validate_features(**context):
    """Validate that feature data exists and is recent.

    This task checks that:
    1. Feature directories exist
    2. Feature files are not empty
    3. Files are recent (optional)
    """
    config_path = context['params'].get('config_path', 'config/model_config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    features_path = Path(config['data']['features_path'])

    # Check train/val/test directories exist
    for split in ['train', 'val', 'test']:
        split_dir = features_path / split
        data_file = split_dir / 'data.parquet'

        if not data_file.exists():
            raise FileNotFoundError(f"Feature file not found: {data_file}")

        file_size = data_file.stat().st_size
        if file_size == 0:
            raise ValueError(f"Feature file is empty: {data_file}")

        print(f"✅ {split} data: {data_file} ({file_size / 1024:.1f} KB)")

    # Push features path to XCom
    context['ti'].xcom_push(key='features_path', value=str(features_path))

    print(f"\n✅ Feature validation complete: {features_path}")
    return str(features_path)


def train_model(**context):
    """Train the model by calling the training script.

    This task:
    1. Calls ml/training/train.py with config
    2. Captures the output directory
    3. Pushes run_dir to XCom for downstream tasks
    """
    config_path = context['params'].get('config_path', 'config/model_config.yaml')
    model_name = context['params'].get('model_name', 'ctr_model')

    # Create output directory for this run
    execution_date = context['execution_date'].strftime('%Y%m%d_%H%M%S')
    run_dir = f"experiments/runs/{model_name}_{execution_date}"

    # Call training script
    cmd = [
        'python',
        'ml/training/train.py',
        '--config', config_path,
        '--output-dir', run_dir
    ]

    print(f"🚀 Running training command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd='/home/user/mlops-learning-plan',
        capture_output=True,
        text=True
    )

    # Print output
    print("=== Training Output ===")
    print(result.stdout)

    if result.returncode != 0:
        print("=== Training Errors ===")
        print(result.stderr)
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # Verify run directory was created
    run_path = Path(run_dir)
    if not run_path.exists():
        raise RuntimeError(f"Training run directory not created: {run_dir}")

    # Push run_dir to XCom
    context['ti'].xcom_push(key='run_dir', value=run_dir)

    print(f"\n✅ Training complete: {run_dir}")
    return run_dir


def evaluate_model(**context):
    """Evaluate the trained model and extract metrics.

    This task:
    1. Loads metrics from training run
    2. Validates metrics meet minimum thresholds
    3. Pushes metrics to XCom
    """
    # Get run_dir from previous task
    run_dir = context['ti'].xcom_pull(key='run_dir', task_ids='train_model')

    if not run_dir:
        raise ValueError("run_dir not found in XCom")

    # Load metrics
    metrics_path = Path(run_dir) / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"📊 Model Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")

    # Validate minimum thresholds
    min_auc = context['params'].get('min_auc', 0.65)
    min_accuracy = context['params'].get('min_accuracy', 0.60)

    if metrics.get('test_auc', 0) < min_auc:
        raise ValueError(f"Model AUC {metrics['test_auc']:.4f} below threshold {min_auc}")

    if metrics.get('test_accuracy', 0) < min_accuracy:
        raise ValueError(f"Model accuracy {metrics['test_accuracy']:.4f} below threshold {min_accuracy}")

    # Push metrics to XCom
    context['ti'].xcom_push(key='metrics', value=metrics)

    print(f"\n✅ Evaluation complete - Model meets quality thresholds")
    return metrics


def register_model_to_staging(**context):
    """Register the trained model to staging.

    This task:
    1. Gets run_dir and metrics from previous tasks
    2. Registers model to staging with ModelRegistry
    3. Pushes model_version to XCom
    """
    # Get data from previous tasks
    run_dir = context['ti'].xcom_pull(key='run_dir', task_ids='train_model')
    metrics = context['ti'].xcom_pull(key='metrics', task_ids='evaluate_model')

    model_name = context['params'].get('model_name', 'ctr_model')

    # Load config
    config_path = Path(run_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize registry
    registry = ModelRegistry(base_dir='models')

    # Register model
    model_path = Path(run_dir) / 'model_best.pt'

    metadata = {
        'run_dir': run_dir,
        'execution_date': context['execution_date'].isoformat(),
        'dag_id': context['dag'].dag_id,
        'task_id': context['task'].task_id
    }

    registered_path = registry.register_model(
        model_name=model_name,
        model_path=str(model_path),
        config=config,
        metrics=metrics,
        stage='staging',
        metadata=metadata
    )

    # Get version from registered path
    version = Path(registered_path).name

    # Push to XCom
    context['ti'].xcom_push(key='model_version', value=version)
    context['ti'].xcom_push(key='registered_path', value=registered_path)

    print(f"\n✅ Model registered: {model_name} {version}")
    print(f"   Path: {registered_path}")

    return registered_path


def notify_completion(**context):
    """Send notification about training completion (placeholder).

    In production, this would:
    - Send Slack message
    - Email stakeholders
    - Update dashboard
    - Trigger downstream processes
    """
    model_name = context['params'].get('model_name', 'ctr_model')
    version = context['ti'].xcom_pull(key='model_version', task_ids='register_model')
    metrics = context['ti'].xcom_pull(key='metrics', task_ids='evaluate_model')
    registered_path = context['ti'].xcom_pull(key='registered_path', task_ids='register_model')

    message = f"""
    🎉 Training Pipeline Complete!

    Model: {model_name}
    Version: {version}
    Execution Date: {context['execution_date']}

    📊 Metrics:
       Test Accuracy: {metrics.get('test_accuracy', 0):.4f}
       Test AUC: {metrics.get('test_auc', 0):.4f}

    📁 Model Location:
       {registered_path}

    Next Steps:
    - Review model performance
    - Test in staging environment
    - Promote to production if satisfactory
    """

    print(message)

    # In production, send to Slack/email
    # slack_webhook(message)
    # send_email(to='ml-team@company.com', body=message)

    return "Notification sent"
```

---

## Step 3: Create the Training DAG

**Create**: `dags/training_pipeline.py`

```python
"""Training pipeline DAG - Orchestrates model training."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import task functions
from training_pipeline_tasks import (
    validate_features,
    train_model,
    evaluate_model,
    register_model_to_staging,
    notify_completion
)


# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Create DAG
with DAG(
    dag_id='training_pipeline',
    default_args=default_args,
    description='Train and register ML models',
    schedule_interval='@weekly',  # Train weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'pytorch'],
    params={
        'config_path': 'config/model_config.yaml',
        'model_name': 'ctr_model',
        'min_auc': 0.65,
        'min_accuracy': 0.60
    }
) as dag:

    # Task 1: Validate feature data exists
    validate_data = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        doc_md="""
        ### Validate Features

        Checks that feature data from Phase 2 pipeline exists and is valid:
        - Train/val/test splits present
        - Files are not empty
        - Schema is correct
        """
    )

    # Task 2: Train the model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        doc_md="""
        ### Train Model

        Executes the PyTorch training script:
        - Loads features
        - Trains neural network
        - Performs early stopping
        - Saves checkpoints
        """
    )

    # Task 3: Evaluate model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        doc_md="""
        ### Evaluate Model

        Evaluates trained model on test set:
        - Computes metrics (accuracy, AUC)
        - Validates against thresholds
        - Fails if model quality is insufficient
        """
    )

    # Task 4: Register to staging
    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model_to_staging,
        doc_md="""
        ### Register Model

        Registers model to staging registry:
        - Versions the model
        - Saves weights + config + metrics
        - Makes available for serving
        """
    )

    # Task 5: Notify completion
    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        doc_md="""
        ### Notify Completion

        Sends notifications about training completion:
        - Slack message
        - Email to ML team
        - Updates dashboard
        """
    )

    # Define dependencies
    validate_data >> train >> evaluate >> register >> notify


# Documentation
dag.doc_md = """
# Training Pipeline

Orchestrates end-to-end model training:

## Workflow
1. **Validate Features**: Ensure data from ETL pipeline is ready
2. **Train Model**: Execute PyTorch training script
3. **Evaluate Model**: Test on held-out set, validate quality
4. **Register Model**: Save to staging registry with version
5. **Notify**: Alert team about completion

## Configuration
Set parameters in DAG params:
- `config_path`: Path to model config YAML
- `model_name`: Name for model registry
- `min_auc`: Minimum AUC threshold (default: 0.65)
- `min_accuracy`: Minimum accuracy threshold (default: 0.60)

## Scheduling
- **Frequency**: Weekly (can be changed to daily/on-demand)
- **Trigger**: Can also be triggered manually or by feature pipeline completion

## Outputs
- Trained model in `experiments/runs/{model_name}_{date}/`
- Registered model in `models/staging/{model_name}/{version}/`
- Metrics logged to XCom and model registry

## Dependencies
- Requires features from Phase 2 feature engineering pipeline
- Python environment with PyTorch, pandas, sklearn

## Next Steps
After successful training:
1. Review metrics in Airflow UI or model registry
2. Test model in staging environment
3. Promote to production using `ModelRegistry.promote_to_production()`
"""
```

---

## Step 4: Set Up Airflow Environment

Ensure Airflow can find your project:

```bash
cd /home/user/mlops-learning-plan

# Set AIRFLOW_HOME
export AIRFLOW_HOME=$(pwd)/airflow

# Add project to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Copy DAGs to Airflow
cp dags/training_pipeline.py $AIRFLOW_HOME/dags/
cp dags/training_pipeline_tasks.py $AIRFLOW_HOME/dags/
```

Or create a symbolic link:

```bash
# Better approach: symlink dags directory
ln -s $(pwd)/dags $AIRFLOW_HOME/dags_local
```

Update `airflow.cfg`:
```ini
[core]
dags_folder = /home/user/mlops-learning-plan/airflow/dags,/home/user/mlops-learning-plan/dags
```

---

## Step 5: Test the DAG

### Test 1: DAG Validation

```bash
# Check DAG loads without errors
airflow dags list | grep training_pipeline

# Parse DAG
python dags/training_pipeline.py

# Check for import errors
airflow dags list-import-errors
```

### Test 2: Individual Task Testing

```bash
# Test validate_features task
airflow tasks test training_pipeline validate_features 2024-01-01

# Test train_model task (takes a few minutes)
airflow tasks test training_pipeline train_model 2024-01-01

# Test evaluate_model task
airflow tasks test training_pipeline evaluate_model 2024-01-01

# Test register_model task
airflow tasks test training_pipeline register_model 2024-01-01
```

### Test 3: Full DAG Run

In Airflow UI:
1. Navigate to http://localhost:8080
2. Find "training_pipeline" DAG
3. Toggle it ON
4. Click "Trigger DAG" (play button)
5. Monitor task progress

Or via CLI:
```bash
# Trigger a DAG run
airflow dags trigger training_pipeline

# Or with custom execution date
airflow dags trigger training_pipeline --exec-date 2024-11-14
```

---

## Step 6: Monitor and Verify

### Check Task Logs

In Airflow UI:
- Click on a task (colored box)
- Click "Log" button
- View execution logs

Look for:
- ✅ marks indicating success
- Training progress (epochs)
- Metrics (accuracy, AUC)
- Model paths

### Check XCom Data

In Airflow UI:
- Go to Admin → XComs
- Filter by `training_pipeline`
- See data passed between tasks:
  - `features_path`
  - `run_dir`
  - `metrics`
  - `model_version`

### Verify Model Registry

```bash
# Check staging models
ls -la models/staging/ctr_model/

# Should see:
# models/staging/ctr_model/
#   v1/
#     model.pt
#     config.yaml
#     metrics.json
#     metadata.json

# Read metrics
cat models/staging/ctr_model/v1/metrics.json
```

---

## Step 7: Connect to Feature Pipeline (Optional)

Make training depend on feature engineering completion:

**Update `dags/training_pipeline.py`**:

```python
from airflow.sensors.external_task import ExternalTaskSensor

with DAG(...) as dag:

    # Wait for feature engineering to complete
    wait_for_features = ExternalTaskSensor(
        task_id='wait_for_features',
        external_dag_id='feature_engineering_pipeline',  # From Phase 2
        external_task_id='save_features',
        allowed_states=['success'],
        timeout=600
    )

    validate_data = PythonOperator(...)

    # Update dependencies
    wait_for_features >> validate_data >> train >> ...
```

This makes training wait for fresh features!

---

## Step 8: Add Manual Trigger with Parameters

Allow triggering with custom config:

**Create**: `dags/trigger_training_manual.py`

```python
"""Manual training trigger with custom parameters."""
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime


with DAG(
    'trigger_training_manual',
    schedule_interval=None,  # Manual only
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='training_pipeline',
        conf={
            'config_path': 'config/model_config_experimental.yaml',
            'model_name': 'ctr_model_experiment',
            'min_auc': 0.70  # Higher threshold for experiments
        }
    )
```

Usage:
```bash
airflow dags trigger trigger_training_manual
```

---

## Step 9: Create a Training Report Script

**Create**: `scripts/training_report.py`

```python
"""Generate training report from DAG run."""
import json
import yaml
from pathlib import Path
from ml.registry.model_registry import ModelRegistry


def generate_training_report(model_name: str, version: str, stage: str = 'staging'):
    """Generate comprehensive training report.

    Args:
        model_name: Name of the model
        version: Model version (e.g., 'v1')
        stage: 'staging' or 'production'
    """
    registry = ModelRegistry()

    # Get model directory
    if stage == 'staging':
        model_dir = Path('models/staging') / model_name / version
    else:
        model_dir = Path('models/production') / model_name / version

    if not model_dir.exists():
        print(f"❌ Model not found: {model_dir}")
        return

    # Load artifacts
    with open(model_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open(model_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Generate report
    report = f"""
    ═══════════════════════════════════════════════════════════
                         TRAINING REPORT
    ═══════════════════════════════════════════════════════════

    Model: {model_name}
    Version: {version}
    Stage: {stage}
    Registered: {metadata.get('registered_at', 'N/A')}

    ───────────────────────────────────────────────────────────
                           CONFIGURATION
    ───────────────────────────────────────────────────────────

    Model Architecture:
      - Hidden Dims: {config['model']['hidden_dims']}
      - Dropout: {config['model']['dropout']}

    Training:
      - Batch Size: {config['training']['batch_size']}
      - Learning Rate: {config['training']['learning_rate']}
      - Epochs: {config['training']['epochs']}
      - Early Stopping: {config['training']['early_stopping_patience']}

    Data:
      - Features Path: {config['data']['features_path']}

    ───────────────────────────────────────────────────────────
                             METRICS
    ───────────────────────────────────────────────────────────

    Test Performance:
      - Accuracy: {metrics.get('test_accuracy', 0):.4f}
      - AUC: {metrics.get('test_auc', 0):.4f}
      - Loss: {metrics.get('test_loss', 0):.4f}

    Best Validation:
      - Val Loss: {metrics.get('best_val_loss', 0):.4f}
      - Val AUC: {metrics.get('best_val_auc', 0):.4f}

    ───────────────────────────────────────────────────────────
                           ARTIFACTS
    ───────────────────────────────────────────────────────────

    Location: {model_dir}
    Files:
      - model.pt (weights)
      - config.yaml (configuration)
      - metrics.json (evaluation metrics)
      - metadata.json (run information)

    ───────────────────────────────────────────────────────────
                        NEXT STEPS
    ───────────────────────────────────────────────────────────

    1. Review metrics above
    2. Test model: python scripts/test_model.py {metadata.get('run_dir', '')}
    3. If satisfactory, promote to production:
       from ml.registry.model_registry import ModelRegistry
       registry = ModelRegistry()
       registry.promote_to_production('{model_name}', '{version}')

    ═══════════════════════════════════════════════════════════
    """

    print(report)

    # Save to file
    report_path = model_dir / 'REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved to {report_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python scripts/training_report.py <model_name> <version> [stage]")
        print("Example: python scripts/training_report.py ctr_model v1 staging")
        sys.exit(1)

    model_name = sys.argv[1]
    version = sys.argv[2]
    stage = sys.argv[3] if len(sys.argv) > 3 else 'staging'

    generate_training_report(model_name, version, stage)
```

Usage:
```bash
python scripts/training_report.py ctr_model v1
```

---

## Deliverables Checklist

- [ ] **Model registry**: `ml/registry/model_registry.py`
- [ ] **DAG tasks**: `dags/training_pipeline_tasks.py`
- [ ] **Training DAG**: `dags/training_pipeline.py`
- [ ] **DAG visible** in Airflow UI
- [ ] **Successful DAG run** (all tasks green)
- [ ] **Model registered** in `models/staging/{model_name}/v1/`
- [ ] **XCom data** captured between tasks
- [ ] **Training report** generated

---

## Key Takeaways

1. **Thin orchestration layer**: DAG calls scripts, doesn't contain ML logic
2. **XCom for coordination**: Pass data between tasks (paths, metrics)
3. **Model registry**: Version and track all models
4. **Validation gates**: Fail fast if data or metrics don't meet thresholds
5. **Reproducibility**: Save config + weights + metrics together

---

## Next Steps

1. **Lab 3.3**: Add experiment tracking to compare multiple runs
2. **Lab 3.4**: Build two-tower ranking model
3. **Phase 4**: Deploy model for serving

---

## Troubleshooting

**Issue**: DAG not appearing in UI
- Check `airflow dags list-import-errors`
- Verify PYTHONPATH includes project root
- Check DAG file syntax

**Issue**: Task fails with import errors
- Ensure all `__init__.py` files exist
- Check virtual environment activated
- Verify dependencies installed

**Issue**: XCom data not available
- Check task_ids match in xcom_pull
- Ensure previous task completed successfully
- View XCom in Admin → XComs

**Issue**: Model registration fails
- Check directory permissions
- Verify run_dir exists from training
- Check model.pt file exists

---

**Excellent work! You've integrated PyTorch training with Airflow! 🎉**

Next: **Lab 3.3 - Experiment Tracking** to systematically compare training runs.
