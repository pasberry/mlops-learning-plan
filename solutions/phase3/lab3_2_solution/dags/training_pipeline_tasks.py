"""Task functions for training pipeline DAG."""
import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/user/mlops-learning-plan/solutions/phase3/lab3_2_solution')

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

    # Call training script (assumes Lab 3.1 solution is available)
    # In production, this would reference the actual training code location
    cmd = [
        'python',
        '../lab3_1_solution/ml/training/train.py',
        '--config', config_path,
        '--output-dir', run_dir
    ]

    print(f"🚀 Running training command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd='/home/user/mlops-learning-plan/solutions/phase3/lab3_2_solution',
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
