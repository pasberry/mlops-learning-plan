"""
Automated Model Retraining DAG

Handles complete retraining workflow:
1. Check if retraining is needed (drift detected or scheduled)
2. Prepare training data
3. Train new model
4. Compare with baseline
5. Promote if better
6. Update monitoring baseline
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import shutil

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago


# Add ml module to path
ML_PATH = Path(__file__).parent.parent / 'ml'
sys.path.insert(0, str(ML_PATH))


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['mlops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_retraining_trigger(**context):
    """Check if retraining should be triggered."""
    import logging
    import json

    logger = logging.getLogger(__name__)

    # Check for drift report
    drift_report_path = context['params'].get('drift_report_path')

    retraining_needed = False
    reason = []

    # Check drift report
    if drift_report_path and Path(drift_report_path).exists():
        with open(drift_report_path) as f:
            drift_report = json.load(f)

        if drift_report.get('overall_drift_detected', False):
            retraining_needed = True
            reason.append("Data drift detected")

            feature_drift = drift_report.get('feature_drift', {})
            drifted_count = feature_drift.get('drifted_features', 0)
            reason.append(f"{drifted_count} features drifted")

    # Check forced retraining flag
    force_retrain = context['params'].get('force_retrain', False)
    if force_retrain:
        retraining_needed = True
        reason.append("Forced retraining requested")

    # Check last training date
    last_training_path = context['params'].get('last_training_timestamp_path')
    max_days_since_training = context['params'].get('max_days_since_training', 30)

    if last_training_path and Path(last_training_path).exists():
        with open(last_training_path) as f:
            last_training_time = datetime.fromisoformat(f.read().strip())

        days_since_training = (datetime.utcnow() - last_training_time).days

        if days_since_training >= max_days_since_training:
            retraining_needed = True
            reason.append(f"{days_since_training} days since last training (threshold: {max_days_since_training})")

    # Log decision
    logger.info("=" * 80)
    logger.info("RETRAINING TRIGGER CHECK")
    logger.info("=" * 80)
    logger.info(f"Retraining needed: {retraining_needed}")
    if reason:
        logger.info(f"Reasons:")
        for r in reason:
            logger.info(f"  - {r}")
    logger.info("=" * 80)

    # Push decision to XCom
    context['ti'].xcom_push(key='retraining_needed', value=retraining_needed)
    context['ti'].xcom_push(key='retraining_reason', value='; '.join(reason))

    return 'prepare_training_data' if retraining_needed else 'skip_retraining'


def prepare_training_data(**context):
    """Prepare data for retraining."""
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.model_selection import train_test_split

    logger = logging.getLogger(__name__)

    # Get configuration
    input_path = context['params']['training_data_path']
    output_dir = context['params']['prepared_data_dir']

    logger.info(f"Preparing training data from {input_path}")

    # Load data
    df = pd.read_csv(input_path)

    # Basic validation
    if len(df) < 1000:
        raise ValueError(f"Insufficient training data: {len(df)} samples (minimum: 1000)")

    # Separate features and target
    target_column = context['params'].get('target_column', 'target')
    feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save splits
    train_df = pd.DataFrame(
        np.column_stack([X_train, y_train]),
        columns=feature_columns + [target_column]
    )
    val_df = pd.DataFrame(
        np.column_stack([X_val, y_val]),
        columns=feature_columns + [target_column]
    )

    train_path = Path(output_dir) / 'train.csv'
    val_path = Path(output_dir) / 'val.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Training data prepared:")
    logger.info(f"  Train: {len(train_df)} samples → {train_path}")
    logger.info(f"  Val:   {len(val_df)} samples → {val_path}")

    # Push paths to XCom
    context['ti'].xcom_push(key='train_path', value=str(train_path))
    context['ti'].xcom_push(key='val_path', value=str(val_path))


def train_new_model(**context):
    """Train new model."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # Get data paths
    train_path = context['ti'].xcom_pull(
        task_ids='prepare_training_data',
        key='train_path'
    )
    val_path = context['ti'].xcom_pull(
        task_ids='prepare_training_data',
        key='val_path'
    )

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.float32)
    X_val = val_df.iloc[:, :-1].values.astype(np.float32)
    y_val = val_df.iloc[:, -1].values.astype(np.float32)

    # Create model
    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # Training configuration
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = context['params'].get('num_epochs', 50)
    batch_size = context['params'].get('batch_size', 32)

    logger.info(f"Training model for {num_epochs} epochs")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val).unsqueeze(1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # Save model
    output_dir = context['params']['models_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    model_path = Path(output_dir) / f'model_{timestamp}.pt'

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'version': timestamp,
        'feature_names': list(train_df.columns[:-1]),
        'model_type': 'neural_network',
        'metadata': {
            'trained_at': datetime.utcnow().isoformat(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_val_loss': best_val_loss,
            'num_epochs': num_epochs
        }
    }

    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")

    # Push model path to XCom
    context['ti'].xcom_push(key='new_model_path', value=str(model_path))


def compare_models(**context):
    """Compare new model with baseline."""
    import logging
    from training.model_comparator import compare_model_checkpoints

    logger = logging.getLogger(__name__)

    # Get paths
    baseline_path = context['params']['baseline_model_path']
    new_model_path = context['ti'].xcom_pull(
        task_ids='train_model',
        key='new_model_path'
    )
    test_data_path = context['params']['test_data_path']

    # Compare models
    logger.info("Comparing new model with baseline")
    results = compare_model_checkpoints(
        baseline_path=baseline_path,
        new_path=new_model_path,
        test_data_path=test_data_path,
        primary_metric=context['params'].get('primary_metric', 'auc'),
        improvement_threshold=context['params'].get('improvement_threshold', 0.02)
    )

    # Save comparison report
    report_dir = context['params'].get('reports_dir', '/tmp/reports')
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    report_path = Path(report_dir) / 'model_comparison.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Comparison report saved to {report_path}")

    # Push results to XCom
    context['ti'].xcom_push(key='should_promote', value=results['should_promote'])
    context['ti'].xcom_push(key='comparison_results', value=results)

    return 'promote_model' if results['should_promote'] else 'skip_promotion'


def promote_new_model(**context):
    """Promote new model to production."""
    import logging

    logger = logging.getLogger(__name__)

    # Get new model path
    new_model_path = context['ti'].xcom_pull(
        task_ids='train_model',
        key='new_model_path'
    )

    # Production model path
    prod_model_path = context['params']['production_model_path']

    # Backup current production model
    if Path(prod_model_path).exists():
        backup_path = str(prod_model_path).replace('.pt', '_backup.pt')
        shutil.copy2(prod_model_path, backup_path)
        logger.info(f"Backed up current production model to {backup_path}")

    # Copy new model to production
    Path(prod_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(new_model_path, prod_model_path)

    logger.info(f"Promoted new model to production: {prod_model_path}")

    # Update last training timestamp
    timestamp_path = context['params'].get('last_training_timestamp_path')
    if timestamp_path:
        Path(timestamp_path).parent.mkdir(parents=True, exist_ok=True)
        with open(timestamp_path, 'w') as f:
            f.write(datetime.utcnow().isoformat())

    logger.info("=" * 80)
    logger.info("MODEL PROMOTION SUCCESSFUL")
    logger.info("=" * 80)


# Create DAG
dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval='0 4 * * 0',  # Weekly on Sunday at 4 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['retraining', 'training', 'production'],
    params={
        'drift_report_path': '/home/user/mlops-learning-plan/data/monitoring/reports/latest_drift_report.json',
        'training_data_path': '/home/user/mlops-learning-plan/data/processed/training_data.csv',
        'test_data_path': '/home/user/mlops-learning-plan/data/processed/test_data.csv',
        'prepared_data_dir': '/home/user/mlops-learning-plan/data/training',
        'models_dir': '/home/user/mlops-learning-plan/models/candidates',
        'baseline_model_path': '/home/user/mlops-learning-plan/models/production/model.pt',
        'production_model_path': '/home/user/mlops-learning-plan/models/production/model.pt',
        'last_training_timestamp_path': '/home/user/mlops-learning-plan/models/last_training.txt',
        'reports_dir': '/home/user/mlops-learning-plan/data/reports',
        'force_retrain': False,
        'max_days_since_training': 30,
        'target_column': 'target',
        'num_epochs': 50,
        'batch_size': 32,
        'primary_metric': 'auc',
        'improvement_threshold': 0.02
    }
)

with dag:
    # Task 1: Check if retraining is needed
    check_trigger = BranchPythonOperator(
        task_id='check_retraining_trigger',
        python_callable=check_retraining_trigger,
        provide_context=True
    )

    # Task 2a: Prepare training data
    prepare_data = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data,
        provide_context=True
    )

    # Task 2b: Skip retraining
    skip_retrain = BashOperator(
        task_id='skip_retraining',
        bash_command='echo "Retraining not needed - skipping"'
    )

    # Task 3: Train new model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_new_model,
        provide_context=True
    )

    # Task 4: Compare models
    compare = BranchPythonOperator(
        task_id='compare_models',
        python_callable=compare_models,
        provide_context=True
    )

    # Task 5a: Promote model
    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_new_model,
        provide_context=True
    )

    # Task 5b: Skip promotion
    skip_promote = BashOperator(
        task_id='skip_promotion',
        bash_command='echo "New model did not meet promotion criteria - keeping baseline"'
    )

    # Define dependencies
    check_trigger >> [prepare_data, skip_retrain]
    prepare_data >> train >> compare
    compare >> [promote, skip_promote]


if __name__ == "__main__":
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=str(Path(__file__).parent), include_examples=False)

    if dag.dag_id in dag_bag.dags:
        print(f"✓ DAG '{dag.dag_id}' loaded successfully")
        print(f"  Tasks: {len(dag.tasks)}")
        print(f"  Schedule: {dag.schedule_interval}")
    else:
        print(f"✗ DAG '{dag.dag_id}' failed to load")
