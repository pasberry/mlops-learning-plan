"""
Model Monitoring DAG

Monitors model performance and detects drift:
1. Load baseline and current data
2. Detect feature drift
3. Detect prediction drift
4. Generate alerts if drift detected
5. Log metrics
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago


# Add monitoring module to path
MONITORING_PATH = Path(__file__).parent.parent / 'monitoring'
sys.path.insert(0, str(MONITORING_PATH))


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['mlops-alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def load_and_prepare_data(**context):
    """Load baseline and current data."""
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    # Get paths from params
    baseline_path = context['params']['baseline_data_path']
    current_path = context['params']['current_data_path']

    logger.info(f"Loading baseline data from {baseline_path}")
    baseline_df = pd.read_csv(baseline_path)

    logger.info(f"Loading current data from {current_path}")
    current_df = pd.read_csv(current_path)

    logger.info(f"Baseline: {len(baseline_df)} rows")
    logger.info(f"Current: {len(current_df)} rows")

    # Basic validation
    if len(baseline_df) == 0:
        raise ValueError("Baseline data is empty")
    if len(current_df) == 0:
        raise ValueError("Current data is empty")

    # Push metadata to XCom
    context['ti'].xcom_push(key='baseline_size', value=len(baseline_df))
    context['ti'].xcom_push(key='current_size', value=len(current_df))


def detect_feature_drift(**context):
    """Detect drift in input features."""
    import pandas as pd
    import logging
    from drift_detector import DriftDetector

    logger = logging.getLogger(__name__)

    # Get paths
    baseline_path = context['params']['baseline_data_path']
    current_path = context['params']['current_data_path']
    feature_columns = context['params']['feature_columns']

    # Get thresholds
    psi_threshold = context['params'].get('psi_threshold', 0.2)
    ks_threshold = context['params'].get('ks_threshold', 0.05)

    # Load data
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    # Create detector
    detector = DriftDetector(
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold
    )

    # Detect drift
    logger.info("Detecting feature drift...")
    drift_results = detector.detect_feature_drift(
        baseline_df,
        current_df,
        feature_columns
    )

    # Log results
    logger.info(f"Feature drift detection complete")
    logger.info(f"Drifted features: {drift_results['drifted_features']}/{drift_results['total_features']}")

    # Push results to XCom
    context['ti'].xcom_push(key='feature_drift_results', value=drift_results)
    context['ti'].xcom_push(key='feature_drift_detected', value=drift_results['overall_drift_detected'])

    return drift_results


def detect_prediction_drift(**context):
    """Detect drift in model predictions."""
    import pandas as pd
    import numpy as np
    import logging
    from drift_detector import DriftDetector

    logger = logging.getLogger(__name__)

    # Get paths
    baseline_pred_path = context['params'].get('baseline_predictions_path')
    current_pred_path = context['params'].get('current_predictions_path')

    if not baseline_pred_path or not current_pred_path:
        logger.info("Prediction paths not provided, skipping prediction drift detection")
        return

    # Load predictions
    baseline_df = pd.read_csv(baseline_pred_path)
    current_df = pd.read_csv(current_pred_path)

    # Extract prediction scores
    pred_column = context['params'].get('prediction_column', 'prediction_score')
    baseline_predictions = baseline_df[pred_column].values
    current_predictions = current_df[pred_column].values

    # Create detector
    detector = DriftDetector(
        psi_threshold=context['params'].get('psi_threshold', 0.2),
        ks_threshold=context['params'].get('ks_threshold', 0.05)
    )

    # Detect drift
    logger.info("Detecting prediction drift...")
    drift_results = detector.detect_prediction_drift(
        baseline_predictions,
        current_predictions
    )

    # Log results
    logger.info(f"Prediction drift detection complete")
    logger.info(f"Drift detected: {drift_results['drift_detected']}")

    # Push results to XCom
    context['ti'].xcom_push(key='prediction_drift_results', value=drift_results)
    context['ti'].xcom_push(key='prediction_drift_detected', value=drift_results['drift_detected'])

    return drift_results


def generate_monitoring_report(**context):
    """Generate comprehensive monitoring report."""
    import logging
    import json
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # Get drift results from XCom
    feature_drift = context['ti'].xcom_pull(
        task_ids='detect_feature_drift',
        key='feature_drift_results'
    )
    prediction_drift = context['ti'].xcom_pull(
        task_ids='detect_prediction_drift',
        key='prediction_drift_results'
    )

    # Compile report
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'feature_drift': feature_drift,
        'prediction_drift': prediction_drift,
        'overall_drift_detected': (
            (feature_drift and feature_drift.get('overall_drift_detected', False)) or
            (prediction_drift and prediction_drift.get('drift_detected', False))
        ),
        'baseline_size': context['ti'].xcom_pull(
            task_ids='load_data',
            key='baseline_size'
        ),
        'current_size': context['ti'].xcom_pull(
            task_ids='load_data',
            key='current_size'
        )
    }

    # Save report
    output_dir = context['params'].get('reports_dir', '/tmp/monitoring_reports')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = Path(output_dir) / f"drift_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Monitoring report saved to {report_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("MONITORING REPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall Drift: {'DETECTED' if report['overall_drift_detected'] else 'NOT DETECTED'}")

    if feature_drift:
        logger.info(f"\nFeature Drift:")
        logger.info(f"  Total features: {feature_drift['total_features']}")
        logger.info(f"  Drifted features: {feature_drift['drifted_features']}")
        logger.info(f"  Drift percentage: {feature_drift['drift_percentage']:.1f}%")

    if prediction_drift:
        logger.info(f"\nPrediction Drift:")
        logger.info(f"  Detected: {prediction_drift['drift_detected']}")
        logger.info(f"  PSI: {prediction_drift['psi']['psi']:.4f}")
        logger.info(f"  Positive rate change: {prediction_drift['positive_rate_change']:.4f}")

    logger.info("=" * 80)

    # Push report path to XCom
    context['ti'].xcom_push(key='report_path', value=str(report_path))
    context['ti'].xcom_push(key='overall_drift_detected', value=report['overall_drift_detected'])

    return report


def check_drift_status(**context):
    """Check if drift was detected and decide next task."""
    drift_detected = context['ti'].xcom_pull(
        task_ids='generate_report',
        key='overall_drift_detected'
    )

    if drift_detected:
        return 'send_drift_alert'
    else:
        return 'log_success'


def send_drift_alert_func(**context):
    """Send alert when drift is detected."""
    import logging

    logger = logging.getLogger(__name__)

    # Get report
    report_path = context['ti'].xcom_pull(
        task_ids='generate_report',
        key='report_path'
    )

    feature_drift = context['ti'].xcom_pull(
        task_ids='detect_feature_drift',
        key='feature_drift_results'
    )

    # Construct alert message
    alert_message = [
        "DRIFT DETECTED IN PRODUCTION MODEL",
        "",
        f"Report: {report_path}",
        "",
    ]

    if feature_drift:
        alert_message.extend([
            f"Feature Drift:",
            f"  - {feature_drift['drifted_features']} out of {feature_drift['total_features']} features drifted",
            f"  - Drift percentage: {feature_drift['drift_percentage']:.1f}%",
            "",
            "Drifted features:",
        ])

        for feature, metrics in feature_drift['features'].items():
            if metrics['drift_detected']:
                alert_message.append(f"  - {feature} (PSI: {metrics['psi']['psi']:.4f})")

    alert_message.extend([
        "",
        "Action required: Review model performance and consider retraining.",
        "",
        f"Timestamp: {datetime.utcnow().isoformat()}"
    ])

    message = "\n".join(alert_message)

    logger.warning("\n" + "=" * 80)
    logger.warning("DRIFT ALERT")
    logger.warning("=" * 80)
    logger.warning(message)
    logger.warning("=" * 80)

    # In production, send to alerting system (Slack, PagerDuty, etc.)
    # For now, just log

    return message


# Create DAG
dag = DAG(
    'model_monitoring',
    default_args=default_args,
    description='Monitor model for drift and performance degradation',
    schedule_interval='0 3 * * *',  # Daily at 3 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'production'],
    params={
        'baseline_data_path': '/home/user/mlops-learning-plan/data/baseline/features.csv',
        'current_data_path': '/home/user/mlops-learning-plan/data/processed/current_features.csv',
        'baseline_predictions_path': '/home/user/mlops-learning-plan/data/baseline/predictions.csv',
        'current_predictions_path': '/home/user/mlops-learning-plan/data/predictions/batch_predictions.csv',
        'feature_columns': [
            'age', 'income', 'credit_score', 'num_purchases',
            'account_age_days', 'avg_transaction', 'num_returns',
            'is_premium', 'region', 'category_preference'
        ],
        'prediction_column': 'prediction_score',
        'psi_threshold': 0.2,
        'ks_threshold': 0.05,
        'reports_dir': '/home/user/mlops-learning-plan/data/monitoring/reports'
    }
)

with dag:
    # Task 1: Load and validate data
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_and_prepare_data,
        provide_context=True
    )

    # Task 2: Detect feature drift
    detect_features = PythonOperator(
        task_id='detect_feature_drift',
        python_callable=detect_feature_drift,
        provide_context=True
    )

    # Task 3: Detect prediction drift
    detect_predictions = PythonOperator(
        task_id='detect_prediction_drift',
        python_callable=detect_prediction_drift,
        provide_context=True
    )

    # Task 4: Generate report
    generate_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_monitoring_report,
        provide_context=True
    )

    # Task 5: Check drift status and branch
    check_drift = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_drift_status,
        provide_context=True
    )

    # Task 6a: Send alert if drift detected
    send_alert = PythonOperator(
        task_id='send_drift_alert',
        python_callable=send_drift_alert_func,
        provide_context=True
    )

    # Task 6b: Log success if no drift
    log_ok = BashOperator(
        task_id='log_success',
        bash_command='echo "No drift detected - model is healthy"'
    )

    # Define dependencies
    load_data >> [detect_features, detect_predictions]
    [detect_features, detect_predictions] >> generate_report
    generate_report >> check_drift
    check_drift >> [send_alert, log_ok]


if __name__ == "__main__":
    # Test the DAG
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=str(Path(__file__).parent), include_examples=False)

    if dag.dag_id in dag_bag.dags:
        print(f"✓ DAG '{dag.dag_id}' loaded successfully")
        print(f"  Tasks: {len(dag.tasks)}")
        print(f"  Schedule: {dag.schedule_interval}")

        # Print task list
        print("\nTask Dependencies:")
        for task in dag.tasks:
            downstream = [t.task_id for t in task.downstream_list]
            if downstream:
                print(f"  {task.task_id} → {', '.join(downstream)}")
            else:
                print(f"  {task.task_id}")
    else:
        print(f"✗ DAG '{dag.dag_id}' failed to load")
        if dag_bag.import_errors:
            print("Errors:")
            for error in dag_bag.import_errors.values():
                print(f"  {error}")
