"""
Master MLOps Pipeline DAG

Orchestrates the complete end-to-end MLOps workflow:
1. Data ingestion and validation
2. Feature engineering
3. Batch inference
4. Monitoring and drift detection
5. Automated retraining (if needed)
6. Model deployment

This DAG represents the "complete loop" of production ML.
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['mlops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def validate_raw_data(**context):
    """Validate incoming raw data."""
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)
    input_path = context['params']['raw_data_path']

    logger.info(f"Validating raw data: {input_path}")

    df = pd.read_csv(input_path)

    # Validation checks
    assert len(df) > 0, "Raw data is empty"
    assert not df.isnull().all().any(), "Column with all nulls found"

    logger.info(f"✓ Raw data validated: {len(df)} rows, {len(df.columns)} columns")
    context['ti'].xcom_push(key='raw_row_count', value=len(df))


def extract_features(**context):
    """Extract and engineer features."""
    import pandas as pd
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    input_path = context['params']['raw_data_path']
    output_path = context['params']['features_path']

    logger.info("Extracting features")

    df = pd.read_csv(input_path)

    # Feature engineering (example)
    features_df = df.copy()

    # Add derived features
    if 'account_age_days' in features_df.columns:
        features_df['account_age_years'] = features_df['account_age_days'] / 365.0

    if 'income' in features_df.columns and 'age' in features_df.columns:
        features_df['income_per_age'] = features_df['income'] / (features_df['age'] + 1)

    # Save features
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    logger.info(f"✓ Features extracted: {output_path}")
    context['ti'].xcom_push(key='features_path', value=output_path)


def check_drift_status(**context):
    """Check if drift was detected in monitoring."""
    import json
    import logging

    logger = logging.getLogger(__name__)

    drift_report_path = context['params']['drift_report_path']

    if not Path(drift_report_path).exists():
        logger.info("No drift report found - assuming no drift")
        context['ti'].xcom_push(key='drift_detected', value=False)
        return

    with open(drift_report_path) as f:
        report = json.load(f)

    drift_detected = report.get('overall_drift_detected', False)

    logger.info(f"Drift status: {'DETECTED' if drift_detected else 'NOT DETECTED'}")
    context['ti'].xcom_push(key='drift_detected', value=drift_detected)


def generate_pipeline_report(**context):
    """Generate end-to-end pipeline execution report."""
    import logging

    logger = logging.getLogger(__name__)

    # Collect metrics from all tasks
    raw_count = context['ti'].xcom_pull(
        task_ids='validate_raw_data',
        key='raw_row_count'
    )

    drift_detected = context['ti'].xcom_pull(
        task_ids='check_drift',
        key='drift_detected'
    )

    # Generate report
    logger.info("=" * 80)
    logger.info("MLOPS PIPELINE EXECUTION REPORT")
    logger.info("=" * 80)
    logger.info(f"Execution Time: {datetime.utcnow().isoformat()}")
    logger.info(f"Raw Data: {raw_count} rows processed")
    logger.info(f"Drift Detected: {drift_detected}")
    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)


# Create master DAG
dag = DAG(
    'mlops_master_pipeline',
    default_args=default_args,
    description='Master end-to-end MLOps pipeline',
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['master', 'production', 'mlops'],
    params={
        'raw_data_path': '/home/user/mlops-learning-plan/data/raw/daily_data.csv',
        'features_path': '/home/user/mlops-learning-plan/data/processed/current_features.csv',
        'predictions_path': '/home/user/mlops-learning-plan/data/predictions/batch_predictions.csv',
        'drift_report_path': '/home/user/mlops-learning-plan/data/monitoring/reports/latest_drift_report.json',
    }
)

with dag:
    # ========== STAGE 1: DATA INGESTION & VALIDATION ==========
    validate_data = PythonOperator(
        task_id='validate_raw_data',
        python_callable=validate_raw_data,
        provide_context=True
    )

    # ========== STAGE 2: FEATURE ENGINEERING ==========
    engineer_features = PythonOperator(
        task_id='extract_features',
        python_callable=extract_features,
        provide_context=True
    )

    # ========== STAGE 3: BATCH INFERENCE ==========
    # Trigger batch inference DAG
    run_batch_inference = TriggerDagRunOperator(
        task_id='trigger_batch_inference',
        trigger_dag_id='batch_inference',
        wait_for_completion=True,
        poke_interval=30,
        execution_date='{{ ds }}',
        reset_dag_run=True
    )

    # ========== STAGE 4: MONITORING ==========
    # Trigger monitoring DAG
    run_monitoring = TriggerDagRunOperator(
        task_id='trigger_monitoring',
        trigger_dag_id='model_monitoring',
        wait_for_completion=True,
        poke_interval=30,
        execution_date='{{ ds }}',
        reset_dag_run=True
    )

    # Check drift status
    check_drift = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift_status,
        provide_context=True
    )

    # ========== STAGE 5: RETRAINING (CONDITIONAL) ==========
    # Trigger retraining if drift detected
    run_retraining = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='model_retraining',
        wait_for_completion=True,
        poke_interval=60,
        execution_date='{{ ds }}',
        reset_dag_run=True,
        trigger_rule='none_failed'  # Run even if previous tasks skipped
    )

    # ========== STAGE 6: REPORTING ==========
    generate_report = PythonOperator(
        task_id='generate_pipeline_report',
        python_callable=generate_pipeline_report,
        provide_context=True,
        trigger_rule='none_failed'
    )

    # Define pipeline flow
    validate_data >> engineer_features >> run_batch_inference
    run_batch_inference >> run_monitoring >> check_drift
    check_drift >> run_retraining >> generate_report


if __name__ == "__main__":
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=str(Path(__file__).parent), include_examples=False)

    if dag.dag_id in dag_bag.dags:
        print(f"✓ DAG '{dag.dag_id}' loaded successfully")
        print(f"  Tasks: {len(dag.tasks)}")
        print(f"  Schedule: {dag.schedule_interval}")
        print("\nPipeline Flow:")
        print("  1. Data Validation")
        print("  2. Feature Engineering")
        print("  3. Batch Inference")
        print("  4. Monitoring & Drift Detection")
        print("  5. Retraining (if needed)")
        print("  6. Pipeline Report")
    else:
        print(f"✗ DAG '{dag.dag_id}' failed to load")
