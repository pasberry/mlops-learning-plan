"""
Batch Inference DAG

Orchestrates batch scoring pipeline:
1. Load and validate input data
2. Run batch predictions
3. Save results
4. Generate reports
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# Add ml module to path
ML_PATH = Path(__file__).parent.parent / 'ml'
sys.path.insert(0, str(ML_PATH))


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def validate_input_data(**context):
    """Validate input data for batch inference."""
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    # Get configuration from context
    input_path = context['params']['input_path']

    logger.info(f"Validating input data: {input_path}")

    # Load data
    df = pd.read_csv(input_path)

    # Validation checks
    errors = []

    if len(df) == 0:
        errors.append("Input data is empty")

    # Check for required columns (example)
    required_columns = ['age', 'income', 'credit_score']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")

    # Check for invalid values
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 0) | (df['age'] > 120)).sum()
        if invalid_age > 0:
            errors.append(f"Invalid age values: {invalid_age}")

    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    # Log statistics
    logger.info(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Push metadata to XCom
    context['ti'].xcom_push(key='row_count', value=len(df))
    context['ti'].xcom_push(key='column_count', value=len(df.columns))


def run_batch_predictions(**context):
    """Run batch predictions."""
    import logging
    from inference.batch_engine import run_batch_inference

    logger = logging.getLogger(__name__)

    # Get configuration
    input_path = context['params']['input_path']
    output_path = context['params']['output_path']
    model_path = context['params']['model_path']
    batch_size = context['params'].get('batch_size', 256)
    id_column = context['params'].get('id_column')

    logger.info(f"Running batch inference")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {model_path}")

    # Run inference
    run_batch_inference(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        batch_size=batch_size,
        id_column=id_column,
        include_features=True
    )

    logger.info("Batch inference complete")

    # Push output path to XCom
    context['ti'].xcom_push(key='output_path', value=output_path)


def generate_predictions_report(**context):
    """Generate report on predictions."""
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    # Get output path from XCom
    output_path = context['ti'].xcom_pull(
        task_ids='run_predictions',
        key='output_path'
    )

    logger.info(f"Generating report for {output_path}")

    # Load predictions
    df = pd.read_csv(output_path)

    # Generate statistics
    report = {
        'total_predictions': len(df),
        'positive_predictions': (df['prediction_class'] == 1).sum(),
        'negative_predictions': (df['prediction_class'] == 0).sum(),
        'avg_score': df['prediction_score'].mean(),
        'min_score': df['prediction_score'].min(),
        'max_score': df['prediction_score'].max(),
        'std_score': df['prediction_score'].std(),
    }

    # Calculate score distribution
    score_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df['score_bin'] = pd.cut(df['prediction_score'], bins=score_bins)
    distribution = df['score_bin'].value_counts().sort_index()

    report['score_distribution'] = distribution.to_dict()

    # Log report
    logger.info("=" * 60)
    logger.info("BATCH INFERENCE REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Predictions: {report['total_predictions']:,}")
    logger.info(f"Positive: {report['positive_predictions']:,} ({report['positive_predictions']/report['total_predictions']*100:.1f}%)")
    logger.info(f"Negative: {report['negative_predictions']:,} ({report['negative_predictions']/report['total_predictions']*100:.1f}%)")
    logger.info(f"\nScore Statistics:")
    logger.info(f"  Mean: {report['avg_score']:.4f}")
    logger.info(f"  Std:  {report['std_score']:.4f}")
    logger.info(f"  Min:  {report['min_score']:.4f}")
    logger.info(f"  Max:  {report['max_score']:.4f}")
    logger.info("\nScore Distribution:")
    for bin_range, count in distribution.items():
        logger.info(f"  {bin_range}: {count:,} ({count/len(df)*100:.1f}%)")
    logger.info("=" * 60)

    # Push report to XCom
    context['ti'].xcom_push(key='report', value=report)

    return report


def archive_predictions(**context):
    """Archive predictions to long-term storage."""
    import shutil
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # Get output path
    output_path = context['ti'].xcom_pull(
        task_ids='run_predictions',
        key='output_path'
    )

    # Get archive configuration
    archive_dir = context['params'].get('archive_dir', '/tmp/predictions_archive')

    # Create archive directory
    Path(archive_dir).mkdir(parents=True, exist_ok=True)

    # Generate archive filename with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    archive_path = Path(archive_dir) / f"predictions_{timestamp}.csv"

    # Copy predictions to archive
    shutil.copy2(output_path, archive_path)

    logger.info(f"Predictions archived to {archive_path}")

    # Push archive path to XCom
    context['ti'].xcom_push(key='archive_path', value=str(archive_path))


# Create DAG
dag = DAG(
    'batch_inference',
    default_args=default_args,
    description='Batch inference pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['inference', 'batch', 'production'],
    params={
        'input_path': '/home/user/mlops-learning-plan/data/processed/batch_input.csv',
        'output_path': '/home/user/mlops-learning-plan/data/predictions/batch_predictions.csv',
        'model_path': '/home/user/mlops-learning-plan/models/production/model.pt',
        'batch_size': 256,
        'id_column': 'user_id',
        'archive_dir': '/home/user/mlops-learning-plan/data/predictions/archive'
    }
)

with dag:
    # Task 1: Validate input data
    validate_input = PythonOperator(
        task_id='validate_input',
        python_callable=validate_input_data,
        provide_context=True
    )

    # Task 2: Run batch predictions
    run_predictions = PythonOperator(
        task_id='run_predictions',
        python_callable=run_batch_predictions,
        provide_context=True
    )

    # Task 3: Generate report
    generate_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_predictions_report,
        provide_context=True
    )

    # Task 4: Archive predictions
    archive = PythonOperator(
        task_id='archive_predictions',
        python_callable=archive_predictions,
        provide_context=True
    )

    # Task 5: Cleanup temporary files (optional)
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='echo "Cleanup complete"',
    )

    # Define task dependencies
    validate_input >> run_predictions >> generate_report >> archive >> cleanup


if __name__ == "__main__":
    # Test the DAG
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=str(Path(__file__).parent), include_examples=False)

    if dag.dag_id in dag_bag.dags:
        print(f"✓ DAG '{dag.dag_id}' loaded successfully")
        print(f"  Tasks: {len(dag.tasks)}")
        print(f"  Schedule: {dag.schedule_interval}")

        # Print task list
        print("\nTasks:")
        for task in dag.tasks:
            print(f"  - {task.task_id}")
    else:
        print(f"✗ DAG '{dag.dag_id}' failed to load")
        if dag_bag.import_errors:
            print("Errors:")
            for error in dag_bag.import_errors.values():
                print(f"  {error}")
