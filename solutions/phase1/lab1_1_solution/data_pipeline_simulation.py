"""
Data Pipeline Simulation - CHALLENGE SOLUTION
Simulates a complete data pipeline with XCom data passing between tasks.

Pipeline: extract_data -> validate_data -> transform_data -> load_data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}


def extract_data(**context):
    """
    Task 1: Extract data from source.

    Simulates extracting data from a database or API.
    Returns a list of numbers that will be passed to the next task via XCom.
    """
    logger.info("Starting data extraction...")
    print("=" * 60)
    print("EXTRACT: Extracting data from source...")
    print("=" * 60)

    # Simulate extracting data (in reality, this might be from a database, API, etc.)
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print(f"Extracted {len(data)} records: {data}")
    logger.info(f"Successfully extracted {len(data)} records")

    # Return data to be stored in XCom
    return data


def validate_data(**context):
    """
    Task 2: Validate the extracted data.

    Pulls data from XCom, validates it, and passes it forward.
    Raises an error if data is invalid.
    """
    logger.info("Starting data validation...")
    print("=" * 60)
    print("VALIDATE: Validating extracted data...")
    print("=" * 60)

    # Pull data from the previous task using XCom
    ti = context['ti']
    data = ti.xcom_pull(task_ids='extract_data')

    print(f"Received data: {data}")

    # Validation checks
    if data is None:
        error_msg = "Validation failed: Data is None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(data, list):
        error_msg = f"Validation failed: Data is not a list, got {type(data)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if len(data) == 0:
        error_msg = "Validation failed: Data is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check for negative values
    if any(x < 0 for x in data):
        error_msg = "Validation failed: Data contains negative values"
        logger.error(error_msg)
        raise ValueError(error_msg)

    print(f"✓ Validation passed: {len(data)} records are valid")
    print(f"✓ All values are non-negative")
    print(f"✓ Data type is correct (list)")
    logger.info("Data validation successful")

    # Pass data forward
    return data


def transform_data(**context):
    """
    Task 3: Transform the validated data.

    Applies transformations to the data (e.g., multiply each number by 2).
    """
    logger.info("Starting data transformation...")
    print("=" * 60)
    print("TRANSFORM: Transforming validated data...")
    print("=" * 60)

    # Pull validated data from XCom
    ti = context['ti']
    data = ti.xcom_pull(task_ids='validate_data')

    print(f"Original data: {data}")

    # Apply transformation: multiply each value by 2
    transformed_data = [x * 2 for x in data]

    print(f"Transformed data: {transformed_data}")
    print(f"Transformation: Each value multiplied by 2")
    logger.info(f"Successfully transformed {len(transformed_data)} records")

    # Calculate some statistics
    print(f"\nStatistics:")
    print(f"  - Record count: {len(transformed_data)}")
    print(f"  - Sum: {sum(transformed_data)}")
    print(f"  - Average: {sum(transformed_data) / len(transformed_data):.2f}")
    print(f"  - Min: {min(transformed_data)}")
    print(f"  - Max: {max(transformed_data)}")

    return transformed_data


def load_data(**context):
    """
    Task 4: Load the transformed data to destination.

    Simulates loading data to a database, data warehouse, or file system.
    """
    logger.info("Starting data loading...")
    print("=" * 60)
    print("LOAD: Loading transformed data to destination...")
    print("=" * 60)

    # Pull transformed data from XCom
    ti = context['ti']
    data = ti.xcom_pull(task_ids='transform_data')

    print(f"Data to load: {data}")

    # Simulate loading to destination
    # In reality, this might write to a database, S3, data warehouse, etc.
    print(f"\n✓ Successfully loaded {len(data)} records to destination")
    print(f"✓ Data loaded successfully: {data}")

    logger.info(f"Successfully loaded {len(data)} records")

    # Return success message
    return {
        'status': 'success',
        'records_loaded': len(data),
        'final_data': data
    }


# Define the DAG
with DAG(
    dag_id='data_pipeline_simulation',
    default_args=default_args,
    description='Simulated data pipeline with extract, validate, transform, and load tasks',
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['tutorial', 'phase1', 'pipeline', 'solution'],
) as dag:

    # Define tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
    )

    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )

    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
    )

    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
    )

    # Define linear pipeline dependencies
    extract_task >> validate_task >> transform_task >> load_task
