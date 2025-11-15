"""
Scheduled ETL Pipeline with Date Partitioning and Backfill Support
Production-ready scheduled pipeline for e-commerce data processing.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Important for data pipelines
    'email': ['data-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Data directories
BASE_DIR = '/tmp/ecommerce_data/scheduled'
RAW_DIR = f'{BASE_DIR}/raw'
PROCESSED_DIR = f'{BASE_DIR}/processed'
METRICS_DIR = f'{BASE_DIR}/metrics'


def get_partition_path(base_path: str, execution_date: datetime) -> str:
    """
    Generate partition path based on execution date.
    Format: base_path/year=YYYY/month=MM/day=DD
    """
    return os.path.join(
        base_path,
        f"year={execution_date.year}",
        f"month={execution_date.month:02d}",
        f"day={execution_date.day:02d}"
    )


def generate_daily_data(**context):
    """
    Task 1: Generate or ingest daily data with date partitioning.
    """
    execution_date = context['execution_date']
    logger.info(f"Generating data for {execution_date.date()}")

    # Generate sample data for the execution date
    from generate_ecommerce_data import EcommerceDataGenerator

    generator = EcommerceDataGenerator(seed=int(execution_date.timestamp()))

    # Generate orders for this specific date
    start_date = execution_date
    end_date = execution_date + timedelta(hours=23, minutes=59, seconds=59)

    orders = generator.generate_orders(
        num_orders=100 + int(np.random.randint(0, 50)),  # Variable daily volume
        start_date=start_date,
        end_date=end_date
    )

    # Create partition directory
    partition_path = get_partition_path(RAW_DIR, execution_date)
    os.makedirs(partition_path, exist_ok=True)

    # Save data with partition
    output_file = f'{partition_path}/orders.csv'
    orders.to_csv(output_file, index=False)

    logger.info(f"Generated {len(orders)} orders for {execution_date.date()}")
    logger.info(f"Saved to {output_file}")

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='records_generated', value=len(orders))
    context['task_instance'].xcom_push(key='partition_path', value=partition_path)
    context['task_instance'].xcom_push(key='execution_date', value=str(execution_date.date()))


def validate_daily_data(**context):
    """
    Task 2: Validate daily data quality.
    """
    execution_date = context['execution_date']
    partition_path = get_partition_path(RAW_DIR, execution_date)

    logger.info(f"Validating data for {execution_date.date()}")

    # Load data
    orders_file = f'{partition_path}/orders.csv'
    if not os.path.exists(orders_file):
        raise FileNotFoundError(f"Data file not found: {orders_file}")

    orders = pd.read_csv(orders_file)

    # Basic validation checks
    validation_results = {
        'execution_date': str(execution_date.date()),
        'total_records': len(orders),
        'checks': []
    }

    # Check 1: Minimum record count
    min_records = 50
    if len(orders) < min_records:
        validation_results['checks'].append({
            'check': 'minimum_records',
            'passed': False,
            'expected': min_records,
            'actual': len(orders)
        })
    else:
        validation_results['checks'].append({
            'check': 'minimum_records',
            'passed': True
        })

    # Check 2: Required columns
    required_columns = ['order_id', 'customer_id', 'total_amount', 'order_date']
    missing_columns = [col for col in required_columns if col not in orders.columns]
    if missing_columns:
        validation_results['checks'].append({
            'check': 'required_columns',
            'passed': False,
            'missing': missing_columns
        })
    else:
        validation_results['checks'].append({
            'check': 'required_columns',
            'passed': True
        })

    # Check 3: No nulls in critical columns
    null_counts = orders[required_columns].isnull().sum()
    if null_counts.any():
        validation_results['checks'].append({
            'check': 'null_values',
            'passed': False,
            'null_counts': null_counts.to_dict()
        })
    else:
        validation_results['checks'].append({
            'check': 'null_values',
            'passed': True
        })

    # Check 4: Date range
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    date_range_check = (
        (orders['order_date'].dt.date >= execution_date.date()) &
        (orders['order_date'].dt.date <= execution_date.date())
    )
    if not date_range_check.all():
        validation_results['checks'].append({
            'check': 'date_range',
            'passed': False,
            'expected_date': str(execution_date.date()),
            'invalid_count': (~date_range_check).sum()
        })
    else:
        validation_results['checks'].append({
            'check': 'date_range',
            'passed': True
        })

    # Overall validation status
    all_passed = all(check['passed'] for check in validation_results['checks'])
    validation_results['overall_passed'] = all_passed

    # Save validation report
    validation_file = f'{partition_path}/validation_report.json'
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    logger.info(f"Validation {'PASSED' if all_passed else 'FAILED'} for {execution_date.date()}")

    # Push to XCom
    context['task_instance'].xcom_push(key='validation_passed', value=all_passed)

    if not all_passed:
        raise ValueError(f"Data validation failed for {execution_date.date()}")


def process_daily_data(**context):
    """
    Task 3: Process and transform daily data.
    """
    execution_date = context['execution_date']
    raw_partition_path = get_partition_path(RAW_DIR, execution_date)
    processed_partition_path = get_partition_path(PROCESSED_DIR, execution_date)

    logger.info(f"Processing data for {execution_date.date()}")

    # Load raw data
    orders = pd.read_csv(f'{raw_partition_path}/orders.csv', parse_dates=['order_date'])

    # Data cleaning
    # Remove duplicates
    initial_count = len(orders)
    orders = orders.drop_duplicates(subset=['order_id'], keep='first')
    duplicates_removed = initial_count - len(orders)

    # Fix negative values
    negative_qty = (orders['quantity'] < 0).sum()
    orders.loc[orders['quantity'] < 0, 'quantity'] = orders['quantity'].abs()

    # Remove nulls in critical columns
    orders = orders.dropna(subset=['order_id', 'customer_id', 'total_amount'])

    # Data transformations
    # Add derived columns
    orders['order_hour'] = orders['order_date'].dt.hour
    orders['order_day_of_week'] = orders['order_date'].dt.dayofweek
    orders['is_weekend'] = orders['order_day_of_week'].isin([5, 6])
    orders['order_value_category'] = pd.cut(
        orders['total_amount'],
        bins=[0, 50, 100, 500, float('inf')],
        labels=['Small', 'Medium', 'Large', 'XLarge']
    )

    # Create aggregations
    daily_summary = {
        'execution_date': str(execution_date.date()),
        'total_orders': len(orders),
        'total_revenue': float(orders['total_amount'].sum()),
        'avg_order_value': float(orders['total_amount'].mean()),
        'unique_customers': int(orders['customer_id'].nunique()),
        'orders_by_category': orders['product_category'].value_counts().to_dict(),
        'orders_by_hour': orders['order_hour'].value_counts().sort_index().to_dict(),
        'weekend_orders': int(orders['is_weekend'].sum()),
        'weekday_orders': int((~orders['is_weekend']).sum()),
        'processing_stats': {
            'duplicates_removed': duplicates_removed,
            'negative_values_fixed': int(negative_qty)
        }
    }

    # Create partition directory
    os.makedirs(processed_partition_path, exist_ok=True)

    # Save processed data
    processed_file = f'{processed_partition_path}/orders_processed.csv'
    orders.to_csv(processed_file, index=False)

    summary_file = f'{processed_partition_path}/daily_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(daily_summary, f, indent=2, default=str)

    logger.info(f"Processed {len(orders)} orders for {execution_date.date()}")
    logger.info(f"Total Revenue: ${daily_summary['total_revenue']:.2f}")

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='processed_records', value=len(orders))
    context['task_instance'].xcom_push(key='total_revenue', value=daily_summary['total_revenue'])
    context['task_instance'].xcom_push(key='daily_summary', value=daily_summary)


def compute_rolling_metrics(**context):
    """
    Task 4: Compute rolling metrics across multiple days.
    """
    execution_date = context['execution_date']
    logger.info(f"Computing rolling metrics for {execution_date.date()}")

    # Load last 7 days of processed data
    daily_summaries = []
    for i in range(7):
        date = execution_date - timedelta(days=i)
        partition_path = get_partition_path(PROCESSED_DIR, date)
        summary_file = f'{partition_path}/daily_summary.json'

        if os.path.exists(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
                daily_summaries.append(summary)

    if not daily_summaries:
        logger.warning("No historical data available for rolling metrics")
        return

    # Compute rolling metrics
    rolling_metrics = {
        'execution_date': str(execution_date.date()),
        'window_days': 7,
        'days_with_data': len(daily_summaries),
        'metrics': {
            'avg_daily_orders': np.mean([s['total_orders'] for s in daily_summaries]),
            'avg_daily_revenue': np.mean([s['total_revenue'] for s in daily_summaries]),
            'total_orders_7d': sum([s['total_orders'] for s in daily_summaries]),
            'total_revenue_7d': sum([s['total_revenue'] for s in daily_summaries]),
            'avg_order_value_7d': np.mean([s['avg_order_value'] for s in daily_summaries]),
            'unique_customers_7d': sum([s['unique_customers'] for s in daily_summaries])
        },
        'trends': {
            'daily_orders': [s['total_orders'] for s in daily_summaries],
            'daily_revenue': [s['total_revenue'] for s in daily_summaries]
        }
    }

    # Calculate growth rate (today vs 7-day average)
    if len(daily_summaries) > 1:
        today_revenue = daily_summaries[0]['total_revenue']
        avg_revenue = rolling_metrics['metrics']['avg_daily_revenue']
        rolling_metrics['growth_rate'] = ((today_revenue - avg_revenue) / avg_revenue) * 100 if avg_revenue > 0 else 0

    # Save rolling metrics
    metrics_partition_path = get_partition_path(METRICS_DIR, execution_date)
    os.makedirs(metrics_partition_path, exist_ok=True)

    metrics_file = f'{metrics_partition_path}/rolling_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(rolling_metrics, f, indent=2, default=str)

    logger.info(f"7-day rolling metrics:")
    logger.info(f"  Avg Daily Orders: {rolling_metrics['metrics']['avg_daily_orders']:.0f}")
    logger.info(f"  Avg Daily Revenue: ${rolling_metrics['metrics']['avg_daily_revenue']:.2f}")
    if 'growth_rate' in rolling_metrics:
        logger.info(f"  Growth Rate: {rolling_metrics['growth_rate']:.1f}%")

    # Push to XCom
    context['task_instance'].xcom_push(key='rolling_metrics', value=rolling_metrics)


def generate_daily_report(**context):
    """
    Task 5: Generate comprehensive daily report.
    """
    execution_date = context['execution_date']
    ti = context['task_instance']

    logger.info(f"Generating daily report for {execution_date.date()}")

    # Pull metrics from previous tasks
    records_generated = ti.xcom_pull(task_ids='generate_daily_data', key='records_generated')
    validation_passed = ti.xcom_pull(task_ids='validate_daily_data', key='validation_passed')
    processed_records = ti.xcom_pull(task_ids='process_daily_data', key='processed_records')
    total_revenue = ti.xcom_pull(task_ids='process_daily_data', key='total_revenue')
    daily_summary = ti.xcom_pull(task_ids='process_daily_data', key='daily_summary')
    rolling_metrics = ti.xcom_pull(task_ids='compute_rolling_metrics', key='rolling_metrics')

    # Create comprehensive report
    report = {
        'execution_date': str(execution_date.date()),
        'pipeline_run': {
            'start_time': str(context['data_interval_start']),
            'end_time': str(context['data_interval_end']),
            'status': 'SUCCESS'
        },
        'data_ingestion': {
            'records_generated': records_generated,
            'validation_passed': validation_passed
        },
        'data_processing': {
            'records_processed': processed_records,
            'total_revenue': total_revenue
        },
        'daily_summary': daily_summary,
        'rolling_metrics': rolling_metrics
    }

    # Save report
    report_partition_path = get_partition_path(METRICS_DIR, execution_date)
    os.makedirs(report_partition_path, exist_ok=True)

    report_file = f'{report_partition_path}/daily_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Also save to a 'latest' file for easy access
    latest_report_file = f'{METRICS_DIR}/latest_daily_report.json'
    with open(latest_report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("="*60)
    logger.info(f"DAILY PIPELINE REPORT - {execution_date.date()}")
    logger.info("="*60)
    logger.info(f"Records Generated: {records_generated}")
    logger.info(f"Records Processed: {processed_records}")
    logger.info(f"Total Revenue: ${total_revenue:.2f}")
    logger.info(f"Unique Customers: {daily_summary['unique_customers']}")
    logger.info(f"Avg Order Value: ${daily_summary['avg_order_value']:.2f}")
    if rolling_metrics and 'growth_rate' in rolling_metrics:
        logger.info(f"7-day Growth Rate: {rolling_metrics['growth_rate']:.1f}%")
    logger.info(f"Report: {report_file}")
    logger.info("="*60)


def cleanup_old_partitions(**context):
    """
    Task 6: Cleanup old data partitions (retention policy).
    """
    execution_date = context['execution_date']
    retention_days = 30  # Keep last 30 days

    logger.info(f"Cleaning up partitions older than {retention_days} days")

    cutoff_date = execution_date - timedelta(days=retention_days)
    deleted_count = 0

    # Cleanup RAW directory
    for base_dir in [RAW_DIR, PROCESSED_DIR]:
        if not os.path.exists(base_dir):
            continue

        for year_dir in Path(base_dir).glob('year=*'):
            for month_dir in year_dir.glob('month=*'):
                for day_dir in month_dir.glob('day=*'):
                    # Parse date from partition path
                    year = int(year_dir.name.split('=')[1])
                    month = int(month_dir.name.split('=')[1])
                    day = int(day_dir.name.split('=')[1])
                    partition_date = datetime(year, month, day)

                    # Delete if older than retention
                    if partition_date < cutoff_date:
                        try:
                            import shutil
                            shutil.rmtree(day_dir)
                            deleted_count += 1
                            logger.info(f"Deleted partition: {day_dir}")
                        except Exception as e:
                            logger.error(f"Error deleting {day_dir}: {e}")

    logger.info(f"Cleanup completed: {deleted_count} partitions deleted")

    # Push to XCom
    context['task_instance'].xcom_push(key='partitions_deleted', value=deleted_count)


# Create the DAG
with DAG(
    'scheduled_ecommerce_etl',
    default_args=default_args,
    description='Scheduled ETL pipeline with date partitioning and backfill support',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2024, 1, 1),  # Start date for backfill
    end_date=None,  # No end date
    catchup=True,  # Enable backfill for historical dates
    max_active_runs=3,  # Allow parallel backfill runs
    tags=['etl', 'scheduled', 'production', 'ecommerce', 'phase2'],
) as dag:

    # Task 0: Create base directories
    create_dirs = BashOperator(
        task_id='create_base_directories',
        bash_command=f'mkdir -p {RAW_DIR} {PROCESSED_DIR} {METRICS_DIR}'
    )

    # Task 1: Generate daily data
    generate_data = PythonOperator(
        task_id='generate_daily_data',
        python_callable=generate_daily_data,
        provide_context=True
    )

    # Task 2: Validate daily data
    validate_data = PythonOperator(
        task_id='validate_daily_data',
        python_callable=validate_daily_data,
        provide_context=True
    )

    # Task 3: Process daily data
    process_data = PythonOperator(
        task_id='process_daily_data',
        python_callable=process_daily_data,
        provide_context=True
    )

    # Task 4: Compute rolling metrics
    compute_metrics = PythonOperator(
        task_id='compute_rolling_metrics',
        python_callable=compute_rolling_metrics,
        provide_context=True
    )

    # Task 5: Generate daily report
    generate_report = PythonOperator(
        task_id='generate_daily_report',
        python_callable=generate_daily_report,
        provide_context=True
    )

    # Task 6: Cleanup old partitions (run weekly)
    cleanup = PythonOperator(
        task_id='cleanup_old_partitions',
        python_callable=cleanup_old_partitions,
        provide_context=True,
        trigger_rule='all_done'  # Run even if previous tasks fail
    )

    # Define task dependencies
    create_dirs >> generate_data >> validate_data >> process_data >> compute_metrics >> generate_report >> cleanup
