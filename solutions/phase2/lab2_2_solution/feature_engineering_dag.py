"""
Feature Engineering DAG for E-commerce Data
Extends ETL pipeline with comprehensive feature engineering.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import json
import os
import logging

# Import feature engineering utilities
from feature_utils import (
    calculate_rfm_features,
    calculate_temporal_features,
    calculate_behavioral_features,
    calculate_customer_value_features,
    create_aggregated_features,
    combine_all_features,
    create_feature_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Data directories
INPUT_DATA_DIR = '/tmp/ecommerce_data/cleaned'
FEATURES_DIR = '/tmp/ecommerce_data/features'


def load_data(**context):
    """
    Task 1: Load cleaned data for feature engineering.
    """
    logger.info("Loading cleaned data...")

    # Load orders and customers
    orders = pd.read_csv(f'{INPUT_DATA_DIR}/orders.csv', parse_dates=['order_date'])
    customers = pd.read_csv(f'{INPUT_DATA_DIR}/customers.csv', parse_dates=['registration_date'])

    logger.info(f"Loaded {len(orders)} orders and {len(customers)} customers")

    # Push data statistics to XCom
    context['task_instance'].xcom_push(key='orders_count', value=len(orders))
    context['task_instance'].xcom_push(key='customers_count', value=len(customers))
    context['task_instance'].xcom_push(key='date_range', value={
        'start': str(orders['order_date'].min()),
        'end': str(orders['order_date'].max())
    })

    # Save to temporary location for feature engineering
    os.makedirs(FEATURES_DIR, exist_ok=True)
    orders.to_csv(f'{FEATURES_DIR}/temp_orders.csv', index=False)
    customers.to_csv(f'{FEATURES_DIR}/temp_customers.csv', index=False)

    logger.info("Data loaded successfully")


def create_rfm_features(**context):
    """
    Task 2: Create RFM (Recency, Frequency, Monetary) features.
    """
    logger.info("Creating RFM features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])

    # Calculate RFM features
    rfm_features = calculate_rfm_features(
        orders,
        customer_id_col='customer_id',
        order_date_col='order_date',
        amount_col='total_amount'
    )

    # Save features
    output_file = f'{FEATURES_DIR}/rfm_features.csv'
    rfm_features.to_csv(output_file, index=False)
    logger.info(f"RFM features saved to {output_file}")

    # Push statistics to XCom
    segment_dist = rfm_features['customer_segment'].value_counts().to_dict()
    context['task_instance'].xcom_push(key='segment_distribution', value=segment_dist)
    context['task_instance'].xcom_push(key='avg_rfm_score', value=float(rfm_features['rfm_score'].mean()))

    logger.info(f"Customer segments: {segment_dist}")
    logger.info("RFM features created successfully")


def create_temporal_features(**context):
    """
    Task 3: Create temporal features from order history.
    """
    logger.info("Creating temporal features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])

    # Calculate temporal features
    temporal_features = calculate_temporal_features(
        orders,
        customer_id_col='customer_id',
        order_date_col='order_date'
    )

    # Save features
    output_file = f'{FEATURES_DIR}/temporal_features.csv'
    temporal_features.to_csv(output_file, index=False)
    logger.info(f"Temporal features saved to {output_file}")

    # Push statistics to XCom
    context['task_instance'].xcom_push(
        key='avg_days_between_orders',
        value=float(temporal_features['avg_days_between_orders'].mean())
    )
    context['task_instance'].xcom_push(
        key='avg_customer_lifetime',
        value=float(temporal_features['customer_lifetime_days'].mean())
    )

    logger.info("Temporal features created successfully")


def create_behavioral_features(**context):
    """
    Task 4: Create behavioral features based on purchase patterns.
    """
    logger.info("Creating behavioral features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])

    # Calculate behavioral features
    behavioral_features = calculate_behavioral_features(
        orders,
        customer_id_col='customer_id',
        product_category_col='product_category',
        quantity_col='quantity',
        status_col='status'
    )

    # Save features
    output_file = f'{FEATURES_DIR}/behavioral_features.csv'
    behavioral_features.to_csv(output_file, index=False)
    logger.info(f"Behavioral features saved to {output_file}")

    # Push statistics to XCom
    top_categories = behavioral_features['most_purchased_category'].value_counts().head(5).to_dict()
    context['task_instance'].xcom_push(key='top_categories', value=top_categories)
    context['task_instance'].xcom_push(
        key='avg_category_diversity',
        value=float(behavioral_features['category_diversity_score'].mean())
    )

    logger.info("Behavioral features created successfully")


def create_customer_value_features(**context):
    """
    Task 5: Create customer value and engagement features.
    """
    logger.info("Creating customer value features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])
    customers = pd.read_csv(f'{FEATURES_DIR}/temp_customers.csv', parse_dates=['registration_date'])

    # Calculate customer value features
    value_features = calculate_customer_value_features(
        orders,
        customers,
        customer_id_col='customer_id',
        order_date_col='order_date',
        amount_col='total_amount'
    )

    # Save features
    output_file = f'{FEATURES_DIR}/customer_value_features.csv'
    value_features.to_csv(output_file, index=False)
    logger.info(f"Customer value features saved to {output_file}")

    # Push statistics to XCom
    context['task_instance'].xcom_push(
        key='avg_total_revenue',
        value=float(value_features['total_revenue'].mean())
    )
    context['task_instance'].xcom_push(
        key='avg_engagement_score',
        value=float(value_features['engagement_score'].mean())
    )

    logger.info("Customer value features created successfully")


def create_aggregated_features(**context):
    """
    Task 6: Create time-windowed aggregated features.
    """
    logger.info("Creating aggregated features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])

    # Create 30-day and 90-day aggregated features
    agg_30d = create_aggregated_features(orders, time_window_days=30)
    agg_90d = create_aggregated_features(orders, time_window_days=90)

    # Save features
    agg_30d.to_csv(f'{FEATURES_DIR}/aggregated_features_30d.csv', index=False)
    agg_90d.to_csv(f'{FEATURES_DIR}/aggregated_features_90d.csv', index=False)
    logger.info("Aggregated features saved")

    # Push statistics to XCom
    context['task_instance'].xcom_push(
        key='active_customers_30d',
        value=len(agg_30d)
    )
    context['task_instance'].xcom_push(
        key='active_customers_90d',
        value=len(agg_90d)
    )

    logger.info("Aggregated features created successfully")


def combine_features(**context):
    """
    Task 7: Combine all features into a master feature set.
    """
    logger.info("Combining all features...")

    # Load data
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv', parse_dates=['order_date'])
    customers = pd.read_csv(f'{FEATURES_DIR}/temp_customers.csv', parse_dates=['registration_date'])

    # Combine all features
    combined_features = combine_all_features(
        orders,
        customers,
        customer_id_col='customer_id'
    )

    # Save combined features
    output_file = f'{FEATURES_DIR}/combined_features.csv'
    combined_features.to_csv(output_file, index=False)
    logger.info(f"Combined features saved to {output_file}")

    # Create feature summary
    feature_summary = create_feature_summary(combined_features)

    # Save summary
    summary_file = f'{FEATURES_DIR}/feature_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(feature_summary, f, indent=2, default=str)
    logger.info(f"Feature summary saved to {summary_file}")

    # Push summary to XCom
    context['task_instance'].xcom_push(key='feature_summary', value=feature_summary)

    logger.info(f"Combined {feature_summary['total_features']} features for {feature_summary['total_customers']} customers")
    logger.info("Feature combination completed successfully")


def create_feature_store(**context):
    """
    Task 8: Create feature store with versioning and metadata.
    """
    logger.info("Creating feature store...")

    # Load combined features
    features = pd.read_csv(f'{FEATURES_DIR}/combined_features.csv')

    # Create feature store directory
    feature_store_dir = f'{FEATURES_DIR}/feature_store'
    os.makedirs(feature_store_dir, exist_ok=True)

    # Version features by date
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_file = f'{feature_store_dir}/features_v{version}.csv'
    features.to_csv(versioned_file, index=False)

    # Also save as 'latest'
    latest_file = f'{feature_store_dir}/features_latest.csv'
    features.to_csv(latest_file, index=False)

    logger.info(f"Features saved to feature store: {versioned_file}")

    # Create metadata
    metadata = {
        'version': version,
        'created_at': str(datetime.now()),
        'num_features': len(features.columns),
        'num_customers': len(features),
        'feature_list': list(features.columns),
        'feature_types': {
            col: str(features[col].dtype) for col in features.columns
        },
        'feature_stats': {
            col: {
                'mean': float(features[col].mean()) if features[col].dtype in [np.float64, np.int64] else None,
                'std': float(features[col].std()) if features[col].dtype in [np.float64, np.int64] else None,
                'min': float(features[col].min()) if features[col].dtype in [np.float64, np.int64] else None,
                'max': float(features[col].max()) if features[col].dtype in [np.float64, np.int64] else None,
                'null_count': int(features[col].isnull().sum())
            } for col in features.columns
        }
    }

    # Save metadata
    metadata_file = f'{feature_store_dir}/metadata_v{version}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    latest_metadata_file = f'{feature_store_dir}/metadata_latest.json'
    with open(latest_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata saved: {metadata_file}")

    # Push metadata to XCom
    context['task_instance'].xcom_push(key='feature_version', value=version)
    context['task_instance'].xcom_push(key='feature_store_path', value=feature_store_dir)

    logger.info("Feature store created successfully")


def generate_feature_report(**context):
    """
    Task 9: Generate comprehensive feature engineering report.
    """
    logger.info("Generating feature engineering report...")

    ti = context['task_instance']

    # Pull metrics from previous tasks
    orders_count = ti.xcom_pull(task_ids='load_data', key='orders_count')
    customers_count = ti.xcom_pull(task_ids='load_data', key='customers_count')
    date_range = ti.xcom_pull(task_ids='load_data', key='date_range')
    segment_dist = ti.xcom_pull(task_ids='create_rfm_features', key='segment_distribution')
    avg_rfm_score = ti.xcom_pull(task_ids='create_rfm_features', key='avg_rfm_score')
    feature_summary = ti.xcom_pull(task_ids='combine_features', key='feature_summary')
    feature_version = ti.xcom_pull(task_ids='create_feature_store', key='feature_version')

    # Create comprehensive report
    report = {
        'pipeline_execution': {
            'execution_date': str(context['execution_date']),
            'completion_time': str(datetime.now()),
            'status': 'SUCCESS'
        },
        'input_data': {
            'orders_count': orders_count,
            'customers_count': customers_count,
            'date_range': date_range
        },
        'rfm_analysis': {
            'avg_rfm_score': avg_rfm_score,
            'segment_distribution': segment_dist
        },
        'feature_summary': feature_summary,
        'feature_store': {
            'version': feature_version,
            'location': f'{FEATURES_DIR}/feature_store'
        }
    }

    # Save report
    report_file = f'{FEATURES_DIR}/feature_engineering_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("="*60)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"Input: {orders_count} orders, {customers_count} customers")
    logger.info(f"Created {feature_summary['total_features']} features")
    logger.info(f"Customer segments: {segment_dist}")
    logger.info(f"Feature version: {feature_version}")
    logger.info(f"Report: {report_file}")
    logger.info("="*60)


# Create the DAG
with DAG(
    'ecommerce_feature_engineering',
    default_args=default_args,
    description='Feature engineering pipeline for e-commerce data',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['feature-engineering', 'ecommerce', 'phase2'],
) as dag:

    # Task 0: Create directories
    create_dirs = BashOperator(
        task_id='create_directories',
        bash_command=f'mkdir -p {FEATURES_DIR}'
    )

    # Task 1: Load data
    load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True
    )

    # Task 2: Create RFM features
    rfm = PythonOperator(
        task_id='create_rfm_features',
        python_callable=create_rfm_features,
        provide_context=True
    )

    # Task 3: Create temporal features
    temporal = PythonOperator(
        task_id='create_temporal_features',
        python_callable=create_temporal_features,
        provide_context=True
    )

    # Task 4: Create behavioral features
    behavioral = PythonOperator(
        task_id='create_behavioral_features',
        python_callable=create_behavioral_features,
        provide_context=True
    )

    # Task 5: Create customer value features
    value = PythonOperator(
        task_id='create_customer_value_features',
        python_callable=create_customer_value_features,
        provide_context=True
    )

    # Task 6: Create aggregated features
    aggregated = PythonOperator(
        task_id='create_aggregated_features',
        python_callable=create_aggregated_features,
        provide_context=True
    )

    # Task 7: Combine all features
    combine = PythonOperator(
        task_id='combine_features',
        python_callable=combine_features,
        provide_context=True
    )

    # Task 8: Create feature store
    store = PythonOperator(
        task_id='create_feature_store',
        python_callable=create_feature_store,
        provide_context=True
    )

    # Task 9: Generate report
    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_feature_report,
        provide_context=True
    )

    # Define task dependencies
    # Load data first, then create all features in parallel, then combine and store
    create_dirs >> load >> [rfm, temporal, behavioral, value, aggregated]
    [rfm, temporal, behavioral, value, aggregated] >> combine >> store >> report
