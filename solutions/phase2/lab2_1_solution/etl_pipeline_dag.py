"""
ETL Pipeline DAG for E-commerce Data
Complete Airflow DAG with ingest, validate, clean, and transform tasks.
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
RAW_DATA_DIR = '/tmp/ecommerce_data/raw'
VALIDATED_DATA_DIR = '/tmp/ecommerce_data/validated'
CLEANED_DATA_DIR = '/tmp/ecommerce_data/cleaned'
TRANSFORMED_DATA_DIR = '/tmp/ecommerce_data/transformed'


def ingest_data(**context):
    """
    Task 1: Ingest raw data from source.
    In a real scenario, this would fetch from an API, database, or S3.
    """
    logger.info("Starting data ingestion...")

    # Create directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # In a real scenario, we would fetch from external sources
    # For this example, we'll generate sample data
    from generate_ecommerce_data import EcommerceDataGenerator

    generator = EcommerceDataGenerator(seed=42)

    # Generate orders
    orders = generator.generate_orders(num_orders=500)
    orders_file = f'{RAW_DATA_DIR}/orders.csv'
    orders.to_csv(orders_file, index=False)
    logger.info(f"Ingested {len(orders)} orders to {orders_file}")

    # Generate customers
    customers = generator.generate_customer_data(num_customers=250)
    customers_file = f'{RAW_DATA_DIR}/customers.csv'
    customers.to_csv(customers_file, index=False)
    logger.info(f"Ingested {len(customers)} customers to {customers_file}")

    # Push metadata to XCom
    context['task_instance'].xcom_push(key='orders_count', value=len(orders))
    context['task_instance'].xcom_push(key='customers_count', value=len(customers))
    context['task_instance'].xcom_push(key='ingestion_timestamp', value=str(datetime.now()))

    logger.info("Data ingestion completed successfully")


def validate_data(**context):
    """
    Task 2: Validate data quality and business rules.
    """
    logger.info("Starting data validation...")

    os.makedirs(VALIDATED_DATA_DIR, exist_ok=True)

    # Load raw data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')
    customers = pd.read_csv(f'{RAW_DATA_DIR}/customers.csv')

    validation_report = {
        'timestamp': str(datetime.now()),
        'orders': {},
        'customers': {}
    }

    # Validate Orders
    logger.info("Validating orders data...")
    orders_issues = {
        'total_records': len(orders),
        'missing_values': orders.isnull().sum().to_dict(),
        'negative_quantities': int((orders['quantity'] < 0).sum()),
        'negative_amounts': int((orders['total_amount'] < 0).sum()),
        'missing_customer_ids': int(orders['customer_id'].isnull().sum()),
        'duplicate_order_ids': int(orders['order_id'].duplicated().sum())
    }
    validation_report['orders'] = orders_issues
    logger.info(f"Orders validation: {orders_issues}")

    # Validate Customers
    logger.info("Validating customers data...")
    customers_issues = {
        'total_records': len(customers),
        'missing_values': customers.isnull().sum().to_dict(),
        'invalid_emails': int(~customers['email'].str.contains('@.*\\.', na=False).sum()),
        'age_outliers': int(((customers['age'] < 18) | (customers['age'] > 100)).sum()),
        'duplicate_customer_ids': int(customers['customer_id'].duplicated().sum())
    }
    validation_report['customers'] = customers_issues
    logger.info(f"Customers validation: {customers_issues}")

    # Save validation report
    report_file = f'{VALIDATED_DATA_DIR}/validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"Validation report saved to {report_file}")

    # Copy data to validated directory
    orders.to_csv(f'{VALIDATED_DATA_DIR}/orders.csv', index=False)
    customers.to_csv(f'{VALIDATED_DATA_DIR}/customers.csv', index=False)

    # Push validation metrics to XCom
    context['task_instance'].xcom_push(key='validation_report', value=validation_report)

    logger.info("Data validation completed successfully")


def clean_data(**context):
    """
    Task 3: Clean data by handling missing values, outliers, and duplicates.
    """
    logger.info("Starting data cleaning...")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

    # Load validated data
    orders = pd.read_csv(f'{VALIDATED_DATA_DIR}/orders.csv')
    customers = pd.read_csv(f'{VALIDATED_DATA_DIR}/customers.csv')

    cleaning_report = {
        'timestamp': str(datetime.now()),
        'orders': {},
        'customers': {}
    }

    # Clean Orders
    logger.info("Cleaning orders data...")
    initial_orders_count = len(orders)

    # Remove duplicates
    orders = orders.drop_duplicates(subset=['order_id'], keep='first')

    # Remove orders with missing critical fields
    orders = orders.dropna(subset=['order_id', 'customer_id'])

    # Fix negative quantities (set to absolute value)
    negative_qty_mask = orders['quantity'] < 0
    orders.loc[negative_qty_mask, 'quantity'] = orders.loc[negative_qty_mask, 'quantity'].abs()

    # Fix negative amounts (recalculate)
    negative_amt_mask = orders['total_amount'] < 0
    orders.loc[negative_amt_mask, 'total_amount'] = (
        orders.loc[negative_amt_mask, 'price'] * orders.loc[negative_amt_mask, 'quantity']
    )

    # Fill missing total_amount
    missing_amt_mask = orders['total_amount'].isnull()
    orders.loc[missing_amt_mask, 'total_amount'] = (
        orders.loc[missing_amt_mask, 'price'] * orders.loc[missing_amt_mask, 'quantity']
    )

    # Convert order_date to datetime
    orders['order_date'] = pd.to_datetime(orders['order_date'])

    final_orders_count = len(orders)
    cleaning_report['orders'] = {
        'initial_count': initial_orders_count,
        'final_count': final_orders_count,
        'removed_count': initial_orders_count - final_orders_count
    }
    logger.info(f"Orders cleaned: {initial_orders_count} -> {final_orders_count}")

    # Clean Customers
    logger.info("Cleaning customers data...")
    initial_customers_count = len(customers)

    # Remove duplicates
    customers = customers.drop_duplicates(subset=['customer_id'], keep='first')

    # Fix invalid emails (replace with placeholder)
    invalid_email_mask = ~customers['email'].str.contains('@.*\\.', na=False)
    customers.loc[invalid_email_mask, 'email'] = (
        customers.loc[invalid_email_mask, 'customer_id'].str.lower() + '@placeholder.com'
    )

    # Fix age outliers (clip to reasonable range)
    customers['age'] = customers['age'].clip(lower=18, upper=100)

    # Convert registration_date to datetime
    customers['registration_date'] = pd.to_datetime(customers['registration_date'])

    final_customers_count = len(customers)
    cleaning_report['customers'] = {
        'initial_count': initial_customers_count,
        'final_count': final_customers_count,
        'removed_count': initial_customers_count - final_customers_count
    }
    logger.info(f"Customers cleaned: {initial_customers_count} -> {final_customers_count}")

    # Save cleaned data
    orders.to_csv(f'{CLEANED_DATA_DIR}/orders.csv', index=False)
    customers.to_csv(f'{CLEANED_DATA_DIR}/customers.csv', index=False)

    # Save cleaning report
    report_file = f'{CLEANED_DATA_DIR}/cleaning_report.json'
    with open(report_file, 'w') as f:
        json.dump(cleaning_report, f, indent=2)
    logger.info(f"Cleaning report saved to {report_file}")

    # Push cleaning metrics to XCom
    context['task_instance'].xcom_push(key='cleaning_report', value=cleaning_report)

    logger.info("Data cleaning completed successfully")


def transform_data(**context):
    """
    Task 4: Transform data for analytics.
    Create aggregated views and derived features.
    """
    logger.info("Starting data transformation...")

    os.makedirs(TRANSFORMED_DATA_DIR, exist_ok=True)

    # Load cleaned data
    orders = pd.read_csv(f'{CLEANED_DATA_DIR}/orders.csv', parse_dates=['order_date'])
    customers = pd.read_csv(f'{CLEANED_DATA_DIR}/customers.csv', parse_dates=['registration_date'])

    # Transform 1: Customer Order Summary
    logger.info("Creating customer order summary...")
    customer_summary = orders.groupby('customer_id').agg({
        'order_id': 'count',
        'total_amount': ['sum', 'mean', 'max'],
        'quantity': 'sum',
        'order_date': ['min', 'max']
    }).round(2)

    # Flatten column names
    customer_summary.columns = ['_'.join(col).strip() for col in customer_summary.columns.values]
    customer_summary = customer_summary.rename(columns={
        'order_id_count': 'total_orders',
        'total_amount_sum': 'total_spent',
        'total_amount_mean': 'avg_order_value',
        'total_amount_max': 'max_order_value',
        'quantity_sum': 'total_items',
        'order_date_min': 'first_order_date',
        'order_date_max': 'last_order_date'
    })
    customer_summary.to_csv(f'{TRANSFORMED_DATA_DIR}/customer_summary.csv')
    logger.info(f"Customer summary created: {len(customer_summary)} customers")

    # Transform 2: Product Category Performance
    logger.info("Creating product category performance...")
    category_performance = orders.groupby('product_category').agg({
        'order_id': 'count',
        'total_amount': 'sum',
        'quantity': 'sum',
        'price': 'mean'
    }).round(2)
    category_performance = category_performance.rename(columns={
        'order_id': 'total_orders',
        'total_amount': 'total_revenue',
        'quantity': 'total_quantity',
        'price': 'avg_price'
    })
    category_performance = category_performance.sort_values('total_revenue', ascending=False)
    category_performance.to_csv(f'{TRANSFORMED_DATA_DIR}/category_performance.csv')
    logger.info(f"Category performance created: {len(category_performance)} categories")

    # Transform 3: Daily Sales Metrics
    logger.info("Creating daily sales metrics...")
    orders['order_date_only'] = orders['order_date'].dt.date
    daily_sales = orders.groupby('order_date_only').agg({
        'order_id': 'count',
        'total_amount': 'sum',
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    daily_sales = daily_sales.rename(columns={
        'order_id': 'total_orders',
        'total_amount': 'total_revenue',
        'customer_id': 'unique_customers',
        'quantity': 'total_items'
    })
    daily_sales.to_csv(f'{TRANSFORMED_DATA_DIR}/daily_sales.csv')
    logger.info(f"Daily sales metrics created: {len(daily_sales)} days")

    # Transform 4: Country Performance
    logger.info("Creating country performance...")
    country_performance = orders.groupby('country').agg({
        'order_id': 'count',
        'total_amount': 'sum',
        'customer_id': 'nunique'
    }).round(2)
    country_performance = country_performance.rename(columns={
        'order_id': 'total_orders',
        'total_amount': 'total_revenue',
        'customer_id': 'unique_customers'
    })
    country_performance = country_performance.sort_values('total_revenue', ascending=False)
    country_performance.to_csv(f'{TRANSFORMED_DATA_DIR}/country_performance.csv')
    logger.info(f"Country performance created: {len(country_performance)} countries")

    # Transform 5: Enhanced Customer Data (join with orders)
    logger.info("Creating enhanced customer data...")
    customer_summary_reset = customer_summary.reset_index()
    enhanced_customers = customers.merge(
        customer_summary_reset,
        on='customer_id',
        how='left'
    )
    # Fill NaN for customers without orders
    enhanced_customers = enhanced_customers.fillna({
        'total_orders': 0,
        'total_spent': 0,
        'avg_order_value': 0,
        'max_order_value': 0,
        'total_items': 0
    })
    enhanced_customers.to_csv(f'{TRANSFORMED_DATA_DIR}/enhanced_customers.csv', index=False)
    logger.info(f"Enhanced customer data created: {len(enhanced_customers)} customers")

    # Create transformation report
    transformation_report = {
        'timestamp': str(datetime.now()),
        'outputs': {
            'customer_summary': len(customer_summary),
            'category_performance': len(category_performance),
            'daily_sales': len(daily_sales),
            'country_performance': len(country_performance),
            'enhanced_customers': len(enhanced_customers)
        },
        'top_categories': category_performance.head(3).to_dict('index'),
        'top_countries': country_performance.head(3).to_dict('index')
    }

    report_file = f'{TRANSFORMED_DATA_DIR}/transformation_report.json'
    with open(report_file, 'w') as f:
        json.dump(transformation_report, f, indent=2, default=str)
    logger.info(f"Transformation report saved to {report_file}")

    # Push transformation metrics to XCom
    context['task_instance'].xcom_push(key='transformation_report', value=transformation_report)

    logger.info("Data transformation completed successfully")


def generate_final_report(**context):
    """
    Task 5: Generate final ETL pipeline report.
    """
    logger.info("Generating final ETL report...")

    ti = context['task_instance']

    # Pull metrics from previous tasks
    ingestion_timestamp = ti.xcom_pull(task_ids='ingest_data', key='ingestion_timestamp')
    orders_count = ti.xcom_pull(task_ids='ingest_data', key='orders_count')
    customers_count = ti.xcom_pull(task_ids='ingest_data', key='customers_count')
    validation_report = ti.xcom_pull(task_ids='validate_data', key='validation_report')
    cleaning_report = ti.xcom_pull(task_ids='clean_data', key='cleaning_report')
    transformation_report = ti.xcom_pull(task_ids='transform_data', key='transformation_report')

    # Create comprehensive report
    final_report = {
        'pipeline_execution': {
            'execution_date': str(context['execution_date']),
            'completion_time': str(datetime.now()),
            'status': 'SUCCESS'
        },
        'ingestion': {
            'timestamp': ingestion_timestamp,
            'orders_ingested': orders_count,
            'customers_ingested': customers_count
        },
        'validation': validation_report,
        'cleaning': cleaning_report,
        'transformation': transformation_report
    }

    # Save final report
    report_file = f'{TRANSFORMED_DATA_DIR}/etl_pipeline_report.json'
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)

    logger.info("="*60)
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Ingested: {orders_count} orders, {customers_count} customers")
    logger.info(f"Cleaned: {cleaning_report['orders']['final_count']} orders, "
                f"{cleaning_report['customers']['final_count']} customers")
    logger.info(f"Created {len(transformation_report['outputs'])} transformed datasets")
    logger.info(f"Final report: {report_file}")
    logger.info("="*60)


# Create the DAG
with DAG(
    'ecommerce_etl_pipeline',
    default_args=default_args,
    description='Complete ETL pipeline for e-commerce data',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'ecommerce', 'phase2'],
) as dag:

    # Task 0: Create directories
    create_dirs = BashOperator(
        task_id='create_directories',
        bash_command=f'mkdir -p {RAW_DATA_DIR} {VALIDATED_DATA_DIR} {CLEANED_DATA_DIR} {TRANSFORMED_DATA_DIR}'
    )

    # Task 1: Ingest data
    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        provide_context=True
    )

    # Task 2: Validate data
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True
    )

    # Task 3: Clean data
    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True
    )

    # Task 4: Transform data
    transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True
    )

    # Task 5: Generate report
    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_final_report,
        provide_context=True
    )

    # Define task dependencies
    create_dirs >> ingest >> validate >> clean >> transform >> report
