"""
Data Validation DAG
Comprehensive data quality validation pipeline using DataValidator.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd
import json
import os
import logging

from data_validator import DataValidator, create_validation_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,  # Alert on validation failures
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Data directories
RAW_DATA_DIR = '/tmp/ecommerce_data/raw'
VALIDATION_DIR = '/tmp/ecommerce_data/validation'


def validate_orders_schema(**context):
    """
    Task 1: Validate orders data schema.
    """
    logger.info("Validating orders schema...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Define expected schema
    expected_schema = {
        'order_id': 'string',
        'customer_id': 'string',
        'order_date': 'string',
        'product_category': 'string',
        'product_name': 'string',
        'quantity': 'int',
        'price': 'float',
        'total_amount': 'float',
        'country': 'string',
        'city': 'string',
        'payment_method': 'string',
        'status': 'string'
    }

    # Create validator and validate
    validator = DataValidator()
    result = validator.validate_schema(orders, expected_schema)

    # Save results
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    with open(f'{VALIDATION_DIR}/orders_schema_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='schema_valid', value=result['passed'])

    if not result['passed']:
        logger.error(f"Schema validation failed: {result['issues']}")
        raise ValueError("Orders schema validation failed")

    logger.info("Orders schema validation passed")


def validate_orders_completeness(**context):
    """
    Task 2: Validate orders data completeness (missing values).
    """
    logger.info("Validating orders data completeness...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Create validator
    validator = DataValidator()

    # Define required columns (no nulls allowed)
    required_columns = ['order_id', 'customer_id', 'product_category', 'product_name']

    # Validate missing values
    result = validator.validate_missing_values(
        orders,
        required_columns=required_columns,
        max_missing_pct=0.10  # Allow max 10% missing for non-required columns
    )

    # Save results
    with open(f'{VALIDATION_DIR}/orders_completeness_validation.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Push to XCom
    context['task_instance'].xcom_push(key='completeness_valid', value=result['passed'])
    context['task_instance'].xcom_push(key='missing_stats', value=result['statistics'])

    logger.info(f"Completeness validation: {'PASSED' if result['passed'] else 'FAILED'}")


def validate_orders_uniqueness(**context):
    """
    Task 3: Validate orders uniqueness (duplicates).
    """
    logger.info("Validating orders uniqueness...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Create validator
    validator = DataValidator()

    # Validate duplicates
    result = validator.validate_duplicates(
        orders,
        unique_columns=['order_id'],
        check_full_duplicates=True
    )

    # Save results
    with open(f'{VALIDATION_DIR}/orders_uniqueness_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='uniqueness_valid', value=result['passed'])

    logger.info(f"Uniqueness validation: {'PASSED' if result['passed'] else 'FAILED'}")


def validate_orders_ranges(**context):
    """
    Task 4: Validate orders value ranges.
    """
    logger.info("Validating orders value ranges...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Create validator
    validator = DataValidator()

    # Define range rules
    range_rules = {
        'quantity': (0, 1000),
        'price': (0, 10000),
        'total_amount': (0, 50000)
    }

    # Validate ranges
    result = validator.validate_value_ranges(orders, range_rules)

    # Save results
    with open(f'{VALIDATION_DIR}/orders_ranges_validation.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Push to XCom
    context['task_instance'].xcom_push(key='ranges_valid', value=result['passed'])

    logger.info(f"Value ranges validation: {'PASSED' if result['passed'] else 'FAILED'}")


def validate_orders_categorical(**context):
    """
    Task 5: Validate orders categorical values.
    """
    logger.info("Validating orders categorical values...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Create validator
    validator = DataValidator()

    # Define allowed categorical values
    allowed_values = {
        'product_category': ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Toys', 'Sports'],
        'payment_method': ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer', 'Apple Pay'],
        'status': ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
        'country': ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan']
    }

    # Validate categorical values
    result = validator.validate_categorical_values(orders, allowed_values)

    # Save results
    with open(f'{VALIDATION_DIR}/orders_categorical_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='categorical_valid', value=result['passed'])

    logger.info(f"Categorical values validation: {'PASSED' if result['passed'] else 'FAILED'}")


def validate_orders_business_rules(**context):
    """
    Task 6: Validate orders business rules.
    """
    logger.info("Validating orders business rules...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')

    # Create validator
    validator = DataValidator()

    # Define business rules
    business_rules = [
        {
            'name': 'total_equals_price_times_quantity',
            'condition': 'abs(total_amount - (price * quantity)) < 0.01',
            'description': 'Total amount should equal price times quantity'
        },
        {
            'name': 'positive_quantity',
            'condition': 'quantity > 0',
            'description': 'Quantity must be positive'
        },
        {
            'name': 'positive_price',
            'condition': 'price > 0',
            'description': 'Price must be positive'
        },
        {
            'name': 'positive_total',
            'condition': 'total_amount > 0',
            'description': 'Total amount must be positive'
        }
    ]

    # Validate business rules
    result = validator.validate_business_rules(orders, business_rules)

    # Save results
    with open(f'{VALIDATION_DIR}/orders_business_rules_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='business_rules_valid', value=result['passed'])

    logger.info(f"Business rules validation: {'PASSED' if result['passed'] else 'FAILED'}")


def validate_customers_schema(**context):
    """
    Task 7: Validate customers data schema.
    """
    logger.info("Validating customers schema...")

    # Load data
    customers = pd.read_csv(f'{RAW_DATA_DIR}/customers.csv')

    # Define expected schema
    expected_schema = {
        'customer_id': 'string',
        'email': 'string',
        'registration_date': 'string',
        'country': 'string',
        'city': 'string',
        'age': 'int',
        'is_premium': 'bool'
    }

    # Create validator and validate
    validator = DataValidator()
    result = validator.validate_schema(customers, expected_schema)

    # Save results
    with open(f'{VALIDATION_DIR}/customers_schema_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='customers_schema_valid', value=result['passed'])

    if not result['passed']:
        logger.error(f"Customers schema validation failed: {result['issues']}")
        raise ValueError("Customers schema validation failed")

    logger.info("Customers schema validation passed")


def validate_customers_quality(**context):
    """
    Task 8: Validate customers data quality (emails, age ranges, etc.).
    """
    logger.info("Validating customers data quality...")

    # Load data
    customers = pd.read_csv(f'{RAW_DATA_DIR}/customers.csv')

    # Create validator
    validator = DataValidator()

    # Validate email format
    email_result = validator.validate_email_format(customers, 'email')

    # Validate age range
    age_result = validator.validate_value_ranges(
        customers,
        {'age': (18, 100)}
    )

    # Validate required fields
    completeness_result = validator.validate_missing_values(
        customers,
        required_columns=['customer_id', 'email'],
        max_missing_pct=0.05
    )

    # Get full report
    full_report = validator.get_validation_report()

    # Save results
    with open(f'{VALIDATION_DIR}/customers_quality_validation.json', 'w') as f:
        json.dump(full_report, f, indent=2, default=str)

    # Push to XCom
    context['task_instance'].xcom_push(key='customers_quality_valid', value=full_report['passed'])

    logger.info(f"Customers quality validation: {'PASSED' if full_report['passed'] else 'FAILED'}")


def validate_referential_integrity(**context):
    """
    Task 9: Validate referential integrity between orders and customers.
    """
    logger.info("Validating referential integrity...")

    # Load data
    orders = pd.read_csv(f'{RAW_DATA_DIR}/orders.csv')
    customers = pd.read_csv(f'{RAW_DATA_DIR}/customers.csv')

    # Create validator
    validator = DataValidator()

    # Validate that all customer_ids in orders exist in customers
    result = validator.validate_referential_integrity(
        orders,
        customers,
        'customer_id',
        df1_name='orders',
        df2_name='customers'
    )

    # Save results
    with open(f'{VALIDATION_DIR}/referential_integrity_validation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Push to XCom
    context['task_instance'].xcom_push(key='referential_integrity_valid', value=result['passed'])

    logger.info(f"Referential integrity validation: {'PASSED' if result['passed'] else 'FAILED'}")


def generate_comprehensive_report(**context):
    """
    Task 10: Generate comprehensive validation report.
    """
    logger.info("Generating comprehensive validation report...")

    ti = context['task_instance']

    # Pull all validation results
    validations = {
        'orders': {
            'schema': ti.xcom_pull(task_ids='validate_orders_schema', key='schema_valid'),
            'completeness': ti.xcom_pull(task_ids='validate_orders_completeness', key='completeness_valid'),
            'uniqueness': ti.xcom_pull(task_ids='validate_orders_uniqueness', key='uniqueness_valid'),
            'ranges': ti.xcom_pull(task_ids='validate_orders_ranges', key='ranges_valid'),
            'categorical': ti.xcom_pull(task_ids='validate_orders_categorical', key='categorical_valid'),
            'business_rules': ti.xcom_pull(task_ids='validate_orders_business_rules', key='business_rules_valid')
        },
        'customers': {
            'schema': ti.xcom_pull(task_ids='validate_customers_schema', key='customers_schema_valid'),
            'quality': ti.xcom_pull(task_ids='validate_customers_quality', key='customers_quality_valid')
        },
        'cross_table': {
            'referential_integrity': ti.xcom_pull(task_ids='validate_referential_integrity', key='referential_integrity_valid')
        }
    }

    # Calculate overall status
    all_passed = all([
        validations['orders']['schema'],
        validations['orders']['completeness'],
        validations['orders']['uniqueness'],
        validations['orders']['ranges'],
        validations['orders']['categorical'],
        validations['orders']['business_rules'],
        validations['customers']['schema'],
        validations['customers']['quality'],
        validations['cross_table']['referential_integrity']
    ])

    # Create comprehensive report
    report = {
        'timestamp': str(datetime.now()),
        'execution_date': str(context['execution_date']),
        'overall_status': 'PASSED' if all_passed else 'FAILED',
        'validations': validations,
        'summary': {
            'total_checks': sum([len(v) for v in validations.values()]),
            'passed_checks': sum([sum([1 for result in v.values() if result]) for v in validations.values()]),
            'failed_checks': sum([sum([1 for result in v.values() if not result]) for v in validations.values()])
        }
    }

    # Save report
    report_file = f'{VALIDATION_DIR}/comprehensive_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("="*60)
    logger.info("DATA VALIDATION PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"Total Checks: {report['summary']['total_checks']}")
    logger.info(f"Passed: {report['summary']['passed_checks']}")
    logger.info(f"Failed: {report['summary']['failed_checks']}")
    logger.info(f"Report: {report_file}")
    logger.info("="*60)

    # If validation failed, raise an alert
    if not all_passed:
        logger.error("Data validation failed! Please review the validation reports.")
        # In production, this would trigger alerts, notifications, etc.


# Create the DAG
with DAG(
    'ecommerce_data_validation',
    default_args=default_args,
    description='Comprehensive data validation pipeline for e-commerce data',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['validation', 'data-quality', 'ecommerce', 'phase2'],
) as dag:

    # Task 0: Create directories
    create_dirs = BashOperator(
        task_id='create_directories',
        bash_command=f'mkdir -p {VALIDATION_DIR}'
    )

    # Orders validation tasks
    validate_orders_schema_task = PythonOperator(
        task_id='validate_orders_schema',
        python_callable=validate_orders_schema,
        provide_context=True
    )

    validate_orders_completeness_task = PythonOperator(
        task_id='validate_orders_completeness',
        python_callable=validate_orders_completeness,
        provide_context=True
    )

    validate_orders_uniqueness_task = PythonOperator(
        task_id='validate_orders_uniqueness',
        python_callable=validate_orders_uniqueness,
        provide_context=True
    )

    validate_orders_ranges_task = PythonOperator(
        task_id='validate_orders_ranges',
        python_callable=validate_orders_ranges,
        provide_context=True
    )

    validate_orders_categorical_task = PythonOperator(
        task_id='validate_orders_categorical',
        python_callable=validate_orders_categorical,
        provide_context=True
    )

    validate_orders_business_rules_task = PythonOperator(
        task_id='validate_orders_business_rules',
        python_callable=validate_orders_business_rules,
        provide_context=True
    )

    # Customers validation tasks
    validate_customers_schema_task = PythonOperator(
        task_id='validate_customers_schema',
        python_callable=validate_customers_schema,
        provide_context=True
    )

    validate_customers_quality_task = PythonOperator(
        task_id='validate_customers_quality',
        python_callable=validate_customers_quality,
        provide_context=True
    )

    # Cross-table validation
    validate_referential_integrity_task = PythonOperator(
        task_id='validate_referential_integrity',
        python_callable=validate_referential_integrity,
        provide_context=True
    )

    # Final report
    generate_report_task = PythonOperator(
        task_id='generate_comprehensive_report',
        python_callable=generate_comprehensive_report,
        provide_context=True
    )

    # Define task dependencies
    # All validations run in parallel after directory creation
    create_dirs >> [
        validate_orders_schema_task,
        validate_orders_completeness_task,
        validate_orders_uniqueness_task,
        validate_orders_ranges_task,
        validate_orders_categorical_task,
        validate_orders_business_rules_task,
        validate_customers_schema_task,
        validate_customers_quality_task,
        validate_referential_integrity_task
    ] >> generate_report_task
