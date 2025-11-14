# Lab 2.3: Data Quality Checks

**Goal**: Implement comprehensive data quality validation with Great Expectations patterns

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Lab 2.1 and 2.2 completed
- Understanding of data validation concepts
- Familiarity with statistical measures

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Implement multi-level data quality checks
- ✅ Use Great Expectations patterns (without the full library)
- ✅ Create reusable validation functions
- ✅ Handle validation failures gracefully
- ✅ Generate data quality reports
- ✅ Set up monitoring and alerting patterns

---

## Background: Data Quality Framework

### Why Data Quality Matters

**In production ML systems, data quality is MORE important than model quality.**

Problems caused by poor data quality:
- ❌ Silent model degradation
- ❌ Incorrect predictions
- ❌ Training failures
- ❌ Wasted compute resources
- ❌ Lost business value

Benefits of good data quality:
- ✅ Catch issues early (before training)
- ✅ Build trust in the system
- ✅ Enable debugging
- ✅ Document data expectations
- ✅ Detect drift over time

### Levels of Data Quality Checks

```
┌────────────────────────────────────────────────┐
│         DATA QUALITY PYRAMID                   │
│                                                │
│              🔺 SEMANTIC                        │
│             (Business Logic)                   │
│         Does the data make sense?              │
│                                                │
│           🔺🔺 STATISTICAL                      │
│         (Distributions, Ranges)                │
│      Are values within expected ranges?        │
│                                                │
│        🔺🔺🔺 SCHEMA                            │
│      (Structure, Types, Nulls)                 │
│    Are columns and types correct?              │
│                                                │
│     🔺🔺🔺🔺 EXISTENCE                          │
│    (File/Data Availability)                    │
│  Does the data exist at all?                   │
│                                                │
└────────────────────────────────────────────────┘
```

### Great Expectations Pattern

[Great Expectations](https://greatexpectations.io/) is a popular Python library for data validation. We'll implement its core patterns:

1. **Expectations**: Assertions about data
2. **Validation Results**: Pass/fail + metadata
3. **Data Docs**: Human-readable reports
4. **Checkpoints**: Reusable validation suites

---

## Part 1: Build Validation Framework

Create `scripts/data_quality.py`:

```python
"""
Data Quality Validation Framework

Implements Great Expectations patterns for data validation.
Provides reusable validation functions and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import json
import logging


class ValidationResult:
    """Result of a single validation check."""

    def __init__(
        self,
        expectation: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        severity: str = 'error'
    ):
        self.expectation = expectation
        self.success = success
        self.details = details or {}
        self.severity = severity  # 'error' or 'warning'
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'expectation': self.expectation,
            'success': self.success,
            'details': self.details,
            'severity': self.severity,
            'timestamp': self.timestamp
        }


class DataValidator:
    """
    Data validation class implementing Great Expectations patterns.

    Usage:
        validator = DataValidator(df)
        validator.expect_column_exists('user_id')
        validator.expect_column_values_not_null('user_id')
        results = validator.get_results()
    """

    def __init__(self, df: pd.DataFrame, name: str = 'dataset'):
        self.df = df
        self.name = name
        self.results: List[ValidationResult] = []

    def expect_column_to_exist(self, column: str) -> ValidationResult:
        """Expect column to exist in dataframe."""
        success = column in self.df.columns
        result = ValidationResult(
            expectation=f"expect_column_to_exist({column})",
            success=success,
            details={'column': column, 'existing_columns': list(self.df.columns)},
            severity='error'
        )
        self.results.append(result)
        return result

    def expect_column_values_to_not_be_null(
        self,
        column: str,
        threshold: float = 0.0
    ) -> ValidationResult:
        """
        Expect column to have no null values (or below threshold).

        Args:
            column: Column name
            threshold: Max allowed null percentage (0.0 = no nulls)
        """
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        null_count = self.df[column].isnull().sum()
        null_pct = (null_count / len(self.df)) * 100
        success = null_pct <= threshold

        result = ValidationResult(
            expectation=f"expect_column_values_to_not_be_null({column})",
            success=success,
            details={
                'column': column,
                'null_count': int(null_count),
                'null_percentage': float(null_pct),
                'threshold': threshold
            },
            severity='error' if threshold == 0.0 else 'warning'
        )
        self.results.append(result)
        return result

    def expect_column_values_to_be_unique(self, column: str) -> ValidationResult:
        """Expect column values to be unique (no duplicates)."""
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        duplicate_count = self.df[column].duplicated().sum()
        success = duplicate_count == 0

        result = ValidationResult(
            expectation=f"expect_column_values_to_be_unique({column})",
            success=success,
            details={
                'column': column,
                'duplicate_count': int(duplicate_count),
                'unique_count': int(self.df[column].nunique()),
                'total_count': len(self.df)
            },
            severity='error'
        )
        self.results.append(result)
        return result

    def expect_column_values_to_be_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        strict: bool = False
    ) -> ValidationResult:
        """
        Expect column values to be within a range.

        Args:
            column: Column name
            min_value: Minimum allowed value (inclusive unless strict=True)
            max_value: Maximum allowed value (inclusive unless strict=True)
            strict: If True, use strict inequalities (< and >)
        """
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        values = self.df[column].dropna()

        if strict:
            violations = values[(values <= min_value) | (values >= max_value)] if min_value is not None and max_value is not None else pd.Series()
        else:
            violations = values[(values < min_value) | (values > max_value)] if min_value is not None and max_value is not None else pd.Series()

        if min_value is not None and max_value is None:
            violations = values[values < min_value] if not strict else values[values <= min_value]
        elif max_value is not None and min_value is None:
            violations = values[values > max_value] if not strict else values[values >= max_value]

        violation_count = len(violations)
        success = violation_count == 0

        result = ValidationResult(
            expectation=f"expect_column_values_to_be_between({column}, {min_value}, {max_value})",
            success=success,
            details={
                'column': column,
                'min_value': min_value,
                'max_value': max_value,
                'violation_count': int(violation_count),
                'min_observed': float(values.min()),
                'max_observed': float(values.max())
            },
            severity='error'
        )
        self.results.append(result)
        return result

    def expect_column_values_to_be_in_set(
        self,
        column: str,
        value_set: List[Any]
    ) -> ValidationResult:
        """Expect column values to be in a defined set."""
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        invalid_values = self.df[~self.df[column].isin(value_set)][column].dropna()
        success = len(invalid_values) == 0

        result = ValidationResult(
            expectation=f"expect_column_values_to_be_in_set({column})",
            success=success,
            details={
                'column': column,
                'allowed_values': value_set,
                'invalid_count': int(len(invalid_values)),
                'invalid_values': list(invalid_values.unique()[:10])  # Sample
            },
            severity='error'
        )
        self.results.append(result)
        return result

    def expect_column_mean_to_be_between(
        self,
        column: str,
        min_value: float,
        max_value: float
    ) -> ValidationResult:
        """Expect column mean to be within a range (statistical check)."""
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        mean_value = self.df[column].mean()
        success = min_value <= mean_value <= max_value

        result = ValidationResult(
            expectation=f"expect_column_mean_to_be_between({column}, {min_value}, {max_value})",
            success=success,
            details={
                'column': column,
                'observed_mean': float(mean_value),
                'expected_min': min_value,
                'expected_max': max_value
            },
            severity='warning'  # Statistical checks are warnings
        )
        self.results.append(result)
        return result

    def expect_table_row_count_to_be_between(
        self,
        min_value: int,
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """Expect table to have row count within a range."""
        row_count = len(self.df)

        if max_value is None:
            success = row_count >= min_value
        else:
            success = min_value <= row_count <= max_value

        result = ValidationResult(
            expectation=f"expect_table_row_count_to_be_between({min_value}, {max_value})",
            success=success,
            details={
                'row_count': row_count,
                'min_expected': min_value,
                'max_expected': max_value
            },
            severity='error'
        )
        self.results.append(result)
        return result

    def expect_column_stdev_to_be_between(
        self,
        column: str,
        min_value: float,
        max_value: float
    ) -> ValidationResult:
        """Expect column standard deviation to be within a range."""
        if column not in self.df.columns:
            return self.expect_column_to_exist(column)

        std_value = self.df[column].std()
        success = min_value <= std_value <= max_value

        result = ValidationResult(
            expectation=f"expect_column_stdev_to_be_between({column}, {min_value}, {max_value})",
            success=success,
            details={
                'column': column,
                'observed_std': float(std_value),
                'expected_min': min_value,
                'expected_max': max_value
            },
            severity='warning'
        )
        self.results.append(result)
        return result

    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        errors = [r for r in self.results if not r.success and r.severity == 'error']
        warnings = [r for r in self.results if not r.success and r.severity == 'warning']

        return {
            'dataset': self.name,
            'total_expectations': total,
            'passed': passed,
            'failed': failed,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'errors': len(errors),
            'warnings': len(warnings),
            'timestamp': datetime.now().isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export validation results as dictionary."""
        return {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results]
        }

    def save_report(self, output_path: str):
        """Save validation report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Validation report saved to {output_path}")


def validate_transactions_schema(df: pd.DataFrame) -> DataValidator:
    """
    Validate transaction data schema and basic constraints.

    Returns:
        DataValidator with all validation results
    """
    validator = DataValidator(df, name='transactions')

    # Schema validations
    required_columns = [
        'transaction_id', 'customer_id', 'product_id',
        'quantity', 'price', 'timestamp', 'total_amount'
    ]

    for col in required_columns:
        validator.expect_column_to_exist(col)

    # Null checks (critical fields)
    validator.expect_column_values_to_not_be_null('transaction_id')
    validator.expect_column_values_to_not_be_null('customer_id')
    validator.expect_column_values_to_not_be_null('product_id')
    validator.expect_column_values_to_not_be_null('price')
    validator.expect_column_values_to_not_be_null('quantity')

    # Allow some nulls in payment_method (5% threshold)
    validator.expect_column_values_to_not_be_null('payment_method', threshold=5.0)

    # Uniqueness
    validator.expect_column_values_to_be_unique('transaction_id')

    # Value ranges
    validator.expect_column_values_to_be_between('price', min_value=0, max_value=10000)
    validator.expect_column_values_to_be_between('quantity', min_value=1, max_value=100)
    validator.expect_column_values_to_be_between('total_amount', min_value=0, max_value=100000)

    # Categorical values
    valid_payment_methods = ['credit_card', 'debit_card', 'paypal', 'crypto', 'unknown']
    validator.expect_column_values_to_be_in_set('payment_method', valid_payment_methods)

    # Table-level checks
    validator.expect_table_row_count_to_be_between(min_value=100)  # At least 100 transactions

    return validator


def validate_transactions_statistics(
    df: pd.DataFrame,
    reference_stats: Optional[Dict] = None
) -> DataValidator:
    """
    Validate transaction data statistics (for drift detection).

    Args:
        df: DataFrame to validate
        reference_stats: Historical statistics to compare against
    """
    validator = DataValidator(df, name='transactions_statistics')

    # Default reference stats if not provided
    if reference_stats is None:
        reference_stats = {
            'price_mean': (20, 200),
            'price_std': (10, 150),
            'quantity_mean': (1, 10),
            'total_amount_mean': (50, 500),
        }

    # Statistical checks
    if 'price_mean' in reference_stats:
        min_val, max_val = reference_stats['price_mean']
        validator.expect_column_mean_to_be_between('price', min_val, max_val)

    if 'price_std' in reference_stats:
        min_val, max_val = reference_stats['price_std']
        validator.expect_column_stdev_to_be_between('price', min_val, max_val)

    if 'quantity_mean' in reference_stats:
        min_val, max_val = reference_stats['quantity_mean']
        validator.expect_column_mean_to_be_between('quantity', min_val, max_val)

    if 'total_amount_mean' in reference_stats:
        min_val, max_val = reference_stats['total_amount_mean']
        validator.expect_column_mean_to_be_between('total_amount', min_val, max_val)

    return validator
```

---

## Part 2: Integrate Validation into DAG

Create `dags/data_quality_pipeline.py`:

```python
"""
Data Quality Pipeline

Comprehensive data quality checks with:
- Schema validation
- Statistical validation
- Drift detection
- Quality reporting
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import logging

# Add scripts to path
sys.path.append('/home/user/mlops-learning-plan')
from scripts.data_quality import (
    DataValidator,
    validate_transactions_schema,
    validate_transactions_statistics
)


default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


@dag(
    dag_id='data_quality_pipeline',
    default_args=default_args,
    description='Comprehensive data quality validation pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data-quality', 'phase2'],
)
def data_quality_pipeline():
    """Data Quality Validation DAG"""

    @task
    def validate_raw_data_schema(ds=None):
        """Validate raw transaction data schema."""
        input_path = f"data/raw/{ds}/transactions.csv"

        logging.info(f"Validating schema for {input_path}")

        df = pd.read_csv(input_path, parse_dates=['timestamp'])

        # Run schema validation
        validator = DataValidator(df, name='raw_transactions')

        # Schema checks
        required_columns = [
            'transaction_id', 'customer_id', 'product_id',
            'quantity', 'price', 'timestamp', 'payment_method'
        ]

        for col in required_columns:
            validator.expect_column_to_exist(col)

        # Basic constraints
        validator.expect_table_row_count_to_be_between(min_value=50)
        validator.expect_column_values_to_not_be_null('transaction_id')
        validator.expect_column_values_to_not_be_null('customer_id')

        # Save report
        report_dir = f"logs/data_quality/{ds}"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/raw_schema_validation.json"
        validator.save_report(report_path)

        # Get summary
        summary = validator.get_summary()
        logging.info(f"Schema validation: {summary['passed']}/{summary['total_expectations']} passed")

        # Fail if there are errors
        errors = [r for r in validator.get_results() if not r.success and r.severity == 'error']
        if errors:
            error_msgs = [f"- {r.expectation}: {r.details}" for r in errors]
            raise ValueError(f"Schema validation failed:\n" + "\n".join(error_msgs))

        return {
            'execution_date': ds,
            'report_path': report_path,
            'summary': summary
        }

    @task
    def validate_processed_data_quality(ds=None):
        """Validate processed transaction data quality."""
        input_path = f"data/processed/{ds}/transactions_clean.parquet"

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Processed data not found: {input_path}")

        logging.info(f"Validating quality for {input_path}")

        df = pd.read_parquet(input_path)

        # Run comprehensive validation
        validator = validate_transactions_schema(df)

        # Save report
        report_dir = f"logs/data_quality/{ds}"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/processed_quality_validation.json"
        validator.save_report(report_path)

        # Get summary
        summary = validator.get_summary()
        logging.info(f"Quality validation: {summary['passed']}/{summary['total_expectations']} passed")
        logging.info(f"Errors: {summary['errors']}, Warnings: {summary['warnings']}")

        # Fail if there are errors (warnings are OK)
        errors = [r for r in validator.get_results() if not r.success and r.severity == 'error']
        if errors:
            error_msgs = [f"- {r.expectation}" for r in errors]
            raise ValueError(f"Quality validation failed with {len(errors)} errors:\n" + "\n".join(error_msgs))

        # Log warnings
        warnings = [r for r in validator.get_results() if not r.success and r.severity == 'warning']
        if warnings:
            warning_msgs = [f"- {r.expectation}" for r in warnings]
            logging.warning(f"Quality validation warnings ({len(warnings)}):\n" + "\n".join(warning_msgs))

        return {
            'execution_date': ds,
            'report_path': report_path,
            'summary': summary
        }

    @task
    def validate_statistics(ds=None, reference_date: str = None):
        """
        Validate statistical properties (for drift detection).

        Compares current data statistics to reference date.
        """
        current_path = f"data/processed/{ds}/transactions_clean.parquet"
        current_df = pd.read_parquet(current_path)

        # Load reference statistics
        if reference_date:
            reference_path = f"data/processed/{reference_date}/transactions_clean.parquet"
            if os.path.exists(reference_path):
                ref_df = pd.read_parquet(reference_path)

                # Compute reference stats with tolerance bands
                reference_stats = {
                    'price_mean': (
                        ref_df['price'].mean() * 0.8,
                        ref_df['price'].mean() * 1.2
                    ),
                    'price_std': (
                        ref_df['price'].std() * 0.7,
                        ref_df['price'].std() * 1.3
                    ),
                    'quantity_mean': (
                        ref_df['quantity'].mean() * 0.8,
                        ref_df['quantity'].mean() * 1.2
                    ),
                }
            else:
                logging.warning(f"Reference date {reference_date} not found, using defaults")
                reference_stats = None
        else:
            reference_stats = None

        # Run statistical validation
        validator = validate_transactions_statistics(current_df, reference_stats)

        # Save report
        report_dir = f"logs/data_quality/{ds}"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/statistical_validation.json"
        validator.save_report(report_path)

        summary = validator.get_summary()
        logging.info(f"Statistical validation: {summary['passed']}/{summary['total_expectations']} passed")

        # Statistical checks are warnings, not errors
        # We log but don't fail the task
        failures = [r for r in validator.get_results() if not r.success]
        if failures:
            failure_msgs = [f"- {r.expectation}: observed={r.details}" for r in failures]
            logging.warning(f"Statistical drift detected ({len(failures)} checks):\n" + "\n".join(failure_msgs))

        return {
            'execution_date': ds,
            'reference_date': reference_date,
            'report_path': report_path,
            'summary': summary,
            'drift_detected': len(failures) > 0
        }

    @task
    def generate_quality_summary(schema_meta: dict, quality_meta: dict, stats_meta: dict):
        """Generate overall data quality summary report."""
        execution_date = schema_meta['execution_date']

        summary = {
            'execution_date': execution_date,
            'timestamp': datetime.now().isoformat(),
            'validations': {
                'schema': schema_meta['summary'],
                'quality': quality_meta['summary'],
                'statistics': stats_meta['summary'],
            },
            'overall_status': 'PASSED',
            'drift_detected': stats_meta['drift_detected']
        }

        # Determine overall status
        if quality_meta['summary']['errors'] > 0:
            summary['overall_status'] = 'FAILED'
        elif quality_meta['summary']['warnings'] > 0 or stats_meta['drift_detected']:
            summary['overall_status'] = 'PASSED_WITH_WARNINGS'

        # Save summary
        output_path = f"logs/data_quality/{execution_date}/summary.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Overall status: {summary['overall_status']}")
        logging.info(f"Quality summary saved to {output_path}")

        return summary

    # Define task dependencies
    schema_meta = validate_raw_data_schema()
    quality_meta = validate_processed_data_quality()
    stats_meta = validate_statistics()

    summary = generate_quality_summary(schema_meta, quality_meta, stats_meta)


# Instantiate DAG
quality_dag = data_quality_pipeline()
```

---

## Part 3: Run and Test

### Step 1: Ensure Data Exists

```bash
# Run ETL if needed
airflow dags trigger etl_ecommerce_pipeline --conf '{"ds": "2024-01-15"}'

# Wait for completion, then run quality checks
airflow dags trigger data_quality_pipeline --conf '{"ds": "2024-01-15"}'
```

### Step 2: View Quality Reports

```bash
# View reports
cat logs/data_quality/2024-01-15/summary.json

# View detailed validations
cat logs/data_quality/2024-01-15/processed_quality_validation.json | jq '.results[] | select(.success == false)'
```

---

## Exercise 1: Add Custom Validation

Create a custom validation for business logic:

```python
def expect_total_amount_equals_price_times_quantity(
    self,
    price_col: str = 'price',
    quantity_col: str = 'quantity',
    total_col: str = 'total_amount',
    tolerance: float = 0.01
) -> ValidationResult:
    """
    Expect total_amount = price × quantity (within tolerance).

    Business logic validation.
    """
    calculated = self.df[price_col] * self.df[quantity_col]
    actual = self.df[total_col]

    diff = (actual - calculated).abs()
    violations = diff > tolerance

    success = violations.sum() == 0

    result = ValidationResult(
        expectation="expect_total_amount_equals_price_times_quantity",
        success=success,
        details={
            'violation_count': int(violations.sum()),
            'max_difference': float(diff.max())
        },
        severity='error'
    )
    self.results.append(result)
    return result
```

Add this method to the `DataValidator` class and use it in your validation pipeline.

---

## Exercise 2: Drift Detection Dashboard

Create a script that compares quality metrics across multiple dates:

```python
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Load all quality reports
reports = []
for path in sorted(glob.glob('logs/data_quality/*/summary.json')):
    with open(path) as f:
        report = json.load(f)
        reports.append(report)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'date': r['execution_date'],
        'passed': r['validations']['quality']['passed'],
        'failed': r['validations']['quality']['failed'],
        'errors': r['validations']['quality']['errors'],
        'warnings': r['validations']['quality']['warnings'],
        'drift': r['drift_detected']
    }
    for r in reports
])

# Plot quality metrics over time
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Success rate
df['success_rate'] = df['passed'] / (df['passed'] + df['failed']) * 100
axes[0].plot(df['date'], df['success_rate'], marker='o')
axes[0].set_title('Data Quality Success Rate Over Time')
axes[0].set_ylabel('Success Rate (%)')
axes[0].axhline(y=95, color='r', linestyle='--', label='Threshold')
axes[0].legend()

# Errors and warnings
axes[1].bar(df['date'], df['errors'], label='Errors', color='red', alpha=0.7)
axes[1].bar(df['date'], df['warnings'], bottom=df['errors'], label='Warnings', color='orange', alpha=0.7)
axes[1].set_title('Validation Errors and Warnings')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('logs/data_quality/quality_dashboard.png')
print("Dashboard saved to logs/data_quality/quality_dashboard.png")
```

---

## Exercise 3: Automated Alerting

Add alerting logic that sends notifications (simulated) when quality degrades:

```python
@task
def check_and_alert(summary: dict):
    """
    Check quality summary and send alerts if needed.

    In production, this would:
    - Send Slack/email notifications
    - Create Jira tickets
    - Update monitoring dashboards
    """
    status = summary['overall_status']
    execution_date = summary['execution_date']

    if status == 'FAILED':
        # Simulate alert
        alert_message = f"""
        🚨 DATA QUALITY ALERT 🚨

        Date: {execution_date}
        Status: {status}
        Errors: {summary['validations']['quality']['errors']}

        Action Required: Review validation reports and fix data issues.
        Report: logs/data_quality/{execution_date}/summary.json
        """

        logging.error(alert_message)

        # In production:
        # send_slack_alert(alert_message)
        # send_email_alert(alert_message)
        # create_jira_ticket(summary)

        raise ValueError("Data quality checks failed - requires immediate attention")

    elif status == 'PASSED_WITH_WARNINGS':
        warning_message = f"""
        ⚠️ DATA QUALITY WARNING ⚠️

        Date: {execution_date}
        Warnings: {summary['validations']['quality']['warnings']}
        Drift Detected: {summary['drift_detected']}

        Action: Monitor for continued degradation.
        """

        logging.warning(warning_message)

        # In production:
        # send_slack_notification(warning_message)

    else:
        logging.info(f"✅ Data quality checks passed for {execution_date}")

    return summary
```

---

## Challenge: Custom Great Expectations Suite

Implement a complete validation suite for the feature dataset:

1. **Load features** from `data/features/v1/train.parquet`
2. **Create validation suite** with:
   - Schema checks (all expected columns exist)
   - Null checks (no nulls in features)
   - Range checks (normalized features between -5 and 5)
   - Distribution checks (mean ≈ 0, std ≈ 1 for normalized features)
3. **Save detailed report**
4. **Visualize** validation results

---

## Key Takeaways

### Data Quality Best Practices

✅ **Validate early**: Check raw data before processing
✅ **Multiple levels**: Schema → values → statistics → business logic
✅ **Fail fast**: Stop pipeline on critical errors
✅ **Warn on drift**: Don't fail on statistical changes, but alert
✅ **Document expectations**: Clear, versioned validation rules
✅ **Generate reports**: Human-readable validation summaries
✅ **Track over time**: Monitor quality trends

### Validation Strategy

| Check Type | Severity | Action |
|------------|----------|--------|
| Schema mismatch | Error | Fail task |
| Missing critical values | Error | Fail task |
| Invalid value ranges | Error | Fail task |
| Statistical drift | Warning | Log and continue |
| Distribution changes | Warning | Log and continue |

### Great Expectations Patterns

1. **Expectations**: Clear, testable assertions
2. **Validators**: Reusable validation logic
3. **Results**: Structured pass/fail with metadata
4. **Reports**: JSON/HTML summaries
5. **Suites**: Collections of related checks

---

## Debugging Tips

### All Validations Failing

```bash
# Check data exists
ls data/processed/2024-01-15/

# Check data can be loaded
python -c "import pandas as pd; print(pd.read_parquet('data/processed/2024-01-15/transactions_clean.parquet').head())"

# Run individual validation
airflow tasks test data_quality_pipeline validate_processed_data_quality 2024-01-15
```

### Understanding Validation Failures

```python
# Load validation report
import json
with open('logs/data_quality/2024-01-15/processed_quality_validation.json') as f:
    report = json.load(f)

# View failures only
failures = [r for r in report['results'] if not r['success']]
for f in failures:
    print(f"{f['expectation']}: {f['details']}")
```

---

## Submission Checklist

Before moving to Lab 2.4:

- ✅ Data quality DAG runs successfully
- ✅ Validation reports generated
- ✅ Understand difference between errors and warnings
- ✅ Can interpret validation results
- ✅ At least one exercise completed

---

## Next Steps

**What you've built**:
- Comprehensive data quality framework
- Great Expectations patterns
- Multi-level validation (schema, values, statistics)
- Quality reporting and monitoring

**Next lab**:
- Schedule pipeline to run daily
- Implement date-based partitioning
- Handle incremental processing
- Perform backfills

---

**Congratulations on building production-grade data quality checks!** 🎉

**Next**: [Lab 2.4 - Scheduled Partitioned Pipeline →](./lab2_4_scheduled_pipeline.md)
