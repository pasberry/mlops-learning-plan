# Lab 2.4: Scheduled Partitioned Pipeline

**Goal**: Build a production-ready scheduled pipeline with date partitioning and backfill support

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Labs 2.1, 2.2, and 2.3 completed
- Understanding of Airflow scheduling
- Familiarity with date partitioning

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Schedule DAGs to run automatically
- ✅ Implement date-based data partitioning
- ✅ Build incremental vs full refresh pipelines
- ✅ Perform backfills for historical data
- ✅ Handle late-arriving data
- ✅ Use Airflow macros and templating
- ✅ Integrate all Phase 2 components into one pipeline

---

## Background: Production Data Pipelines

### Why Scheduling Matters

In production, ML systems need fresh data continuously:

```
Day 1: New data arrives → Process → Train model
Day 2: New data arrives → Process → Retrain model
Day 3: New data arrives → Process → Retrain model
...
```

**Manual triggering doesn't scale.** You need automated, scheduled pipelines.

### Date Partitioning

**Partitioning** = Organizing data by date/time for efficient processing.

```
data/
  raw/
    2024-01-01/
      transactions.csv
    2024-01-02/
      transactions.csv
    2024-01-03/
      transactions.csv
```

**Benefits**:
- ✅ Process only new data (incremental)
- ✅ Parallel processing of partitions
- ✅ Easy to reprocess specific dates
- ✅ Clear data lineage
- ✅ Efficient storage and queries

### Full Refresh vs Incremental

**Full Refresh**:
```python
# Process ALL data every time
df = load_all_historical_data()
df_processed = transform(df)
save(df_processed, 'output/full.parquet')
```

**Pros**: Simple, always consistent
**Cons**: Slow, expensive, doesn't scale

**Incremental**:
```python
# Process only today's data
df = load_data_for_date(execution_date)
df_processed = transform(df)
save(df_processed, f'output/{execution_date}/data.parquet')
```

**Pros**: Fast, scalable, efficient
**Cons**: More complex, need to handle late data

### Airflow Scheduling Concepts

#### 1. schedule_interval

How often the DAG runs:

```python
# Common schedules
schedule='@daily'      # Every day at midnight
schedule='@hourly'     # Every hour
schedule='0 9 * * *'   # Every day at 9 AM (cron expression)
schedule='*/30 * * * *'  # Every 30 minutes
```

#### 2. Execution Date vs Run Date

- **execution_date**: The date the DAG is processing (data date)
- **run_date**: When the DAG actually runs

Example:
```
Schedule: @daily
Run at: 2024-01-16 00:00:00
Execution date: 2024-01-15 (processes yesterday's data)
```

**Why?** Because data for a date is only complete AFTER the date ends.

#### 3. Catchup and Backfills

**catchup=True**: Run all missed DAG runs since start_date
**catchup=False**: Only run from now forward (skip history)

```python
# Example with catchup
DAG(
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=True  # Will run for all dates from Jan 1 to today
)
```

**Backfill**: Manually reprocess historical dates

```bash
# Backfill specific range
airflow dags backfill my_dag \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

---

## Part 1: Convert ETL Pipeline to Scheduled

### Step 1: Update ETL DAG with Scheduling

Modify `dags/etl_pipeline.py`:

```python
"""
Scheduled ETL Pipeline

Runs daily to process new transaction data.
Uses date partitioning for incremental processing.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
import json


default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,  # Don't wait for previous runs
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='etl_ecommerce_scheduled',
    default_args=default_args,
    description='Scheduled ETL pipeline with date partitioning',
    schedule='0 1 * * *',  # Run daily at 1 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill automatically
    max_active_runs=3,  # Allow max 3 concurrent runs
    tags=['etl', 'scheduled', 'production'],
)
def etl_scheduled_pipeline():
    """Scheduled ETL DAG"""

    @task
    def check_data_availability(ds=None, **context):
        """
        Check if raw data is available for processing.

        In production, this would check if upstream data has landed.
        For this lab, we'll generate it if missing.
        """
        execution_date = ds
        data_path = f"data/raw/{execution_date}/transactions.csv"

        logging.info(f"Checking data availability for {execution_date}")

        if os.path.exists(data_path):
            logging.info(f"✓ Data exists: {data_path}")
            return {
                'execution_date': execution_date,
                'data_available': True,
                'data_path': data_path
            }
        else:
            logging.warning(f"✗ Data not found: {data_path}")
            return {
                'execution_date': execution_date,
                'data_available': False,
                'data_path': data_path
            }

    @task
    def generate_or_skip(check_result: dict):
        """
        Generate data if not available, or skip if already exists.
        """
        import sys
        sys.path.append('/home/user/mlops-learning-plan')

        execution_date = check_result['execution_date']

        if check_result['data_available']:
            logging.info(f"Data already exists for {execution_date}, skipping generation")
            return check_result

        # Generate data
        logging.info(f"Generating data for {execution_date}")

        from scripts.generate_ecommerce_data import main as generate
        result = generate(execution_date=execution_date, output_dir='data/raw')

        return {
            'execution_date': execution_date,
            'data_available': True,
            'data_path': check_result['data_path'],
            'generated': True,
            'num_transactions': result['num_transactions']
        }

    @task
    def process_daily_data(metadata: dict):
        """
        Process data for the execution date.

        This is an incremental processing task - only handles one day.
        """
        execution_date = metadata['execution_date']
        input_path = f"data/raw/{execution_date}/transactions.csv"

        logging.info(f"Processing data for {execution_date}")

        # Read data
        df = pd.read_csv(input_path, parse_dates=['timestamp'])

        # Basic validation
        required_cols = ['transaction_id', 'customer_id', 'product_id', 'price', 'quantity']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Clean data
        initial_rows = len(df)

        df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        df = df[df['price'] >= 0]
        df = df[df['quantity'] > 0]
        df['payment_method'] = df['payment_method'].fillna('unknown')

        # Add derived columns
        df['total_amount'] = df['price'] * df['quantity']
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour

        final_rows = len(df)

        logging.info(f"Cleaned: {initial_rows} → {final_rows} rows")

        # Write to processed layer (partitioned by date)
        output_dir = f"data/processed/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/transactions_clean.parquet"

        df.to_parquet(output_path, index=False)

        logging.info(f"✓ Processed data saved to {output_path}")

        return {
            'execution_date': execution_date,
            'input_rows': initial_rows,
            'output_rows': final_rows,
            'output_path': output_path
        }

    @task
    def update_metadata_registry(process_result: dict):
        """
        Update metadata registry with processing results.

        Maintains a log of all processed dates for tracking.
        """
        execution_date = process_result['execution_date']

        registry_path = "data/processed/processing_registry.jsonl"

        # Append to registry (JSONL format - one JSON per line)
        registry_entry = {
            'execution_date': execution_date,
            'processed_at': datetime.now().isoformat(),
            'input_rows': process_result['input_rows'],
            'output_rows': process_result['output_rows'],
            'output_path': process_result['output_path']
        }

        with open(registry_path, 'a') as f:
            f.write(json.dumps(registry_entry) + '\n')

        logging.info(f"✓ Registry updated: {registry_path}")

        return registry_entry

    # Task dependencies
    availability = check_data_availability()
    generated = generate_or_skip(availability)
    processed = process_daily_data(generated)
    metadata = update_metadata_registry(processed)


# Instantiate DAG
etl_scheduled_dag = etl_scheduled_pipeline()
```

---

## Part 2: Build End-to-End Scheduled Pipeline

Create `dags/end_to_end_scheduled_pipeline.py`:

```python
"""
End-to-End Scheduled Data Pipeline

Combines:
1. ETL (extract, validate, clean)
2. Data Quality checks
3. Feature Engineering
4. Metadata tracking

Runs daily to process new data and prepare features for training.
"""

from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging
import json
import sys

sys.path.append('/home/user/mlops-learning-plan')


default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': 'data-team@example.com',  # Would be real email in production
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='end_to_end_data_pipeline',
    default_args=default_args,
    description='Complete scheduled data pipeline from raw to features',
    schedule='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Prevent concurrent runs
    tags=['production', 'end-to-end', 'scheduled'],
)
def end_to_end_pipeline():
    """End-to-End Data Pipeline"""

    # ========== INGESTION STAGE ==========

    @task
    def ingest_raw_data(ds=None):
        """Ingest raw data for execution date."""
        import sys
        sys.path.append('/home/user/mlops-learning-plan')
        from scripts.generate_ecommerce_data import main as generate

        execution_date = ds
        data_path = f"data/raw/{execution_date}/transactions.csv"

        # Check if data exists
        if os.path.exists(data_path):
            logging.info(f"Data already exists: {data_path}")
        else:
            logging.info(f"Generating data for {execution_date}")
            generate(execution_date=execution_date, output_dir='data/raw')

        # Load and return metadata
        df = pd.read_csv(data_path)

        return {
            'execution_date': execution_date,
            'num_rows': len(df),
            'data_path': data_path
        }

    # ========== VALIDATION STAGE ==========

    @task
    def validate_raw_data(ingest_meta: dict):
        """Validate raw data quality."""
        from scripts.data_quality import DataValidator

        execution_date = ingest_meta['execution_date']
        df = pd.read_csv(ingest_meta['data_path'], parse_dates=['timestamp'])

        validator = DataValidator(df, name='raw_transactions')

        # Schema checks
        for col in ['transaction_id', 'customer_id', 'product_id', 'price', 'quantity']:
            validator.expect_column_to_exist(col)

        # Basic quality checks
        validator.expect_table_row_count_to_be_between(min_value=100)
        validator.expect_column_values_to_be_between('price', min_value=0, max_value=10000)

        # Save report
        report_dir = f"logs/data_quality/{execution_date}"
        os.makedirs(report_dir, exist_ok=True)
        validator.save_report(f"{report_dir}/raw_validation.json")

        summary = validator.get_summary()

        # Fail on errors
        if summary['errors'] > 0:
            raise ValueError(f"Validation failed with {summary['errors']} errors")

        return {
            'execution_date': execution_date,
            'validation_passed': True,
            'summary': summary
        }

    # ========== TRANSFORMATION STAGE ==========

    @task
    def clean_and_transform(validation_meta: dict):
        """Clean and transform data."""
        execution_date = validation_meta['execution_date']
        input_path = f"data/raw/{execution_date}/transactions.csv"

        df = pd.read_csv(input_path, parse_dates=['timestamp'])

        # Clean
        df = df.drop_duplicates(subset=['transaction_id'])
        df = df[df['price'] >= 0]
        df = df[df['quantity'] > 0]
        df['payment_method'] = df['payment_method'].fillna('unknown')

        # Transform
        df['total_amount'] = df['price'] * df['quantity']
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Save
        output_dir = f"data/processed/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/transactions_clean.parquet"

        df.to_parquet(output_path, index=False)

        return {
            'execution_date': execution_date,
            'num_rows': len(df),
            'output_path': output_path
        }

    # ========== FEATURE ENGINEERING STAGE ==========

    @task
    def compute_features(transform_meta: dict):
        """Compute features from cleaned data."""
        execution_date = transform_meta['execution_date']
        df = pd.read_parquet(transform_meta['output_path'])

        # Customer aggregations
        customer_features = df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean'],
        }).reset_index()

        customer_features.columns = [
            'customer_id',
            'customer_num_purchases',
            'customer_total_spent',
            'customer_avg_order_value'
        ]

        # Merge back
        df = df.merge(customer_features, on='customer_id', how='left')

        # Save features
        feature_dir = f"data/features/daily/{execution_date}"
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = f"{feature_dir}/features.parquet"

        df.to_parquet(feature_path, index=False)

        return {
            'execution_date': execution_date,
            'num_features': len(df.columns),
            'feature_path': feature_path
        }

    # ========== METADATA STAGE ==========

    @task
    def update_pipeline_metadata(feature_meta: dict):
        """Update pipeline metadata registry."""
        execution_date = feature_meta['execution_date']

        metadata = {
            'execution_date': execution_date,
            'pipeline_run_at': datetime.now().isoformat(),
            'num_features': feature_meta['num_features'],
            'feature_path': feature_meta['feature_path'],
            'status': 'success'
        }

        registry_path = "data/pipeline_registry.jsonl"
        with open(registry_path, 'a') as f:
            f.write(json.dumps(metadata) + '\n')

        logging.info(f"✅ Pipeline completed for {execution_date}")

        return metadata

    # ========== DEFINE PIPELINE FLOW ==========

    ingest_meta = ingest_raw_data()
    validate_meta = validate_raw_data(ingest_meta)
    transform_meta = clean_and_transform(validate_meta)
    feature_meta = compute_features(transform_meta)
    final_meta = update_pipeline_metadata(feature_meta)


# Instantiate DAG
end_to_end_dag = end_to_end_pipeline()
```

---

## Part 3: Working with Schedules and Backfills

### Enable and Monitor Scheduled DAG

```bash
# List DAGs
airflow dags list | grep end_to_end

# Unpause DAG (enable scheduling)
airflow dags unpause end_to_end_data_pipeline

# Check next run time
airflow dags next-execution end_to_end_data_pipeline

# View DAG info
airflow dags show end_to_end_data_pipeline
```

### Manual Trigger for Testing

```bash
# Trigger for specific date
airflow dags trigger end_to_end_data_pipeline \
    --exec-date 2024-01-15

# Trigger for today
airflow dags trigger end_to_end_data_pipeline
```

### Perform Backfill

```bash
# Backfill a date range
airflow dags backfill end_to_end_data_pipeline \
    --start-date 2024-01-01 \
    --end-date 2024-01-10 \
    --reset-dagruns  # Clear existing runs

# Backfill with concurrency limit
airflow dags backfill end_to_end_data_pipeline \
    --start-date 2024-01-01 \
    --end-date 2024-01-10 \
    --max-active-runs 2  # Process 2 dates at a time
```

### Clear and Rerun Specific Dates

```bash
# Clear a specific date
airflow dags clear end_to_end_data_pipeline \
    --start-date 2024-01-15 \
    --end-date 2024-01-15

# Clear and rerun
airflow dags clear end_to_end_data_pipeline \
    --start-date 2024-01-15 \
    --end-date 2024-01-15 \
    --yes  # Auto-confirm

# Then trigger will rerun automatically (if scheduled)
# Or manually trigger:
airflow dags trigger end_to_end_data_pipeline --exec-date 2024-01-15
```

---

## Part 4: Advanced Scheduling Patterns

### Pattern 1: Sensor-Based Triggering

Wait for upstream data before processing:

```python
from airflow.sensors.filesystem import FileSensor

@task
def wait_for_data(ds=None):
    """Wait for data file to appear."""
    from airflow.sensors.base import PokeReturnValue

    data_path = f"data/raw/{ds}/transactions.csv"

    if os.path.exists(data_path):
        return PokeReturnValue(is_done=True, xcom_value={'path': data_path})
    else:
        return PokeReturnValue(is_done=False)

# Or use built-in FileSensor (in non-TaskFlow API)
wait_task = FileSensor(
    task_id='wait_for_file',
    filepath='data/raw/{{ ds }}/transactions.csv',
    poke_interval=60,  # Check every 60 seconds
    timeout=3600,  # Give up after 1 hour
)
```

### Pattern 2: Conditional Processing

Process only if new data arrived:

```python
@task.branch
def check_if_data_changed(ds=None):
    """
    Check if data has changed since last run.
    If not, skip processing.
    """
    current_path = f"data/raw/{ds}/transactions.csv"
    previous_date = (pd.to_datetime(ds) - timedelta(days=1)).strftime('%Y-%m-%d')
    previous_path = f"data/raw/{previous_date}/transactions.csv"

    if not os.path.exists(current_path):
        return 'skip_processing'

    if not os.path.exists(previous_path):
        return 'process_data'  # First run

    # Compare file hashes or row counts
    current_size = os.path.getsize(current_path)
    previous_size = os.path.getsize(previous_path)

    if current_size == previous_size:
        return 'skip_processing'
    else:
        return 'process_data'

# Use branching
check = check_if_data_changed()
check >> [process_data_task, skip_task]
```

### Pattern 3: Dynamic Date Ranges

Process last N days of data:

```python
@task
def load_last_n_days(ds=None, n_days: int = 7):
    """
    Load data from last N days for aggregations.

    Useful for computing rolling features.
    """
    end_date = pd.to_datetime(ds)
    start_date = end_date - timedelta(days=n_days)

    logging.info(f"Loading data from {start_date} to {end_date}")

    all_data = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_path = f"data/processed/{date_str}/transactions_clean.parquet"

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            all_data.append(df)
            logging.info(f"Loaded {len(df)} rows from {date_str}")

        current_date += timedelta(days=1)

    if not all_data:
        raise FileNotFoundError(f"No data found between {start_date} and {end_date}")

    combined_df = pd.concat(all_data, ignore_index=True)

    logging.info(f"Combined: {len(combined_df)} total rows")

    return {
        'execution_date': ds,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'num_rows': len(combined_df)
    }
```

---

## Exercise 1: Add Late Data Handling

Handle data that arrives late:

```python
@task
def check_and_reprocess_late_data(ds=None, lookback_days: int = 3):
    """
    Check if data for previous days has been updated.
    If so, reprocess those dates.

    This handles late-arriving data.
    """
    execution_date = pd.to_datetime(ds)
    dates_to_reprocess = []

    for i in range(1, lookback_days + 1):
        check_date = execution_date - timedelta(days=i)
        date_str = check_date.strftime('%Y-%m-%d')

        raw_path = f"data/raw/{date_str}/transactions.csv"
        processed_path = f"data/processed/{date_str}/transactions_clean.parquet"

        if not os.path.exists(raw_path):
            continue

        # Check if raw is newer than processed
        if os.path.exists(processed_path):
            raw_mtime = os.path.getmtime(raw_path)
            processed_mtime = os.path.getmtime(processed_path)

            if raw_mtime > processed_mtime:
                logging.info(f"Late data detected for {date_str}")
                dates_to_reprocess.append(date_str)

    if dates_to_reprocess:
        logging.warning(f"Reprocessing {len(dates_to_reprocess)} dates: {dates_to_reprocess}")
        # Trigger reprocessing (could use TriggerDagRunOperator)

    return dates_to_reprocess
```

---

## Exercise 2: Implement SLA Monitoring

Add SLA (Service Level Agreement) monitoring:

```python
from airflow.models import TaskInstance

default_args = {
    ...
    'sla': timedelta(hours=2),  # Task should complete within 2 hours
}

@dag(
    ...
    sla_miss_callback=sla_miss_alert,  # Function to call on SLA miss
)
def my_dag():
    pass

def sla_miss_alert(dag, task_list, blocking_task_list, slas, blocking_tis):
    """
    Called when SLA is missed.

    Args:
        dag: DAG object
        task_list: Tasks that missed SLA
        blocking_task_list: Tasks blocking others
        slas: SLA configurations
        blocking_tis: Task instances blocking
    """
    logging.error(f"SLA MISSED for {dag.dag_id}")
    logging.error(f"Tasks: {[t.task_id for t in task_list]}")

    # Send alert
    # send_slack_alert(f"SLA missed for {dag.dag_id}")
```

---

## Exercise 3: Create Processing Dashboard

Build a dashboard showing pipeline runs:

```python
"""
Pipeline Monitoring Dashboard

Reads pipeline_registry.jsonl and generates a report.
"""

import pandas as pd
import json

# Load registry
records = []
with open('data/pipeline_registry.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Convert dates
df['execution_date'] = pd.to_datetime(df['execution_date'])
df['pipeline_run_at'] = pd.to_datetime(df['pipeline_run_at'])

# Analyze
print("Pipeline Runs Summary")
print("=" * 50)
print(f"Total runs: {len(df)}")
print(f"Date range: {df['execution_date'].min()} to {df['execution_date'].max()}")
print(f"Success rate: {(df['status'] == 'success').mean() * 100:.1f}%")

print("\nRecent runs:")
print(df.tail(10)[['execution_date', 'pipeline_run_at', 'num_features', 'status']])

# Check for missing dates
all_dates = pd.date_range(df['execution_date'].min(), df['execution_date'].max())
processed_dates = df['execution_date'].dt.date.unique()
missing_dates = [d for d in all_dates if d.date() not in processed_dates]

if missing_dates:
    print(f"\n⚠️ Missing dates: {len(missing_dates)}")
    print(missing_dates[:10])  # Show first 10
else:
    print("\n✅ No missing dates")
```

---

## Challenge: Multi-DAG Orchestration

Create a master DAG that orchestrates multiple pipelines:

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor

with DAG('master_orchestrator', ...) as dag:

    # Wait for upstream ETL to complete
    wait_for_etl = ExternalTaskSensor(
        task_id='wait_for_etl',
        external_dag_id='etl_ecommerce_scheduled',
        external_task_id='process_daily_data',
        timeout=600,
    )

    # Then trigger feature engineering
    trigger_features = TriggerDagRunOperator(
        task_id='trigger_feature_engineering',
        trigger_dag_id='feature_engineering_pipeline',
        wait_for_completion=True,
    )

    # Then trigger model training (Phase 3)
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_model_training',
        trigger_dag_id='model_training_pipeline',
        wait_for_completion=True,
    )

    wait_for_etl >> trigger_features >> trigger_training
```

---

## Key Takeaways

### Scheduling Best Practices

✅ **Set appropriate schedule**: Match data arrival frequency
✅ **Use catchup wisely**: False for new DAGs, True for backfills
✅ **Limit concurrency**: Prevent resource overload
✅ **Monitor SLAs**: Track pipeline performance
✅ **Handle late data**: Look back and reprocess if needed

### Partitioning Strategy

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| Daily | Most common | Simple, manageable | May be too coarse |
| Hourly | High-frequency data | Fine-grained | More partitions to manage |
| Full refresh | Small datasets | Always consistent | Expensive |
| Incremental | Large datasets | Efficient, scalable | More complex |

### Backfill Checklist

- ✅ Clear existing runs if needed (`--reset-dagruns`)
- ✅ Set appropriate concurrency (`--max-active-runs`)
- ✅ Monitor progress (UI or CLI)
- ✅ Verify output data for all dates
- ✅ Check for failures and rerun if needed

---

## Debugging Tips

### DAG Not Scheduled

```bash
# Check if paused
airflow dags list | grep my_dag

# Unpause
airflow dags unpause my_dag

# Check scheduler is running
ps aux | grep "airflow scheduler"

# View scheduler logs
tail -f logs/scheduler/latest/*.log
```

### Wrong Execution Date

```python
# Common mistake: Using current date instead of execution date
# WRONG
today = datetime.now().date()

# RIGHT
@task
def my_task(ds=None):  # ds = execution date string (YYYY-MM-DD)
    execution_date = ds
```

### Backfill Stuck

```bash
# Check running tasks
airflow tasks list-runs my_dag --state running

# Kill stuck tasks
airflow tasks kill my_dag task_id 2024-01-15

# Clear and retry
airflow dags clear my_dag --start-date 2024-01-15 --end-date 2024-01-15
```

---

## Submission Checklist

Before completing Phase 2:

- ✅ Scheduled DAG runs automatically
- ✅ Can perform backfills successfully
- ✅ Data properly partitioned by date
- ✅ Pipeline metadata tracked
- ✅ Understand incremental vs full refresh
- ✅ At least one exercise completed
- ✅ End-to-end pipeline from raw to features works

---

## Phase 2 Completion Criteria

You've completed Phase 2 when you can:

✅ **Design and implement multi-stage ETL pipelines**
✅ **Build comprehensive data validation frameworks**
✅ **Create feature engineering workflows**
✅ **Use date partitioning for incremental processing**
✅ **Schedule and backfill production pipelines**
✅ **Debug and monitor data pipeline failures**
✅ **Explain idempotency, data versioning, and quality patterns**

---

## Next Phase Preview: Phase 3

**Phase 3: Modeling & Training with PyTorch**

You'll learn to:
- Integrate PyTorch training into Airflow
- Build tabular and ranking models
- Implement experiment tracking
- Version and register models
- Create training pipelines

Your data pipelines from Phase 2 will feed directly into training pipelines!

---

## Resources

### Airflow Scheduling
- [Airflow Scheduling & Triggers](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#scheduling-and-triggers)
- [Backfills](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html#backfill)
- [Cron Expressions](https://crontab.guru/)

### Data Partitioning
- [Data Partitioning Best Practices](https://docs.databricks.com/delta/best-practices.html)
- [Hive-style Partitioning](https://arrow.apache.org/docs/python/parquet.html#partitioned-datasets-multiple-files)

### Monitoring
- [Airflow SLAs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html#slas)
- [Email Alerts](https://airflow.apache.org/docs/apache-airflow/stable/howto/email-config.html)

---

**Congratulations on completing Phase 2!** 🎉🎉🎉

**You've built a production-grade data pipeline system. This is a HUGE achievement!**

**Next**: Ready for Phase 3? Let's build ML models and integrate training into your pipelines! 🚀
