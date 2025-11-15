# Lab 2.4 Solution: Scheduled Pipeline with Date Partitioning

This solution provides a production-ready scheduled ETL pipeline with date partitioning, backfill support, and data retention policies.

## Overview

The scheduled pipeline features:

1. **Date Partitioning**: Data organized by year/month/day
2. **Daily Scheduling**: Automatic daily execution at 2 AM
3. **Backfill Support**: Process historical dates automatically
4. **Rolling Metrics**: 7-day rolling window calculations
5. **Data Retention**: Automatic cleanup of old partitions
6. **Comprehensive Reporting**: Daily execution reports

## Files

- `scheduled_etl_dag.py` - Complete scheduled DAG with partitioning and backfill
- `README.md` - This file

## Prerequisites

```bash
# Install required packages
pip install apache-airflow pandas numpy

# Ensure you have the data generator from Lab 2.1
# Copy generate_ecommerce_data.py to DAGs directory

# Airflow must be initialized
airflow db init
```

## Setup

### 1. Copy Files to Airflow

```bash
# Set AIRFLOW_HOME
export AIRFLOW_HOME=~/airflow

# Copy files to DAGs directory
cp scheduled_etl_dag.py $AIRFLOW_HOME/dags/

# Also copy the data generator from Lab 2.1
cp ../lab2_1_solution/generate_ecommerce_data.py $AIRFLOW_HOME/dags/
```

### 2. Configure Airflow for Backfill

Edit `$AIRFLOW_HOME/airflow.cfg`:

```ini
[core]
# Allow parallel backfill runs
max_active_runs_per_dag = 3

# Set parallelism
parallelism = 32
dag_concurrency = 16

[scheduler]
# Enable catchup by default
catchup_by_default = True
```

### 3. Start Airflow

**Terminal 1 - Webserver:**
```bash
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
airflow scheduler
```

## Running the Pipeline

### Option 1: Let it Run on Schedule

The pipeline is configured to run daily at 2 AM:

```python
schedule_interval='0 2 * * *'  # Cron: 2 AM daily
```

Simply enable the DAG and it will run automatically.

### Option 2: Manual Trigger

```bash
# Trigger for today
airflow dags trigger scheduled_ecommerce_etl

# Trigger for specific date
airflow dags trigger scheduled_ecommerce_etl --exec-date 2024-01-15
```

### Option 3: Backfill Historical Dates

```bash
# Backfill last 7 days
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-01 \
    --end-date 2024-01-07

# Backfill with parallelism
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --reset-dagruns  # Rerun if already executed

# Backfill without depends_on_past (faster)
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --ignore-dependencies
```

## Date Partitioning Explained

### Directory Structure

Data is partitioned using Hive-style partitioning:

```
/tmp/ecommerce_data/scheduled/
├── raw/
│   ├── year=2024/
│   │   ├── month=01/
│   │   │   ├── day=01/
│   │   │   │   ├── orders.csv
│   │   │   │   └── validation_report.json
│   │   │   ├── day=02/
│   │   │   │   ├── orders.csv
│   │   │   │   └── validation_report.json
│   │   │   └── ...
│   │   ├── month=02/
│   │   │   └── ...
│   │   └── ...
│   └── year=2025/
│       └── ...
├── processed/
│   ├── year=2024/
│   │   ├── month=01/
│   │   │   ├── day=01/
│   │   │   │   ├── orders_processed.csv
│   │   │   │   └── daily_summary.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── metrics/
    ├── year=2024/
    │   └── month=01/
    │       ├── day=01/
    │       │   ├── rolling_metrics.json
    │       │   └── daily_report.json
    │       └── ...
    └── latest_daily_report.json
```

### Benefits of Partitioning

1. **Efficient Queries**: Query specific dates without scanning all data
2. **Parallel Processing**: Process multiple partitions in parallel
3. **Easy Backfill**: Reprocess specific dates independently
4. **Data Management**: Delete old partitions easily
5. **Cost Optimization**: Store hot data separately from cold data

### Partition Path Generation

```python
def get_partition_path(base_path: str, execution_date: datetime) -> str:
    """Generate partition path: base_path/year=YYYY/month=MM/day=DD"""
    return os.path.join(
        base_path,
        f"year={execution_date.year}",
        f"month={execution_date.month:02d}",
        f"day={execution_date.day:02d}"
    )

# Example usage
partition = get_partition_path('/data/raw', datetime(2024, 1, 15))
# Returns: /data/raw/year=2024/month=01/day=15
```

## Pipeline Tasks Explained

### Task 1: Generate Daily Data

```python
Execution: Daily at 2 AM
Purpose: Generate/ingest data for the execution date
Output:
  - year=YYYY/month=MM/day=DD/orders.csv
```

- Generates synthetic data for the specific date
- Uses execution_date to determine which day's data to create
- Supports backfill by generating historical data

### Task 2: Validate Daily Data

```python
Execution: After data generation
Purpose: Validate data quality for the partition
Checks:
  - Minimum record count (50 orders/day)
  - Required columns present
  - No nulls in critical fields
  - Dates within expected range
```

Fails the pipeline if validation doesn't pass.

### Task 3: Process Daily Data

```python
Execution: After validation passes
Purpose: Clean and transform daily data
Operations:
  - Remove duplicates
  - Fix negative values
  - Add derived columns (hour, day_of_week, etc.)
  - Create daily summary statistics
```

### Task 4: Compute Rolling Metrics

```python
Execution: After processing
Purpose: Calculate 7-day rolling window metrics
Metrics:
  - Average daily orders
  - Average daily revenue
  - Total orders (7 days)
  - Growth rate vs average
```

Looks back 7 days from execution date.

### Task 5: Generate Daily Report

```python
Execution: After metrics computation
Purpose: Create comprehensive daily report
Includes:
  - Pipeline execution metadata
  - Data ingestion stats
  - Processing results
  - Rolling metrics
  - Growth trends
```

### Task 6: Cleanup Old Partitions

```python
Execution: After report generation
Purpose: Delete partitions older than 30 days
Retention: 30 days (configurable)
```

Implements data retention policy.

## Backfill Support

### What is Backfill?

Backfill processes historical dates that were missed or need reprocessing.

### How Backfill Works

1. **catchup=True**: Enables backfill mode
2. **start_date**: First date to process
3. **execution_date**: Current date being processed in the backfill
4. **depends_on_past=True**: Ensures sequential processing

### Example: Backfill Last Month

```bash
# Backfill January 2024
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --reset-dagruns

# Monitor backfill progress
airflow dags list-runs -d scheduled_ecommerce_etl
```

### Backfill Scenarios

**Scenario 1: New Pipeline**
```bash
# Pipeline deployed on Jan 15, need data from Jan 1
# Airflow automatically backfills Jan 1-14 when DAG is enabled
# (if catchup=True and start_date < today)
```

**Scenario 2: Failed Date**
```bash
# Jan 10 failed, reprocess just that date
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-10 \
    --end-date 2024-01-10 \
    --reset-dagruns
```

**Scenario 3: Data Correction**
```bash
# Bug fixed, reprocess last 7 days
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-08 \
    --end-date 2024-01-14 \
    --reset-dagruns \
    --rerun-failed-tasks
```

## Schedule Configuration

### Cron Schedule

```python
# Daily at 2 AM
schedule_interval='0 2 * * *'

# Common schedules:
schedule_interval='@daily'        # Midnight
schedule_interval='@hourly'       # Every hour
schedule_interval='0 */6 * * *'   # Every 6 hours
schedule_interval='0 0 * * 1'     # Weekly (Monday)
schedule_interval='0 0 1 * *'     # Monthly (1st day)
```

### Execution Date vs Run Date

- **execution_date**: Date/time the data represents (logical date)
- **run_date**: Actual date/time when DAG runs (physical date)

Example:
```
execution_date: 2024-01-01 00:00:00 (data for Jan 1)
run_date: 2024-01-02 02:00:00 (runs on Jan 2 at 2 AM)
```

### Data Interval

```python
# Access in tasks
execution_date = context['execution_date']
data_interval_start = context['data_interval_start']
data_interval_end = context['data_interval_end']

# For daily DAG:
# execution_date: 2024-01-01 00:00:00
# data_interval_start: 2024-01-01 00:00:00
# data_interval_end: 2024-01-02 00:00:00
```

## Monitoring and Reporting

### View Latest Report

```bash
# View latest daily report
cat /tmp/ecommerce_data/scheduled/metrics/latest_daily_report.json

# View specific date
cat /tmp/ecommerce_data/scheduled/metrics/year=2024/month=01/day=15/daily_report.json
```

### Query Partitioned Data

```python
import pandas as pd
from glob import glob

# Load all January data
files = glob('/tmp/ecommerce_data/scheduled/processed/year=2024/month=01/day=*/orders_processed.csv')
df = pd.concat([pd.read_csv(f) for f in files])
print(f"January: {len(df)} orders")

# Load specific date range
def load_date_range(start_date, end_date):
    dfs = []
    current = start_date
    while current <= end_date:
        path = f'/tmp/ecommerce_data/scheduled/processed/year={current.year}/month={current.month:02d}/day={current.day:02d}/orders_processed.csv'
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        current += timedelta(days=1)
    return pd.concat(dfs) if dfs else pd.DataFrame()

# Load last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
df = load_date_range(start_date, end_date)
```

### Create Dashboard

```python
import matplotlib.pyplot as plt
import json
from glob import glob

# Load daily summaries
summaries = []
for file in sorted(glob('/tmp/ecommerce_data/scheduled/processed/year=2024/month=01/day=*/daily_summary.json')):
    with open(file) as f:
        summaries.append(json.load(f))

# Plot revenue trend
dates = [s['execution_date'] for s in summaries]
revenues = [s['total_revenue'] for s in summaries]

plt.figure(figsize=(12, 6))
plt.plot(dates, revenues, marker='o')
plt.title('Daily Revenue Trend')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('revenue_trend.png')
```

## Advanced Features

### Parallel Backfill

Process multiple dates in parallel:

```python
# In DAG configuration
max_active_runs=3  # Allow 3 parallel backfill runs

# Backfill will run Jan 1, 2, 3 in parallel, then 4, 5, 6, etc.
```

### Conditional Task Execution

```python
# Only cleanup on Sundays
from airflow.operators.python import BranchPythonOperator

def should_cleanup(**context):
    if context['execution_date'].weekday() == 6:  # Sunday
        return 'cleanup_old_partitions'
    else:
        return 'skip_cleanup'

branch = BranchPythonOperator(
    task_id='check_cleanup_day',
    python_callable=should_cleanup
)
```

### Dynamic DAGs

Generate DAGs dynamically for multiple regions:

```python
regions = ['us-east', 'us-west', 'eu-west']

for region in regions:
    dag_id = f'scheduled_ecommerce_etl_{region}'

    dag = DAG(
        dag_id,
        default_args=default_args,
        schedule_interval='0 2 * * *',
        ...
    )
    # Define tasks...
```

## Data Retention Policy

### Configure Retention

```python
# In cleanup_old_partitions function
retention_days = 30  # Keep last 30 days

# Or use different retention by data type
retention_policies = {
    'raw': 7,        # Keep raw data 7 days
    'processed': 30, # Keep processed 30 days
    'metrics': 90    # Keep metrics 90 days
}
```

### Manual Cleanup

```bash
# Delete partitions older than specific date
python -c "
import shutil
from pathlib import Path
from datetime import datetime

cutoff = datetime(2024, 1, 1)
for day_dir in Path('/tmp/ecommerce_data/scheduled/raw').glob('year=*/month=*/day=*'):
    year = int(day_dir.parent.parent.name.split('=')[1])
    month = int(day_dir.parent.name.split('=')[1])
    day = int(day_dir.name.split('=')[1])
    if datetime(year, month, day) < cutoff:
        shutil.rmtree(day_dir)
        print(f'Deleted {day_dir}')
"
```

## Troubleshooting

### Backfill Not Running

```bash
# Check if catchup is enabled
airflow dags show scheduled_ecommerce_etl | grep catchup

# Enable catchup
airflow dags update --property catchup=True scheduled_ecommerce_etl
```

### Partition Already Exists

```bash
# Clear specific date
airflow tasks clear scheduled_ecommerce_etl \
    --start-date 2024-01-15 \
    --end-date 2024-01-15

# Force rerun
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-15 \
    --end-date 2024-01-15 \
    --reset-dagruns
```

### Slow Backfill

```bash
# Increase parallelism in airflow.cfg
[core]
parallelism = 64
dag_concurrency = 32

# Or use --ignore-dependencies for independent dates
airflow dags backfill scheduled_ecommerce_etl \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --ignore-dependencies
```

## Best Practices

1. **Use Partitioning**: Always partition data by date for efficient querying
2. **Enable Catchup Carefully**: Only enable if you want automatic backfill
3. **Set depends_on_past**: For data pipelines with dependencies
4. **Implement Idempotency**: Tasks should produce same result when rerun
5. **Monitor Backfills**: Watch for failures during backfill operations
6. **Test Backfill**: Test with small date range first
7. **Document Partitions**: Keep partition schema documented
8. **Retention Policy**: Define and enforce data retention
9. **Validate Partitions**: Check partition completeness regularly
10. **Use Templating**: Leverage Jinja templates for dynamic values

## Production Deployment

### 1. Configure for Production

```python
# Use production database
sql_alchemy_conn = 'postgresql://user:pass@host/airflow'

# Use production executor
executor = CeleryExecutor  # or KubernetesExecutor

# Set appropriate parallelism
parallelism = 64
dag_concurrency = 32
```

### 2. Set Up Monitoring

```python
# Add monitoring callbacks
from airflow.utils.email import send_email

def failure_callback(context):
    send_email(
        to=['ops@example.com'],
        subject=f"DAG {context['dag'].dag_id} Failed",
        html_content=f"Task {context['task'].task_id} failed"
    )

default_args = {
    'on_failure_callback': failure_callback
}
```

### 3. Use External Storage

```python
# Instead of /tmp, use S3, GCS, or HDFS
BASE_DIR = 's3://my-bucket/ecommerce_data/scheduled'

# Or
BASE_DIR = 'gs://my-bucket/ecommerce_data/scheduled'

# Or
BASE_DIR = 'hdfs:///user/airflow/ecommerce_data/scheduled'
```

## Next Steps

1. **Production Deployment**: Deploy to production Airflow cluster
2. **Integrate with Data Warehouse**: Load partitions into BigQuery, Snowflake, etc.
3. **Add Data Quality**: Integrate Lab 2.3 validations
4. **Feature Engineering**: Add Lab 2.2 feature creation
5. **Monitoring**: Set up dashboards and alerts
6. **Optimization**: Add partition pruning, caching
7. **Documentation**: Document partition schema and processes

## Resources

- [Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/concepts/index.html)
- [Data Partitioning Best Practices](https://cloud.google.com/bigquery/docs/partitioned-tables)
- [Backfill Guide](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#backfill)

## License

MIT License - Free to use for learning purposes
