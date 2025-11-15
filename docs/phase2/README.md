# Phase 2: Data & Pipelines with Airflow

**Duration**: 2-3 weeks
**Goal**: Build production-grade ETL and feature engineering pipelines

---

## Overview

Welcome to Phase 2! This is where you transform from "I can write a DAG" to "I can architect production data pipelines."

In Phase 1, you learned the basics of Airflow and PyTorch. Now you'll build real-world data pipelines that:
- Ingest data from various sources
- Validate data quality
- Transform and engineer features
- Version and partition data
- Handle failures gracefully
- Run on schedules with incremental processing

### What You'll Learn

1. **ETL Pipeline Architecture**
   - Extract: Reading from files, databases, APIs
   - Transform: Cleaning, validation, feature engineering
   - Load: Writing partitioned, versioned datasets
   - Idempotency and incremental processing

2. **Data Validation & Quality**
   - Schema validation
   - Statistical checks (nulls, ranges, distributions)
   - Data profiling
   - Failing fast vs graceful degradation
   - Great Expectations concepts

3. **Feature Engineering in Pipelines**
   - Train/validation/test splits
   - Feature transformations
   - Avoiding data leakage
   - Feature versioning
   - Reproducibility

4. **Data Partitioning Strategies**
   - Date-based partitioning (daily, hourly)
   - Full refresh vs incremental
   - Backfills and catchup
   - Late-arriving data

5. **Airflow Advanced Patterns**
   - XComs for inter-task communication
   - Task groups for organization
   - Dynamic task generation
   - Retry logic and SLAs
   - DAG parameterization
   - Templating with Jinja

---

## The Data Pipeline Mental Model

Before building pipelines, internalize this architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA PIPELINE STAGES                       │
│                                                             │
│  1. INGEST (Extract)                                        │
│     ├─ Read from source (files, DB, API)                   │
│     ├─ Land in raw layer                                    │
│     └─ Partition by date/batch                              │
│          ↓                                                  │
│  2. VALIDATE (Quality Gate)                                 │
│     ├─ Check schema (columns, types)                        │
│     ├─ Check stats (nulls, ranges)                          │
│     ├─ Fail or warn on violations                           │
│     └─ Log data profile                                     │
│          ↓                                                  │
│  3. CLEAN (Transform)                                       │
│     ├─ Handle missing values                                │
│     ├─ Remove duplicates                                    │
│     ├─ Fix data types                                       │
│     └─ Filter invalid rows                                  │
│          ↓                                                  │
│  4. ENGINEER FEATURES                                       │
│     ├─ Compute derived features                             │
│     ├─ Encode categoricals                                  │
│     ├─ Scale numericals                                     │
│     └─ Create train/val/test splits                         │
│          ↓                                                  │
│  5. LOAD (Persist)                                          │
│     ├─ Write to processed layer                             │
│     ├─ Version the output                                   │
│     └─ Ready for training                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure for Data

```
data/
  raw/                    # Immutable, as-received data
    2024-01-01/
      customers.csv
    2024-01-02/
      customers.csv

  processed/              # Cleaned, validated data
    2024-01-01/
      customers_clean.parquet

  features/               # Feature-engineered datasets
    v1/                   # Feature version
      train.parquet
      val.parquet
      test.parquet

  predictions/            # Model outputs (Phase 4)
    2024-01-01/
      batch_predictions.parquet
```

**Key Principles**:
- **Raw layer is immutable**: Never modify source data
- **Idempotent transforms**: Rerunning produces same output
- **Versioning**: Track schema and feature changes
- **Partitioning**: Enable incremental processing

---

## ETL vs ELT

### ETL (Extract-Transform-Load)
```
Source → Transform → Load to Data Warehouse
```
- Transform BEFORE loading
- Common in traditional data warehouses
- More control over what's loaded

### ELT (Extract-Load-Transform)
```
Source → Load to Data Lake → Transform
```
- Load raw data first, transform later
- Common in modern data lakes
- Better for exploratory analysis

**We'll use ETL** for this course because:
- More relevant for ML (we transform before training)
- Teaches proper validation patterns
- Matches production ML workflows

---

## Data Quality: The Foundation of Good ML

**Garbage in, garbage out.**

### Why Data Quality Matters

Poor data quality leads to:
- ❌ Models that don't generalize
- ❌ Silent failures in production
- ❌ Broken experiments and wasted time
- ❌ Data drift going undetected

Good data quality ensures:
- ✅ Reproducible experiments
- ✅ Trustworthy model performance
- ✅ Early detection of issues
- ✅ Confidence in production systems

### Types of Data Quality Checks

#### 1. Schema Validation
```python
# Expected schema
expected_columns = ['user_id', 'age', 'purchase_amount', 'timestamp']
expected_types = {
    'user_id': 'int64',
    'age': 'int64',
    'purchase_amount': 'float64',
    'timestamp': 'datetime64[ns]'
}

# Validate
assert set(df.columns) == set(expected_columns)
assert df.dtypes.to_dict() == expected_types
```

#### 2. Statistical Checks
```python
# Nulls
assert df['user_id'].isnull().sum() == 0, "user_id cannot be null"

# Ranges
assert df['age'].between(0, 120).all(), "Invalid age values"

# Uniqueness
assert df['user_id'].is_unique, "Duplicate user_ids found"

# Distributions (flag warnings, don't fail)
if df['purchase_amount'].mean() > threshold:
    logging.warning("Purchase amount mean higher than expected")
```

#### 3. Freshness Checks
```python
# Check data recency
max_timestamp = df['timestamp'].max()
hours_old = (datetime.now() - max_timestamp).total_seconds() / 3600

if hours_old > 24:
    raise ValueError(f"Data is {hours_old:.1f} hours old")
```

---

## Feature Engineering Best Practices

### 1. Avoid Data Leakage

**Data Leakage**: Using information from the future or test set during training.

```python
# BAD: Scaling before split (test set info leaks to train)
df['amount_scaled'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
train, test = train_test_split(df)

# GOOD: Split first, then scale
train, test = train_test_split(df)
mean, std = train['amount'].mean(), train['amount'].std()
train['amount_scaled'] = (train['amount'] - mean) / std
test['amount_scaled'] = (test['amount'] - mean) / std  # Use train stats!
```

### 2. Feature Versioning

Why version features?
- Track which features correspond to which model
- Reproduce experiments exactly
- Roll back to previous feature sets

```python
# Save feature version metadata
metadata = {
    'version': 'v2',
    'created_at': datetime.now().isoformat(),
    'feature_columns': list(features.columns),
    'transformations': [
        'log_transform(amount)',
        'one_hot_encode(category)',
        'date_features(timestamp)'
    ]
}

with open('data/features/v2/metadata.json', 'w') as f:
    json.dump(metadata, f)
```

### 3. Reproducible Splits

```python
# BAD: Different split each time
train, test = train_test_split(df)

# GOOD: Deterministic split
train, test = train_test_split(df, random_state=42)

# BETTER: Time-based split for temporal data
cutoff_date = '2024-01-15'
train = df[df['date'] < cutoff_date]
test = df[df['date'] >= cutoff_date]
```

---

## Airflow Advanced Patterns

### 1. XComs (Cross-Communication)

Pass small amounts of data between tasks:

```python
@task
def extract():
    data = fetch_data()
    return len(data)  # Automatically pushed to XCom

@task
def validate(num_records: int):  # Automatically pulled from XCom
    if num_records < 100:
        raise ValueError("Not enough records")
    return num_records

@dag(...)
def my_dag():
    records = extract()
    validate(records)
```

**Best Practices**:
- ✅ Use for metadata (counts, paths, status)
- ❌ Don't use for large data (use files instead)

### 2. Task Groups

Organize related tasks:

```python
from airflow.utils.task_group import TaskGroup

with DAG(...) as dag:
    with TaskGroup('ingestion') as ingest_group:
        fetch = PythonOperator(task_id='fetch', ...)
        validate = PythonOperator(task_id='validate', ...)
        fetch >> validate

    with TaskGroup('feature_engineering') as feature_group:
        encode = PythonOperator(task_id='encode', ...)
        scale = PythonOperator(task_id='scale', ...)
        encode >> scale

    ingest_group >> feature_group
```

### 3. Templating with Jinja

Use macros for dynamic values:

```python
@task
def process_data(ds, **context):
    # ds is execution date in YYYY-MM-DD format
    input_path = f"data/raw/{ds}/data.csv"
    output_path = f"data/processed/{ds}/data.parquet"

    # Access other context variables
    run_id = context['run_id']
    dag_id = context['dag_id']

    # Process...
```

Common macros:
- `{{ ds }}`: Execution date (YYYY-MM-DD)
- `{{ ds_nodash }}`: Execution date (YYYYMMDD)
- `{{ yesterday_ds }}`: Previous day
- `{{ params.my_param }}`: Custom parameters

### 4. Retry Logic

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Task-specific overrides
task = PythonOperator(
    task_id='flaky_api_call',
    python_callable=call_api,
    retries=5,  # This task is extra flaky
)
```

### 5. Sensors (Waiting for Conditions)

```python
from airflow.sensors.filesystem import FileSensor

wait_for_file = FileSensor(
    task_id='wait_for_data',
    filepath='/data/raw/{{ ds }}/data.csv',
    poke_interval=60,  # Check every 60 seconds
    timeout=3600,       # Give up after 1 hour
)

wait_for_file >> process_data
```

---

## Partitioning Strategies

### 1. Date Partitioning

**Daily partitions**:
```
data/raw/2024-01-01/
data/raw/2024-01-02/
data/raw/2024-01-03/
```

**Hourly partitions** (for high-volume data):
```
data/raw/2024-01-01/00/
data/raw/2024-01-01/01/
data/raw/2024-01-01/02/
```

### 2. Full Refresh vs Incremental

**Full Refresh**:
```python
# Process entire dataset each run
df = load_all_data()
df_processed = transform(df)
df_processed.to_parquet('data/processed/full.parquet')
```

**Pros**: Simple, always consistent
**Cons**: Slow for large datasets

**Incremental**:
```python
# Process only new data
df_new = load_data_for_date(execution_date)
df_processed = transform(df_new)
df_processed.to_parquet(f'data/processed/{execution_date}/data.parquet')
```

**Pros**: Fast, scalable
**Cons**: More complex, need to handle late data

### 3. Backfills

Reprocess historical data:

```bash
# Backfill specific date range
airflow dags backfill my_dag \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

# Clear and rerun
airflow tasks clear my_dag \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

**Use cases**:
- Bug fixes in pipeline logic
- Schema changes
- Reprocessing with new features

---

## Phase 2 Labs

### Lab 2.1: ETL Pipeline for E-commerce Data
**Goal**: Build a complete ETL DAG with ingestion, validation, and cleaning
- Extract synthetic e-commerce transaction data
- Validate schema and data quality
- Clean and transform data
- Write partitioned output

[→ Go to Lab 2.1](./lab2_1_etl_pipeline.md)

### Lab 2.2: Feature Engineering Pipeline
**Goal**: Build features for a predictive model
- Load cleaned data from Lab 2.1
- Engineer time-based and aggregation features
- Create train/val/test splits
- Version feature outputs

[→ Go to Lab 2.2](./lab2_2_feature_engineering.md)

### Lab 2.3: Data Quality Checks
**Goal**: Implement comprehensive quality checks
- Schema validation
- Statistical tests
- Anomaly detection
- Failure handling strategies
- Great Expectations patterns

[→ Go to Lab 2.3](./lab2_3_data_quality.md)

### Lab 2.4: Scheduled Partitioned Pipeline
**Goal**: Schedule pipeline with incremental processing
- Daily scheduled runs
- Date-based partitioning
- Incremental vs full refresh
- Backfilling historical data

[→ Go to Lab 2.4](./lab2_4_scheduled_pipeline.md)

---

## Success Criteria

You've completed Phase 2 when you can:

✅ Design multi-stage ETL pipelines
✅ Implement data validation and quality checks
✅ Build feature engineering workflows
✅ Use date partitioning for incremental processing
✅ Schedule DAGs with proper parameterization
✅ Debug data pipeline failures
✅ Explain idempotency and data versioning
✅ Use XComs and task groups effectively

---

## Best Practices to Internalize

### 1. Idempotency

**Running the same task with the same inputs produces the same output.**

```python
# BAD: Appends, not idempotent
df.to_csv('data.csv', mode='a')

# GOOD: Overwrites, idempotent
df.to_parquet(f'data/processed/{execution_date}/data.parquet')
```

### 2. Fail Fast

```python
# Validate early in the pipeline
def validate_schema(df):
    required_cols = ['user_id', 'timestamp', 'amount']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

# Don't let bad data propagate downstream
```

### 3. Explicit Over Implicit

```python
# BAD: Unclear where data comes from
def process_data():
    df = pd.read_csv('data.csv')  # Which data.csv?

# GOOD: Explicit paths
def process_data(input_path: str, output_path: str, execution_date: str):
    df = pd.read_csv(input_path)
    # ... transform ...
    df.to_parquet(output_path)
```

### 4. Separate Raw and Processed

```
data/
  raw/         # NEVER modify
  processed/   # Output of pipelines
  features/    # Ready for training
```

### 5. Log Everything

```python
import logging

logging.info(f"Processing {len(df)} records for date {execution_date}")
logging.info(f"Removed {num_nulls} rows with nulls")
logging.info(f"Output written to {output_path}")
```

---

## Common Pitfalls

### 1. Not Handling Missing Files

```python
# BAD
df = pd.read_csv(f'data/raw/{date}/data.csv')

# GOOD
import os
path = f'data/raw/{date}/data.csv'
if not os.path.exists(path):
    raise FileNotFoundError(f"Data not found for {date}")
df = pd.read_csv(path)
```

### 2. Leaking Future Data

```python
# BAD: Using global stats (includes future data)
df['amount_scaled'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

# GOOD: Use only training data stats
train_mean = train['amount'].mean()
train_std = train['amount'].std()
df['amount_scaled'] = (df['amount'] - train_mean) / train_std
```

### 3. Large XComs

```python
# BAD: Passing entire dataframe
@task
def extract():
    return df  # 1GB dataframe!

# GOOD: Pass file path
@task
def extract():
    path = 'data/raw/data.parquet'
    df.to_parquet(path)
    return path
```

### 4. Hardcoded Dates

```python
# BAD
df = pd.read_csv('data/raw/2024-01-01/data.csv')

# GOOD
@task
def process(ds=None):
    df = pd.read_csv(f'data/raw/{ds}/data.csv')
```

---

## Tools & Libraries

### Data Processing
- **pandas**: DataFrame operations
- **pyarrow**: Fast Parquet I/O
- **polars**: Faster alternative to pandas (optional)

### Data Validation
- **pandera**: DataFrame schema validation
- **Great Expectations**: Comprehensive data quality (we'll introduce the concepts)

### File Formats
- **CSV**: Human-readable, but slow and no schema
- **Parquet**: Columnar, compressed, schema-enforced (recommended)
- **Feather**: Fast but not as compressed

```python
# Why Parquet?
# CSV
df.to_csv('data.csv')  # 100 MB
pd.read_csv('data.csv')  # 5 seconds

# Parquet
df.to_parquet('data.parquet')  # 20 MB
pd.read_parquet('data.parquet')  # 0.5 seconds
```

---

## Next Steps

1. **Review Phase 2 concepts** (you're here!)
2. **Complete Lab 2.1** (ETL Pipeline)
3. **Share your code** for review
4. **Iterate** based on feedback
5. **Move to Lab 2.2** when ready

---

## Resources

### Airflow
- [Airflow DAG Authoring](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html)
- [XComs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html)
- [Task Groups](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#taskgroups)

### Data Quality
- [Great Expectations](https://greatexpectations.io/)
- [Pandera](https://pandera.readthedocs.io/)

### Data Engineering
- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) (book)
- [Data Partitioning Best Practices](https://docs.databricks.com/delta/best-practices.html)

---

**Ready to build production data pipelines? Let's start with Lab 2.1!** 🚀
