# Lab 2.1: ETL Pipeline for E-commerce Data

**Goal**: Build a complete ETL DAG with data ingestion, validation, and cleaning

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Phase 1 completed
- Airflow running locally
- Comfortable with pandas

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Build a multi-stage ETL pipeline
- ✅ Generate and ingest synthetic data
- ✅ Implement data validation checks
- ✅ Clean and transform data
- ✅ Write partitioned outputs
- ✅ Use XComs to pass metadata between tasks

---

## Background: The E-commerce Dataset

We'll build an ETL pipeline for a simulated e-commerce platform that tracks:
- **Transactions**: Customer purchases
- **Products**: Item catalog
- **Customers**: User demographics

This mimics real-world scenarios where you have:
- Multiple data sources
- Data quality issues (nulls, duplicates, invalid values)
- Need for validation and cleaning

### Dataset Schema

**transactions.csv**
```
transaction_id, customer_id, product_id, quantity, price, timestamp, payment_method
```

**products.csv**
```
product_id, name, category, cost, stock
```

**customers.csv**
```
customer_id, name, email, age, country, signup_date
```

---

## Lab Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ETL DAG FLOW                       │
│                                                     │
│  generate_data                                      │
│       ↓                                             │
│  ingest_transactions                                │
│       ↓                                             │
│  validate_transactions                              │
│       ↓                                             │
│  clean_transactions                                 │
│       ↓                                             │
│  write_processed_data                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Part 1: Setup and Data Generation

### Step 1: Create Directory Structure

```bash
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs/data_quality
mkdir -p scripts
```

### Step 2: Create Data Generator Script

Create `scripts/generate_ecommerce_data.py`:

```python
"""
Generate synthetic e-commerce data for ETL pipeline testing.
Creates realistic but artificial transaction, product, and customer data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_transactions(num_records=1000, execution_date=None):
    """Generate synthetic transaction data."""
    if execution_date is None:
        execution_date = datetime.now().date()
    else:
        execution_date = pd.to_datetime(execution_date).date()

    np.random.seed(hash(str(execution_date)) % 2**32)  # Deterministic based on date

    # Generate timestamps for the execution date
    start_time = datetime.combine(execution_date, datetime.min.time())
    timestamps = [
        start_time + timedelta(seconds=np.random.randint(0, 86400))
        for _ in range(num_records)
    ]

    transactions = pd.DataFrame({
        'transaction_id': range(1, num_records + 1),
        'customer_id': np.random.randint(1, 200, num_records),
        'product_id': np.random.randint(1, 50, num_records),
        'quantity': np.random.randint(1, 10, num_records),
        'price': np.round(np.random.uniform(5.0, 500.0, num_records), 2),
        'timestamp': timestamps,
        'payment_method': np.random.choice(
            ['credit_card', 'debit_card', 'paypal', 'crypto', None],
            num_records,
            p=[0.5, 0.25, 0.15, 0.05, 0.05]  # 5% nulls
        )
    })

    # Introduce some data quality issues for validation testing

    # 1. Some negative prices (data error)
    num_errors = int(num_records * 0.02)  # 2% bad data
    error_indices = np.random.choice(num_records, num_errors, replace=False)
    transactions.loc[error_indices, 'price'] = -np.abs(
        transactions.loc[error_indices, 'price']
    )

    # 2. Some extremely high quantities (outliers)
    outlier_indices = np.random.choice(num_records, int(num_records * 0.01), replace=False)
    transactions.loc[outlier_indices, 'quantity'] = np.random.randint(100, 1000, len(outlier_indices))

    # 3. Some duplicate transaction IDs
    dup_indices = np.random.choice(num_records // 2, int(num_records * 0.01), replace=False)
    transactions.loc[dup_indices, 'transaction_id'] = transactions.loc[
        dup_indices + num_records // 2, 'transaction_id'
    ].values

    return transactions


def generate_products():
    """Generate product catalog."""
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    products = []

    for i in range(1, 51):
        products.append({
            'product_id': i,
            'name': f'Product_{i}',
            'category': np.random.choice(categories),
            'cost': np.round(np.random.uniform(3.0, 300.0), 2),
            'stock': np.random.randint(0, 500)
        })

    return pd.DataFrame(products)


def generate_customers():
    """Generate customer data."""
    countries = ['US', 'UK', 'CA', 'DE', 'FR', 'AU']
    customers = []

    for i in range(1, 201):
        signup_date = datetime.now().date() - timedelta(days=np.random.randint(1, 730))
        customers.append({
            'customer_id': i,
            'name': f'Customer_{i}',
            'email': f'customer{i}@example.com',
            'age': np.random.randint(18, 80),
            'country': np.random.choice(countries),
            'signup_date': signup_date
        })

    return pd.DataFrame(customers)


def main(execution_date=None, output_dir='data/raw'):
    """Generate all datasets for a given date."""
    if execution_date is None:
        execution_date = datetime.now().date()
    else:
        execution_date = pd.to_datetime(execution_date).date()

    # Create date-partitioned directory
    date_str = execution_date.strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, date_str)
    os.makedirs(output_path, exist_ok=True)

    # Generate data
    print(f"Generating data for {date_str}...")

    transactions = generate_transactions(execution_date=execution_date)
    products = generate_products()
    customers = generate_customers()

    # Save to CSV
    transactions.to_csv(f'{output_path}/transactions.csv', index=False)
    products.to_csv(f'{output_path}/products.csv', index=False)
    customers.to_csv(f'{output_path}/customers.csv', index=False)

    print(f"Generated {len(transactions)} transactions")
    print(f"Generated {len(products)} products")
    print(f"Generated {len(customers)} customers")
    print(f"Data saved to {output_path}/")

    return {
        'num_transactions': len(transactions),
        'num_products': len(products),
        'num_customers': len(customers),
        'output_path': output_path
    }


if __name__ == '__main__':
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    main(execution_date=date)
```

**Save this file** and test it:

```bash
python scripts/generate_ecommerce_data.py 2024-01-15
```

You should see data in `data/raw/2024-01-15/`.

---

## Part 2: Build the ETL DAG

Create `dags/etl_pipeline.py`:

```python
"""
ETL Pipeline for E-commerce Data

This DAG demonstrates:
- Data generation (simulating ingestion)
- Schema and quality validation
- Data cleaning and transformation
- Writing partitioned outputs
- XCom usage for metadata
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
import json


# Default arguments
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


@dag(
    dag_id='etl_ecommerce_pipeline',
    default_args=default_args,
    description='ETL pipeline for e-commerce transaction data',
    schedule=None,  # Manual trigger for now (Lab 2.4 will add scheduling)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'phase2', 'ecommerce'],
)
def etl_pipeline():
    """ETL Pipeline DAG"""

    @task
    def generate_data(ds=None):
        """
        Generate synthetic e-commerce data for the execution date.

        In production, this would be replaced with actual data ingestion
        from APIs, databases, or file systems.
        """
        import sys
        sys.path.append('/home/user/mlops-learning-plan')

        from scripts.generate_ecommerce_data import main as generate

        logging.info(f"Generating data for execution date: {ds}")

        result = generate(execution_date=ds, output_dir='data/raw')

        logging.info(f"Data generation complete: {result}")

        # Return metadata for downstream tasks
        return {
            'execution_date': ds,
            'num_transactions': result['num_transactions'],
            'output_path': result['output_path']
        }

    @task
    def ingest_transactions(metadata: dict):
        """
        Ingest transaction data from raw layer.

        Returns:
            dict: Ingestion metadata (row count, file path)
        """
        execution_date = metadata['execution_date']
        input_path = f"data/raw/{execution_date}/transactions.csv"

        logging.info(f"Ingesting transactions from {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Transaction file not found: {input_path}")

        # Read data
        df = pd.read_csv(input_path)

        logging.info(f"Ingested {len(df)} transactions")
        logging.info(f"Columns: {list(df.columns)}")
        logging.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return {
            'execution_date': execution_date,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'input_path': input_path
        }

    @task
    def validate_transactions(metadata: dict):
        """
        Validate transaction data quality.

        Checks:
        - Schema (expected columns and types)
        - Required fields (no nulls)
        - Value ranges (price > 0, quantity > 0)
        - Duplicates

        Returns:
            dict: Validation results
        """
        execution_date = metadata['execution_date']
        input_path = metadata['input_path']

        logging.info(f"Validating transactions from {input_path}")

        df = pd.read_csv(input_path, parse_dates=['timestamp'])

        # Validation results
        validation_results = {
            'execution_date': execution_date,
            'total_rows': len(df),
            'validation_errors': [],
            'validation_warnings': []
        }

        # 1. Schema validation
        expected_columns = [
            'transaction_id', 'customer_id', 'product_id',
            'quantity', 'price', 'timestamp', 'payment_method'
        ]

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            error = f"Missing columns: {missing_columns}"
            validation_results['validation_errors'].append(error)
            logging.error(error)

        # 2. Check for nulls in critical fields
        critical_fields = ['transaction_id', 'customer_id', 'product_id', 'price', 'quantity']
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    error = f"Field '{field}' has {null_count} null values"
                    validation_results['validation_errors'].append(error)
                    logging.error(error)

        # 3. Check for invalid values
        if 'price' in df.columns:
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                error = f"Found {negative_prices} negative prices"
                validation_results['validation_errors'].append(error)
                logging.error(error)

            zero_prices = (df['price'] == 0).sum()
            if zero_prices > 0:
                warning = f"Found {zero_prices} zero prices"
                validation_results['validation_warnings'].append(warning)
                logging.warning(warning)

        if 'quantity' in df.columns:
            invalid_quantities = (df['quantity'] <= 0).sum()
            if invalid_quantities > 0:
                error = f"Found {invalid_quantities} invalid quantities (<= 0)"
                validation_results['validation_errors'].append(error)
                logging.error(error)

            high_quantities = (df['quantity'] > 50).sum()
            if high_quantities > 0:
                warning = f"Found {high_quantities} unusually high quantities (> 50)"
                validation_results['validation_warnings'].append(warning)
                logging.warning(warning)

        # 4. Check for duplicates
        if 'transaction_id' in df.columns:
            duplicate_ids = df['transaction_id'].duplicated().sum()
            if duplicate_ids > 0:
                error = f"Found {duplicate_ids} duplicate transaction IDs"
                validation_results['validation_errors'].append(error)
                logging.error(error)

        # 5. Data freshness (check timestamp range)
        if 'timestamp' in df.columns:
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()
            logging.info(f"Timestamp range: {min_ts} to {max_ts}")

        # Save validation report
        report_dir = f"logs/data_quality/{execution_date}"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/validation_report.json"

        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

        logging.info(f"Validation report saved to {report_path}")

        # Fail task if there are errors
        if validation_results['validation_errors']:
            error_msg = "\n".join(validation_results['validation_errors'])
            raise ValueError(f"Data validation failed:\n{error_msg}")

        # Log warnings but continue
        if validation_results['validation_warnings']:
            warning_msg = "\n".join(validation_results['validation_warnings'])
            logging.warning(f"Data validation warnings:\n{warning_msg}")

        logging.info("✓ Validation passed")

        return validation_results

    @task
    def clean_transactions(validation_metadata: dict):
        """
        Clean and transform transaction data.

        Transformations:
        - Remove duplicate transaction IDs
        - Remove rows with negative prices
        - Remove rows with invalid quantities
        - Fill missing payment methods
        - Convert timestamps to datetime
        - Add derived columns (total_amount, date, hour)

        Returns:
            dict: Cleaning metadata
        """
        execution_date = validation_metadata['execution_date']
        input_path = f"data/raw/{execution_date}/transactions.csv"

        logging.info(f"Cleaning transactions from {input_path}")

        # Read data
        df = pd.read_csv(input_path, parse_dates=['timestamp'])
        initial_rows = len(df)

        logging.info(f"Initial row count: {initial_rows}")

        # 1. Remove duplicates
        df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        after_dedup = len(df)
        logging.info(f"Removed {initial_rows - after_dedup} duplicate transaction IDs")

        # 2. Remove negative prices
        df = df[df['price'] >= 0]
        after_price_filter = len(df)
        logging.info(f"Removed {after_dedup - after_price_filter} rows with negative prices")

        # 3. Remove invalid quantities
        df = df[df['quantity'] > 0]
        after_quantity_filter = len(df)
        logging.info(f"Removed {after_price_filter - after_quantity_filter} rows with invalid quantities")

        # 4. Fill missing payment methods
        df['payment_method'] = df['payment_method'].fillna('unknown')

        # 5. Add derived columns
        df['total_amount'] = df['price'] * df['quantity']
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

        # 6. Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        removal_pct = (rows_removed / initial_rows) * 100

        logging.info(f"Final row count: {final_rows}")
        logging.info(f"Total removed: {rows_removed} ({removal_pct:.2f}%)")

        # Return metadata
        return {
            'execution_date': execution_date,
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'removal_percentage': removal_pct,
            'cleaned_df': df  # Pass DataFrame to next task (small enough for XCom)
        }

    @task
    def write_processed_data(cleaning_metadata: dict):
        """
        Write cleaned data to processed layer in Parquet format.

        Parquet benefits:
        - Columnar storage (faster reads)
        - Compression (smaller files)
        - Schema enforcement
        - Metadata included
        """
        execution_date = cleaning_metadata['execution_date']
        df = cleaning_metadata['cleaned_df']

        # Convert back to DataFrame if it was serialized
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create output directory
        output_dir = f"data/processed/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/transactions_clean.parquet"

        # Write to Parquet
        df.to_parquet(output_path, index=False, compression='snappy')

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        logging.info(f"Wrote {len(df)} rows to {output_path}")
        logging.info(f"File size: {file_size_mb:.2f} MB")

        # Write summary statistics
        stats = {
            'execution_date': execution_date,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'file_path': output_path,
            'file_size_mb': file_size_mb,
            'total_amount_sum': float(df['total_amount'].sum()),
            'avg_transaction_value': float(df['total_amount'].mean()),
            'unique_customers': int(df['customer_id'].nunique()),
            'unique_products': int(df['product_id'].nunique()),
        }

        stats_path = f"{output_dir}/summary_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logging.info(f"Summary statistics saved to {stats_path}")
        logging.info("✓ ETL pipeline complete!")

        return stats

    # Define task dependencies
    data_metadata = generate_data()
    ingest_metadata = ingest_transactions(data_metadata)
    validation_metadata = validate_transactions(ingest_metadata)
    cleaning_metadata = clean_transactions(validation_metadata)
    final_stats = write_processed_data(cleaning_metadata)

    # Implicit dependency chain:
    # generate_data >> ingest >> validate >> clean >> write


# Instantiate the DAG
etl_dag = etl_pipeline()
```

---

## Part 3: Run and Test Your DAG

### Step 1: Test the DAG File

```bash
# Check for syntax errors
python dags/etl_pipeline.py

# List DAGs
airflow dags list | grep etl_ecommerce

# Test individual tasks
airflow tasks test etl_ecommerce_pipeline generate_data 2024-01-15
```

### Step 2: Run in Airflow UI

1. **Start Airflow** (if not running):
   ```bash
   airflow webserver --port 8080  # Terminal 1
   airflow scheduler               # Terminal 2
   ```

2. **Access UI**: http://localhost:8080

3. **Enable the DAG**: Toggle `etl_ecommerce_pipeline` to active

4. **Trigger the DAG**: Click the play button and trigger with config:
   ```json
   {
     "ds": "2024-01-15"
   }
   ```

5. **Monitor execution**: Watch tasks turn green

### Step 3: Verify Output

```bash
# Check generated data
ls -lh data/raw/2024-01-15/

# Check processed data
ls -lh data/processed/2024-01-15/

# View summary stats
cat data/processed/2024-01-15/summary_stats.json

# View validation report
cat logs/data_quality/2024-01-15/validation_report.json
```

### Step 4: Inspect Processed Data

```python
import pandas as pd

# Read processed data
df = pd.read_parquet('data/processed/2024-01-15/transactions_clean.parquet')

print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nSummary stats:")
print(df.describe())
```

---

## Exercise 1: Add Product Enrichment

Extend the pipeline to join product data with transactions:

**Add this task**:

```python
@task
def enrich_with_products(cleaning_metadata: dict):
    """
    Join transaction data with product catalog.
    Adds product name, category, and calculates profit margin.
    """
    execution_date = cleaning_metadata['execution_date']

    # Load cleaned transactions
    transactions = cleaning_metadata['cleaned_df']
    if not isinstance(transactions, pd.DataFrame):
        transactions = pd.DataFrame(transactions)

    # Load products
    products_path = f"data/raw/{execution_date}/products.csv"
    products = pd.read_csv(products_path)

    logging.info(f"Enriching {len(transactions)} transactions with product data")

    # Join on product_id
    enriched = transactions.merge(
        products[['product_id', 'name', 'category', 'cost']],
        on='product_id',
        how='left'
    )

    # Calculate profit
    enriched['profit'] = (enriched['price'] - enriched['cost']) * enriched['quantity']

    logging.info(f"Added columns: {['name', 'category', 'cost', 'profit']}")

    return {
        'execution_date': execution_date,
        'enriched_df': enriched
    }
```

**Modify the dependency chain** to include this task between `clean_transactions` and `write_processed_data`.

---

## Exercise 2: Add Aggregation Task

Create a task that generates daily summary statistics:

```python
@task
def generate_daily_summary(stats: dict):
    """
    Generate daily summary aggregations.
    Useful for dashboards and monitoring.
    """
    execution_date = stats['execution_date']
    df = pd.read_parquet(f"data/processed/{execution_date}/transactions_clean.parquet")

    summary = {
        'execution_date': execution_date,
        'total_transactions': len(df),
        'total_revenue': float(df['total_amount'].sum()),
        'avg_transaction_value': float(df['total_amount'].mean()),
        'unique_customers': int(df['customer_id'].nunique()),
        'unique_products': int(df['product_id'].nunique()),
        'transactions_by_hour': df.groupby('hour')['transaction_id'].count().to_dict(),
        'revenue_by_payment_method': df.groupby('payment_method')['total_amount'].sum().to_dict(),
    }

    output_path = f"data/processed/{execution_date}/daily_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Daily summary saved to {output_path}")
    return summary
```

---

## Exercise 3: Simulate Validation Failures

Modify `generate_ecommerce_data.py` to introduce more severe data quality issues:

- Increase the percentage of negative prices to 10%
- Add some completely null rows
- Add invalid customer IDs (negative numbers)

Then run your DAG and observe:
- Does the validation task catch the errors?
- Does the DAG fail as expected?
- Are the error messages clear?

Fix the data generator and re-run.

---

## Challenge: Multi-Source ETL

Extend the pipeline to process all three datasets (transactions, products, customers):

1. **Ingest all three sources** (parallel tasks)
2. **Validate each source** (parallel tasks)
3. **Join all datasets** (after all validations pass)
4. **Write enriched dataset** with customer and product information

Use **task groups** to organize the parallel ingestion and validation tasks.

**Hint**:
```python
from airflow.utils.task_group import TaskGroup

with TaskGroup('ingestion') as ingest_group:
    ingest_transactions_task = ingest_transactions(...)
    ingest_products_task = ingest_products(...)
    ingest_customers_task = ingest_customers(...)

with TaskGroup('validation') as validate_group:
    validate_transactions_task = validate_transactions(...)
    validate_products_task = validate_products(...)
    validate_customers_task = validate_customers(...)

ingest_group >> validate_group >> join_datasets
```

---

## Key Takeaways

### ETL Best Practices

✅ **Validate early**: Fail fast on bad data
✅ **Idempotent transformations**: Same input → same output
✅ **Partition by date**: Enable incremental processing
✅ **Use Parquet**: Faster and smaller than CSV
✅ **Log metadata**: Track row counts, file sizes, timing
✅ **Explicit paths**: Use execution date in file paths

### Data Quality Checks

✅ **Schema validation**: Expected columns and types
✅ **Null checks**: Required fields must be present
✅ **Range checks**: Values within valid ranges
✅ **Uniqueness checks**: No duplicate IDs
✅ **Freshness checks**: Data not too old

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Large DataFrames in XCom | Pass file paths, not DataFrames |
| Validation failures | Check validation logs, fix upstream data |
| Hardcoded dates | Use `ds` parameter (execution date) |
| Slow processing | Use Parquet instead of CSV |
| Missing files | Check paths, ensure upstream tasks succeeded |

---

## Debugging Tips

### Task Fails with "File Not Found"

```bash
# Check if previous task succeeded
airflow tasks state etl_ecommerce_pipeline generate_data 2024-01-15

# Check file exists
ls data/raw/2024-01-15/

# Check task logs
airflow tasks test etl_ecommerce_pipeline generate_data 2024-01-15
```

### Validation Task Fails

```bash
# View validation report
cat logs/data_quality/2024-01-15/validation_report.json

# Check raw data
python -c "import pandas as pd; print(pd.read_csv('data/raw/2024-01-15/transactions.csv').info())"
```

### XCom Size Error

If you get "XCom value too large":

```python
# Don't do this (DataFrame too large)
return df

# Do this instead
df.to_parquet(output_path)
return {'path': output_path, 'num_rows': len(df)}
```

---

## Submission Checklist

Before moving to Lab 2.2:

- ✅ ETL DAG runs successfully end-to-end
- ✅ All tasks turn green in Airflow UI
- ✅ Processed data exists in `data/processed/<date>/`
- ✅ Validation reports generated
- ✅ At least one exercise completed
- ✅ You understand each task's purpose

---

## Next Steps

**What you've built**:
- Complete ETL pipeline with validation and cleaning
- Date-partitioned data structure
- Quality checks and error handling

**Next lab**:
- Build features on top of cleaned data
- Create train/val/test splits
- Version feature sets

**Share your code**:
1. Commit your changes
2. Run the DAG for 2-3 different dates
3. Share any challenges or questions

---

**Congratulations on building your first production-style ETL pipeline!** 🎉

**Next**: [Lab 2.2 - Feature Engineering Pipeline →](./lab2_2_feature_engineering.md)
