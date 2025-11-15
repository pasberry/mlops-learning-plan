# Lab 2.2: Feature Engineering Pipeline

**Goal**: Build a feature engineering DAG that creates ML-ready datasets

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Lab 2.1 completed
- Understanding of train/test splits
- Basic feature engineering concepts

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Build features from cleaned transaction data
- ✅ Create time-based and aggregation features
- ✅ Implement proper train/validation/test splits
- ✅ Version feature datasets
- ✅ Avoid data leakage
- ✅ Integrate with the ETL pipeline

---

## Background: Feature Engineering for ML

### What Are Features?

Features are the inputs to your ML model. Good features make the difference between a mediocre model and a great one.

**Raw data** (from Lab 2.1):
```
transaction_id, customer_id, product_id, quantity, price, timestamp
```

**Engineered features** (what models actually use):
```
customer_total_purchases, customer_avg_order_value,
days_since_last_purchase, favorite_category,
is_weekend, hour_of_day, ...
```

### Types of Features

1. **Numeric transformations**: log, square root, scaling
2. **Categorical encoding**: one-hot, label encoding, embeddings
3. **Temporal features**: hour, day of week, is_weekend
4. **Aggregations**: customer lifetime value, product popularity
5. **Interactions**: price × quantity, category × hour

### Avoiding Data Leakage

**Data leakage** = Using information from the future or test set during training.

```python
# ❌ WRONG: Leakage - using all data to compute stats
df['price_normalized'] = (df['price'] - df['price'].mean()) / df['price'].std()
train, test = split(df)

# ✅ CORRECT: Compute stats on train only
train, test = split(df)
train_mean = train['price'].mean()
train_std = train['price'].std()
train['price_normalized'] = (train['price'] - train_mean) / train_std
test['price_normalized'] = (test['price'] - train_mean) / train_std  # Use train stats!
```

---

## Lab Architecture

```
┌──────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING PIPELINE               │
│                                                      │
│  load_cleaned_data                                   │
│       ↓                                              │
│  compute_customer_features (aggregations)            │
│       ↓                                              │
│  compute_product_features (aggregations)             │
│       ↓                                              │
│  compute_temporal_features                           │
│       ↓                                              │
│  merge_all_features                                  │
│       ↓                                              │
│  create_train_val_test_split                         │
│       ↓                                              │
│  write_feature_datasets (versioned)                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Part 1: Create Feature Engineering DAG

Create `dags/feature_engineering_pipeline.py`:

```python
"""
Feature Engineering Pipeline

This DAG creates ML-ready feature datasets from cleaned transaction data.

Features created:
- Customer aggregations (total purchases, avg order value, etc.)
- Product aggregations (popularity, avg price, etc.)
- Temporal features (hour, day of week, is_weekend)
- Interaction features

Outputs:
- Versioned train/val/test splits
- Feature metadata for reproducibility
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Dict, Tuple


default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


@dag(
    dag_id='feature_engineering_pipeline',
    default_args=default_args,
    description='Feature engineering pipeline for ML model training',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['feature-engineering', 'phase2', 'ml'],
)
def feature_engineering_pipeline():
    """Feature Engineering DAG"""

    @task
    def load_cleaned_data(ds=None, execution_dates: list = None):
        """
        Load cleaned transaction data.

        For time-series data, we may want to load multiple days
        to compute historical aggregations.

        Args:
            ds: Execution date (for single-day processing)
            execution_dates: List of dates to load (for multi-day)
        """
        if execution_dates is None:
            execution_dates = [ds]

        logging.info(f"Loading cleaned data for dates: {execution_dates}")

        all_data = []
        for date in execution_dates:
            file_path = f"data/processed/{date}/transactions_clean.parquet"

            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}, skipping")
                continue

            df = pd.read_parquet(file_path)
            all_data.append(df)
            logging.info(f"Loaded {len(df)} rows from {date}")

        if not all_data:
            raise FileNotFoundError(f"No data found for dates: {execution_dates}")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        logging.info(f"Total rows loaded: {len(combined_df)}")
        logging.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

        return {
            'num_rows': len(combined_df),
            'num_dates': len(execution_dates),
            'execution_date': ds,
        }

    @task
    def compute_customer_features(load_metadata: dict):
        """
        Compute customer-level aggregation features.

        Features:
        - Total number of purchases
        - Total amount spent
        - Average order value
        - Days since first purchase
        - Favorite payment method
        - Purchase frequency (orders per day)
        """
        execution_date = load_metadata['execution_date']

        # Load data
        df = pd.read_parquet(f"data/processed/{execution_date}/transactions_clean.parquet")

        logging.info(f"Computing customer features for {df['customer_id'].nunique()} customers")

        # Compute aggregations per customer
        customer_features = df.groupby('customer_id').agg({
            'transaction_id': 'count',              # num_purchases
            'total_amount': ['sum', 'mean', 'std'],  # spending stats
            'timestamp': ['min', 'max'],            # first and last purchase
            'payment_method': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',  # favorite payment method
        }).reset_index()

        # Flatten column names
        customer_features.columns = [
            'customer_id',
            'customer_num_purchases',
            'customer_total_spent',
            'customer_avg_order_value',
            'customer_order_std',
            'customer_first_purchase',
            'customer_last_purchase',
            'customer_favorite_payment_method'
        ]

        # Compute derived features
        customer_features['customer_order_std'] = customer_features['customer_order_std'].fillna(0)

        # Days since first purchase (tenure)
        customer_features['customer_tenure_days'] = (
            customer_features['customer_last_purchase'] - customer_features['customer_first_purchase']
        ).dt.days

        # Purchase frequency (orders per day)
        customer_features['customer_purchase_frequency'] = (
            customer_features['customer_num_purchases'] /
            (customer_features['customer_tenure_days'] + 1)  # +1 to avoid division by zero
        )

        # Drop timestamp columns (we've derived features from them)
        customer_features = customer_features.drop(
            columns=['customer_first_purchase', 'customer_last_purchase']
        )

        # Save intermediate results
        output_dir = f"data/features/intermediate/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/customer_features.parquet"

        customer_features.to_parquet(output_path, index=False)

        logging.info(f"Computed {len(customer_features.columns)} customer features")
        logging.info(f"Saved to {output_path}")

        return {
            'execution_date': execution_date,
            'num_customers': len(customer_features),
            'num_features': len(customer_features.columns) - 1,  # -1 for customer_id
            'output_path': output_path
        }

    @task
    def compute_product_features(load_metadata: dict):
        """
        Compute product-level aggregation features.

        Features:
        - Product popularity (num purchases)
        - Average price sold at
        - Total quantity sold
        - Revenue generated
        """
        execution_date = load_metadata['execution_date']

        # Load data
        df = pd.read_parquet(f"data/processed/{execution_date}/transactions_clean.parquet")

        logging.info(f"Computing product features for {df['product_id'].nunique()} products")

        # Compute aggregations per product
        product_features = df.groupby('product_id').agg({
            'transaction_id': 'count',       # popularity
            'price': ['mean', 'std', 'min', 'max'],
            'quantity': 'sum',
            'total_amount': 'sum',           # revenue
        }).reset_index()

        # Flatten column names
        product_features.columns = [
            'product_id',
            'product_popularity',
            'product_avg_price',
            'product_price_std',
            'product_min_price',
            'product_max_price',
            'product_total_quantity_sold',
            'product_total_revenue',
        ]

        # Handle missing values
        product_features['product_price_std'] = product_features['product_price_std'].fillna(0)

        # Compute derived features
        product_features['product_price_range'] = (
            product_features['product_max_price'] - product_features['product_min_price']
        )

        # Save intermediate results
        output_dir = f"data/features/intermediate/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/product_features.parquet"

        product_features.to_parquet(output_path, index=False)

        logging.info(f"Computed {len(product_features.columns)} product features")
        logging.info(f"Saved to {output_path}")

        return {
            'execution_date': execution_date,
            'num_products': len(product_features),
            'num_features': len(product_features.columns) - 1,
            'output_path': output_path
        }

    @task
    def compute_temporal_features(load_metadata: dict):
        """
        Compute temporal features from transaction timestamps.

        Features:
        - Hour of day
        - Day of week
        - Is weekend
        - Is business hours (9-5)
        - Month
        - Quarter
        """
        execution_date = load_metadata['execution_date']

        # Load data
        df = pd.read_parquet(f"data/processed/{execution_date}/transactions_clean.parquet")

        logging.info("Computing temporal features")

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract temporal components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Derived temporal features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
        df['is_morning'] = (df['hour'] < 12).astype(int)
        df['is_evening'] = (df['hour'] >= 18).astype(int)

        # Select only the features we created (plus transaction_id for joining)
        temporal_features = df[[
            'transaction_id',
            'hour',
            'day_of_week',
            'day_of_month',
            'month',
            'quarter',
            'is_weekend',
            'is_business_hours',
            'is_morning',
            'is_evening'
        ]]

        # Save intermediate results
        output_dir = f"data/features/intermediate/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/temporal_features.parquet"

        temporal_features.to_parquet(output_path, index=False)

        logging.info(f"Computed {len(temporal_features.columns) - 1} temporal features")
        logging.info(f"Saved to {output_path}")

        return {
            'execution_date': execution_date,
            'num_features': len(temporal_features.columns) - 1,
            'output_path': output_path
        }

    @task
    def merge_all_features(customer_meta: dict, product_meta: dict, temporal_meta: dict):
        """
        Merge all feature sets into a single dataset.

        Joins:
        - Base transactions with customer features (on customer_id)
        - With product features (on product_id)
        - With temporal features (on transaction_id)
        """
        execution_date = customer_meta['execution_date']

        logging.info("Merging all feature sets")

        # Load base transactions
        base_df = pd.read_parquet(f"data/processed/{execution_date}/transactions_clean.parquet")

        # Load feature sets
        customer_features = pd.read_parquet(customer_meta['output_path'])
        product_features = pd.read_parquet(product_meta['output_path'])
        temporal_features = pd.read_parquet(temporal_meta['output_path'])

        # Merge customer features
        df = base_df.merge(customer_features, on='customer_id', how='left')
        logging.info(f"After customer merge: {df.shape}")

        # Merge product features
        df = df.merge(product_features, on='product_id', how='left')
        logging.info(f"After product merge: {df.shape}")

        # Merge temporal features
        df = df.merge(temporal_features, on='transaction_id', how='left')
        logging.info(f"After temporal merge: {df.shape}")

        # Select final feature columns
        # (Exclude raw identifiers and intermediate columns)
        feature_columns = [
            # Target variable (what we want to predict)
            'total_amount',

            # Base features
            'price',
            'quantity',

            # Customer features
            'customer_num_purchases',
            'customer_total_spent',
            'customer_avg_order_value',
            'customer_order_std',
            'customer_tenure_days',
            'customer_purchase_frequency',

            # Product features
            'product_popularity',
            'product_avg_price',
            'product_price_std',
            'product_total_quantity_sold',
            'product_total_revenue',
            'product_price_range',

            # Temporal features
            'hour',
            'day_of_week',
            'month',
            'is_weekend',
            'is_business_hours',
            'is_morning',
            'is_evening',
        ]

        # Also keep IDs for tracking (but won't use in training)
        id_columns = ['transaction_id', 'customer_id', 'product_id', 'timestamp']

        final_df = df[id_columns + feature_columns].copy()

        # Handle any missing values
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

        logging.info(f"Final feature set: {final_df.shape}")
        logging.info(f"Feature columns: {feature_columns}")

        # Save merged features
        output_dir = f"data/features/intermediate/{execution_date}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/merged_features.parquet"

        final_df.to_parquet(output_path, index=False)

        logging.info(f"Merged features saved to {output_path}")

        return {
            'execution_date': execution_date,
            'num_rows': len(final_df),
            'num_features': len(feature_columns),
            'feature_columns': feature_columns,
            'output_path': output_path
        }

    @task
    def create_train_val_test_split(merge_meta: dict, feature_version: str = 'v1'):
        """
        Create train/validation/test splits.

        Strategy:
        - Time-based split (most realistic for time-series data)
        - 70% train, 15% validation, 15% test
        - Ensure no data leakage

        Note: For production, you'd typically use historical data for train,
        recent data for validation, and hold out the most recent for test.
        """
        execution_date = merge_meta['execution_date']
        feature_columns = merge_meta['feature_columns']

        logging.info("Creating train/val/test splits")

        # Load merged features
        df = pd.read_parquet(merge_meta['output_path'])

        # Sort by timestamp to ensure temporal ordering
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Time-based split
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logging.info(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        logging.info(f"Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        logging.info(f"Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

        # Verify no temporal leakage
        assert train_df['timestamp'].max() < val_df['timestamp'].min()
        assert val_df['timestamp'].max() < test_df['timestamp'].min()
        logging.info("✓ Temporal split verified - no leakage")

        # Compute normalization statistics on TRAIN set only
        numeric_features = [col for col in feature_columns if col != 'total_amount']

        train_stats = {
            'mean': train_df[numeric_features].mean().to_dict(),
            'std': train_df[numeric_features].std().to_dict(),
        }

        # Apply normalization to all splits using TRAIN statistics
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for col in numeric_features:
                mean = train_stats['mean'][col]
                std = train_stats['std'][col]
                if std > 0:  # Avoid division by zero
                    split_df[f'{col}_normalized'] = (split_df[col] - mean) / std
                else:
                    split_df[f'{col}_normalized'] = 0

        logging.info("✓ Features normalized using train set statistics")

        return {
            'execution_date': execution_date,
            'feature_version': feature_version,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'train_stats': train_stats,
            'feature_columns': feature_columns,
        }

    @task
    def write_feature_datasets(split_meta: dict):
        """
        Write versioned train/val/test datasets.

        Versioning allows:
        - Reproducing experiments
        - Tracking which features correspond to which models
        - Rolling back to previous feature sets
        """
        execution_date = split_meta['execution_date']
        feature_version = split_meta['feature_version']

        # Create versioned output directory
        output_dir = f"data/features/{feature_version}"
        os.makedirs(output_dir, exist_ok=True)

        # Write splits
        for split_name in ['train', 'val', 'test']:
            df = split_meta[f'{split_name}_df']
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            output_path = f"{output_dir}/{split_name}.parquet"
            df.to_parquet(output_path, index=False)

            logging.info(f"Wrote {split_name}: {len(df)} rows to {output_path}")

        # Write feature metadata
        metadata = {
            'feature_version': feature_version,
            'created_at': datetime.now().isoformat(),
            'execution_date': execution_date,
            'feature_columns': split_meta['feature_columns'],
            'num_features': len(split_meta['feature_columns']),
            'train_size': len(split_meta['train_df']),
            'val_size': len(split_meta['val_df']),
            'test_size': len(split_meta['test_df']),
            'normalization_stats': split_meta['train_stats'],
            'description': 'Customer, product, and temporal features for transaction amount prediction'
        }

        metadata_path = f"{output_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logging.info(f"Feature metadata saved to {metadata_path}")
        logging.info("✓ Feature engineering pipeline complete!")

        return {
            'feature_version': feature_version,
            'output_dir': output_dir,
            'metadata_path': metadata_path,
            **metadata
        }

    # Define task dependencies
    load_meta = load_cleaned_data()

    # Compute features in parallel
    customer_meta = compute_customer_features(load_meta)
    product_meta = compute_product_features(load_meta)
    temporal_meta = compute_temporal_features(load_meta)

    # Merge and split
    merge_meta = merge_all_features(customer_meta, product_meta, temporal_meta)
    split_meta = create_train_val_test_split(merge_meta)
    final_output = write_feature_datasets(split_meta)


# Instantiate the DAG
feature_dag = feature_engineering_pipeline()
```

---

## Part 2: Run the Pipeline

### Step 1: Ensure ETL Pipeline Has Run

```bash
# Check that cleaned data exists
ls data/processed/2024-01-15/transactions_clean.parquet
```

If not, run the ETL pipeline from Lab 2.1 first:
```bash
airflow dags trigger etl_ecommerce_pipeline
```

### Step 2: Trigger Feature Engineering Pipeline

```bash
# Via CLI
airflow dags trigger feature_engineering_pipeline

# Or via UI
# Navigate to http://localhost:8080
# Find "feature_engineering_pipeline"
# Click trigger
```

### Step 3: Verify Output

```bash
# Check feature datasets
ls -lh data/features/v1/

# Should see:
# train.parquet
# val.parquet
# test.parquet
# metadata.json

# View metadata
cat data/features/v1/metadata.json

# Check intermediate features
ls data/features/intermediate/2024-01-15/
```

---

## Part 3: Inspect Features

Create a notebook or script to explore the features:

```python
import pandas as pd
import json

# Load feature metadata
with open('data/features/v1/metadata.json') as f:
    metadata = json.load(f)

print("Feature Version:", metadata['feature_version'])
print("Number of Features:", metadata['num_features'])
print("\nFeatures:", metadata['feature_columns'])

# Load datasets
train = pd.read_parquet('data/features/v1/train.parquet')
val = pd.read_parquet('data/features/v1/val.parquet')
test = pd.read_parquet('data/features/v1/test.parquet')

print(f"\nTrain: {len(train)} rows")
print(f"Val: {len(val)} rows")
print(f"Test: {len(test)} rows")

# Explore train set
print("\nTrain set sample:")
print(train.head())

print("\nFeature statistics:")
print(train.describe())

# Check for data leakage - dates shouldn't overlap
print("\nDate ranges (verifying no leakage):")
print(f"Train: {train['timestamp'].min()} to {train['timestamp'].max()}")
print(f"Val:   {val['timestamp'].min()} to {val['timestamp'].max()}")
print(f"Test:  {test['timestamp'].min()} to {test['timestamp'].max()}")
```

---

## Exercise 1: Add Interaction Features

Interaction features capture relationships between variables:

```python
@task
def compute_interaction_features(merge_meta: dict):
    """
    Compute interaction features.

    Examples:
    - price_per_quantity = price / quantity
    - customer_product_affinity = has this customer bought this product before?
    - price_vs_product_avg = price / product_avg_price
    """
    execution_date = merge_meta['execution_date']
    df = pd.read_parquet(merge_meta['output_path'])

    # Price interactions
    df['price_per_quantity'] = df['price'] / (df['quantity'] + 1e-9)  # Avoid division by zero
    df['price_vs_product_avg'] = df['price'] / (df['product_avg_price'] + 1e-9)

    # Customer interactions
    df['customer_price_ratio'] = df['price'] / (df['customer_avg_order_value'] + 1e-9)

    # Temporal interactions
    df['weekend_premium'] = df['is_weekend'] * df['price']

    # Save...
    # Return updated metadata with new feature columns
```

Add this task to your DAG between `merge_all_features` and `create_train_val_test_split`.

---

## Exercise 2: Add Lagged Features

For time-series prediction, previous values are often predictive:

```python
@task
def compute_lagged_features(merge_meta: dict):
    """
    Compute lagged features (previous purchase behavior).

    For each customer:
    - Amount of previous purchase
    - Days since last purchase
    - Rolling average of last 3 purchases
    """
    execution_date = merge_meta['execution_date']
    df = pd.read_parquet(merge_meta['output_path'])

    # Sort by customer and timestamp
    df = df.sort_values(['customer_id', 'timestamp'])

    # Lagged purchase amount
    df['prev_purchase_amount'] = df.groupby('customer_id')['total_amount'].shift(1)

    # Days since last purchase
    df['prev_purchase_timestamp'] = df.groupby('customer_id')['timestamp'].shift(1)
    df['days_since_last_purchase'] = (
        df['timestamp'] - df['prev_purchase_timestamp']
    ).dt.total_seconds() / 86400

    # Rolling average (last 3 purchases)
    df['rolling_avg_amount'] = df.groupby('customer_id')['total_amount'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Fill NaNs for first purchases
    df['prev_purchase_amount'] = df['prev_purchase_amount'].fillna(0)
    df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(999)

    # Drop intermediate columns
    df = df.drop(columns=['prev_purchase_timestamp'])

    # Save updated dataframe...
```

---

## Exercise 3: Create Multiple Feature Versions

Modify the DAG to support creating different feature versions:

1. **v1**: Basic features (current implementation)
2. **v2**: v1 + interaction features
3. **v3**: v2 + lagged features

**Hint**: Use DAG params:

```python
@dag(
    ...
    params={
        'feature_version': 'v1',
        'include_interactions': False,
        'include_lags': False,
    }
)
def feature_engineering_pipeline():
    ...

    # In tasks, use:
    feature_version = context['params']['feature_version']
    include_interactions = context['params']['include_interactions']
```

Then trigger with different configs:

```bash
airflow dags trigger feature_engineering_pipeline \
    --conf '{"feature_version": "v2", "include_interactions": true}'
```

---

## Challenge: End-to-End Pipeline

Create a master DAG that runs both ETL and feature engineering:

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

with DAG('end_to_end_data_pipeline', ...) as dag:

    run_etl = TriggerDagRunOperator(
        task_id='run_etl_pipeline',
        trigger_dag_id='etl_ecommerce_pipeline',
        wait_for_completion=True,
    )

    run_feature_engineering = TriggerDagRunOperator(
        task_id='run_feature_engineering',
        trigger_dag_id='feature_engineering_pipeline',
        wait_for_completion=True,
    )

    run_etl >> run_feature_engineering
```

---

## Key Takeaways

### Feature Engineering Best Practices

✅ **Split before scaling**: Compute statistics on train set only
✅ **Time-based splits**: For temporal data, respect time ordering
✅ **Version features**: Track what features correspond to which models
✅ **Document features**: Save metadata about feature creation
✅ **Avoid leakage**: Never use future information
✅ **Handle nulls**: Explicit strategy (fill, drop, or flag)

### Common Feature Types

| Type | Examples | Use Case |
|------|----------|----------|
| Aggregations | Sum, mean, count, std | Customer lifetime value |
| Temporal | Hour, day, is_weekend | Seasonality patterns |
| Interactions | price × quantity | Capturing relationships |
| Lagged | Previous value, rolling avg | Time-series prediction |
| Categorical | One-hot, label encoding | Category features |

### Data Leakage Checklist

- ✅ Computed normalization stats on train only?
- ✅ Split data before feature engineering?
- ✅ No future information in features?
- ✅ Temporal ordering respected?
- ✅ Test set truly held out?

---

## Debugging Tips

### Features Have NaN Values

```python
# Check for NaNs
print(train.isnull().sum())

# Identify which features
null_features = train.columns[train.isnull().any()].tolist()
print(f"Features with nulls: {null_features}")

# Fix: Explicit null handling
df[numeric_cols] = df[numeric_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna('unknown')
```

### Splits Overlap in Time

```bash
# Verify no overlap
python -c "
import pandas as pd
train = pd.read_parquet('data/features/v1/train.parquet')
val = pd.read_parquet('data/features/v1/val.parquet')
print('Train max:', train['timestamp'].max())
print('Val min:', val['timestamp'].min())
assert train['timestamp'].max() < val['timestamp'].min()
"
```

---

## Submission Checklist

Before moving to Lab 2.3:

- ✅ Feature engineering DAG runs successfully
- ✅ Train/val/test splits created
- ✅ Feature metadata saved
- ✅ No data leakage in splits
- ✅ At least one exercise completed
- ✅ You can explain each feature's purpose

---

## Next Steps

**What you've built**:
- Complete feature engineering pipeline
- Versioned train/val/test datasets
- Customer, product, and temporal features
- Proper handling of data leakage

**Next lab**:
- Add comprehensive data quality checks
- Implement Great Expectations patterns
- Handle validation failures gracefully

---

**Congratulations! Your data is now ML-ready.** 🎉

**Next**: [Lab 2.3 - Data Quality Checks →](./lab2_3_data_quality.md)
