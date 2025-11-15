# Lab 2.2 Solution: Feature Engineering Pipeline

This solution provides a comprehensive feature engineering pipeline that creates rich features from e-commerce data for machine learning.

## Overview

The feature engineering pipeline creates multiple types of features:

1. **RFM Features**: Recency, Frequency, Monetary analysis
2. **Temporal Features**: Time-based patterns and trends
3. **Behavioral Features**: Purchase behavior and preferences
4. **Customer Value Features**: CLV, engagement, and value metrics
5. **Aggregated Features**: Time-windowed summary statistics

## Files

- `feature_utils.py` - Feature engineering utility functions
- `feature_engineering_dag.py` - Complete Airflow DAG for feature engineering
- `README.md` - This file

## Prerequisites

```bash
# Install required packages
pip install apache-airflow pandas numpy scikit-learn

# Ensure you have completed Lab 2.1 (ETL Pipeline)
# The feature engineering pipeline depends on cleaned data from Lab 2.1
```

## Setup

### 1. Run ETL Pipeline First

The feature engineering pipeline requires cleaned data from Lab 2.1:

```bash
# Make sure Lab 2.1 ETL pipeline has run and produced cleaned data
# Expected files:
#   /tmp/ecommerce_data/cleaned/orders.csv
#   /tmp/ecommerce_data/cleaned/customers.csv
```

### 2. Copy Files to Airflow

```bash
# Set AIRFLOW_HOME
export AIRFLOW_HOME=~/airflow

# Copy files to DAGs directory
cp feature_engineering_dag.py $AIRFLOW_HOME/dags/
cp feature_utils.py $AIRFLOW_HOME/dags/
```

### 3. Start Airflow (if not already running)

**Terminal 1 - Webserver:**
```bash
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
airflow scheduler
```

## Running the Pipeline

### Option 1: Via Airflow UI

1. Go to http://localhost:8080
2. Find the DAG named `ecommerce_feature_engineering`
3. Toggle the DAG to "On"
4. Click the "Play" button to trigger a run
5. Monitor execution in the Graph view

### Option 2: Via Command Line

```bash
# Trigger the DAG
airflow dags trigger ecommerce_feature_engineering

# Test individual tasks
airflow tasks test ecommerce_feature_engineering load_data 2024-01-01
airflow tasks test ecommerce_feature_engineering create_rfm_features 2024-01-01
airflow tasks test ecommerce_feature_engineering create_temporal_features 2024-01-01
airflow tasks test ecommerce_feature_engineering create_behavioral_features 2024-01-01
airflow tasks test ecommerce_feature_engineering combine_features 2024-01-01
```

### Option 3: Use Feature Utils Standalone

You can use the feature engineering functions independently:

```python
from feature_utils import (
    calculate_rfm_features,
    calculate_temporal_features,
    combine_all_features
)
import pandas as pd

# Load your data
orders = pd.read_csv('/tmp/ecommerce_data/cleaned/orders.csv')
customers = pd.read_csv('/tmp/ecommerce_data/cleaned/customers.csv')

# Create RFM features
rfm = calculate_rfm_features(orders)
print(rfm.head())

# Create all features
all_features = combine_all_features(orders, customers)
print(f"Created {len(all_features.columns)} features")
```

## Feature Catalog

### RFM Features (11 features)

```
recency_days              - Days since last purchase
frequency                 - Total number of orders
monetary_total            - Total spending
monetary_mean             - Average order value
monetary_std              - Standard deviation of order values
monetary_min              - Minimum order value
monetary_max              - Maximum order value
recency_score            - Recency score (1-5)
frequency_score          - Frequency score (1-5)
monetary_score           - Monetary score (1-5)
rfm_score                - Combined RFM score
customer_segment         - Customer segment (Champions, Loyal, etc.)
```

**Customer Segments:**
- **Champions** (RFM 4-5): Best customers, buy frequently, high value
- **Loyal** (RFM 3-4): Regular customers, good value
- **Needs Attention** (RFM 2-3): Average customers, need engagement
- **At Risk** (RFM 0-2): Low engagement, may churn

### Temporal Features (13 features)

```
first_order_date          - Date of first purchase
last_order_date           - Date of most recent purchase
customer_lifetime_days    - Days since first purchase
avg_days_between_orders   - Average inter-purchase interval
std_days_between_orders   - Std dev of inter-purchase intervals
min_days_between_orders   - Minimum days between orders
max_days_between_orders   - Maximum days between orders
most_common_order_day     - Most frequent day of week (0=Mon, 6=Sun)
most_common_order_hour    - Most frequent hour of day (0-23)
weekend_order_ratio       - Proportion of weekend orders
weekday_orders           - Number of weekday orders
weekend_orders           - Number of weekend orders
```

### Behavioral Features (12 features)

```
most_purchased_category   - Top product category
num_unique_categories     - Number of different categories purchased
category_diversity_score  - Category diversity (0-1)
avg_items_per_order      - Average items per order
total_items_purchased    - Total items bought
max_items_in_order       - Maximum items in single order
cancelled_orders         - Number of cancelled orders
delivered_orders         - Number of delivered orders
cancellation_rate        - Cancellation rate (0-1)
delivery_rate            - Delivery success rate (0-1)
bulk_buyer_score         - Proportion of bulk orders (>3 items)
```

### Customer Value Features (10+ features)

```
total_revenue            - Total lifetime value
avg_order_value          - Average order value
tenure_days              - Days as a customer
num_orders               - Total number of orders
orders_per_month         - Purchase frequency per month
value_trend              - Recent vs historical spending trend
engagement_score         - Engagement metric
days_since_last_order    - Recency
predicted_next_purchase_days - Estimated days to next purchase
account_age_days         - Days since registration (if customer data available)
is_premium               - Premium status (if available)
```

### Aggregated Features (12+ features)

**30-day window:**
```
orders_last_30d           - Orders in last 30 days
revenue_last_30d          - Revenue in last 30 days
avg_order_value_last_30d  - Avg order value (30d)
max_order_value_last_30d  - Max order value (30d)
items_purchased_last_30d  - Items purchased (30d)
top_category_last_30d     - Most purchased category (30d)
```

**90-day window:**
```
orders_last_90d           - Orders in last 90 days
revenue_last_90d          - Revenue in last 90 days
avg_order_value_last_90d  - Avg order value (90d)
max_order_value_last_90d  - Max order value (90d)
items_purchased_last_90d  - Items purchased (90d)
top_category_last_90d     - Most purchased category (90d)
```

## Output Directory Structure

```
/tmp/ecommerce_data/features/
├── rfm_features.csv
├── temporal_features.csv
├── behavioral_features.csv
├── customer_value_features.csv
├── aggregated_features_30d.csv
├── aggregated_features_90d.csv
├── combined_features.csv
├── feature_summary.json
├── feature_engineering_report.json
└── feature_store/
    ├── features_v20240101_120000.csv
    ├── features_latest.csv
    ├── metadata_v20240101_120000.json
    └── metadata_latest.json
```

## Pipeline Tasks Explained

### Task 1: Load Data
Loads cleaned data from the ETL pipeline output.

### Task 2-6: Create Individual Feature Sets
Creates different types of features in parallel:
- RFM features
- Temporal features
- Behavioral features
- Customer value features
- Aggregated features

### Task 7: Combine Features
Merges all feature sets into a unified feature matrix.

### Task 8: Create Feature Store
Versions and stores features with metadata for ML consumption.

### Task 9: Generate Report
Creates comprehensive report with feature statistics.

## Using Features for Machine Learning

### Load Features

```python
import pandas as pd

# Load latest features
features = pd.read_csv('/tmp/ecommerce_data/features/feature_store/features_latest.csv')

# Load specific version
features = pd.read_csv('/tmp/ecommerce_data/features/feature_store/features_v20240101_120000.csv')
```

### Prepare for ML

```python
# Select numeric features
numeric_features = features.select_dtypes(include=['float64', 'int64'])

# Handle categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['customer_segment_encoded'] = le.fit_transform(features['customer_segment'])
features['most_purchased_category_encoded'] = le.fit_transform(features['most_purchased_category'])

# Split features and target
X = features.drop(['customer_id', 'customer_segment'], axis=1)
y = features['customer_segment']  # or any other target

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Example: Customer Segmentation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
clustering_features = [
    'rfm_score', 'total_revenue', 'frequency',
    'avg_days_between_orders', 'category_diversity_score'
]

X = features[clustering_features].fillna(0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
features['cluster'] = kmeans.fit_predict(X_scaled)

print(features.groupby('cluster')[clustering_features].mean())
```

### Example: Churn Prediction

```python
from sklearn.ensemble import RandomForestClassifier

# Define churn (no orders in last 30 days)
features['is_churned'] = (features['orders_last_30d'] == 0).astype(int)

# Select predictive features
predictive_features = [
    'recency_days', 'frequency', 'monetary_total',
    'avg_days_between_orders', 'engagement_score',
    'orders_last_90d', 'category_diversity_score'
]

X = features[predictive_features].fillna(0)
y = features['is_churned']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Customization

### Add Custom Features

Edit `feature_utils.py` to add new feature functions:

```python
def calculate_custom_features(orders_df, customer_id_col='customer_id'):
    """Calculate custom domain-specific features."""
    custom_features = []

    for customer_id, group in orders_df.groupby(customer_id_col):
        # Your custom logic here
        feature = {
            customer_id_col: customer_id,
            'custom_metric_1': ...,
            'custom_metric_2': ...
        }
        custom_features.append(feature)

    return pd.DataFrame(custom_features)
```

Then add to the DAG:

```python
def create_custom_features(**context):
    orders = pd.read_csv(f'{FEATURES_DIR}/temp_orders.csv')
    custom = calculate_custom_features(orders)
    custom.to_csv(f'{FEATURES_DIR}/custom_features.csv', index=False)
```

### Adjust Time Windows

```python
# Create features for different time periods
agg_7d = create_aggregated_features(orders, time_window_days=7)
agg_180d = create_aggregated_features(orders, time_window_days=180)
```

## Monitoring

### Check Feature Quality

```bash
# View feature summary
cat /tmp/ecommerce_data/features/feature_summary.json

# Check feature distributions
python -c "
import pandas as pd
features = pd.read_csv('/tmp/ecommerce_data/features/combined_features.csv')
print(features.describe())
"
```

### Track Feature Drift

```python
import pandas as pd
import json

# Compare two feature versions
v1 = pd.read_csv('/tmp/ecommerce_data/features/feature_store/features_v20240101_120000.csv')
v2 = pd.read_csv('/tmp/ecommerce_data/features/feature_store/features_v20240102_120000.csv')

# Compare distributions
for col in v1.select_dtypes(include=['float64', 'int64']).columns:
    drift = abs(v1[col].mean() - v2[col].mean()) / v1[col].std()
    if drift > 0.5:  # Significant drift
        print(f"Feature {col} has drifted: {drift:.2f}")
```

## Best Practices

1. **Feature Documentation**: Always document what each feature represents
2. **Feature Versioning**: Use timestamps to version feature sets
3. **Feature Validation**: Check for NaN, inf, and outliers
4. **Feature Selection**: Not all features are useful - select based on importance
5. **Feature Monitoring**: Track feature distributions over time
6. **Reproducibility**: Save preprocessing parameters with features

## Troubleshooting

### Missing Input Data

```bash
# Ensure Lab 2.1 ETL pipeline has run
ls -l /tmp/ecommerce_data/cleaned/

# If missing, run ETL pipeline first
airflow dags trigger ecommerce_etl_pipeline
```

### Import Errors

```bash
# Ensure feature_utils.py is in DAGs directory
cp feature_utils.py $AIRFLOW_HOME/dags/

# Or add to Python path
export PYTHONPATH=$PYTHONPATH:$AIRFLOW_HOME/dags
```

### Memory Issues

For large datasets:

```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_features(chunk)
```

## Next Steps

1. **Lab 2.3**: Add data quality validation for features
2. **Lab 2.4**: Schedule feature engineering pipeline
3. **Use in ML**: Train models using engineered features
4. **Feature Store**: Deploy to production feature store (Feast, Tecton)
5. **Real-time Features**: Add streaming feature computation

## Resources

- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)
- [RFM Analysis Guide](https://clevertap.com/blog/rfm-analysis/)
- [Feast Feature Store](https://feast.dev/)

## License

MIT License - Free to use for learning purposes
