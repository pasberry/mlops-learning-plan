# Capstone Module 3: Feature Engineering

## Overview

Transforms raw user activity and item data into ML-ready features for the two-tower ranking model.

## Feature Pipeline

### User Features

**Behavioral Features:**
```python
def engineer_user_features(user_id, activity_df):
    """
    Engagement metrics:
    - total_interactions: Count of all events
    - unique_items_viewed: Distinct items
    - like_rate: likes / total_interactions
    - share_rate: shares / total_interactions
    - avg_dwell_time: Mean time spent per item
    - session_frequency: Sessions per day

    Diversity metrics:
    - category_diversity: Entropy of categories
    - creator_diversity: Unique creators followed

    Temporal patterns:
    - peak_hour: Most active hour
    - weekend_ratio: Weekend activity %
    - recency: Days since last interaction
    """
```

**Preference Features:**
```python
def engineer_user_preferences(user_id, activity_df):
    """
    Category affinity:
    - top_categories: [cat1, cat2, cat3] with weights
    - category_vector: Sparse vector of categories

    Creator affinity:
    - following_count: Number of creators followed
    - creator_vector: Sparse vector of creators

    Content type preference:
    - video_rate: % video interactions
    - text_rate: % text interactions
    - image_rate: % image interactions
    """
```

### Item Features

**Content Features:**
```python
def engineer_item_features(item_id, item_data):
    """
    Metadata:
    - category: One-hot encoded
    - tags: Multi-hot encoded
    - content_type: video|text|image
    - length_seconds: Duration/reading time

    Creator features:
    - creator_followers: Creator popularity
    - creator_quality: Historical engagement rate
    - creator_activity: Posts per day

    Quality signals:
    - completion_rate: % users who finish
    - positive_rate: likes / (likes + dislikes)
    - reshare_rate: shares / views
    """
```

**Temporal Features:**
```python
def engineer_item_temporal(item_id, current_time):
    """
    Freshness:
    - age_hours: Hours since published
    - age_score: Decay function of age
    - is_trending: Recent spike in views

    Time patterns:
    - created_hour: Hour of day published
    - created_day: Day of week
    - seasonal: Holiday/event related
    """
```

**Popularity Features:**
```python
def engineer_item_popularity(item_id, activity_df):
    """
    Rolling window metrics:
    - views_1h, views_24h, views_7d
    - engagement_1h, engagement_24h, engagement_7d
    - unique_viewers_24h
    - viral_score: Acceleration metric

    Cohort analysis:
    - new_user_rate: % viewers who are new
    - retention_rate: % who come back
    """
```

## Feature Transformations

### Normalization

```python
# Numeric features
def normalize_features(features):
    """
    StandardScaler for normal distributions:
    - age, dwell_time

    MinMaxScaler for bounded features:
    - like_rate, share_rate (0-1)

    LogTransform for skewed distributions:
    - follower_count, view_count
    """
```

### Encoding

```python
# Categorical features
def encode_categories(features):
    """
    One-hot encoding:
    - category (if < 50 unique)
    - country (if < 100 unique)

    Embedding:
    - creator_id (if many unique)
    - item_id (if many unique)

    Target encoding:
    - category (weighted by historical CTR)
    """
```

## Feature Store

### Implementation

```python
class FeatureStore:
    """Centralized feature storage and serving."""

    def __init__(self):
        self.online_store = RedisCache()
        self.offline_store = S3Storage()

    def write_features(self, entity, features):
        """
        Write to both stores:
        - Online (Redis): For real-time serving
        - Offline (S3): For training
        """
        # Online
        self.online_store.hset(
            f"features:{entity}",
            mapping=features,
            ex=3600  # 1 hour TTL
        )

        # Offline
        self.offline_store.write_parquet(
            f"features/{entity}/date={today}",
            features
        )

    def read_features(self, entity, feature_names):
        """
        Read from online store:
        - Fast for serving
        - Falls back to offline if cache miss
        """
        features = self.online_store.hmget(
            f"features:{entity}",
            feature_names
        )

        if None in features:
            # Cache miss - load from offline
            features = self._load_from_offline(entity, feature_names)
            self._warm_cache(entity, features)

        return features
```

### Feature Versioning

```python
# Track feature versions
feature_metadata = {
    'user_features_v1': {
        'features': ['age', 'like_rate', 'total_interactions'],
        'created': '2025-01-01',
        'schema': {...}
    },
    'user_features_v2': {
        'features': ['age', 'like_rate', 'total_interactions', 'category_diversity'],
        'created': '2025-02-01',
        'schema': {...}
    }
}
```

## Feature Quality

### Validation

```python
def validate_features(features):
    """
    Checks:
    ✓ No missing values (or imputed)
    ✓ Values in expected range
    ✓ No infinity or NaN
    ✓ Correct data types
    ✓ No leakage (future data)
    """
    assert features.isnull().sum() == 0
    assert (features['like_rate'] >= 0).all()
    assert (features['like_rate'] <= 1).all()
    assert not features.isna().any().any()
```

### Monitoring

```python
# Track feature distributions
def monitor_features(features):
    """
    Metrics to track:
    - Mean, median, std per feature
    - Distribution shifts (PSI)
    - Correlation changes
    - Sparsity rates
    """
```

## Training Dataset Creation

### Combining Features

```python
def create_training_dataset(date):
    """
    1. Load user-item interaction pairs
    2. Fetch user features from feature store
    3. Fetch item features from feature store
    4. Combine into training examples
    5. Add labels (engagement/no engagement)
    6. Save to storage
    """

    # Load pairs
    pairs = load_interaction_pairs(date)

    # Fetch features
    user_features = fetch_batch_features('user', pairs['user_id'])
    item_features = fetch_batch_features('item', pairs['item_id'])

    # Combine
    training_data = pd.concat([
        pairs[['user_id', 'item_id', 'label']],
        user_features,
        item_features
    ], axis=1)

    # Save
    save_training_dataset(training_data, date)
```

## Airflow DAG

```python
# Feature engineering DAG
with DAG('feature_engineering', schedule='0 1 * * *') as dag:

    extract_user_activity = PythonOperator(
        task_id='extract_activity',
        python_callable=extract_daily_activity
    )

    compute_user_features = PythonOperator(
        task_id='compute_user_features',
        python_callable=engineer_user_features
    )

    compute_item_features = PythonOperator(
        task_id='compute_item_features',
        python_callable=engineer_item_features
    )

    validate_features = PythonOperator(
        task_id='validate',
        python_callable=validate_feature_quality
    )

    write_feature_store = PythonOperator(
        task_id='write_features',
        python_callable=write_to_feature_store
    )

    # Dependencies
    extract_user_activity >> [compute_user_features, compute_item_features]
    [compute_user_features, compute_item_features] >> validate_features
    validate_features >> write_feature_store
```

## Key Decisions

**1. Why Feature Store?**
- Centralized: Single source of truth
- Reusable: Share features across models
- Consistent: Same features in training/serving

**2. Why Redis for Online Features?**
- Fast: < 1ms latency
- TTL: Auto-expire stale features
- Atomic: Update features safely

**3. Feature Selection Strategy?**
- Start simple: Basic engagement metrics
- Iterate: Add features based on importance
- Remove: Drop low-importance features

## Integration Points

```
Module 2 (Data) → Raw data source
Module 4 (Training) → Consumes features
Module 5 (Serving) → Real-time features
Module 6 (Monitoring) → Feature drift
```

## Learning Outcomes

✅ Behavioral feature engineering
✅ Temporal feature design
✅ Feature store architecture
✅ Feature validation
✅ Training dataset creation
