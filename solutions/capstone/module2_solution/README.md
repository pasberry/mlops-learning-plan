# Capstone Module 2: Data Collection & Storage

## Overview

Sets up data infrastructure for the feed ranking system: user activity tracking, item catalog, and storage.

## Data Sources

### 1. User Activity Stream

**Events Tracked:**
```python
{
    "user_id": "12345",
    "item_id": "67890",
    "event_type": "view|like|share|comment",
    "timestamp": "2025-11-15T10:30:00Z",
    "context": {
        "device": "mobile",
        "platform": "ios",
        "session_id": "abc123"
    },
    "dwell_time_ms": 5420
}
```

**Volume:** ~1M events/day for 100K active users

### 2. Item Catalog

**Item Metadata:**
```python
{
    "item_id": "67890",
    "creator_id": "54321",
    "category": "tech",
    "tags": ["ai", "machine-learning", "python"],
    "created_at": "2025-11-15T08:00:00Z",
    "media_type": "video|image|text",
    "content_embedding": [0.1, 0.2, ...],  # Pre-computed
}
```

**Volume:** ~10M items, growing 50K/day

### 3. User Profiles

**User Data:**
```python
{
    "user_id": "12345",
    "created_at": "2024-01-15T00:00:00Z",
    "demographics": {
        "age_range": "25-34",
        "country": "US",
        "language": "en"
    },
    "preferences": {
        "categories": ["tech", "sports"],
        "following": ["54321", "98765"]
    }
}
```

## Storage Architecture

### Hot Storage (Redis)

**Use Case:** Real-time serving
```
• User embeddings (24h TTL)
• Item embeddings (7d TTL)
• Top rankings per user (12h TTL)
• User features cache (1h TTL)
```

### Warm Storage (PostgreSQL)

**Use Case:** Operational queries
```
• User profiles
• Item catalog
• Recent interactions (30 days)
• Model metadata
```

### Cold Storage (S3/Parquet)

**Use Case:** Training & analytics
```
• Historical interactions (all time)
• Training datasets
• Model checkpoints
• Experiment results
```

## Data Pipeline

### Ingestion

```python
# Kafka consumer for real-time events
@streaming_pipeline
def ingest_user_activity():
    """
    1. Consume from Kafka topic
    2. Validate events
    3. Write to PostgreSQL (recent)
    4. Batch to S3 (historical)
    5. Update Redis cache
    """
```

### Batch Processing

```python
# Daily Airflow DAG
@daily_pipeline
def process_daily_data():
    """
    1. Load yesterday's activity from S3
    2. Aggregate user features
    3. Update item statistics
    4. Generate training dataset
    5. Archive processed data
    """
```

## Data Quality

### Validation Rules

```python
def validate_event(event):
    """
    ✓ user_id exists and valid
    ✓ item_id in catalog
    ✓ timestamp recent (< 1 hour old)
    ✓ event_type in allowed list
    ✓ No duplicate events (dedup key)
    """
```

### Monitoring

```python
metrics = {
    'events_per_second': gauge,
    'validation_failures': counter,
    'storage_lag_seconds': histogram,
    'duplicate_rate': gauge
}
```

## Schema Management

### User Activity Table

```sql
CREATE TABLE user_activity (
    event_id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    event_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    context JSONB,
    dwell_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_timestamp (user_id, timestamp),
    INDEX idx_item_timestamp (item_id, timestamp)
);
```

### Item Catalog Table

```sql
CREATE TABLE items (
    item_id BIGINT PRIMARY KEY,
    creator_id BIGINT NOT NULL,
    category VARCHAR(50),
    tags TEXT[],
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    metadata JSONB,
    INDEX idx_category (category),
    INDEX idx_created_at (created_at)
);
```

## Data Preparation for ML

### Feature Store

```python
# Daily feature computation
def compute_user_features(date):
    """
    Rolling window features (7d, 30d):
    - Total interactions
    - Unique items viewed
    - Category distribution
    - Like/share rates
    - Avg dwell time
    - Active days
    """

def compute_item_features(date):
    """
    Time-windowed features:
    - View count (1d, 7d, 30d)
    - Engagement rate
    - Unique viewers
    - Average rating
    - Freshness score
    """
```

### Training Data Generation

```python
# Generate user-item pairs with labels
def create_training_dataset(date):
    """
    Positive samples:
    - User engaged with item (view, like, share)

    Negative samples:
    - Random items user didn't interact with
    - Hard negatives: Popular items user skipped

    Output: (user_id, item_id, label, features)
    """
```

## Sample Implementation

```python
# data/collection.py
class DataCollector:
    """Collect and validate user activity."""

    def __init__(self):
        self.db = PostgresDB()
        self.cache = RedisCache()
        self.storage = S3Storage()

    def collect_event(self, event):
        # Validate
        if not self.validate(event):
            raise ValidationError()

        # Store in DB
        self.db.insert('user_activity', event)

        # Update cache
        self.cache.hincrby(f"user:{event['user_id']}", 'total_views', 1)

        # Batch to S3 (buffered)
        self.storage.append_to_batch(event)

    def daily_export(self, date):
        # Export day's data to S3
        data = self.db.query(f"SELECT * FROM user_activity WHERE date = '{date}'")
        self.storage.write_parquet(f"activity/{date}.parquet", data)
```

## Key Decisions

**1. Why Kafka for ingestion?**
- High throughput (1M+ events/sec)
- Durability and replay
- Decouples producers/consumers

**2. Why Redis for caching?**
- Sub-millisecond latency
- Built-in TTL support
- Easy invalidation

**3. Why Parquet for archives?**
- Columnar format (fast analytics)
- Great compression
- Schema evolution support

## Integration Points

```
Module 1 (Design) → Data requirements
Module 3 (Features) → Uses collected data
Module 4 (Training) → Reads from storage
Module 5 (Serving) → Queries cache
```

## Learning Outcomes

✅ Event streaming architecture
✅ Multi-tier storage strategy
✅ Data quality validation
✅ Schema design for ML
✅ Feature store concepts
