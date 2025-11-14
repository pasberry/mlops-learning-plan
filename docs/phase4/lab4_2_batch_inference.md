# Lab 4.2: Batch Inference Pipeline

**Goal**: Build an Airflow DAG for scheduled batch inference

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Trained model from Phase 3
- Airflow setup and running
- Understanding of Airflow DAGs

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Create an Airflow DAG for batch scoring
- ✅ Load and score large datasets efficiently
- ✅ Version predictions by date/run
- ✅ Handle incremental scoring
- ✅ Monitor batch job performance
- ✅ Understand batch vs online inference tradeoffs

---

## Background: Why Batch Inference?

### Use Cases

**1. Nightly Recommendations**
```
Every night at 2 AM:
- Load all active users (10M users)
- Generate top 10 product recommendations
- Write to Redis cache
- Users see recommendations in morning
```

**2. Daily Risk Scoring**
```
Every day after market close:
- Score all accounts for fraud risk
- Flag high-risk transactions
- Alert fraud team
```

**3. Weekly Campaign Targeting**
```
Every Monday:
- Score all customers for churn likelihood
- Export top 10K at-risk customers
- Send to marketing team
```

### Batch vs Online Inference

| Aspect | Online | Batch |
|--------|--------|-------|
| **Latency** | < 100ms | Minutes to hours |
| **Throughput** | 100s-1000s QPS | Millions per run |
| **Use case** | Real-time decisions | Bulk scoring |
| **Cost** | Higher per prediction | Lower per prediction |
| **Freshness** | Always current | Updated on schedule |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Batch Inference DAG                     │
│                                                      │
│  ┌────────────┐      ┌───────────────┐             │
│  │  Read      │  ──> │  Load Model   │             │
│  │  Unscored  │      │  & Features   │             │
│  │  Data      │      └───────┬───────┘             │
│  └────────────┘              │                      │
│                              ▼                      │
│                      ┌───────────────┐              │
│                      │  Generate     │              │
│                      │  Predictions  │              │
│                      │  (Batched)    │              │
│                      └───────┬───────┘              │
│                              │                      │
│                              ▼                      │
│  ┌────────────┐      ┌───────────────┐             │
│  │  Write     │  <── │  Post-process │             │
│  │  Results   │      │  & Validate   │             │
│  │  (Dated)   │      └───────────────┘             │
│  └────────────┘                                     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Part 1: Create Batch Inference Module

Create `ml/batch/batch_inference.py`:

```python
"""
Batch inference utilities
Efficiently score large datasets
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """
    Efficiently score large datasets in batches
    """
    def __init__(
        self,
        model_path: str,
        feature_names: List[str],
        batch_size: int = 1024,
        device: str = "cpu"
    ):
        """
        Initialize batch inference engine

        Args:
            model_path: Path to model checkpoint
            feature_names: List of feature names in order
            batch_size: Number of samples per batch
            device: 'cpu' or 'cuda'
        """
        self.model_path = model_path
        self.feature_names = feature_names
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.model_version = None

        self._load_model()

    def _load_model(self):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Import model architecture (adjust to your model)
        from ml.serving.model_service import SimpleClassifier

        input_dim = checkpoint.get('input_dim', len(self.feature_names))
        hidden_dim = checkpoint.get('hidden_dim', 64)
        output_dim = checkpoint.get('output_dim', 2)

        self.model = SimpleClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)

        self.model_version = checkpoint.get('version', 'unknown')
        logger.info(f"Model loaded: {self.model_version}")

    def score_dataframe(
        self,
        df: pd.DataFrame,
        id_column: str = 'id'
    ) -> pd.DataFrame:
        """
        Score a pandas DataFrame

        Args:
            df: Input dataframe with features
            id_column: Name of ID column

        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Scoring {len(df)} samples...")

        # Validate features exist
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features in correct order
        X = df[self.feature_names].values.astype(np.float32)

        # Score in batches
        all_predictions = []
        all_probabilities = []

        num_batches = (len(X) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(X))

            batch_X = torch.tensor(X[start_idx:end_idx], device=self.device)

            with torch.no_grad():
                logits = self.model(batch_X)
                probs = torch.softmax(logits, dim=1)

            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            probabilities = probs.cpu().numpy()

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {end_idx}/{len(X)} samples")

        # Create output dataframe
        result_df = df[[id_column]].copy()
        result_df['prediction'] = all_predictions
        result_df['prediction_proba'] = [
            prob[pred] for prob, pred in zip(all_probabilities, all_predictions)
        ]

        # Add class probabilities
        num_classes = all_probabilities[0].shape[0]
        for class_idx in range(num_classes):
            result_df[f'prob_class_{class_idx}'] = [
                prob[class_idx] for prob in all_probabilities
            ]

        result_df['model_version'] = self.model_version
        result_df['scored_at'] = datetime.utcnow()

        logger.info(f"Scoring complete: {len(result_df)} predictions")

        return result_df

    def score_csv(
        self,
        input_path: str,
        output_path: str,
        id_column: str = 'id',
        chunksize: Optional[int] = None
    ):
        """
        Score a CSV file (supports large files with chunking)

        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
            id_column: Name of ID column
            chunksize: If set, process file in chunks
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if chunksize is None:
            # Load entire file
            df = pd.read_csv(input_path)
            result_df = self.score_dataframe(df, id_column=id_column)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results written to {output_path}")

        else:
            # Process in chunks (for very large files)
            logger.info(f"Processing file in chunks of {chunksize}")

            first_chunk = True
            for chunk_df in pd.read_csv(input_path, chunksize=chunksize):
                result_chunk = self.score_dataframe(chunk_df, id_column=id_column)

                # Write chunk to CSV
                result_chunk.to_csv(
                    output_path,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False

            logger.info(f"Results written to {output_path}")


def get_unscored_data(
    data_dir: str,
    date_partition: str
) -> pd.DataFrame:
    """
    Load unscored data for a specific date

    Args:
        data_dir: Base data directory
        date_partition: Date string (YYYY-MM-DD)

    Returns:
        DataFrame with unscored data
    """
    file_path = Path(data_dir) / "unscored" / date_partition / "data.csv"

    if not file_path.exists():
        logger.warning(f"No data found for {date_partition}")
        return pd.DataFrame()

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} samples")

    return df


def write_predictions(
    predictions_df: pd.DataFrame,
    output_dir: str,
    date_partition: str
):
    """
    Write predictions to dated partition

    Args:
        predictions_df: DataFrame with predictions
        output_dir: Base output directory
        date_partition: Date string (YYYY-MM-DD)
    """
    output_path = Path(output_dir) / "predictions" / date_partition / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Predictions written to {output_path}")

    # Write summary stats
    stats = {
        'total_predictions': len(predictions_df),
        'class_distribution': predictions_df['prediction'].value_counts().to_dict(),
        'avg_confidence': float(predictions_df['prediction_proba'].mean()),
        'date_partition': date_partition,
        'scored_at': datetime.utcnow().isoformat()
    }

    stats_path = output_path.parent / "stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats written to {stats_path}")
```

---

## Part 2: Create Airflow DAG

Create `dags/batch_inference_dag.py`:

```python
"""
Batch Inference DAG
Daily batch scoring of new data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.batch.batch_inference import (
    BatchInferenceEngine,
    get_unscored_data,
    write_predictions
)

logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_path': 'models/production/model_latest.pt',
    'feature_names': [
        'age', 'income', 'tenure_days', 'num_purchases',
        'avg_transaction_value', 'days_since_last_purchase'
    ],
    'data_dir': 'data',
    'batch_size': 1024,
    'device': 'cpu'
}


def check_data_availability(**context):
    """
    Check if data is available for scoring
    """
    execution_date = context['ds']  # YYYY-MM-DD
    logger.info(f"Checking data availability for {execution_date}")

    df = get_unscored_data(
        data_dir=CONFIG['data_dir'],
        date_partition=execution_date
    )

    if df.empty:
        raise ValueError(f"No data available for {execution_date}")

    # Push row count to XCom
    context['ti'].xcom_push(key='num_samples', value=len(df))
    logger.info(f"Found {len(df)} samples to score")


def load_and_score_data(**context):
    """
    Load data and generate predictions
    """
    execution_date = context['ds']
    logger.info(f"Starting batch scoring for {execution_date}")

    # Load data
    df = get_unscored_data(
        data_dir=CONFIG['data_dir'],
        date_partition=execution_date
    )

    # Initialize inference engine
    engine = BatchInferenceEngine(
        model_path=CONFIG['model_path'],
        feature_names=CONFIG['feature_names'],
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device']
    )

    # Score data
    predictions_df = engine.score_dataframe(df, id_column='id')

    # Write predictions
    write_predictions(
        predictions_df=predictions_df,
        output_dir=CONFIG['data_dir'],
        date_partition=execution_date
    )

    # Push metrics to XCom
    metrics = {
        'total_scored': len(predictions_df),
        'class_0_count': int((predictions_df['prediction'] == 0).sum()),
        'class_1_count': int((predictions_df['prediction'] == 1).sum()),
        'avg_confidence': float(predictions_df['prediction_proba'].mean()),
        'model_version': predictions_df['model_version'].iloc[0]
    }

    context['ti'].xcom_push(key='scoring_metrics', value=metrics)
    logger.info(f"Scoring complete: {metrics}")


def validate_predictions(**context):
    """
    Validate prediction quality
    """
    execution_date = context['ds']
    metrics = context['ti'].xcom_pull(
        task_ids='score_data',
        key='scoring_metrics'
    )

    logger.info(f"Validating predictions for {execution_date}")
    logger.info(f"Metrics: {metrics}")

    # Validation checks
    checks = []

    # Check 1: Confidence threshold
    if metrics['avg_confidence'] < 0.5:
        checks.append(f"Low confidence: {metrics['avg_confidence']:.2f}")

    # Check 2: Class balance (warn if heavily skewed)
    total = metrics['total_scored']
    class_0_pct = metrics['class_0_count'] / total
    class_1_pct = metrics['class_1_count'] / total

    if class_0_pct > 0.95 or class_1_pct > 0.95:
        checks.append(f"Skewed distribution: {class_0_pct:.1%} vs {class_1_pct:.1%}")

    # Check 3: Ensure predictions were generated
    if metrics['total_scored'] == 0:
        raise ValueError("No predictions generated!")

    # Log warnings
    if checks:
        logger.warning(f"Validation warnings: {checks}")
    else:
        logger.info("All validation checks passed")


def send_completion_notification(**context):
    """
    Send notification that batch scoring is complete
    """
    execution_date = context['ds']
    metrics = context['ti'].xcom_pull(
        task_ids='score_data',
        key='scoring_metrics'
    )

    logger.info(f"Batch inference complete for {execution_date}")
    logger.info(f"Scored {metrics['total_scored']} samples")
    logger.info(f"Model version: {metrics['model_version']}")

    # In production, send to Slack/email/monitoring system
    # For now, just log
    message = f"""
    Batch Inference Complete
    ========================
    Date: {execution_date}
    Samples scored: {metrics['total_scored']:,}
    Class 0: {metrics['class_0_count']:,} ({metrics['class_0_count']/metrics['total_scored']:.1%})
    Class 1: {metrics['class_1_count']:,} ({metrics['class_1_count']/metrics['total_scored']:.1%})
    Avg confidence: {metrics['avg_confidence']:.2%}
    Model version: {metrics['model_version']}
    """
    logger.info(message)


# Define DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='batch_inference',
    default_args=default_args,
    description='Daily batch inference pipeline',
    schedule='0 2 * * *',  # Run daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['inference', 'production'],
) as dag:

    # Task 1: Check data availability
    check_data = PythonOperator(
        task_id='check_data_availability',
        python_callable=check_data_availability,
        provide_context=True,
    )

    # Task 2: Score data
    score_data = PythonOperator(
        task_id='score_data',
        python_callable=load_and_score_data,
        provide_context=True,
    )

    # Task 3: Validate predictions
    validate = PythonOperator(
        task_id='validate_predictions',
        python_callable=validate_predictions,
        provide_context=True,
    )

    # Task 4: Send notification
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_completion_notification,
        provide_context=True,
    )

    # Define dependencies
    check_data >> score_data >> validate >> notify
```

---

## Part 3: Test Data Generator

Create `ml/batch/generate_test_data.py`:

```python
"""
Generate test data for batch inference
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_test_data(
    num_samples: int = 10000,
    output_dir: str = 'data',
    date: str = None
):
    """
    Generate synthetic test data

    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory
        date: Date partition (YYYY-MM-DD), defaults to today
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    np.random.seed(42)

    # Generate features
    data = {
        'id': [f'user_{i:06d}' for i in range(num_samples)],
        'age': np.random.randint(18, 80, num_samples),
        'income': np.random.lognormal(11, 0.5, num_samples),
        'tenure_days': np.random.randint(1, 1000, num_samples),
        'num_purchases': np.random.poisson(5, num_samples),
        'avg_transaction_value': np.random.lognormal(4, 0.8, num_samples),
        'days_since_last_purchase': np.random.exponential(30, num_samples)
    }

    df = pd.DataFrame(data)

    # Write to dated partition
    output_path = Path(output_dir) / 'unscored' / date / 'data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples")
    print(f"Written to {output_path}")


if __name__ == '__main__':
    # Generate data for today
    generate_test_data(num_samples=10000)

    # Generate data for yesterday (testing backfill)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    generate_test_data(num_samples=5000, date=yesterday)
```

---

## Part 4: Running Batch Inference

### 1. Generate Test Data

```bash
python ml/batch/generate_test_data.py
```

### 2. Test Locally

```python
# test_batch_inference.py
from ml.batch.batch_inference import BatchInferenceEngine
import pandas as pd

# Initialize engine
engine = BatchInferenceEngine(
    model_path='models/production/model_latest.pt',
    feature_names=['age', 'income', 'tenure_days',
                   'num_purchases', 'avg_transaction_value',
                   'days_since_last_purchase'],
    batch_size=1024
)

# Load test data
df = pd.read_csv('data/unscored/2024-01-15/data.csv')

# Score
predictions = engine.score_dataframe(df, id_column='id')

print(predictions.head())
print(f"\nScored {len(predictions)} samples")
print(f"Class distribution:\n{predictions['prediction'].value_counts()}")
```

### 3. Run Airflow DAG

```bash
# Trigger DAG manually
airflow dags trigger batch_inference

# Or trigger for specific date
airflow dags trigger batch_inference --exec-date 2024-01-15

# Monitor in UI
# http://localhost:8080
```

---

## Exercises

### Exercise 1: Add Incremental Scoring

Only score new/updated records:

```python
def get_unscored_data(data_dir: str, date_partition: str) -> pd.DataFrame:
    """Load only records not already scored"""

    # Load new data
    new_data = pd.read_csv(f"{data_dir}/unscored/{date_partition}/data.csv")

    # Load previous predictions
    prev_predictions_path = Path(f"{data_dir}/predictions/{date_partition}/predictions.csv")

    if prev_predictions_path.exists():
        prev_predictions = pd.read_csv(prev_predictions_path)
        already_scored = set(prev_predictions['id'])

        # Filter to only new IDs
        new_data = new_data[~new_data['id'].isin(already_scored)]

    return new_data
```

### Exercise 2: Add Performance Monitoring

Track inference speed and resource usage:

```python
import time
import psutil

def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # ... scoring logic ...

    elapsed_time = time.time() - start_time
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024

    logger.info(f"Performance metrics:")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  Time: {elapsed_time:.2f}s")
    logger.info(f"  Throughput: {len(df)/elapsed_time:.0f} samples/sec")
    logger.info(f"  Memory: {end_memory - start_memory:.1f} MB")

    return result_df
```

### Exercise 3: Add Data Quality Checks

Validate input data before scoring:

```python
def validate_input_data(df: pd.DataFrame, feature_names: List[str]):
    """Validate data quality before scoring"""

    issues = []

    # Check for missing values
    missing = df[feature_names].isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

    # Check for outliers (example: age)
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 0) | (df['age'] > 120)).sum()
        if invalid_age > 0:
            issues.append(f"Invalid age values: {invalid_age}")

    # Check for duplicates
    duplicates = df.duplicated(subset=['id']).sum()
    if duplicates > 0:
        issues.append(f"Duplicate IDs: {duplicates}")

    if issues:
        raise ValueError(f"Data quality issues: {issues}")
```

---

## Production Considerations

### 1. Handling Large Files

For files too large to fit in memory:

```python
# Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    predictions = engine.score_dataframe(chunk)
    # Write incrementally
    predictions.to_csv('output.csv', mode='a', header=first_chunk)
```

### 2. Distributed Processing

Use Spark for very large datasets:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("batch_inference").getOrCreate()

# Read data
df = spark.read.csv("s3://bucket/data/*.csv", header=True)

# Score in parallel
predictions = df.rdd.map(lambda row: score_row(row)).toDF()

# Write results
predictions.write.parquet("s3://bucket/predictions/")
```

### 3. Monitoring and Alerting

```python
def check_prediction_drift(**context):
    """Compare today's predictions to historical baseline"""

    today_metrics = context['ti'].xcom_pull(key='scoring_metrics')

    # Load historical metrics
    historical_avg_confidence = 0.75

    # Alert if significant deviation
    if abs(today_metrics['avg_confidence'] - historical_avg_confidence) > 0.1:
        send_alert(f"Prediction confidence drift detected")
```

---

## Key Takeaways

✅ **Batch for throughput**: Process millions of records efficiently
✅ **Version outputs**: Date-partition predictions for reproducibility
✅ **Monitor performance**: Track speed, memory, quality
✅ **Validate inputs and outputs**: Catch issues early
✅ **Handle failures gracefully**: Retry logic, logging

---

## Next Steps

- ✅ Complete this lab and test batch scoring
- ✅ Share your implementation for review
- → Move to **Lab 4.3: Monitoring & Drift Detection**

---

**Congratulations! You've built a production batch inference pipeline! 🚀**

**Next**: [Lab 4.3 - Monitoring →](./lab4_3_monitoring.md)
