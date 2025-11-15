# Lab 4.3: Monitoring & Drift Detection

**Goal**: Implement production monitoring and drift detection

**Estimated Time**: 120-150 minutes

**Prerequisites**:
- Batch inference pipeline from Lab 4.2
- Understanding of statistical distributions
- NumPy and SciPy installed

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Track feature and prediction distributions
- ✅ Implement drift detection (PSI, KL divergence)
- ✅ Create alerting logic for drift events
- ✅ Build monitoring dashboards
- ✅ Understand when to retrain models
- ✅ Store and visualize metrics over time

---

## Background: Why Models Fail in Production

### The Core Problem

```
Training time (2023):          Production (2024):
─────────────────────          ─────────────────
Age: 25-45 (μ=35)       ≠     Age: 18-30 (μ=24)
Income: $50-100k        ≠     Income: $30-70k
Urban: 80%              ≠     Urban: 60%

Result: Model accuracy drops from 85% → 68%
```

**Models are static, but data is dynamic.**

### Types of Drift

#### 1. Data Drift (Covariate Shift)
**The input distribution changes**

```python
# Training: P(X)
X_train ~ N(μ=50, σ=10)

# Production: P'(X) ≠ P(X)
X_prod ~ N(μ=40, σ=15)
```

**Impact**: Model receives out-of-distribution inputs
**Detection**: Compare feature distributions

#### 2. Concept Drift
**The relationship between X and Y changes**

```python
# Training: Y = f(X)
# Price is main factor for purchases

# Production: Y = g(X), where g ≠ f
# Reviews now matter more than price
```

**Impact**: Model predictions become inaccurate
**Detection**: Monitor prediction accuracy (requires labels)

#### 3. Prediction Drift
**The output distribution changes**

```python
# Training predictions
P(Y=1) = 0.4

# Production predictions
P(Y=1) = 0.7  # Significantly different
```

**Impact**: May indicate data or concept drift
**Detection**: Compare prediction distributions

---

## Drift Detection Techniques

### 1. Population Stability Index (PSI)

**Formula**:
```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```

**Interpretation**:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.2: Moderate change, investigate
- PSI ≥ 0.2: Significant change, retrain recommended

**Intuition**: Measures how much the distribution has shifted

### 2. Kullback-Leibler (KL) Divergence

**Formula**:
```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

**Interpretation**:
- KL = 0: Distributions are identical
- KL > 0: Distributions differ (higher = more different)

**Intuition**: Information lost when Q approximates P

### 3. Statistical Tests

**Kolmogorov-Smirnov Test**:
- Compares continuous distributions
- Returns p-value (< 0.05 = significantly different)

**Chi-Squared Test**:
- Compares categorical distributions
- Returns p-value (< 0.05 = significantly different)

---

## Part 1: Drift Detection Module

Create `ml/monitoring/drift_detection.py`:

```python
"""
Drift detection utilities
Implements PSI, KL divergence, and statistical tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.special import kl_div
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect distribution drift using multiple methods
    """

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
        eps: float = 1e-10
    ) -> float:
        """
        Calculate Population Stability Index

        Args:
            expected: Reference distribution (training data)
            actual: Current distribution (production data)
            buckets: Number of bins for discretization
            eps: Small constant to avoid log(0)

        Returns:
            PSI score
        """
        # Create bins based on expected distribution
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            buckets + 1
        )

        # Calculate percentages in each bin
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Add small constant to avoid division by zero
        expected_percents = expected_percents + eps
        actual_percents = actual_percents + eps

        # Calculate PSI
        psi = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )

        return float(psi)

    @staticmethod
    def calculate_kl_divergence(
        p: np.ndarray,
        q: np.ndarray,
        bins: int = 50,
        eps: float = 1e-10
    ) -> float:
        """
        Calculate Kullback-Leibler divergence

        Args:
            p: Reference distribution
            q: Current distribution
            bins: Number of bins
            eps: Small constant to avoid log(0)

        Returns:
            KL divergence score
        """
        # Create bins
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        bins_array = np.linspace(min_val, max_val, bins + 1)

        # Calculate histograms (normalized)
        p_hist = np.histogram(p, bins=bins_array, density=True)[0]
        q_hist = np.histogram(q, bins=bins_array, density=True)[0]

        # Normalize to probabilities
        p_hist = p_hist / (p_hist.sum() + eps)
        q_hist = q_hist / (q_hist.sum() + eps)

        # Add epsilon to avoid log(0)
        p_hist = p_hist + eps
        q_hist = q_hist + eps

        # Calculate KL divergence
        kl = np.sum(kl_div(p_hist, q_hist))

        return float(kl)

    @staticmethod
    def ks_test(
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for continuous distributions

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    @staticmethod
    def chi2_test(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, float]:
        """
        Chi-squared test for categorical or binned data

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for continuous data

        Returns:
            (statistic, p_value)
        """
        # Create bins
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1
        )

        # Bin the data
        ref_binned = np.histogram(reference, bins=breakpoints)[0]
        curr_binned = np.histogram(current, bins=breakpoints)[0]

        # Chi-squared test
        statistic, p_value = stats.chisquare(curr_binned, ref_binned)

        return float(statistic), float(p_value)


class FeatureMonitor:
    """
    Monitor feature distributions over time
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: List[str],
        categorical_features: List[str] = None
    ):
        """
        Initialize feature monitor

        Args:
            reference_data: Training/reference data
            feature_names: List of features to monitor
            categorical_features: List of categorical feature names
        """
        self.reference_data = reference_data[feature_names]
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.detector = DriftDetector()

        # Store reference statistics
        self.reference_stats = self._compute_statistics(self.reference_data)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics"""
        stats = {}

        for feature in self.feature_names:
            stats[feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'median': float(df[feature].median()),
                'q25': float(df[feature].quantile(0.25)),
                'q75': float(df[feature].quantile(0.75))
            }

        return stats

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
        p_value_threshold: float = 0.05
    ) -> Dict:
        """
        Detect drift in current data vs reference

        Args:
            current_data: Current production data
            psi_threshold: PSI threshold for alerting
            kl_threshold: KL divergence threshold
            p_value_threshold: Statistical test p-value threshold

        Returns:
            Dictionary with drift metrics and alerts
        """
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_samples': len(current_data),
            'features': {},
            'drift_detected': False,
            'alerts': []
        }

        for feature in self.feature_names:
            reference_values = self.reference_data[feature].values
            current_values = current_data[feature].values

            # Calculate drift metrics
            psi = self.detector.calculate_psi(reference_values, current_values)
            kl_div = self.detector.calculate_kl_divergence(
                reference_values,
                current_values
            )
            ks_stat, ks_pval = self.detector.ks_test(
                reference_values,
                current_values
            )

            # Current statistics
            current_stats = {
                'mean': float(current_values.mean()),
                'std': float(current_values.std())
            }

            # Check for drift
            drift_flags = {
                'psi_drift': psi > psi_threshold,
                'kl_drift': kl_div > kl_threshold,
                'ks_drift': ks_pval < p_value_threshold
            }

            feature_drifted = any(drift_flags.values())

            # Store results
            results['features'][feature] = {
                'psi': psi,
                'kl_divergence': kl_div,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'reference_mean': self.reference_stats[feature]['mean'],
                'current_mean': current_stats['mean'],
                'reference_std': self.reference_stats[feature]['std'],
                'current_std': current_stats['std'],
                'drift_detected': feature_drifted,
                'drift_flags': drift_flags
            }

            # Generate alerts
            if feature_drifted:
                results['drift_detected'] = True
                alert_msg = f"Drift detected in '{feature}': "
                if drift_flags['psi_drift']:
                    alert_msg += f"PSI={psi:.3f} (>{psi_threshold}) "
                if drift_flags['kl_drift']:
                    alert_msg += f"KL={kl_div:.3f} (>{kl_threshold}) "
                if drift_flags['ks_drift']:
                    alert_msg += f"KS p-value={ks_pval:.3f} (<{p_value_threshold})"

                results['alerts'].append(alert_msg)
                logger.warning(alert_msg)

        return results


class PredictionMonitor:
    """
    Monitor prediction distributions
    """

    def __init__(self, reference_predictions: np.ndarray):
        """
        Initialize prediction monitor

        Args:
            reference_predictions: Reference predictions (from validation set)
        """
        self.reference_predictions = reference_predictions
        self.reference_distribution = self._compute_distribution(
            reference_predictions
        )
        self.detector = DriftDetector()

    def _compute_distribution(self, predictions: np.ndarray) -> Dict:
        """Compute prediction distribution"""
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique.astype(int), counts / len(predictions)))

        return {
            'distribution': distribution,
            'mean': float(predictions.mean()),
            'std': float(predictions.std())
        }

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect drift in prediction distribution

        Args:
            current_predictions: Current production predictions
            threshold: Distribution change threshold

        Returns:
            Drift detection results
        """
        current_dist = self._compute_distribution(current_predictions)

        # Calculate drift
        psi = self.detector.calculate_psi(
            self.reference_predictions,
            current_predictions
        )

        # Check distribution shift
        max_shift = 0
        shifts = {}

        for class_label in self.reference_distribution['distribution'].keys():
            ref_pct = self.reference_distribution['distribution'][class_label]
            curr_pct = current_dist['distribution'].get(class_label, 0)
            shift = abs(curr_pct - ref_pct)
            shifts[f'class_{class_label}'] = {
                'reference': ref_pct,
                'current': curr_pct,
                'shift': shift
            }
            max_shift = max(max_shift, shift)

        drift_detected = psi > threshold or max_shift > threshold

        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_predictions': len(current_predictions),
            'psi': psi,
            'class_shifts': shifts,
            'max_shift': max_shift,
            'drift_detected': drift_detected,
            'reference_mean': self.reference_distribution['mean'],
            'current_mean': current_dist['mean']
        }

        if drift_detected:
            logger.warning(
                f"Prediction drift detected: PSI={psi:.3f}, "
                f"Max shift={max_shift:.3f}"
            )

        return results
```

---

## Part 2: Monitoring Storage

Create `ml/monitoring/metrics_store.py`:

```python
"""
Store and retrieve monitoring metrics
Uses JSON files for simplicity (use database in production)
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd


class MetricsStore:
    """
    Store monitoring metrics over time
    """

    def __init__(self, storage_dir: str = "monitoring/metrics"):
        """
        Initialize metrics store

        Args:
            storage_dir: Directory to store metrics
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_drift_metrics(
        self,
        metrics: Dict,
        date: str = None
    ):
        """
        Save drift detection metrics

        Args:
            metrics: Drift metrics dictionary
            date: Date partition (YYYY-MM-DD)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Create date directory
        date_dir = self.storage_dir / "drift" / date
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        filepath = date_dir / "metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Append to history
        self._append_to_history('drift', metrics)

    def save_prediction_metrics(
        self,
        metrics: Dict,
        date: str = None
    ):
        """Save prediction distribution metrics"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        date_dir = self.storage_dir / "predictions" / date
        date_dir.mkdir(parents=True, exist_ok=True)

        filepath = date_dir / "metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        self._append_to_history('predictions', metrics)

    def _append_to_history(self, metric_type: str, metrics: Dict):
        """Append metrics to historical CSV"""
        history_file = self.storage_dir / f"{metric_type}_history.csv"

        # Flatten metrics for CSV
        flat_metrics = self._flatten_dict(metrics)

        # Create dataframe
        df = pd.DataFrame([flat_metrics])

        # Append to file
        if history_file.exists():
            df.to_csv(history_file, mode='a', header=False, index=False)
        else:
            df.to_csv(history_file, mode='w', header=True, index=False)

    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_history(
        self,
        metric_type: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Retrieve historical metrics

        Args:
            metric_type: 'drift' or 'predictions'
            days: Number of days to retrieve

        Returns:
            DataFrame with historical metrics
        """
        history_file = self.storage_dir / f"{metric_type}_history.csv"

        if not history_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(history_file)

        # Filter by date if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]

        return df
```

---

## Part 3: Monitoring DAG

Create `dags/monitoring_dag.py`:

```python
"""
Monitoring DAG
Detect drift and alert when thresholds exceeded
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.monitoring.drift_detection import FeatureMonitor, PredictionMonitor
from ml.monitoring.metrics_store import MetricsStore

logger = logging.getLogger(__name__)


def load_reference_data():
    """Load reference/training data"""
    # In practice, load from your data lake
    # For now, generate synthetic reference data
    np.random.seed(42)

    reference_df = pd.DataFrame({
        'age': np.random.normal(35, 10, 10000),
        'income': np.random.lognormal(11, 0.5, 10000),
        'tenure_days': np.random.exponential(200, 10000),
    })

    return reference_df


def detect_feature_drift(**context):
    """
    Detect drift in feature distributions
    """
    execution_date = context['ds']
    logger.info(f"Detecting feature drift for {execution_date}")

    # Load reference data
    reference_df = load_reference_data()

    # Load current production data
    current_data_path = f"data/unscored/{execution_date}/data.csv"

    if not Path(current_data_path).exists():
        logger.warning(f"No data found for {execution_date}")
        return

    current_df = pd.read_csv(current_data_path)

    # Initialize monitor
    monitor = FeatureMonitor(
        reference_data=reference_df,
        feature_names=['age', 'income', 'tenure_days']
    )

    # Detect drift
    drift_results = monitor.detect_drift(
        current_data=current_df,
        psi_threshold=0.2,
        kl_threshold=0.1
    )

    # Store metrics
    metrics_store = MetricsStore()
    metrics_store.save_drift_metrics(drift_results, date=execution_date)

    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='drift_detected', value=drift_results['drift_detected'])
    context['ti'].xcom_push(key='drift_alerts', value=drift_results['alerts'])

    logger.info(f"Drift detection complete. Drift detected: {drift_results['drift_detected']}")


def detect_prediction_drift(**context):
    """
    Detect drift in prediction distributions
    """
    execution_date = context['ds']
    logger.info(f"Detecting prediction drift for {execution_date}")

    # Load reference predictions (from validation set)
    reference_predictions = np.random.binomial(1, 0.4, 10000)

    # Load current predictions
    predictions_path = f"data/predictions/{execution_date}/predictions.csv"

    if not Path(predictions_path).exists():
        logger.warning(f"No predictions found for {execution_date}")
        return

    predictions_df = pd.read_csv(predictions_path)
    current_predictions = predictions_df['prediction'].values

    # Initialize monitor
    monitor = PredictionMonitor(reference_predictions=reference_predictions)

    # Detect drift
    drift_results = monitor.detect_prediction_drift(
        current_predictions=current_predictions,
        threshold=0.1
    )

    # Store metrics
    metrics_store = MetricsStore()
    metrics_store.save_prediction_metrics(drift_results, date=execution_date)

    logger.info(f"Prediction drift: {drift_results['drift_detected']}")


def send_drift_alerts(**context):
    """
    Send alerts if drift detected
    """
    execution_date = context['ds']

    drift_detected = context['ti'].xcom_pull(
        task_ids='detect_feature_drift',
        key='drift_detected'
    )

    if drift_detected:
        alerts = context['ti'].xcom_pull(
            task_ids='detect_feature_drift',
            key='drift_alerts'
        )

        logger.warning(f"DRIFT ALERT for {execution_date}")
        for alert in alerts:
            logger.warning(f"  - {alert}")

        # In production: send to Slack, PagerDuty, etc.
        # send_slack_message(alerts)
        # trigger_retraining_dag()

    else:
        logger.info(f"No drift detected for {execution_date}")


# Define DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='monitoring_drift_detection',
    default_args=default_args,
    description='Monitor for data and prediction drift',
    schedule='0 3 * * *',  # Run daily at 3 AM (after batch inference)
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift'],
) as dag:

    detect_features = PythonOperator(
        task_id='detect_feature_drift',
        python_callable=detect_feature_drift,
        provide_context=True,
    )

    detect_predictions = PythonOperator(
        task_id='detect_prediction_drift',
        python_callable=detect_prediction_drift,
        provide_context=True,
    )

    send_alerts = PythonOperator(
        task_id='send_drift_alerts',
        python_callable=send_drift_alerts,
        provide_context=True,
    )

    [detect_features, detect_predictions] >> send_alerts
```

---

## Part 4: Visualization

Create `ml/monitoring/visualize.py`:

```python
"""
Visualize monitoring metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ml.monitoring.metrics_store import MetricsStore


def plot_drift_over_time(
    metric_type: str = 'drift',
    feature: str = 'age',
    days: int = 30
):
    """
    Plot drift metrics over time

    Args:
        metric_type: 'drift' or 'predictions'
        feature: Feature name to plot
        days: Number of days to plot
    """
    store = MetricsStore()
    history = store.get_history(metric_type, days=days)

    if history.empty:
        print("No historical data available")
        return

    # Extract relevant columns
    psi_col = f'features.{feature}.psi'
    kl_col = f'features.{feature}.kl_divergence'

    if psi_col in history.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot PSI
        axes[0].plot(history['timestamp'], history[psi_col], marker='o')
        axes[0].axhline(y=0.1, color='orange', linestyle='--', label='Warning')
        axes[0].axhline(y=0.2, color='red', linestyle='--', label='Alert')
        axes[0].set_title(f'PSI Over Time - {feature}')
        axes[0].set_ylabel('PSI')
        axes[0].legend()
        axes[0].grid(True)

        # Plot KL Divergence
        axes[1].plot(history['timestamp'], history[kl_col], marker='o', color='green')
        axes[1].axhline(y=0.1, color='red', linestyle='--', label='Alert')
        axes[1].set_title(f'KL Divergence Over Time - {feature}')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('KL Divergence')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'monitoring/plots/{feature}_drift_over_time.png')
        print(f"Plot saved to monitoring/plots/{feature}_drift_over_time.png")


if __name__ == '__main__':
    plot_drift_over_time(feature='age', days=30)
```

---

## Exercises

### Exercise 1: Add Model Performance Monitoring

If you have ground truth labels:

```python
def monitor_model_accuracy(**context):
    """Monitor actual model performance"""
    execution_date = context['ds']

    # Load predictions
    predictions = pd.read_csv(f"data/predictions/{execution_date}/predictions.csv")

    # Load ground truth labels (after they become available)
    labels = pd.read_csv(f"data/labels/{execution_date}/labels.csv")

    # Calculate metrics
    from sklearn.metrics import accuracy_score, roc_auc_score

    accuracy = accuracy_score(labels['true_label'], predictions['prediction'])
    auc = roc_auc_score(labels['true_label'], predictions['prob_class_1'])

    # Alert if performance drops
    if accuracy < 0.80:
        send_alert(f"Model accuracy dropped to {accuracy:.2%}")
```

### Exercise 2: Add Multivariate Drift Detection

Detect drift in multiple features together:

```python
from scipy.spatial.distance import euclidean

def detect_multivariate_drift(reference_df, current_df, features):
    """Detect drift using multivariate distance"""

    # Compute mean vectors
    ref_mean = reference_df[features].mean().values
    curr_mean = current_df[features].mean().values

    # Euclidean distance between means
    distance = euclidean(ref_mean, curr_mean)

    # Normalize by reference std
    ref_std = reference_df[features].std().mean()
    normalized_distance = distance / ref_std

    return normalized_distance
```

---

## Production Best Practices

### 1. Sampling for Large Datasets

Don't monitor every prediction:

```python
# Sample 10% of data for monitoring
sample_size = int(len(data) * 0.1)
sample_data = data.sample(n=sample_size, random_state=42)

# Detect drift on sample
drift_results = monitor.detect_drift(sample_data)
```

### 2. Alerting Fatigue

Use smart alerting:

```python
def should_alert(current_drift, previous_drift_history):
    """Only alert on sustained drift"""

    # Alert if drift detected for 3 consecutive days
    recent_drift = previous_drift_history[-3:]
    if len(recent_drift) == 3 and all(recent_drift):
        return True

    # Alert if drift magnitude is very high
    if current_drift['max_psi'] > 0.5:
        return True

    return False
```

### 3. Automatic Retraining Triggers

```python
def check_retraining_needed(**context):
    """Decide if retraining should be triggered"""

    drift_detected = context['ti'].xcom_pull(key='drift_detected')

    # Get drift severity
    drift_metrics = context['ti'].xcom_pull(key='drift_metrics')

    # Retrain if severe drift or scheduled time
    days_since_training = get_days_since_last_training()

    if drift_detected and drift_metrics['max_psi'] > 0.3:
        trigger_retraining_dag()
    elif days_since_training > 30:
        trigger_retraining_dag()
```

---

## Key Takeaways

✅ **Monitor continuously**: Don't wait for complaints
✅ **Use multiple metrics**: PSI, KL, statistical tests
✅ **Track over time**: Trends matter more than single points
✅ **Alert smartly**: Avoid fatigue with smart thresholds
✅ **Automate responses**: Trigger retraining when needed

---

## Next Steps

- ✅ Complete this lab and test drift detection
- ✅ Share your implementation for review
- → Move to **Lab 4.4: Automated Retraining Pipeline**

---

**Congratulations! You've implemented production monitoring! 🚀**

**Next**: [Lab 4.4 - Retraining Pipeline →](./lab4_4_retraining_pipeline.md)
