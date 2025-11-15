# Module 5: Monitoring & Drift Detection

**Estimated Time**: 2-3 days
**Difficulty**: Medium-Hard

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Implement feature drift detection (KS test, PSI, Chi-square)
- ✅ Monitor prediction drift and distribution shifts
- ✅ Track model performance in production
- ✅ Build automated alerting systems
- ✅ Create monitoring DAGs in Airflow
- ✅ Generate drift reports and visualizations

## Overview

This module builds a **production monitoring system** that:
1. Detects feature drift (input distribution changes)
2. Monitors prediction drift (output distribution changes)
3. Tracks model performance metrics
4. Generates alerts when drift is detected
5. Orchestrates monitoring with Airflow
6. Produces reports for investigation

**Key Principle**: "You can't improve what you don't measure." Monitoring is critical for maintaining model quality in production.

## Background

### Why Models Degrade in Production

**Data Drift**: Input distributions change over time
- User behavior evolves
- New user segments emerge
- Seasonal patterns
- Product changes affect features

**Concept Drift**: Relationship between inputs and outputs changes
- User preferences shift
- Market conditions change
- Competitors affect behavior

**Both lead to**: Model predictions become less accurate → User experience degrades → Business metrics suffer

### Drift Detection Methods

**For Continuous Features**:
1. **Kolmogorov-Smirnov (KS) Test**:
   - Measures maximum difference between CDFs
   - Returns statistic ∈ [0, 1] and p-value
   - Drift if KS statistic > threshold (e.g., 0.1)

2. **Population Stability Index (PSI)**:
   - Bins data, compares percentages
   - PSI = Σ (actual% - expected%) × ln(actual%/expected%)
   - PSI < 0.1: No drift, 0.1-0.25: Moderate, >0.25: Significant

**For Categorical Features**:
1. **Chi-Square Test**:
   - Compares frequency distributions
   - Returns χ² statistic and p-value

**For Predictions**:
1. **Mean/Std Shift**: Track changes in prediction statistics
2. **Distribution Comparison**: Use KS test on prediction scores

## Step 1: Implement Drift Detection

### File Structure
```
src/monitoring/
├── __init__.py
├── drift.py          # Drift detection algorithms
├── metrics.py        # Performance metrics
└── alerting.py       # Alert generation
```

### Drift Detection Implementation

**File**: `src/monitoring/drift.py`

```python
"""
Drift detection implementations.

Supports:
- Feature drift (KS test, PSI, Chi-square)
- Prediction drift
- Concept drift (performance degradation)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path


class DriftDetector:
    """Detect data drift in features and predictions."""

    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config or {
            'ks_threshold': 0.1,
            'psi_threshold': 0.2,
            'prediction_mean_threshold': 0.1
        }

    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Detect drift in features.

        Args:
            reference_data: Training/baseline data
            current_data: Recent production data
            feature_cols: List of feature column names

        Returns:
            Dict with drift results per feature
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_reference': len(reference_data),
            'n_current': len(current_data),
            'features': {},
            'summary': {
                'total_features': len(feature_cols),
                'drifted_features': 0,
                'drift_detected': False
            }
        }

        for feature in feature_cols:
            # Skip if feature not in both datasets
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            # Determine if continuous or categorical
            if pd.api.types.is_numeric_dtype(ref_values):
                # Continuous: use KS test and PSI
                drift_result = self._detect_continuous_drift(
                    ref_values, curr_values, feature
                )
            else:
                # Categorical: use Chi-square
                drift_result = self._detect_categorical_drift(
                    ref_values, curr_values, feature
                )

            results['features'][feature] = drift_result

            if drift_result['drifted']:
                results['summary']['drifted_features'] += 1

        # Overall drift detected if any feature drifted
        results['summary']['drift_detected'] = results['summary']['drifted_features'] > 0

        return results

    def _detect_continuous_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> Dict:
        """Detect drift in continuous feature using KS test and PSI."""
        # KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference, current)

        # PSI
        psi_value = self._calculate_psi(reference, current)

        # Drift detected if either test indicates drift
        ks_drift = ks_statistic > self.config['ks_threshold']
        psi_drift = psi_value > self.config['psi_threshold']
        drifted = ks_drift or psi_drift

        return {
            'feature': feature_name,
            'type': 'continuous',
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'psi': float(psi_value),
            'drifted': drifted,
            'drift_reason': self._get_drift_reason(ks_drift, psi_drift),
            'reference_mean': float(reference.mean()),
            'current_mean': float(current.mean()),
            'reference_std': float(reference.std()),
            'current_std': float(current.std())
        }

    def _detect_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> Dict:
        """Detect drift in categorical feature using Chi-square test."""
        # Get value counts
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()

        # Align categories
        all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))

        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

        # Chi-square test
        chi2_statistic, chi2_pvalue = stats.chisquare(curr_freq, ref_freq)

        drifted = chi2_pvalue < 0.05  # Significant at 5% level

        return {
            'feature': feature_name,
            'type': 'categorical',
            'chi2_statistic': float(chi2_statistic),
            'chi2_pvalue': float(chi2_pvalue),
            'drifted': drifted,
            'reference_categories': len(ref_counts),
            'current_categories': len(curr_counts)
        }

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = Σ (actual% - expected%) × ln(actual%/expected%)
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        if len(breakpoints) <= 1:
            return 0.0  # Can't compute PSI

        # Bin both datasets
        ref_binned = np.digitize(reference, breakpoints[1:-1])
        curr_binned = np.digitize(current, breakpoints[1:-1])

        # Calculate percentages
        ref_percents = np.bincount(ref_binned, minlength=bins) / len(reference)
        curr_percents = np.bincount(curr_binned, minlength=bins) / len(current)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_percents = ref_percents + epsilon
        curr_percents = curr_percents + epsilon

        # PSI formula
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))

        return float(psi)

    def _get_drift_reason(self, ks_drift: bool, psi_drift: bool) -> str:
        """Get human-readable drift reason."""
        if ks_drift and psi_drift:
            return "Both KS test and PSI indicate drift"
        elif ks_drift:
            return "KS test indicates drift"
        elif psi_drift:
            return "PSI indicates drift"
        else:
            return "No drift detected"

    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series
    ) -> Dict:
        """
        Detect drift in model predictions.

        Args:
            reference_predictions: Baseline prediction scores
            current_predictions: Recent prediction scores

        Returns:
            Dict with prediction drift analysis
        """
        # Mean shift
        ref_mean = reference_predictions.mean()
        curr_mean = current_predictions.mean()
        mean_shift = abs(curr_mean - ref_mean) / ref_mean if ref_mean > 0 else 0

        # Std shift
        ref_std = reference_predictions.std()
        curr_std = current_predictions.std()

        # KS test on predictions
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)

        # Drift if mean shifted >10% or KS test significant
        mean_drift = mean_shift > self.config['prediction_mean_threshold']
        ks_drift = ks_statistic > self.config['ks_threshold']
        drifted = mean_drift or ks_drift

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'drifted': drifted,
            'mean_shift_pct': float(mean_shift * 100),
            'reference_mean': float(ref_mean),
            'current_mean': float(curr_mean),
            'reference_std': float(ref_std),
            'current_std': float(curr_std),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue)
        }


class PerformanceMonitor:
    """Monitor model performance in production."""

    def __init__(self):
        pass

    def calculate_online_metrics(
        self,
        predictions_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate online performance metrics from logged predictions.

        Assumes predictions_df has: user_id, item_id, score, click (if available)

        Args:
            predictions_df: DataFrame with prediction logs

        Returns:
            Dict with performance metrics
        """
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_predictions': len(predictions_df),
            'mean_score': float(predictions_df['score'].mean()),
            'std_score': float(predictions_df['score'].std())
        }

        # If we have ground truth clicks (joined from logs)
        if 'click' in predictions_df.columns:
            from sklearn.metrics import roc_auc_score, log_loss

            # AUC
            try:
                auc = roc_auc_score(predictions_df['click'], predictions_df['score'])
                metrics['auc'] = float(auc)
            except:
                metrics['auc'] = None

            # CTR
            ctr = predictions_df['click'].mean()
            metrics['ctr'] = float(ctr)

            # Log loss
            try:
                logloss = log_loss(predictions_df['click'], predictions_df['score'])
                metrics['log_loss'] = float(logloss)
            except:
                metrics['log_loss'] = None

        return metrics


def main():
    """Test drift detection."""
    # Load reference data (training set)
    reference_df = pd.read_parquet("data/features/train_features.parquet")

    # Simulate current data (we'll use test set as a stand-in)
    # In production, this would be recent prediction logs with features
    current_df = pd.read_parquet("data/features/test_features.parquet")

    # Feature columns
    feature_cols = [
        'user_historical_ctr', 'user_avg_dwell_time',
        'user_interaction_count', 'user_days_active',
        'item_ctr', 'item_avg_dwell_time',
        'item_popularity', 'item_age_days',
        'hour_of_day', 'day_of_week'
    ]

    # Detect drift
    detector = DriftDetector()
    drift_results = detector.detect_feature_drift(
        reference_df, current_df, feature_cols
    )

    print("\n📊 Drift Detection Results:")
    print(f"Total features: {drift_results['summary']['total_features']}")
    print(f"Drifted features: {drift_results['summary']['drifted_features']}")
    print(f"Drift detected: {drift_results['summary']['drift_detected']}")

    # Show drifted features
    if drift_results['summary']['drifted_features'] > 0:
        print("\nDrifted features:")
        for feature, result in drift_results['features'].items():
            if result['drifted']:
                print(f"  {feature}: {result['drift_reason']}")

    # Save results
    output_dir = Path("data/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "drift_report.json", 'w') as f:
        json.dump(drift_results, f, indent=2)

    print(f"\n✅ Drift report saved to data/monitoring/drift_report.json")


if __name__ == "__main__":
    main()
```

### Alerting System

**File**: `src/monitoring/alerting.py`

```python
"""
Alerting system for drift and performance issues.
"""

from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path


class AlertManager:
    """Manage alerts for drift and performance issues."""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'alert_on_drift': True,
            'alert_on_performance_degradation': True,
            'min_drifted_features': 1,
            'performance_threshold': 0.05  # 5% AUC drop
        }

        self.alerts = []

    def check_drift_alerts(self, drift_results: Dict) -> List[Dict]:
        """
        Generate alerts based on drift detection results.

        Args:
            drift_results: Output from DriftDetector.detect_feature_drift()

        Returns:
            List of alert dicts
        """
        alerts = []

        if not self.config['alert_on_drift']:
            return alerts

        # Check if drift detected
        if drift_results['summary']['drift_detected']:
            drifted_count = drift_results['summary']['drifted_features']

            if drifted_count >= self.config['min_drifted_features']:
                # Create alert
                alert = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'feature_drift',
                    'severity': self._get_drift_severity(drifted_count),
                    'message': f"Feature drift detected in {drifted_count} features",
                    'details': {
                        'drifted_features': [
                            feature for feature, result in drift_results['features'].items()
                            if result.get('drifted', False)
                        ],
                        'total_features': drift_results['summary']['total_features']
                    },
                    'action': 'Consider retraining model with recent data'
                }

                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def check_performance_alerts(
        self,
        current_metrics: Dict,
        baseline_metrics: Dict
    ) -> List[Dict]:
        """
        Generate alerts based on performance degradation.

        Args:
            current_metrics: Recent performance metrics
            baseline_metrics: Baseline (training) metrics

        Returns:
            List of alert dicts
        """
        alerts = []

        if not self.config['alert_on_performance_degradation']:
            return alerts

        # Check AUC degradation
        if 'auc' in current_metrics and 'auc' in baseline_metrics:
            auc_drop = baseline_metrics['auc'] - current_metrics['auc']

            if auc_drop > self.config['performance_threshold']:
                alert = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'performance_degradation',
                    'severity': 'high' if auc_drop > 0.1 else 'medium',
                    'message': f"AUC dropped by {auc_drop:.3f}",
                    'details': {
                        'baseline_auc': baseline_metrics['auc'],
                        'current_auc': current_metrics['auc'],
                        'drop': auc_drop
                    },
                    'action': 'Retrain model immediately'
                }

                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def _get_drift_severity(self, drifted_count: int) -> str:
        """Determine alert severity based on number of drifted features."""
        if drifted_count >= 5:
            return 'high'
        elif drifted_count >= 2:
            return 'medium'
        else:
            return 'low'

    def save_alerts(self, output_path: str = "data/monitoring/alerts.json"):
        """Save all alerts to file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.alerts, f, indent=2)

    def send_alert(self, alert: Dict):
        """
        Send alert (email, Slack, PagerDuty, etc.).

        In this implementation, just print and log.
        In production, integrate with alerting service.
        """
        severity_emoji = {
            'low': '⚠️',
            'medium': '⚠️⚠️',
            'high': '🚨'
        }

        emoji = severity_emoji.get(alert['severity'], '⚠️')

        print(f"\n{emoji} ALERT: {alert['type']}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Message: {alert['message']}")
        print(f"   Action: {alert['action']}")

        # In production, send to Slack:
        # slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        # requests.post(slack_webhook, json={"text": alert['message']})
```

## Step 2: Build Monitoring DAG

**File**: `dags/monitoring_dag.py`

```python
"""
Monitoring and Drift Detection DAG.

Runs daily to:
1. Load reference data (training set)
2. Load recent predictions
3. Detect feature drift
4. Detect prediction drift
5. Monitor performance
6. Generate alerts
7. Trigger retraining if needed
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')

import pandas as pd
import json
from pathlib import Path

from src.monitoring.drift import DriftDetector, PerformanceMonitor
from src.monitoring.alerting import AlertManager


default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def load_reference_data(**context):
    """Load reference (training) data."""
    print("Loading reference data...")

    reference_df = pd.read_parquet("data/features/train_features.parquet")

    # Store metadata in XCom
    context['ti'].xcom_push(key='reference_size', value=len(reference_df))

    print(f"Loaded {len(reference_df)} reference samples")


def load_current_data(**context):
    """Load recent production data (from prediction logs)."""
    print("Loading current production data...")

    # In production, this would query prediction logs
    # For this project, we'll use test set as a proxy
    current_df = pd.read_parquet("data/features/test_features.parquet")

    context['ti'].xcom_push(key='current_size', value=len(current_df))

    print(f"Loaded {len(current_df)} current samples")


def detect_feature_drift(**context):
    """Detect feature drift."""
    print("Detecting feature drift...")

    # Load data
    reference_df = pd.read_parquet("data/features/train_features.parquet")
    current_df = pd.read_parquet("data/features/test_features.parquet")

    # Feature columns
    feature_cols = [
        'user_historical_ctr', 'user_avg_dwell_time',
        'user_interaction_count', 'user_days_active',
        'item_ctr', 'item_avg_dwell_time',
        'item_popularity', 'item_age_days',
        'hour_of_day', 'day_of_week'
    ]

    # Detect drift
    detector = DriftDetector()
    drift_results = detector.detect_feature_drift(
        reference_df, current_df, feature_cols
    )

    # Save results
    output_dir = Path("data/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "drift_report.json", 'w') as f:
        json.dump(drift_results, f, indent=2)

    # Push to XCom
    context['ti'].xcom_push(key='drift_detected', value=drift_results['summary']['drift_detected'])
    context['ti'].xcom_push(key='drifted_features', value=drift_results['summary']['drifted_features'])

    print(f"Drift detected: {drift_results['summary']['drift_detected']}")
    print(f"Drifted features: {drift_results['summary']['drifted_features']}")


def monitor_performance(**context):
    """Monitor model performance."""
    print("Monitoring model performance...")

    # Load baseline metrics (from training)
    with open("models/training/test_metrics.json", 'r') as f:
        baseline_metrics = json.load(f)

    # In production, calculate current metrics from recent predictions
    # For now, use test set metrics
    current_metrics = baseline_metrics  # Placeholder

    # Save metrics
    output_dir = Path("data/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "performance_metrics.json", 'w') as f:
        json.dump({
            'baseline': baseline_metrics,
            'current': current_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }, f, indent=2)

    print(f"Baseline AUC: {baseline_metrics.get('auc', 'N/A')}")
    print(f"Current AUC: {current_metrics.get('auc', 'N/A')}")


def generate_alerts(**context):
    """Generate alerts based on drift and performance."""
    print("Generating alerts...")

    # Load drift results
    with open("data/monitoring/drift_report.json", 'r') as f:
        drift_results = json.load(f)

    # Load performance metrics
    with open("data/monitoring/performance_metrics.json", 'r') as f:
        perf_data = json.load(f)

    # Generate alerts
    alert_manager = AlertManager()

    # Check drift alerts
    drift_alerts = alert_manager.check_drift_alerts(drift_results)

    # Check performance alerts
    perf_alerts = alert_manager.check_performance_alerts(
        perf_data['current'],
        perf_data['baseline']
    )

    # Send alerts
    all_alerts = drift_alerts + perf_alerts
    for alert in all_alerts:
        alert_manager.send_alert(alert)

    # Save alerts
    alert_manager.save_alerts()

    # Push to XCom (for retraining trigger)
    should_retrain = len(all_alerts) > 0
    context['ti'].xcom_push(key='should_retrain', value=should_retrain)

    print(f"Generated {len(all_alerts)} alerts")


def decide_retraining(**context):
    """Decide if retraining should be triggered."""
    ti = context['ti']

    should_retrain = ti.xcom_pull(key='should_retrain', task_ids='generate_alerts')

    print(f"Should retrain: {should_retrain}")

    return should_retrain


with DAG(
    'monitoring_and_drift_detection',
    default_args=default_args,
    description='Monitor model performance and detect drift',
    schedule_interval='0 4 * * *',  # Daily at 4 AM
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=['monitoring', 'drift'],
) as dag:

    load_ref_task = PythonOperator(
        task_id='load_reference_data',
        python_callable=load_reference_data,
    )

    load_curr_task = PythonOperator(
        task_id='load_current_data',
        python_callable=load_current_data,
    )

    drift_task = PythonOperator(
        task_id='detect_feature_drift',
        python_callable=detect_feature_drift,
    )

    perf_task = PythonOperator(
        task_id='monitor_performance',
        python_callable=monitor_performance,
    )

    alert_task = PythonOperator(
        task_id='generate_alerts',
        python_callable=generate_alerts,
    )

    # Conditional retraining trigger (will implement in Module 6)
    # retrain_trigger = TriggerDagRunOperator(
    #     task_id='trigger_retraining',
    #     trigger_dag_id='automated_retraining',
    #     trigger_rule=TriggerRule.ALL_SUCCESS,
    #     python_callable=decide_retraining,
    # )

    # Dependencies
    [load_ref_task, load_curr_task] >> drift_task
    drift_task >> perf_task >> alert_task
    # >> retrain_trigger
```

## Step 3: Create Monitoring Dashboard (Optional)

**File**: `notebooks/03_monitoring_dashboard.ipynb`

```python
# Jupyter notebook for visualizing monitoring results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load drift report
with open("../data/monitoring/drift_report.json", 'r') as f:
    drift_results = json.load(f)

# Visualize drifted features
drifted_features = [
    (feature, result['ks_statistic'])
    for feature, result in drift_results['features'].items()
    if result.get('drifted', False)
]

if drifted_features:
    features, ks_stats = zip(*drifted_features)

    plt.figure(figsize=(10, 6))
    plt.barh(features, ks_stats)
    plt.xlabel('KS Statistic')
    plt.title('Feature Drift Detection')
    plt.axvline(x=0.1, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../data/monitoring/drift_visualization.png')
    plt.show()

# Load performance metrics
with open("../data/monitoring/performance_metrics.json", 'r') as f:
    perf_data = json.load(f)

print(f"Baseline AUC: {perf_data['baseline'].get('auc', 'N/A'):.4f}")
print(f"Current AUC: {perf_data['current'].get('auc', 'N/A'):.4f}")
```

## Testing

```bash
cd capstone_project

# 1. Test drift detection
python src/monitoring/drift.py

# 2. Test monitoring DAG
airflow dags test monitoring_and_drift_detection 2024-11-15

# 3. Check outputs
cat data/monitoring/drift_report.json
cat data/monitoring/alerts.json

# 4. View visualization
# Open notebooks/03_monitoring_dashboard.ipynb
```

## Review Checklist

- [ ] Drift detection works for continuous features
- [ ] Drift detection works for categorical features
- [ ] Prediction drift detection implemented
- [ ] Performance monitoring works
- [ ] Alerts generated correctly
- [ ] Monitoring DAG runs successfully
- [ ] Reports saved to data/monitoring/
- [ ] Visualizations created

## What to Submit

1. **Code**:
   - `src/monitoring/drift.py`
   - `src/monitoring/alerting.py`
   - `dags/monitoring_dag.py`

2. **Reports**:
   - Sample drift report
   - Sample alert
   - Performance metrics

3. **Visualizations**: Drift charts

4. **Reflection**:
   - What drift did you detect?
   - How would you tune thresholds?
   - What other metrics would you monitor?

## Next Steps

With monitoring in place, proceed to [Module 6: Retraining & Promotion](module6_retraining.md)!
