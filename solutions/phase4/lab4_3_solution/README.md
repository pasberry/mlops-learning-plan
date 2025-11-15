# Lab 4.3: Model Monitoring Solution

Complete drift detection and monitoring system for production models.

## Overview

This solution implements comprehensive model monitoring:
- **Feature Drift Detection** - Population Stability Index (PSI), KS test, JS divergence
- **Prediction Drift Detection** - Monitor model output distribution
- **Automated Alerting** - Notify when drift exceeds thresholds
- **Scheduled Monitoring** - Daily automated checks via Airflow

## Architecture

```
lab4_3_solution/
├── monitoring/
│   └── drift_detector.py    # Drift detection algorithms
├── dags/
│   └── monitoring_dag.py    # Airflow monitoring orchestration
└── README.md                # This file
```

## Drift Detection Methods

### 1. Population Stability Index (PSI)

Measures distribution shift between baseline and current data:
- **PSI < 0.1**: No significant change
- **0.1 ≤ PSI < 0.2**: Small change
- **PSI ≥ 0.2**: Significant drift (alert!)

### 2. Kolmogorov-Smirnov Test

Statistical test comparing cumulative distributions:
- **p-value < 0.05**: Distributions are significantly different (drift)
- **p-value ≥ 0.05**: No significant difference

### 3. Jensen-Shannon Divergence

Similarity measure between probability distributions:
- **JS = 0**: Identical distributions
- **JS = 1**: Completely different
- **JS ≥ 0.1**: Significant drift

## Setup

### 1. Install Dependencies

```bash
pip install numpy pandas scipy airflow
```

### 2. Create Test Data

```bash
# Create directories
mkdir -p /home/user/mlops-learning-plan/data/baseline
mkdir -p /home/user/mlops-learning-plan/data/monitoring/reports

# Generate baseline data
python3 << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)

# Baseline data (1 month ago)
baseline_data = {
    'age': np.random.normal(35, 10, 5000).clip(18, 80),
    'income': np.random.normal(65000, 20000, 5000).clip(20000, 200000),
    'credit_score': np.random.normal(700, 50, 5000).clip(300, 850),
    'num_purchases': np.random.poisson(5, 5000),
    'account_age_days': np.random.uniform(1, 3650, 5000),
    'avg_transaction': np.random.lognormal(4.5, 0.8, 5000),
    'num_returns': np.random.poisson(1, 5000),
    'is_premium': np.random.binomial(1, 0.3, 5000),
    'region': np.random.randint(1, 5, 5000),
    'category_preference': np.random.randint(1, 10, 5000)
}

baseline_df = pd.DataFrame(baseline_data)
baseline_df.to_csv(
    '/home/user/mlops-learning-plan/data/baseline/features.csv',
    index=False
)

print(f"✓ Baseline data created: {len(baseline_df)} samples")

# Create baseline predictions
baseline_predictions = {
    'prediction_score': np.random.beta(2, 3, 5000),
    'prediction_class': np.random.binomial(1, 0.4, 5000)
}
pd.DataFrame(baseline_predictions).to_csv(
    '/home/user/mlops-learning-plan/data/baseline/predictions.csv',
    index=False
)

# Current data (with some drift)
current_data = {
    'age': np.random.normal(38, 12, 5000).clip(18, 80),  # Shifted mean
    'income': np.random.normal(70000, 25000, 5000).clip(20000, 200000),  # Increased variance
    'credit_score': np.random.normal(695, 55, 5000).clip(300, 850),  # Slight shift
    'num_purchases': np.random.poisson(6, 5000),  # Increased
    'account_age_days': np.random.uniform(1, 3650, 5000),
    'avg_transaction': np.random.lognormal(4.7, 0.9, 5000),  # Shifted
    'num_returns': np.random.poisson(1, 5000),
    'is_premium': np.random.binomial(1, 0.35, 5000),  # Increased premium rate
    'region': np.random.randint(1, 5, 5000),
    'category_preference': np.random.randint(1, 10, 5000)
}

current_df = pd.DataFrame(current_data)
current_df.to_csv(
    '/home/user/mlops-learning-plan/data/processed/current_features.csv',
    index=False
)

print(f"✓ Current data created: {len(current_df)} samples (with drift)")
EOF
```

## Running Drift Detection

### Method 1: Standalone Script

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_3_solution

# Run drift detection
python monitoring/drift_detector.py \
    --baseline /home/user/mlops-learning-plan/data/baseline/features.csv \
    --current /home/user/mlops-learning-plan/data/processed/current_features.csv \
    --features age income credit_score num_purchases avg_transaction \
    --output /home/user/mlops-learning-plan/data/monitoring/reports/drift_report.json \
    --psi-threshold 0.2 \
    --ks-threshold 0.05
```

### Method 2: Python API

```python
from monitoring.drift_detector import DriftDetector
import pandas as pd

# Load data
baseline_df = pd.read_csv('data/baseline/features.csv')
current_df = pd.read_csv('data/processed/current_features.csv')

# Create detector
detector = DriftDetector(
    psi_threshold=0.2,
    ks_threshold=0.05,
    js_threshold=0.1
)

# Detect drift
feature_columns = ['age', 'income', 'credit_score', 'num_purchases']
report = detector.generate_drift_report(
    baseline_df,
    current_df,
    feature_columns
)

# Check results
if report['overall_drift_detected']:
    print("⚠️  DRIFT DETECTED!")
    print(f"Drifted features: {report['feature_drift']['drifted_features']}")
else:
    print("✓ No drift detected")

# Save report
detector.save_report(report, 'drift_report.json')
```

### Method 3: Airflow DAG

```bash
# Copy files to Airflow
cp dags/monitoring_dag.py $AIRFLOW_HOME/dags/
mkdir -p $AIRFLOW_HOME/monitoring
cp monitoring/drift_detector.py $AIRFLOW_HOME/monitoring/

# Trigger DAG
airflow dags trigger model_monitoring

# Monitor execution
airflow dags list-runs -d model_monitoring

# View logs
airflow tasks logs model_monitoring detect_feature_drift 2025-11-15
```

## Expected Output

### Console Output

```
================================================================================
DRIFT DETECTION REPORT
================================================================================
Overall Drift Detected: YES

Baseline: 5,000 samples
Current:  5,000 samples

Feature Drift:
  Total features: 5
  Drifted features: 3
  Drift percentage: 60.0%

  Drifted Features:
    - age:
        PSI: 0.2456
        KS p-value: 0.0023
        JS divergence: 0.1234
    - income:
        PSI: 0.1876
        KS p-value: 0.0145
        JS divergence: 0.0987
    - avg_transaction:
        PSI: 0.2134
        KS p-value: 0.0089
        JS divergence: 0.1123

Prediction Drift:
  Drift detected: NO
  PSI: 0.0876
  Positive rate change: 0.0234
================================================================================
```

### Drift Report JSON

```json
{
  "overall_drift_detected": true,
  "feature_drift": {
    "total_features": 5,
    "drifted_features": 3,
    "drift_percentage": 60.0,
    "features": {
      "age": {
        "feature": "age",
        "psi": {
          "psi": 0.2456,
          "drift_detected": true,
          "threshold": 0.2,
          "expected_mean": 35.2,
          "actual_mean": 38.1
        },
        "ks_test": {
          "ks_statistic": 0.1234,
          "p_value": 0.0023,
          "drift_detected": true
        },
        "drift_detected": true
      }
    }
  },
  "timestamp": "2025-11-15T10:30:00.123456"
}
```

## Monitoring DAG Workflow

```
┌─────────────┐
│  Load Data  │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
┌──────▼──────┐  ┌───▼────────┐     │
│Feature Drift│  │Prediction  │     │
│  Detection  │  │   Drift    │     │
└──────┬──────┘  └─────┬──────┘     │
       │              │              │
       └──────┬───────┘              │
              │                      │
       ┌──────▼────────┐             │
       │Generate Report│             │
       └──────┬────────┘             │
              │                      │
       ┌──────▼─────┐                │
       │Check Drift?│                │
       └──────┬─────┘                │
              │                      │
     ┌────────┴────────┐             │
     │                 │             │
┌────▼────┐      ┌─────▼────┐       │
│Send     │      │Log       │       │
│Alert    │      │Success   │       │
└─────────┘      └──────────┘       │
```

## Alerting and Actions

### When Drift is Detected

1. **Immediate Actions:**
   - Send alert to ML team
   - Log detailed drift report
   - Archive baseline and current data

2. **Investigation:**
   - Review drifted features
   - Check data quality
   - Verify data pipeline
   - Analyze user behavior changes

3. **Remediation:**
   - Update baseline if drift is expected
   - Retrain model with new data
   - Adjust feature engineering
   - Update monitoring thresholds

### Configuring Alerts

```python
# In monitoring_dag.py

def send_drift_alert_func(**context):
    # Send to Slack
    send_slack_message(
        channel="#ml-alerts",
        message=alert_message
    )

    # Send to PagerDuty
    trigger_pagerduty_incident(
        severity='warning',
        details=drift_results
    )

    # Send email
    send_email(
        to='ml-team@example.com',
        subject='Model Drift Detected',
        body=alert_message
    )
```

## Tuning Detection Thresholds

### Conservative (fewer false alarms)

```python
detector = DriftDetector(
    psi_threshold=0.3,    # Higher = less sensitive
    ks_threshold=0.01,    # Lower = less sensitive
    js_threshold=0.15     # Higher = less sensitive
)
```

### Aggressive (catch early drift)

```python
detector = DriftDetector(
    psi_threshold=0.1,    # Lower = more sensitive
    ks_threshold=0.1,     # Higher = more sensitive
    js_threshold=0.05     # Lower = more sensitive
)
```

### Recommended (balanced)

```python
detector = DriftDetector(
    psi_threshold=0.2,
    ks_threshold=0.05,
    js_threshold=0.1
)
```

## Testing

### Test Drift Detection

```python
import numpy as np
from monitoring.drift_detector import DriftDetector

# Create detector
detector = DriftDetector()

# Test 1: No drift
baseline = np.random.normal(0, 1, 1000)
current = np.random.normal(0, 1, 1000)
psi, metrics = detector.calculate_psi(baseline, current)
assert psi < 0.1, "No drift should be detected"

# Test 2: Significant drift
baseline = np.random.normal(0, 1, 1000)
current = np.random.normal(2, 1, 1000)  # Mean shifted by 2
psi, metrics = detector.calculate_psi(baseline, current)
assert psi > 0.2, "Drift should be detected"

print("✓ All tests passed")
```

### Test Monitoring DAG

```bash
# Test individual tasks
airflow tasks test model_monitoring load_data 2025-11-15
airflow tasks test model_monitoring detect_feature_drift 2025-11-15
airflow tasks test model_monitoring generate_report 2025-11-15

# Run full DAG backfill
airflow dags backfill model_monitoring \
    --start-date 2025-11-14 \
    --end-date 2025-11-15
```

## Production Best Practices

### 1. Baseline Management

```python
# Update baseline periodically (e.g., monthly)
def update_baseline(**context):
    """Update baseline with recent production data."""
    current_df = pd.read_csv('data/processed/current_features.csv')

    # Save as new baseline
    timestamp = datetime.utcnow().strftime('%Y%m%d')
    baseline_path = f'data/baseline/features_{timestamp}.csv'
    current_df.to_csv(baseline_path, index=False)

    # Archive old baseline
    archive_old_baseline()
```

### 2. Monitoring Metrics Storage

```python
# Store metrics in time-series database
def store_metrics(drift_results):
    """Store drift metrics for trending."""
    import influxdb

    client = influxdb.InfluxDBClient(...)

    for feature, metrics in drift_results['features'].items():
        point = {
            "measurement": "drift_metrics",
            "tags": {"feature": feature},
            "fields": {
                "psi": metrics['psi']['psi'],
                "ks_pvalue": metrics['ks_test']['p_value'],
                "drift_detected": metrics['drift_detected']
            },
            "time": datetime.utcnow()
        }
        client.write_points([point])
```

### 3. Gradual Drift Detection

```python
# Track drift trends over time
def detect_gradual_drift(historical_psi_values):
    """Detect gradual drift using moving average."""
    from scipy.stats import linregress

    # Fit linear trend
    x = range(len(historical_psi_values))
    slope, intercept, r_value, p_value, std_err = linregress(x, historical_psi_values)

    # Alert if positive trend with high confidence
    if slope > 0.01 and p_value < 0.05:
        return True  # Gradual drift detected

    return False
```

## Troubleshooting

### Issue: All Features Showing Drift

**Cause:** Thresholds too sensitive or data pipeline issue

**Solution:**
- Review thresholds
- Check data pipeline for systematic changes
- Verify data collection process

### Issue: No Drift Ever Detected

**Cause:** Thresholds too lenient or insufficient sample size

**Solution:**
- Lower thresholds
- Increase sample size
- Test with synthetic drifted data

### Issue: Unstable Drift Detection

**Cause:** High variance in small samples

**Solution:**
- Use larger sample sizes (>1000 samples)
- Apply smoothing/aggregation
- Use multiple detection methods

## Next Steps

1. **Lab 4.4**: Implement automated retraining when drift is detected
2. **Lab 4.5**: Integrate monitoring into complete MLOps system
3. **Advanced**: Add performance monitoring (accuracy, precision, recall)
4. **Dashboard**: Create visualization dashboard for drift metrics

## Learning Outcomes

After completing this lab, you understand:
- ✅ Statistical methods for drift detection (PSI, KS, JS)
- ✅ Implementing production monitoring systems
- ✅ Setting appropriate alert thresholds
- ✅ Orchestrating monitoring with Airflow
- ✅ Handling drift in production models
- ✅ Building automated alerting systems
- ✅ Best practices for baseline management
