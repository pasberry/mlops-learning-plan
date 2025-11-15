# Lab 4.4: Automated Retraining Solution

Complete automated model retraining pipeline with intelligent model comparison and promotion.

## Overview

Implements the critical "close the loop" phase of MLOps:
- **Trigger Detection** - Retrain when drift detected or on schedule
- **Automated Training** - Train new model candidate
- **A/B Comparison** - Compare new model with production baseline
- **Smart Promotion** - Promote only if statistically better
- **Safe Rollback** - Backup old model before promotion

## Components

### 1. Model Comparator (`ml/training/model_comparator.py`)

**Features:**
- Multiple metric evaluation (AUC, F1, accuracy, precision, recall)
- Statistical significance testing (bootstrap)
- Configurable improvement thresholds
- Comprehensive comparison reports

**Decision Logic:**
```python
promote = (
    improvement >= threshold AND
    statistically_significant
)
```

### 2. Retraining DAG (`dags/retraining_dag.py`)

**Workflow:**
```
Check Trigger → Prepare Data → Train Model → Compare → Promote/Skip
     ↓                                              ↓
Skip if no drift                            Promote if better
```

**Triggers:**
- Data drift detected
- Scheduled interval (e.g., weekly)
- Manual trigger
- Days since last training threshold

## Quick Start

### 1. Setup

```bash
# Create directories
mkdir -p /home/user/mlops-learning-plan/models/candidates
mkdir -p /home/user/mlops-learning-plan/data/training
mkdir -p /home/user/mlops-learning-plan/data/reports

# Create test training data
python3 << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

# Generate features
data = {
    'age': np.random.normal(35, 10, n_samples).clip(18, 80),
    'income': np.random.normal(65000, 20000, n_samples).clip(20000, 200000),
    'credit_score': np.random.normal(700, 50, n_samples).clip(300, 850),
    'num_purchases': np.random.poisson(5, n_samples),
    'account_age_days': np.random.uniform(1, 3650, n_samples),
    'avg_transaction': np.random.lognormal(4.5, 0.8, n_samples),
    'num_returns': np.random.poisson(1, n_samples),
    'is_premium': np.random.binomial(1, 0.3, n_samples),
    'region': np.random.randint(1, 5, n_samples),
    'category_preference': np.random.randint(1, 10, n_samples),
}

# Generate target (example: higher income + premium → positive)
target = (
    (data['income'] > 70000).astype(int) * 0.3 +
    (data['is_premium'] == 1).astype(int) * 0.4 +
    np.random.random(n_samples) * 0.3
)
target = (target > 0.5).astype(int)

df = pd.DataFrame(data)
df['target'] = target

# Save training and test sets
train_df = df.iloc[:8000]
test_df = df.iloc[8000:]

train_df.to_csv('/home/user/mlops-learning-plan/data/processed/training_data.csv', index=False)
test_df.to_csv('/home/user/mlops-learning-plan/data/processed/test_data.csv', index=False)

print(f"✓ Training data: {len(train_df)} samples")
print(f"✓ Test data: {len(test_df)} samples")
EOF
```

### 2. Run Model Comparison

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_4_solution

# Compare two models
python ml/training/model_comparator.py \
    --baseline /home/user/mlops-learning-plan/models/production/model.pt \
    --new /home/user/mlops-learning-plan/models/candidates/model_new.pt \
    --test-data /home/user/mlops-learning-plan/data/processed/test_data.csv \
    --output /home/user/mlops-learning-plan/data/reports/comparison.json \
    --metric auc \
    --threshold 0.02
```

### 3. Run Retraining DAG

```bash
# Copy to Airflow
cp dags/retraining_dag.py $AIRFLOW_HOME/dags/
cp -r ml/ $AIRFLOW_HOME/

# Trigger retraining
airflow dags trigger model_retraining

# Force retraining (ignore triggers)
airflow dags trigger model_retraining \
    --conf '{"force_retrain": true}'

# Monitor progress
airflow dags list-runs -d model_retraining --state running
```

## Expected Output

### Model Comparison Report

```
================================================================================
MODEL COMPARISON RESULTS
================================================================================

Primary Metric: AUC
Baseline: 0.8234
New:      0.8456
Improvement: 0.0222 (2.70%)

All Metrics:
  accuracy    : 0.7845 → 0.8012 (+2.13%)
  precision   : 0.7654 → 0.7823 (+2.21%)
  recall      : 0.8123 → 0.8298 (+2.15%)
  f1          : 0.7876 → 0.8052 (+2.24%)
  auc         : 0.8234 → 0.8456 (+2.70%)
  log_loss    : 0.4523 → 0.4312 (-4.66%)

Decision Factors:
  Meets threshold:          True
  Statistically significant: True

PROMOTION DECISION: PROMOTE
Reason: New model shows 0.0222 improvement in auc (threshold: 0.02) and is statistically significant
================================================================================
```

### DAG Execution Log

```
[2025-11-15 10:00:00] INFO - Checking retraining trigger
[2025-11-15 10:00:01] INFO - Drift detected - retraining needed
[2025-11-15 10:00:02] INFO - Preparing training data
[2025-11-15 10:00:05] INFO - Training model for 50 epochs
[2025-11-15 10:05:30] INFO - Training complete. Best val loss: 0.4123
[2025-11-15 10:05:35] INFO - Comparing new model with baseline
[2025-11-15 10:05:40] INFO - New model promoted to production
```

## Configuration

### Retraining Triggers

```python
# In DAG params
params = {
    'force_retrain': False,              # Manual trigger
    'max_days_since_training': 30,       # Auto-retrain after N days
    'drift_report_path': '...',          # Check for drift
}
```

### Model Comparison Thresholds

```python
comparator = ModelComparator(
    primary_metric='auc',           # auc, f1, accuracy
    improvement_threshold=0.02,     # 2% minimum improvement
    statistical_significance_threshold=0.05  # 95% confidence
)
```

### Training Configuration

```python
params = {
    'num_epochs': 50,
    'batch_size': 32,
    'target_column': 'target',
}
```

## Best Practices

### 1. Always Use Holdout Test Set

```python
# Never use validation set for final comparison!
train_df, temp_df = train_test_split(df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)

# Train on train_df
# Tune on val_df
# Compare on test_df (unseen during training)
```

### 2. Statistical Significance Testing

```python
# Bootstrap test ensures improvement is real, not random
is_significant = comparator._bootstrap_significance_test(
    baseline_model, new_model, X_test, y_test,
    metric='auc', n_bootstrap=100
)
```

### 3. Safe Model Promotion

```python
# Always backup before promoting
if Path(prod_model_path).exists():
    backup_path = prod_model_path.replace('.pt', '_backup.pt')
    shutil.copy2(prod_model_path, backup_path)

# Then promote
shutil.copy2(new_model_path, prod_model_path)
```

### 4. Track Training History

```python
# Save metadata with each training
metadata = {
    'trained_at': datetime.utcnow().isoformat(),
    'train_samples': len(X_train),
    'val_loss': best_val_loss,
    'drift_detected': True,
    'promoted': True
}
```

## Advanced Usage

### Custom Retraining Logic

```python
def check_retraining_trigger(**context):
    """Custom trigger logic."""
    # Check performance degradation
    recent_metrics = get_recent_production_metrics()
    if recent_metrics['auc'] < 0.75:
        return 'prepare_training_data'

    # Check data volume
    new_samples = count_new_samples_since_last_training()
    if new_samples > 100000:
        return 'prepare_training_data'

    return 'skip_retraining'
```

### Multi-Metric Promotion

```python
def should_promote_model(results):
    """Promote if multiple metrics improve."""
    improvements = results['improvements']

    # Check multiple metrics
    auc_improved = improvements['auc']['absolute_improvement'] > 0.02
    f1_improved = improvements['f1']['absolute_improvement'] > 0.01
    recall_improved = improvements['recall']['absolute_improvement'] > 0

    # All must improve
    return auc_improved and f1_improved and recall_improved
```

### Champion/Challenger Pattern

```python
# Keep both models in production
# Route small % of traffic to challenger
# Promote after observing real performance
def deploy_as_challenger(new_model_path):
    challenger_path = 'models/challenger/model.pt'
    shutil.copy2(new_model_path, challenger_path)

    # Update routing config to send 10% to challenger
    update_traffic_routing(challenger_percentage=10)
```

## Troubleshooting

**Issue:** New model never promoted
- **Solution:** Lower improvement_threshold or check if models are actually different

**Issue:** Retraining triggered too often
- **Solution:** Increase drift thresholds or max_days_since_training

**Issue:** Training fails with OOM
- **Solution:** Reduce batch_size or model size

## Integration with Other Labs

```
Lab 4.3 (Monitoring) → Lab 4.4 (Retraining)
        ↓                        ↓
    Drift Report          New Model
                             ↓
                    Lab 4.1 (Serving)
```

## Next Steps

- **Lab 4.5**: Integrate all components into master pipeline
- **Capstone**: Apply to feed ranking system
- **Advanced**: Implement A/B testing, multi-model ensembles

## Learning Outcomes

✅ Automated retraining trigger logic
✅ Statistical model comparison
✅ Safe model promotion strategies
✅ Production retraining best practices
✅ Closing the MLOps loop
