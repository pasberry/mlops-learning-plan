# Lab 4.5: Complete MLOps System

The final integration - complete end-to-end MLOps pipeline orchestrating all previous labs into a production system.

## Overview

This master pipeline represents the **complete MLOps loop**:

```
┌─────────────────────────────────────────────────────────┐
│                   PRODUCTION MLOPS LOOP                  │
└─────────────────────────────────────────────────────────┘

Data → Features → Inference → Monitor → Retrain → Deploy
  ↑                                                     ↓
  └─────────────────────────────────────────────────────┘
                    (Close the Loop)
```

## Architecture

```
master_dag.py
├── Stage 1: Data Ingestion & Validation
├── Stage 2: Feature Engineering
├── Stage 3: Batch Inference (Lab 4.2)
├── Stage 4: Monitoring & Drift Detection (Lab 4.3)
├── Stage 5: Automated Retraining (Lab 4.4)
└── Stage 6: Pipeline Reporting
```

## Complete System Components

### Labs Integration

```
Lab 4.1 (Serving)    → Real-time API predictions
Lab 4.2 (Batch)      → Batch scoring pipeline
Lab 4.3 (Monitoring) → Drift detection
Lab 4.4 (Retraining) → Model updates
Lab 4.5 (This Lab)   → Master orchestration
```

### DAG Dependencies

```
mlops_master_pipeline (daily)
├─→ batch_inference (triggered)
├─→ model_monitoring (triggered)
└─→ model_retraining (triggered if drift)
```

## Setup

### 1. Prepare All Dependencies

```bash
# Copy all DAGs to Airflow
cp ../lab4_2_solution/dags/batch_inference_dag.py $AIRFLOW_HOME/dags/
cp ../lab4_3_solution/dags/monitoring_dag.py $AIRFLOW_HOME/dags/
cp ../lab4_4_solution/dags/retraining_dag.py $AIRFLOW_HOME/dags/
cp dags/master_dag.py $AIRFLOW_HOME/dags/

# Copy ML modules
cp -r ../lab4_2_solution/ml $AIRFLOW_HOME/
cp -r ../lab4_3_solution/monitoring $AIRFLOW_HOME/
cp -r ../lab4_4_solution/ml/training $AIRFLOW_HOME/ml/

# Verify DAGs loaded
airflow dags list | grep -E '(mlops_master|batch_inference|model_monitoring|model_retraining)'
```

### 2. Create Daily Data Generator

```bash
# Simulate daily data ingestion
python3 << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

def generate_daily_data(date_str):
    """Generate simulated daily data."""
    np.random.seed(hash(date_str) % 2**32)

    n_samples = 1000

    data = {
        'user_id': range(n_samples),
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

    df = pd.DataFrame(data)

    # Save
    output_path = Path('/home/user/mlops-learning-plan/data/raw/daily_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Generated {len(df)} samples for {date_str}")

# Generate for today
from datetime import datetime
generate_daily_data(datetime.utcnow().strftime('%Y-%m-%d'))
EOF
```

## Running the Complete System

### Method 1: Trigger Master DAG

```bash
# Run complete pipeline
airflow dags trigger mlops_master_pipeline

# Monitor execution
airflow dags list-runs -d mlops_master_pipeline --state running

# Watch logs
airflow dags show mlops_master_pipeline
```

### Method 2: Scheduled Execution

The master DAG runs daily at 1 AM automatically:
```python
schedule_interval='0 1 * * *'  # Daily at 1 AM
```

### Method 3: Manual Sub-DAG Triggering

```bash
# Run components individually
airflow dags trigger batch_inference
airflow dags trigger model_monitoring
airflow dags trigger model_retraining
```

## Pipeline Execution Flow

### Day 1: Normal Operation

```
1. validate_raw_data          ✓ 1000 rows ingested
2. extract_features           ✓ Features engineered
3. trigger_batch_inference    ✓ 1000 predictions generated
4. trigger_monitoring         ✓ No drift detected
5. check_drift                ✓ Drift status: False
6. trigger_retraining         ⊘ Skipped (no drift)
7. generate_pipeline_report   ✓ Pipeline complete
```

### Day 30: Drift Detected

```
1. validate_raw_data          ✓ 1000 rows ingested
2. extract_features           ✓ Features engineered
3. trigger_batch_inference    ✓ 1000 predictions generated
4. trigger_monitoring         ⚠ DRIFT DETECTED (3 features)
5. check_drift                ⚠ Drift status: True
6. trigger_retraining         ▶ Training new model...
   └─→ prepare_data           ✓ Data prepared
   └─→ train_model            ✓ Model trained
   └─→ compare_models         ✓ New model better (+3.2% AUC)
   └─→ promote_model          ✓ Model promoted to production
7. generate_pipeline_report   ✓ Pipeline complete with retraining
```

## Expected Output

### Successful Run (No Drift)

```
================================================================================
MLOPS PIPELINE EXECUTION REPORT
================================================================================
Execution Time: 2025-11-15T01:00:00.000000
Raw Data: 1000 rows processed
Drift Detected: False
================================================================================
Pipeline completed successfully!
================================================================================
```

### Run with Drift and Retraining

```
================================================================================
MLOPS PIPELINE EXECUTION REPORT
================================================================================
Execution Time: 2025-11-15T01:00:00.000000
Raw Data: 1000 rows processed
Drift Detected: True

Sub-Pipeline Results:
- Batch Inference: 1000 predictions generated
- Monitoring: 3/10 features drifted (30%)
- Retraining: New model promoted (AUC: 0.8234 → 0.8556)
================================================================================
Pipeline completed successfully!
================================================================================
```

## Monitoring the System

### Airflow UI

1. Open http://localhost:8080
2. View DAG Graph: `mlops_master_pipeline`
3. Check task status and logs
4. Monitor sub-DAG triggers

### Logs

```bash
# Master pipeline logs
airflow tasks logs mlops_master_pipeline generate_pipeline_report 2025-11-15

# Batch inference logs
airflow tasks logs batch_inference run_predictions 2025-11-15

# Monitoring logs
airflow tasks logs model_monitoring detect_feature_drift 2025-11-15

# Retraining logs
airflow tasks logs model_retraining train_model 2025-11-15
```

### Metrics

```bash
# Check pipeline success rate
airflow dags list-runs -d mlops_master_pipeline --state success

# Check average execution time
airflow dags list-runs -d mlops_master_pipeline
```

## Production Considerations

### 1. Error Handling

```python
# Add alerting for critical failures
from airflow.operators.email import EmailOperator

send_alert = EmailOperator(
    task_id='send_failure_alert',
    to='oncall@example.com',
    subject='MLOps Pipeline Failed',
    html_content='Pipeline failed at {{ task_instance.task_id }}',
    trigger_rule='one_failed'
)
```

### 2. SLA Monitoring

```python
# Set SLAs for each stage
validate_data = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw_data,
    sla=timedelta(minutes=10)  # Must complete within 10 min
)
```

### 3. Data Versioning

```python
# Version all data artifacts
execution_date = context['execution_date']
features_path = f'data/features/features_{execution_date}.csv'
predictions_path = f'data/predictions/pred_{execution_date}.csv'
```

### 4. Backfill Strategy

```bash
# Reprocess historical dates
airflow dags backfill mlops_master_pipeline \
    --start-date 2025-11-01 \
    --end-date 2025-11-15 \
    --reset-dagruns
```

### 5. Resource Management

```python
# Set resource pools for heavy tasks
run_batch_inference = TriggerDagRunOperator(
    task_id='trigger_batch_inference',
    pool='gpu_pool',  # Use GPU pool
    priority_weight=10  # High priority
)
```

## Testing the Complete System

### End-to-End Test

```bash
# 1. Generate test data
python scripts/generate_daily_data.py --date 2025-11-15

# 2. Run pipeline
airflow dags test mlops_master_pipeline 2025-11-15

# 3. Verify outputs
ls -lh data/predictions/
ls -lh data/monitoring/reports/
ls -lh models/production/
```

### Integration Test

```python
def test_complete_pipeline():
    """Test entire pipeline end-to-end."""
    # Generate test data
    generate_test_data()

    # Trigger pipeline
    run_id = trigger_dag('mlops_master_pipeline')

    # Wait for completion
    wait_for_dag_run(run_id, timeout=3600)

    # Verify outputs
    assert Path('data/predictions/batch_predictions.csv').exists()
    assert Path('data/monitoring/reports/latest_drift_report.json').exists()

    # Check predictions
    df = pd.read_csv('data/predictions/batch_predictions.csv')
    assert len(df) > 0
    assert 'prediction_score' in df.columns

    print("✓ End-to-end test passed")
```

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SYSTEM                      │
└─────────────────────────────────────────────────────────────┘

┌──────────┐         ┌──────────┐         ┌──────────┐
│  Raw     │────────>│ Feature  │────────>│  Batch   │
│  Data    │         │Engineering│        │Inference │
└──────────┘         └──────────┘         └──────────┘
                                                 │
                                                 ▼
┌──────────┐         ┌──────────┐         ┌──────────┐
│  Model   │<────────│Retraining│<────────│Monitoring│
│Deployment│         │  (Auto)  │         │& Drift   │
└──────────┘         └──────────┘         └──────────┘
     │                                           │
     └──────────────────────────────────────────┘
              (Close the Loop)
```

## Key Achievements

### You've Built:

✅ **Complete Data Pipeline** - Ingestion → Features → Predictions
✅ **Automated Monitoring** - Drift detection and alerting
✅ **Intelligent Retraining** - Trigger-based model updates
✅ **Safe Deployment** - Model comparison and promotion
✅ **Production Orchestration** - Reliable, scheduled execution
✅ **Closed Loop System** - Self-healing ML infrastructure

### This System Handles:

✅ Daily batch predictions at scale
✅ Continuous monitoring for drift
✅ Automatic retraining when needed
✅ Safe model promotion with validation
✅ Complete audit trail and logging
✅ Error recovery and alerting

## Next Steps

### Immediate

1. Deploy to production Airflow cluster
2. Connect to real data sources
3. Set up production monitoring dashboards
4. Configure alerting (PagerDuty, Slack)

### Short Term

1. Add A/B testing for model deployment
2. Implement canary releases
3. Add performance regression tests
4. Set up MLflow for experiment tracking

### Long Term

1. Multi-model ensembles
2. Online learning integration
3. Distributed training (Spark, Ray)
4. Real-time streaming inference

## Capstone Project

You're now ready for the **Capstone: Feed Ranking System**

The capstone applies everything you've learned to build:
- Two-tower neural architecture
- Real-time and batch inference
- Complete monitoring and retraining
- Production-grade system

**Proceed to**: `/solutions/capstone/module7_solution/`

## Learning Outcomes

After completing this lab, you've mastered:

✅ **End-to-end MLOps** - Complete production ML lifecycle
✅ **Airflow Orchestration** - Complex DAG dependencies and triggers
✅ **Production Best Practices** - Error handling, monitoring, versioning
✅ **System Integration** - Connecting multiple components seamlessly
✅ **Operational ML** - Running and maintaining ML in production

**Congratulations! You've built a production MLOps system.** 🎉
