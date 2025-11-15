# Lab 3.2 Solution: Training DAG - Airflow Integration

Complete solution for orchestrating PyTorch model training with Apache Airflow.

## Overview

This solution implements:
- End-to-end training orchestration with Airflow
- Model registry for versioning and tracking
- Quality gates and validation checks
- XCom-based data passing between tasks
- Automated model registration to staging

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              TRAINING PIPELINE DAG                  │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐             │
│  │   Validate   │───▶│    Train     │             │
│  │    Data      │    │    Model     │             │
│  └──────────────┘    └──────────────┘             │
│                             │                      │
│                             ▼                      │
│                      ┌──────────────┐             │
│                      │   Evaluate   │             │
│                      │    Model     │             │
│                      └──────────────┘             │
│                             │                      │
│                             ▼                      │
│                      ┌──────────────┐             │
│                      │   Register   │             │
│                      │    Model     │             │
│                      └──────────────┘             │
│                             │                      │
│                             ▼                      │
│                      ┌──────────────┐             │
│                      │    Notify    │             │
│                      └──────────────┘             │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
lab3_2_solution/
├── dags/
│   ├── training_pipeline.py         # Main DAG definition
│   └── training_pipeline_tasks.py   # Task implementations
├── ml/
│   ├── __init__.py
│   └── registry/
│       ├── __init__.py
│       └── model_registry.py        # Model registry
└── README.md
```

## Prerequisites

1. **Lab 3.1 completed** - Training code must be available
2. **Airflow installed and running**
3. **Sample data generated** - Run Lab 3.1 data generation script

## Quick Start

### Step 1: Setup Airflow Environment

```bash
cd /home/user/mlops-learning-plan/solutions/phase3/lab3_2_solution

# Set AIRFLOW_HOME
export AIRFLOW_HOME=/home/user/mlops-learning-plan/airflow

# Add to PYTHONPATH
export PYTHONPATH=/home/user/mlops-learning-plan/solutions/phase3/lab3_2_solution:$PYTHONPATH
```

### Step 2: Copy DAGs to Airflow

```bash
# Copy DAG files to Airflow dags folder
cp dags/*.py $AIRFLOW_HOME/dags/
```

### Step 3: Start Airflow (if not running)

```bash
# Terminal 1: Webserver
airflow webserver --port 8080

# Terminal 2: Scheduler (new terminal)
export AIRFLOW_HOME=/home/user/mlops-learning-plan/airflow
airflow scheduler
```

### Step 4: Verify DAG Loaded

```bash
# Check DAG is recognized
airflow dags list | grep training_pipeline

# Check for import errors
airflow dags list-import-errors
```

### Step 5: Test Individual Tasks

```bash
# Test validate_features task
airflow tasks test training_pipeline validate_features 2024-11-15

# Test train_model task (takes longer)
airflow tasks test training_pipeline train_model 2024-11-15

# Test evaluate_model task
airflow tasks test training_pipeline evaluate_model 2024-11-15

# Test register_model task
airflow tasks test training_pipeline register_model 2024-11-15
```

### Step 6: Trigger Complete DAG Run

**Via Airflow UI:**
1. Navigate to http://localhost:8080
2. Find "training_pipeline" DAG
3. Toggle it ON (unpause)
4. Click "Trigger DAG" (play button ▶)
5. Monitor task progress in Graph or Grid view

**Via CLI:**
```bash
airflow dags trigger training_pipeline
```

### Step 7: Monitor Execution

Check task logs in Airflow UI:
1. Click on a task (colored box in Graph view)
2. Click "Log" button
3. View real-time execution logs

Expected output in logs:
```
[validate_features] ✅ train data: data/features/v1/train/data.parquet (234.5 KB)
[validate_features] ✅ val data: data/features/v1/val/data.parquet (50.2 KB)
[validate_features] ✅ test data: data/features/v1/test/data.parquet (50.1 KB)

[train_model] 🚀 Running training command...
[train_model] Epoch 1/30: Train Loss: 0.5234, Val AUC: 0.7654
...
[train_model] ✅ Training complete

[evaluate_model] 📊 Model Metrics:
[evaluate_model]    test_accuracy: 0.7453
[evaluate_model]    test_auc: 0.7891

[register_model] ✅ Model registered: ctr_model v1 (staging)
[register_model]    Path: models/staging/ctr_model/v1

[notify_completion] 🎉 Training Pipeline Complete!
```

## Model Registry Structure

After successful run:

```
models/
├── staging/
│   └── ctr_model/
│       └── v1/
│           ├── model.pt           # Model weights
│           ├── config.yaml        # Training configuration
│           ├── metrics.json       # Evaluation metrics
│           └── metadata.json      # Registry metadata
└── production/
    └── ctr_model/
        ├── current -> v1/         # Symlink to current version
        └── v1/                    # Promoted model
            ├── model.pt
            ├── config.yaml
            ├── metrics.json
            └── metadata.json
```

## DAG Tasks Explained

### 1. validate_features
**Purpose**: Verify training data exists and is valid
- Checks train/val/test directories exist
- Validates files are not empty
- Passes data path to downstream tasks via XCom

### 2. train_model
**Purpose**: Execute PyTorch training script
- Calls Lab 3.1 training code
- Creates timestamped run directory
- Saves model checkpoints and metrics
- Passes run_dir to downstream via XCom

### 3. evaluate_model
**Purpose**: Validate model quality
- Loads metrics from training run
- Checks against minimum thresholds (configurable)
- Fails pipeline if quality insufficient
- Passes metrics to downstream via XCom

### 4. register_model
**Purpose**: Version and register the model
- Copies model to staging registry
- Generates version number (v1, v2, ...)
- Saves all artifacts (weights, config, metrics)
- Passes version to downstream via XCom

### 5. notify_completion
**Purpose**: Send completion notifications
- Prints summary message (production: Slack/email)
- Includes metrics and model location
- Provides next steps for team

## Configuration

### DAG Parameters

Edit in DAG file or trigger with custom params:

```python
params={
    'config_path': '../lab3_1_solution/config/model_config.yaml',
    'model_name': 'ctr_model',
    'min_auc': 0.65,        # Minimum AUC threshold
    'min_accuracy': 0.60     # Minimum accuracy threshold
}
```

### Model Registry Usage

```python
from ml.registry.model_registry import ModelRegistry

# Initialize
registry = ModelRegistry(base_dir='models')

# List staging models
print(registry.list_models('staging'))
# Output: {'ctr_model': ['v1', 'v2', 'v3']}

# Get latest version
latest = registry.get_latest_version('ctr_model', stage='staging')
print(latest)  # 'v3'

# Promote to production
registry.promote_to_production('ctr_model', 'v3')
# Creates models/production/ctr_model/v3/
# Creates symlink: models/production/ctr_model/current -> v3/
```

## Scheduling Options

```python
# Weekly (default)
schedule_interval='@weekly'

# Daily
schedule_interval='@daily'

# Cron: Every Monday at 2 AM
schedule_interval='0 2 * * 1'

# Manual only
schedule_interval=None
```

## Integration with Feature Pipeline

Make training wait for fresh features:

```python
from airflow.sensors.external_task import ExternalTaskSensor

# Add to DAG
wait_for_features = ExternalTaskSensor(
    task_id='wait_for_features',
    external_dag_id='feature_engineering_pipeline',
    external_task_id='save_features',
    allowed_states=['success'],
    timeout=600
)

# Update dependencies
wait_for_features >> validate_data >> train >> ...
```

## XCom Data Flow

Data passed between tasks:

```
validate_features → features_path → train_model
train_model → run_dir → evaluate_model
train_model → run_dir → register_model
evaluate_model → metrics → register_model
evaluate_model → metrics → notify_completion
register_model → model_version → notify_completion
register_model → registered_path → notify_completion
```

View XCom in Airflow UI:
- Admin → XComs
- Filter by DAG: training_pipeline
- See all key-value pairs

## Testing

```bash
# Parse DAG (check syntax)
python dags/training_pipeline.py

# List DAG
airflow dags list | grep training_pipeline

# Test specific task
airflow tasks test training_pipeline validate_features 2024-11-15

# Trigger full run
airflow dags trigger training_pipeline

# Check run status
airflow dags list-runs -d training_pipeline
```

## Key Features

1. **Thin Orchestration Layer**: DAG calls scripts, doesn't contain ML logic
2. **XCom Coordination**: Pass data between tasks seamlessly
3. **Model Versioning**: Automatic version numbering (v1, v2, ...)
4. **Quality Gates**: Fail pipeline if model doesn't meet thresholds
5. **Reproducibility**: Config + weights + metrics saved together
6. **Idempotent**: Can re-run without side effects

## Troubleshooting

**Issue**: DAG not appearing in UI
```bash
# Check import errors
airflow dags list-import-errors

# Check PYTHONPATH
echo $PYTHONPATH

# Check DAG file location
ls $AIRFLOW_HOME/dags/
```

**Issue**: Task fails with import errors
```bash
# Ensure __init__.py files exist
find ml -name "__init__.py"

# Check Python can import
python -c "from ml.registry.model_registry import ModelRegistry"
```

**Issue**: XCom data not available
- Check task_ids match in xcom_pull
- Ensure previous task completed successfully
- View XCom in Admin → XComs

**Issue**: Model registration fails
```bash
# Check directory permissions
ls -la models/

# Verify run_dir exists
ls experiments/runs/

# Check model file exists
ls experiments/runs/ctr_model_*/model_best.pt
```

**Issue**: Training script not found
- Update path in training_pipeline_tasks.py
- Ensure Lab 3.1 solution is accessible
- Use absolute paths if needed

## Production Enhancements

For production, add:

1. **Notifications**: Slack webhook, email alerts
2. **A/B Testing**: Shadow traffic, champion/challenger
3. **Rollback**: Automatic revert if quality degrades
4. **Model Cards**: Auto-generate documentation
5. **Drift Detection**: Compare new model vs production
6. **Resource Management**: GPU allocation, memory limits
7. **Logging**: Centralized logging (ELK stack)
8. **Monitoring**: Model training metrics dashboard

## Next Steps

1. **Lab 3.3**: Add experiment tracking to compare multiple runs
2. **Lab 3.4**: Build two-tower ranking model
3. **Phase 4**: Deploy model for serving

## References

- Apache Airflow Documentation: https://airflow.apache.org/
- XCom Guide: https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html
- DAG Best Practices
