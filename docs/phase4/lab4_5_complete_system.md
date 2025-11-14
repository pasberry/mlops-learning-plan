# Lab 4.5: Complete MLOps System Integration

**Goal**: Integrate all components into a production-ready MLOps system

**Estimated Time**: 180-240 minutes

**Prerequisites**:
- All previous Phase 4 labs completed
- Understanding of the full ML lifecycle
- Familiarity with all pipeline components

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Integrate all MLOps components end-to-end
- ✅ Orchestrate the complete ML lifecycle
- ✅ Understand production deployment patterns
- ✅ Implement operational best practices
- ✅ Build a portfolio-worthy MLOps system
- ✅ Know how to scale and evolve the system

---

## The Complete MLOps Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    COMPLETE MLOPS SYSTEM                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  DATA LAYER                                          │     │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐        │     │
│  │  │Raw Data  │──>│Feature   │──>│Training  │        │     │
│  │  │Pipeline  │   │Pipeline  │   │Data      │        │     │
│  │  └──────────┘   └──────────┘   └──────────┘        │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                          │                                     │
│  ┌───────────────────────▼──────────────────────────────┐     │
│  │  TRAINING LAYER                                      │     │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐        │     │
│  │  │Train     │──>│Evaluate  │──>│Register  │        │     │
│  │  │Model     │   │Model     │   │Model     │        │     │
│  │  └──────────┘   └──────────┘   └──────────┘        │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                          │                                     │
│  ┌───────────────────────▼──────────────────────────────┐     │
│  │  SERVING LAYER                                       │     │
│  │  ┌──────────┐         ┌──────────┐                  │     │
│  │  │Online    │         │Batch     │                  │     │
│  │  │Inference │         │Inference │                  │     │
│  │  │(FastAPI) │         │(Airflow) │                  │     │
│  │  └────┬─────┘         └────┬─────┘                  │     │
│  └───────┼────────────────────┼────────────────────────┘     │
│          │                    │                               │
│  ┌───────┴────────────────────▼────────────────────────┐     │
│  │  MONITORING LAYER                                   │     │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐       │     │
│  │  │Drift     │──>│Alert     │──>│Trigger   │       │     │
│  │  │Detection │   │System    │   │Retrain   │       │     │
│  │  └──────────┘   └──────────┘   └────┬─────┘       │     │
│  └────────────────────────────────────┬─┘─────────────┘     │
│                                       │                       │
│  ┌────────────────────────────────────▼───────────────┐      │
│  │  RETRAINING LAYER (closes the loop)                │      │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐       │      │
│  │  │Retrain   │──>│Compare   │──>│Promote   │       │      │
│  │  │Model     │   │Models    │   │to Prod   │       │      │
│  │  └──────────┘   └──────────┘   └──────────┘       │      │
│  └──────────────────────────────────────────────────────     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Master Configuration

Create `config/mlops_config.yaml`:

```yaml
# Complete MLOps System Configuration

project:
  name: "mlops-production-system"
  version: "1.0.0"
  environment: "production"  # or "staging", "development"

# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  features_dir: "data/features"
  validation_dir: "data/validation"
  predictions_dir: "data/predictions"

# Model configuration
model:
  input_dim: 6
  hidden_dim: 64
  output_dim: 2
  architecture: "SimpleClassifier"

# Features
features:
  names:
    - "age"
    - "income"
    - "tenure_days"
    - "num_purchases"
    - "avg_transaction_value"
    - "days_since_last_purchase"

  categorical: []

  preprocessing:
    log_transform: ["income", "avg_transaction_value"]
    clip_ranges:
      age: [18, 100]
      tenure_days: [0, 3650]

# Training configuration
training:
  batch_size: 128
  learning_rate: 0.001
  num_epochs: 50
  optimizer: "adam"
  early_stopping_patience: 5
  validation_split: 0.2

# Serving configuration
serving:
  online:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    timeout: 30

  batch:
    batch_size: 1024
    schedule: "0 2 * * *"  # Daily at 2 AM

# Monitoring configuration
monitoring:
  drift_detection:
    psi_threshold: 0.2
    kl_threshold: 0.1
    p_value_threshold: 0.05
    check_frequency: "daily"

  alerting:
    slack_webhook: "${SLACK_WEBHOOK_URL}"
    email_recipients:
      - "mlops-team@company.com"

# Retraining configuration
retraining:
  triggers:
    drift_detected: true
    scheduled_days: 30
    performance_threshold: 0.80

  validation:
    primary_metric: "roc_auc"
    min_improvement: 0.01

  promotion:
    auto_promote: true  # Promote automatically if better
    rollback_enabled: true
    keep_previous_versions: 5

# Model registry
registry:
  staging_dir: "models/staging"
  production_dir: "models/production"
  archive_dir: "models/archive"
  metadata_tracking: true

# Airflow DAG defaults
airflow:
  default_args:
    owner: "mlops-team"
    depends_on_past: false
    email_on_failure: true
    email_on_retry: false
    retries: 2
    retry_delay_minutes: 5

  schedules:
    data_pipeline: "0 1 * * *"      # 1 AM daily
    training: "0 4 * * 0"           # 4 AM Sunday
    batch_inference: "0 2 * * *"    # 2 AM daily
    monitoring: "0 3 * * *"         # 3 AM daily

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/mlops.log"
```

---

## Part 2: Configuration Manager

Create `ml/config_manager.py`:

```python
"""
Configuration management for the MLOps system
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str = "config/mlops_config.yaml"):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file
        """
        if self._config is not None:
            logger.info("Configuration already loaded")
            return self._config

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path) as f:
            config_raw = yaml.safe_load(f)

        # Replace environment variables
        self._config = self._substitute_env_vars(config_raw)

        logger.info("Configuration loaded successfully")
        return self._config

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config
        Format: ${VAR_NAME} or ${VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_spec = config[2:-1]
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_spec, config)
        return config

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            self.load_config()

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict:
        """Get entire configuration"""
        if self._config is None:
            self.load_config()
        return self._config


# Global instance
config = ConfigManager()
```

---

## Part 3: Master DAG Orchestrator

Create `dags/master_mlops_dag.py`:

```python
"""
Master MLOps DAG
Orchestrates the entire ML lifecycle
"""

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.config_manager import config

logger = logging.getLogger(__name__)


def check_system_health(**context):
    """
    Check overall system health before proceeding
    """
    logger.info("Checking system health...")

    checks = {
        'data_pipeline': check_data_pipeline_health(),
        'model_serving': check_model_serving_health(),
        'monitoring': check_monitoring_health()
    }

    all_healthy = all(checks.values())

    if not all_healthy:
        failed = [k for k, v in checks.items() if not v]
        logger.error(f"Health check failed for: {failed}")
        raise RuntimeError(f"System health check failed: {failed}")

    logger.info("System health check passed ✓")


def check_data_pipeline_health():
    """Check if data pipeline is healthy"""
    # Check if recent data exists
    data_dir = Path(config.get('data.raw_dir'))
    recent_data = list(data_dir.glob("**/data.csv"))
    return len(recent_data) > 0


def check_model_serving_health():
    """Check if model serving is healthy"""
    # Check if production model exists
    model_path = Path(config.get('registry.production_dir')) / "model_latest.pt"
    return model_path.exists()


def check_monitoring_health():
    """Check if monitoring is working"""
    # Check if recent monitoring data exists
    monitoring_dir = Path("monitoring/metrics")
    return monitoring_dir.exists()


def log_pipeline_execution(**context):
    """
    Log pipeline execution for auditing
    """
    execution_date = context['ds']
    dag_id = context['dag'].dag_id
    run_id = context['run_id']

    log_entry = {
        'timestamp': context['ts'],
        'execution_date': execution_date,
        'dag_id': dag_id,
        'run_id': run_id,
        'environment': config.get('project.environment')
    }

    logger.info(f"Pipeline execution: {log_entry}")

    # In production, write to database or logging service
    # write_to_audit_log(log_entry)


# Load configuration
config.load_config()

# Define DAG
default_args = {
    'owner': config.get('airflow.default_args.owner'),
    'depends_on_past': False,
    'email_on_failure': config.get('airflow.default_args.email_on_failure'),
    'retries': config.get('airflow.default_args.retries'),
    'retry_delay': timedelta(
        minutes=config.get('airflow.default_args.retry_delay_minutes')
    ),
}

with DAG(
    dag_id='master_mlops_pipeline',
    default_args=default_args,
    description='Master orchestrator for the complete MLOps system',
    schedule='0 0 * * *',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['master', 'production'],
) as dag:

    # Health check
    health_check = PythonOperator(
        task_id='system_health_check',
        python_callable=check_system_health,
        provide_context=True,
    )

    # Trigger data pipeline
    trigger_data = TriggerDagRunOperator(
        task_id='trigger_data_pipeline',
        trigger_dag_id='data_pipeline',  # From Phase 2
        wait_for_completion=True,
        poke_interval=60,
    )

    # Trigger batch inference
    trigger_inference = TriggerDagRunOperator(
        task_id='trigger_batch_inference',
        trigger_dag_id='batch_inference',  # From Lab 4.2
        wait_for_completion=True,
        poke_interval=60,
    )

    # Trigger monitoring
    trigger_monitoring = TriggerDagRunOperator(
        task_id='trigger_monitoring',
        trigger_dag_id='monitoring_drift_detection',  # From Lab 4.3
        wait_for_completion=True,
        poke_interval=60,
    )

    # Log execution
    log_execution = PythonOperator(
        task_id='log_pipeline_execution',
        python_callable=log_pipeline_execution,
        provide_context=True,
    )

    # Dependencies
    health_check >> trigger_data >> trigger_inference >> trigger_monitoring >> log_execution
```

---

## Part 4: System Monitoring Dashboard

Create `ml/dashboard/metrics_dashboard.py`:

```python
"""
Simple metrics dashboard for monitoring
"""

import pandas as pd
from pathlib import Path
from ml.monitoring.metrics_store import MetricsStore
from typing import Dict, List
import json


class MLOpsDashboard:
    """
    Generate dashboard data for the MLOps system
    """

    def __init__(self):
        self.metrics_store = MetricsStore()

    def get_system_overview(self, days: int = 7) -> Dict:
        """
        Get high-level system metrics

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with system metrics
        """
        # Drift metrics
        drift_history = self.metrics_store.get_history('drift', days=days)

        # Prediction metrics
        pred_history = self.metrics_store.get_history('predictions', days=days)

        # Model info
        from ml.retraining.model_promotion import ModelPromoter
        promoter = ModelPromoter()
        model_info = promoter.get_production_model_info()

        overview = {
            'model': {
                'version': model_info.get('version', 'unknown'),
                'deployed_at': model_info.get('promoted_at', 'unknown'),
            },
            'drift': {
                'num_checks': len(drift_history),
                'drift_detected_count': int(drift_history['drift_detected'].sum()) if not drift_history.empty else 0,
                'last_check': drift_history['timestamp'].max() if not drift_history.empty else None,
            },
            'predictions': {
                'total_predictions': int(pred_history['num_predictions'].sum()) if not pred_history.empty else 0,
                'last_batch': pred_history['timestamp'].max() if not pred_history.empty else None,
            }
        }

        return overview

    def get_drift_summary(self, days: int = 30) -> pd.DataFrame:
        """
        Get drift metrics summary

        Args:
            days: Number of days to analyze

        Returns:
            DataFrame with drift summary
        """
        drift_history = self.metrics_store.get_history('drift', days=days)

        if drift_history.empty:
            return pd.DataFrame()

        # Extract feature-level PSI values
        feature_cols = [c for c in drift_history.columns if '.psi' in c]

        summary = []
        for col in feature_cols:
            feature_name = col.split('.')[1]  # Extract feature name
            summary.append({
                'feature': feature_name,
                'mean_psi': drift_history[col].mean(),
                'max_psi': drift_history[col].max(),
                'current_psi': drift_history[col].iloc[-1] if len(drift_history) > 0 else None,
                'drift_frequency': (drift_history[col] > 0.2).sum() / len(drift_history)
            })

        return pd.DataFrame(summary)

    def get_prediction_trends(self, days: int = 30) -> pd.DataFrame:
        """
        Get prediction distribution trends

        Args:
            days: Number of days to analyze

        Returns:
            DataFrame with prediction trends
        """
        pred_history = self.metrics_store.get_history('predictions', days=days)

        if pred_history.empty:
            return pd.DataFrame()

        # Select relevant columns
        trend_cols = ['timestamp', 'num_predictions', 'psi', 'current_mean']

        return pred_history[trend_cols] if all(c in pred_history.columns for c in trend_cols) else pred_history

    def generate_report(self, output_path: str = "monitoring/reports/system_report.json"):
        """
        Generate comprehensive system report

        Args:
            output_path: Path to save report
        """
        report = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'overview': self.get_system_overview(days=7),
            'drift_summary': self.get_drift_summary(days=30).to_dict('records'),
            'prediction_trends': self.get_prediction_trends(days=30).to_dict('records')
        }

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report generated: {output_path}")
        return report


if __name__ == '__main__':
    dashboard = MLOpsDashboard()
    report = dashboard.generate_report()

    # Print summary
    print("\n=== MLOPS SYSTEM OVERVIEW ===")
    print(f"Model version: {report['overview']['model']['version']}")
    print(f"Drift checks: {report['overview']['drift']['num_checks']}")
    print(f"Drift detected: {report['overview']['drift']['drift_detected_count']} times")
    print(f"Total predictions: {report['overview']['predictions']['total_predictions']:,}")
```

---

## Part 5: End-to-End Test

Create `tests/test_e2e_system.py`:

```python
"""
End-to-end system test
Validates the complete MLOps pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.batch.batch_inference import BatchInferenceEngine
from ml.monitoring.drift_detection import FeatureMonitor, PredictionMonitor
from ml.retraining.model_comparison import ModelComparator
from ml.serving.model_service import ModelService


class TestMLOpsSystem:
    """
    End-to-end tests for the complete MLOps system
    """

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)

        return pd.DataFrame({
            'id': [f'user_{i}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'income': np.random.lognormal(11, 0.5, 100),
            'tenure_days': np.random.randint(1, 1000, 100),
            'num_purchases': np.random.poisson(5, 100),
            'avg_transaction_value': np.random.lognormal(4, 0.8, 100),
            'days_since_last_purchase': np.random.exponential(30, 100),
            'label': np.random.binomial(1, 0.4, 100)
        })

    @pytest.fixture
    def feature_names(self):
        """Feature names used in the system"""
        return [
            'age', 'income', 'tenure_days', 'num_purchases',
            'avg_transaction_value', 'days_since_last_purchase'
        ]

    def test_model_service(self, sample_data, feature_names):
        """Test model loading and prediction"""
        service = ModelService()

        # This would use a real model in production
        # For testing, we'll skip if model doesn't exist
        model_path = Path("models/production/model_latest.pt")

        if not model_path.exists():
            pytest.skip("Production model not found")

        service.load_model(
            model_path=str(model_path),
            feature_names=feature_names
        )

        # Test single prediction
        features = sample_data[feature_names].iloc[0].to_dict()
        prediction = service.predict(features)

        assert 'prediction' in prediction
        assert 'probabilities' in prediction
        assert prediction['prediction'] in [0, 1]

    def test_batch_inference(self, sample_data, feature_names):
        """Test batch inference engine"""
        model_path = Path("models/production/model_latest.pt")

        if not model_path.exists():
            pytest.skip("Production model not found")

        engine = BatchInferenceEngine(
            model_path=str(model_path),
            feature_names=feature_names,
            batch_size=32
        )

        predictions = engine.score_dataframe(sample_data, id_column='id')

        assert len(predictions) == len(sample_data)
        assert 'prediction' in predictions.columns
        assert 'prediction_proba' in predictions.columns

    def test_drift_detection(self, sample_data, feature_names):
        """Test drift detection"""
        reference_data = sample_data.copy()

        # Create drifted data
        drifted_data = sample_data.copy()
        drifted_data['age'] = drifted_data['age'] + 10  # Age drift

        monitor = FeatureMonitor(
            reference_data=reference_data,
            feature_names=feature_names
        )

        results = monitor.detect_drift(drifted_data)

        assert 'features' in results
        assert 'drift_detected' in results
        # Age should show drift
        assert results['features']['age']['psi'] > 0

    def test_prediction_monitoring(self):
        """Test prediction distribution monitoring"""
        reference_preds = np.random.binomial(1, 0.4, 1000)
        current_preds = np.random.binomial(1, 0.6, 1000)  # Drifted

        monitor = PredictionMonitor(reference_preds)
        results = monitor.detect_prediction_drift(current_preds)

        assert 'psi' in results
        assert 'drift_detected' in results
        # Should detect drift
        assert results['drift_detected']

    def test_model_comparison(self, sample_data, feature_names):
        """Test model comparison logic"""
        # Save sample validation data
        val_path = Path("data/test_validation.csv")
        val_path.parent.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(val_path, index=False)

        model_path = Path("models/production/model_latest.pt")

        if not model_path.exists():
            pytest.skip("Production model not found")

        comparator = ModelComparator(str(val_path))
        metrics = comparator.evaluate_model(str(model_path), feature_names)

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_config_manager(self):
        """Test configuration management"""
        from ml.config_manager import ConfigManager

        config = ConfigManager()
        config.load_config()

        assert config.get('project.name') is not None
        assert config.get('features.names') is not None
        assert isinstance(config.get('features.names'), list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Part 6: Deployment Guide

Create `docs/DEPLOYMENT_GUIDE.md`:

```markdown
# MLOps System Deployment Guide

## Prerequisites

- Python 3.10+
- Docker (optional, for containerization)
- PostgreSQL or compatible database (for Airflow backend)
- Redis (optional, for caching)

## Environment Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your settings
```

Required variables:
- `AIRFLOW_HOME`: Path to Airflow directory
- `MLFLOW_TRACKING_URI`: MLflow server URI (if using)
- `SLACK_WEBHOOK_URL`: For alerting
- `DATABASE_URL`: Database connection string

### 3. Initialize databases

```bash
# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

## Deployment Options

### Option 1: Local Development

```bash
# Terminal 1: Airflow webserver
airflow webserver --port 8080

# Terminal 2: Airflow scheduler
airflow scheduler

# Terminal 3: Model serving API
uvicorn ml.serving.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Compose

```bash
docker-compose up -d
```

### Option 3: Kubernetes

```bash
kubectl apply -f k8s/
```

## Post-Deployment Verification

### 1. Check Airflow

```bash
# List DAGs
airflow dags list

# Test a DAG
airflow dags test master_mlops_pipeline 2024-01-15
```

### 2. Check Model Serving

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 75000, ...}}'
```

### 3. Check Monitoring

```bash
python ml/dashboard/metrics_dashboard.py
```

## Scaling Considerations

### Horizontal Scaling (API)

- Deploy multiple FastAPI instances behind a load balancer
- Use Kubernetes HPA or AWS Auto Scaling

### Batch Processing

- Use Airflow CeleryExecutor for distributed task execution
- Consider Spark for very large datasets

### Database

- Use managed databases (AWS RDS, Google Cloud SQL)
- Implement read replicas for high read traffic

## Monitoring and Alerting

### Metrics to Monitor

- API latency (p50, p95, p99)
- Prediction throughput
- Model drift scores
- Pipeline success rate
- Data quality metrics

### Alert Channels

- Slack: Real-time alerts
- PagerDuty: Critical incidents
- Email: Daily summaries

## Backup and Recovery

### What to Backup

- Model artifacts
- Configuration files
- Airflow metadata database
- Historical predictions (for audit)

### Backup Schedule

```bash
# Daily model backups
0 4 * * * backup-models.sh

# Weekly database backups
0 2 * * 0 backup-database.sh
```

## Security

### API Security

- Use API keys or OAuth for authentication
- Rate limiting
- Input validation
- HTTPS only in production

### Data Security

- Encrypt sensitive data at rest
- Use IAM roles for cloud access
- Implement data retention policies

## Troubleshooting

### Common Issues

**Airflow DAG not appearing**
```bash
# Check for Python errors
python dags/your_dag.py

# Check Airflow logs
tail -f logs/scheduler/latest/*.log
```

**Model serving errors**
```bash
# Check model file exists
ls -lh models/production/

# Test model loading
python -c "import torch; torch.load('models/production/model_latest.pt')"
```

**Drift detection not working**
```bash
# Check monitoring data
ls -lh monitoring/metrics/
```

## Maintenance

### Weekly Tasks

- Review drift alerts
- Check model performance
- Update documentation

### Monthly Tasks

- Review and archive old models
- Update dependencies
- Capacity planning review

### Quarterly Tasks

- Security audit
- Performance optimization
- Architecture review
```

---

## Production Best Practices Checklist

### Code Quality
- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Unit tests (>80% coverage)
- ✅ Integration tests
- ✅ Code linting (black, flake8)

### Operations
- ✅ Logging at appropriate levels
- ✅ Monitoring and alerting
- ✅ Graceful error handling
- ✅ Automated backups
- ✅ Disaster recovery plan

### ML-Specific
- ✅ Model versioning
- ✅ Experiment tracking
- ✅ Data versioning
- ✅ Feature store (optional)
- ✅ Model registry

### Security
- ✅ Authentication/authorization
- ✅ Input validation
- ✅ Secrets management
- ✅ Encryption at rest/transit
- ✅ Audit logging

### Performance
- ✅ API response time < 100ms (p95)
- ✅ Batch processing throughput documented
- ✅ Resource utilization monitored
- ✅ Cost optimization
- ✅ Caching strategy

---

## Exercises

### Exercise 1: Implement Health Checks

Add comprehensive health checks:

```python
def comprehensive_health_check():
    """Check all system components"""
    health = {
        'database': check_database_connection(),
        'model': check_model_loaded(),
        'data': check_recent_data_available(),
        'monitoring': check_monitoring_active(),
        'disk_space': check_disk_space()
    }

    overall_status = "healthy" if all(health.values()) else "degraded"

    return {
        'status': overall_status,
        'components': health,
        'timestamp': datetime.now().isoformat()
    }
```

### Exercise 2: Add Performance Benchmarks

Benchmark your system:

```python
def benchmark_inference_speed():
    """Measure inference latency"""
    import time

    # Generate test data
    test_data = generate_test_samples(1000)

    # Benchmark single prediction
    start = time.time()
    for sample in test_data[:100]:
        predict(sample)
    single_latency = (time.time() - start) / 100

    # Benchmark batch prediction
    start = time.time()
    predict_batch(test_data)
    batch_latency = time.time() - start

    print(f"Single prediction: {single_latency*1000:.2f}ms")
    print(f"Batch throughput: {len(test_data)/batch_latency:.0f} samples/sec")
```

### Exercise 3: Implement Circuit Breaker

Add fault tolerance:

```python
class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    """
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

---

## Scaling to Production

### Startup Scale (< 1M predictions/day)

```
- Single server for Airflow
- FastAPI with 4 workers
- PostgreSQL database
- Local file storage
- Cost: ~$200/month
```

### Growth Scale (1M-100M predictions/day)

```
- Airflow with CeleryExecutor (3-5 workers)
- Auto-scaling FastAPI instances (5-20 instances)
- Managed PostgreSQL (RDS/Cloud SQL)
- S3/GCS for storage
- Redis for caching
- Cost: ~$2,000-5,000/month
```

### Enterprise Scale (100M+ predictions/day)

```
- Kubernetes cluster for Airflow
- Auto-scaling model serving (50+ instances)
- Distributed database (Snowflake/BigQuery)
- Feature store (Feast, Tecton)
- Real-time monitoring (Datadog, NewRelic)
- Cost: $20,000+/month
```

---

## Key Takeaways

✅ **Integration is key**: All components must work together seamlessly
✅ **Configuration management**: Centralize and version all configs
✅ **Monitoring first**: If you can't measure it, you can't improve it
✅ **Automation everywhere**: Minimize manual interventions
✅ **Fail gracefully**: Plan for failures, implement fallbacks
✅ **Documentation**: Your future self will thank you

---

## Congratulations! 🎉

You've built a complete, production-ready MLOps system that includes:

1. ✅ **Data pipelines** - Automated ETL and feature engineering
2. ✅ **Training pipelines** - Reproducible model training
3. ✅ **Serving** - Both online and batch inference
4. ✅ **Monitoring** - Drift detection and alerting
5. ✅ **Retraining** - Automated model updates
6. ✅ **Orchestration** - Airflow DAGs for everything

**This is portfolio-grade work.** You now understand how to:
- Design and implement end-to-end ML systems
- Deploy models to production
- Monitor and maintain ML systems
- Scale systems from prototype to production

---

## What's Next?

### Level Up Your System

1. **Add feature store** (Feast, Tecton)
2. **Implement A/B testing** framework
3. **Add real-time monitoring** (Prometheus, Grafana)
4. **Distributed training** (Horovod, PyTorch Distributed)
5. **Model explainability** (SHAP, LIME)

### Keep Learning

- Study production ML at scale (Google, Netflix, Uber blogs)
- Join MLOps communities
- Contribute to open source
- Build projects in different domains

### Share Your Work

- Document your system in a blog post
- Create a video walkthrough
- Open source components
- Present at meetups

---

**You've completed the MLOps Mastery course!** 🚀

**Remember**: The best MLOps engineers are those who ship to production.

**Now go build amazing ML systems!**
