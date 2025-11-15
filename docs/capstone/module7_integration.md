# Module 7: Final Integration & End-to-End Testing

**Estimated Time**: 2-3 days
**Difficulty**: Medium

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Integrate all components into a cohesive system
- ✅ Build master orchestration DAGs
- ✅ Perform end-to-end system testing
- ✅ Create comprehensive documentation
- ✅ Demonstrate the complete ML lifecycle
- ✅ Identify and fix integration issues
- ✅ Plan for production deployment

## Overview

This final module brings everything together:
1. Integrate all 6 previous modules
2. Build master orchestration DAG
3. Test the complete system end-to-end
4. Document the system thoroughly
5. Create a demo showcasing the ML lifecycle
6. Prepare for production deployment

**Key Principle**: "The whole is greater than the sum of its parts." Individual components work, but integration reveals the true system behavior.

## Step 1: Integration Checklist

### Component Integration Matrix

| Component | Status | Integration Points | Dependencies |
|-----------|--------|-------------------|--------------|
| **Data Generation** | ✅ | → ETL DAG | None |
| **ETL Pipeline** | ✅ | → Training DAG<br>→ Monitoring DAG | Data Generation |
| **Model Training** | ✅ | → Model Registry<br>→ Serving API | ETL Pipeline |
| **Model Serving** | ✅ | → Prediction Logs<br>→ Monitoring | Model Registry |
| **Monitoring** | ✅ | → Retraining DAG (trigger) | Prediction Logs, ETL |
| **Retraining** | ✅ | → Training DAG<br>→ Model Promotion | Monitoring, ETL |

### Integration Tasks

**1. Data Flow Integration**
- [ ] ETL outputs feed into training
- [ ] Training outputs load into serving
- [ ] Serving logs feed into monitoring
- [ ] Monitoring triggers retraining

**2. DAG Orchestration**
- [ ] ETL DAG triggers training
- [ ] Monitoring DAG conditionally triggers retraining
- [ ] All DAGs handle failures gracefully

**3. Model Lifecycle**
- [ ] Models flow from training → registry → production
- [ ] Model versions tracked consistently
- [ ] Rollback mechanism works

**4. Configuration Management**
- [ ] All components use centralized config
- [ ] Config changes propagate correctly
- [ ] No hardcoded values

## Step 2: Master Orchestration DAG

**File**: `dags/master_orchestration_dag.py`

```python
"""
Master Orchestration DAG.

Coordinates all pipelines:
1. Daily ETL + Training
2. Monitoring
3. Conditional Retraining

This DAG doesn't run tasks itself; it triggers other DAGs in the right order.
"""

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta


default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_system_health(**context):
    """Check if all components are healthy before starting."""
    print("Checking system health...")

    # Check if required directories exist
    from pathlib import Path

    required_dirs = [
        'data/raw',
        'data/features',
        'models/training',
        'models/production'
    ]

    all_healthy = True
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            all_healthy = False
        else:
            print(f"✅ {dir_path}")

    if not all_healthy:
        raise ValueError("System health check failed")

    print("✅ System healthy!")


with DAG(
    'master_orchestration',
    default_args=default_args,
    description='Master orchestration for entire ML system',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=['orchestration', 'master'],
) as dag:

    # Health check
    health_check = PythonOperator(
        task_id='system_health_check',
        python_callable=check_system_health,
    )

    # Trigger ETL pipeline
    trigger_etl = TriggerDagRunOperator(
        task_id='trigger_etl_pipeline',
        trigger_dag_id='etl_and_feature_engineering',
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=['success'],
        failed_states=['failed']
    )

    # Wait for ETL to complete
    # (TriggerDagRunOperator with wait_for_completion handles this)

    # Trigger training pipeline
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_pipeline',
        trigger_dag_id='model_training',
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=['success'],
        failed_states=['failed']
    )

    # Trigger monitoring (runs in parallel with training completion)
    trigger_monitoring = TriggerDagRunOperator(
        task_id='trigger_monitoring_pipeline',
        trigger_dag_id='monitoring_and_drift_detection',
        wait_for_completion=False,  # Don't wait, let it run async
    )

    # Dependencies
    health_check >> trigger_etl >> trigger_training >> trigger_monitoring
```

## Step 3: End-to-End Testing

### Test Script

**File**: `tests/test_end_to_end.py`

```python
"""
End-to-end integration tests.

Tests the complete ML pipeline from data generation to serving.
"""

import pytest
import pandas as pd
import requests
import time
from pathlib import Path
import json
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture(scope="class")
    def setup_system(self):
        """Setup system for testing."""
        print("\n🚀 Setting up system for E2E test...")

        # Ensure directories exist
        directories = [
            'data/raw',
            'data/features',
            'data/predictions',
            'data/monitoring',
            'models/training',
            'models/production'
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        yield

        print("\n✅ E2E test complete")

    def test_01_data_generation(self, setup_system):
        """Test data generation."""
        from src.data.generator import FeedDataGenerator

        generator = FeedDataGenerator()
        interactions_df = generator.generate_interactions()

        assert len(interactions_df) > 0, "No interactions generated"
        assert 'user_id' in interactions_df.columns
        assert 'item_id' in interactions_df.columns
        assert 'click' in interactions_df.columns

        # Save for next tests
        interactions_df.to_csv("data/raw/interactions.csv", index=False)
        generator.users.to_csv("data/raw/users.csv", index=False)
        generator.items.to_csv("data/raw/items.csv", index=False)

        print(f"✅ Generated {len(interactions_df)} interactions")

    def test_02_data_validation(self, setup_system):
        """Test data validation."""
        from src.data.validator import DataValidator

        validator = DataValidator()
        results = validator.validate_interactions("data/raw/interactions.csv")

        assert results['failed'] == 0, f"Validation failed: {results['failures']}"

        print("✅ Data validation passed")

    def test_03_feature_engineering(self, setup_system):
        """Test feature engineering."""
        from src.data.features import FeatureEngineer

        # Load data
        interactions_df = pd.read_csv("data/raw/interactions.csv")
        users_df = pd.read_csv("data/raw/users.csv")
        items_df = pd.read_csv("data/raw/items.csv")

        # Engineer features
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(interactions_df, users_df, items_df)

        assert len(features_df) > 0, "No features generated"
        assert 'click' in features_df.columns

        # Create splits
        splits = engineer.create_splits(features_df)

        for split_name, split_df in splits.items():
            assert len(split_df) > 0, f"{split_name} split is empty"
            split_df.to_parquet(f"data/features/{split_name}_features.parquet", index=False)

        print("✅ Feature engineering complete")

    def test_04_model_training(self, setup_system):
        """Test model training."""
        import torch
        from torch.utils.data import DataLoader
        import yaml
        from src.models.ranker import create_model
        from src.models.trainer import RankingDataset, RankingTrainer

        # Load data
        train_df = pd.read_parquet("data/features/train_features.parquet")
        val_df = pd.read_parquet("data/features/val_features.parquet")

        # Load config
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        model_type = config['model']['type']

        # Create datasets
        train_dataset = RankingDataset(train_df, model_type=model_type)
        val_dataset = RankingDataset(val_df, model_type=model_type)

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        # Create model
        if model_type == 'two_tower':
            feature_dims = {
                'user': train_dataset.user_features.shape[1],
                'item': train_dataset.item_features.shape[1]
            }
        else:
            feature_dims = {'total': train_dataset.features.shape[1]}

        model = create_model(config, feature_dims)

        # Quick training (fewer epochs for testing)
        config['training']['epochs'] = 3
        trainer = RankingTrainer()

        # Write config temporarily
        with open("config/model_config_test.yaml", 'w') as f:
            yaml.dump(config, f)

        trainer.config = config
        history = trainer.train(model, train_loader, val_loader, experiment_name="e2e_test")

        # Check model was saved
        assert Path("models/training/best_model.pt").exists(), "Model not saved"

        print("✅ Model training complete")

    def test_05_model_evaluation(self, setup_system):
        """Test model evaluation."""
        import torch
        from torch.utils.data import DataLoader
        import yaml
        from src.models.ranker import create_model
        from src.models.trainer import RankingDataset
        from src.models.evaluator import RankingEvaluator

        # Load test data
        test_df = pd.read_parquet("data/features/test_features.parquet")

        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        model_type = config['model']['type']
        test_dataset = RankingDataset(test_df, model_type=model_type)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        # Load model
        if model_type == 'two_tower':
            feature_dims = {
                'user': test_dataset.user_features.shape[1],
                'item': test_dataset.item_features.shape[1]
            }
        else:
            feature_dims = {'total': test_dataset.features.shape[1]}

        model = create_model(config, feature_dims)
        model.load_state_dict(torch.load("models/training/best_model.pt"))

        # Evaluate
        evaluator = RankingEvaluator()
        metrics = evaluator.evaluate(model, test_loader)

        assert metrics['auc'] > 0.5, f"Model AUC too low: {metrics['auc']}"

        # Save metrics
        with open("models/training/test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Model evaluation complete - AUC: {metrics['auc']:.4f}")

    def test_06_model_serving(self, setup_system):
        """Test model serving API."""
        # Note: This requires the API to be running
        # For automated testing, you'd start the API in a subprocess

        # Skip if API not running
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            api_running = response.status_code == 200
        except:
            api_running = False

        if not api_running:
            pytest.skip("API not running - start with: python src/serving/app.py")

        # Test prediction
        request_data = {
            "user_id": 123,
            "item_ids": [1001, 1002, 1003, 1004, 1005],
            "context": {"device": "mobile"}
        }

        response = requests.post("http://localhost:8000/predict", json=request_data)

        assert response.status_code == 200, f"Prediction failed: {response.text}"

        data = response.json()
        assert data['user_id'] == 123
        assert len(data['predictions']) == 5
        assert data['predictions'][0]['rank'] == 1

        print(f"✅ Model serving works - Latency: {data['latency_ms']:.2f}ms")

    def test_07_monitoring(self, setup_system):
        """Test drift detection."""
        from src.monitoring.drift import DriftDetector

        # Load data
        reference_df = pd.read_parquet("data/features/train_features.parquet")
        current_df = pd.read_parquet("data/features/test_features.parquet")

        feature_cols = [
            'user_historical_ctr', 'user_avg_dwell_time',
            'item_ctr', 'item_popularity'
        ]

        # Detect drift
        detector = DriftDetector()
        drift_results = detector.detect_feature_drift(
            reference_df, current_df, feature_cols
        )

        assert 'summary' in drift_results
        assert 'features' in drift_results

        # Save results
        with open("data/monitoring/drift_report.json", 'w') as f:
            json.dump(drift_results, f, indent=2)

        print(f"✅ Monitoring works - Drift detected: {drift_results['summary']['drift_detected']}")

    def test_08_complete_flow(self, setup_system):
        """Test complete ML lifecycle flow."""
        print("\n📊 Testing Complete ML Lifecycle Flow...")

        # 1. Data exists
        assert Path("data/raw/interactions.csv").exists()

        # 2. Features exist
        assert Path("data/features/train_features.parquet").exists()

        # 3. Model trained
        assert Path("models/training/best_model.pt").exists()

        # 4. Metrics calculated
        assert Path("models/training/test_metrics.json").exists()

        # 5. Monitoring ran
        assert Path("data/monitoring/drift_report.json").exists()

        print("✅ Complete ML lifecycle verified!")


def run_e2e_test():
    """Run end-to-end test."""
    pytest.main([__file__, '-v', '-s'])


if __name__ == "__main__":
    run_e2e_test()
```

### Run Tests

```bash
cd capstone_project

# Install pytest if not already installed
pip install pytest

# Run E2E tests
python tests/test_end_to_end.py

# Or with pytest directly
pytest tests/test_end_to_end.py -v -s
```

## Step 4: System Documentation

### Complete README

**File**: `capstone_project/README.md`

```markdown
# Mini Feed Ranking System

Production-grade MLOps system demonstrating end-to-end ML lifecycle management.

## Overview

This system implements a complete feed ranking pipeline similar to what you'd find at
Meta, TikTok, or LinkedIn, including:

- **ETL Pipeline**: Data generation, validation, feature engineering
- **Model Training**: PyTorch neural ranking models with MLflow tracking
- **Model Serving**: FastAPI REST API for real-time predictions
- **Monitoring**: Feature drift and performance monitoring
- **Retraining**: Automated model retraining and promotion

## Architecture

```
Data → ETL → Training → Serving → Monitoring → Retraining → [Loop]
```

See [ARCHITECTURE.md](docs/architecture.md) for detailed design.

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB RAM
- 10GB disk space

### Setup

```bash
# Clone repo
git clone <repo-url>
cd capstone_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\\Scripts\\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Initialize Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \\
    --username admin \\
    --firstname Admin \\
    --lastname User \\
    --role Admin \\
    --email admin@example.com

# Start Airflow
airflow webserver -p 8080 &
airflow scheduler &
```

### Run Complete Pipeline

```bash
# 1. Generate data
python src/data/generator.py

# 2. Run ETL
airflow dags trigger etl_and_feature_engineering

# 3. Train model
airflow dags trigger model_training

# 4. Start serving API
python src/serving/app.py &

# 5. Test prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": 123,
    "item_ids": [1001, 1002, 1003]
  }'

# 6. Run monitoring
airflow dags trigger monitoring_and_drift_detection

# 7. (Optional) Trigger retraining
airflow dags trigger automated_retraining
```

## Project Structure

```
capstone_project/
├── config/              # Configuration files
├── dags/                # Airflow DAGs
├── src/                 # Source code
│   ├── data/           # Data processing
│   ├── models/         # ML models
│   ├── serving/        # API serving
│   └── monitoring/     # Monitoring & drift
├── tests/              # Tests
├── data/               # Data directory (gitignored)
├── models/             # Model artifacts (gitignored)
└── docs/               # Documentation
```

## Components

### ETL Pipeline (`dags/etl_dag.py`)
- Generates synthetic feed interaction data
- Validates data quality with Great Expectations
- Engineers ranking features
- Creates train/val/test splits

### Model Training (`dags/training_dag.py`)
- Trains PyTorch neural ranking models
- Logs experiments to MLflow
- Evaluates on test set
- Registers models to MLflow registry

### Model Serving (`src/serving/app.py`)
- FastAPI REST API
- `/predict` endpoint for ranking
- `/health` and `/metrics` endpoints
- Request validation and logging

### Monitoring (`dags/monitoring_dag.py`)
- Feature drift detection (KS test, PSI)
- Prediction drift monitoring
- Performance tracking
- Alert generation

### Retraining (`dags/retraining_dag.py`)
- Triggered by drift or schedule
- Compares new model vs production
- Safe model promotion with rollback
- Automated deployment

## Configuration

Edit `config/*.yaml` files to customize:

- `data_config.yaml`: Data generation parameters
- `model_config.yaml`: Model architecture and training
- `monitoring_config.yaml`: Drift thresholds and alerts

## Testing

```bash
# Unit tests
pytest tests/

# End-to-end test
python tests/test_end_to_end.py

# Load test
locust --host=http://localhost:8000
```

## Monitoring

- **Airflow UI**: http://localhost:8080
- **MLflow UI**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

## Troubleshooting

**Issue**: Airflow DAG not appearing
- Solution: Check DAG file for syntax errors, restart scheduler

**Issue**: Model training fails
- Solution: Check data exists, verify GPU/CPU availability

**Issue**: API returns 503
- Solution: Ensure model is trained and saved to `models/training/best_model.pt`

## Production Deployment

See [docs/deployment.md](docs/deployment.md) for production deployment guide.

## Contributing

[Add contribution guidelines if open-source]

## License

[Add license]

## Contact

[Your contact information]
```

## Step 5: Demo Script

**File**: `scripts/demo.sh`

```bash
#!/bin/bash

# Demo script showing complete ML lifecycle
# Run this to demonstrate the entire system

set -e  # Exit on error

cd /home/user/mlops-learning-plan/capstone_project

echo "=========================================="
echo "  MINI FEED RANKING SYSTEM DEMO"
echo "=========================================="
echo ""

echo "📊 Step 1: Generate Synthetic Data"
python src/data/generator.py
echo "✅ Data generated"
echo ""

echo "📊 Step 2: Validate Data Quality"
python src/data/validator.py
echo "✅ Data validated"
echo ""

echo "📊 Step 3: Engineer Features"
python src/data/features.py
echo "✅ Features engineered"
echo ""

echo "📊 Step 4: Train Ranking Model"
python src/models/trainer.py
echo "✅ Model trained"
echo ""

echo "📊 Step 5: Evaluate Model"
python src/models/evaluator.py
echo "✅ Model evaluated"
echo ""

echo "📊 Step 6: Start Model Serving API (in background)"
python src/serving/app.py &
API_PID=$!
sleep 5  # Wait for API to start
echo "✅ API started (PID: $API_PID)"
echo ""

echo "📊 Step 7: Test Prediction Endpoint"
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "item_ids": [1001, 1002, 1003, 1004, 1005],
    "context": {"device": "mobile"}
  }' | python -m json.tool
echo ""
echo "✅ Prediction successful"
echo ""

echo "📊 Step 8: Run Drift Detection"
python src/monitoring/drift.py
echo "✅ Drift detection complete"
echo ""

echo "📊 Step 9: Compare Models (for retraining demo)"
python src/models/comparator.py
echo "✅ Model comparison complete"
echo ""

echo "=========================================="
echo "  DEMO COMPLETE!"
echo "=========================================="
echo ""
echo "🎉 Successfully demonstrated:"
echo "  - Data generation and validation"
echo "  - Feature engineering"
echo "  - Model training and evaluation"
echo "  - Model serving via API"
echo "  - Drift detection"
echo "  - Model comparison"
echo ""
echo "📍 Outputs:"
echo "  - Data: data/raw/, data/features/"
echo "  - Models: models/training/"
echo "  - Monitoring: data/monitoring/"
echo ""
echo "🔗 Next Steps:"
echo "  - View API docs: http://localhost:8000/docs"
echo "  - View MLflow UI: mlflow ui"
echo "  - Run Airflow DAGs for full orchestration"
echo ""

# Kill API server
kill $API_PID
echo "🛑 Stopped API server"
```

Make it executable:
```bash
chmod +x scripts/demo.sh
```

## Step 6: Performance Benchmarking

**File**: `scripts/benchmark.py`

```python
"""
Performance benchmarking script.

Measures:
- ETL pipeline time
- Training time
- Serving latency
- Monitoring time
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path


def benchmark_pipeline():
    """Benchmark complete pipeline."""
    results = {}

    # 1. Data Generation
    print("Benchmarking data generation...")
    start = time.time()
    from src.data.generator import FeedDataGenerator
    generator = FeedDataGenerator()
    interactions_df = generator.generate_interactions()
    results['data_generation_seconds'] = time.time() - start

    # 2. Feature Engineering
    print("Benchmarking feature engineering...")
    users_df = pd.read_csv("data/raw/users.csv")
    items_df = pd.read_csv("data/raw/items.csv")

    start = time.time()
    from src.data.features import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(interactions_df, users_df, items_df)
    results['feature_engineering_seconds'] = time.time() - start

    # 3. Training Time (abbreviated for benchmark)
    print("Benchmarking training...")
    # Load existing metrics
    import json
    # ... (add training benchmark if needed)

    # 4. Serving Latency
    print("Benchmarking serving...")
    import requests

    try:
        latencies = []
        for _ in range(100):
            start = time.time()
            response = requests.post(
                "http://localhost:8000/predict",
                json={
                    "user_id": 123,
                    "item_ids": list(range(50))
                }
            )
            latencies.append((time.time() - start) * 1000)

        results['serving_p50_ms'] = np.percentile(latencies, 50)
        results['serving_p99_ms'] = np.percentile(latencies, 99)
    except:
        print("API not running, skipping serving benchmark")

    # Print results
    print("\n📊 Benchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}")

    # Save results
    with open("benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    benchmark_pipeline()
```

## Review Checklist

- [ ] All components integrated successfully
- [ ] Master orchestration DAG works
- [ ] End-to-end tests pass
- [ ] Complete documentation written
- [ ] Demo script runs successfully
- [ ] Performance benchmarked
- [ ] All outputs documented
- [ ] System ready for production

## What to Submit

### 1. Code Repository
- All source code
- Configuration files
- DAGs
- Tests
- Documentation

### 2. Demo Video (10-15 minutes)
Record yourself running the demo, showing:
1. Data generation and validation
2. Feature engineering
3. Model training (show MLflow)
4. Live predictions via API
5. Drift detection
6. Automated retraining
7. Model promotion

### 3. Final Report (5-10 pages)
Include:
- **Executive Summary**: What you built and why
- **Architecture**: System design diagram and explanation
- **Implementation**: Key technical decisions
- **Results**: Metrics, performance, benchmarks
- **Challenges**: What was difficult, how you solved it
- **Production Plan**: How you'd deploy to production
- **Future Work**: What you'd improve/add

### 4. Metrics Summary
- Model Performance: AUC, log loss, precision@K
- System Performance: Latency, throughput
- Data Quality: Validation pass rate
- Drift Detection: Features with drift

## Common Issues & Solutions

**Issue 1**: DAGs don't trigger each other
- Check trigger task IDs match DAG IDs
- Verify Airflow scheduler is running
- Check DAG dependencies in UI

**Issue 2**: Model promotion fails
- Ensure comparison results exist
- Check file permissions
- Verify model paths

**Issue 3**: API can't load model
- Check model file exists
- Verify feature dimensions match
- Check device (CPU vs GPU)

## Production Deployment Considerations

### Before Production

- [ ] Security: Add API authentication
- [ ] Logging: Centralize logs (ELK/Cloud Logging)
- [ ] Monitoring: Set up Prometheus + Grafana
- [ ] Scaling: Containerize with Docker
- [ ] CI/CD: Set up GitHub Actions
- [ ] Backup: Implement model/data backups
- [ ] Documentation: API docs, runbooks, on-call guides

### Recommended Production Stack

```
- Data: S3/GCS + Spark
- Features: Feast feature store
- Orchestration: Airflow on Kubernetes
- Training: GPU instances (on-demand)
- Serving: FastAPI on K8s (auto-scaling)
- Monitoring: Prometheus + Grafana
- Logging: ELK Stack
- Registry: MLflow on dedicated server
```

## Scaling Plan

**Current**: 1M interactions, 10K users
**Target**: 1B interactions, 1M users

| Component | Current | Target | Changes Needed |
|-----------|---------|--------|----------------|
| Data | CSV | Parquet/Delta | Distributed storage |
| Processing | Pandas | Spark | Distributed compute |
| Training | CPU | GPU | Cloud GPUs |
| Serving | 1 instance | 100+ instances | K8s auto-scaling |
| Monitoring | Batch | Streaming | Kafka + Flink |

## Congratulations!

You've built a complete, production-grade MLOps system! 🎉

You now understand:
- ✅ End-to-end ML pipelines
- ✅ Production ML system design
- ✅ MLOps best practices
- ✅ Automation and orchestration
- ✅ Monitoring and maintenance
- ✅ Model lifecycle management

### Next Steps

1. **Add to Portfolio**: Showcase this project to employers
2. **Open Source**: Publish on GitHub
3. **Extend**: Add more features (e.g., real feature store, true A/B testing)
4. **Deploy**: Try deploying to cloud (AWS/GCP/Azure)
5. **Teach**: Write blog posts about what you learned

### Resources for Continued Learning

- [Google's MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Chip Huyen's ML Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Community](https://mlops.community/)

---

**You did it!** This capstone represents the culmination of your MLOps learning journey. Well done! 🚀
