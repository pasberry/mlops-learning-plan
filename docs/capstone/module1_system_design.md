# Module 1: System Design & Planning

**Estimated Time**: 1-2 days
**Difficulty**: Medium

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Define clear requirements and success criteria for an ML system
- ✅ Design a complete system architecture with proper component separation
- ✅ Create data models and schemas for a ranking system
- ✅ Design APIs for model serving
- ✅ Plan DAG dependencies and orchestration logic
- ✅ Document architectural decisions and tradeoffs

## Overview

Before writing any code, we need to design the system. This module focuses on the **planning phase** of the ML lifecycle, where we define what we're building, why, and how all the pieces fit together.

**What you'll create**:
- System design document
- Data schemas
- API specifications
- DAG dependency plan
- Technology stack decisions
- Project structure

## Background

### Why Design First?

In production ML systems, **design errors are expensive**. Consider:
- Changing data schemas after collecting millions of records
- Refactoring DAGs that are already running in production
- Redesigning APIs that have external consumers

Good design upfront prevents:
- Technical debt accumulation
- Rework and refactoring
- Integration issues between components
- Performance bottlenecks

### Feed Ranking Systems in Production

Real-world examples:
- **Instagram/Facebook**: Rank posts, stories, reels for user's feed
- **TikTok**: Rank videos for "For You" page
- **LinkedIn**: Rank posts, jobs, connections
- **Twitter/X**: Rank tweets in timeline
- **YouTube**: Rank videos for recommendations

**Common Pattern**:
1. User makes request → System retrieves candidate items (100-1000s)
2. System scores each item with ML model
3. Items ranked by score
4. Top N items returned to user

**Key Requirements**:
- **Low latency**: Users expect instant results (<200ms)
- **High accuracy**: Better predictions = better user experience = more engagement
- **Freshness**: Models must adapt to changing user preferences
- **Scale**: Handle millions of users and billions of items

## Step 1: Define Requirements

### Functional Requirements

**Core Functionality**:
```
1. Data Processing
   - Generate/ingest user-item interaction data
   - Validate data quality
   - Engineer features from raw data
   - Split data for training/validation/testing

2. Model Training
   - Train neural ranking model
   - Track experiments
   - Evaluate model performance
   - Version and register models

3. Model Serving
   - Load production model
   - Serve predictions via REST API
   - Log all predictions
   - Monitor latency and throughput

4. Monitoring
   - Detect feature drift
   - Detect prediction drift
   - Track model performance
   - Generate alerts

5. Retraining
   - Automatically retrain on drift/schedule
   - Compare new model vs production
   - Promote better models
   - Handle rollback if needed
```

### Non-Functional Requirements

**Performance**:
- Training: Complete in <30 minutes on CPU
- Serving: <100ms p99 latency for batch of 100 items
- Throughput: Support >100 QPS on single instance

**Reliability**:
- Data validation: Catch 95%+ of data quality issues
- Model evaluation: Proper train/val/test splits
- Monitoring: Detect drift within 24 hours

**Scalability**:
- Handle 10K users, 50K items initially
- Design to scale to 100K users, 500K items

**Maintainability**:
- Configuration-driven (no hardcoded values)
- Modular code (easy to test/modify)
- Comprehensive logging
- Clear documentation

### Success Metrics

**Model Performance**:
- AUC > 0.70 on test set
- Log loss < 0.5
- Precision@10 > 0.15

**System Performance**:
- ETL pipeline completes in <10 minutes
- Training completes in <30 minutes
- API latency <100ms p99
- Drift detection accuracy >90%

**Business Metrics** (simulated):
- CTR (click-through rate)
- Engagement rate (likes, shares)
- Session length
- User retention

## Step 2: Design System Architecture

### High-Level Architecture

Create a document describing your system's components:

```
Components:
1. Data Layer
   - Raw data storage
   - Feature store (simulated with Parquet files)
   - Model artifacts storage

2. Processing Layer
   - ETL pipeline (Airflow)
   - Feature engineering
   - Data validation

3. Training Layer
   - PyTorch training scripts
   - MLflow experiment tracking
   - Model registry

4. Serving Layer
   - FastAPI application
   - Model loader
   - Prediction logger

5. Monitoring Layer
   - Drift detection
   - Performance tracking
   - Alerting

6. Orchestration Layer
   - Airflow DAGs
   - DAG dependencies
   - Scheduler
```

### Component Interaction

Document how components interact:

```
Data Flow:
Raw Data → [ETL DAG] → Features → [Training DAG] → Model → [Serving] → Predictions
                                                                          ↓
                                                                   [Monitoring DAG]
                                                                          ↓
                                                                   [Retraining DAG]
```

**Questions to Answer**:
1. How does training DAG know ETL completed?
2. How does serving load the latest model?
3. How does monitoring trigger retraining?
4. What happens if a component fails?

## Step 3: Design Data Models

### Raw Interaction Data Schema

```python
# data/raw/interactions.csv
{
    'interaction_id': str,        # Unique ID (UUID)
    'user_id': int,               # 0-9999
    'item_id': int,               # 0-49999
    'timestamp': str,             # ISO 8601 format
    'click': int,                 # 0 or 1
    'like': int,                  # 0 or 1
    'share': int,                 # 0 or 1
    'dwell_time': float,          # Seconds (0-300)
    'device': str,                # 'mobile', 'desktop', 'tablet'
    'time_of_day': str,           # 'morning', 'afternoon', 'evening', 'night'
    'day_of_week': int            # 0-6 (Monday-Sunday)
}
```

### Feature Schema

```python
# data/features/train_features.parquet
{
    # Target
    'target': int,                # 0 or 1 (click)

    # User features
    'user_id': int,
    'user_historical_ctr': float,
    'user_avg_dwell_time': float,
    'user_interaction_count': int,
    'user_days_active': int,

    # Item features
    'item_id': int,
    'item_popularity': float,
    'item_ctr': float,
    'item_avg_dwell_time': float,
    'item_age_days': int,

    # Interaction features
    'hour_of_day': int,
    'day_of_week': int,
    'device_mobile': int,         # One-hot encoded
    'device_desktop': int,
    'device_tablet': int,

    # User-item cross features
    'user_item_previous_interactions': int,
    'recency_score': float
}
```

### Model Artifact Schema

```python
# models/production/model_metadata.json
{
    'model_id': str,              # UUID
    'model_version': str,         # e.g., "v1.2.3"
    'training_date': str,         # ISO 8601
    'mlflow_run_id': str,
    'model_type': str,            # 'two_tower' or 'deep_mlp'
    'metrics': {
        'auc': float,
        'log_loss': float,
        'precision_at_10': float
    },
    'feature_schema_version': str,
    'training_data_date_range': {
        'start': str,
        'end': str
    }
}
```

### Prediction Log Schema

```python
# data/predictions/predictions_<date>.csv
{
    'prediction_id': str,
    'timestamp': str,
    'user_id': int,
    'item_id': int,
    'score': float,
    'rank': int,
    'model_version': str,
    'latency_ms': float,
    'features': dict             # JSON blob of features used
}
```

## Step 4: Design APIs

### Model Serving API

**Endpoint 1: Predict**
```python
POST /predict
Request:
{
    "user_id": 123,
    "item_ids": [1001, 1002, 1003, ...],  # up to 100 items
    "context": {
        "device": "mobile",
        "timestamp": "2024-11-15T10:30:00Z"
    }
}

Response:
{
    "user_id": 123,
    "predictions": [
        {"item_id": 1002, "score": 0.87, "rank": 1},
        {"item_id": 1001, "score": 0.72, "rank": 2},
        {"item_id": 1003, "score": 0.45, "rank": 3}
    ],
    "model_version": "v1.2.3",
    "latency_ms": 45.2
}
```

**Endpoint 2: Health Check**
```python
GET /health
Response:
{
    "status": "healthy",
    "model_version": "v1.2.3",
    "model_loaded_at": "2024-11-15T08:00:00Z",
    "uptime_seconds": 3600,
    "last_prediction": "2024-11-15T10:30:00Z"
}
```

**Endpoint 3: Metrics**
```python
GET /metrics
Response:
{
    "requests_total": 10000,
    "requests_per_second": 25.3,
    "latency_p50_ms": 32.1,
    "latency_p99_ms": 87.4,
    "error_rate": 0.001,
    "predictions_total": 250000
}
```

### Configuration API (Internal)

```python
GET /config
Response:
{
    "model_config": {
        "type": "two_tower",
        "embedding_dim": 128
    },
    "feature_config": {
        "version": "v2",
        "features": ["user_ctr", "item_popularity", ...]
    }
}
```

## Step 5: Plan DAG Dependencies

### DAG 1: ETL Pipeline
```python
Name: etl_and_feature_engineering
Schedule: Daily at 2:00 AM
Tasks:
  1. generate_data (or ingest_data)
  2. validate_data
  3. compute_features
  4. create_train_val_test_splits
  5. save_features
Dependencies: None
Triggers: training_dag (on success)
```

### DAG 2: Model Training
```python
Name: model_training
Schedule: Triggered by ETL success
Tasks:
  1. load_features
  2. train_model
  3. evaluate_model
  4. log_to_mlflow
  5. register_model
Dependencies: etl_and_feature_engineering
Triggers: None (manual deployment initially)
```

### DAG 3: Monitoring
```python
Name: monitoring_and_drift_detection
Schedule: Daily at 4:00 AM
Tasks:
  1. load_reference_data
  2. load_recent_predictions
  3. compute_feature_drift
  4. compute_prediction_drift
  5. evaluate_online_performance
  6. generate_alerts
Dependencies: etl_and_feature_engineering (needs recent data)
Triggers: retraining_dag (if drift detected)
```

### DAG 4: Retraining
```python
Name: automated_retraining
Schedule: Weekly (Sunday midnight) OR triggered by alerts
Tasks:
  1. check_trigger_conditions
  2. fetch_recent_data
  3. trigger_etl_dag
  4. trigger_training_dag
  5. evaluate_new_model
  6. compare_with_production
  7. promote_if_better
Dependencies: monitoring_and_drift_detection
Triggers: training_dag
```

### Dependency Graph
```
   [ETL DAG]
       ↓
   ┌───┴───────┬─────────┐
   ↓           ↓         ↓
[Training] [Monitoring] [Serving]
   ↓           ↓
   └───────┬───┘
           ↓
    [Retraining]
           ↓
    [Auto-promote]
```

## Step 6: Technology Stack Decisions

### Core Technologies

| Component | Technology | Why? |
|-----------|-----------|------|
| **Orchestration** | Apache Airflow | Industry standard, great for complex DAGs |
| **ML Framework** | PyTorch | Flexible, great for research & production |
| **Experiment Tracking** | MLflow | Open-source, integrates well with PyTorch |
| **Data Validation** | Great Expectations | Comprehensive data quality checks |
| **API Framework** | FastAPI | Fast, modern, automatic docs |
| **Data Processing** | Pandas | Simple, sufficient for our scale |
| **Storage** | Parquet files | Efficient, schema-preserving |

### Alternative Considerations

**If scaling to production**:
- Data Processing: Spark (for massive datasets)
- Feature Store: Feast, Tecton
- Model Serving: TorchServe, TensorFlow Serving
- Monitoring: Prometheus, Grafana, Evidently

**Tradeoffs**:
- Simpler stack = easier to learn and debug
- Production stack = better performance and features
- **Our choice**: Start simple, design for future migration

## Step 7: Create Project Structure

```bash
capstone_project/
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
│
├── config/                            # Configuration files
│   ├── data_config.yaml               # Data generation/processing
│   ├── model_config.yaml              # Model architecture/training
│   └── monitoring_config.yaml         # Drift thresholds
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py               # Synthetic data generation
│   │   ├── validator.py               # Great Expectations validation
│   │   └── features.py                # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ranker.py                  # PyTorch model definitions
│   │   └── trainer.py                 # Training loop
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application
│   │   └── schemas.py                 # Pydantic models
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py                   # Drift detection
│       └── metrics.py                 # Performance metrics
│
├── dags/                              # Airflow DAGs
│   ├── etl_dag.py
│   ├── training_dag.py
│   ├── monitoring_dag.py
│   └── retraining_dag.py
│
├── tests/                             # Unit tests
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_drift.py
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_results_analysis.ipynb
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Raw interaction data
│   ├── processed/                     # Cleaned data
│   ├── features/                      # Feature sets
│   └── predictions/                   # Prediction logs
│
├── models/                            # Model artifacts (gitignored)
│   ├── training/                      # Training checkpoints
│   └── production/                    # Production models
│
├── mlruns/                            # MLflow tracking (gitignored)
│
└── docs/                              # Documentation
    ├── architecture.md
    ├── api_docs.md
    └── runbook.md
```

## Step 8: Write System Design Document

Create `capstone_project/docs/system_design.md`:

### Suggested Sections

1. **Executive Summary**
   - What we're building
   - Why it matters
   - Key technical decisions

2. **Requirements**
   - Functional requirements
   - Non-functional requirements
   - Success criteria

3. **Architecture**
   - Component diagram
   - Data flow
   - Technology stack

4. **Data Models**
   - Schemas for all data types
   - Versioning strategy

5. **APIs**
   - Endpoint specifications
   - Request/response formats

6. **Orchestration**
   - DAG descriptions
   - Dependencies
   - Scheduling

7. **Monitoring & Alerting**
   - Metrics to track
   - Alert conditions
   - Response procedures

8. **Security & Privacy** (future)
   - Data anonymization
   - API authentication
   - Model security

9. **Scaling Plan** (future)
   - Bottlenecks
   - Migration path to production stack

10. **Open Questions**
    - Risks and mitigations
    - Future improvements

## Implementation Guide

### Step-by-Step

1. **Create project directory**:
```bash
cd /home/user/mlops-learning-plan
mkdir -p capstone_project
cd capstone_project
```

2. **Create directory structure**:
```bash
mkdir -p {config,src/{data,models,serving,monitoring},dags,tests,notebooks,data/{raw,processed,features,predictions},models/{training,production},docs}
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/serving/__init__.py src/monitoring/__init__.py
```

3. **Create initial README.md**:
```markdown
# Mini Feed Ranking System

Production-grade MLOps capstone project demonstrating end-to-end ML lifecycle.

## Quick Start
(To be filled in as you build)

## Architecture
See docs/architecture.md

## Components
- ETL Pipeline
- Model Training
- Model Serving
- Monitoring
- Retraining

## Setup
(To be filled in)
```

4. **Create requirements.txt**:
```txt
# Core
apache-airflow==2.7.0
torch==2.0.1
mlflow==2.7.1
great-expectations==0.18.0
fastapi==0.104.1
uvicorn==0.24.0

# Data processing
pandas==2.1.1
numpy==1.24.3
pyarrow==13.0.0
scikit-learn==1.3.1

# Utilities
pyyaml==6.0.1
pydantic==2.4.2
python-dotenv==1.0.0
requests==2.31.0

# Monitoring
scipy==1.11.3

# Development
pytest==7.4.2
black==23.9.1
flake8==6.1.0
jupyter==1.0.0
```

5. **Create base configuration files**:

**config/data_config.yaml**:
```yaml
data_generation:
  num_users: 10000
  num_items: 50000
  num_interactions: 1000000
  date_range:
    start: "2024-01-01"
    end: "2024-11-15"

feature_engineering:
  user_features:
    - user_historical_ctr
    - user_avg_dwell_time
    - user_interaction_count
  item_features:
    - item_popularity
    - item_ctr
    - item_age_days
  interaction_features:
    - hour_of_day
    - day_of_week
    - device

split:
  train: 0.7
  val: 0.15
  test: 0.15
  method: "temporal"  # temporal split, not random
```

**config/model_config.yaml**:
```yaml
model:
  type: "two_tower"  # or "deep_mlp"
  user_tower_dims: [256, 128]
  item_tower_dims: [256, 128]
  embedding_dim: 128
  dropout: 0.3

training:
  batch_size: 1024
  learning_rate: 0.001
  epochs: 20
  optimizer: "adam"
  early_stopping:
    patience: 3
    min_delta: 0.001

evaluation:
  metrics:
    - "auc"
    - "log_loss"
    - "precision_at_10"
```

**config/monitoring_config.yaml**:
```yaml
drift_detection:
  feature_drift:
    method: "ks_test"
    threshold: 0.1
  prediction_drift:
    method: "psi"
    threshold: 0.2

performance_monitoring:
  auc_degradation_threshold: 0.05  # Alert if drops >5%
  latency_p99_threshold_ms: 100

alerting:
  email: "your-email@example.com"  # (simulated)
  slack_webhook: null

retraining_triggers:
  feature_drift: true
  performance_degradation: true
  scheduled: "weekly"
```

## Deliverable: System Design Document

### What to Include

Your system design document should answer:

1. **What are we building?**
   - High-level description
   - Key features
   - User journey

2. **Why these design choices?**
   - Technology selection rationale
   - Architectural patterns
   - Tradeoffs made

3. **How do components interact?**
   - Clear diagrams
   - Data flow descriptions
   - API contracts

4. **What are the risks?**
   - Technical risks
   - Mitigation strategies
   - Open questions

5. **How will we measure success?**
   - Model metrics
   - System metrics
   - Business metrics

### Example Outline

```markdown
# Mini Feed Ranking System - System Design

## 1. Overview
[Brief description]

## 2. Requirements
### Functional
- [List]
### Non-Functional
- [List]
### Success Metrics
- [List]

## 3. Architecture
### High-Level Design
[Diagram]
### Component Details
[Descriptions]

## 4. Data Models
### Raw Data
[Schema]
### Features
[Schema]
### Models
[Schema]

## 5. APIs
### Prediction API
[Spec]

## 6. Orchestration
### DAG Dependencies
[Diagram]

## 7. Monitoring
[Strategy]

## 8. Risks & Mitigations
[List]

## 9. Future Work
[Ideas]
```

## Review Checklist

Before moving to Module 2, ensure you have:

- [ ] Clearly defined requirements (functional + non-functional)
- [ ] Complete system architecture diagram
- [ ] All data schemas documented
- [ ] API specifications written
- [ ] DAG dependencies planned
- [ ] Technology stack decisions made and justified
- [ ] Project directory structure created
- [ ] Configuration files created
- [ ] System design document written
- [ ] Risks and mitigations identified

## What to Submit

Create a document (Markdown preferred) containing:

1. **System Design Document** (see template above)
2. **Project Structure** (screenshot of directory tree)
3. **Configuration Files** (YAML files created)
4. **Architectural Diagrams** (ASCII art or actual diagrams)
5. **Reflection**:
   - What was most challenging in the design phase?
   - What design decisions are you uncertain about?
   - What would you do differently in a real production system?

## Common Pitfalls

🚫 **Don't**:
- Skip requirements definition
- Design components in isolation (consider integration!)
- Hardcode values that should be configurable
- Over-engineer for scale you don't need yet
- Under-engineer for scale you'll need soon

✅ **Do**:
- Think about failure modes
- Consider operational complexity
- Document design decisions and rationale
- Plan for monitoring from day 1
- Keep it simple, but make it right

## Next Steps

Once you've completed your system design:

1. Review your design document with the checklist
2. Get feedback (from peers, mentors, or self-review)
3. Revise based on feedback
4. Proceed to [Module 2: ETL & Feature Engineering](module2_data_pipeline.md)

---

**Time Check**: Have you spent 1-2 days on design? If you're moving faster, that's fine, but make sure you've thought through the architecture carefully. Design mistakes are expensive to fix later!

**Ready to start building?** Move on to [Module 2: ETL & Feature Engineering DAG](module2_data_pipeline.md)
