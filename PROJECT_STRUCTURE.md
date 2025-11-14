# MLOps Learning Project - Directory Structure

This document outlines the complete project structure you'll build throughout the course.

## Final Project Structure (End of Phase 4)

```
mlops-learning-plan/
│
├── README.md                          # Project overview and setup
├── CURRICULUM.md                      # Complete learning path
├── PROJECT_STRUCTURE.md               # This file
│
├── setup/                             # Environment setup scripts
│   ├── requirements.txt               # Python dependencies
│   ├── install_airflow.sh            # Airflow setup script
│   └── setup_dev_env.sh              # Complete dev environment setup
│
├── dags/                              # Airflow DAG definitions
│   ├── __init__.py
│   ├── etl_pipeline.py               # Phase 2: Data ingestion and processing
│   ├── feature_engineering_pipeline.py # Phase 2: Feature creation
│   ├── training_pipeline.py          # Phase 3: Model training orchestration
│   ├── batch_inference_pipeline.py   # Phase 4: Batch scoring
│   ├── monitoring_pipeline.py        # Phase 4: Model monitoring
│   └── retraining_pipeline.py        # Phase 4: Automated retraining
│
├── ml/                                # ML code (PyTorch models and training)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py               # PyTorch Dataset classes
│   │   ├── loaders.py                # DataLoader utilities
│   │   └── preprocessing.py          # Data preprocessing functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tabular.py                # Tabular classification/regression models
│   │   ├── ranking.py                # Two-tower and ranking models
│   │   └── base.py                   # Base model class
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                  # Training loop
│   │   ├── evaluate.py               # Evaluation logic
│   │   └── utils.py                  # Training utilities
│   └── registry/
│       ├── __init__.py
│       └── model_registry.py         # Model versioning and registration
│
├── serving/                           # Model serving layer
│   ├── __init__.py
│   ├── api.py                        # FastAPI serving application
│   ├── model_loader.py               # Load models for inference
│   ├── schemas.py                    # Request/response schemas
│   └── client.py                     # Example client
│
├── monitoring/                        # Monitoring and observability
│   ├── __init__.py
│   ├── drift_detection.py            # Feature and prediction drift
│   ├── metrics.py                    # Monitoring metrics calculation
│   ├── alerts.py                     # Alerting logic
│   └── reports.py                    # Reporting utilities
│
├── etl/                               # ETL and data processing utilities
│   ├── __init__.py
│   ├── ingestion.py                  # Data ingestion functions
│   ├── validation.py                 # Data validation
│   ├── feature_engineering.py        # Feature creation logic
│   └── transforms.py                 # Data transformations
│
├── config/                            # Configuration files
│   ├── data_config.yaml              # Data pipeline configs
│   ├── model_config.yaml             # Model hyperparameters
│   ├── training_config.yaml          # Training configs
│   └── serving_config.yaml           # Serving configs
│
├── experiments/                       # Experiment tracking
│   ├── runs/                         # Individual experiment runs
│   │   └── {run_id}/
│   │       ├── config.yaml           # Run configuration
│   │       ├── metrics.json          # Training metrics
│   │       └── artifacts/            # Model checkpoints, plots
│   └── experiments.db                # Experiment metadata (if using MLflow)
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                          # Raw ingested data
│   │   └── {date}/                   # Date-partitioned raw data
│   ├── processed/                    # Cleaned and validated data
│   │   └── {date}/
│   ├── features/                     # Engineered features
│   │   └── {version}/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── predictions/                  # Model predictions
│   │   └── {date}/
│   └── monitoring/                   # Monitoring data
│       └── {date}/
│
├── models/                            # Model artifacts (gitignored)
│   ├── staging/                      # Models in staging
│   │   └── {model_name}/
│   │       └── {version}/
│   │           ├── model.pt          # PyTorch model weights
│   │           ├── config.yaml       # Model config
│   │           └── metrics.json      # Evaluation metrics
│   └── production/                   # Production models
│       └── {model_name}/
│           ├── current/              # Symlink to current version
│           └── {version}/
│
├── logs/                              # Application logs (gitignored)
│   ├── airflow/                      # Airflow logs
│   ├── training/                     # Training logs
│   └── serving/                      # Serving logs
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_dags/                    # DAG tests
│   ├── test_ml/                      # ML code tests
│   ├── test_serving/                 # Serving tests
│   └── test_monitoring/              # Monitoring tests
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_monitoring_analysis.ipynb
│
├── scripts/                           # Utility scripts
│   ├── download_data.py              # Download sample datasets
│   ├── reset_environment.sh          # Reset for fresh start
│   ├── run_training.py               # Standalone training script
│   └── run_inference.py              # Standalone inference script
│
└── docs/                              # Documentation
    ├── phase1/                       # Phase 1 materials and labs
    ├── phase2/                       # Phase 2 materials and labs
    ├── phase3/                       # Phase 3 materials and labs
    ├── phase4/                       # Phase 4 materials and labs
    └── architecture/                 # System architecture docs
```

## Phase-by-Phase Evolution

### After Phase 1
```
mlops-learning-plan/
├── setup/
│   └── requirements.txt
├── dags/
│   └── hello_airflow.py              # Simple DAG
├── ml/
│   └── mnist_train.py                # Simple PyTorch script
└── docs/
    └── phase1/
```

### After Phase 2
```
mlops-learning-plan/
├── dags/
│   ├── etl_pipeline.py
│   └── feature_engineering_pipeline.py
├── etl/
│   ├── ingestion.py
│   ├── validation.py
│   └── feature_engineering.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
└── config/
    └── data_config.yaml
```

### After Phase 3
```
mlops-learning-plan/
├── dags/
│   ├── etl_pipeline.py
│   └── training_pipeline.py
├── ml/
│   ├── data/
│   ├── models/
│   └── training/
├── experiments/
│   └── runs/
├── models/
│   └── staging/
└── config/
    ├── data_config.yaml
    └── model_config.yaml
```

### After Phase 4 (Complete)
Full structure as shown above.

---

## Key Principles

### 1. Separation of Concerns
- **dags/**: Orchestration logic only (thin layer)
- **ml/**: ML-specific code (PyTorch models, training)
- **etl/**: Data processing logic
- **serving/**: Inference layer
- **monitoring/**: Observability and monitoring

### 2. Configuration-Driven
- All configs in `config/` directory
- YAML for human-readable configs
- Environment-specific overrides

### 3. Versioning
- Data versioned by date or explicit version
- Models versioned (semantic versioning)
- Features versioned
- Predictions timestamped

### 4. Data Partitioning
- Raw data partitioned by ingestion date
- Features partitioned by version
- Predictions partitioned by prediction date
- Enables incremental processing and backfills

### 5. Model Registry
- Staging vs production separation
- Versioned model artifacts
- Metadata (config, metrics) alongside weights
- Clear promotion path

### 6. Experiment Tracking
- Each run gets unique ID
- Configs, metrics, and artifacts stored together
- Reproducibility through config snapshots
- Easy comparison across runs

---

## .gitignore Recommendations

```gitignore
# Data (usually too large, regenerable)
data/
!data/.gitkeep

# Models (large binary files)
models/
!models/.gitkeep

# Experiments (can be many)
experiments/runs/
!experiments/runs/.gitkeep

# Logs
logs/
*.log

# Airflow
airflow.db
airflow.cfg
airflow-webserver.pid
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.env.local
```

---

## Next Steps

This structure will be built incrementally as you progress through the phases. Don't worry about creating everything upfront—we'll create each component as needed during the labs.

**Start with Phase 1**, and we'll build this step by step!
