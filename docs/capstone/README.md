# MLOps Capstone Project: Mini Feed Ranking System

## Project Overview

Welcome to the capstone project for the MLOps course! You will build a **production-grade feed ranking system** from scratch, simulating the type of ML systems used at companies like Meta, TikTok, Twitter, and LinkedIn.

This project integrates everything you've learned across all four phases:
- **Phase 1**: Airflow orchestration, PyTorch fundamentals, experiment tracking
- **Phase 2**: ETL pipelines, feature engineering, data quality
- **Phase 3**: Model training, two-tower architectures, MLflow integration
- **Phase 4**: Model serving, monitoring, retraining automation

### What is a Feed Ranking System?

A feed ranking system predicts which content (posts, videos, ads) a user is most likely to engage with, then ranks items accordingly. Every time you scroll through Instagram, TikTok, or LinkedIn, a ranking model is scoring hundreds of items in real-time to show you the most relevant content first.

**Your Task**: Build an end-to-end ML system that:
1. Generates synthetic user-item interaction data
2. Engineers features from raw interactions
3. Trains a neural ranking model
4. Serves predictions via REST API
5. Monitors for data drift and model degradation
6. Automatically retrains when performance degrades

## Learning Objectives

By completing this capstone, you will demonstrate mastery of:

✅ **System Design**: Architecting multi-component ML systems with proper data flow and dependencies

✅ **Data Engineering**: Building production ETL pipelines with validation, versioning, and monitoring

✅ **ML Engineering**: Training, evaluating, and versioning ranking models with experiment tracking

✅ **MLOps**: Orchestrating complex workflows with Airflow DAGs and managing DAG dependencies

✅ **Model Serving**: Deploying models via REST APIs with proper logging and health checks

✅ **Monitoring**: Detecting feature drift, prediction drift, and model degradation in production

✅ **Automation**: Building closed-loop systems that retrain and promote models automatically

✅ **Production Best Practices**: Code quality, testing, documentation, and reproducibility

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MINI FEED RANKING SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │   Raw Interaction Data   │
                    │  (user_id, item_id,      │
                    │   timestamp, click, etc) │
                    └──────────┬───────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │         ETL & FEATURE ENGINEERING DAG        │
        │  - Data generation/ingestion                 │
        │  - Data validation (Great Expectations)      │
        │  - Feature engineering (embeddings, CTR)     │
        │  - Train/val/test splits                     │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │           MODEL TRAINING DAG                 │
        │  - Load features                             │
        │  - Train PyTorch ranking model               │
        │  - Evaluate (AUC, log loss, NDCG)            │
        │  - Log to MLflow                             │
        │  - Register model                            │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │          FASTAPI MODEL SERVER                │
        │  - Load production model                     │
        │  - /predict endpoint                         │
        │  - Request/response logging                  │
        │  - Health checks                             │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │          MONITORING DAG                      │
        │  - Feature drift detection (KS, PSI)         │
        │  - Prediction drift monitoring               │
        │  - Model performance tracking                │
        │  - Alert generation                          │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │          RETRAINING DAG                      │
        │  - Triggered by alerts or schedule           │
        │  - Train new model on recent data            │
        │  - A/B test vs production model              │
        │  - Auto-promote if better                    │
        └──────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow | DAG scheduling and dependency management |
| **Data Processing** | Pandas, NumPy | Feature engineering and data manipulation |
| **Data Validation** | Great Expectations | Schema validation and data quality checks |
| **ML Framework** | PyTorch | Neural network training and inference |
| **Experiment Tracking** | MLflow | Model versioning, metrics, and registry |
| **Model Serving** | FastAPI | REST API for real-time predictions |
| **Monitoring** | Custom + MLflow | Drift detection and alerting |
| **Storage** | Local filesystem | Data lake simulation (CSV/Parquet) |
| **Configuration** | YAML | Declarative config management |

## Project Modules

The capstone is divided into 7 modules, each building on the previous:

### Module 1: System Design & Planning (1-2 days)
- Define requirements and success metrics
- Design system architecture
- Plan data models and APIs
- Create project structure
- **Deliverable**: System design document

### Module 2: ETL & Feature Engineering DAG (2-3 days)
- Generate synthetic feed interaction data
- Build data validation pipeline
- Engineer ranking features (user/item embeddings, engagement signals)
- Create train/val/test splits
- **Deliverable**: Working ETL DAG that produces feature sets

### Module 3: Model Training DAG (2-3 days)
- Implement PyTorch ranking model (two-tower or deep MLP)
- Build training DAG with MLflow integration
- Implement evaluation metrics (AUC, log loss, NDCG)
- Register models to MLflow
- **Deliverable**: Automated training pipeline

### Module 4: Model Serving (2 days)
- Build FastAPI application
- Implement /predict endpoint with request validation
- Add prediction logging
- Deploy locally with health checks
- **Deliverable**: Production-ready API server

### Module 5: Monitoring & Drift Detection (2-3 days)
- Implement feature drift detection (KS test, PSI)
- Build prediction drift monitoring
- Create monitoring DAG
- Set up alerting logic
- **Deliverable**: Automated monitoring system

### Module 6: Retraining & Model Promotion (2-3 days)
- Build retraining DAG triggered by drift/schedule
- Implement model comparison logic
- Create auto-promotion pipeline
- Add A/B testing concepts
- **Deliverable**: Closed-loop retraining system

### Module 7: Final Integration & Testing (2-3 days)
- End-to-end system testing
- Master orchestration DAG
- Performance optimization
- Documentation and demo
- **Deliverable**: Complete production-ready system

## Getting Started

### Prerequisites

Before starting the capstone, ensure you have completed:
- ✅ All Phase 1 labs (Airflow, PyTorch basics)
- ✅ All Phase 2 labs (ETL, feature engineering, data quality)
- ✅ All Phase 3 labs (Model training, two-tower models, experiment tracking)
- ✅ All Phase 4 labs (Serving, monitoring, retraining)

### Environment Setup

1. **Create project directory structure**:
```bash
cd /home/user/mlops-learning-plan
mkdir -p capstone_project/{data,dags,models,api,config,tests,notebooks}
mkdir -p capstone_project/data/{raw,processed,features,predictions}
mkdir -p capstone_project/models/{training,production}
```

2. **Verify your environment**:
```bash
# Should have these installed from previous phases
python --version  # Python 3.8+
pip list | grep -E "airflow|torch|mlflow|fastapi|great-expectations|pandas"
```

3. **Start with Module 1**: Read through `module1_system_design.md` to begin your design phase.

### Recommended Timeline

**Week 1: Foundation**
- Days 1-2: System design (Module 1)
- Days 3-5: ETL pipeline (Module 2)
- Days 6-7: Model training (Module 3)

**Week 2: Deployment & Automation**
- Days 1-2: Model serving (Module 4)
- Days 3-4: Monitoring (Module 5)
- Days 5-6: Retraining (Module 6)
- Day 7: Integration (Module 7)

**Week 3: Polish & Documentation**
- Testing, debugging, optimization
- Documentation
- Demo preparation

**Total Estimated Time**: 2-3 weeks (15-20 hours/week)

## Success Criteria

Your capstone project will be considered successful when you can demonstrate:

### 1. Working System Components
- [ ] ETL DAG runs successfully and produces valid feature sets
- [ ] Training DAG trains models and logs to MLflow
- [ ] FastAPI server serves predictions with <100ms p99 latency
- [ ] Monitoring DAG detects drift and generates alerts
- [ ] Retraining DAG automatically improves model performance

### 2. Code Quality
- [ ] Clean, modular, well-documented code
- [ ] Proper error handling and logging
- [ ] Configuration-driven (no hardcoded values)
- [ ] Follows Python best practices (PEP 8)

### 3. Testing
- [ ] Unit tests for critical functions
- [ ] Integration tests for DAGs
- [ ] End-to-end system test demonstrating full flow

### 4. Documentation
- [ ] System architecture diagram
- [ ] Component documentation
- [ ] API documentation
- [ ] Runbook for deployment and troubleshooting

### 5. Demo
- [ ] 10-minute recorded demo showing:
  - Data flowing through ETL
  - Model training and experiment tracking
  - Live predictions via API
  - Drift detection triggering retraining
  - New model auto-promotion

## Project Structure

```
capstone_project/
├── README.md                 # Project overview and setup
├── config/
│   ├── data_config.yaml      # Data generation and processing config
│   ├── model_config.yaml     # Model architecture and training config
│   └── monitoring_config.yaml # Drift detection thresholds
├── dags/
│   ├── etl_dag.py           # ETL and feature engineering
│   ├── training_dag.py      # Model training
│   ├── monitoring_dag.py    # Drift detection
│   ├── retraining_dag.py    # Automated retraining
│   └── master_dag.py        # Orchestrates all DAGs
├── src/
│   ├── data/
│   │   ├── generator.py     # Synthetic data generation
│   │   ├── validator.py     # Data quality checks
│   │   └── features.py      # Feature engineering
│   ├── models/
│   │   ├── ranker.py        # PyTorch ranking model
│   │   └── trainer.py       # Training logic
│   ├── serving/
│   │   ├── app.py           # FastAPI application
│   │   └── schemas.py       # Request/response models
│   └── monitoring/
│       ├── drift.py         # Drift detection
│       └── metrics.py       # Performance tracking
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_results_analysis.ipynb
├── data/                    # gitignored
├── models/                  # gitignored
└── mlruns/                  # gitignored
```

## Learning Resources

- **Architecture Details**: See `ARCHITECTURE.md`
- **Module Guides**: See `module1_*.md` through `module7_*.md`
- **Feed Ranking Papers**:
  - [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/)
  - [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
  - [Two-Tower Models for Recommendations](https://research.google/pubs/pub48840/)
- **MLOps Best Practices**:
  - [Google ML Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
  - [Chip Huyen's ML Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)

## Getting Help

- Review previous phase labs for specific implementations
- Check `ARCHITECTURE.md` for system design guidance
- Each module has detailed step-by-step instructions
- Use the review checklists at the end of each module

## Next Steps

1. Read through `ARCHITECTURE.md` to understand the complete system design
2. Start with `module1_system_design.md` to plan your implementation
3. Work through modules sequentially - each builds on the previous
4. Test thoroughly at each stage before moving forward

---

**Ready to build a production ML system?** Let's get started with [Module 1: System Design](module1_system_design.md)!
