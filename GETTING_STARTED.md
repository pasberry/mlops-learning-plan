# Getting Started with MLOps Mastery

**Welcome!** This guide will help you start the course and navigate the materials effectively.

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8+ installed (3.10+ recommended)
- ✅ 8GB+ RAM (16GB recommended)
- ✅ 10GB+ free disk space
- ✅ Basic command line knowledge
- ✅ Git installed

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Set Up Your Environment

```bash
# Clone or navigate to the project
cd /home/user/mlops-learning-plan

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r setup/requirements.txt

# Initialize Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Create Airflow admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Create project directories
mkdir -p dags ml/{data,models,training,tracking,registry} \
         data/{raw,processed,features,predictions} \
         models/{staging,production} \
         experiments/runs logs config
```

### Step 2: Verify Installation

```bash
python3 << 'EOF'
import airflow
import torch
import pandas as pd
print(f"✅ Airflow {airflow.__version__}")
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ Pandas {pd.__version__}")
print("🎉 Environment ready!")
EOF
```

### Step 3: Start Airflow

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler (in new terminal)
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler

# Access UI at http://localhost:8080
# Login: admin / admin
```

---

## 📚 Course Structure

### **Phase 1: Foundations** (1-2 weeks)
Learn the MLOps lifecycle, Airflow basics, and PyTorch fundamentals.

**Start here**: `docs/phase1/README.md`

**Labs**:
1. [First Airflow DAG](docs/phase1/lab1_1_first_airflow_dag.md) - 45 min
2. [PyTorch Training](docs/phase1/lab1_2_pytorch_training.md) - 90 min
3. [Config-Driven Training](docs/phase1/lab1_3_config_driven_training.md) - 60 min

**Complete when**: You can write Airflow DAGs and train PyTorch models

---

### **Phase 2: Data & Pipelines** (2-3 weeks)
Build production ETL and feature engineering with Airflow.

**Start here**: `docs/phase2/README.md`

**Labs**:
1. [ETL Pipeline](docs/phase2/lab2_1_etl_pipeline.md) - 90 min
2. [Feature Engineering](docs/phase2/lab2_2_feature_engineering.md) - 90 min
3. [Data Quality](docs/phase2/lab2_3_data_quality.md) - 90 min
4. [Scheduled Pipeline](docs/phase2/lab2_4_scheduled_pipeline.md) - 90 min

**Complete when**: You can build complete data pipelines with validation

---

### **Phase 3: Modeling & Training** (2-3 weeks)
Integrate PyTorch training into Airflow, implement experiment tracking.

**Start here**: `docs/phase3/README.md`

**Labs**:
1. [Tabular Model](docs/phase3/lab3_1_tabular_model.md) - 120 min
2. [Training DAG](docs/phase3/lab3_2_training_dag.md) - 90 min
3. [Experiment Tracking](docs/phase3/lab3_3_experiment_tracking.md) - 90 min
4. [Two-Tower Model](docs/phase3/lab3_4_two_tower_model.md) - 120 min

**Complete when**: You can orchestrate training and track experiments

---

### **Phase 4: Deployment & Monitoring** (2-3 weeks)
Deploy models, implement monitoring, close the feedback loop.

**Start here**: `docs/phase4/README.md`

**Labs**:
1. [Model Serving](docs/phase4/lab4_1_model_serving.md) - 90 min
2. [Batch Inference](docs/phase4/lab4_2_batch_inference.md) - 90 min
3. [Monitoring](docs/phase4/lab4_3_monitoring.md) - 90 min
4. [Retraining Pipeline](docs/phase4/lab4_4_retraining_pipeline.md) - 120 min
5. [Complete System](docs/phase4/lab4_5_complete_system.md) - 180 min

**Complete when**: You have a complete, production-grade MLOps system

---

## 🎯 How to Use This Course

### For Self-Paced Learning

1. **Read phase overviews first**: Understand the big picture
2. **Complete labs in order**: Each builds on previous work
3. **Do the exercises**: Hands-on practice is essential
4. **Experiment**: Try variations and break things
5. **Build incrementally**: Your final project grows with each lab

### Time Commitment

- **Minimum**: 10 hours/week → Complete in 8-10 weeks
- **Recommended**: 15 hours/week → Complete in 6-8 weeks
- **Intensive**: 20+ hours/week → Complete in 4-6 weeks

### Learning Tips

✅ **Run every code example**: Don't just read, execute
✅ **Modify the code**: Change parameters, add features
✅ **Debug failures**: Understanding errors teaches more than successes
✅ **Take notes**: Document your learnings
✅ **Build your own dataset**: Apply concepts to your domain

---

## 📁 Project Organization

Your project will grow like this:

```
mlops-learning-plan/
├── dags/                    # Airflow DAGs (Phases 1-4)
├── ml/                      # ML code (Phase 1+)
│   ├── data/               # Data utilities (Phase 3)
│   ├── models/             # Model definitions (Phase 3)
│   ├── training/           # Training scripts (Phase 3)
│   ├── tracking/           # Experiment tracking (Phase 3)
│   └── registry/           # Model registry (Phase 3)
├── serving/                 # Model serving (Phase 4)
├── monitoring/              # Monitoring code (Phase 4)
├── config/                  # Configuration files (All phases)
├── data/                    # Data storage (gitignored)
├── models/                  # Model artifacts (gitignored)
├── experiments/             # Experiment runs (gitignored)
└── docs/                    # Course materials (read-only)
```

---

## 🔧 Common Commands

### Airflow

```bash
# Start services
airflow webserver --port 8080
airflow scheduler

# List DAGs
airflow dags list

# Test a task
airflow tasks test <dag_id> <task_id> 2024-01-01

# Trigger DAG
airflow dags trigger <dag_id>

# View logs
airflow tasks logs <dag_id> <task_id> <execution_date>
```

### Python/PyTorch

```bash
# Run training
python ml/train_mnist.py
python ml/train_mnist_config.py --config config/mnist_config.yaml

# Test model
python ml/test_mnist.py

# View TensorBoard
tensorboard --logdir=runs
```

### Project Management

```bash
# Activate environment
source venv/bin/activate

# Install new packages
pip install <package>
pip freeze > setup/requirements.txt

# Clean up
rm -rf data/* models/* experiments/runs/*
airflow db reset
```

---

## 🐛 Troubleshooting

### "airflow: command not found"

```bash
# Make sure venv is activated
source venv/bin/activate
which airflow  # Should show path in venv
```

### "DAG doesn't appear in UI"

```bash
# Check for syntax errors
python dags/your_dag.py

# Check Airflow logs
cat airflow/logs/scheduler/latest/*.log

# Verify dags_folder
grep dags_folder airflow/airflow.cfg
```

### "ModuleNotFoundError"

```bash
# Ensure all dependencies are installed
pip install -r setup/requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### "Port 8080 already in use"

```bash
# Find and kill process
lsof -i :8080
kill -9 <PID>

# Or use different port
airflow webserver --port 8081
```

---

## 📖 Additional Resources

### Official Documentation
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

### Course Materials
- [Full Curriculum](CURRICULUM.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Environment Setup](docs/phase1/environment_setup.md)

### Community
- [Airflow Slack](https://apache-airflow.slack.com/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [MLOps Community](https://mlops.community/)

---

## ✅ Phase Completion Checklist

### Phase 1 ✓
- [ ] Airflow DAG runs successfully
- [ ] PyTorch model trains to >95% accuracy
- [ ] Config-driven training works
- [ ] Understand MLOps lifecycle

### Phase 2 ✓
- [ ] ETL pipeline processes data
- [ ] Feature engineering creates train/val/test splits
- [ ] Data quality checks catch issues
- [ ] Scheduled pipeline runs daily

### Phase 3 ✓
- [ ] Tabular model trains via Airflow
- [ ] Experiment tracking logs all runs
- [ ] Two-tower model built and tested
- [ ] Model registry manages versions

### Phase 4 ✓
- [ ] FastAPI serves model predictions
- [ ] Batch inference scores large datasets
- [ ] Monitoring detects drift
- [ ] Retraining pipeline closes the loop
- [ ] Complete system integrates all components

---

## 🎓 What You'll Achieve

By completing this course, you will:

1. **Build** end-to-end ML systems from scratch
2. **Orchestrate** complex pipelines with Airflow
3. **Train** production models with PyTorch
4. **Deploy** models for batch and online inference
5. **Monitor** model performance and data quality
6. **Automate** the complete ML lifecycle
7. **Speak confidently** about MLOps at senior+ level
8. **Have a portfolio project** demonstrating all skills

---

## 🚀 Ready to Begin?

**Start here**: [Phase 1 - Foundations](docs/phase1/README.md)

Then follow the lab sequence:
1. [Lab 1.1 - First Airflow DAG](docs/phase1/lab1_1_first_airflow_dag.md)
2. Continue through all phases...
3. Complete the course with a production MLOps system!

---

**Questions or stuck?** Review the troubleshooting section or consult the official documentation.

**Let's build production ML systems!** 🔥
