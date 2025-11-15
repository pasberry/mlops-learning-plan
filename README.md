# MLOps Mastery: Production ML with PyTorch & Airflow

**A comprehensive, hands-on course to master end-to-end MLOps.**

Transform from novice to expert in production machine learning systems using:
- **Apache Airflow** for ETL and orchestration
- **PyTorch** for model development and training
- **Production patterns** for deployment, monitoring, and retraining

## 🎯 What You'll Build

By the end of this course, you'll have a complete, production-grade MLOps system:

- ✅ ETL pipelines with data validation and quality checks
- ✅ Feature engineering workflows with versioning
- ✅ PyTorch models (tabular, ranking/two-tower architectures)
- ✅ Automated training pipelines with experiment tracking
- ✅ Model serving API (batch and online inference)
- ✅ Monitoring and drift detection
- ✅ Automated retraining loops

## 📚 Course Materials

- **[CURRICULUM.md](CURRICULUM.md)**: Complete 4-phase learning path with detailed labs
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Project organization and best practices

## 🚀 Quick Start

### Phase 1: Foundations (Start Here!)
Learn the MLOps lifecycle, Airflow basics, and PyTorch fundamentals.

### Phase 2: Data & Pipelines
Build production ETL and feature engineering with Airflow.

### Phase 3: Modeling & Training
Integrate PyTorch training into Airflow, implement experiment tracking.

### Phase 4: Deployment & Monitoring
Deploy models, implement monitoring, close the feedback loop.

### 🏆 Capstone Project: Mini Feed Ranking System
Build a complete, production-grade MLOps system integrating all concepts:
- End-to-end feed ranking pipeline (like Meta/TikTok)
- Automated ETL, training, serving, monitoring, and retraining
- 7 comprehensive modules over 2-3 weeks
- **[Start Capstone →](docs/capstone/README.md)**

## 🎓 Who Is This For?

**Target Student**: Senior Software Engineer transitioning to ML/MLOps
- Strong Python and backend systems experience
- New to PyTorch, Airflow, and end-to-end MLOps
- Want to understand AND implement the full ML lifecycle

## 📊 The MLOps Loop

```
DATA → TRAIN → DEPLOY → MONITOR → RETRAIN → (loop)
```

Every component you build fits into this loop. By Phase 4, you'll orchestrate the entire cycle automatically.

## 🛠 Technologies

- **Orchestration**: Apache Airflow
- **Modeling**: PyTorch
- **Serving**: FastAPI
- **Monitoring**: Custom drift detection
- **Experiment Tracking**: MLflow or custom
- **Languages**: Python 3.10+

---

## ⚙️ Environment Setup (Linux)

### Prerequisites
```bash
# Verify Python 3.8+ is installed
python3 --version

# Install if needed (Ubuntu/Debian)
# sudo apt-get update && sudo apt-get install python3.10 python3.10-venv
```

### Quick Setup (5 minutes)

```bash
# 1. Navigate to project directory
cd /home/user/mlops-learning-plan

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r setup/requirements.txt

# 4. Initialize Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# 5. Create Airflow admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# 6. Create project directories
mkdir -p dags ml/{data,models,training} data/{raw,processed,features} logs models/{staging,production} experiments/runs

# 7. Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
venv/
env/

# Data and Models
data/
models/
experiments/runs/

# Airflow
airflow/
logs/
*.log

# Jupyter
.ipynb_checkpoints/

# Environment
.env

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
EOF
```

### Start Airflow

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler (in new terminal, activate venv first)
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler

# Access UI at http://localhost:8080
# Login: admin / admin
```

### Verify Installation

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

**Detailed setup guide**: [docs/phase1/environment_setup.md](docs/phase1/environment_setup.md)

---

## 📈 Progress

- [ ] Phase 1: Foundations
- [ ] Phase 2: Data & Pipelines
- [ ] Phase 3: Modeling & Training
- [ ] Phase 4: Deployment & Monitoring
- [ ] 🏆 Capstone: Mini Feed Ranking System

## 🏗 Project Structure

The project grows incrementally through each phase. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the complete architecture.

## 🤝 Learning Approach

1. **Code-first, pragmatic**: Real implementations, no pseudocode
2. **Incremental complexity**: Build on previous work
3. **Review & iterate**: Get feedback, improve, master
4. **Portfolio-grade**: Every lab contributes to the final system

## 📖 Next Steps

1. Read [CURRICULUM.md](CURRICULUM.md) for the complete learning path
2. Set up your development environment (guide in Phase 1)
3. Start with Lab 1.1: Your first Airflow DAG

---

**Let's build production ML systems! 🔥**
