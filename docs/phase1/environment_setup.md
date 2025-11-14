# Environment Setup for Phase 1

This guide will help you set up your local development environment for the MLOps course.

---

## Prerequisites

### Required
- **Python 3.8+** (3.10 or 3.11 recommended)
- **8GB+ RAM** (16GB recommended for later phases)
- **10GB+ free disk space**
- **Command line basics** (terminal comfort)
- **Git** (for version control)

### Operating System
- **Linux** (preferred)
- **macOS** (works great)
- **Windows** (use WSL2 for best experience)

---

## Setup Steps

### Step 1: Verify Python Version

```bash
python --version
# or
python3 --version

# Should show Python 3.8 or higher
```

If you need to install Python:
- **Ubuntu/Debian**: `sudo apt-get install python3.10`
- **macOS**: `brew install python@3.10`
- **Windows**: Download from [python.org](https://www.python.org/)

---

### Step 2: Create Virtual Environment

Always use a virtual environment to avoid dependency conflicts.

```bash
# Navigate to project directory
cd mlops-learning-plan

# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Git Bash):
source venv/Scripts/activate

# You should see (venv) in your prompt
```

**Important**: Activate this environment every time you work on the project!

---

### Step 3: Install Core Dependencies

Create a requirements file:

```bash
# This will be created as setup/requirements.txt
```

**setup/requirements.txt**:
```
# Phase 1 Requirements

# Airflow
apache-airflow==2.8.1
apache-airflow-providers-sqlite==3.7.0

# PyTorch (CPU version - lighter for development)
torch==2.1.2
torchvision==0.16.2

# Data Science
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0

# Configuration
pyyaml==6.0.1

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1

# Jupyter (optional, for exploration)
jupyter==1.0.0
ipykernel==6.27.1
```

Install dependencies:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r setup/requirements.txt
```

**Note**: PyTorch installation may vary by system. For GPU support, see [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---

### Step 4: Initialize Airflow

Airflow needs initialization before first use.

```bash
# Set Airflow home (where config and databases live)
export AIRFLOW_HOME=$(pwd)/airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# This creates:
# - airflow/airflow.db (SQLite database)
# - airflow/airflow.cfg (configuration)
```

**Important**: Add `airflow/` to `.gitignore` (database and logs shouldn't be committed).

---

### Step 5: Configure Airflow

Edit `airflow/airflow.cfg` for development:

```bash
# Find these lines and update:

# Load examples (set to False to hide example DAGs)
load_examples = False

# DAGs folder (where you'll put your DAG files)
dags_folder = /path/to/mlops-learning-plan/dags

# Default timezone
default_timezone = America/Los_Angeles  # or your timezone
```

Or set via environment variables:
```bash
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
```

---

### Step 6: Create Project Directories

```bash
# Create directories for Phase 1
mkdir -p dags
mkdir -p ml
mkdir -p data/{raw,processed,features}
mkdir -p logs
mkdir -p models/{staging,production}
mkdir -p experiments/runs

# Create .gitkeep files for empty directories
touch data/.gitkeep
touch models/.gitkeep
touch experiments/runs/.gitkeep
```

---

### Step 7: Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# Data and Models (usually too large)
data/
models/
experiments/runs/

# Airflow
airflow/
!airflow/.gitkeep
logs/
*.log

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

---

### Step 8: Verify Installation

Create a simple test script:

```bash
cat > test_setup.py << 'EOF'
"""Test environment setup."""

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import airflow
        print(f"✅ Airflow {airflow.__version__}")

        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")

        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")

        import yaml
        print(f"✅ PyYAML (available)")

        print("\n🎉 All imports successful! Environment is ready.")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    return True


def test_pytorch():
    """Test basic PyTorch functionality."""
    import torch

    # Create tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2

    print("\n🧪 PyTorch Test:")
    print(f"  Input: {x}")
    print(f"  Output: {y}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    return True


def test_airflow():
    """Test Airflow setup."""
    import os
    from airflow import __version__

    airflow_home = os.getenv('AIRFLOW_HOME', 'airflow')

    print("\n🧪 Airflow Test:")
    print(f"  Version: {__version__}")
    print(f"  AIRFLOW_HOME: {airflow_home}")
    print(f"  Database exists: {os.path.exists(f'{airflow_home}/airflow.db')}")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Testing MLOps Environment Setup")
    print("=" * 50)

    test_imports()
    test_pytorch()
    test_airflow()

    print("\n" + "=" * 50)
    print("Setup test complete!")
    print("=" * 50)
EOF

python test_setup.py
```

Expected output:
```
==================================================
Testing MLOps Environment Setup
==================================================
✅ Airflow 2.8.1
✅ PyTorch 2.1.2
✅ NumPy 1.24.3
✅ Pandas 2.1.4
✅ Scikit-learn 1.3.2
✅ PyYAML (available)

🎉 All imports successful! Environment is ready.

🧪 PyTorch Test:
  Input: tensor([1., 2., 3.])
  Output: tensor([2., 4., 6.])
  CUDA available: False

🧪 Airflow Test:
  Version: 2.8.1
  AIRFLOW_HOME: /path/to/airflow
  Database exists: True

==================================================
Setup test complete!
==================================================
```

---

### Step 9: Start Airflow (for testing)

```bash
# In one terminal: Start the webserver
airflow webserver --port 8080

# In another terminal: Start the scheduler
airflow scheduler

# Access UI at http://localhost:8080
# Login: admin / admin
```

You should see the Airflow UI with no DAGs (since we haven't created any yet).

**Tip**: You'll start these services when working on Airflow labs.

---

## Quick Start Commands

Add these to your shell profile for convenience:

```bash
# Add to ~/.bashrc or ~/.zshrc

alias activate-mlops='cd /path/to/mlops-learning-plan && source venv/bin/activate'
alias start-airflow='airflow webserver & airflow scheduler'
alias stop-airflow='pkill -f airflow'

# Set AIRFLOW_HOME
export AIRFLOW_HOME="/path/to/mlops-learning-plan/airflow"
```

Then:
```bash
source ~/.bashrc  # or ~/.zshrc
activate-mlops    # Activates venv and cd's to project
```

---

## Troubleshooting

### Issue: "airflow: command not found"

**Solution**: Make sure your virtual environment is activated:
```bash
source venv/bin/activate
which airflow  # Should show path in venv
```

### Issue: "No module named 'airflow'"

**Solution**: Reinstall Airflow:
```bash
pip install apache-airflow==2.8.1
```

### Issue: PyTorch installation is slow or fails

**Solution**:
1. Install CPU-only version (smaller):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
2. Or install from PyTorch website with exact commands for your system

### Issue: Airflow webserver fails to start

**Solution**: Check if port 8080 is already in use:
```bash
lsof -i :8080
# Kill the process or use a different port:
airflow webserver --port 8081
```

### Issue: Permission errors on macOS/Linux

**Solution**: Make sure you own the project directory:
```bash
sudo chown -R $USER:$USER /path/to/mlops-learning-plan
```

---

## Optional: Docker Setup

If you prefer Docker (advanced):

```bash
# Coming in future phases
# For now, local setup is recommended for learning
```

---

## Verification Checklist

Before proceeding to labs, verify:

- ✅ Python 3.8+ installed
- ✅ Virtual environment created and activated
- ✅ All packages installed (run `test_setup.py`)
- ✅ Airflow initialized (database and user created)
- ✅ Airflow UI accessible at http://localhost:8080
- ✅ Project directories created
- ✅ `.gitignore` configured

---

## Next Steps

Now that your environment is ready:

1. **Proceed to Lab 1.1**: Your First Airflow DAG
2. **Keep your virtual environment activated**: Always run `source venv/bin/activate`
3. **Start Airflow when needed**: `airflow webserver` + `airflow scheduler`

---

**Environment ready? Let's build! 🚀**
