# Phase 1: Foundations of MLOps & Tooling

**Duration**: 1-2 weeks
**Goal**: Understand the MLOps lifecycle and get hands-on with core tools

---

## Overview

Welcome to Phase 1! This is where you build your mental model of MLOps and get comfortable with the core tools: **Apache Airflow** and **PyTorch**.

### What You'll Learn

1. **The MLOps Lifecycle**
   - The complete loop: data → train → deploy → monitor → retrain
   - Why orchestration matters
   - Where each tool fits

2. **Apache Airflow Fundamentals**
   - What is a DAG (Directed Acyclic Graph)?
   - Tasks, operators, and dependencies
   - Writing your first pipeline
   - Best practices from day one

3. **PyTorch Fundamentals**
   - Tensors and computation graphs
   - Building neural networks with `nn.Module`
   - Training loops
   - Data loading with `DataLoader`

4. **Configuration-Driven Development**
   - Why configs matter in ML
   - YAML configs for experiments
   - Separating code from configuration

---

## The MLOps Mental Model

Before writing any code, internalize this:

```
┌─────────────────────────────────────────────────────────┐
│                  THE ML LIFECYCLE                       │
│                                                         │
│  1. DATA LOOP                                           │
│     ├─ Ingest raw data (batch or streaming)            │
│     ├─ Validate quality                                 │
│     ├─ Engineer features                                │
│     └─ Version and partition                            │
│                                                         │
│  2. MODEL LOOP                                          │
│     ├─ Train models                                     │
│     ├─ Evaluate performance                             │
│     ├─ Track experiments                                │
│     └─ Register best model                              │
│                                                         │
│  3. DEPLOYMENT LOOP                                     │
│     ├─ Load model for inference                         │
│     ├─ Serve predictions (batch or online)              │
│     └─ Log predictions and inputs                       │
│                                                         │
│  4. MONITORING LOOP                                     │
│     ├─ Track model performance                          │
│     ├─ Detect data/prediction drift                     │
│     ├─ Trigger alerts                                   │
│     └─ Initiate retraining                              │
│                                                         │
│  Then back to DATA LOOP with new data... ♻️             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Where Tools Fit

- **Airflow**: Orchestrates ALL loops (schedules, triggers, chains tasks)
- **PyTorch**: Powers the MODEL LOOP (training and inference)
- **Experiment Tracking**: Captures runs, metrics, artifacts
- **Serving**: Exposes models (FastAPI, batch jobs)
- **Monitoring**: Watches for drift and degradation

**Key Insight**: Every stage is a pipeline. Airflow orchestrates pipelines. Everything else is a task in those pipelines.

---

## MLOps vs DevOps vs Traditional ML

### Traditional ML (Notebooks)
```python
# Load data
df = pd.read_csv('data.csv')

# Train model
model.fit(X, y)

# Save
pickle.dump(model, 'model.pkl')

# Hope it works in production... 🤞
```

**Problems**:
- Not reproducible (where's the code version? data version?)
- Not automated (manual runs)
- Not monitored (how's it performing?)
- Not scalable (one-off scripts)

### DevOps (Software Engineering)
```
Code → Build → Test → Deploy → Monitor
```

**Focus**: Shipping code reliably and quickly

### MLOps (ML Engineering)
```
Data → Feature → Train → Evaluate → Deploy → Monitor → Retrain
         ↑                                              |
         └──────────────────────────────────────────────┘
```

**Focus**: Shipping ML systems reliably and quickly
**Key Difference**: Data and models are versioned artifacts, not just code

### MLOps = DevOps + Data Versioning + Model Lifecycle Management

---

## Why Apache Airflow?

**Airflow is an orchestrator**: It schedules and runs workflows.

### Core Concepts

#### 1. DAG (Directed Acyclic Graph)
A DAG is a workflow: a collection of tasks with dependencies.

```python
# Example DAG structure
ingest_data → validate_data → engineer_features
                                     ↓
                              train_model → evaluate_model
```

- **Directed**: Tasks have order (A before B)
- **Acyclic**: No loops (can't go back)
- **Graph**: Tasks and dependencies

#### 2. Tasks
Individual units of work:
- `PythonOperator`: Run Python functions
- `BashOperator`: Run shell commands
- `@task` decorator: Modern TaskFlow API

#### 3. Dependencies
Define execution order:
```python
task_a >> task_b  # task_a runs before task_b
task_b << task_a  # same thing
[task_a, task_b] >> task_c  # parallel tasks
```

#### 4. Scheduling
Run DAGs on a schedule:
- Daily: `@daily` or `0 0 * * *`
- Hourly: `@hourly`
- Weekly: `0 0 * * 0`
- Custom cron expressions

### Why Use Airflow for ML?

✅ **Reproducibility**: DAG code versions your workflow
✅ **Scheduling**: Automatic daily training, batch inference
✅ **Monitoring**: Built-in UI to see task status
✅ **Retry logic**: Handle transient failures
✅ **Backfills**: Rerun historical data
✅ **Dependency management**: Ensure tasks run in order
✅ **Scalability**: Distributed execution

---

## Why PyTorch?

**PyTorch is a deep learning framework**: Flexible, pythonic, production-ready.

### Core Concepts

#### 1. Tensors
Multi-dimensional arrays (like NumPy, but GPU-accelerated):
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2  # [2.0, 4.0, 6.0]
```

#### 2. Autograd (Automatic Differentiation)
PyTorch tracks operations and computes gradients automatically:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute gradient
print(x.grad)  # dy/dx = 2*x = 4.0
```

#### 3. nn.Module (Building Blocks)
Define models as classes:
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
```

#### 4. Training Loop
Standard pattern:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Why PyTorch for Production ML?

✅ **Flexibility**: Easy to customize and debug
✅ **Pythonic**: Feels like regular Python
✅ **Production-ready**: Used at Meta, Tesla, Uber, etc.
✅ **TorchServe**: Official model serving
✅ **Distributed training**: Built-in support
✅ **ONNX export**: Deploy anywhere

---

## Phase 1 Labs

### Lab 1.1: First Airflow DAG
**Goal**: Write and run a simple 3-task DAG
- Set up Airflow locally
- Create a DAG with dependencies
- Run and monitor in Airflow UI

[→ Go to Lab 1.1](./lab1_1_first_airflow_dag.md)

### Lab 1.2: PyTorch Training Script
**Goal**: Build a complete training script from scratch
- Load MNIST dataset
- Define a simple neural network
- Implement training and evaluation loops
- Save model checkpoint

[→ Go to Lab 1.2](./lab1_2_pytorch_training.md)

### Lab 1.3: Config-Driven Training
**Goal**: Separate configuration from code
- Create YAML config file
- Load config in Python
- Make training script config-driven
- Run experiments with different configs

[→ Go to Lab 1.3](./lab1_3_config_driven_training.md)

---

## Success Criteria

You've completed Phase 1 when you can:

✅ Explain the ML lifecycle and where each tool fits
✅ Write and debug Airflow DAGs
✅ Build and train PyTorch models
✅ Use configs to drive experiments
✅ Understand orchestration vs computation (Airflow vs PyTorch)

---

## Best Practices to Internalize Now

### 1. Idempotency
**Re-running a task with the same inputs produces the same outputs.**

```python
# BAD: Appends to file (not idempotent)
with open('data.csv', 'a') as f:
    f.write(new_data)

# GOOD: Overwrites file (idempotent)
with open('data.csv', 'w') as f:
    f.write(all_data)
```

### 2. Data Partitioning
**Organize data by date/version for incremental processing.**

```
data/
  raw/
    2024-01-01/
    2024-01-02/
  processed/
    2024-01-01/
    2024-01-02/
```

### 3. Configuration Over Hardcoding
```python
# BAD
learning_rate = 0.001
batch_size = 32

# GOOD
config = load_config('config.yaml')
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
```

### 4. Explicit Over Implicit
```python
# BAD
def train():
    # Where's the data? Where's the model saved?
    pass

# GOOD
def train(data_path: str, model_output_path: str, config: dict):
    # Clear inputs and outputs
    pass
```

---

## Next Steps

1. **Set up your environment** (see next section)
2. **Complete Lab 1.1** (First Airflow DAG)
3. **Share your code** for review
4. **Iterate** based on feedback
5. **Move to Lab 1.2** when ready

---

## Resources

### Airflow
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [TaskFlow API Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### MLOps Concepts
- [MLOps Principles](https://ml-ops.org/)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

**Ready to get started? Let's set up your environment!** 🚀
