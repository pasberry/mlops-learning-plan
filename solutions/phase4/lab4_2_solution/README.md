# Lab 4.2: Batch Inference Solution

Complete batch inference pipeline with Airflow orchestration.

## Overview

This solution implements an end-to-end batch inference system:
- Efficient batch processing of large datasets
- PyTorch DataLoader integration for optimal performance
- Airflow DAG for orchestration and scheduling
- Data validation and quality checks
- Automated reporting and archival

## Architecture

```
lab4_2_solution/
├── dags/
│   └── batch_inference_dag.py    # Airflow orchestration
├── ml/
│   └── inference/
│       └── batch_engine.py       # Batch processing engine
└── README.md                      # This file
```

## Components

### 1. Batch Inference Engine (`ml/inference/batch_engine.py`)

**Features:**
- PyTorch DataLoader for efficient batching
- GPU acceleration support
- Chunked processing for large files
- Memory-efficient streaming
- Progress tracking and logging

**Key Methods:**
- `predict_dataframe()` - Predict on pandas DataFrame
- `predict_csv()` - Process CSV files directly
- `predict_batch()` - Predict on list of dictionaries
- `get_stats()` - Get inference statistics

**Performance:**
- Batch size: 256 (configurable)
- Multi-worker data loading
- GPU support when available
- Handles millions of rows efficiently

### 2. Airflow DAG (`dags/batch_inference_dag.py`)

**Pipeline Steps:**
1. **Validate Input** - Check data quality and schema
2. **Run Predictions** - Execute batch inference
3. **Generate Report** - Create prediction statistics
4. **Archive** - Save results to long-term storage
5. **Cleanup** - Remove temporary files

**Schedule:** Daily at 2 AM (configurable)

**Parameters:**
- `input_path` - Input data CSV path
- `output_path` - Output predictions path
- `model_path` - Model checkpoint path
- `batch_size` - Inference batch size
- `id_column` - ID column to preserve
- `archive_dir` - Archive directory

## Setup

### 1. Install Dependencies

```bash
pip install torch pandas numpy airflow
```

### 2. Create Test Data

```bash
# Create directories
mkdir -p /home/user/mlops-learning-plan/data/processed
mkdir -p /home/user/mlops-learning-plan/data/predictions/archive

# Generate test data
python3 << 'EOF'
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 10000

data = {
    'user_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'num_purchases': np.random.randint(0, 50, n_samples),
    'account_age_days': np.random.randint(1, 3650, n_samples),
    'avg_transaction': np.random.uniform(10, 500, n_samples),
    'num_returns': np.random.randint(0, 10, n_samples),
    'is_premium': np.random.randint(0, 2, n_samples),
    'region': np.random.randint(1, 5, n_samples),
    'category_preference': np.random.randint(1, 10, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('/home/user/mlops-learning-plan/data/processed/batch_input.csv', index=False)

print(f"✓ Generated {len(df)} test samples")
print(f"  Columns: {list(df.columns)}")
print(f"  Size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
EOF
```

### 3. Ensure Model Exists

```bash
# Check if model exists
if [ -f /home/user/mlops-learning-plan/models/production/model.pt ]; then
    echo "✓ Model found"
else
    echo "Creating test model..."
    # Use the model creation script from Lab 4.1
fi
```

## Running Batch Inference

### Method 1: Standalone Script

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_2_solution

# Run batch inference directly
python ml/inference/batch_engine.py \
    --input /home/user/mlops-learning-plan/data/processed/batch_input.csv \
    --output /home/user/mlops-learning-plan/data/predictions/batch_predictions.csv \
    --model /home/user/mlops-learning-plan/models/production/model.pt \
    --batch-size 256 \
    --id-column user_id \
    --include-features
```

### Method 2: Python API

```python
from ml.inference.batch_engine import BatchInferenceEngine

# Create engine
engine = BatchInferenceEngine(
    model_path='/home/user/mlops-learning-plan/models/production/model.pt',
    batch_size=256
)

# Option 1: Process CSV
engine.predict_csv(
    input_path='data/processed/batch_input.csv',
    output_path='data/predictions/batch_predictions.csv',
    id_column='user_id',
    include_features=True
)

# Option 2: Process DataFrame
import pandas as pd
df = pd.read_csv('data/processed/batch_input.csv')
results_df = engine.predict_dataframe(df, id_column='user_id')
results_df.to_csv('data/predictions/batch_predictions.csv', index=False)

# Option 3: Process list of dicts
features_list = [
    {'age': 35, 'income': 75000, ...},
    {'age': 45, 'income': 95000, ...}
]
results = engine.predict_batch(features_list)
```

### Method 3: Airflow DAG

```bash
# Copy DAG to Airflow dags folder
cp dags/batch_inference_dag.py $AIRFLOW_HOME/dags/

# Copy ml module
mkdir -p $AIRFLOW_HOME/ml
cp -r ml/inference $AIRFLOW_HOME/ml/

# Trigger DAG
airflow dags trigger batch_inference

# Trigger with custom parameters
airflow dags trigger batch_inference \
    --conf '{
        "input_path": "/path/to/input.csv",
        "output_path": "/path/to/output.csv",
        "batch_size": 512
    }'

# Monitor DAG run
airflow dags list-runs -d batch_inference
```

## Expected Output

### Predictions CSV

```csv
user_id,prediction_score,prediction_class,timestamp,age,income,credit_score,...
1,0.7234,1,2025-11-15 10:30:00.123456,35,75000,720,...
2,0.3456,0,2025-11-15 10:30:00.123456,25,45000,650,...
3,0.8901,1,2025-11-15 10:30:00.123456,45,95000,780,...
```

### Console Output

```
INFO - Loading model from /home/user/mlops-learning-plan/models/production/model.pt
INFO - Using device: cpu
INFO - Model loaded successfully
INFO - Feature columns: ['age', 'income', 'credit_score', ...]
INFO - Running batch inference on 10000 rows
INFO - Processed 25600 rows
INFO - Processed 51200 rows
INFO - Processed 76800 rows
INFO - Batch inference complete: 10000 predictions
INFO - Predictions saved to /home/user/mlops-learning-plan/data/predictions/batch_predictions.csv
```

### Inference Report

```
============================================================
BATCH INFERENCE REPORT
============================================================
Total Predictions: 10,000
Positive: 6,234 (62.3%)
Negative: 3,766 (37.7%)

Score Statistics:
  Mean: 0.6123
  Std:  0.2456
  Min:  0.0234
  Max:  0.9876

Score Distribution:
  (0.0, 0.2]: 1,234 (12.3%)
  (0.2, 0.4]: 1,567 (15.7%)
  (0.4, 0.6]: 2,345 (23.5%)
  (0.6, 0.8]: 3,456 (34.6%)
  (0.8, 1.0]: 1,398 (14.0%)
============================================================
```

## Performance Optimization

### 1. Batch Size Tuning

```python
# Small batch (low memory, slower)
engine = BatchInferenceEngine(model_path='model.pt', batch_size=64)

# Medium batch (balanced)
engine = BatchInferenceEngine(model_path='model.pt', batch_size=256)

# Large batch (high memory, faster)
engine = BatchInferenceEngine(model_path='model.pt', batch_size=1024)
```

### 2. GPU Acceleration

```python
# Force GPU
engine = BatchInferenceEngine(model_path='model.pt', device='cuda')

# Auto-select
engine = BatchInferenceEngine(model_path='model.pt', device=None)

# Force CPU
engine = BatchInferenceEngine(model_path='model.pt', device='cpu')
```

### 3. Chunked Processing

```python
# For very large files (100M+ rows)
engine.predict_csv(
    input_path='huge_file.csv',
    output_path='predictions.csv',
    chunksize=100000  # Process 100k rows at a time
)
```

### 4. Multi-Worker Data Loading

```python
# More workers = faster data loading
engine = BatchInferenceEngine(
    model_path='model.pt',
    num_workers=8  # Use 8 CPU cores for data loading
)
```

## Monitoring and Debugging

### Check Progress

```python
import logging
logging.basicConfig(level=logging.INFO)

# Engine will log progress every 100 batches
engine.predict_csv(...)
```

### Memory Profiling

```bash
# Monitor memory usage
python -m memory_profiler ml/inference/batch_engine.py --input ... --output ...
```

### Performance Profiling

```python
import time
import cProfile

# Time the inference
start = time.time()
engine.predict_csv(...)
print(f"Inference took {time.time() - start:.2f}s")

# Profile the code
cProfile.run('engine.predict_csv(...)')
```

## Airflow DAG Testing

### 1. Validate DAG

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_2_solution

# Test DAG syntax
python dags/batch_inference_dag.py

# Check for import errors
airflow dags list-import-errors
```

### 2. Test Individual Tasks

```bash
# Test validate_input task
airflow tasks test batch_inference validate_input 2025-11-15

# Test run_predictions task
airflow tasks test batch_inference run_predictions 2025-11-15

# Test generate_report task
airflow tasks test batch_inference generate_report 2025-11-15
```

### 3. Run Full DAG

```bash
# Trigger manual run
airflow dags trigger batch_inference

# Check DAG status
airflow dags list-runs -d batch_inference --state running

# View task logs
airflow tasks logs batch_inference run_predictions 2025-11-15
```

## Production Considerations

### 1. Error Handling

```python
def run_batch_predictions(**context):
    try:
        run_batch_inference(...)
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        # Send alert
        send_alert(f"Batch inference failed: {str(e)}")
        raise
```

### 2. Retry Logic

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}
```

### 3. Data Partitioning

```python
# Process data by date partition
input_path = f'/data/processed/batch_input_{execution_date}.csv'
output_path = f'/data/predictions/predictions_{execution_date}.csv'
```

### 4. Output Validation

```python
def validate_output(**context):
    """Validate predictions output."""
    output_path = context['ti'].xcom_pull(key='output_path')
    df = pd.read_csv(output_path)

    # Check row count matches input
    input_count = context['ti'].xcom_pull(key='row_count')
    if len(df) != input_count:
        raise ValueError("Output row count doesn't match input")

    # Check for missing predictions
    if df['prediction_score'].isnull().any():
        raise ValueError("Null predictions found")
```

### 5. SLA Monitoring

```python
dag = DAG(
    'batch_inference',
    default_args=default_args,
    sla_miss_callback=lambda *args: send_alert("SLA missed!"),
)

run_predictions = PythonOperator(
    task_id='run_predictions',
    python_callable=run_batch_predictions,
    sla=timedelta(hours=2),  # Must complete within 2 hours
)
```

## Scaling for Large Datasets

### Distributed Processing

```python
# Use Spark for very large datasets
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("batch_inference").getOrCreate()

# Read data
df = spark.read.csv(input_path, header=True)

# Partition and process
df.repartition(100).foreach_partition(lambda partition:
    engine.predict_batch(list(partition))
)
```

### Parallel Processing

```python
from multiprocessing import Pool

def process_chunk(chunk_path):
    engine = BatchInferenceEngine(model_path='model.pt')
    engine.predict_csv(chunk_path, chunk_path.replace('.csv', '_pred.csv'))

# Process multiple files in parallel
with Pool(processes=4) as pool:
    pool.map(process_chunk, chunk_paths)
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or use chunked processing

```python
engine = BatchInferenceEngine(model_path='model.pt', batch_size=64)
# or
engine.predict_csv(..., chunksize=10000)
```

### Issue: Slow Inference

**Solution:** Increase batch size, use GPU, or multi-worker loading

```python
engine = BatchInferenceEngine(
    model_path='model.pt',
    batch_size=512,
    device='cuda',
    num_workers=8
)
```

### Issue: DAG Import Errors

**Solution:** Check Python path and dependencies

```bash
# Add ml module to Python path
export PYTHONPATH=$PYTHONPATH:/home/user/mlops-learning-plan/solutions/phase4/lab4_2_solution
```

## Next Steps

1. **Lab 4.3**: Add monitoring and drift detection
2. **Lab 4.4**: Implement automated retraining
3. **Lab 4.5**: Integrate into complete system
4. **Optimization**: Profile and optimize for production scale

## Learning Outcomes

After completing this lab, you understand:
- ✅ Building efficient batch inference pipelines
- ✅ Using PyTorch DataLoader for batch processing
- ✅ Orchestrating batch jobs with Airflow
- ✅ Handling large datasets efficiently
- ✅ Implementing data validation and quality checks
- ✅ Performance optimization techniques
- ✅ Production scaling strategies
