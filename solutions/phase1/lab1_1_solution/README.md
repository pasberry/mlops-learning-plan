# Lab 1.1 Solution: Your First Airflow DAG

Complete, working solutions for Lab 1.1 exercises.

## Files Included

1. **hello_airflow.py** - Basic DAG with 3 tasks and parallel execution
2. **data_pipeline_simulation.py** - Challenge solution: Complete data pipeline with XCom

## Prerequisites

- Airflow installed and initialized
- Airflow webserver and scheduler running
- Virtual environment activated

## Setup Instructions

### 1. Copy DAGs to Airflow

Copy the solution files to your Airflow dags folder:

```bash
# Find your Airflow dags folder location
airflow config get-value core dags_folder

# Copy solution files (replace with your actual dags folder path)
cp /home/user/mlops-learning-plan/solutions/phase1/lab1_1_solution/*.py ~/airflow/dags/

# Or create a symbolic link
ln -s /home/user/mlops-learning-plan/solutions/phase1/lab1_1_solution/*.py ~/airflow/dags/
```

### 2. Start Airflow (if not already running)

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler
```

### 3. Verify DAGs Appear

```bash
# List all DAGs
airflow dags list | grep -E "(hello_airflow|data_pipeline_simulation)"

# Check for parsing errors
airflow dags list-import-errors
```

You should see:
```
hello_airflow                | /path/to/dags/hello_airflow.py | ...
data_pipeline_simulation     | /path/to/dags/data_pipeline_simulation.py | ...
```

## Running the Solutions

### Solution 1: hello_airflow

**What it does:**
- Prints a welcome message
- Displays current date/time
- Shows a random motivational quote
- Demonstrates parallel task execution

**How to run:**

1. **Via Airflow UI:**
   - Go to http://localhost:8080
   - Login with `admin` / `admin`
   - Find `hello_airflow` in the DAG list
   - Toggle it to "Active" (unpause)
   - Click the "Play" button → "Trigger DAG"

2. **Via CLI:**
   ```bash
   airflow dags trigger hello_airflow
   ```

3. **Test individual tasks:**
   ```bash
   # Test each task without running full DAG
   airflow tasks test hello_airflow print_welcome 2024-01-01
   airflow tasks test hello_airflow print_date 2024-01-01
   airflow tasks test hello_airflow print_random_quote 2024-01-01
   ```

**Expected Output:**

When you check the task logs, you should see:

*print_welcome task:*
```
==================================================
Welcome to Airflow!
This is your first DAG running successfully.
==================================================
```

*print_date task:*
```
Current date and time: 2024-11-15 10:30:45.123456
Date: 2024-11-15
Time: 10:30:45.123456
```

*print_random_quote task:*
```
==================================================
Quote of the day: A model in production is worth a thousand in notebooks.
==================================================
```

**Task Dependencies:**
```
print_welcome
    ├── print_date
    └── print_random_quote
```

The welcome task runs first, then date and quote run in parallel.

### Solution 2: data_pipeline_simulation (Challenge)

**What it does:**
- **Extract**: Generates sample data (list of numbers)
- **Validate**: Checks data quality (not empty, no negatives)
- **Transform**: Applies transformation (multiplies by 2)
- **Load**: Simulates loading to destination
- **XCom**: Passes data between tasks using Airflow's XCom feature

**How to run:**

1. **Via Airflow UI:**
   - Find `data_pipeline_simulation` in DAG list
   - Toggle to "Active"
   - Click "Trigger DAG"
   - Watch tasks turn green sequentially

2. **Via CLI:**
   ```bash
   airflow dags trigger data_pipeline_simulation
   ```

3. **Test the full pipeline:**
   ```bash
   # Test each task in order
   airflow tasks test data_pipeline_simulation extract_data 2024-01-01
   airflow tasks test data_pipeline_simulation validate_data 2024-01-01
   airflow tasks test data_pipeline_simulation transform_data 2024-01-01
   airflow tasks test data_pipeline_simulation load_data 2024-01-01
   ```

**Expected Output:**

*extract_data task:*
```
============================================================
EXTRACT: Extracting data from source...
============================================================
Extracted 10 records: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

*validate_data task:*
```
============================================================
VALIDATE: Validating extracted data...
============================================================
Received data: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
✓ Validation passed: 10 records are valid
✓ All values are non-negative
✓ Data type is correct (list)
```

*transform_data task:*
```
============================================================
TRANSFORM: Transforming validated data...
============================================================
Original data: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Transformed data: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
Transformation: Each value multiplied by 2

Statistics:
  - Record count: 10
  - Sum: 1100
  - Average: 110.00
  - Min: 20
  - Max: 200
```

*load_data task:*
```
============================================================
LOAD: Loading transformed data to destination...
============================================================
Data to load: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

✓ Successfully loaded 10 records to destination
✓ Data loaded successfully: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
```

**Task Dependencies:**
```
extract_data → validate_data → transform_data → load_data
```

All tasks run sequentially, passing data via XCom.

## Viewing XCom Data

To see the data passed between tasks:

1. **Via UI:**
   - Click on a DAG run
   - Click on a task
   - Go to "XCom" tab
   - You'll see the returned values from each task

2. **Via CLI:**
   ```bash
   # List XComs for a specific DAG run
   airflow db shell
   # Then run SQL query:
   # SELECT * FROM xcom WHERE dag_id='data_pipeline_simulation';
   ```

## Monitoring and Debugging

### Check DAG Status

```bash
# View DAG runs
airflow dags list-runs -d hello_airflow

# View task instances
airflow tasks list hello_airflow
```

### View Logs

```bash
# Via CLI (replace with actual execution date)
airflow tasks logs hello_airflow print_welcome 2024-11-15

# Logs are also at: ~/airflow/logs/
```

### Common Issues and Troubleshooting

**Issue 1: DAG doesn't appear in UI**

*Solution:*
```bash
# Check for Python syntax errors
python ~/airflow/dags/hello_airflow.py

# Check Airflow import errors
airflow dags list-import-errors

# Refresh DAGs (wait 30 seconds or restart scheduler)
```

**Issue 2: Task fails with "No module named..."**

*Solution:*
```bash
# Ensure virtual environment is activated
source ~/venv/bin/activate

# Check Python path in task logs
```

**Issue 3: XCom data is None**

*Solution:*
- Ensure the previous task completed successfully (green)
- Check the task return value in logs
- Verify task_ids match in xcom_pull

**Issue 4: Scheduler not picking up DAG changes**

*Solution:*
```bash
# Restart the scheduler
pkill -f "airflow scheduler"
airflow scheduler

# Or reduce dag_dir_list_interval in airflow.cfg
```

## Testing Without Airflow UI

You can test DAGs without the webserver:

```bash
# Test entire DAG
airflow dags test hello_airflow 2024-01-01

# Test with backfill
airflow dags backfill hello_airflow -s 2024-01-01 -e 2024-01-01
```

## Key Concepts Demonstrated

### hello_airflow.py
- ✅ DAG definition with context manager
- ✅ PythonOperator usage
- ✅ Task dependencies with >> operator
- ✅ Parallel task execution
- ✅ Default arguments
- ✅ DAG tags and metadata

### data_pipeline_simulation.py
- ✅ XCom for inter-task communication
- ✅ Context and task instance (ti)
- ✅ Data validation and error handling
- ✅ Linear pipeline dependencies
- ✅ Logging best practices
- ✅ Simulating real-world data pipelines

## Next Steps

After successfully running these solutions:

1. **Experiment**: Modify the DAGs to add new tasks
2. **Extend**: Add more validation rules or transformations
3. **Combine**: Create a DAG that combines both patterns
4. **Advanced**: Try the TaskFlow API (@task decorator)

## Additional Resources

- [Airflow Task Dependencies](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#task-dependencies)
- [XCom Documentation](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html)
- [PythonOperator Reference](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/python.html)

## Success Criteria

You've successfully completed Lab 1.1 if:

- ✅ Both DAGs appear in Airflow UI
- ✅ `hello_airflow` runs without errors (all tasks green)
- ✅ `data_pipeline_simulation` successfully passes data between tasks
- ✅ You can view task logs
- ✅ You can see XCom data in the UI
- ✅ You understand task dependencies and execution order

---

**Congratulations on completing Lab 1.1!** 🎉

Proceed to Lab 1.2 for PyTorch training.
