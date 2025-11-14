# Lab 1.1: Your First Airflow DAG

**Goal**: Write and run a simple Airflow DAG with 3 tasks and dependencies

**Estimated Time**: 45-60 minutes

**Prerequisites**:
- Environment setup complete
- Airflow initialized and running

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Understand DAG structure and syntax
- ✅ Create tasks using `PythonOperator` and `@task` decorator
- ✅ Define task dependencies
- ✅ Run and monitor DAGs in Airflow UI
- ✅ Debug task failures

---

## Background: Anatomy of an Airflow DAG

### Core Components

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# 1. Define the DAG
with DAG(
    dag_id='my_dag',              # Unique identifier
    start_date=datetime(2024, 1, 1),  # When DAG can start running
    schedule='@daily',             # How often to run
    catchup=False,                 # Don't backfill past runs
) as dag:

    # 2. Define tasks
    def my_function():
        print("Hello from Airflow!")

    task1 = PythonOperator(
        task_id='my_task',
        python_callable=my_function
    )

    # 3. Define dependencies
    # (We'll add more tasks and dependencies below)
```

### Task Dependencies Syntax

```python
# Method 1: Bitshift operators (recommended)
task_a >> task_b >> task_c  # Linear: a → b → c

# Method 2: Explicit set_downstream
task_a.set_downstream(task_b)

# Multiple dependencies
task_a >> [task_b, task_c]  # Parallel: a → b, a → c
[task_b, task_c] >> task_d  # Join: b → d, c → d
```

### Modern TaskFlow API (Alternative)

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False
)
def my_dag():
    @task
    def task_one():
        return "Data from task one"

    @task
    def task_two(input_data):
        print(f"Received: {input_data}")

    # Dependencies are implicit through function calls
    data = task_one()
    task_two(data)

# Instantiate the DAG
my_dag_instance = my_dag()
```

Both approaches work. For this lab, we'll use the traditional approach, then introduce TaskFlow in later labs.

---

## Lab Instructions

### Part 1: Create Your First DAG

Create `dags/hello_airflow.py`:

```python
"""
My First Airflow DAG
A simple pipeline with 3 tasks demonstrating dependencies.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Default arguments applied to all tasks
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,  # Don't wait for past runs to succeed
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,              # Retry once if task fails
    'retry_delay': timedelta(minutes=1),
}


def print_welcome():
    """Task 1: Print a welcome message."""
    print("=" * 50)
    print("Welcome to Airflow!")
    print("This is your first DAG running successfully.")
    print("=" * 50)
    return "Welcome message printed"


def print_date():
    """Task 2: Print the current date and time."""
    from datetime import datetime
    current_time = datetime.now()
    print(f"Current date and time: {current_time}")
    print(f"Date: {current_time.date()}")
    print(f"Time: {current_time.time()}")
    return str(current_time)


def print_random_quote():
    """Task 3: Print a random motivational quote."""
    import random

    quotes = [
        "Data is the new oil, but models are the refineries.",
        "In MLOps we trust, in pipelines we automate.",
        "A model in production is worth a thousand in notebooks.",
        "Monitoring is caring.",
        "Retrain or refrain from production gains.",
    ]

    quote = random.choice(quotes)
    print("=" * 50)
    print(f"Quote of the day: {quote}")
    print("=" * 50)
    return quote


# Define the DAG
with DAG(
    dag_id='hello_airflow',
    default_args=default_args,
    description='My first Airflow DAG - A simple 3-task pipeline',
    schedule=None,  # Manual trigger only (for now)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['tutorial', 'phase1'],
) as dag:

    # Define tasks
    task_welcome = PythonOperator(
        task_id='print_welcome',
        python_callable=print_welcome,
    )

    task_date = PythonOperator(
        task_id='print_date',
        python_callable=print_date,
    )

    task_quote = PythonOperator(
        task_id='print_random_quote',
        python_callable=print_random_quote,
    )

    # Define dependencies
    # welcome runs first, then date and quote run in parallel
    task_welcome >> [task_date, task_quote]
```

**Save the file**: `dags/hello_airflow.py`

---

### Part 2: Verify DAG Appears in Airflow

1. **Make sure Airflow is running**:
   ```bash
   # Terminal 1: Webserver
   airflow webserver --port 8080

   # Terminal 2: Scheduler
   airflow scheduler
   ```

2. **Check for DAG parsing errors**:
   ```bash
   # In a third terminal (with venv activated)
   airflow dags list | grep hello_airflow

   # Should show:
   # hello_airflow | /path/to/dags/hello_airflow.py | ...
   ```

3. **Open Airflow UI**:
   - Go to http://localhost:8080
   - Login with `admin` / `admin`
   - You should see `hello_airflow` in the DAG list

4. **Verify DAG structure**:
   - Click on the `hello_airflow` DAG
   - View the Graph view
   - You should see:
     ```
     print_welcome → print_date
                  → print_random_quote
     ```

---

### Part 3: Run Your DAG

1. **Enable the DAG**:
   - In the UI, toggle the DAG from "Paused" to "Active" (switch on the left)

2. **Trigger the DAG manually**:
   - Click the "Play" button (▶️) on the right
   - Select "Trigger DAG"
   - Or use CLI: `airflow dags trigger hello_airflow`

3. **Monitor execution**:
   - Click on the DAG name
   - Watch tasks turn from white → yellow → green
   - Click on individual tasks to see logs

4. **View task logs**:
   - Click on `print_welcome` task
   - Click "Logs"
   - You should see the welcome message printed

---

### Part 4: Understand Task States

| Color | State | Meaning |
|-------|-------|---------|
| ⬜ White | None | Not yet scheduled |
| 🟡 Yellow | Running | Currently executing |
| 🟢 Green | Success | Completed successfully |
| 🔴 Red | Failed | Task failed |
| 🟠 Orange | Upstream Failed | Parent task failed |
| 🔵 Blue | Skipped | Skipped by branching logic |

---

## Exercise: Extend Your DAG

Now that you have a working DAG, extend it:

### Exercise 1: Add a Fourth Task

Add a task that:
- Runs **after** both `print_date` and `print_random_quote`
- Prints a summary message like "DAG execution complete!"

**Hint**:
```python
def print_summary():
    print("All tasks completed successfully!")

task_summary = PythonOperator(
    task_id='print_summary',
    python_callable=print_summary,
)

# Add dependencies here
[task_date, task_quote] >> task_summary
```

### Exercise 2: Pass Data Between Tasks

Modify tasks to return values and use XComs:

```python
def get_current_hour():
    from datetime import datetime
    hour = datetime.now().hour
    print(f"Current hour: {hour}")
    return hour  # This will be stored in XCom

def check_time_of_day(**context):
    # Pull value from XCom
    hour = context['ti'].xcom_pull(task_ids='get_hour')

    if hour < 12:
        print("Good morning!")
    elif hour < 18:
        print("Good afternoon!")
    else:
        print("Good evening!")

task_get_hour = PythonOperator(
    task_id='get_hour',
    python_callable=get_current_hour,
)

task_greet = PythonOperator(
    task_id='greet_based_on_time',
    python_callable=check_time_of_day,
)

task_get_hour >> task_greet
```

### Exercise 3: Simulate a Failure

Add a task that deliberately fails:

```python
def simulate_failure():
    raise ValueError("This task intentionally failed for testing!")

task_fail = PythonOperator(
    task_id='intentional_failure',
    python_callable=simulate_failure,
)
```

Add it to your DAG and observe:
- What happens to downstream tasks?
- How does the UI show the failure?
- What do the logs say?

Then comment it out or remove it.

---

## Challenge: Build a Data Pipeline Simulation

Create a new DAG (`dags/data_pipeline_simulation.py`) that simulates a data pipeline:

```
extract_data → validate_data → transform_data → load_data
```

Requirements:
- `extract_data`: Print "Extracting data from source..." and return a list of numbers
- `validate_data`: Check if data is not empty, raise error if it is
- `transform_data`: Multiply each number by 2
- `load_data`: Print "Data loaded successfully: {result}"

Use XComs to pass data between tasks.

**Bonus**: Add proper logging, docstrings, and error handling.

---

## Key Takeaways

### DAG Best Practices

✅ **Idempotent tasks**: Running a task multiple times with the same input produces the same output
✅ **Atomic tasks**: Each task does one thing well
✅ **Explicit dependencies**: Make task order clear
✅ **Descriptive names**: `task_id` should be clear and readable
✅ **Set `catchup=False`**: Unless you need historical backfills
✅ **Use tags**: Organize DAGs by category

### Common Pitfalls to Avoid

❌ **Don't put heavy computation in the DAG file**: DAG files are parsed frequently
❌ **Don't use dynamic task generation carelessly**: Can create huge DAGs
❌ **Don't ignore failures**: Always check logs
❌ **Don't hardcode paths**: Use variables and configs

---

## Testing Your DAG

Before running in Airflow, test locally:

```bash
# Test if DAG file has syntax errors
python dags/hello_airflow.py

# List all DAGs
airflow dags list

# Test a specific task
airflow tasks test hello_airflow print_welcome 2024-01-01
```

---

## Debugging Tips

### DAG doesn't appear in UI?

1. Check for Python syntax errors:
   ```bash
   python dags/hello_airflow.py
   ```

2. Check Airflow logs:
   ```bash
   cat logs/scheduler/latest/hello_airflow.py.log
   ```

3. Verify `dags_folder` in `airflow.cfg` points to the right directory

### Task fails but logs are unclear?

- Add more `print()` statements
- Use Python's `logging` module:
  ```python
  import logging
  logging.info("This is a log message")
  ```

### Task stuck in "running" state?

- Check scheduler is running: `ps aux | grep airflow`
- Restart scheduler: `pkill -f 'airflow scheduler' && airflow scheduler`

---

## Submission Checklist

Before moving to the next lab, ensure:

- ✅ `hello_airflow.py` DAG runs successfully
- ✅ All three tasks complete (green in UI)
- ✅ You can view task logs
- ✅ At least one exercise completed
- ✅ You understand task dependencies

---

## Next Steps

**Share your code**:
1. Commit your DAG: `git add dags/hello_airflow.py`
2. Share the file or describe what you built
3. Mention any challenges or questions

**After review**:
- Incorporate feedback
- Move to **Lab 1.2: PyTorch Training Script**

---

## Additional Resources

- [Airflow DAG Documentation](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html)
- [Airflow Operators](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html)
- [TaskFlow API](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)

---

**Congratulations on creating your first Airflow DAG! 🎉**

**Next**: [Lab 1.2 - PyTorch Training Script →](./lab1_2_pytorch_training.md)
