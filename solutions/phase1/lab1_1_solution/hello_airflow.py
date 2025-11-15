"""
My First Airflow DAG - SOLUTION
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
    tags=['tutorial', 'phase1', 'solution'],
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
