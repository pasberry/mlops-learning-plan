"""Training pipeline DAG - Orchestrates model training."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import task functions
from training_pipeline_tasks import (
    validate_features,
    train_model,
    evaluate_model,
    register_model_to_staging,
    notify_completion
)


# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Create DAG
with DAG(
    dag_id='training_pipeline',
    default_args=default_args,
    description='Train and register ML models',
    schedule_interval='@weekly',  # Train weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'pytorch'],
    params={
        'config_path': '../lab3_1_solution/config/model_config.yaml',
        'model_name': 'ctr_model',
        'min_auc': 0.65,
        'min_accuracy': 0.60
    }
) as dag:

    # Task 1: Validate feature data exists
    validate_data = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        doc_md="""
        ### Validate Features

        Checks that feature data from Phase 2 pipeline exists and is valid:
        - Train/val/test splits present
        - Files are not empty
        - Schema is correct
        """
    )

    # Task 2: Train the model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        doc_md="""
        ### Train Model

        Executes the PyTorch training script:
        - Loads features
        - Trains neural network
        - Performs early stopping
        - Saves checkpoints
        """
    )

    # Task 3: Evaluate model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        doc_md="""
        ### Evaluate Model

        Evaluates trained model on test set:
        - Computes metrics (accuracy, AUC)
        - Validates against thresholds
        - Fails if model quality is insufficient
        """
    )

    # Task 4: Register to staging
    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model_to_staging,
        doc_md="""
        ### Register Model

        Registers model to staging registry:
        - Versions the model
        - Saves weights + config + metrics
        - Makes available for serving
        """
    )

    # Task 5: Notify completion
    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        doc_md="""
        ### Notify Completion

        Sends notifications about training completion:
        - Slack message
        - Email to ML team
        - Updates dashboard
        """
    )

    # Define dependencies
    validate_data >> train >> evaluate >> register >> notify


# Documentation
dag.doc_md = """
# Training Pipeline

Orchestrates end-to-end model training:

## Workflow
1. **Validate Features**: Ensure data from ETL pipeline is ready
2. **Train Model**: Execute PyTorch training script
3. **Evaluate Model**: Test on held-out set, validate quality
4. **Register Model**: Save to staging registry with version
5. **Notify**: Alert team about completion

## Configuration
Set parameters in DAG params:
- `config_path`: Path to model config YAML
- `model_name`: Name for model registry
- `min_auc`: Minimum AUC threshold (default: 0.65)
- `min_accuracy`: Minimum accuracy threshold (default: 0.60)

## Scheduling
- **Frequency**: Weekly (can be changed to daily/on-demand)
- **Trigger**: Can also be triggered manually or by feature pipeline completion

## Outputs
- Trained model in `experiments/runs/{model_name}_{date}/`
- Registered model in `models/staging/{model_name}/{version}/`
- Metrics logged to XCom and model registry

## Dependencies
- Requires features from Phase 2 feature engineering pipeline
- Python environment with PyTorch, pandas, sklearn

## Next Steps
After successful training:
1. Review metrics in Airflow UI or model registry
2. Test model in staging environment
3. Promote to production using `ModelRegistry.promote_to_production()`
"""
