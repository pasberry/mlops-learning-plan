# Lab 4.4: Automated Retraining Pipeline

**Goal**: Close the MLOps loop with automated retraining

**Estimated Time**: 120-150 minutes

**Prerequisites**:
- Training pipeline from Phase 3
- Monitoring system from Lab 4.3
- Understanding of model evaluation

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Create a drift-triggered retraining DAG
- ✅ Implement automated model comparison
- ✅ Build model promotion logic
- ✅ Understand A/B testing concepts
- ✅ Complete the full MLOps loop
- ✅ Handle model rollback scenarios

---

## Background: Closing the Loop

### The MLOps Feedback Loop

```
┌─────────────────────────────────────────────────┐
│         CONTINUOUS ML LIFECYCLE                 │
│                                                 │
│  1. DATA COLLECTION                             │
│     └─> New data arrives daily                 │
│                                                 │
│  2. MONITORING                                  │
│     └─> Detect drift or performance drop       │
│                                                 │
│  3. TRIGGER RETRAINING                          │
│     └─> Automatic or manual trigger            │
│                                                 │
│  4. TRAIN NEW MODEL                             │
│     └─> Use latest data                        │
│                                                 │
│  5. VALIDATE & COMPARE                          │
│     └─> New model vs production model          │
│                                                 │
│  6. PROMOTE TO PRODUCTION                       │
│     └─> If new model is better                 │
│                                                 │
│  7. MONITOR NEW MODEL                           │
│     └─> Back to step 2... ♻️                    │
└─────────────────────────────────────────────────┘
```

### Retraining Strategies

#### 1. Scheduled Retraining
```python
# Every Monday at 2 AM
schedule = '0 2 * * 1'

# Pros: Predictable, simple
# Cons: May retrain unnecessarily
```

#### 2. Drift-Triggered Retraining
```python
if psi > 0.2:
    trigger_retraining()

# Pros: Responsive to data changes
# Cons: May be too sensitive
```

#### 3. Performance-Triggered Retraining
```python
if accuracy < 0.80:
    trigger_retraining()

# Pros: Directly tied to business metric
# Cons: Requires labeled data
```

#### 4. Hybrid (Recommended)
```python
# Retrain if any condition met
should_retrain = (
    psi > 0.2 or
    accuracy < 0.80 or
    days_since_training > 30
)
```

---

## Part 1: Model Comparison Module

Create `ml/retraining/model_comparison.py`:

```python
"""
Compare models to decide which should be in production
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare two models on validation data
    """

    def __init__(self, validation_data_path: str):
        """
        Initialize model comparator

        Args:
            validation_data_path: Path to validation dataset
        """
        self.validation_data = pd.read_csv(validation_data_path)
        logger.info(f"Loaded validation data: {len(self.validation_data)} samples")

    def evaluate_model(
        self,
        model_path: str,
        feature_names: list
    ) -> Dict:
        """
        Evaluate a model on validation data

        Args:
            model_path: Path to model checkpoint
            feature_names: List of feature names

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating model: {model_path}")

        # Load model
        from ml.serving.model_service import SimpleClassifier

        checkpoint = torch.load(model_path, map_location='cpu')

        model = SimpleClassifier(
            input_dim=checkpoint.get('input_dim', len(feature_names)),
            hidden_dim=checkpoint.get('hidden_dim', 64),
            output_dim=checkpoint.get('output_dim', 2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Extract features and labels
        X = self.validation_data[feature_names].values.astype(np.float32)
        y_true = self.validation_data['label'].values

        # Generate predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1).numpy()
            y_proba = probs[:, 1].numpy()  # Probability of class 1

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
            'num_samples': len(y_true),
            'model_path': str(model_path),
            'evaluated_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Evaluation complete: {metrics}")
        return metrics

    def compare_models(
        self,
        model_a_path: str,
        model_b_path: str,
        feature_names: list,
        primary_metric: str = 'roc_auc',
        min_improvement: float = 0.01
    ) -> Tuple[str, Dict]:
        """
        Compare two models and determine winner

        Args:
            model_a_path: Path to first model (e.g., current production)
            model_b_path: Path to second model (e.g., new candidate)
            feature_names: List of feature names
            primary_metric: Metric to use for comparison
            min_improvement: Minimum improvement required to promote new model

        Returns:
            (winner, comparison_results)
        """
        logger.info("Comparing models...")
        logger.info(f"  Model A: {model_a_path}")
        logger.info(f"  Model B: {model_b_path}")

        # Evaluate both models
        metrics_a = self.evaluate_model(model_a_path, feature_names)
        metrics_b = self.evaluate_model(model_b_path, feature_names)

        # Compare on primary metric
        score_a = metrics_a[primary_metric]
        score_b = metrics_b[primary_metric]

        improvement = score_b - score_a
        improvement_pct = (improvement / score_a) * 100

        # Determine winner
        if improvement >= min_improvement:
            winner = 'model_b'
            decision = f"Model B wins: {improvement:.4f} improvement ({improvement_pct:.2f}%)"
        elif improvement <= -min_improvement:
            winner = 'model_a'
            decision = f"Model A wins: Model B is {abs(improvement):.4f} worse"
        else:
            winner = 'model_a'  # Tie goes to current model
            decision = f"Tie: Difference too small ({improvement:.4f}), keeping Model A"

        comparison = {
            'winner': winner,
            'decision': decision,
            'primary_metric': primary_metric,
            'model_a': {
                'path': model_a_path,
                'metrics': metrics_a
            },
            'model_b': {
                'path': model_b_path,
                'metrics': metrics_b
            },
            'improvement': {
                'absolute': improvement,
                'percentage': improvement_pct
            },
            'compared_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Comparison result: {decision}")
        return winner, comparison

    def save_comparison_results(
        self,
        comparison: Dict,
        output_path: str
    ):
        """Save comparison results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison results saved to {output_path}")
```

---

## Part 2: Model Promotion Module

Create `ml/retraining/model_promotion.py`:

```python
"""
Promote models between environments (staging → production)
"""

import shutil
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Manage model promotion between environments
    """

    def __init__(
        self,
        staging_dir: str = "models/staging",
        production_dir: str = "models/production",
        archive_dir: str = "models/archive"
    ):
        """
        Initialize model promoter

        Args:
            staging_dir: Directory for staging models
            production_dir: Directory for production models
            archive_dir: Directory for archived models
        """
        self.staging_dir = Path(staging_dir)
        self.production_dir = Path(production_dir)
        self.archive_dir = Path(archive_dir)

        # Create directories
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def promote_to_production(
        self,
        staging_model_path: str,
        model_version: str,
        comparison_results: dict = None,
        keep_previous: bool = True
    ):
        """
        Promote a staging model to production

        Args:
            staging_model_path: Path to staging model
            model_version: Version string (e.g., 'v1.2.0')
            comparison_results: Results from model comparison
            keep_previous: Whether to archive previous production model
        """
        staging_path = Path(staging_model_path)

        if not staging_path.exists():
            raise FileNotFoundError(f"Staging model not found: {staging_path}")

        logger.info(f"Promoting model to production: {model_version}")

        # Archive current production model
        current_production = self.production_dir / "model_latest.pt"

        if current_production.exists() and keep_previous:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = self.archive_dir / f"model_{timestamp}.pt"

            shutil.copy(current_production, archive_path)
            logger.info(f"Archived previous model to {archive_path}")

        # Copy staging model to production
        production_path = self.production_dir / f"model_{model_version}.pt"
        shutil.copy(staging_path, production_path)

        # Update "latest" symlink
        latest_path = self.production_dir / "model_latest.pt"
        if latest_path.exists():
            latest_path.unlink()

        shutil.copy(production_path, latest_path)

        logger.info(f"Model promoted to production: {production_path}")

        # Save promotion metadata
        metadata = {
            'version': model_version,
            'staging_path': str(staging_path),
            'production_path': str(production_path),
            'promoted_at': datetime.utcnow().isoformat(),
            'comparison_results': comparison_results
        }

        metadata_path = self.production_dir / f"model_{model_version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Promotion complete")

    def rollback_to_previous(self):
        """
        Rollback to previous production model
        """
        logger.warning("Rolling back to previous model...")

        # Find most recent archived model
        archived_models = sorted(
            self.archive_dir.glob("model_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not archived_models:
            raise RuntimeError("No archived models available for rollback")

        previous_model = archived_models[0]
        logger.info(f"Rolling back to: {previous_model}")

        # Copy back to production
        latest_path = self.production_dir / "model_latest.pt"
        shutil.copy(previous_model, latest_path)

        logger.warning("Rollback complete")

    def get_production_model_info(self) -> dict:
        """Get information about current production model"""
        latest_path = self.production_dir / "model_latest.pt"

        if not latest_path.exists():
            return {"error": "No production model found"}

        # Find corresponding metadata
        metadata_files = list(self.production_dir.glob("model_*_metadata.json"))

        if metadata_files:
            # Get most recent metadata
            latest_metadata = sorted(
                metadata_files,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[0]

            with open(latest_metadata) as f:
                return json.load(f)

        return {
            "path": str(latest_path),
            "modified_at": datetime.fromtimestamp(
                latest_path.stat().st_mtime
            ).isoformat()
        }
```

---

## Part 3: Retraining DAG

Create `dags/retraining_dag.py`:

```python
"""
Automated Retraining DAG
Triggered by drift detection or schedule
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.retraining.model_comparison import ModelComparator
from ml.retraining.model_promotion import ModelPromoter

logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'feature_names': [
        'age', 'income', 'tenure_days', 'num_purchases',
        'avg_transaction_value', 'days_since_last_purchase'
    ],
    'training_script': 'ml/training/train.py',
    'validation_data': 'data/validation/validation.csv',
    'staging_dir': 'models/staging',
    'production_dir': 'models/production',
    'primary_metric': 'roc_auc',
    'min_improvement': 0.01
}


def check_retraining_needed(**context):
    """
    Decide if retraining is needed
    """
    logger.info("Checking if retraining is needed...")

    # Option 1: Check for drift
    drift_detected = context['dag_run'].conf.get('drift_detected', False)

    if drift_detected:
        logger.info("Retraining triggered by drift detection")
        return 'train_new_model'

    # Option 2: Check schedule (e.g., retrain weekly)
    last_training_date = get_last_training_date()

    if last_training_date:
        days_since_training = (datetime.now() - last_training_date).days
        if days_since_training >= 7:
            logger.info(f"Retraining triggered by schedule ({days_since_training} days)")
            return 'train_new_model'

    # Option 3: Manual trigger
    manual_trigger = context['dag_run'].conf.get('force_retrain', False)

    if manual_trigger:
        logger.info("Retraining triggered manually")
        return 'train_new_model'

    logger.info("No retraining needed")
    return 'skip_retraining'


def get_last_training_date():
    """Get timestamp of last training run"""
    staging_dir = Path(CONFIG['staging_dir'])

    if not staging_dir.exists():
        return None

    models = list(staging_dir.glob("model_*.pt"))

    if not models:
        return None

    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    return datetime.fromtimestamp(latest_model.stat().st_mtime)


def prepare_training_data(**context):
    """
    Prepare data for retraining
    Collect latest data from production
    """
    logger.info("Preparing training data...")

    # In production, aggregate recent production data
    # For now, use existing data

    execution_date = context['ds']
    logger.info(f"Using data up to {execution_date}")

    # Could filter data, create train/val split, etc.
    # This is a placeholder
    logger.info("Training data ready")


def train_new_model(**context):
    """
    Train a new model
    """
    logger.info("Training new model...")

    execution_date = context['ds']
    model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run training script
    # In practice, use your Phase 3 training pipeline
    logger.info(f"Training model version: {model_version}")

    # Placeholder for actual training
    # In real implementation, trigger your training DAG or run training code

    # For demo, just copy a model
    import shutil
    staging_path = Path(CONFIG['staging_dir']) / f"model_{model_version}.pt"
    staging_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy production model as placeholder (replace with actual training)
    prod_model = Path(CONFIG['production_dir']) / "model_latest.pt"
    if prod_model.exists():
        shutil.copy(prod_model, staging_path)
        logger.info(f"New model saved to {staging_path}")

    # Push model path to XCom
    context['ti'].xcom_push(key='new_model_path', value=str(staging_path))
    context['ti'].xcom_push(key='model_version', value=model_version)


def evaluate_and_compare(**context):
    """
    Compare new model to production model
    """
    logger.info("Evaluating and comparing models...")

    new_model_path = context['ti'].xcom_pull(
        task_ids='train_new_model',
        key='new_model_path'
    )

    production_model_path = str(
        Path(CONFIG['production_dir']) / "model_latest.pt"
    )

    # Initialize comparator
    comparator = ModelComparator(
        validation_data_path=CONFIG['validation_data']
    )

    # Compare models
    winner, comparison = comparator.compare_models(
        model_a_path=production_model_path,
        model_b_path=new_model_path,
        feature_names=CONFIG['feature_names'],
        primary_metric=CONFIG['primary_metric'],
        min_improvement=CONFIG['min_improvement']
    )

    # Save comparison results
    comparison_path = Path("models/comparisons") / \
                     f"comparison_{context['ts_nodash']}.json"
    comparator.save_comparison_results(comparison, str(comparison_path))

    # Push results to XCom
    context['ti'].xcom_push(key='comparison_winner', value=winner)
    context['ti'].xcom_push(key='comparison_results', value=comparison)

    logger.info(f"Comparison complete: {winner} wins")


def decide_promotion(**context):
    """
    Decide whether to promote new model
    """
    winner = context['ti'].xcom_pull(
        task_ids='evaluate_and_compare',
        key='comparison_winner'
    )

    if winner == 'model_b':
        logger.info("New model is better - promoting to production")
        return 'promote_to_production'
    else:
        logger.info("Current model is better - keeping in production")
        return 'keep_current_model'


def promote_model(**context):
    """
    Promote new model to production
    """
    logger.info("Promoting model to production...")

    new_model_path = context['ti'].xcom_pull(
        task_ids='train_new_model',
        key='new_model_path'
    )

    model_version = context['ti'].xcom_pull(
        task_ids='train_new_model',
        key='model_version'
    )

    comparison_results = context['ti'].xcom_pull(
        task_ids='evaluate_and_compare',
        key='comparison_results'
    )

    # Initialize promoter
    promoter = ModelPromoter(
        staging_dir=CONFIG['staging_dir'],
        production_dir=CONFIG['production_dir']
    )

    # Promote model
    promoter.promote_to_production(
        staging_model_path=new_model_path,
        model_version=model_version,
        comparison_results=comparison_results
    )

    logger.info(f"Model {model_version} promoted to production")


def send_retraining_notification(**context):
    """
    Send notification about retraining results
    """
    winner = context['ti'].xcom_pull(
        task_ids='evaluate_and_compare',
        key='comparison_winner'
    )

    comparison = context['ti'].xcom_pull(
        task_ids='evaluate_and_compare',
        key='comparison_results'
    )

    if winner == 'model_b':
        message = f"""
        ✅ New model promoted to production!

        Improvement: {comparison['improvement']['percentage']:.2f}%
        Primary metric ({comparison['primary_metric']}):
          - Old model: {comparison['model_a']['metrics'][comparison['primary_metric']]:.4f}
          - New model: {comparison['model_b']['metrics'][comparison['primary_metric']]:.4f}
        """
    else:
        message = f"""
        ℹ️ Retraining completed - keeping current model

        Reason: {comparison['decision']}
        """

    logger.info(message)
    # In production: send to Slack, email, etc.


# Define DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='automated_retraining',
    default_args=default_args,
    description='Automated model retraining and promotion',
    schedule=None,  # Triggered by monitoring DAG or manually
    start_date=days_ago(1),
    catchup=False,
    tags=['retraining', 'production'],
) as dag:

    # Task 1: Check if retraining needed
    check_retraining = BranchPythonOperator(
        task_id='check_retraining_needed',
        python_callable=check_retraining_needed,
        provide_context=True,
    )

    # Task 2: Prepare data
    prepare_data = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data,
        provide_context=True,
    )

    # Task 3: Train new model
    train_model = PythonOperator(
        task_id='train_new_model',
        python_callable=train_new_model,
        provide_context=True,
    )

    # Task 4: Evaluate and compare
    evaluate = PythonOperator(
        task_id='evaluate_and_compare',
        python_callable=evaluate_and_compare,
        provide_context=True,
    )

    # Task 5: Decide promotion
    decide = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=decide_promotion,
        provide_context=True,
    )

    # Task 6a: Promote to production
    promote = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_model,
        provide_context=True,
    )

    # Task 6b: Keep current model
    keep_current = BashOperator(
        task_id='keep_current_model',
        bash_command='echo "Keeping current production model"'
    )

    # Task 7: Send notification
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_retraining_notification,
        provide_context=True,
        trigger_rule='none_failed'  # Run even if some branches skipped
    )

    # Task 8: Skip retraining
    skip = BashOperator(
        task_id='skip_retraining',
        bash_command='echo "No retraining needed"'
    )

    # Define dependencies
    check_retraining >> [prepare_data, skip]
    prepare_data >> train_model >> evaluate >> decide
    decide >> [promote, keep_current]
    [promote, keep_current] >> notify
```

---

## Part 4: Trigger Retraining from Monitoring

Update `dags/monitoring_dag.py` to add:

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

def trigger_retraining_if_needed(**context):
    """
    Trigger retraining DAG if drift detected
    """
    drift_detected = context['ti'].xcom_pull(
        task_ids='detect_feature_drift',
        key='drift_detected'
    )

    if drift_detected:
        logger.warning("Drift detected - triggering retraining DAG")
        return 'trigger_retraining'
    else:
        return 'no_action'

# In monitoring DAG
trigger_retrain = TriggerDagRunOperator(
    task_id='trigger_retraining',
    trigger_dag_id='automated_retraining',
    conf={'drift_detected': True},
)

# Add to DAG
send_alerts >> trigger_retrain
```

---

## Part 5: A/B Testing Concepts

### Gradual Rollout

```python
def serve_prediction(user_id, features):
    """
    Serve predictions with A/B testing
    """
    # Hash user ID to assign to test group
    user_hash = hash(user_id) % 100

    # 95% get production model, 5% get new model
    if user_hash < 5:
        model = load_model("models/staging/model_new.pt")
        variant = "B"
    else:
        model = load_model("models/production/model_latest.pt")
        variant = "A"

    prediction = model.predict(features)

    # Log for analysis
    log_prediction(user_id, prediction, variant)

    return prediction
```

### Monitor A/B Test Results

```python
def analyze_ab_test():
    """
    Compare metrics between variant A and B
    """
    logs = load_prediction_logs()

    variant_a = logs[logs['variant'] == 'A']
    variant_b = logs[logs['variant'] == 'B']

    # Compare metrics (if you have labels)
    if 'true_label' in logs.columns:
        accuracy_a = accuracy_score(variant_a['true_label'], variant_a['prediction'])
        accuracy_b = accuracy_score(variant_b['true_label'], variant_b['prediction'])

        print(f"Variant A accuracy: {accuracy_a:.3f}")
        print(f"Variant B accuracy: {accuracy_b:.3f}")

        # Statistical significance test
        from scipy.stats import chi2_contingency

        contingency_table = pd.crosstab(
            logs['variant'],
            logs['prediction'] == logs['true_label']
        )

        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        if p_value < 0.05:
            print("Difference is statistically significant")
```

---

## Exercises

### Exercise 1: Add Canary Deployment

Gradually increase traffic to new model:

```python
def canary_rollout(new_model_path, stages=[5, 25, 50, 100]):
    """
    Gradually roll out new model
    """
    for traffic_pct in stages:
        # Deploy with traffic_pct% of traffic
        deploy_model(new_model_path, traffic_pct=traffic_pct)

        # Monitor for 24 hours
        time.sleep(24 * 3600)

        # Check metrics
        metrics = get_production_metrics()

        if metrics['error_rate'] > threshold:
            # Rollback
            rollback_deployment()
            break
```

### Exercise 2: Add Shadow Mode

Run new model in parallel without affecting production:

```python
def shadow_mode_prediction(features):
    """
    Run both models, only return production prediction
    """
    # Production model (actual response)
    prod_prediction = production_model.predict(features)

    # New model (shadow, logged but not returned)
    new_prediction = new_model.predict(features)

    # Log both for comparison
    log_shadow_prediction(prod_prediction, new_prediction)

    return prod_prediction  # Only production prediction returned
```

---

## Production Best Practices

### 1. Automated Testing Before Promotion

```python
def run_model_tests(model_path):
    """
    Run automated tests before promotion
    """
    tests = [
        test_model_loading(model_path),
        test_prediction_shape(model_path),
        test_inference_speed(model_path),
        test_on_known_samples(model_path)
    ]

    if all(tests):
        logger.info("All tests passed")
        return True
    else:
        logger.error("Tests failed - blocking promotion")
        return False
```

### 2. Rollback Capability

Always keep previous model:

```python
def deploy_with_rollback(new_model):
    """
    Deploy new model with quick rollback option
    """
    # Keep old model warm
    old_model = get_production_model()

    # Deploy new model
    deploy(new_model)

    # Monitor for 1 hour
    metrics = monitor_for_issues(duration_hours=1)

    if metrics['has_issues']:
        # Instant rollback
        deploy(old_model)
        alert("Rolled back to previous model")
```

### 3. Model Metadata Tracking

```python
model_metadata = {
    'version': 'v1.2.0',
    'training_date': '2024-01-15',
    'training_data': 'data_v2_2024-01-01_to_2024-01-14',
    'features': feature_names,
    'metrics': {
        'accuracy': 0.87,
        'roc_auc': 0.92
    },
    'promoted_at': '2024-01-16T10:00:00Z',
    'comparison_results': {...}
}
```

---

## Key Takeaways

✅ **Automate retraining**: Don't wait for manual triggers
✅ **Always compare**: New model must beat current model
✅ **Promote carefully**: Use staging → production flow
✅ **Keep rollback ready**: Always maintain previous version
✅ **Monitor post-deployment**: Watch for regressions
✅ **Test in production**: A/B tests, canary deployments

---

## Complete MLOps Loop

You've now completed the entire loop:

```
1. Data Pipeline (Phase 2) ✅
   └─> Collect and process data

2. Training Pipeline (Phase 3) ✅
   └─> Train and track models

3. Serving (Lab 4.1-4.2) ✅
   └─> Deploy for inference

4. Monitoring (Lab 4.3) ✅
   └─> Detect drift and issues

5. Retraining (Lab 4.4) ✅
   └─> Retrain and promote

6. Back to step 1... ♻️
```

---

## Next Steps

- ✅ Complete this lab and test automated retraining
- ✅ Share your implementation for review
- → Move to **Lab 4.5: Complete MLOps System**

---

**Congratulations! You've closed the MLOps loop! 🚀**

**Next**: [Lab 4.5 - Complete System →](./lab4_5_complete_system.md)
