# Module 6: Automated Retraining & Model Promotion

**Estimated Time**: 2-3 days
**Difficulty**: Medium-Hard

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Build automated retraining pipelines triggered by drift/schedule
- ✅ Implement model comparison and validation logic
- ✅ Create safe model promotion workflows
- ✅ Design A/B testing frameworks (conceptually)
- ✅ Handle model rollback scenarios
- ✅ Close the ML lifecycle loop

## Overview

This module builds the **automated retraining system** that:
1. Monitors for retraining triggers (drift, schedule, performance)
2. Fetches recent data and retrains model
3. Evaluates new model on holdout data
4. Compares new model vs production model
5. Promotes new model if better (with safety checks)
6. Orchestrates everything with Airflow

**Key Principle**: "Automate what's routine, make critical decisions explicit." Retraining should be automatic, but promotion should have guardrails.

## Background

### Why Automated Retraining?

**Manual Retraining Problems**:
- ❌ Slow to respond to drift
- ❌ Requires constant human attention
- ❌ Inconsistent (depends on who does it)
- ❌ Doesn't scale

**Automated Retraining Benefits**:
- ✅ Fast response to changes
- ✅ Consistent process
- ✅ Scales to many models
- ✅ Frees humans for strategic work

### Retraining Triggers

**Time-Based**:
- Daily, weekly, monthly
- Pros: Predictable, simple
- Cons: May retrain unnecessarily

**Event-Based**:
- Drift detected
- Performance degradation
- New data volume threshold
- Pros: Responds to actual need
- Cons: More complex

**Best Practice**: Combine both (scheduled baseline + event triggers)

### Model Promotion Strategies

**1. Immediate Replacement** (Risky)
```
New Model Better → Replace Production Model
```

**2. Shadow Mode** (Safer)
```
New Model → Run in parallel → Compare → Promote if better
```

**3. Canary Deployment** (Safest)
```
New Model → 1% traffic → 10% → 50% → 100%
```

**4. A/B Testing** (Most Rigorous)
```
Split traffic → Measure business metrics → Promote winner
```

## Step 1: Implement Model Comparison Logic

**File**: `src/models/comparator.py`

```python
"""
Model comparison logic for promotion decisions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, log_loss
import json
from pathlib import Path


class ModelComparator:
    """Compare two models for promotion decision."""

    def __init__(self, config: Dict = None):
        """
        Args:
            config: Comparison configuration
        """
        self.config = config or {
            'min_auc_improvement': 0.01,  # 1% AUC improvement required
            'max_performance_degradation': 0.005,  # Max 0.5% degradation allowed
            'metrics': ['auc', 'log_loss']
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compare_models(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        test_loader: DataLoader,
        model_type: str
    ) -> Dict:
        """
        Compare two models on test data.

        Args:
            model_a: First model (typically production model)
            model_b: Second model (typically new candidate)
            test_loader: Test data loader
            model_type: 'two_tower' or 'deep_mlp'

        Returns:
            Dict with comparison results and decision
        """
        print("Comparing models...")

        # Evaluate both models
        metrics_a = self._evaluate_model(model_a, test_loader, model_type)
        metrics_b = self._evaluate_model(model_b, test_loader, model_type)

        # Calculate improvements
        improvements = {
            metric: metrics_b[metric] - metrics_a[metric]
            for metric in self.config['metrics']
            if metric in metrics_a and metric in metrics_b
        }

        # Make decision
        decision = self._make_decision(metrics_a, metrics_b, improvements)

        comparison_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_a_metrics': metrics_a,
            'model_b_metrics': metrics_b,
            'improvements': improvements,
            'decision': decision['action'],
            'reason': decision['reason'],
            'confidence': decision['confidence']
        }

        return comparison_results

    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_type: str
    ) -> Dict:
        """Evaluate a single model."""
        model.eval()
        model = model.to(self.device)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                if model_type == 'two_tower':
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    targets = batch['target']
                    outputs = model(user_features, item_features)
                else:
                    features = batch['features'].to(self.device)
                    targets = batch['target']
                    outputs = model(features)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # Calculate metrics
        metrics = {
            'auc': float(roc_auc_score(targets, predictions)),
            'log_loss': float(log_loss(targets, predictions))
        }

        return metrics

    def _make_decision(
        self,
        metrics_a: Dict,
        metrics_b: Dict,
        improvements: Dict
    ) -> Dict:
        """
        Make promotion decision based on metrics.

        Decision Logic:
        - If AUC improvement >= min_threshold → PROMOTE
        - If AUC degradation > max_degradation → REJECT
        - If small improvement (0-1%) → A_B_TEST
        - Otherwise → REJECT
        """
        auc_improvement = improvements.get('auc', 0)

        # Strong improvement → promote
        if auc_improvement >= self.config['min_auc_improvement']:
            return {
                'action': 'PROMOTE',
                'reason': f"AUC improved by {auc_improvement:.4f} (>= {self.config['min_auc_improvement']:.4f})",
                'confidence': 'high'
            }

        # Significant degradation → reject
        if auc_improvement < -self.config['max_performance_degradation']:
            return {
                'action': 'REJECT',
                'reason': f"AUC degraded by {abs(auc_improvement):.4f} (> threshold {self.config['max_performance_degradation']:.4f})",
                'confidence': 'high'
            }

        # Small improvement → A/B test
        if 0 < auc_improvement < self.config['min_auc_improvement']:
            return {
                'action': 'A_B_TEST',
                'reason': f"Small AUC improvement ({auc_improvement:.4f}), recommend A/B test",
                'confidence': 'medium'
            }

        # No clear winner → keep current
        return {
            'action': 'REJECT',
            'reason': f"No significant improvement (AUC change: {auc_improvement:.4f})",
            'confidence': 'medium'
        }


def main():
    """Test model comparison."""
    import yaml
    from src.models.ranker import create_model
    from src.models.trainer import RankingDataset

    # Load test data
    test_df = pd.read_parquet("data/features/test_features.parquet")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']
    test_dataset = RankingDataset(test_df, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Load production model
    if model_type == 'two_tower':
        feature_dims = {
            'user': test_dataset.user_features.shape[1],
            'item': test_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {'total': test_dataset.features.shape[1]}

    model_prod = create_model(config, feature_dims)
    model_prod.load_state_dict(torch.load("models/production/model.pt"))

    # Load new model (for demo, same as prod)
    model_new = create_model(config, feature_dims)
    model_new.load_state_dict(torch.load("models/training/best_model.pt"))

    # Compare
    comparator = ModelComparator()
    results = comparator.compare_models(model_prod, model_new, test_loader, model_type)

    print("\n📊 Model Comparison Results:")
    print(f"Decision: {results['decision']}")
    print(f"Reason: {results['reason']}")
    print(f"Confidence: {results['confidence']}")

    print(f"\nProduction Model AUC: {results['model_a_metrics']['auc']:.4f}")
    print(f"New Model AUC: {results['model_b_metrics']['auc']:.4f}")
    print(f"Improvement: {results['improvements']['auc']:.4f}")

    # Save results
    with open("models/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

## Step 2: Implement Model Promotion

**File**: `src/models/promoter.py`

```python
"""
Model promotion logic with safety checks.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import mlflow


class ModelPromoter:
    """Safely promote models to production."""

    def __init__(self):
        self.production_dir = Path("models/production")
        self.backup_dir = Path("models/backups")
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def promote_model(
        self,
        model_path: str,
        comparison_results: Dict,
        force: bool = False
    ) -> Dict:
        """
        Promote model to production.

        Args:
            model_path: Path to new model checkpoint
            comparison_results: Results from ModelComparator
            force: Force promotion even if decision is not PROMOTE

        Returns:
            Dict with promotion results
        """
        decision = comparison_results['decision']

        if decision != 'PROMOTE' and not force:
            return {
                'promoted': False,
                'reason': f"Decision was {decision}, not PROMOTE. Use force=True to override.",
                'timestamp': datetime.utcnow().isoformat()
            }

        # Safety check: backup current production model
        self._backup_current_model()

        # Promote new model
        self._copy_new_model(model_path)

        # Update metadata
        metadata = self._create_metadata(comparison_results)
        self._save_metadata(metadata)

        # Update MLflow registry
        self._update_mlflow_registry()

        print("✅ Model promoted to production!")

        return {
            'promoted': True,
            'reason': comparison_results['reason'],
            'timestamp': datetime.utcnow().isoformat(),
            'backup_created': True,
            'metadata': metadata
        }

    def _backup_current_model(self):
        """Backup current production model."""
        prod_model = self.production_dir / "model.pt"

        if prod_model.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"model_{timestamp}.pt"

            shutil.copy(prod_model, backup_path)
            print(f"Backed up current model to {backup_path}")

    def _copy_new_model(self, model_path: str):
        """Copy new model to production."""
        src = Path(model_path)
        dst = self.production_dir / "model.pt"

        shutil.copy(src, dst)
        print(f"Copied new model from {src} to {dst}")

    def _create_metadata(self, comparison_results: Dict) -> Dict:
        """Create production model metadata."""
        return {
            'model_version': f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'promoted_at': datetime.utcnow().isoformat(),
            'metrics': comparison_results['model_b_metrics'],
            'decision': comparison_results['decision'],
            'reason': comparison_results['reason']
        }

    def _save_metadata(self, metadata: Dict):
        """Save production model metadata."""
        metadata_path = self.production_dir / "metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    def _update_mlflow_registry(self):
        """Update MLflow model registry (mark as production)."""
        try:
            client = mlflow.tracking.MlflowClient()

            # Get latest version of registered model
            model_name = "feed_ranker"
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            if latest_versions:
                latest_version = latest_versions[0].version

                # Transition to production
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production"
                )

                print(f"Updated MLflow registry: version {latest_version} → Production")

        except Exception as e:
            print(f"Warning: Could not update MLflow registry: {e}")

    def rollback(self, backup_timestamp: str = None) -> Dict:
        """
        Rollback to previous model.

        Args:
            backup_timestamp: Specific backup to restore (format: YYYYMMDD_HHMMSS)
                            If None, uses most recent backup

        Returns:
            Dict with rollback results
        """
        # Find backup to restore
        if backup_timestamp:
            backup_path = self.backup_dir / f"model_{backup_timestamp}.pt"
        else:
            # Get most recent backup
            backups = sorted(self.backup_dir.glob("model_*.pt"), reverse=True)
            if not backups:
                return {
                    'success': False,
                    'reason': 'No backups found'
                }
            backup_path = backups[0]

        if not backup_path.exists():
            return {
                'success': False,
                'reason': f'Backup not found: {backup_path}'
            }

        # Backup current (failed) model
        self._backup_current_model()

        # Restore backup
        prod_model = self.production_dir / "model.pt"
        shutil.copy(backup_path, prod_model)

        print(f"✅ Rolled back to {backup_path}")

        return {
            'success': True,
            'restored_from': str(backup_path),
            'timestamp': datetime.utcnow().isoformat()
        }


def main():
    """Test model promotion."""
    # Load comparison results
    with open("models/comparison_results.json", 'r') as f:
        comparison_results = json.load(f)

    # Promote model
    promoter = ModelPromoter()
    result = promoter.promote_model(
        model_path="models/training/best_model.pt",
        comparison_results=comparison_results,
        force=True  # Force for demo
    )

    print(f"\nPromotion result: {result}")


if __name__ == "__main__":
    main()
```

## Step 3: Build Retraining DAG

**File**: `dags/retraining_dag.py`

```python
"""
Automated Retraining DAG.

Triggers:
1. Scheduled (weekly)
2. Drift detection alert
3. Performance degradation

Workflow:
1. Check trigger conditions
2. Fetch recent data
3. Run ETL pipeline
4. Train new model
5. Evaluate new model
6. Compare with production
7. Promote if better
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')

import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.models.ranker import create_model
from src.models.trainer import RankingDataset, RankingTrainer
from src.models.comparator import ModelComparator
from src.models.promoter import ModelPromoter


default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_trigger_conditions(**context):
    """
    Check if retraining should proceed.

    Reasons to retrain:
    1. Scheduled time
    2. Drift detected
    3. Performance degraded
    """
    print("Checking retraining trigger conditions...")

    # Check if drift detected
    try:
        with open("data/monitoring/drift_report.json", 'r') as f:
            drift_results = json.load(f)

        drift_detected = drift_results['summary']['drift_detected']
    except:
        drift_detected = False

    # Check if alerts exist
    try:
        with open("data/monitoring/alerts.json", 'r') as f:
            alerts = json.load(f)

        has_alerts = len(alerts) > 0
    except:
        has_alerts = False

    # Determine if should retrain
    should_retrain = drift_detected or has_alerts

    print(f"Drift detected: {drift_detected}")
    print(f"Has alerts: {has_alerts}")
    print(f"Should retrain: {should_retrain}")

    if not should_retrain:
        print("⏭️  Skipping retraining - no triggers detected")

    context['ti'].xcom_push(key='should_retrain', value=should_retrain)

    return should_retrain


def fetch_recent_data(**context):
    """Fetch recent data for retraining."""
    print("Fetching recent data...")

    # In production, this would fetch data from the last N days
    # For this project, we'll use existing data

    # Simulate fetching "new" data
    # In reality, you'd query your data warehouse/lake
    raw_data = pd.read_csv("data/raw/interactions.csv")

    print(f"Fetched {len(raw_data)} recent interactions")


def trigger_etl(**context):
    """Trigger ETL pipeline to process recent data."""
    print("Triggering ETL pipeline...")

    # In production, this would trigger the ETL DAG
    # For this project, we'll assume ETL has already run

    print("ETL pipeline triggered (simulated)")


def train_new_model(**context):
    """Train new model on recent data."""
    print("Training new model...")

    # Load data
    train_df = pd.read_parquet("data/features/train_features.parquet")
    val_df = pd.read_parquet("data/features/val_features.parquet")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']

    # Create datasets
    train_dataset = RankingDataset(train_df, model_type=model_type)
    val_dataset = RankingDataset(val_df, model_type=model_type)

    # Create loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    if model_type == 'two_tower':
        feature_dims = {
            'user': train_dataset.user_features.shape[1],
            'item': train_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {'total': train_dataset.features.shape[1]}

    model = create_model(config, feature_dims)

    # Train (with warm start from previous model - optional)
    trainer = RankingTrainer()
    history = trainer.train(
        model, train_loader, val_loader,
        experiment_name="feed_ranking_retraining"
    )

    print("New model training complete!")


def compare_models(**context):
    """Compare new model with production model."""
    print("Comparing new model with production...")

    # Load test data
    test_df = pd.read_parquet("data/features/test_features.parquet")

    # Load config
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']
    test_dataset = RankingDataset(test_df, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Load models
    if model_type == 'two_tower':
        feature_dims = {
            'user': test_dataset.user_features.shape[1],
            'item': test_dataset.item_features.shape[1]
        }
    else:
        feature_dims = {'total': test_dataset.features.shape[1]}

    # Production model
    model_prod = create_model(config, feature_dims)
    try:
        model_prod.load_state_dict(torch.load("models/production/model.pt"))
    except:
        print("Warning: No production model found, using training model as baseline")
        model_prod.load_state_dict(torch.load("models/training/best_model.pt"))

    # New model
    model_new = create_model(config, feature_dims)
    model_new.load_state_dict(torch.load("models/training/best_model.pt"))

    # Compare
    comparator = ModelComparator()
    results = comparator.compare_models(model_prod, model_new, test_loader, model_type)

    # Save results
    with open("models/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Push decision to XCom
    context['ti'].xcom_push(key='promotion_decision', value=results['decision'])

    print(f"\nComparison complete!")
    print(f"Decision: {results['decision']}")
    print(f"Reason: {results['reason']}")


def promote_model(**context):
    """Promote new model if comparison was positive."""
    ti = context['ti']
    decision = ti.xcom_pull(key='promotion_decision', task_ids='compare_models')

    print(f"Promotion decision: {decision}")

    if decision == 'PROMOTE':
        # Load comparison results
        with open("models/comparison_results.json", 'r') as f:
            comparison_results = json.load(f)

        # Promote
        promoter = ModelPromoter()
        result = promoter.promote_model(
            model_path="models/training/best_model.pt",
            comparison_results=comparison_results
        )

        print(f"Promotion result: {result}")

        # TODO: Restart serving API to load new model
        # In production: send signal to API, use blue-green deployment, etc.

    elif decision == 'A_B_TEST':
        print("⚠️  Recommended A/B test - manual intervention required")
        # In production: set up A/B test infrastructure

    else:
        print("❌ Model not promoted - keeping current production model")


with DAG(
    'automated_retraining',
    default_args=default_args,
    description='Automated model retraining and promotion',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday midnight
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=['retraining', 'automation'],
) as dag:

    check_trigger_task = PythonOperator(
        task_id='check_trigger_conditions',
        python_callable=check_trigger_conditions,
    )

    fetch_data_task = PythonOperator(
        task_id='fetch_recent_data',
        python_callable=fetch_recent_data,
    )

    trigger_etl_task = PythonOperator(
        task_id='trigger_etl',
        python_callable=trigger_etl,
    )

    train_task = PythonOperator(
        task_id='train_new_model',
        python_callable=train_new_model,
    )

    compare_task = PythonOperator(
        task_id='compare_models',
        python_callable=compare_models,
    )

    promote_task = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model,
    )

    # Dependencies
    check_trigger_task >> fetch_data_task >> trigger_etl_task >> train_task >> compare_task >> promote_task
```

## Step 4: A/B Testing Framework (Conceptual)

**File**: `docs/ab_testing_design.md`

```markdown
# A/B Testing Framework Design

## Overview

When model improvement is marginal (e.g., 0.5% AUC gain), offline metrics may not
reflect real user impact. A/B testing measures business metrics in production.

## Architecture

```
User Request
    ↓
[Traffic Split]
    ↓           ↓
Model A     Model B
(50%)       (50%)
    ↓           ↓
[Log Assignments]
    ↓
[Measure Business Metrics]
    ↓
[Statistical Significance Test]
    ↓
[Promote Winner]
```

## Implementation Steps

1. **Traffic Splitting**:
   - Hash user_id → assign to A or B
   - Ensure stable assignments (same user always gets same model)

2. **Assignment Logging**:
   - Log: user_id, model_version, timestamp
   - Join with outcome data (clicks, engagement)

3. **Metric Collection**:
   - Primary: CTR, engagement rate
   - Secondary: Session length, retention

4. **Statistical Testing**:
   - Two-proportion z-test for CTR
   - Minimum sample size: 10,000 users per variant
   - Significance level: α = 0.05

5. **Decision Criteria**:
   - If B significantly better → Promote B
   - If no significant difference → Keep A (simpler/cheaper)
   - If B significantly worse → Reject B

## Code Sketch

```python
def assign_model_version(user_id):
    """Stable assignment based on user_id hash."""
    hash_val = hash(str(user_id))
    return 'model_b' if hash_val % 2 == 0 else 'model_a'

def analyze_ab_test(assignments_df, outcomes_df):
    """Analyze A/B test results."""
    # Join assignments with outcomes
    data = assignments_df.merge(outcomes_df, on='user_id')

    # Calculate metrics per variant
    metrics_a = data[data['variant'] == 'model_a']['click'].mean()
    metrics_b = data[data['variant'] == 'model_b']['click'].mean()

    # Statistical test
    from statsmodels.stats.proportion import proportions_ztest
    counts = [
        data[data['variant'] == 'model_a']['click'].sum(),
        data[data['variant'] == 'model_b']['click'].sum()
    ]
    nobs = [
        len(data[data['variant'] == 'model_a']),
        len(data[data['variant'] == 'model_b'])
    ]

    z_stat, p_value = proportions_ztest(counts, nobs)

    return {
        'ctr_a': metrics_a,
        'ctr_b': metrics_b,
        'lift': (metrics_b - metrics_a) / metrics_a,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```
```

## Testing

```bash
cd capstone_project

# 1. Test model comparison
python src/models/comparator.py

# 2. Test model promotion
python src/models/promoter.py

# 3. Test retraining DAG
airflow dags test automated_retraining 2024-11-15

# 4. Check results
cat models/comparison_results.json
cat models/production/metadata.json
```

## Review Checklist

- [ ] Model comparison logic works
- [ ] Promotion creates backups
- [ ] Rollback functionality works
- [ ] Retraining DAG runs successfully
- [ ] Models promoted only when better
- [ ] Metadata saved correctly
- [ ] MLflow registry updated
- [ ] A/B testing design documented

## What to Submit

1. **Code**:
   - `src/models/comparator.py`
   - `src/models/promoter.py`
   - `dags/retraining_dag.py`

2. **Results**:
   - Model comparison report
   - Promotion logs
   - Production model metadata

3. **Design Document**: A/B testing framework

4. **Reflection**:
   - How did you ensure safe promotions?
   - What additional safety checks would you add?
   - How would you implement true A/B testing?

## Next Steps

You've now built the complete retraining loop! Proceed to [Module 7: Final Integration](module7_integration.md) to tie everything together.
