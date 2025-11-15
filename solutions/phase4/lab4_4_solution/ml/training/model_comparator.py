"""
Model comparison and validation.

Compares new model against baseline to decide if it should be promoted.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare models and determine if new model should replace current."""

    def __init__(
        self,
        primary_metric: str = 'auc',
        improvement_threshold: float = 0.02,
        statistical_significance_threshold: float = 0.05
    ):
        """Initialize model comparator.

        Args:
            primary_metric: Primary metric for comparison (auc, f1, accuracy)
            improvement_threshold: Minimum improvement required (e.g., 0.02 = 2%)
            statistical_significance_threshold: P-value threshold for significance
        """
        self.primary_metric = primary_metric
        self.improvement_threshold = improvement_threshold
        self.significance_threshold = statistical_significance_threshold

    def evaluate_model(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            model: PyTorch model
            X: Feature matrix
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        model.eval()

        # Get predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred_proba = model(X_tensor).numpy().squeeze()

        # Binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
        }

        return metrics

    def compare_models(
        self,
        baseline_model: torch.nn.Module,
        new_model: torch.nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compare baseline and new models.

        Args:
            baseline_model: Current production model
            new_model: Newly trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Comparison results with promotion decision
        """
        logger.info("Comparing baseline and new models")

        # Evaluate both models
        baseline_metrics = self.evaluate_model(baseline_model, X_test, y_test)
        new_metrics = self.evaluate_model(new_model, X_test, y_test)

        # Calculate improvements
        improvements = {}
        for metric in baseline_metrics:
            baseline_val = baseline_metrics[metric]
            new_val = new_metrics[metric]

            # For log_loss, lower is better
            if metric == 'log_loss':
                improvement = baseline_val - new_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            else:
                improvement = new_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

            improvements[metric] = {
                'baseline': float(baseline_val),
                'new': float(new_val),
                'absolute_improvement': float(improvement),
                'percent_improvement': float(improvement_pct)
            }

        # Check primary metric improvement
        primary_improvement = improvements[self.primary_metric]['absolute_improvement']
        meets_threshold = primary_improvement >= self.improvement_threshold

        # Statistical significance (simple bootstrap test)
        is_significant = self._bootstrap_significance_test(
            baseline_model, new_model, X_test, y_test, self.primary_metric
        )

        # Promotion decision
        should_promote = meets_threshold and is_significant

        # Compile results
        results = {
            'baseline_metrics': baseline_metrics,
            'new_metrics': new_metrics,
            'improvements': improvements,
            'primary_metric': self.primary_metric,
            'primary_improvement': float(primary_improvement),
            'improvement_threshold': self.improvement_threshold,
            'meets_threshold': meets_threshold,
            'statistically_significant': is_significant,
            'should_promote': should_promote,
            'promotion_reason': self._get_promotion_reason(
                meets_threshold, is_significant, primary_improvement
            )
        }

        # Log results
        self._log_comparison_results(results)

        return results

    def _bootstrap_significance_test(
        self,
        baseline_model: torch.nn.Module,
        new_model: torch.nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric: str,
        n_bootstrap: int = 100
    ) -> bool:
        """Perform bootstrap significance test.

        Args:
            baseline_model: Baseline model
            new_model: New model
            X_test: Test features
            y_test: Test labels
            metric: Metric to test
            n_bootstrap: Number of bootstrap samples

        Returns:
            True if difference is statistically significant
        """
        logger.info(f"Running bootstrap significance test ({n_bootstrap} samples)")

        baseline_model.eval()
        new_model.eval()

        # Get predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            baseline_preds = baseline_model(X_tensor).numpy().squeeze()
            new_preds = new_model(X_tensor).numpy().squeeze()

        differences = []
        n_samples = len(y_test)

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_boot = y_test[indices]
            baseline_boot = baseline_preds[indices]
            new_boot = new_preds[indices]

            # Calculate metric
            if metric == 'auc':
                baseline_score = roc_auc_score(y_boot, baseline_boot)
                new_score = roc_auc_score(y_boot, new_boot)
            elif metric == 'log_loss':
                baseline_score = log_loss(y_boot, baseline_boot)
                new_score = log_loss(y_boot, new_boot)
                # For log_loss, lower is better, so reverse the difference
                differences.append(baseline_score - new_score)
                continue
            else:
                baseline_score = accuracy_score(y_boot, (baseline_boot >= 0.5).astype(int))
                new_score = accuracy_score(y_boot, (new_boot >= 0.5).astype(int))

            differences.append(new_score - baseline_score)

        # Check if improvement is consistently positive
        p_value = (np.array(differences) <= 0).mean()
        is_significant = p_value < self.significance_threshold

        logger.info(f"Bootstrap p-value: {p_value:.4f} (threshold: {self.significance_threshold})")

        return is_significant

    def _get_promotion_reason(
        self,
        meets_threshold: bool,
        is_significant: bool,
        improvement: float
    ) -> str:
        """Get human-readable promotion reason."""
        if meets_threshold and is_significant:
            return f"New model shows {improvement:.4f} improvement in {self.primary_metric} (threshold: {self.improvement_threshold}) and is statistically significant"
        elif not meets_threshold:
            return f"Improvement ({improvement:.4f}) does not meet threshold ({self.improvement_threshold})"
        elif not is_significant:
            return "Improvement is not statistically significant"
        else:
            return "Unknown reason"

    def _log_comparison_results(self, results: Dict[str, Any]):
        """Log comparison results."""
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nPrimary Metric: {results['primary_metric'].upper()}")
        logger.info(f"Baseline: {results['baseline_metrics'][results['primary_metric']]:.4f}")
        logger.info(f"New:      {results['new_metrics'][results['primary_metric']]:.4f}")
        logger.info(f"Improvement: {results['primary_improvement']:.4f} ({results['improvements'][results['primary_metric']]['percent_improvement']:.2f}%)")

        logger.info(f"\nAll Metrics:")
        for metric, values in results['improvements'].items():
            logger.info(f"  {metric:12s}: {values['baseline']:.4f} → {values['new']:.4f} ({values['percent_improvement']:+.2f}%)")

        logger.info(f"\nDecision Factors:")
        logger.info(f"  Meets threshold:          {results['meets_threshold']}")
        logger.info(f"  Statistically significant: {results['statistically_significant']}")

        logger.info(f"\nPROMOTION DECISION: {'PROMOTE' if results['should_promote'] else 'DO NOT PROMOTE'}")
        logger.info(f"Reason: {results['promotion_reason']}")

        logger.info("=" * 80)

    def save_comparison_report(self, results: Dict[str, Any], output_path: str):
        """Save comparison report to JSON.

        Args:
            results: Comparison results
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comparison report saved to {output_path}")


def load_model_from_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    """Load PyTorch model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model architecture
    from torch import nn

    input_dim = checkpoint.get('input_dim', 10)
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model.eval()
    return model


def compare_model_checkpoints(
    baseline_path: str,
    new_path: str,
    test_data_path: str,
    output_path: Optional[str] = None,
    primary_metric: str = 'auc',
    improvement_threshold: float = 0.02
) -> Dict[str, Any]:
    """Compare two model checkpoints.

    Args:
        baseline_path: Path to baseline model
        new_path: Path to new model
        test_data_path: Path to test data CSV
        output_path: Optional path to save report
        primary_metric: Primary comparison metric
        improvement_threshold: Minimum improvement required

    Returns:
        Comparison results
    """
    logger.info(f"Comparing models:")
    logger.info(f"  Baseline: {baseline_path}")
    logger.info(f"  New:      {new_path}")

    # Load models
    baseline_model = load_model_from_checkpoint(baseline_path)
    new_model = load_model_from_checkpoint(new_path)

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Assuming last column is target
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)

    # Create comparator
    comparator = ModelComparator(
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold
    )

    # Compare models
    results = comparator.compare_models(
        baseline_model, new_model, X_test, y_test
    )

    # Save report if path provided
    if output_path:
        comparator.save_comparison_report(results, output_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--baseline", required=True, help="Baseline model path")
    parser.add_argument("--new", required=True, help="New model path")
    parser.add_argument("--test-data", required=True, help="Test data CSV")
    parser.add_argument("--output", help="Output report path")
    parser.add_argument("--metric", default="auc", help="Primary metric")
    parser.add_argument("--threshold", type=float, default=0.02, help="Improvement threshold")

    args = parser.parse_args()

    results = compare_model_checkpoints(
        baseline_path=args.baseline,
        new_path=args.new,
        test_data_path=args.test_data,
        output_path=args.output,
        primary_metric=args.metric,
        improvement_threshold=args.threshold
    )

    # Exit code based on promotion decision
    exit(0 if results['should_promote'] else 1)
