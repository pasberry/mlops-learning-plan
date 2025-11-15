"""
Data and model drift detection.

Implements statistical tests for detecting distribution shifts:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) test
- Jensen-Shannon Divergence
- Prediction drift monitoring
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and model drift."""

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        js_threshold: float = 0.1
    ):
        """Initialize drift detector.

        Args:
            psi_threshold: PSI threshold (>0.2 indicates drift)
            ks_threshold: KS test p-value threshold (<0.05 indicates drift)
            js_threshold: Jensen-Shannon divergence threshold (>0.1 indicates drift)
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Population Stability Index (PSI).

        PSI measures the shift in a variable's distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Small change
        - PSI >= 0.2: Significant change (drift detected)

        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution
            bins: Number of bins for discretization

        Returns:
            Tuple of (PSI value, detailed metrics)
        """
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Ensure we have enough bins
        if len(breakpoints) < 3:
            breakpoints = np.linspace(
                min(expected.min(), actual.min()),
                max(expected.max(), actual.max()),
                bins + 1
            )

        # Bin the data
        expected_binned = np.digitize(expected, breakpoints[1:-1])
        actual_binned = np.digitize(actual, breakpoints[1:-1])

        # Calculate proportions
        expected_counts = np.bincount(expected_binned, minlength=bins)
        actual_counts = np.bincount(actual_binned, minlength=bins)

        expected_props = expected_counts / len(expected)
        actual_props = actual_counts / len(actual)

        # Avoid division by zero
        expected_props = np.where(expected_props == 0, 0.0001, expected_props)
        actual_props = np.where(actual_props == 0, 0.0001, actual_props)

        # Calculate PSI
        psi_values = (actual_props - expected_props) * np.log(actual_props / expected_props)
        psi = np.sum(psi_values)

        # Detailed metrics
        metrics = {
            'psi': float(psi),
            'drift_detected': psi >= self.psi_threshold,
            'threshold': self.psi_threshold,
            'num_bins': len(breakpoints) - 1,
            'expected_mean': float(np.mean(expected)),
            'actual_mean': float(np.mean(actual)),
            'expected_std': float(np.std(expected)),
            'actual_std': float(np.std(actual)),
        }

        return psi, metrics

    def kolmogorov_smirnov_test(
        self,
        expected: np.ndarray,
        actual: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Perform Kolmogorov-Smirnov test.

        KS test measures the maximum distance between cumulative distributions.
        A low p-value (<0.05) indicates distributions are significantly different.

        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution

        Returns:
            Tuple of (KS statistic, p-value, detailed metrics)
        """
        ks_stat, p_value = stats.ks_2samp(expected, actual)

        metrics = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < self.ks_threshold,
            'threshold': self.ks_threshold,
            'expected_median': float(np.median(expected)),
            'actual_median': float(np.median(actual)),
        }

        return ks_stat, p_value, metrics

    def jensen_shannon_divergence(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 50
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Jensen-Shannon Divergence.

        JS divergence measures similarity between probability distributions.
        Range: [0, 1], where 0 = identical, 1 = completely different.

        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution
            bins: Number of bins for histogram

        Returns:
            Tuple of (JS divergence, detailed metrics)
        """
        # Create histograms
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bins_edges = np.linspace(min_val, max_val, bins + 1)

        expected_hist, _ = np.histogram(expected, bins=bins_edges, density=True)
        actual_hist, _ = np.histogram(actual, bins=bins_edges, density=True)

        # Normalize
        expected_hist = expected_hist / expected_hist.sum()
        actual_hist = actual_hist / actual_hist.sum()

        # Avoid zeros
        expected_hist = np.where(expected_hist == 0, 1e-10, expected_hist)
        actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)

        # Calculate JS divergence
        m = 0.5 * (expected_hist + actual_hist)
        js_div = 0.5 * stats.entropy(expected_hist, m) + 0.5 * stats.entropy(actual_hist, m)

        metrics = {
            'js_divergence': float(js_div),
            'drift_detected': js_div >= self.js_threshold,
            'threshold': self.js_threshold,
        }

        return js_div, metrics

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Detect drift for multiple features.

        Args:
            baseline_df: Baseline (reference) data
            current_df: Current (production) data
            feature_columns: List of feature columns to check

        Returns:
            Dictionary with drift metrics for each feature
        """
        logger.info(f"Detecting drift for {len(feature_columns)} features")

        results = {}

        for feature in feature_columns:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature '{feature}' not found in data, skipping")
                continue

            # Get feature values
            baseline_values = baseline_df[feature].dropna().values
            current_values = current_df[feature].dropna().values

            if len(baseline_values) == 0 or len(current_values) == 0:
                logger.warning(f"Feature '{feature}' has no valid values, skipping")
                continue

            # Calculate all drift metrics
            psi, psi_metrics = self.calculate_psi(baseline_values, current_values)
            ks_stat, p_value, ks_metrics = self.kolmogorov_smirnov_test(
                baseline_values, current_values
            )
            js_div, js_metrics = self.jensen_shannon_divergence(
                baseline_values, current_values
            )

            # Combine results
            feature_drift = {
                'feature': feature,
                'psi': psi_metrics,
                'ks_test': ks_metrics,
                'js_divergence': js_metrics,
                'drift_detected': (
                    psi_metrics['drift_detected'] or
                    ks_metrics['drift_detected'] or
                    js_metrics['drift_detected']
                ),
                'baseline_samples': len(baseline_values),
                'current_samples': len(current_values),
            }

            results[feature] = feature_drift

            # Log drift detection
            if feature_drift['drift_detected']:
                logger.warning(f"Drift detected for feature '{feature}'")
                logger.warning(f"  PSI: {psi:.4f} (threshold: {self.psi_threshold})")
                logger.warning(f"  KS p-value: {p_value:.4f} (threshold: {self.ks_threshold})")
                logger.warning(f"  JS divergence: {js_div:.4f} (threshold: {self.js_threshold})")

        # Summary statistics
        total_features = len(results)
        drifted_features = sum(1 for r in results.values() if r['drift_detected'])

        summary = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0,
            'overall_drift_detected': drifted_features > 0,
            'features': results,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"Drift detection complete: {drifted_features}/{total_features} features drifted")

        return summary

    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detect drift in model predictions.

        Args:
            baseline_predictions: Baseline prediction scores
            current_predictions: Current prediction scores

        Returns:
            Dictionary with prediction drift metrics
        """
        logger.info("Detecting prediction drift")

        # Calculate drift metrics
        psi, psi_metrics = self.calculate_psi(baseline_predictions, current_predictions)
        ks_stat, p_value, ks_metrics = self.kolmogorov_smirnov_test(
            baseline_predictions, current_predictions
        )
        js_div, js_metrics = self.jensen_shannon_divergence(
            baseline_predictions, current_predictions
        )

        # Calculate additional prediction statistics
        baseline_pos_rate = (baseline_predictions >= 0.5).mean()
        current_pos_rate = (current_predictions >= 0.5).mean()

        results = {
            'psi': psi_metrics,
            'ks_test': ks_metrics,
            'js_divergence': js_metrics,
            'drift_detected': (
                psi_metrics['drift_detected'] or
                ks_metrics['drift_detected'] or
                js_metrics['drift_detected']
            ),
            'baseline_positive_rate': float(baseline_pos_rate),
            'current_positive_rate': float(current_pos_rate),
            'positive_rate_change': float(current_pos_rate - baseline_pos_rate),
            'baseline_samples': len(baseline_predictions),
            'current_samples': len(current_predictions),
            'timestamp': datetime.utcnow().isoformat()
        }

        if results['drift_detected']:
            logger.warning("Prediction drift detected!")
            logger.warning(f"  PSI: {psi:.4f}")
            logger.warning(f"  KS p-value: {p_value:.4f}")
            logger.warning(f"  Positive rate change: {results['positive_rate_change']:.4f}")

        return results

    def generate_drift_report(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: List[str],
        baseline_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive drift report.

        Args:
            baseline_df: Baseline data
            current_df: Current data
            feature_columns: Feature columns to check
            baseline_predictions: Optional baseline predictions
            current_predictions: Optional current predictions

        Returns:
            Complete drift report
        """
        logger.info("Generating drift report")

        # Feature drift
        feature_drift = self.detect_feature_drift(
            baseline_df, current_df, feature_columns
        )

        # Prediction drift (if provided)
        prediction_drift = None
        if baseline_predictions is not None and current_predictions is not None:
            prediction_drift = self.detect_prediction_drift(
                baseline_predictions, current_predictions
            )

        # Overall assessment
        overall_drift_detected = feature_drift['overall_drift_detected']
        if prediction_drift:
            overall_drift_detected = overall_drift_detected or prediction_drift['drift_detected']

        report = {
            'overall_drift_detected': overall_drift_detected,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'baseline_period': {
                'samples': len(baseline_df),
                'features': len(feature_columns)
            },
            'current_period': {
                'samples': len(current_df),
                'features': len(feature_columns)
            },
            'thresholds': {
                'psi': self.psi_threshold,
                'ks_pvalue': self.ks_threshold,
                'js_divergence': self.js_threshold
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        # Print summary
        self._print_drift_report(report)

        return report

    def _print_drift_report(self, report: Dict[str, Any]):
        """Print formatted drift report."""
        logger.info("=" * 80)
        logger.info("DRIFT DETECTION REPORT")
        logger.info("=" * 80)

        logger.info(f"Overall Drift Detected: {'YES' if report['overall_drift_detected'] else 'NO'}")
        logger.info(f"\nBaseline: {report['baseline_period']['samples']:,} samples")
        logger.info(f"Current:  {report['current_period']['samples']:,} samples")

        # Feature drift summary
        feature_drift = report['feature_drift']
        logger.info(f"\nFeature Drift:")
        logger.info(f"  Total features: {feature_drift['total_features']}")
        logger.info(f"  Drifted features: {feature_drift['drifted_features']}")
        logger.info(f"  Drift percentage: {feature_drift['drift_percentage']:.1f}%")

        if feature_drift['drifted_features'] > 0:
            logger.info(f"\n  Drifted Features:")
            for feature, metrics in feature_drift['features'].items():
                if metrics['drift_detected']:
                    logger.info(f"    - {feature}:")
                    logger.info(f"        PSI: {metrics['psi']['psi']:.4f}")
                    logger.info(f"        KS p-value: {metrics['ks_test']['p_value']:.4f}")
                    logger.info(f"        JS divergence: {metrics['js_divergence']['js_divergence']:.4f}")

        # Prediction drift
        if report['prediction_drift']:
            pred_drift = report['prediction_drift']
            logger.info(f"\nPrediction Drift:")
            logger.info(f"  Drift detected: {'YES' if pred_drift['drift_detected'] else 'NO'}")
            logger.info(f"  PSI: {pred_drift['psi']['psi']:.4f}")
            logger.info(f"  Positive rate change: {pred_drift['positive_rate_change']:.4f}")

        logger.info("=" * 80)

    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save drift report to JSON file.

        Args:
            report: Drift report dictionary
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Drift report saved to {output_path}")


def detect_drift_from_files(
    baseline_path: str,
    current_path: str,
    feature_columns: List[str],
    output_path: Optional[str] = None,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.05
) -> Dict[str, Any]:
    """Detect drift from CSV files.

    Args:
        baseline_path: Path to baseline CSV
        current_path: Path to current CSV
        feature_columns: Feature columns to check
        output_path: Optional path to save report
        psi_threshold: PSI threshold
        ks_threshold: KS test threshold

    Returns:
        Drift report dictionary
    """
    # Load data
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    # Create detector
    detector = DriftDetector(
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold
    )

    # Generate report
    report = detector.generate_drift_report(
        baseline_df, current_df, feature_columns
    )

    # Save report if path provided
    if output_path:
        detector.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect data and model drift")
    parser.add_argument("--baseline", required=True, help="Baseline data CSV")
    parser.add_argument("--current", required=True, help="Current data CSV")
    parser.add_argument("--features", required=True, nargs='+', help="Feature columns")
    parser.add_argument("--output", help="Output report path")
    parser.add_argument("--psi-threshold", type=float, default=0.2, help="PSI threshold")
    parser.add_argument("--ks-threshold", type=float, default=0.05, help="KS test threshold")

    args = parser.parse_args()

    report = detect_drift_from_files(
        baseline_path=args.baseline,
        current_path=args.current,
        feature_columns=args.features,
        output_path=args.output,
        psi_threshold=args.psi_threshold,
        ks_threshold=args.ks_threshold
    )

    if report['overall_drift_detected']:
        exit(1)  # Exit with error if drift detected
    else:
        exit(0)
