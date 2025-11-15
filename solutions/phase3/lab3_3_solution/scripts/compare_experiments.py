"""Compare experiment runs."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.tracking.experiment_tracker import ExperimentTracker
import pandas as pd


def compare_experiments(experiment_name: str):
    """Compare all runs for an experiment."""
    tracker = ExperimentTracker()

    # Get all runs
    runs = tracker.list_runs(experiment_name)

    if not runs:
        print(f"No runs found for experiment: {experiment_name}")
        return

    print(f"Found {len(runs)} runs for '{experiment_name}'")

    # Get run IDs
    run_ids = [r['run_id'] for r in runs]

    # Compare
    comparison_df = tracker.compare_runs(run_ids)

    # Sort by test_auc (descending) if available
    if 'test_auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('test_auc', ascending=False)
    elif 'val_auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('val_auc', ascending=False)

    # Display
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPARISON: {experiment_name}")
    print("=" * 80)

    # Select columns to display
    display_cols = ['run_id', 'status']

    # Add param columns
    param_cols = [c for c in comparison_df.columns if c not in ['run_id', 'experiment_name', 'status', 'created_at']]
    metric_cols = [c for c in param_cols if c.startswith('test_') or c.startswith('val_') or c.endswith('_loss') or c.endswith('_acc') or c.endswith('_auc')]
    param_only_cols = [c for c in param_cols if c not in metric_cols]

    # Prioritize display: run_id, status, params, metrics
    display_cols.extend(param_only_cols[:5])  # Show up to 5 params
    display_cols.extend(metric_cols[:5])  # Show up to 5 metrics

    display_cols = [c for c in display_cols if c in comparison_df.columns]

    # Set display options for pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(comparison_df[display_cols].to_string(index=False))

    # Show best run
    print("\n" + "=" * 80)
    print("BEST RUN:")
    print("=" * 80)

    # Determine which metric to use for best
    best_metric = None
    if 'test_auc' in comparison_df.columns:
        best_metric = 'test_auc'
    elif 'val_auc' in comparison_df.columns:
        best_metric = 'val_auc'
    elif 'test_accuracy' in comparison_df.columns:
        best_metric = 'test_accuracy'

    if best_metric:
        best_run = tracker.get_best_run(best_metric, experiment_name)
        if best_run:
            print(f"Run ID: {best_run['run_id']}")
            print(f"Best {best_metric}: {best_run['metrics'][best_metric][-1]['value']:.4f}")

            print(f"\nParameters:")
            for key, value in best_run['params'].items():
                print(f"  {key}: {value}")

            print(f"\nFinal Metrics:")
            for metric_name, metric_history in best_run['metrics'].items():
                if metric_history:
                    final_value = metric_history[-1]['value']
                    print(f"  {metric_name}: {final_value:.4f}")

    return comparison_df


def plot_metric_comparison(run_ids: list, metric: str):
    """Plot metric across epochs for multiple runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return

    tracker = ExperimentTracker()

    plt.figure(figsize=(10, 6))

    for run_id in run_ids:
        run = tracker.get_run(run_id)
        if not run or metric not in run['metrics']:
            print(f"⚠️  Metric '{metric}' not found in run {run_id}")
            continue

        metric_history = run['metrics'][metric]
        steps = [m['step'] for m in metric_history if m['step'] is not None]
        values = [m['value'] for m in metric_history if m['step'] is not None]

        if not steps:
            print(f"⚠️  No step data for metric '{metric}' in run {run_id}")
            continue

        # Get learning rate for label
        lr = run['params'].get('learning_rate', 'unknown')
        label = f"{run_id} (lr={lr})"

        plt.plot(steps, values, marker='o', label=label)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison Across Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = f'experiments/metric_comparison_{metric}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to {plot_path}")

    plt.show()


def list_all_experiments():
    """List all experiments and their runs."""
    tracker = ExperimentTracker()
    all_runs = tracker.list_runs()

    if not all_runs:
        print("No experiments found.")
        return

    # Group by experiment name
    experiments = {}
    for run in all_runs:
        exp_name = run['experiment_name']
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(run)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS")
    print("=" * 80)

    for exp_name, runs in experiments.items():
        completed = sum(1 for r in runs if r['status'] == 'completed')
        print(f"\n{exp_name}:")
        print(f"  Total runs: {len(runs)}")
        print(f"  Completed: {completed}")
        print(f"  Run IDs: {', '.join([r['run_id'] for r in runs[:5]])}", end='')
        if len(runs) > 5:
            print(f"... (+{len(runs) - 5} more)")
        else:
            print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/compare_experiments.py <experiment_name>")
        print("  python scripts/compare_experiments.py ctr_model")
        print("\nOr to plot a specific metric:")
        print("  python scripts/compare_experiments.py <experiment_name> plot <metric> <run_ids...>")
        print("  python scripts/compare_experiments.py ctr_model plot val_auc run_001 run_002")
        print("\nOr to list all experiments:")
        print("  python scripts/compare_experiments.py --list")
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_all_experiments()
    elif len(sys.argv) > 2 and sys.argv[2] == 'plot':
        if len(sys.argv) < 5:
            print("Error: plot requires experiment_name, metric, and at least one run_id")
            print("Usage: python scripts/compare_experiments.py <experiment_name> plot <metric> <run_ids...>")
            sys.exit(1)
        experiment_name = sys.argv[1]
        metric = sys.argv[3]
        run_ids = sys.argv[4:]
        plot_metric_comparison(run_ids, metric)
    else:
        experiment_name = sys.argv[1]
        compare_experiments(experiment_name)
