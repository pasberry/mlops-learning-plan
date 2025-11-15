"""
Compare multiple experiments - SOLUTION
Load and compare results from multiple training runs.
"""

import json
from pathlib import Path
import pandas as pd
import argparse
import sys


def load_experiment_results(base_dir='./models/staging/mnist'):
    """
    Load all experiment results from a base directory.

    Args:
        base_dir: Base directory containing experiment subdirectories

    Returns:
        pandas.DataFrame: DataFrame with experiment results
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory not found: {base_dir}")
        return pd.DataFrame()

    results = []

    # Iterate through experiment directories
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / 'experiment_summary.json'

            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        data = json.load(f)

                    # Extract relevant information
                    result = {
                        'name': data['experiment_name'],
                        'best_acc': data['final_metrics']['best_val_acc'],
                        'epochs': data['final_metrics']['final_epoch'],
                        'lr': data['config']['training']['learning_rate'],
                        'batch_size': data['config']['training']['batch_size'],
                        'optimizer': data['config']['training']['optimizer'],
                        'conv1_ch': data['config']['model']['architecture']['conv1_channels'],
                        'conv2_ch': data['config']['model']['architecture']['conv2_channels'],
                        'fc1_size': data['config']['model']['architecture']['fc1_size'],
                        'dropout_conv': data['config']['model']['architecture']['dropout_conv'],
                        'dropout_fc': data['config']['model']['architecture']['dropout_fc'],
                        'timestamp': data.get('timestamp', 'N/A'),
                    }
                    results.append(result)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse {summary_file}: {e}")

    if not results:
        print(f"No experiment results found in {base_dir}")
        return pd.DataFrame()

    # Create DataFrame and sort by accuracy
    df = pd.DataFrame(results)
    df = df.sort_values('best_acc', ascending=False)

    return df


def print_comparison_table(df, show_all=False):
    """
    Print experiment comparison table.

    Args:
        df: DataFrame with experiment results
        show_all: Whether to show all columns
    """
    if df.empty:
        print("No experiments to compare.")
        return

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*100}\n")

    if show_all:
        # Show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
    else:
        # Show key columns only
        key_columns = ['name', 'best_acc', 'epochs', 'lr', 'batch_size', 'optimizer']
        if all(col in df.columns for col in key_columns):
            print(df[key_columns].to_string(index=False))
        else:
            print(df.to_string(index=False))

    print(f"\n{'='*100}")
    print(f"Total experiments: {len(df)}")
    print(f"Best accuracy: {df['best_acc'].max():.2f}% ({df.iloc[0]['name']})")
    print(f"{'='*100}\n")


def print_statistics(df):
    """
    Print summary statistics.

    Args:
        df: DataFrame with experiment results
    """
    if df.empty:
        return

    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}\n")

    print(f"Accuracy Statistics:")
    print(f"  Mean: {df['best_acc'].mean():.2f}%")
    print(f"  Std:  {df['best_acc'].std():.2f}%")
    print(f"  Min:  {df['best_acc'].min():.2f}%")
    print(f"  Max:  {df['best_acc'].max():.2f}%")

    print(f"\nEpochs Statistics:")
    print(f"  Mean: {df['epochs'].mean():.1f}")
    print(f"  Min:  {df['epochs'].min()}")
    print(f"  Max:  {df['epochs'].max()}")

    print(f"\nOptimizer Distribution:")
    print(df['optimizer'].value_counts().to_string())

    print(f"\nBatch Size Distribution:")
    print(df['batch_size'].value_counts().sort_index().to_string())

    print(f"\n{'='*100}\n")


def find_best_config(df, metric='best_acc'):
    """
    Find best configuration based on a metric.

    Args:
        df: DataFrame with experiment results
        metric: Metric to optimize

    Returns:
        dict: Best experiment configuration
    """
    if df.empty:
        return None

    best_idx = df[metric].idxmax()
    best_exp = df.loc[best_idx]

    print(f"\n{'='*100}")
    print(f"BEST CONFIGURATION (by {metric})")
    print(f"{'='*100}\n")

    for key, value in best_exp.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*100}\n")

    return best_exp.to_dict()


def compare_by_hyperparameter(df, param='lr'):
    """
    Compare experiments grouped by a hyperparameter.

    Args:
        df: DataFrame with experiment results
        param: Hyperparameter to group by
    """
    if df.empty or param not in df.columns:
        return

    print(f"\n{'='*100}")
    print(f"COMPARISON BY {param.upper()}")
    print(f"{'='*100}\n")

    grouped = df.groupby(param)['best_acc'].agg(['mean', 'std', 'min', 'max', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)

    print(grouped.to_string())
    print(f"\n{'='*100}\n")


def export_to_csv(df, output_path='experiment_comparison.csv'):
    """
    Export comparison table to CSV.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save CSV file
    """
    if df.empty:
        print("No data to export.")
        return

    df.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare ML experiment results')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='./models/staging/mnist',
        help='Base directory containing experiment results'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all columns in comparison table'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--compare-by',
        type=str,
        choices=['lr', 'batch_size', 'optimizer'],
        help='Compare experiments by hyperparameter'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    args = parser.parse_args()

    # Load experiment results
    print(f"Loading experiments from: {args.base_dir}")
    df = load_experiment_results(args.base_dir)

    if df.empty:
        print("\nNo experiments found. Train some models first:")
        print("  python train_mnist_config.py --config config.yaml")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(df, show_all=args.show_all)

    # Print statistics if requested
    if args.stats:
        print_statistics(df)

    # Find best configuration
    find_best_config(df, metric='best_acc')

    # Compare by hyperparameter if requested
    if args.compare_by:
        compare_by_hyperparameter(df, args.compare_by)

    # Export to CSV if requested
    if args.export:
        export_to_csv(df, args.export)


if __name__ == "__main__":
    main()
