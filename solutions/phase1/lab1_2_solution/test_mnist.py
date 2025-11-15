"""
Test a saved MNIST model - SOLUTION
Load a trained model checkpoint and evaluate on test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from pathlib import Path
import sys

# Import the model from train_mnist
from train_mnist import SimpleCNN, get_dataloaders


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        tuple: (model, checkpoint_dict)
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize model
    model = SimpleCNN().to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Validation Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  - Validation Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")

    return model, checkpoint


def test_model(model, test_loader, device):
    """
    Test the model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to test on

    Returns:
        dict: Test metrics
    """
    model.eval()
    correct = 0
    total = 0

    # Track per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10

    print("\nTesting model...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Calculate overall accuracy
    accuracy = 100. * correct / total

    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            class_accuracies[i] = class_acc

    return {
        'overall_accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies
    }


def print_results(results):
    """
    Print test results in a formatted way.

    Args:
        results: Dictionary of test results
    """
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct predictions: {results['correct']}/{results['total']}")

    print(f"\nPer-Class Accuracy:")
    print(f"{'-'*40}")
    for digit in range(10):
        acc = results['class_accuracies'].get(digit, 0)
        print(f"  Digit {digit}: {acc:.2f}%")

    print(f"\n{'='*60}")

    # Check if model meets target accuracy
    if results['overall_accuracy'] >= 97.0:
        print("✓ Model meets target accuracy (>97%)")
    else:
        print("✗ Model below target accuracy (<97%)")
    print(f"{'='*60}")


def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    Visualize some predictions (prints predictions vs actual labels).

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device
        num_samples: Number of samples to show
    """
    model.eval()
    print(f"\nSample Predictions (first {num_samples}):")
    print(f"{'-'*40}")

    with torch.no_grad():
        # Get one batch
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)

        # Show predictions
        for i in range(min(num_samples, len(target))):
            actual = target[i].item()
            pred = predicted[i].item()
            status = "✓" if actual == pred else "✗"
            print(f"  Sample {i+1}: Predicted={pred}, Actual={actual} {status}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test a trained MNIST model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./models/staging/mnist/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing MNIST data'
    )
    parser.add_argument(
        '--show-samples',
        type=int,
        default=10,
        help='Number of sample predictions to show'
    )
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST test set...")
    _, test_loader = get_dataloaders(args.batch_size, args.data_dir)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    try:
        model, checkpoint = load_model(args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've trained a model first:")
        print("  python train_mnist.py")
        sys.exit(1)

    # Test model
    results = test_model(model, test_loader, device)

    # Print results
    print_results(results)

    # Show some predictions
    if args.show_samples > 0:
        visualize_predictions(model, test_loader, device, args.show_samples)


if __name__ == "__main__":
    main()
