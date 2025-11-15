"""
Configuration management for ML experiments - SOLUTION
Provides utilities for loading, saving, and managing configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import torch


class Config:
    """Configuration class with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Recursively converts nested dictionaries to Config objects
        for convenient dot notation access.

        Args:
            config_dict: Configuration dictionary

        Example:
            >>> config = Config({'model': {'name': 'CNN'}})
            >>> config.model.name
            'CNN'
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to Config objects
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        """
        Allow dictionary-style access.

        Args:
            key: Attribute name

        Returns:
            Attribute value

        Example:
            >>> config['model']  # Same as config.model
        """
        return getattr(self, key)

    def __repr__(self):
        """String representation of config."""
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config object back to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            >>> config = Config({'a': 1, 'b': {'c': 2}})
            >>> config.to_dict()
            {'a': 1, 'b': {'c': 2}}
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with dot notation access

    Raises:
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> config = load_config('config/mnist_config.yaml')
        >>> print(config.training.learning_rate)
        0.001
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        save_path: Path to save YAML file

    Example:
        >>> config = load_config('config.yaml')
        >>> save_config(config, 'output/config_backup.yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)

    print(f"Config saved to: {save_path}")


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch to ensure
    reproducible results across runs.

    Args:
        seed: Random seed

    Example:
        >>> set_seed(42)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Note: For full reproducibility, also set:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # However, this may reduce performance


def get_device(config: Config) -> torch.device:
    """
    Get device from config.

    Args:
        config: Config object

    Returns:
        torch.device (cuda or cpu)

    Example:
        >>> config = load_config('config.yaml')
        >>> device = get_device(config)
        >>> print(device)
        cuda:0  # or cpu
    """
    if config.device.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if config.device.use_cuda and not torch.cuda.is_available():
            print("CUDA requested but not available. Using CPU.")

    return device


def validate_config(config: Config) -> bool:
    """
    Validate configuration file.

    Checks for required fields and valid value ranges.

    Args:
        config: Config object

    Returns:
        True if config is valid

    Raises:
        ValueError: If config is invalid

    Example:
        >>> config = load_config('config.yaml')
        >>> validate_config(config)
        True
    """
    # Check required top-level fields
    required_fields = ['experiment', 'data', 'model', 'training', 'logging', 'device']
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required field: {field}")

    # Validate training parameters
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    # Validate optimizer
    valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
    if config.training.optimizer.lower() not in valid_optimizers:
        raise ValueError(f"optimizer must be one of {valid_optimizers}")

    # Validate data parameters
    if config.data.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    # Validate early stopping parameters
    if config.training.early_stopping.enabled:
        if config.training.early_stopping.patience <= 0:
            raise ValueError("early_stopping.patience must be positive")

    print("✓ Config validation passed")
    return True


# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config('config.yaml')

    # Access with dot notation
    print(f"\nExperiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Optimizer: {config.training.optimizer}")

    # Access with dictionary notation
    print(f"\nModel: {config['model']['name']}")
    print(f"Conv1 channels: {config['model']['architecture']['conv1_channels']}")

    # Validate config
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Validation error: {e}")

    # Convert to dict
    print(f"\nFull config:")
    import json
    print(json.dumps(config.to_dict(), indent=2))

    # Set random seed
    set_seed(config.device.seed)
    print(f"\nRandom seed set to: {config.device.seed}")

    # Get device
    device = get_device(config)
    print(f"Device: {device}")
