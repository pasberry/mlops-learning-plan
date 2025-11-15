"""Base model interface for all PyTorch models."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models.

    All models should inherit from this class and implement:
    - forward()
    - get_config()
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization.

        Returns:
            Dictionary of model configuration
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model weights and config.

        Args:
            path: Path to save model (without extension)
        """
        import os
        import json

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), f"{path}.pt")

        # Save config
        with open(f"{path}_config.json", 'w') as f:
            json.dump(self.get_config(), f, indent=2)

        print(f"Model saved to {path}.pt")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from weights and config.

        Args:
            path: Path to model file (without extension)
            **kwargs: Additional arguments to override config

        Returns:
            Loaded model instance
        """
        import json

        # Load config
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)

        # Override with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(torch.load(f"{path}.pt"))

        print(f"Model loaded from {path}.pt")
        return model
