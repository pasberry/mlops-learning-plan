"""Tabular classification models."""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from ml.models.base import BaseModel


class TabularClassifier(BaseModel):
    """Multi-layer perceptron for tabular classification.

    Architecture:
        Input → Linear → ReLU → Dropout →
        [Linear → ReLU → Dropout] × (n_layers - 1) →
        Linear → Sigmoid

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions (e.g., [256, 128, 64])
        dropout: Dropout probability (default: 0.3)
        output_dim: Output dimension (default: 1 for binary classification)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.output_dim = output_dim

        # Build layers
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        logits = self.layers(x)
        probs = self.sigmoid(logits)
        return probs

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'output_dim': self.output_dim
        }

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"TabularClassifier(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  dropout={self.dropout},\n"
            f"  output_dim={self.output_dim},\n"
            f"  params={n_params:,}\n"
            f")"
        )


# Example usage and testing
if __name__ == '__main__':
    # Create model
    model = TabularClassifier(
        input_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )

    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 16)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test save/load
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model')
        model.save(model_path)
        loaded_model = TabularClassifier.load(model_path)
        print("\n✅ Save/load test passed")
