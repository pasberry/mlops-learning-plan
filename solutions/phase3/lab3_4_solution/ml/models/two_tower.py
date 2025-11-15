"""Two-tower models for ranking and recommendations."""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from ml.models.base import BaseModel


class TwoTowerModel(BaseModel):
    """Two-tower architecture for ranking.

    Separate encoders for user and item, with dot product similarity.

    Architecture:
        User Features → User Tower → User Embedding (embedding_dim)
        Item Features → Item Tower → Item Embedding (embedding_dim)

        Similarity = dot(User Embedding, Item Embedding)
        Score = sigmoid(Similarity)

    Args:
        user_input_dim: Dimension of user features
        item_input_dim: Dimension of item features
        embedding_dim: Dimension of final embeddings (default: 128)
        user_hidden_dims: Hidden layer sizes for user tower (default: [256, 128])
        item_hidden_dims: Hidden layer sizes for item tower (default: [256, 128])
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int = 128,
        user_hidden_dims: List[int] = None,
        item_hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()

        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim
        self.user_hidden_dims = user_hidden_dims or [256, 128]
        self.item_hidden_dims = item_hidden_dims or [256, 128]
        self.dropout = dropout

        # Build user tower
        self.user_tower = self._build_tower(
            user_input_dim,
            self.user_hidden_dims,
            embedding_dim,
            dropout
        )

        # Build item tower
        self.item_tower = self._build_tower(
            item_input_dim,
            self.item_hidden_dims,
            embedding_dim,
            dropout
        )

        # Initialize weights
        self._init_weights()

    def _build_tower(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Sequential:
        """Build a tower (encoder) network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout probability

        Returns:
            Sequential module
        """
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

        # Output layer (embedding)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features to embedding.

        Args:
            user_features: (batch_size, user_input_dim)

        Returns:
            User embeddings: (batch_size, embedding_dim)
        """
        embeddings = self.user_tower(user_features)
        # L2 normalize
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features to embedding.

        Args:
            item_features: (batch_size, item_input_dim)

        Returns:
            Item embeddings: (batch_size, embedding_dim)
        """
        embeddings = self.item_tower(item_features)
        # L2 normalize
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute affinity scores.

        Args:
            user_features: (batch_size, user_input_dim)
            item_features: (batch_size, item_input_dim)

        Returns:
            Affinity scores: (batch_size, 1)
        """
        # Get embeddings
        user_embeddings = self.encode_user(user_features)
        item_embeddings = self.encode_item(item_features)

        # Compute dot product (element-wise multiply + sum)
        # (batch_size, embedding_dim) * (batch_size, embedding_dim)
        # -> (batch_size, embedding_dim) -> (batch_size)
        similarity = (user_embeddings * item_embeddings).sum(dim=1, keepdim=True)

        # Apply sigmoid to get probability
        score = torch.sigmoid(similarity)

        return score

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'user_input_dim': self.user_input_dim,
            'item_input_dim': self.item_input_dim,
            'embedding_dim': self.embedding_dim,
            'user_hidden_dims': self.user_hidden_dims,
            'item_hidden_dims': self.item_hidden_dims,
            'dropout': self.dropout
        }

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"TwoTowerModel(\n"
            f"  user_input_dim={self.user_input_dim},\n"
            f"  item_input_dim={self.item_input_dim},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  user_hidden_dims={self.user_hidden_dims},\n"
            f"  item_hidden_dims={self.item_hidden_dims},\n"
            f"  dropout={self.dropout},\n"
            f"  params={n_params:,}\n"
            f")"
        )


# Test
if __name__ == '__main__':
    # Create model
    model = TwoTowerModel(
        user_input_dim=20,
        item_input_dim=15,
        embedding_dim=128,
        user_hidden_dims=[256, 128],
        item_hidden_dims=[256, 128],
        dropout=0.3
    )

    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    user_features = torch.randn(batch_size, 20)
    item_features = torch.randn(batch_size, 15)

    scores = model(user_features, item_features)
    print(f"\nInput shapes:")
    print(f"  User: {user_features.shape}")
    print(f"  Item: {item_features.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test encoding separately
    user_emb = model.encode_user(user_features)
    item_emb = model.encode_item(item_features)
    print(f"\nEmbedding shapes:")
    print(f"  User: {user_emb.shape}")
    print(f"  Item: {item_emb.shape}")

    # Check normalization
    user_norms = torch.norm(user_emb, dim=1)
    print(f"User embedding norms: {user_norms.mean():.3f} (should be ~1.0)")

    print("\n✅ Two-tower model working correctly")
