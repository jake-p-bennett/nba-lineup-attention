"""
Models for lineup-level prediction.

Input: 5 players (indices + stats)
Output: Predicted net rating for the lineup

This tests whether self-attention can capture within-team chemistry
better than simple averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LineupBaseline(nn.Module):
    """
    Baseline: Average player representations, then predict.

    Architecture:
        Player Stats → Linear Projection → Add Residual Embedding
                                    ↓
                        Simple Average of 5 Players
                                    ↓
                              MLP → Net Rating
    """

    def __init__(
        self,
        num_players: int,
        stat_dim: int = 7,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.stat_proj = nn.Linear(stat_dim, embed_dim)
        self.player_embedding = nn.Embedding(num_players, embed_dim)

        # Initialize embeddings small (residuals to stats)
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def get_player_repr(self, player_indices: torch.Tensor, player_stats: torch.Tensor) -> torch.Tensor:
        """Get player representation: projected stats + residual embedding."""
        stat_repr = self.stat_proj(player_stats)
        residual = self.player_embedding(player_indices)
        return stat_repr + residual

    def forward(
        self,
        player_indices: torch.Tensor,  # (batch, 5)
        player_stats: torch.Tensor,     # (batch, 5, stat_dim)
    ) -> torch.Tensor:
        # Get representations: (batch, 5, embed_dim)
        player_repr = self.get_player_repr(player_indices, player_stats)

        # Simple average: (batch, embed_dim)
        lineup_repr = player_repr.mean(dim=1)

        # Predict
        return self.head(lineup_repr).squeeze(-1)


class LineupAttention(nn.Module):
    """
    Attention-based lineup model.

    Architecture:
        Player Stats → Linear Projection → Add Residual Embedding
                                    ↓
                        Self-Attention (5 players attend to each other)
                                    ↓
                        Attention-Weighted Pooling
                                    ↓
                              MLP → Net Rating

    The hypothesis: Self-attention can learn which player combinations
    have chemistry (synergy or anti-synergy) beyond individual quality.
    """

    def __init__(
        self,
        num_players: int,
        stat_dim: int = 7,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()

        self.stat_proj = nn.Linear(stat_dim, embed_dim)
        self.player_embedding = nn.Embedding(num_players, embed_dim)

        # Initialize embeddings small
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Attention pooling: learn which players to weight more
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def get_player_repr(self, player_indices: torch.Tensor, player_stats: torch.Tensor) -> torch.Tensor:
        """Get player representation: projected stats + residual embedding."""
        stat_repr = self.stat_proj(player_stats)
        residual = self.player_embedding(player_indices)
        return stat_repr + residual

    def forward(
        self,
        player_indices: torch.Tensor,  # (batch, 5)
        player_stats: torch.Tensor,     # (batch, 5, stat_dim)
    ) -> torch.Tensor:
        batch_size = player_indices.size(0)

        # Get representations: (batch, 5, embed_dim)
        player_repr = self.get_player_repr(player_indices, player_stats)

        # Self-attention: players attend to each other
        attn_out, _ = self.self_attn(player_repr, player_repr, player_repr)
        player_repr = self.attn_norm(player_repr + attn_out)

        # Attention pooling: (batch, 1, embed_dim)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, player_repr, player_repr)
        lineup_repr = pooled.squeeze(1)  # (batch, embed_dim)

        # Predict
        return self.head(lineup_repr).squeeze(-1)


class LineupDeepAttention(nn.Module):
    """
    Deeper attention model with multiple layers.

    Tests if more capacity helps capture complex interactions.
    """

    def __init__(
        self,
        num_players: int,
        stat_dim: int = 7,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.stat_proj = nn.Linear(stat_dim, embed_dim)
        self.player_embedding = nn.Embedding(num_players, embed_dim)
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)

        # Multiple self-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Simple average pooling (after attention has mixed information)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def get_player_repr(self, player_indices: torch.Tensor, player_stats: torch.Tensor) -> torch.Tensor:
        stat_repr = self.stat_proj(player_stats)
        residual = self.player_embedding(player_indices)
        return stat_repr + residual

    def forward(
        self,
        player_indices: torch.Tensor,
        player_stats: torch.Tensor,
    ) -> torch.Tensor:
        # Get representations
        x = self.get_player_repr(player_indices, player_stats)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Average pool
        lineup_repr = x.mean(dim=1)

        return self.head(lineup_repr).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_embedding_norm(model: nn.Module) -> float:
    """Compute L2 norm of embeddings."""
    if hasattr(model, 'player_embedding'):
        return model.player_embedding.weight.norm().item()
    return 0.0
