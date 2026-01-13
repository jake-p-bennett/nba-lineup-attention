"""
Hybrid models: Stats + Learned Residual Embeddings

The key insight:
- Stats capture what we KNOW (scoring, defense, etc.)
- Learned embeddings capture what stats MISS (chemistry, intangibles)

Player representation = f(stats) + residual_embedding

The residual starts small and only grows if it helps prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridBaseline(nn.Module):
    """
    Hybrid model: project stats + add learned residual embedding.

    For each player:
        repr = Linear(stats) + embedding

    Then aggregate across team using participation weights.
    """

    def __init__(
        self,
        num_players: int,
        stat_dim: int = 5,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.stat_dim = stat_dim
        self.embed_dim = embed_dim

        # Project stats to representation space
        self.stat_proj = nn.Linear(stat_dim, embed_dim)

        # Learned residual embeddings (initialized small!)
        self.player_embedding = nn.Embedding(num_players, embed_dim)

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Stats projection: normal initialization
        nn.init.xavier_uniform_(self.stat_proj.weight)
        nn.init.zeros_(self.stat_proj.bias)

        # Residual embeddings: START SMALL so stats dominate initially
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)

        # Output layers
        for module in self.output:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def get_player_repr(
        self,
        player_indices: torch.Tensor,
        player_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get player representations: stats projection + residual embedding.

        Args:
            player_indices: (batch, top_k)
            player_stats: (batch, top_k, stat_dim)

        Returns:
            (batch, top_k, embed_dim)
        """
        # Project stats: (batch, top_k, stat_dim) -> (batch, top_k, embed_dim)
        stat_repr = self.stat_proj(player_stats)

        # Get residual embeddings: (batch, top_k) -> (batch, top_k, embed_dim)
        residual = self.player_embedding(player_indices)

        # Combine
        return stat_repr + residual

    def forward(
        self,
        away_indices: torch.Tensor,
        home_indices: torch.Tensor,
        away_stats: torch.Tensor,
        home_stats: torch.Tensor,
        away_weights: torch.Tensor,
        home_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            away_indices: (batch, top_k) player indices
            home_indices: (batch, top_k)
            away_stats: (batch, top_k, stat_dim) player stats
            home_stats: (batch, top_k, stat_dim)
            away_weights: (batch, top_k) participation weights
            home_weights: (batch, top_k)

        Returns:
            (batch,) predicted point differential
        """
        # Get representations
        away_repr = self.get_player_repr(away_indices, away_stats)  # (batch, k, embed)
        home_repr = self.get_player_repr(home_indices, home_stats)

        # Weighted pooling: (batch, k, embed) * (batch, k, 1) -> sum -> (batch, embed)
        away_pooled = (away_repr * away_weights.unsqueeze(-1)).sum(dim=1)
        home_pooled = (home_repr * home_weights.unsqueeze(-1)).sum(dim=1)

        # Predict
        combined = torch.cat([away_pooled, home_pooled], dim=-1)
        return self.output(combined).squeeze(-1)


class HybridAttention(nn.Module):
    """
    Hybrid model with attention over players.

    Uses stats + residual embeddings, then applies attention
    to capture player interactions within and across teams.
    """

    def __init__(
        self,
        num_players: int,
        stat_dim: int = 5,
        embed_dim: int = 32,
        num_heads: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stat_dim = stat_dim
        self.embed_dim = embed_dim

        # Stats projection
        self.stat_proj = nn.Linear(stat_dim, embed_dim)

        # Learned residual embeddings
        self.player_embedding = nn.Embedding(num_players, embed_dim)

        # Team indicator
        self.team_embedding = nn.Embedding(2, embed_dim)

        # Self-attention within team
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention between teams
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Output
        self.output = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.stat_proj.weight)
        nn.init.zeros_(self.stat_proj.bias)
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.team_embedding.weight)

    def get_player_repr(
        self,
        player_indices: torch.Tensor,
        player_stats: torch.Tensor,
        team_id: int,
    ) -> torch.Tensor:
        """Get player representations with team indicator."""
        stat_repr = self.stat_proj(player_stats)
        residual = self.player_embedding(player_indices)

        # Add team embedding
        team_emb = self.team_embedding(
            torch.tensor([team_id], device=player_indices.device)
        )

        return stat_repr + residual + team_emb

    def forward(
        self,
        away_indices: torch.Tensor,
        home_indices: torch.Tensor,
        away_stats: torch.Tensor,
        home_stats: torch.Tensor,
        away_weights: torch.Tensor,
        home_weights: torch.Tensor,
        return_attention: bool = False,
    ):
        # Get representations
        away_repr = self.get_player_repr(away_indices, away_stats, team_id=0)
        home_repr = self.get_player_repr(home_indices, home_stats, team_id=1)

        # Self-attention
        away_self, away_attn = self.self_attn(away_repr, away_repr, away_repr)
        home_self, home_attn = self.self_attn(home_repr, home_repr, home_repr)

        away_repr = self.norm1(away_repr + away_self)
        home_repr = self.norm1(home_repr + home_self)

        # Cross-attention
        away_cross, _ = self.cross_attn(away_repr, home_repr, home_repr)
        home_cross, _ = self.cross_attn(home_repr, away_repr, away_repr)

        away_repr = self.norm2(away_repr + away_cross)
        home_repr = self.norm2(home_repr + home_cross)

        # Feed-forward
        away_repr = self.norm3(away_repr + self.ff(away_repr))
        home_repr = self.norm3(home_repr + self.ff(home_repr))

        # Weighted pooling
        away_pooled = (away_repr * away_weights.unsqueeze(-1)).sum(dim=1)
        home_pooled = (home_repr * home_weights.unsqueeze(-1)).sum(dim=1)

        # Predict
        combined = torch.cat([home_pooled, away_pooled], dim=-1)
        pred = self.output(combined).squeeze(-1)

        if return_attention:
            return pred, {'self': (away_attn, home_attn)}
        return pred


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_embedding_norm(model: nn.Module) -> float:
    """Measure how much the residual embeddings have grown."""
    if hasattr(model, 'player_embedding'):
        return model.player_embedding.weight.norm().item()
    return 0.0


if __name__ == "__main__":
    # Test
    batch_size = 32
    top_k = 10
    num_players = 1181
    stat_dim = 5

    away_idx = torch.randint(0, num_players, (batch_size, top_k))
    home_idx = torch.randint(0, num_players, (batch_size, top_k))
    away_stats = torch.randn(batch_size, top_k, stat_dim)
    home_stats = torch.randn(batch_size, top_k, stat_dim)
    away_weights = F.softmax(torch.randn(batch_size, top_k), dim=-1)
    home_weights = F.softmax(torch.randn(batch_size, top_k), dim=-1)

    print("Testing hybrid models:")
    print("=" * 50)

    # Baseline
    baseline = HybridBaseline(num_players, stat_dim)
    out = baseline(away_idx, home_idx, away_stats, home_stats, away_weights, home_weights)
    print(f"HybridBaseline output shape: {out.shape}")
    print(f"HybridBaseline parameters: {count_parameters(baseline):,}")
    print(f"Initial embedding norm: {count_embedding_norm(baseline):.4f}")

    # Attention
    attention = HybridAttention(num_players, stat_dim)
    out = attention(away_idx, home_idx, away_stats, home_stats, away_weights, home_weights)
    print(f"\nHybridAttention output shape: {out.shape}")
    print(f"HybridAttention parameters: {count_parameters(attention):,}")
    print(f"Initial embedding norm: {count_embedding_norm(attention):.4f}")
