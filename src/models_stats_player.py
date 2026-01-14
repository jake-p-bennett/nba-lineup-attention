"""
Alternative Step 2: Stats as player embeddings with attention.

Key difference from Step 2 (actual):
- Operates at player level (10 players per team), not team level
- Can apply attention over players

Key difference from Step 3 (Hybrid):
- No learned per-player residual embeddings
- Just stats projected to embedding space
"""

import torch
import torch.nn as nn


class StatsPlayerBaseline(nn.Module):
    """
    Stats-based model at player level with simple averaging.

    For each player: repr = Linear(stats)
    Then aggregate across team using participation weights.
    """

    def __init__(
        self,
        stat_dim: int = 7,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.stat_dim = stat_dim
        self.embed_dim = embed_dim

        # Project stats to representation space
        self.stat_proj = nn.Linear(stat_dim, embed_dim)

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
        nn.init.xavier_uniform_(self.stat_proj.weight)
        nn.init.zeros_(self.stat_proj.bias)
        for module in self.output:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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
            away_indices: (batch, top_k) - NOT USED, kept for API compatibility
            home_indices: (batch, top_k) - NOT USED
            away_stats: (batch, top_k, stat_dim) player stats
            home_stats: (batch, top_k, stat_dim)
            away_weights: (batch, top_k) participation weights
            home_weights: (batch, top_k)

        Returns:
            (batch,) predicted point differential
        """
        # Project stats to embedding space: (batch, k, stat_dim) -> (batch, k, embed_dim)
        away_repr = self.stat_proj(away_stats)
        home_repr = self.stat_proj(home_stats)

        # Weighted pooling: (batch, k, embed) * (batch, k, 1) -> sum -> (batch, embed)
        away_pooled = (away_repr * away_weights.unsqueeze(-1)).sum(dim=1)
        home_pooled = (home_repr * home_weights.unsqueeze(-1)).sum(dim=1)

        # Predict
        combined = torch.cat([away_pooled, home_pooled], dim=-1)
        return self.output(combined).squeeze(-1)


class StatsPlayerAttention(nn.Module):
    """
    Stats-based model at player level with attention.

    For each player: repr = Linear(stats)
    Then apply self-attention within team and cross-attention between teams.
    """

    def __init__(
        self,
        stat_dim: int = 7,
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
        nn.init.xavier_uniform_(self.team_embedding.weight)

    def get_player_repr(
        self,
        player_stats: torch.Tensor,
        team_id: int,
    ) -> torch.Tensor:
        """Get player representations from stats with team indicator."""
        stat_repr = self.stat_proj(player_stats)

        # Add team embedding
        team_emb = self.team_embedding(
            torch.tensor([team_id], device=player_stats.device)
        )

        return stat_repr + team_emb

    def forward(
        self,
        away_indices: torch.Tensor,
        home_indices: torch.Tensor,
        away_stats: torch.Tensor,
        home_stats: torch.Tensor,
        away_weights: torch.Tensor,
        home_weights: torch.Tensor,
    ):
        # Get representations (no per-player embeddings, just projected stats)
        away_repr = self.get_player_repr(away_stats, team_id=0)
        home_repr = self.get_player_repr(home_stats, team_id=1)

        # Self-attention
        away_self, _ = self.self_attn(away_repr, away_repr, away_repr)
        home_self, _ = self.self_attn(home_repr, home_repr, home_repr)

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
        return self.output(combined).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch.nn.functional as F

    # Test
    batch_size = 32
    top_k = 10
    stat_dim = 7

    away_idx = torch.randint(0, 1000, (batch_size, top_k))
    home_idx = torch.randint(0, 1000, (batch_size, top_k))
    away_stats = torch.randn(batch_size, top_k, stat_dim)
    home_stats = torch.randn(batch_size, top_k, stat_dim)
    away_weights = F.softmax(torch.randn(batch_size, top_k), dim=-1)
    home_weights = F.softmax(torch.randn(batch_size, top_k), dim=-1)

    print("Testing stats-at-player-level models:")
    print("=" * 50)

    # Baseline
    baseline = StatsPlayerBaseline(stat_dim)
    out = baseline(away_idx, home_idx, away_stats, home_stats, away_weights, home_weights)
    print(f"StatsPlayerBaseline output shape: {out.shape}")
    print(f"StatsPlayerBaseline parameters: {count_parameters(baseline):,}")

    # Attention
    attention = StatsPlayerAttention(stat_dim)
    out = attention(away_idx, home_idx, away_stats, home_stats, away_weights, home_weights)
    print(f"\nStatsPlayerAttention output shape: {out.shape}")
    print(f"StatsPlayerAttention parameters: {count_parameters(attention):,}")
