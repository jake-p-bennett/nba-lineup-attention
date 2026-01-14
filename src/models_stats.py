"""
Stats-based models for game prediction.

These models take precomputed player statistics as input
(weighted by participation) rather than learning embeddings.

Input format:
- away_features: (batch, num_features) - team-level stats
- home_features: (batch, num_features) - team-level stats

Features: [off_rating, def_rating, net_rating, possessions, games_played]
(all normalized to mean=0, std=1)
"""

import torch
import torch.nn as nn


class StatsBaseline(nn.Module):
    """
    Simple MLP baseline using stats as input.

    Takes concatenated team stats and predicts point differential.
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        away_features: torch.Tensor,
        home_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            away_features: (batch, num_features)
            home_features: (batch, num_features)

        Returns:
            (batch,) predicted point differential (home - away)
        """
        combined = torch.cat([away_features, home_features], dim=-1)
        return self.fc(combined).squeeze(-1)


class StatsInteraction(nn.Module):
    """
    Model that captures stat interactions between teams.

    Idea: The impact of a team's offense depends on the other team's defense.
    This model explicitly models these matchup effects.
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_features = num_features

        # Process individual team stats
        self.away_encoder = nn.Linear(num_features, hidden_dim)
        self.home_encoder = nn.Linear(num_features, hidden_dim)

        # Interaction layer: outer product of encoded features
        # This captures how home offense interacts with away defense, etc.
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Combine individual + interaction features
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.away_encoder.weight)
        nn.init.xavier_uniform_(self.home_encoder.weight)
        for module in self.interaction:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        for module in self.output:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        away_features: torch.Tensor,
        home_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = away_features.shape[0]

        # Encode individual team stats
        away_enc = torch.relu(self.away_encoder(away_features))  # (batch, hidden)
        home_enc = torch.relu(self.home_encoder(home_features))  # (batch, hidden)

        # Outer product for interactions
        # (batch, hidden, 1) @ (batch, 1, hidden) -> (batch, hidden, hidden)
        interaction = torch.bmm(
            home_enc.unsqueeze(2),
            away_enc.unsqueeze(1)
        ).view(batch_size, -1)  # (batch, hidden*hidden)

        interaction_enc = self.interaction(interaction)  # (batch, hidden)

        # Combine all features
        combined = torch.cat([away_enc, home_enc, interaction_enc], dim=-1)
        return self.output(combined).squeeze(-1)


class StatsAttention(nn.Module):
    """
    Attention-based model for stats.

    Uses cross-attention between team stat features to model
    how each stat interacts with the opponent's stats.
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 1,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Project each stat to embedding space
        # Treat each stat as a "token"
        self.stat_projection = nn.Linear(1, hidden_dim)

        # Position encoding for stats (so model knows which stat is which)
        self.stat_embedding = nn.Embedding(num_features, hidden_dim)

        # Team encoding
        self.team_embedding = nn.Embedding(2, hidden_dim)

        # Self-attention within team
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Cross-attention between teams
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.stat_projection.weight)
        nn.init.xavier_uniform_(self.stat_embedding.weight)
        nn.init.xavier_uniform_(self.team_embedding.weight)

    def encode_team(
        self,
        features: torch.Tensor,
        team_id: int,
    ) -> torch.Tensor:
        """
        Encode team features as sequence of stat embeddings.

        Args:
            features: (batch, num_features)
            team_id: 0 for away, 1 for home

        Returns:
            (batch, num_features, hidden_dim)
        """
        batch_size = features.shape[0]
        device = features.device

        # Project each stat value: (batch, num_features) -> (batch, num_features, hidden)
        stat_emb = self.stat_projection(features.unsqueeze(-1))

        # Add position encoding (which stat is which)
        positions = torch.arange(self.num_features, device=device)
        pos_emb = self.stat_embedding(positions)  # (num_features, hidden)
        stat_emb = stat_emb + pos_emb

        # Add team encoding
        team = torch.tensor([team_id], device=device)
        team_emb = self.team_embedding(team)  # (1, hidden)
        stat_emb = stat_emb + team_emb

        return stat_emb

    def forward(
        self,
        away_features: torch.Tensor,
        home_features: torch.Tensor,
    ) -> torch.Tensor:
        # Encode teams as sequences
        away_emb = self.encode_team(away_features, team_id=0)  # (batch, 5, hidden)
        home_emb = self.encode_team(home_features, team_id=1)  # (batch, 5, hidden)

        # Self-attention within team
        away_self, _ = self.self_attn(away_emb, away_emb, away_emb)
        home_self, _ = self.self_attn(home_emb, home_emb, home_emb)

        away_emb = self.norm1(away_emb + away_self)
        home_emb = self.norm1(home_emb + home_self)

        # Cross-attention: home attends to away, away attends to home
        home_cross, _ = self.cross_attn(home_emb, away_emb, away_emb)
        away_cross, _ = self.cross_attn(away_emb, home_emb, home_emb)

        home_emb = self.norm2(home_emb + home_cross)
        away_emb = self.norm2(away_emb + away_cross)

        # Pool: mean over stats
        away_pooled = away_emb.mean(dim=1)  # (batch, hidden)
        home_pooled = home_emb.mean(dim=1)  # (batch, hidden)

        # Predict
        combined = torch.cat([home_pooled, away_pooled], dim=-1)
        return self.output(combined).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    batch_size = 32
    num_features = 5

    away = torch.randn(batch_size, num_features)
    home = torch.randn(batch_size, num_features)

    print("Testing stats models:")
    print("=" * 50)

    # Baseline
    baseline = StatsBaseline(num_features)
    out = baseline(away, home)
    print(f"StatsBaseline output shape: {out.shape}")
    print(f"StatsBaseline parameters: {count_parameters(baseline):,}")

    # Interaction
    interaction = StatsInteraction(num_features)
    out = interaction(away, home)
    print(f"\nStatsInteraction output shape: {out.shape}")
    print(f"StatsInteraction parameters: {count_parameters(interaction):,}")

    # Attention
    attention = StatsAttention(num_features)
    out = attention(away, home)
    print(f"\nStatsAttention output shape: {out.shape}")
    print(f"StatsAttention parameters: {count_parameters(attention):,}")
