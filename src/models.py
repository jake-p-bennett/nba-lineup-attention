"""
NBA Lineup Models

Two approaches:
1. BaselineModel: NBA2Vec-style averaging (no interaction modeling)
2. LineupTransformer: Attention-based model (captures player synergies)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaselineModel(nn.Module):
    """
    NBA2Vec-style baseline: embed players, average, predict.

    This model CANNOT capture player interactions because averaging
    destroys pairwise information. It only captures "average quality"
    of a lineup.
    """

    def __init__(
        self,
        num_players: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.player_embedding = nn.Embedding(num_players, embedding_dim)

        # After averaging 5 players, we have embedding_dim features
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.player_embedding.weight)
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, player_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            player_indices: (batch_size, 5) player indices

        Returns:
            (batch_size,) predicted net rating
        """
        # Embed players: (batch, 5, embedding_dim)
        embeddings = self.player_embedding(player_indices)

        # Average across players: (batch, embedding_dim)
        lineup_embedding = embeddings.mean(dim=1)

        # Predict: (batch, 1) -> (batch,)
        return self.fc(lineup_embedding).squeeze(-1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            return_attention: whether to return attention weights

        Returns:
            output: (batch, seq_len, embed_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, v)

        # Reshape back: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class TransformerBlock(nn.Module):
    """Standard transformer block: attention + feed-forward with residuals."""

    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(self.norm1(x), return_attention)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class LineupTransformer(nn.Module):
    """
    Attention-based lineup model.

    Unlike the baseline, this model captures player-to-player interactions
    through self-attention. The attention weights reveal which player
    combinations matter.

    Architecture:
        1. Embed each player
        2. Add learned position encoding (optional, for role-awareness)
        3. Self-attention layers to model interactions
        4. Aggregate and predict
    """

    def __init__(
        self,
        num_players: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_position_encoding = use_position_encoding

        # Player embeddings
        self.player_embedding = nn.Embedding(num_players, embedding_dim)

        # Learned position encoding for 5 lineup slots
        # This lets the model learn that "slot 1" might be point guard, etc.
        if use_position_encoding:
            self.position_embedding = nn.Embedding(5, embedding_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        # Aggregation: learned weighted sum instead of simple average
        self.attention_pool = nn.Linear(embedding_dim, 1)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.player_embedding.weight)
        if self.use_position_encoding:
            nn.init.xavier_uniform_(self.position_embedding.weight)

    def forward(
        self,
        player_indices: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            player_indices: (batch_size, 5) player indices
            return_attention: whether to return attention weights from all layers

        Returns:
            predictions: (batch_size,) predicted net rating
            attention_weights: list of (batch, num_heads, 5, 5) if return_attention
        """
        batch_size = player_indices.shape[0]

        # Embed players: (batch, 5, embedding_dim)
        x = self.player_embedding(player_indices)

        # Add position encoding
        if self.use_position_encoding:
            positions = torch.arange(5, device=player_indices.device)
            x = x + self.position_embedding(positions)

        # Pass through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, return_attention)
            if return_attention:
                attention_weights.append(attn)

        x = self.norm(x)

        # Attention-weighted pooling: (batch, 5, embed_dim) -> (batch, embed_dim)
        pool_weights = F.softmax(self.attention_pool(x).squeeze(-1), dim=-1)  # (batch, 5)
        x = torch.einsum('bn,bnd->bd', pool_weights, x)  # (batch, embed_dim)

        # Predict
        predictions = self.output_head(x).squeeze(-1)

        if return_attention:
            return predictions, attention_weights
        return predictions, None

    def get_player_synergy_scores(
        self,
        player_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get pairwise synergy scores from attention weights.

        Args:
            player_indices: (batch_size, 5) player indices

        Returns:
            synergy_matrix: (batch_size, 5, 5) pairwise synergy scores
        """
        _, attention_weights = self.forward(player_indices, return_attention=True)

        # Average attention across all heads and layers
        # Each attn is (batch, num_heads, 5, 5)
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 2))  # (batch, 5, 5)

        return avg_attention


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    num_players = 913
    batch_size = 32

    # Create random input
    player_indices = torch.randint(0, num_players, (batch_size, 5))

    # Test baseline
    baseline = BaselineModel(num_players)
    out = baseline(player_indices)
    print(f"Baseline output shape: {out.shape}")
    print(f"Baseline parameters: {count_parameters(baseline):,}")

    # Test transformer
    transformer = LineupTransformer(num_players)
    out, attn = transformer(player_indices, return_attention=True)
    print(f"\nTransformer output shape: {out.shape}")
    print(f"Transformer parameters: {count_parameters(transformer):,}")
    print(f"Attention weights shapes: {[a.shape for a in attn]}")
