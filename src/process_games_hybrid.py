"""
Process game data for hybrid model (stats + learned embeddings).

For each game, we need:
- Player IDs (for learned embeddings)
- Player stats (for known features)
- Participation weights (for aggregation)

This allows the model to use stats as a strong baseline
while learning residual embeddings for what stats miss.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
MIN_YEAR = 2015


def load_data():
    """Load play-by-play data and player stats."""
    print("Loading data...")

    # Load player stats
    df_stats = pd.read_parquet(DATA_DIR / 'player_stats_computed.parquet')
    print(f"Loaded stats for {len(df_stats)} players")

    # Create player ID mapping
    all_players = sorted(df_stats['player_id'].tolist())
    player_to_idx = {p: i for i, p in enumerate(all_players)}

    # Create stats lookup (normalized)
    stat_cols = ['off_rating_norm', 'def_rating_norm', 'net_rating_norm',
                 'possessions_norm', 'games_played_norm']

    player_stats = {}
    for _, row in df_stats.iterrows():
        player_stats[row['player_id']] = np.array([row[col] for col in stat_cols])

    # Load play-by-play
    cols = [
        'GameID', 'Date', 'Possession',
        'AwayScore', 'HomeScore',
        'A1', 'A2', 'A3', 'A4', 'A5',
        'H1', 'H2', 'H3', 'H4', 'H5',
    ]
    df = pd.read_csv(DATA_DIR / 'all_games.csv', usecols=cols, low_memory=False)

    df['AwayScore'] = pd.to_numeric(df['AwayScore'], errors='coerce')
    df['HomeScore'] = pd.to_numeric(df['HomeScore'], errors='coerce')
    df['Year'] = df['Date'].str[-4:].astype(int)
    df = df[df['Year'] >= MIN_YEAR].copy()

    print(f"Loaded {len(df):,} plays")
    print(f"Player vocabulary: {len(player_to_idx)} players")

    return df, player_stats, player_to_idx, len(stat_cols)


def aggregate_to_games(df: pd.DataFrame, player_stats: dict, player_to_idx: dict, num_features: int):
    """
    Aggregate to game level with both IDs and stats.

    For each game, return:
    - Top-K player indices for each team
    - Their participation weights
    - Their stats
    """
    print("Aggregating to game level...")

    TOP_K = 10  # Keep top 10 players per team by participation

    games = []
    skipped = 0

    for game_id, game_df in tqdm(df.groupby('GameID'), desc="Processing games"):
        # Get final scores
        final_away = game_df['AwayScore'].dropna().iloc[-1] if len(game_df['AwayScore'].dropna()) > 0 else None
        final_home = game_df['HomeScore'].dropna().iloc[-1] if len(game_df['HomeScore'].dropna()) > 0 else None

        if final_away is None or final_home is None:
            continue

        point_diff = final_home - final_away

        # Count possessions for each player
        away_counts = defaultdict(int)
        home_counts = defaultdict(int)

        for _, play in game_df.iterrows():
            for col in ['A1', 'A2', 'A3', 'A4', 'A5']:
                player = play[col]
                if pd.notna(player) and player in player_to_idx:
                    away_counts[player] += 1

            for col in ['H1', 'H2', 'H3', 'H4', 'H5']:
                player = play[col]
                if pd.notna(player) and player in player_to_idx:
                    home_counts[player] += 1

        # Skip if not enough valid players
        if len(away_counts) < 5 or len(home_counts) < 5:
            skipped += 1
            continue

        # Get top-K players by participation
        away_sorted = sorted(away_counts.items(), key=lambda x: -x[1])[:TOP_K]
        home_sorted = sorted(home_counts.items(), key=lambda x: -x[1])[:TOP_K]

        # Pad to TOP_K if needed
        while len(away_sorted) < TOP_K:
            away_sorted.append((away_sorted[0][0], 0))  # Repeat first player with 0 weight
        while len(home_sorted) < TOP_K:
            home_sorted.append((home_sorted[0][0], 0))

        # Extract indices, weights, stats
        away_indices = [player_to_idx[p] for p, _ in away_sorted]
        away_weights = [c for _, c in away_sorted]
        away_stats_list = [player_stats[p] for p, _ in away_sorted]

        home_indices = [player_to_idx[p] for p, _ in home_sorted]
        home_weights = [c for _, c in home_sorted]
        home_stats_list = [player_stats[p] for p, _ in home_sorted]

        # Normalize weights
        away_total = sum(away_weights)
        home_total = sum(home_weights)
        away_weights = [w / away_total for w in away_weights]
        home_weights = [w / home_total for w in home_weights]

        games.append({
            'game_id': game_id,
            'point_diff': point_diff,
            'away_indices': away_indices,
            'away_weights': away_weights,
            'away_stats': away_stats_list,
            'home_indices': home_indices,
            'home_weights': home_weights,
            'home_stats': home_stats_list,
        })

    print(f"Processed {len(games):,} games (skipped {skipped})")
    return games


def prepare_tensors(games: list, num_features: int, top_k: int = 10):
    """Convert to tensors."""
    print("Preparing tensors...")

    n = len(games)

    # Shape: (num_games, top_k)
    away_indices = torch.zeros(n, top_k, dtype=torch.long)
    home_indices = torch.zeros(n, top_k, dtype=torch.long)
    away_weights = torch.zeros(n, top_k)
    home_weights = torch.zeros(n, top_k)

    # Shape: (num_games, top_k, num_features)
    away_stats = torch.zeros(n, top_k, num_features)
    home_stats = torch.zeros(n, top_k, num_features)

    targets = torch.zeros(n)

    for i, game in enumerate(games):
        away_indices[i] = torch.tensor(game['away_indices'])
        home_indices[i] = torch.tensor(game['home_indices'])
        away_weights[i] = torch.tensor(game['away_weights'])
        home_weights[i] = torch.tensor(game['home_weights'])
        away_stats[i] = torch.tensor(np.array(game['away_stats']))
        home_stats[i] = torch.tensor(np.array(game['home_stats']))
        targets[i] = game['point_diff']

    return {
        'away_indices': away_indices,
        'home_indices': home_indices,
        'away_weights': away_weights,
        'home_weights': home_weights,
        'away_stats': away_stats,
        'home_stats': home_stats,
        'targets': targets,
    }


def train_val_test_split(data: dict, val_frac: float = 0.15, test_frac: float = 0.15):
    """Split data."""
    n = len(data['targets'])
    indices = np.random.permutation(n)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {}
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[split_name] = {
            'away_indices': data['away_indices'][idx],
            'home_indices': data['home_indices'][idx],
            'away_weights': data['away_weights'][idx],
            'home_weights': data['home_weights'][idx],
            'away_stats': data['away_stats'][idx],
            'home_stats': data['home_stats'][idx],
            'targets': data['targets'][idx],
        }

    return splits


def main():
    np.random.seed(42)

    df, player_stats, player_to_idx, num_features = load_data()
    games = aggregate_to_games(df, player_stats, player_to_idx, num_features)
    data = prepare_tensors(games, num_features)

    splits = train_val_test_split(data)

    # Save
    output_dir = DATA_DIR / 'games_hybrid_processed'
    output_dir.mkdir(exist_ok=True)

    for split_name, split_data in splits.items():
        torch.save(split_data, output_dir / f'{split_name}.pt')
        print(f"Saved {split_name}: {len(split_data['targets'])} samples")

    # Metadata
    metadata = {
        'num_players': len(player_to_idx),
        'num_features': num_features,
        'top_k': 10,
        'target_mean': float(data['targets'].mean()),
        'target_std': float(data['targets'].std()),
        'feature_names': ['off_rating_norm', 'def_rating_norm', 'net_rating_norm',
                          'possessions_norm', 'games_played_norm'],
        'player_to_idx': player_to_idx,
    }
    torch.save(metadata, output_dir / 'metadata.pt')

    # Reverse mapping
    idx_to_player = {v: k for k, v in player_to_idx.items()}
    torch.save(idx_to_player, output_dir / 'idx_to_player.pt')

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total games: {len(games):,}")
    print(f"Players: {len(player_to_idx)}")
    print(f"Features per player: {num_features}")
    print(f"Top-K players per team: 10")
    print(f"Train: {len(splits['train']['targets']):,}")
    print(f"Val: {len(splits['val']['targets']):,}")
    print(f"Test: {len(splits['test']['targets']):,}")
    print(f"\nTarget stats:")
    print(f"  Mean: {metadata['target_mean']:.2f}")
    print(f"  Std: {metadata['target_std']:.2f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
