"""
Process play-by-play data into lineup-level aggregations.

Each observation is a unique 5-player lineup with their aggregated performance
across all minutes played together.

This preserves within-team interactions for attention to potentially learn from.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch

DATA_DIR = Path(__file__).parent.parent / "data"
MIN_YEAR = 2015
MIN_POSSESSIONS = 50  # Minimum possessions for a lineup to be included


def load_data():
    """Load play-by-play data and player stats."""
    print("Loading data...")

    # Load player stats (use rich stats - 7 features)
    df_stats = pd.read_parquet(DATA_DIR / 'player_stats_rich.parquet')
    print(f"Loaded stats for {len(df_stats)} players")

    # Create player ID mapping
    all_players = sorted(df_stats['player_id'].tolist())
    player_to_idx = {p: i for i, p in enumerate(all_players)}

    # Create stats lookup (normalized) - 7 features
    stat_cols = ['off_rating_norm', 'def_rating_norm', 'net_rating_norm',
                 'pts_per100_norm', 'reb_per100_norm',
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

    return df, player_stats, player_to_idx


def extract_lineups(df: pd.DataFrame, player_stats: dict, player_to_idx: dict):
    """
    Extract lineup-level statistics.

    For each unique 5-player combination, compute:
    - Total possessions played together
    - Points scored and allowed
    - Net rating (per 100 possessions)
    """
    print("Extracting lineup statistics...")

    # Track stats for each lineup (as frozen set of 5 players)
    # Separate tracking for home and away to get both offensive and defensive context
    lineup_stats = defaultdict(lambda: {
        'possessions': 0,
        'points_for': 0,
        'points_against': 0,
        'games': set(),
    })

    prev_scores = {}  # GameID -> (away_score, home_score)
    prev_lineups = {}  # GameID -> (away_lineup, home_lineup)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing plays"):
        game_id = row['GameID']

        # Get current scores
        away_score = row['AwayScore']
        home_score = row['HomeScore']

        if pd.isna(away_score) or pd.isna(home_score):
            continue

        # Get current lineups (only players we have stats for)
        away_players = tuple(sorted([
            row[c] for c in ['A1', 'A2', 'A3', 'A4', 'A5']
            if pd.notna(row[c]) and row[c] in player_to_idx
        ]))
        home_players = tuple(sorted([
            row[c] for c in ['H1', 'H2', 'H3', 'H4', 'H5']
            if pd.notna(row[c]) and row[c] in player_to_idx
        ]))

        # Skip if we don't have full lineups
        if len(away_players) != 5 or len(home_players) != 5:
            continue

        # If we have previous data for this game, compute deltas
        if game_id in prev_scores:
            prev_away, prev_home = prev_scores[game_id]
            prev_away_lineup, prev_home_lineup = prev_lineups[game_id]

            away_delta = away_score - prev_away
            home_delta = home_score - prev_home

            # Update stats for the previous lineups (they were on court for this scoring)
            if prev_away_lineup and len(prev_away_lineup) == 5:
                lineup_stats[prev_away_lineup]['possessions'] += 1
                lineup_stats[prev_away_lineup]['points_for'] += away_delta
                lineup_stats[prev_away_lineup]['points_against'] += home_delta
                lineup_stats[prev_away_lineup]['games'].add(game_id)

            if prev_home_lineup and len(prev_home_lineup) == 5:
                lineup_stats[prev_home_lineup]['possessions'] += 1
                lineup_stats[prev_home_lineup]['points_for'] += home_delta
                lineup_stats[prev_home_lineup]['points_against'] += away_delta
                lineup_stats[prev_home_lineup]['games'].add(game_id)

        # Store current state
        prev_scores[game_id] = (away_score, home_score)
        prev_lineups[game_id] = (away_players, home_players)

    print(f"Found {len(lineup_stats):,} unique lineups")

    return lineup_stats


def create_dataset(lineup_stats: dict, player_stats: dict, player_to_idx: dict, min_poss: int):
    """
    Create dataset from lineup statistics.

    Filter to lineups with sufficient possessions and compute features/targets.
    """
    print(f"Creating dataset (min {min_poss} possessions)...")

    records = []

    for lineup, stats in lineup_stats.items():
        if stats['possessions'] < min_poss:
            continue

        # Check all players have stats
        if not all(p in player_stats for p in lineup):
            continue

        poss = stats['possessions']
        pts_for = stats['points_for']
        pts_against = stats['points_against']

        # Compute ratings per 100 possessions
        off_rating = (pts_for / poss) * 100
        def_rating = (pts_against / poss) * 100
        net_rating = off_rating - def_rating

        # Get player indices and stats
        player_indices = [player_to_idx[p] for p in lineup]
        player_stat_vectors = [player_stats[p] for p in lineup]

        records.append({
            'lineup': lineup,
            'player_indices': player_indices,
            'player_stats': player_stat_vectors,
            'possessions': poss,
            'games': len(stats['games']),
            'off_rating': off_rating,
            'def_rating': def_rating,
            'net_rating': net_rating,
        })

    print(f"Created {len(records):,} lineup observations (from {len(lineup_stats):,} total)")

    return records


def prepare_tensors(records: list, num_features: int = 7):
    """Convert records to tensors."""
    print("Preparing tensors...")

    n = len(records)

    # Shape: (num_lineups, 5)
    player_indices = torch.zeros(n, 5, dtype=torch.long)

    # Shape: (num_lineups, 5, num_features)
    player_stats = torch.zeros(n, 5, num_features)

    # Targets
    net_ratings = torch.zeros(n)
    possessions = torch.zeros(n)

    for i, rec in enumerate(records):
        player_indices[i] = torch.tensor(rec['player_indices'])
        player_stats[i] = torch.tensor(np.array(rec['player_stats']))
        net_ratings[i] = rec['net_rating']
        possessions[i] = rec['possessions']

    return {
        'player_indices': player_indices,
        'player_stats': player_stats,
        'net_ratings': net_ratings,
        'possessions': possessions,
    }


def train_val_test_split(data: dict, val_frac: float = 0.15, test_frac: float = 0.15):
    """Split data randomly."""
    n = len(data['net_ratings'])
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
            'player_indices': data['player_indices'][idx],
            'player_stats': data['player_stats'][idx],
            'net_ratings': data['net_ratings'][idx],
            'possessions': data['possessions'][idx],
        }

    return splits


def main():
    np.random.seed(42)

    df, player_stats, player_to_idx = load_data()
    lineup_stats = extract_lineups(df, player_stats, player_to_idx)
    records = create_dataset(lineup_stats, player_stats, player_to_idx, MIN_POSSESSIONS)

    if len(records) < 100:
        print(f"\nWARNING: Only {len(records)} lineups with {MIN_POSSESSIONS}+ possessions.")
        print("Trying with lower threshold...")
        for threshold in [30, 20, 10]:
            records = create_dataset(lineup_stats, player_stats, player_to_idx, threshold)
            if len(records) >= 500:
                print(f"Using threshold of {threshold} possessions")
                break

    data = prepare_tensors(records)
    splits = train_val_test_split(data)

    # Save
    output_dir = DATA_DIR / 'lineups_processed'
    output_dir.mkdir(exist_ok=True)

    for split_name, split_data in splits.items():
        torch.save(split_data, output_dir / f'{split_name}.pt')
        print(f"Saved {split_name}: {len(split_data['net_ratings'])} samples")

    # Compute target stats
    all_targets = data['net_ratings']
    target_mean = float(all_targets.mean())
    target_std = float(all_targets.std())

    # Metadata
    metadata = {
        'num_players': len(player_to_idx),
        'num_features': 7,
        'num_lineup_players': 5,
        'target_mean': target_mean,
        'target_std': target_std,
        'min_possessions': MIN_POSSESSIONS,
        'total_lineups': len(records),
        'player_to_idx': player_to_idx,
    }
    torch.save(metadata, output_dir / 'metadata.pt')

    print(f"\n{'='*60}")
    print("LINEUP-LEVEL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total lineups: {len(records):,}")
    print(f"Players: {len(player_to_idx)}")
    print(f"Features per player: 7")
    print(f"Min possessions filter: {MIN_POSSESSIONS}")
    print(f"Train: {len(splits['train']['net_ratings']):,}")
    print(f"Val: {len(splits['val']['net_ratings']):,}")
    print(f"Test: {len(splits['test']['net_ratings']):,}")
    print(f"\nTarget (net rating) stats:")
    print(f"  Mean: {target_mean:.2f}")
    print(f"  Std: {target_std:.2f}")
    print(f"  Min: {float(all_targets.min()):.2f}")
    print(f"  Max: {float(all_targets.max()):.2f}")

    # Show possession distribution
    all_poss = data['possessions']
    print(f"\nPossessions per lineup:")
    print(f"  Mean: {float(all_poss.mean()):.0f}")
    print(f"  Median: {float(all_poss.median()):.0f}")
    print(f"  Max: {float(all_poss.max()):.0f}")

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
