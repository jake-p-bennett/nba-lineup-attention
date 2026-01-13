"""
Compute player-level statistics from play-by-play data.

For each player, compute:
- Offensive rating (points per 100 possessions while on court)
- Defensive rating (opponent points per 100 possessions while on court)
- Net rating
- Total possessions played
- Games played
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
MIN_YEAR = 2015  # Match the game processing


def load_data() -> pd.DataFrame:
    """Load play-by-play data."""
    print("Loading play-by-play data...")

    cols = [
        'GameID', 'Date', 'Possession',
        'AwayScore', 'HomeScore',
        'A1', 'A2', 'A3', 'A4', 'A5',
        'H1', 'H2', 'H3', 'H4', 'H5',
    ]

    df = pd.read_csv(DATA_DIR / 'all_games.csv', usecols=cols, low_memory=False)

    # Convert scores
    df['AwayScore'] = pd.to_numeric(df['AwayScore'], errors='coerce')
    df['HomeScore'] = pd.to_numeric(df['HomeScore'], errors='coerce')

    # Filter by year
    df['Year'] = df['Date'].str[-4:].astype(int)
    df = df[df['Year'] >= MIN_YEAR].copy()

    print(f"Loaded {len(df):,} plays from {df['Year'].min()}-{df['Year'].max()}")

    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player statistics from play-by-play."""
    print("Computing player statistics...")

    # Track stats per player
    player_stats = defaultdict(lambda: {
        'off_possessions': 0,
        'def_possessions': 0,
        'off_points': 0,
        'def_points': 0,
        'games': set(),
        'plus_minus': 0,
    })

    # Process game by game
    for game_id, game_df in tqdm(df.groupby('GameID'), desc="Processing games"):
        game_df = game_df.sort_values('Possession')

        # Get score changes between possessions
        game_df['AwayScorePrev'] = game_df['AwayScore'].shift(1).fillna(0)
        game_df['HomeScorePrev'] = game_df['HomeScore'].shift(1).fillna(0)
        game_df['AwayPtsScored'] = game_df['AwayScore'] - game_df['AwayScorePrev']
        game_df['HomePtsScored'] = game_df['HomeScore'] - game_df['HomeScorePrev']

        # Cap at reasonable values (filter out data errors)
        game_df['AwayPtsScored'] = game_df['AwayPtsScored'].clip(0, 4)
        game_df['HomePtsScored'] = game_df['HomePtsScored'].clip(0, 4)

        # Final score for plus/minus
        final_away = game_df['AwayScore'].dropna().iloc[-1] if len(game_df['AwayScore'].dropna()) > 0 else 0
        final_home = game_df['HomeScore'].dropna().iloc[-1] if len(game_df['HomeScore'].dropna()) > 0 else 0

        # Track each player's contribution
        for _, play in game_df.iterrows():
            away_pts = play['AwayPtsScored'] if pd.notna(play['AwayPtsScored']) else 0
            home_pts = play['HomePtsScored'] if pd.notna(play['HomePtsScored']) else 0

            # Away players
            for col in ['A1', 'A2', 'A3', 'A4', 'A5']:
                player = play[col]
                if pd.notna(player):
                    stats = player_stats[player]
                    stats['off_possessions'] += 1
                    stats['def_possessions'] += 1
                    stats['off_points'] += away_pts
                    stats['def_points'] += home_pts
                    stats['games'].add(game_id)

            # Home players
            for col in ['H1', 'H2', 'H3', 'H4', 'H5']:
                player = play[col]
                if pd.notna(player):
                    stats = player_stats[player]
                    stats['off_possessions'] += 1
                    stats['def_possessions'] += 1
                    stats['off_points'] += home_pts
                    stats['def_points'] += away_pts
                    stats['games'].add(game_id)

    # Convert to DataFrame
    records = []
    for player_id, stats in player_stats.items():
        if stats['off_possessions'] < 100:  # Minimum threshold
            continue

        off_rating = (stats['off_points'] / stats['off_possessions']) * 100
        def_rating = (stats['def_points'] / stats['def_possessions']) * 100
        net_rating = off_rating - def_rating

        records.append({
            'player_id': player_id,
            'off_rating': off_rating,
            'def_rating': def_rating,
            'net_rating': net_rating,
            'possessions': stats['off_possessions'],
            'games_played': len(stats['games']),
        })

    df_stats = pd.DataFrame(records)

    # Normalize stats to have mean 0, std 1
    for col in ['off_rating', 'def_rating', 'net_rating', 'possessions', 'games_played']:
        mean = df_stats[col].mean()
        std = df_stats[col].std()
        df_stats[f'{col}_norm'] = (df_stats[col] - mean) / std

    print(f"Computed stats for {len(df_stats)} players")

    return df_stats


def main():
    df = load_data()
    df_stats = compute_stats(df)

    # Save
    output_path = DATA_DIR / 'player_stats_computed.parquet'
    df_stats.to_parquet(output_path)

    print(f"\nSaved to {output_path}")
    print(f"\nSample stats:")
    print(df_stats.head(10).to_string())

    print(f"\nStats summary:")
    print(df_stats[['off_rating', 'def_rating', 'net_rating', 'possessions', 'games_played']].describe())


if __name__ == "__main__":
    main()
