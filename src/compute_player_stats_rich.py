"""
Compute RICH player-level statistics from play-by-play data.

Adds traditional box score stats (pts, ast, reb, stl, blk) to the
basic efficiency metrics.

For each player, compute:
- Offensive rating, Defensive rating, Net rating
- Points, Assists, Rebounds, Steals, Blocks (per 100 possessions)
- Total possessions, Games played
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re

DATA_DIR = Path(__file__).parent.parent / "data"
MIN_YEAR = 2015


def load_data() -> pd.DataFrame:
    """Load play-by-play data."""
    print("Loading play-by-play data...")

    cols = [
        'GameID', 'Date', 'Possession',
        'AwayScore', 'HomeScore',
        'AwayEvent', 'HomeEvent',
        'A1', 'A2', 'A3', 'A4', 'A5',
        'H1', 'H2', 'H3', 'H4', 'H5',
        'ActivePlayers',
    ]

    df = pd.read_csv(DATA_DIR / 'all_games.csv', usecols=cols, low_memory=False)

    df['AwayScore'] = pd.to_numeric(df['AwayScore'], errors='coerce')
    df['HomeScore'] = pd.to_numeric(df['HomeScore'], errors='coerce')

    df['Year'] = df['Date'].str[-4:].astype(int)
    df = df[df['Year'] >= MIN_YEAR].copy()

    print(f"Loaded {len(df):,} plays from {df['Year'].min()}-{df['Year'].max()}")

    return df


def parse_box_score_events(df: pd.DataFrame) -> dict:
    """
    Parse play-by-play events to extract box score stats.

    Returns dict: player_id -> {pts, reb}

    Note: Only points and rebounds are reliably parseable from the play-by-play.
    """
    print("Parsing box score events...")

    player_box = defaultdict(lambda: {'pts': 0, 'reb': 0})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing events"):
        for event_col, player_cols in [('AwayEvent', ['A1','A2','A3','A4','A5']),
                                        ('HomeEvent', ['H1','H2','H3','H4','H5'])]:
            event = row[event_col]
            if pd.isna(event):
                continue
            event = str(event)

            # Get players on court for this team
            players_on_court = [row[c] for c in player_cols if pd.notna(row[c])]

            # Try to match events to players using ActivePlayers field
            active = row.get('ActivePlayers', '')
            if pd.notna(active):
                try:
                    active_list = eval(active) if isinstance(active, str) else active
                except:
                    active_list = []
            else:
                active_list = []

            # Points
            if 'makes 2-pt' in event:
                for player in active_list:
                    if player in players_on_court:
                        player_box[player]['pts'] += 2
                        break
            elif 'makes 3-pt' in event:
                for player in active_list:
                    if player in players_on_court:
                        player_box[player]['pts'] += 3
                        break
            elif 'makes free throw' in event:
                for player in active_list:
                    if player in players_on_court:
                        player_box[player]['pts'] += 1
                        break

            # Rebounds
            if 'rebound' in event.lower():
                for player in active_list:
                    if player in players_on_court:
                        player_box[player]['reb'] += 1
                        break

    return player_box


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
    })

    # Process game by game for efficiency stats
    for game_id, game_df in tqdm(df.groupby('GameID'), desc="Processing games"):
        game_df = game_df.sort_values('Possession')

        game_df['AwayScorePrev'] = game_df['AwayScore'].shift(1).fillna(0)
        game_df['HomeScorePrev'] = game_df['HomeScore'].shift(1).fillna(0)
        game_df['AwayPtsScored'] = game_df['AwayScore'] - game_df['AwayScorePrev']
        game_df['HomePtsScored'] = game_df['HomeScore'] - game_df['HomeScorePrev']

        game_df['AwayPtsScored'] = game_df['AwayPtsScored'].clip(0, 4)
        game_df['HomePtsScored'] = game_df['HomePtsScored'].clip(0, 4)

        for _, play in game_df.iterrows():
            away_pts = play['AwayPtsScored'] if pd.notna(play['AwayPtsScored']) else 0
            home_pts = play['HomePtsScored'] if pd.notna(play['HomePtsScored']) else 0

            for col in ['A1', 'A2', 'A3', 'A4', 'A5']:
                player = play[col]
                if pd.notna(player):
                    stats = player_stats[player]
                    stats['off_possessions'] += 1
                    stats['def_possessions'] += 1
                    stats['off_points'] += away_pts
                    stats['def_points'] += home_pts
                    stats['games'].add(game_id)

            for col in ['H1', 'H2', 'H3', 'H4', 'H5']:
                player = play[col]
                if pd.notna(player):
                    stats = player_stats[player]
                    stats['off_possessions'] += 1
                    stats['def_possessions'] += 1
                    stats['off_points'] += home_pts
                    stats['def_points'] += away_pts
                    stats['games'].add(game_id)

    # Parse box score events
    player_box = parse_box_score_events(df)

    # Convert to DataFrame
    records = []
    for player_id, stats in player_stats.items():
        if stats['off_possessions'] < 100:
            continue

        poss = stats['off_possessions']
        off_rating = (stats['off_points'] / poss) * 100
        def_rating = (stats['def_points'] / poss) * 100
        net_rating = off_rating - def_rating

        # Box score per 100 possessions (only pts and reb work from play-by-play parsing)
        box = player_box.get(player_id, {'pts': 0, 'reb': 0})
        pts_per100 = (box.get('pts', 0) / poss) * 100
        reb_per100 = (box.get('reb', 0) / poss) * 100

        records.append({
            'player_id': player_id,
            'off_rating': off_rating,
            'def_rating': def_rating,
            'net_rating': net_rating,
            'pts_per100': pts_per100,
            'reb_per100': reb_per100,
            'possessions': poss,
            'games_played': len(stats['games']),
        })

    df_stats = pd.DataFrame(records)

    # Normalize all stats (7 features total)
    stat_cols = ['off_rating', 'def_rating', 'net_rating',
                 'pts_per100', 'reb_per100',
                 'possessions', 'games_played']

    for col in stat_cols:
        mean = df_stats[col].mean()
        std = df_stats[col].std()
        df_stats[f'{col}_norm'] = (df_stats[col] - mean) / std

    print(f"Computed stats for {len(df_stats)} players")

    return df_stats


def main():
    df = load_data()
    df_stats = compute_stats(df)

    output_path = DATA_DIR / 'player_stats_rich.parquet'
    df_stats.to_parquet(output_path)

    print(f"\nSaved to {output_path}")
    print(f"\nFeatures: {[c for c in df_stats.columns if c.endswith('_norm')]}")
    print(f"\nSample stats:")
    print(df_stats.head(10).to_string())

    print(f"\nStats summary:")
    raw_cols = ['off_rating', 'def_rating', 'net_rating', 'pts_per100', 'reb_per100']
    print(df_stats[raw_cols].describe())


if __name__ == "__main__":
    main()
