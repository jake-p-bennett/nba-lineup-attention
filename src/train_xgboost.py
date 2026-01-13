"""
XGBoost baseline for game-level prediction.

This serves as a sanity check: if XGBoost with simple averaged features
matches the neural network, it confirms the problem is fundamentally
about additive player contributions (no complex interactions to learn).

Usage:
    python src/train_xgboost.py
    python src/train_xgboost.py --feature-mode concat  # Use concatenated features
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_DIR = Path(__file__).parent.parent / "data" / "games_hybrid_rich_processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_data():
    """Load preprocessed game data."""
    metadata = torch.load(DATA_DIR / "metadata.pt")

    splits = {}
    for name in ["train", "val", "test"]:
        data = torch.load(DATA_DIR / f"{name}.pt")
        splits[name] = {
            "away_stats": data["away_stats"].numpy(),      # (n, 10, 7)
            "home_stats": data["home_stats"].numpy(),      # (n, 10, 7)
            "away_weights": data["away_weights"].numpy(),  # (n, 10)
            "home_weights": data["home_weights"].numpy(),  # (n, 10)
            "targets": data["targets"].numpy(),            # (n,)
        }

    return splits, metadata


def create_features_averaged(away_stats, home_stats, away_weights, home_weights):
    """
    Create features by computing weighted average of player stats per team.

    Returns: (n_samples, 14) array - 7 features per team
    """
    # Weighted average: (n, 10, 7) * (n, 10, 1) -> sum -> (n, 7)
    away_avg = np.sum(away_stats * away_weights[:, :, np.newaxis], axis=1)
    home_avg = np.sum(home_stats * home_weights[:, :, np.newaxis], axis=1)

    # Concatenate: (n, 14)
    return np.concatenate([away_avg, home_avg], axis=1)


def create_features_concat(away_stats, home_stats, away_weights, home_weights, top_k=5):
    """
    Create features by concatenating top-K players' stats.

    Returns: (n_samples, top_k * 7 * 2) array
    """
    # Take top-K players (already sorted by weight)
    away_top = away_stats[:, :top_k, :].reshape(away_stats.shape[0], -1)  # (n, top_k*7)
    home_top = home_stats[:, :top_k, :].reshape(home_stats.shape[0], -1)  # (n, top_k*7)

    # Also include weights as features
    away_w = away_weights[:, :top_k]  # (n, top_k)
    home_w = home_weights[:, :top_k]  # (n, top_k)

    return np.concatenate([away_top, away_w, home_top, home_w], axis=1)


def create_features_summary(away_stats, home_stats, away_weights, home_weights):
    """
    Create features using summary statistics (mean, std, min, max) per team.

    Returns: (n_samples, 7 * 4 * 2) = (n_samples, 56) array
    """
    def summarize(stats, weights):
        # Weighted mean
        weighted_mean = np.sum(stats * weights[:, :, np.newaxis], axis=1)
        # Unweighted stats for diversity
        std = np.std(stats, axis=1)
        min_val = np.min(stats, axis=1)
        max_val = np.max(stats, axis=1)
        return np.concatenate([weighted_mean, std, min_val, max_val], axis=1)

    away_summary = summarize(away_stats, away_weights)
    home_summary = summarize(home_stats, home_weights)

    return np.concatenate([away_summary, home_summary], axis=1)


def train_and_evaluate(feature_mode: str = "averaged"):
    """Train XGBoost and evaluate."""
    print(f"Loading data...")
    splits, metadata = load_data()

    # Select feature creation function
    if feature_mode == "averaged":
        create_features = create_features_averaged
        print("Feature mode: Weighted average (14 features)")
    elif feature_mode == "concat":
        create_features = lambda a, h, aw, hw: create_features_concat(a, h, aw, hw, top_k=10)
        print("Feature mode: Concatenated top-10 (160 features)")
    elif feature_mode == "summary":
        create_features = create_features_summary
        print("Feature mode: Summary statistics (56 features)")
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    # Create feature matrices
    X_train = create_features(
        splits["train"]["away_stats"],
        splits["train"]["home_stats"],
        splits["train"]["away_weights"],
        splits["train"]["home_weights"],
    )
    y_train = splits["train"]["targets"]

    X_val = create_features(
        splits["val"]["away_stats"],
        splits["val"]["home_stats"],
        splits["val"]["away_weights"],
        splits["val"]["home_weights"],
    )
    y_val = splits["val"]["targets"]

    X_test = create_features(
        splits["test"]["away_stats"],
        splits["test"]["home_stats"],
        splits["test"]["away_weights"],
        splits["test"]["home_weights"],
    )
    y_test = splits["test"]["targets"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train XGBoost with early stopping
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration}")

    # Evaluate on test set
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Win accuracy
    pred_home_win = y_pred > 0
    actual_home_win = y_test > 0
    win_accuracy = np.mean(pred_home_win == actual_home_win)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (XGBoost - {feature_mode})")
    print(f"{'='*60}")
    print(f"RMSE:         {rmse:.2f} points")
    print(f"MAE:          {mae:.2f} points")
    print(f"R²:           {r2:.4f}")
    print(f"Win Accuracy: {win_accuracy:.1%}")
    print(f"Features:     {X_train.shape[1]}")

    # Feature importance (top 10)
    if feature_mode == "averaged":
        feature_names = [
            "away_off_rating", "away_def_rating", "away_net_rating",
            "away_pts_per100", "away_reb_per100", "away_poss", "away_games",
            "home_off_rating", "home_def_rating", "home_net_rating",
            "home_pts_per100", "home_reb_per100", "home_poss", "home_games",
        ]
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        print(f"\nTop 10 Feature Importances:")
        for i in sorted_idx[:10]:
            print(f"  {feature_names[i]}: {importance[i]:.4f}")

    # Save results
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    results = {
        "model_type": f"xgboost_{feature_mode}",
        "test_metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "win_accuracy": float(win_accuracy),
        },
        "n_features": X_train.shape[1],
        "best_iteration": model.best_iteration,
    }

    with open(CHECKPOINT_DIR / f"xgboost_{feature_mode}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model.save_model(CHECKPOINT_DIR / f"xgboost_{feature_mode}.json")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="averaged",
        choices=["averaged", "concat", "summary"],
        help="How to create features from player stats"
    )
    args = parser.parse_args()

    train_and_evaluate(args.feature_mode)
