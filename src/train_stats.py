"""
Training script for stats-based models.

Usage:
    python src/train_stats.py --model baseline
    python src/train_stats.py --model interaction
    python src/train_stats.py --model attention
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models_stats import StatsBaseline, StatsInteraction, StatsAttention, count_parameters


DATA_DIR = Path(__file__).parent.parent / "data" / "games_stats_processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_data(batch_size: int = 64):
    """Load preprocessed stats data."""
    metadata = torch.load(DATA_DIR / "metadata.pt")

    def load_split(name: str) -> TensorDataset:
        data = torch.load(DATA_DIR / f"{name}.pt")
        return TensorDataset(
            data["away_features"],
            data["home_features"],
            data["targets"],
        )

    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, metadata


def create_model(model_type: str, num_features: int) -> nn.Module:
    """Create model by type."""
    if model_type == "baseline":
        return StatsBaseline(num_features, hidden_dim=64)
    elif model_type == "interaction":
        return StatsInteraction(num_features, hidden_dim=64)
    elif model_type == "attention":
        return StatsAttention(num_features, hidden_dim=64, num_heads=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for away_feat, home_feat, targets in loader:
        away_feat = away_feat.to(device)
        home_feat = home_feat.to(device)
        targets = targets.to(device)

        # Normalize targets
        targets_norm = (targets - target_mean) / target_std

        optimizer.zero_grad()

        predictions = model(away_feat, home_feat)

        # MSE loss
        loss = ((predictions - targets_norm) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict:
    """Evaluate model."""
    model.eval()

    all_preds = []
    all_targets = []

    for away_feat, home_feat, targets in loader:
        away_feat = away_feat.to(device)
        home_feat = home_feat.to(device)

        predictions = model(away_feat, home_feat)

        # Denormalize
        predictions = predictions * target_std + target_mean

        all_preds.append(predictions.cpu())
        all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Metrics
    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = (all_preds - all_targets).abs().mean().item()

    # R²
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    # Win prediction accuracy (predict home win if pred > 0)
    pred_home_win = all_preds > 0
    actual_home_win = all_targets > 0
    win_accuracy = (pred_home_win == actual_home_win).float().mean().item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2.item(),
        "win_accuracy": win_accuracy,
    }


def train(
    model_type: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "auto",
):
    """Main training function."""
    if device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, metadata = load_data(batch_size)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    target_mean = metadata["target_mean"]
    target_std = metadata["target_std"]
    num_features = metadata["num_features"]

    print(f"Features: {metadata['feature_names']}")
    print(f"Target mean: {target_mean:.2f}, std: {target_std:.2f}")

    # Create model
    model = create_model(model_type, num_features).to(device)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_rmse = float("inf")
    epochs_without_improvement = 0

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("\nTraining...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, target_mean, target_std)
        val_metrics = evaluate(model, val_loader, device, target_mean, target_std)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"Win Acc: {val_metrics['win_accuracy']:.1%}"
        )

        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "metadata": metadata,
            }, CHECKPOINT_DIR / f"stats_{model_type}_best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best and evaluate on test
    checkpoint = torch.load(CHECKPOINT_DIR / f"stats_{model_type}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, target_mean, target_std)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({model_type})")
    print(f"{'='*60}")
    print(f"RMSE:         {test_metrics['rmse']:.2f} points")
    print(f"MAE:          {test_metrics['mae']:.2f} points")
    print(f"R²:           {test_metrics['r2']:.4f}")
    print(f"Win Accuracy: {test_metrics['win_accuracy']:.1%}")

    # Save results
    results = {
        "model_type": f"stats_{model_type}",
        "test_metrics": test_metrics,
        "best_val_metrics": checkpoint["val_metrics"],
        "epochs_trained": checkpoint["epoch"] + 1,
        "parameters": count_parameters(model),
    }

    with open(CHECKPOINT_DIR / f"stats_{model_type}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "interaction", "attention"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )
