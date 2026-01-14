"""
Training script for NBA lineup models.

Usage:
    python src/train.py --model baseline
    python src/train.py --model attention
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import BaselineModel, LineupTransformer, count_parameters


# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_data(batch_size: int = 64):
    """Load preprocessed data."""
    metadata = torch.load(DATA_DIR / "metadata.pt")

    def load_split(name: str) -> TensorDataset:
        data = torch.load(DATA_DIR / f"{name}.pt")
        return TensorDataset(
            data["player_indices"],
            data["targets"],
            data["minutes"],
        )

    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, metadata


def create_model(model_type: str, num_players: int) -> nn.Module:
    """Create model by type."""
    if model_type == "baseline":
        return BaselineModel(num_players, embedding_dim=64, hidden_dim=128)
    elif model_type == "attention":
        return LineupTransformer(
            num_players,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            dropout=0.1,
        )
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
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for player_indices, targets, minutes in loader:
        player_indices = player_indices.to(device)
        targets = targets.to(device)
        minutes = minutes.to(device)

        # Normalize targets
        targets_norm = (targets - target_mean) / target_std

        # Forward
        optimizer.zero_grad()
        predictions, _ = model(player_indices, return_attention=False) if hasattr(model, 'layers') else (model(player_indices), None)

        # MSE loss, optionally weighted by minutes
        # Weight by sqrt(minutes) to give more weight to reliable samples
        weights = torch.sqrt(minutes)
        weights = weights / weights.sum() * len(weights)  # Normalize

        loss = (weights * (predictions - targets_norm) ** 2).mean()

        # Backward
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
    """Evaluate model, return metrics."""
    model.eval()

    all_preds = []
    all_targets = []

    for player_indices, targets, minutes in loader:
        player_indices = player_indices.to(device)

        if hasattr(model, 'layers'):
            predictions, _ = model(player_indices, return_attention=False)
        else:
            predictions = model(player_indices)

        # Denormalize predictions
        predictions = predictions * target_std + target_mean

        all_preds.append(predictions.cpu())
        all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Metrics
    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = (all_preds - all_targets).abs().mean().item()

    # R² score
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    # Win prediction accuracy
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
    patience: int = 10,
    device: str = "auto",
):
    """Main training function."""
    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, metadata = load_data(batch_size)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    target_mean = metadata["target_mean"]
    target_std = metadata["target_std"]
    num_players = metadata["num_players"]

    # Create model
    model = create_model(model_type, num_players).to(device)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_metrics": []}

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("\nTraining...")
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, target_mean, target_std)

        # Validate
        val_metrics = evaluate(model, val_loader, device, target_mean, target_std)

        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        # Print progress
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"Win Acc: {val_metrics['win_accuracy']:.1%}"
        )

        # Learning rate scheduling
        scheduler.step(val_metrics["rmse"])

        # Early stopping
        if val_metrics["rmse"] < best_val_loss:
            best_val_loss = val_metrics["rmse"]
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "metadata": metadata,
            }, CHECKPOINT_DIR / f"{model_type}_best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    checkpoint = torch.load(CHECKPOINT_DIR / f"{model_type}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, target_mean, target_std)
    print(f"\n{'='*50}")
    print(f"TEST RESULTS ({model_type})")
    print(f"{'='*50}")
    print(f"RMSE:         {test_metrics['rmse']:.2f}")
    print(f"MAE:          {test_metrics['mae']:.2f}")
    print(f"R²:           {test_metrics['r2']:.4f}")
    print(f"Win Accuracy: {test_metrics['win_accuracy']:.1%}")

    # Save final results
    results = {
        "model_type": model_type,
        "test_metrics": test_metrics,
        "best_val_metrics": checkpoint["val_metrics"],
        "epochs_trained": checkpoint["epoch"] + 1,
        "parameters": count_parameters(model),
    }

    with open(CHECKPOINT_DIR / f"{model_type}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, test_metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "attention"])
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
