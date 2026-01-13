"""
Training script for hybrid models with RICH stats (7 features).

Usage:
    python src/train_hybrid_rich.py --model baseline
    python src/train_hybrid_rich.py --model attention
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models_hybrid import HybridBaseline, HybridAttention, count_parameters, count_embedding_norm


DATA_DIR = Path(__file__).parent.parent / "data" / "games_hybrid_rich_processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_data(batch_size: int = 64):
    """Load preprocessed hybrid data with rich stats."""
    metadata = torch.load(DATA_DIR / "metadata.pt")

    def load_split(name: str) -> TensorDataset:
        data = torch.load(DATA_DIR / f"{name}.pt")
        return TensorDataset(
            data["away_indices"],
            data["home_indices"],
            data["away_stats"],
            data["home_stats"],
            data["away_weights"],
            data["home_weights"],
            data["targets"],
        )

    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, metadata


def create_model(model_type: str, num_players: int, num_features: int) -> nn.Module:
    """Create model by type."""
    if model_type == "baseline":
        return HybridBaseline(num_players, stat_dim=num_features, embed_dim=32, hidden_dim=64)
    elif model_type == "attention":
        return HybridAttention(num_players, stat_dim=num_features, embed_dim=32, hidden_dim=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mean: float,
    target_std: float,
    emb_reg: float = 0.0,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_emb_loss = 0.0

    for away_idx, home_idx, away_stats, home_stats, away_w, home_w, targets in loader:
        away_idx = away_idx.to(device)
        home_idx = home_idx.to(device)
        away_stats = away_stats.to(device)
        home_stats = home_stats.to(device)
        away_w = away_w.to(device)
        home_w = home_w.to(device)
        targets = targets.to(device)

        # Normalize targets
        targets_norm = (targets - target_mean) / target_std

        optimizer.zero_grad()

        # Forward
        if isinstance(model, HybridAttention):
            predictions = model(away_idx, home_idx, away_stats, home_stats, away_w, home_w)
        else:
            predictions = model(away_idx, home_idx, away_stats, home_stats, away_w, home_w)

        # MSE loss
        mse_loss = ((predictions - targets_norm) ** 2).mean()

        # Embedding regularization: L2 penalty on embedding weights
        emb_loss = 0.0
        if emb_reg > 0 and hasattr(model, 'player_embedding'):
            emb_loss = emb_reg * (model.player_embedding.weight ** 2).mean()

        loss = mse_loss + emb_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += mse_loss.item()
        total_emb_loss += emb_loss.item() if isinstance(emb_loss, torch.Tensor) else emb_loss

    return total_loss / len(loader), total_emb_loss / len(loader)


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

    for away_idx, home_idx, away_stats, home_stats, away_w, home_w, targets in loader:
        away_idx = away_idx.to(device)
        home_idx = home_idx.to(device)
        away_stats = away_stats.to(device)
        home_stats = home_stats.to(device)
        away_w = away_w.to(device)
        home_w = home_w.to(device)

        if isinstance(model, HybridAttention):
            predictions = model(away_idx, home_idx, away_stats, home_stats, away_w, home_w)
        else:
            predictions = model(away_idx, home_idx, away_stats, home_stats, away_w, home_w)

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
    patience: int = 15,
    device: str = "auto",
    emb_reg: float = 0.0,
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
    num_players = metadata["num_players"]
    num_features = metadata["num_features"]

    print(f"Players: {num_players}, Features: {num_features}")
    print(f"Feature names: {metadata['feature_names']}")
    print(f"Target mean: {target_mean:.2f}, std: {target_std:.2f}")
    print(f"Embedding regularization: {emb_reg}")

    # Create model
    model = create_model(model_type, num_players, num_features).to(device)
    print(f"\nModel: hybrid_rich_{model_type}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Initial embedding norm: {count_embedding_norm(model):.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    suffix = f"_reg{emb_reg}" if emb_reg > 0 else ""

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("\nTraining...")
    for epoch in range(epochs):
        train_loss, emb_loss = train_epoch(model, train_loader, optimizer, device, target_mean, target_std, emb_reg)
        val_metrics = evaluate(model, val_loader, device, target_mean, target_std)

        emb_norm = count_embedding_norm(model)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"Win Acc: {val_metrics['win_accuracy']:.1%} | "
            f"Emb: {emb_norm:.1f}"
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
                "embedding_norm": emb_norm,
            }, CHECKPOINT_DIR / f"hybrid_rich_{model_type}{suffix}_best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best and evaluate on test
    checkpoint = torch.load(CHECKPOINT_DIR / f"hybrid_rich_{model_type}{suffix}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, target_mean, target_std)
    final_emb_norm = count_embedding_norm(model)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (hybrid_rich_{model_type}{suffix})")
    print(f"{'='*60}")
    print(f"RMSE:         {test_metrics['rmse']:.2f} points")
    print(f"MAE:          {test_metrics['mae']:.2f} points")
    print(f"R²:           {test_metrics['r2']:.4f}")
    print(f"Win Accuracy: {test_metrics['win_accuracy']:.1%}")
    print(f"Embedding Norm: {final_emb_norm:.2f} (started at ~0.18)")

    # Save results
    results = {
        "model_type": f"hybrid_rich_{model_type}{suffix}",
        "test_metrics": test_metrics,
        "best_val_metrics": checkpoint["val_metrics"],
        "epochs_trained": checkpoint["epoch"] + 1,
        "parameters": count_parameters(model),
        "final_embedding_norm": final_emb_norm,
        "emb_reg": emb_reg,
        "num_features": num_features,
        "feature_names": metadata["feature_names"],
    }

    with open(CHECKPOINT_DIR / f"hybrid_rich_{model_type}{suffix}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "attention"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--emb-reg", type=float, default=0.0, help="L2 regularization on embeddings")
    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        emb_reg=args.emb_reg,
    )
