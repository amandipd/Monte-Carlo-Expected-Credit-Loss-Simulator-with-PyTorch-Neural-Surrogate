"""Train the ECL surrogate model on synthetic data."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from risk_engine.config import MODELS_DIR, SYNTHETIC_DATASET_PATH
from risk_engine.surrogate.dataset import ECLDataset, load_training_dataframe, split_features_labels
from risk_engine.surrogate.model import ECLSurrogate
from risk_engine.surrogate.scalers import save_scaler_artifact

DEFAULT_MODEL_PATH = MODELS_DIR / "surrogate_v1.pt"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"

@dataclass
class TrainConfig:
    dataset_path: Path = SYNTHETIC_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    scaler_path: Path = DEFAULT_SCALER_PATH
    val_fraction: float = 0.2
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 30
    seed: int = 42

@dataclass
class TrainMetrics:
    train_mse: float
    train_mae: float
    val_mse: float
    val_mae: float
    epochs_run: int

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def _compute_metrics(
    model: ECLSurrogate,
    loader: DataLoader,
    device: torch.device,
    label_scaler: StandardScaler,
) -> tuple[float, float]:
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)

            pred_orig = label_scaler.inverse_transform(predictions.cpu().numpy())
            label_orig = label_scaler.inverse_transform(labels.cpu().numpy())

            diff = pred_orig - label_orig
            total_squared_error += float(np.square(diff).sum())
            total_absolute_error += float(np.abs(diff).sum())
            total_samples += len(features)

    mse = total_squared_error / total_samples
    mae = total_absolute_error / total_samples
    return mse, mae

def train_model(config: TrainConfig | None = None) -> TrainMetrics:
    config = TrainConfig() if config is None else config
    _set_seed(config.seed)
    device = torch.device("cpu")

    df = load_training_dataframe(config.dataset_path)
    features, labels = split_features_labels(df)

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=config.val_fraction,
        random_state=config.seed,
    )

    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_val_scaled = feature_scaler.transform(x_val)
    y_train_scaled = label_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = label_scaler.transform(y_val.reshape(-1, 1)).ravel()

    train_loader = DataLoader(
        ECLDataset(x_train_scaled, y_train_scaled),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ECLDataset(x_val_scaled, y_val_scaled),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = ECLSurrogate().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_mse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

        train_mse, train_mae = _compute_metrics(model, train_loader, device, label_scaler)
        val_mse, val_mae = _compute_metrics(model, val_loader, device, label_scaler)

        if epoch == 1 or epoch % 50 == 0 or val_mse < best_val_mse:
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE: {train_mse:,.0f}  MAE: ${train_mae:,.0f} | "
                f"val MSE: {val_mse:,.0f}  MAE: ${val_mae:,.0f}"
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")

    model.load_state_dict(best_state)
    train_mse, train_mae = _compute_metrics(model, train_loader, device, label_scaler)
    val_mse, val_mae = _compute_metrics(model, val_loader, device, label_scaler)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    save_scaler_artifact(config.scaler_path, feature_scaler, label_scaler)

    print(f"Saved model to {config.model_path}")
    print(f"Saved scaler to {config.scaler_path}")
    print(
        f"Final best metrics | train MSE: {train_mse:,.0f}  MAE: ${train_mae:,.0f} | "
        f"val MSE: {val_mse:,.0f}  MAE: ${val_mae:,.0f}"
    )

    return TrainMetrics(
        train_mse=train_mse,
        train_mae=train_mae,
        val_mse=val_mse,
        val_mae=val_mae,
        epochs_run=epochs_run,
    )

def _parse_args():
    parser = argparse.ArgumentParser(description="Train ECL surrogate model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=SYNTHETIC_DATASET_PATH,
        help="Path to synthetic_ecl_dataset.csv",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for model weights",
    )
    parser.add_argument(
        "--scaler-out",
        type=Path,
        default=DEFAULT_SCALER_PATH,
        help="Output path for feature scaler",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = TrainConfig(
        dataset_path=args.dataset,
        model_path=args.model_out,
        scaler_path=args.scaler_out,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
    train_model(config)

if __name__ == "__main__":
    main()
