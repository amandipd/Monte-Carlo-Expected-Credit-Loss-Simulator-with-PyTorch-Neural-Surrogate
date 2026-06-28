"""Load and save feature/label scaler artifacts."""
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler


def save_scaler_artifact(
    path: Path,
    feature_scaler: StandardScaler,
    label_scaler: StandardScaler,
) -> None:
    joblib.dump(
        {"feature_scaler": feature_scaler, "label_scaler": label_scaler},
        path,
    )


def load_scaler_artifact(path: Path) -> tuple[StandardScaler, StandardScaler]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Unsupported scaler artifact at {path}. Retrain with the current pipeline."
        )
    return artifact["feature_scaler"], artifact["label_scaler"]
