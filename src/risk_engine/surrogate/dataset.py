"""PyTorch Dataset for ECL surrogate training."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from risk_engine.surrogate.generate_training_data import DATASET_COLUMNS
from risk_engine.surrogate.sampling import FEATURE_NAMES

LABEL_COLUMN = "expected_credit_loss"

def load_training_dataframe(path: Path) -> pd.DataFrame:
    """Load and validate the synthetic training CSV."""
    df = pd.read_csv(path)
    if list(df.columns) != DATASET_COLUMNS:
        raise ValueError(
            f"Unexpected columns: {list(df.columns)} (expected {DATASET_COLUMNS})"
        )
    if df.isna().any().any():
        raise ValueError("Training data contains NaN values")
    return df

def split_features_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.float64)
    return features, labels

class ECLDataset(Dataset):
    """Tensor dataset of macro features and ECL labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
