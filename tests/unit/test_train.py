"""Tests for ECL training dataset and training pipeline."""

import joblib
import pytest
import torch

from risk_engine.surrogate.dataset import ECLDataset, load_training_dataframe, split_features_labels
from risk_engine.surrogate.generate_training_data import generate_dataset
from risk_engine.surrogate.model import ECLSurrogate
from risk_engine.surrogate.train import TrainConfig, train_model

def test_ecl_dataset_shapes():
    import numpy as np

    features = np.zeros((10, 3), dtype=np.float64)
    labels = np.ones(10, dtype=np.float64)
    dataset = ECLDataset(features, labels)

    x, y = dataset[0]
    assert x.shape == (3,)
    assert y.shape == (1,)
    assert len(dataset) == 10

def test_load_training_dataframe(tmp_path):
    csv_path = tmp_path / "data.csv"
    generate_dataset(n_samples=8, n_loans=5_000, output_path=csv_path, seed=1)

    df = load_training_dataframe(csv_path)
    features, labels = split_features_labels(df)

    assert features.shape == (8, 3)
    assert labels.shape == (8,)

def test_train_model_writes_artifacts(tmp_path):
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "surrogate_v1.pt"
    scaler_path = tmp_path / "scaler_v1.pkl"

    generate_dataset(n_samples=40, n_loans=5_000, output_path=csv_path, seed=2)

    metrics = train_model(
        TrainConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            epochs=50,
            batch_size=8,
            patience=10,
            seed=2,
        )
    )

    assert model_path.is_file()
    assert scaler_path.is_file()
    assert metrics.val_mae > 0

    model = ECLSurrogate()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    artifact = joblib.load(scaler_path)

    assert isinstance(artifact, dict)
    assert artifact["feature_scaler"].mean_.shape == (3,)
    assert artifact["label_scaler"].mean_.shape == (1,)

    x = torch.tensor(
        artifact["feature_scaler"].transform([[4.0, 3.0, 100.0]]),
        dtype=torch.float32,
    )
    y = model(x)
    assert y.shape == (1, 1)
