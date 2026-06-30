"""Shared pytest fixtures for the surrogate pipeline."""

import pytest
from fastapi.testclient import TestClient

from risk_engine.surrogate.app import app
from risk_engine.surrogate.cache import ECLCache
from risk_engine.surrogate.generate_training_data import generate_dataset
from risk_engine.surrogate.train import TrainConfig, train_model
from risk_engine.testing.fakes import FakeRedis


@pytest.fixture
def trained_artifacts(tmp_path):
    """Train a small surrogate on a tiny synthetic dataset."""
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "surrogate_v1.pt"
    scaler_path = tmp_path / "scaler_v1.pkl"

    generate_dataset(n_samples=40, n_loans=5_000, output_path=csv_path, seed=11)
    train_model(
        TrainConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            epochs=30,
            batch_size=8,
            patience=10,
            seed=11,
        )
    )
    return model_path, scaler_path


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """FastAPI TestClient with a freshly trained model and in-memory cache."""
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "surrogate_v1.pt"
    scaler_path = tmp_path / "scaler_v1.pkl"

    generate_dataset(n_samples=40, n_loans=5_000, output_path=csv_path, seed=21)
    train_model(
        TrainConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            epochs=40,
            batch_size=8,
            patience=10,
            seed=21,
        )
    )

    monkeypatch.setattr(
        "risk_engine.surrogate.inference.DEFAULT_MODEL_PATH",
        model_path,
    )
    monkeypatch.setattr(
        "risk_engine.surrogate.inference.DEFAULT_SCALER_PATH",
        scaler_path,
    )
    fake_cache = ECLCache(enabled=True, ttl_seconds=86400, redis_client=FakeRedis())
    monkeypatch.setattr(
        "risk_engine.surrogate.app.ECLCache.connect",
        lambda: fake_cache,
    )

    with TestClient(app) as client:
        yield client
