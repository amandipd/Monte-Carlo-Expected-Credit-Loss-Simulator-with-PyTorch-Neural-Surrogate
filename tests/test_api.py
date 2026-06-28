"""Tests for FastAPI surrogate inference endpoint."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.app import app
from ai_surrogate.cache import ECLCache
from ai_surrogate.generate_training_data import generate_dataset
from ai_surrogate.inference import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH
from ai_surrogate.train import TrainConfig, train_model
from test_cache import FakeRedis


@pytest.fixture
def api_client(tmp_path, monkeypatch):
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
        "ai_surrogate.inference.DEFAULT_MODEL_PATH",
        model_path,
    )
    monkeypatch.setattr(
        "ai_surrogate.inference.DEFAULT_SCALER_PATH",
        scaler_path,
    )
    fake_cache = ECLCache(enabled=True, ttl_seconds=86400, redis_client=FakeRedis())
    monkeypatch.setattr(
        "ai_surrogate.app.ECLCache.connect",
        lambda: fake_cache,
    )

    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "cache_enabled" in body
    assert "cache_available" in body


def test_predict_endpoint(api_client):
    response = api_client.post(
        "/api/v2/predict",
        json={
            "unemployment_rate": 6.5,
            "interest_rate": 5.25,
            "housing_price_index": 95.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input_macro_coordinates"]["unemployment_rate"] == 6.5
    assert body["predicted_ecl"] > 0
    assert body["inference_ms"] >= 0
    assert body["cached"] is False


def test_predict_uses_cache_on_repeat(api_client):
    payload = {
        "unemployment_rate": 5.0,
        "interest_rate": 4.0,
        "housing_price_index": 98.0,
    }
    first = api_client.post("/api/v2/predict", json=payload)
    second = api_client.post("/api/v2/predict", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert first.json()["predicted_ecl"] == second.json()["predicted_ecl"]


def test_predict_validation_error(api_client):
    response = api_client.post(
        "/api/v2/predict",
        json={
            "unemployment_rate": -1.0,
            "interest_rate": 5.0,
            "housing_price_index": 100.0,
        },
    )
    assert response.status_code == 422


def test_predict_shock_endpoint(api_client, monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "true")

    response = api_client.post(
        "/api/v2/predict_shock",
        json={
            "scenario_description": (
                "A severe crisis with high unemployment, rate hikes, and a housing crash."
            ),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input_macro_coordinates"]["unemployment_rate"] == 10.0
    assert body["predicted_ecl"] > 0
    assert body["executive_summary"]
    assert body["inference_ms"] >= 0


@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.is_file() or not DEFAULT_SCALER_PATH.is_file(),
    reason="Production model artifacts not present",
)
def test_predict_with_production_model():
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/predict",
            json={
                "unemployment_rate": 4.0,
                "interest_rate": 3.0,
                "housing_price_index": 100.0,
            },
        )
    assert response.status_code == 200
    assert response.json()["predicted_ecl"] > 0
