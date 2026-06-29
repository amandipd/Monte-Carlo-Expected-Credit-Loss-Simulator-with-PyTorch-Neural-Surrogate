"""Tests for surrogate evaluation."""
import sys

import pytest

from risk_engine.surrogate.evaluate import EvalConfig, evaluate_model, main
from risk_engine.surrogate.generate_training_data import generate_dataset
from risk_engine.surrogate.train import TrainConfig, train_model

def test_evaluate_model_passes_with_trained_artifacts(tmp_path):
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "surrogate_v1.pt"
    scaler_path = tmp_path / "scaler_v1.pkl"

    generate_dataset(n_samples=80, n_loans=5_000, output_path=csv_path, seed=42)
    train_model(
        TrainConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            epochs=200,
            batch_size=16,
            patience=25,
            seed=42,
        )
    )

    summary = evaluate_model(
        EvalConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            n_loans=5_000,
            label_seed=42,
            mae_threshold_fraction=0.15,
            spot_check_tolerance_fraction=0.20,
            spot_check_count=3,
        )
    )

    assert summary.mae_passed
    assert summary.spot_check_passed
    assert summary.passed

def test_evaluate_model_fails_with_loose_threshold(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "surrogate_v1.pt"
    scaler_path = tmp_path / "scaler_v1.pkl"

    generate_dataset(n_samples=40, n_loans=5_000, output_path=csv_path, seed=3)
    train_model(
        TrainConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            epochs=30,
            batch_size=8,
            patience=10,
            seed=3,
        )
    )

    summary = evaluate_model(
        EvalConfig(
            dataset_path=csv_path,
            model_path=model_path,
            scaler_path=scaler_path,
            n_loans=5_000,
            label_seed=3,
            mae_threshold_fraction=0.0001,
            spot_check_tolerance_fraction=0.0001,
        )
    )
    assert summary.passed is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--dataset",
            str(csv_path),
            "--model",
            str(model_path),
            "--scaler",
            str(scaler_path),
            "--mae-threshold",
            "0.0001",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
