"""Tests for ECL surrogate inference."""

import pytest

from risk_engine.surrogate.inference import SurrogatePredictor, load_predictor, predict_ecl
from risk_engine.config import MACRO_BOUNDS

def test_load_predictor(trained_artifacts):
    model_path, scaler_path = trained_artifacts
    predictor = load_predictor(model_path, scaler_path)
    assert isinstance(predictor, SurrogatePredictor)

def test_predict_ecl_returns_float(trained_artifacts):
    model_path, scaler_path = trained_artifacts
    ecl = predict_ecl(4.0, 3.0, 100.0, model_path=model_path, scaler_path=scaler_path)
    assert isinstance(ecl, float)
    assert ecl > 0

def test_predict_ecl_clips_out_of_bounds_inputs(trained_artifacts):
    model_path, scaler_path = trained_artifacts
    predictor = load_predictor(model_path, scaler_path)

    clipped_ecl = predictor.predict_ecl(-5.0, 99.0, 200.0)
    bounded_ecl = predictor.predict_ecl(
        MACRO_BOUNDS["unemployment_rate"][0],
        MACRO_BOUNDS["interest_rate"][1],
        MACRO_BOUNDS["housing_price_index"][1],
    )

    assert clipped_ecl == bounded_ecl

def test_predict_ecl_reuses_predictor(trained_artifacts):
    model_path, scaler_path = trained_artifacts
    predictor = load_predictor(model_path, scaler_path)

    first = predictor.predict_ecl(6.0, 5.0, 95.0)
    second = predict_ecl(6.0, 5.0, 95.0, predictor=predictor)

    assert first == second

def test_load_predictor_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_predictor(tmp_path / "missing.pt", tmp_path / "missing.pkl")
