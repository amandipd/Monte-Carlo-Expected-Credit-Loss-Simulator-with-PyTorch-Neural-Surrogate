"""Tests for macro scenario sampling."""

import numpy as np
import pytest

from risk_engine.config import MACRO_BOUNDS, N_SAMPLES
from risk_engine.surrogate.sampling import FEATURE_NAMES, clip_macro_scenarios, sample_macro_scenarios

def test_sample_macro_scenarios_shape():
    rng = np.random.default_rng(0)
    scenarios = sample_macro_scenarios(50, rng)
    assert scenarios.shape == (50, 3)

def test_sample_macro_scenarios_within_bounds():
    rng = np.random.default_rng(1)
    scenarios = sample_macro_scenarios(200, rng)

    for idx, name in enumerate(FEATURE_NAMES):
        low, high = MACRO_BOUNDS[name]
        assert scenarios[:, idx].min() >= low
        assert scenarios[:, idx].max() <= high

def test_sample_macro_scenarios_reproducible_with_rng():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = sample_macro_scenarios(10, rng_a)
    b = sample_macro_scenarios(10, rng_b)
    np.testing.assert_array_equal(a, b)

def test_uniform_sampling_method():
    rng = np.random.default_rng(7)
    scenarios = sample_macro_scenarios(100, rng, method="uniform")
    assert scenarios.shape == (100, 3)

def test_invalid_sample_count_raises():
    with pytest.raises(ValueError, match="n must be a positive integer"):
        sample_macro_scenarios(0)

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_macro_scenarios(5, method="invalid")

def test_clip_macro_scenarios():
    lows = np.array([MACRO_BOUNDS[name][0] for name in FEATURE_NAMES])
    highs = np.array([MACRO_BOUNDS[name][1] for name in FEATURE_NAMES])
    raw = np.array([[-1.0, 99.0, 200.0]])
    clipped = clip_macro_scenarios(raw)
    assert clipped[0, 0] == pytest.approx(lows[0])
    assert clipped[0, 1] == pytest.approx(highs[1])
    assert clipped[0, 2] == pytest.approx(highs[2])

def test_n_samples_config_default():
    assert N_SAMPLES > 0
