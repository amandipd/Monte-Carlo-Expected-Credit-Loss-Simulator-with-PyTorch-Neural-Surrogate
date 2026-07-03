"""Unit tests for the parameterized ECL engine."""

import pytest

from risk_engine.config import (
    BASE_HAZARD_RATE,
    MACRO_BASELINE_HPI,
    MACRO_BASELINE_INTEREST,
    MACRO_BASELINE_UNEMPLOYMENT,
    MACRO_BOUNDS,
)
from risk_engine.monte_carlo.ecl_engine import (
    clip_macro_inputs,
    compute_ecl,
    macro_to_hazard_rate,
    pd_from_macro_inputs,
    simulate_defaults,
)

def test_compute_ecl_is_deterministic_with_fixed_seed():
    first = compute_ecl(6.0, 5.0, 95.0, n_loans=10_000, seed=123)
    second = compute_ecl(6.0, 5.0, 95.0, n_loans=10_000, seed=123)
    assert first == second
    assert first > 0

def test_simulate_defaults_is_deterministic_with_fixed_seed():
    first = simulate_defaults(10_000, 6.0, 5.0, 95.0, seed=99)
    second = simulate_defaults(10_000, 6.0, 5.0, 95.0, seed=99)
    assert first == second

def test_macro_to_hazard_rate_at_baseline():
    hazard = macro_to_hazard_rate(
        MACRO_BASELINE_UNEMPLOYMENT,
        MACRO_BASELINE_INTEREST,
        MACRO_BASELINE_HPI,
    )
    assert hazard == pytest.approx(BASE_HAZARD_RATE, rel=1e-9)

def test_clip_macro_inputs_enforces_bounds():
    u_min, u_max = MACRO_BOUNDS["unemployment_rate"]
    i_min, i_max = MACRO_BOUNDS["interest_rate"]
    h_min, h_max = MACRO_BOUNDS["housing_price_index"]

    unemployment, interest, hpi = clip_macro_inputs(-1.0, 99.0, 200.0)
    assert unemployment == u_min
    assert interest == i_max
    assert hpi == h_max

def test_stressed_macros_increase_pd():
    baseline_pd = pd_from_macro_inputs()
    stressed_pd = pd_from_macro_inputs(10.0, 8.0, 80.0)
    assert stressed_pd > baseline_pd

def test_compute_ecl_rejects_invalid_n_loans():
    with pytest.raises(ValueError, match="n_loans"):
        compute_ecl(4.0, 3.0, 100.0, n_loans=0, seed=1)
