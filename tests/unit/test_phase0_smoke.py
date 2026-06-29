"""Phase 0 smoke tests: refactored ECL engine and vectorized calculator."""
import re

import pytest

from risk_engine.config import AVG_EXPOSURE, HAZARD_RATE, LGD, RESULTS_DIR, TIME_HORIZON
from risk_engine.monte_carlo.ecl_engine import (
    pd_from_macro_inputs,
    portfolio_ecl_from_defaults,
    probability_of_default,
)
from risk_engine.monte_carlo.vectorized_calc import run_simulation

def test_baseline_pd_matches_legacy_hazard_rate():
    """At baseline macros, PD should match the original fixed HAZARD_RATE formula."""
    expected_pd = probability_of_default(HAZARD_RATE, TIME_HORIZON)
    actual_pd = pd_from_macro_inputs()
    assert actual_pd == pytest.approx(expected_pd, rel=1e-9)

def test_portfolio_ecl_formula():
    defaults = 1000
    expected = defaults * AVG_EXPOSURE * LGD
    assert portfolio_ecl_from_defaults(defaults) == expected

def test_vectorized_run_simulation_small_portfolio():
    results = run_simulation(n_loans=50_000, seed=42)

    assert results["total_loans"] == 50_000
    assert 0 <= results["defaults"] <= 50_000
    assert results["expected_credit_loss"] == portfolio_ecl_from_defaults(
        results["defaults"]
    )

    expected_pd = pd_from_macro_inputs()
    observed_rate = results["default_rate"]
    # Monte Carlo noise: allow ~15% relative tolerance on default rate
    assert observed_rate == pytest.approx(expected_pd, rel=0.15)

def test_vectorized_results_file_written():
    """After run_simulation(), results/vectorized_results.txt must exist with key fields."""
    output_file = RESULTS_DIR / "vectorized_results.txt"
    assert output_file.is_file(), f"Missing {output_file}"

    text = output_file.read_text(encoding="utf-8")
    assert "Total Loans:" in text
    assert "Defaults:" in text
    assert "Default Rate:" in text
    assert "Expected Credit Loss:" in text
    assert "NumPy Vectorization" in text
    assert "Macro Scenario:" in text

    defaults_match = re.search(r"Defaults:\s*([\d,]+)", text)
    ecl_match = re.search(r"Expected Credit Loss:\s*\$([\d,]+\.\d{2})", text)
    rate_match = re.search(r"Default Rate:\s*([\d.]+)", text)
    total_loans_match = re.search(r"Total Loans:\s*([\d,]+)", text)

    assert defaults_match and ecl_match and rate_match and total_loans_match

    defaults = int(defaults_match.group(1).replace(",", ""))
    ecl = float(ecl_match.group(1).replace(",", ""))
    default_rate = float(rate_match.group(1))
    total_loans = int(total_loans_match.group(1).replace(",", ""))

    assert ecl == pytest.approx(portfolio_ecl_from_defaults(defaults), rel=1e-9)
    assert default_rate == pytest.approx(defaults / total_loans, rel=1e-6)

    expected_pd = pd_from_macro_inputs()
    assert default_rate == pytest.approx(expected_pd, rel=0.02)
