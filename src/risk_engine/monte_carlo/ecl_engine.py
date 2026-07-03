"""
Parameterized Expected Credit Loss (ECL) engine.

Maps macroeconomic inputs to a hazard rate, runs a vectorized Monte Carlo
default simulation, and returns total portfolio ECL.
"""

import numpy as np

from risk_engine.config import (
    AVG_EXPOSURE,
    BASE_HAZARD_RATE,
    LGD,
    MACRO_BASELINE_HPI,
    MACRO_BASELINE_INTEREST,
    MACRO_BASELINE_UNEMPLOYMENT,
    MACRO_BOUNDS,
    TIME_HORIZON,
)

def clip_macro_inputs(
    unemployment: float,
    interest_rate: float,
    hpi: float,
) -> tuple[float, float, float]:
    """Clip macro features to configured safe bounds."""
    u_min, u_max = MACRO_BOUNDS["unemployment_rate"]
    i_min, i_max = MACRO_BOUNDS["interest_rate"]
    h_min, h_max = MACRO_BOUNDS["housing_price_index"]
    return (
        float(np.clip(unemployment, u_min, u_max)),
        float(np.clip(interest_rate, i_min, i_max)),
        float(np.clip(hpi, h_min, h_max)),
    )

def macro_to_hazard_rate(
    unemployment: float,
    interest_rate: float,
    hpi: float,
) -> float:
    """
    Map macro features to an annual hazard rate.

    At baseline macro levels the hazard rate equals BASE_HAZARD_RATE.
    Higher unemployment and interest rates increase hazard; lower HPI increases hazard.
    """
    unemployment, interest_rate, hpi = clip_macro_inputs(
        unemployment, interest_rate, hpi
    )

    u_ref = MACRO_BASELINE_UNEMPLOYMENT
    i_ref = MACRO_BASELINE_INTEREST
    hpi_ref = MACRO_BASELINE_HPI

    unemployment_factor = 1.0 + 0.5 * (unemployment - u_ref) / max(u_ref, 1e-6)
    interest_factor = 1.0 + 0.25 * (interest_rate - i_ref) / max(i_ref, 1e-6)
    hpi_factor = 1.0 + 0.5 * (hpi_ref - hpi) / max(hpi_ref, 1e-6)

    hazard_rate = (
        BASE_HAZARD_RATE * unemployment_factor * interest_factor * hpi_factor
    )
    return max(float(hazard_rate), 1e-6)

def probability_of_default(
    hazard_rate: float,
    time_horizon: float | None = None,
) -> float:
    """PD = 1 - exp(-hazard_rate * time_horizon)."""
    horizon = TIME_HORIZON if time_horizon is None else time_horizon
    return float(1.0 - np.exp(-hazard_rate * horizon))

def default_macro_inputs() -> tuple[float, float, float]:
    """Baseline macro scenario (neutral stress; hazard multipliers = 1.0)."""
    return (
        MACRO_BASELINE_UNEMPLOYMENT,
        MACRO_BASELINE_INTEREST,
        MACRO_BASELINE_HPI,
    )

def resolve_macro_inputs(
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
) -> tuple[float, float, float]:
    """Use baseline macros when any input is omitted."""
    if unemployment is None and interest_rate is None and hpi is None:
        return default_macro_inputs()
    if unemployment is None or interest_rate is None or hpi is None:
        raise ValueError(
            "Provide all macro inputs (unemployment, interest_rate, hpi) or none."
        )
    return unemployment, interest_rate, hpi

def pd_from_macro_inputs(
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
) -> float:
    """Probability of default for the given (or baseline) macro scenario."""
    unemployment, interest_rate, hpi = resolve_macro_inputs(
        unemployment, interest_rate, hpi
    )
    hazard_rate = macro_to_hazard_rate(unemployment, interest_rate, hpi)
    return probability_of_default(hazard_rate)

def portfolio_ecl_from_defaults(defaults: int) -> float:
    """Convert a simulated default count to total portfolio ECL."""
    return float(defaults * AVG_EXPOSURE * LGD)

def simulate_defaults(
    n_loans: int,
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
    seed: int | None = None,
) -> int:
    """Run a vectorized Monte Carlo default simulation."""
    if n_loans <= 0:
        raise ValueError("n_loans must be a positive integer")

    pd = pd_from_macro_inputs(unemployment, interest_rate, hpi)

    rng = np.random.default_rng(seed)
    random_rolls = rng.random(n_loans)
    return int(np.count_nonzero(random_rolls < pd))

def compute_ecl(
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
    n_loans: int = 0,
    seed: int | None = None,
) -> float:
    """
    Run a vectorized Monte Carlo simulation and return total portfolio ECL.

    ECL = default_count * AVG_EXPOSURE * LGD
    """
    defaults = simulate_defaults(
        n_loans,
        unemployment=unemployment,
        interest_rate=interest_rate,
        hpi=hpi,
        seed=seed,
    )
    return portfolio_ecl_from_defaults(defaults)
