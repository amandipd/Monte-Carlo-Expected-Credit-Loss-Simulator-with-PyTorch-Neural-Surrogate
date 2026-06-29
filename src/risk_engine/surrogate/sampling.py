"""Macro scenario sampling for synthetic training data."""

import numpy as np
from numpy.random import Generator
from scipy.stats import qmc

from risk_engine.config import MACRO_BOUNDS

FEATURE_NAMES = ("unemployment_rate", "interest_rate", "housing_price_index")

def _bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lows = np.array([MACRO_BOUNDS[name][0] for name in FEATURE_NAMES], dtype=np.float64)
    highs = np.array([MACRO_BOUNDS[name][1] for name in FEATURE_NAMES], dtype=np.float64)
    return lows, highs

def clip_macro_scenarios(scenarios: np.ndarray) -> np.ndarray:
    """Clip sampled scenarios to configured macro bounds."""
    scenarios = np.asarray(scenarios, dtype=np.float64)
    if scenarios.ndim != 2 or scenarios.shape[1] != 3:
        raise ValueError("scenarios must have shape (n, 3)")

    lows, highs = _bounds_arrays()
    return np.clip(scenarios, lows, highs)

def sample_macro_scenarios(
    n: int,
    rng: Generator | None = None,
    *,
    method: str = "latin_hypercube",
) -> np.ndarray:
    """
    Sample macro scenarios within configured bounds.

    Returns array of shape (n, 3):
        [unemployment_rate, interest_rate, housing_price_index]
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng() if rng is None else rng
    lows, highs = _bounds_arrays()

    if method == "uniform":
        samples = rng.uniform(lows, highs, size=(n, 3))
    elif method == "latin_hypercube":
        seed = int(rng.integers(0, 2**31 - 1))
        engine = qmc.LatinHypercube(d=3, seed=seed)
        unit_samples = engine.random(n)
        samples = qmc.scale(unit_samples, lows, highs)
    else:
        raise ValueError(f"Unknown sampling method: {method!r}. Use 'uniform' or 'latin_hypercube'.")

    return clip_macro_scenarios(samples)
