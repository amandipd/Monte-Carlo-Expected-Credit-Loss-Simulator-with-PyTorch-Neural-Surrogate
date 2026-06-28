"""
Load simulation settings from environment / .env.
Edit .env in the project root to change N_LOANS, HAZARD_RATE, TIME_HORIZON, Redis, etc.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

PROJECT_ROOT = _env_path.parent
RESULTS_DIR = PROJECT_ROOT / "results"


N_LOANS = int(os.getenv("N_LOANS", "100000000"))


HAZARD_RATE = float(os.getenv("HAZARD_RATE", "0.05"))  # default 5% chance of default per year
TIME_HORIZON = float(os.getenv("TIME_HORIZON", "1.0"))  # default 1 year
BASE_HAZARD_RATE = float(os.getenv("BASE_HAZARD_RATE", os.getenv("HAZARD_RATE", "0.05")))
AVG_EXPOSURE = float(os.getenv("AVG_EXPOSURE", "250000"))  # dollars per loan
LGD = float(os.getenv("LGD", "0.45"))  # loss given default (fraction)
MACRO_BASELINE_UNEMPLOYMENT = float(os.getenv("MACRO_BASELINE_UNEMPLOYMENT", "4.0"))
MACRO_BASELINE_INTEREST = float(os.getenv("MACRO_BASELINE_INTEREST", "3.0"))
MACRO_BASELINE_HPI = float(os.getenv("MACRO_BASELINE_HPI", "100.0"))

# Macro sampling / clipping bounds (percentages for rates; index level for HPI)
MACRO_BOUNDS = {
    "unemployment_rate": (
        float(os.getenv("MACRO_UNEMPLOYMENT_MIN", "2.0")),
        float(os.getenv("MACRO_UNEMPLOYMENT_MAX", "15.0")),
    ),
    "interest_rate": (
        float(os.getenv("MACRO_INTEREST_MIN", "0.0")),
        float(os.getenv("MACRO_INTEREST_MAX", "12.0")),
    ),
    "housing_price_index": (
        float(os.getenv("MACRO_HPI_MIN", "70.0")),
        float(os.getenv("MACRO_HPI_MAX", "130.0")),
    ),
}


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# How many jobs to split work into (more jobs = more workers can run in parallel)
N_JOBS = int(os.getenv("N_JOBS", "10"))

