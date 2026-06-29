"""
Load simulation settings from environment / .env.
Edit .env in the project root to change N_LOANS, HAZARD_RATE, TIME_HORIZON, Redis, etc.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_env_path = PROJECT_ROOT / ".env"
load_dotenv(_env_path)

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_ecl_dataset.csv"

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

# Synthetic training data generation
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
TRAINING_N_LOANS = int(os.getenv("TRAINING_N_LOANS", "500000"))
TRAINING_LABEL_SEED = int(os.getenv("TRAINING_LABEL_SEED", "42"))
EVAL_MAE_THRESHOLD = float(os.getenv("EVAL_MAE_THRESHOLD", "0.05"))
EVAL_SPOT_CHECK_TOLERANCE = float(os.getenv("EVAL_SPOT_CHECK_TOLERANCE", "0.10"))
EVAL_SPOT_CHECK_COUNT = int(os.getenv("EVAL_SPOT_CHECK_COUNT", "5"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ECL inference cache (separate Redis keys from simulation job queues)
ECL_CACHE_ENABLED = os.getenv("ECL_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
ECL_CACHE_TTL = int(os.getenv("ECL_CACHE_TTL", "86400"))

# LLM agent layer (Ollama — free, local; no paid API key required)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_MOCK = os.getenv("LLM_MOCK", "false").lower() in {"1", "true", "yes"}
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# How many jobs to split work into (more jobs = more workers can run in parallel)
N_JOBS = int(os.getenv("N_JOBS", "10"))

