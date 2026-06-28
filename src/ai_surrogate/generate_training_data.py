"""Generate synthetic ECL training dataset using the Monte Carlo engine."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import N_SAMPLES, SYNTHETIC_DATASET_PATH, TRAINING_N_LOANS
from ai_surrogate.sampling import FEATURE_NAMES, sample_macro_scenarios
from computations.ecl_engine import compute_ecl

DATASET_COLUMNS = [*FEATURE_NAMES, "expected_credit_loss"]


def generate_dataset(
    n_samples: int | None = None,
    n_loans: int | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
    method: str = "latin_hypercube",
) -> pd.DataFrame:
    """
    Sample macro scenarios and label each row with compute_ecl().

    Returns a DataFrame with columns:
        unemployment_rate, interest_rate, housing_price_index, expected_credit_loss
    """
    n_samples = N_SAMPLES if n_samples is None else n_samples
    n_loans = TRAINING_N_LOANS if n_loans is None else n_loans
    output_path = SYNTHETIC_DATASET_PATH if output_path is None else Path(output_path)

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if n_loans <= 0:
        raise ValueError("n_loans must be a positive integer")

    rng = np.random.default_rng(seed)
    scenarios = sample_macro_scenarios(n_samples, rng, method=method)

    labels: list[float] = []
    start_time = time.time()
    for idx, (unemployment, interest_rate, hpi) in enumerate(scenarios):
        row_seed = None if seed is None else seed + idx
        ecl = compute_ecl(
            unemployment=float(unemployment),
            interest_rate=float(interest_rate),
            hpi=float(hpi),
            n_loans=n_loans,
            seed=row_seed,
        )
        labels.append(ecl)

        if (idx + 1) % max(1, n_samples // 10) == 0 or idx + 1 == n_samples:
            print(f"  Labeled {idx + 1:,} / {n_samples:,} scenarios")

    df = pd.DataFrame(scenarios, columns=list(FEATURE_NAMES))
    df["expected_credit_loss"] = labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"Wrote {len(df):,} rows to {output_path} in {elapsed:.2f}s")
    return df


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic macro → ECL training dataset."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=f"Number of scenarios to generate (default: {N_SAMPLES} from config).",
    )
    parser.add_argument(
        "--n-loans",
        type=int,
        default=None,
        help=f"Loans per label simulation (default: {TRAINING_N_LOANS} from config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SYNTHETIC_DATASET_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--method",
        choices=("latin_hypercube", "uniform"),
        default="latin_hypercube",
        help="Macro sampling method.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    print(
        f"Generating dataset: n_samples={args.n_samples or N_SAMPLES:,}, "
        f"n_loans={args.n_loans or TRAINING_N_LOANS:,}"
    )
    generate_dataset(
        n_samples=args.n_samples,
        n_loans=args.n_loans,
        output_path=args.output,
        seed=args.seed,
        method=args.method,
    )


if __name__ == "__main__":
    main()
