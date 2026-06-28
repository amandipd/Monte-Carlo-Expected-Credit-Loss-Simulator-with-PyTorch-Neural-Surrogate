"""Validate synthetic ECL training dataset quality."""
import argparse
import sys
from pathlib import Path

import pandas as pd

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import MACRO_BOUNDS, N_SAMPLES, SYNTHETIC_DATASET_PATH, TRAINING_N_LOANS
from ai_surrogate.generate_training_data import DATASET_COLUMNS
from ai_surrogate.sampling import FEATURE_NAMES
from computations.ecl_engine import compute_ecl


class ValidationError(Exception):
    """Raised when dataset validation fails."""


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = SYNTHETIC_DATASET_PATH if path is None else Path(path)
    if not path.is_file():
        raise ValidationError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(
    df: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    n_loans: int | None = None,
    seed: int | None = 42,
    spot_check_count: int = 3,
) -> dict:
    """
    Validate dataset schema, completeness, bounds, labels, and spot-check rows.

    Returns summary statistics. Raises ValidationError on failure.
    """
    expected_rows = N_SAMPLES if expected_rows is None else expected_rows
    n_loans = TRAINING_N_LOANS if n_loans is None else n_loans

    if list(df.columns) != DATASET_COLUMNS:
        raise ValidationError(
            f"Unexpected columns: {list(df.columns)} (expected {DATASET_COLUMNS})"
        )

    if len(df) == 0:
        raise ValidationError("Dataset is empty")

    if expected_rows is not None and len(df) != expected_rows:
        raise ValidationError(
            f"Row count mismatch: got {len(df):,}, expected {expected_rows:,}"
        )

    nan_counts = df.isna().sum()
    if nan_counts.any():
        raise ValidationError(f"NaN values found:\n{nan_counts[nan_counts > 0]}")

    for feature in FEATURE_NAMES:
        low, high = MACRO_BOUNDS[feature]
        col_min = df[feature].min()
        col_max = df[feature].max()
        if col_min < low or col_max > high:
            raise ValidationError(
                f"{feature} out of bounds [{low}, {high}]: min={col_min}, max={col_max}"
            )

    labels = df["expected_credit_loss"]
    if (labels <= 0).any():
        raise ValidationError("expected_credit_loss must be strictly positive")

    summary = {
        "row_count": len(df),
        "feature_ranges": {
            feature: {"min": float(df[feature].min()), "max": float(df[feature].max())}
            for feature in FEATURE_NAMES
        },
        "label_stats": {
            "min": float(labels.min()),
            "max": float(labels.max()),
            "mean": float(labels.mean()),
            "std": float(labels.std()),
        },
        "spot_checks": [],
    }

    if spot_check_count > 0:
        if seed is None:
            raise ValidationError("seed is required for spot-check validation")

        indices = _spot_check_indices(len(df), spot_check_count)
        for idx in indices:
            row = df.iloc[idx]
            recomputed = compute_ecl(
                unemployment=float(row["unemployment_rate"]),
                interest_rate=float(row["interest_rate"]),
                hpi=float(row["housing_price_index"]),
                n_loans=n_loans,
                seed=seed + idx,
            )
            stored = float(row["expected_credit_loss"])
            if recomputed != stored:
                raise ValidationError(
                    f"Spot-check failed at row {idx}: stored={stored}, recomputed={recomputed}"
                )
            summary["spot_checks"].append(
                {"row_index": idx, "expected_credit_loss": stored}
            )

    return summary


def _spot_check_indices(n_rows: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    if count == 2:
        return [0, n_rows - 1]

    middle = n_rows // 2
    indices = [0, middle, n_rows - 1]
    return sorted(set(indices))[:count]


def print_summary(summary: dict, path: Path) -> None:
    print(f"Dataset validation passed: {path}")
    print(f"  Rows: {summary['row_count']:,}")
    print("  Feature ranges:")
    for feature, stats in summary["feature_ranges"].items():
        print(f"    {feature}: [{stats['min']:.4f}, {stats['max']:.4f}]")
    label_stats = summary["label_stats"]
    print("  expected_credit_loss:")
    print(f"    min:  ${label_stats['min']:,.2f}")
    print(f"    max:  ${label_stats['max']:,.2f}")
    print(f"    mean: ${label_stats['mean']:,.2f}")
    print(f"    std:  ${label_stats['std']:,.2f}")
    if summary["spot_checks"]:
        checked = ", ".join(str(item["row_index"]) for item in summary["spot_checks"])
        print(f"  Spot-check rows revalidated: {checked}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate synthetic ECL training dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=SYNTHETIC_DATASET_PATH,
        help="Path to synthetic_ecl_dataset.csv",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help=f"Expected row count (default: {N_SAMPLES} from config).",
    )
    parser.add_argument(
        "--n-loans",
        type=int,
        default=None,
        help=f"Loans used during labeling (default: {TRAINING_N_LOANS} from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during dataset generation (for spot-checks).",
    )
    parser.add_argument(
        "--spot-checks",
        type=int,
        default=3,
        help="Number of rows to revalidate against compute_ecl().",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        df = load_dataset(args.input)
        summary = validate_dataset(
            df,
            expected_rows=args.expected_rows,
            n_loans=args.n_loans,
            seed=args.seed,
            spot_check_count=args.spot_checks,
        )
        print_summary(summary, args.input)
    except ValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
