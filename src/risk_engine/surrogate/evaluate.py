"""Evaluate trained ECL surrogate against validation metrics and engine spot-checks."""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from risk_engine.config import (
    EVAL_MAE_THRESHOLD,
    EVAL_SPOT_CHECK_COUNT,
    EVAL_SPOT_CHECK_TOLERANCE,
    SYNTHETIC_DATASET_PATH,
    TRAINING_LABEL_SEED,
    TRAINING_N_LOANS,
)
from risk_engine.surrogate.dataset import load_training_dataframe, split_features_labels
from risk_engine.surrogate.inference import SurrogatePredictor, load_predictor
from risk_engine.surrogate.train import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH
from risk_engine.monte_carlo.ecl_engine import compute_ecl

class EvaluationError(Exception):
    """Raised when the surrogate fails validation gates."""

@dataclass
class EvalConfig:
    dataset_path: Path = SYNTHETIC_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    scaler_path: Path = DEFAULT_SCALER_PATH
    val_fraction: float = 0.2
    seed: int = 42
    label_seed: int = TRAINING_LABEL_SEED
    n_loans: int = TRAINING_N_LOANS
    mae_threshold_fraction: float = EVAL_MAE_THRESHOLD
    spot_check_count: int = EVAL_SPOT_CHECK_COUNT
    spot_check_tolerance_fraction: float = EVAL_SPOT_CHECK_TOLERANCE

@dataclass
class EvalSummary:
    val_mae: float
    val_mean_ecl: float
    val_mae_ratio: float
    mae_threshold_fraction: float
    mae_passed: bool
    spot_checks: list[dict]
    spot_check_passed: bool
    passed: bool

def _validation_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
    )
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def evaluate_model(
    config: EvalConfig | None = None,
    predictor: SurrogatePredictor | None = None,
) -> EvalSummary:
    config = EvalConfig() if config is None else config
    predictor = load_predictor(config.model_path, config.scaler_path) if predictor is None else predictor

    df = load_training_dataframe(config.dataset_path)
    _, val_df = _validation_split(df, config.val_fraction, config.seed)

    _, labels = split_features_labels(val_df)
    predictions = np.array(
        [
            predictor.predict_ecl(
                float(row.unemployment_rate),
                float(row.interest_rate),
                float(row.housing_price_index),
            )
            for row in val_df.itertuples(index=False)
        ],
        dtype=np.float64,
    )

    val_mae = float(np.mean(np.abs(predictions - labels)))
    val_mean_ecl = float(np.mean(labels))
    val_mae_ratio = val_mae / val_mean_ecl if val_mean_ecl > 0 else float("inf")
    mae_passed = val_mae_ratio < config.mae_threshold_fraction

    spot_indices = _spot_check_row_indices(len(val_df), config.spot_check_count)
    spot_checks: list[dict] = []
    spot_check_passed = True

    for position in spot_indices:
        row = val_df.iloc[position]
        original_index = val_df.index[position]
        stored_label = float(row["expected_credit_loss"])
        predicted = predictor.predict_ecl(
            float(row["unemployment_rate"]),
            float(row["interest_rate"]),
            float(row["housing_price_index"]),
        )
        engine_label = compute_ecl(
            unemployment=float(row["unemployment_rate"]),
            interest_rate=float(row["interest_rate"]),
            hpi=float(row["housing_price_index"]),
            n_loans=config.n_loans,
            seed=config.label_seed + int(original_index),
        )

        rel_error = abs(predicted - engine_label) / engine_label if engine_label > 0 else float("inf")
        row_passed = rel_error <= config.spot_check_tolerance_fraction
        spot_check_passed = spot_check_passed and row_passed

        spot_checks.append(
            {
                "row_index": int(original_index),
                "unemployment_rate": float(row["unemployment_rate"]),
                "interest_rate": float(row["interest_rate"]),
                "housing_price_index": float(row["housing_price_index"]),
                "stored_label": stored_label,
                "engine_label": engine_label,
                "predicted_ecl": predicted,
                "relative_error": rel_error,
                "passed": row_passed,
            }
        )

    passed = mae_passed and spot_check_passed
    return EvalSummary(
        val_mae=val_mae,
        val_mean_ecl=val_mean_ecl,
        val_mae_ratio=val_mae_ratio,
        mae_threshold_fraction=config.mae_threshold_fraction,
        mae_passed=mae_passed,
        spot_checks=spot_checks,
        spot_check_passed=spot_check_passed,
        passed=passed,
    )

def _spot_check_row_indices(n_rows: int, count: int) -> list[int]:
    if count <= 0 or n_rows == 0:
        return []
    if count == 1:
        return [0]
    if count == 2:
        return [0, n_rows - 1]
    middle = n_rows // 2
    indices = [0, middle, n_rows - 1]
    return sorted(set(indices))[:count]

def print_summary(summary: EvalSummary) -> None:
    print("Surrogate evaluation summary")
    print(f"  Val MAE:        ${summary.val_mae:,.2f}")
    print(f"  Val mean ECL:   ${summary.val_mean_ecl:,.2f}")
    print(
        f"  Val MAE ratio:  {summary.val_mae_ratio:.2%} "
        f"(threshold < {summary.mae_threshold_fraction:.0%}) "
        f"{'PASS' if summary.mae_passed else 'FAIL'}"
    )
    print("  Spot-checks vs compute_ecl():")
    for check in summary.spot_checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"    row {check['row_index']:4d} | "
            f"engine ${check['engine_label']:,.0f} | "
            f"predicted ${check['predicted_ecl']:,.0f} | "
            f"error {check['relative_error']:.2%} | {status}"
        )
    print(f"  Overall: {'PASS' if summary.passed else 'FAIL'}")

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ECL surrogate.")
    parser.add_argument("--dataset", type=Path, default=SYNTHETIC_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=TRAINING_LABEL_SEED)
    parser.add_argument("--n-loans", type=int, default=TRAINING_N_LOANS)
    parser.add_argument("--mae-threshold", type=float, default=EVAL_MAE_THRESHOLD)
    parser.add_argument("--spot-checks", type=int, default=EVAL_SPOT_CHECK_COUNT)
    parser.add_argument(
        "--spot-check-tolerance",
        type=float,
        default=EVAL_SPOT_CHECK_TOLERANCE,
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = EvalConfig(
        dataset_path=args.dataset,
        model_path=args.model,
        scaler_path=args.scaler,
        val_fraction=args.val_fraction,
        seed=args.seed,
        label_seed=args.label_seed,
        n_loans=args.n_loans,
        mae_threshold_fraction=args.mae_threshold,
        spot_check_count=args.spot_checks,
        spot_check_tolerance_fraction=args.spot_check_tolerance,
    )

    try:
        summary = evaluate_model(config)
        print_summary(summary)
        if not summary.passed:
            raise EvaluationError("Surrogate failed one or more validation gates")
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
