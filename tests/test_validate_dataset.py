"""Tests for dataset validation."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.generate_training_data import DATASET_COLUMNS, generate_dataset
from ai_surrogate.validate_dataset import ValidationError, load_dataset, validate_dataset


def test_validate_dataset_passes_generated_file(tmp_path):
    output = tmp_path / "synthetic_ecl_dataset.csv"
    generate_dataset(n_samples=10, n_loans=10_000, output_path=output, seed=42)

    df = load_dataset(output)
    summary = validate_dataset(df, expected_rows=10, n_loans=10_000, seed=42)

    assert summary["row_count"] == 10
    assert len(summary["spot_checks"]) == 3


def test_validate_dataset_missing_file(tmp_path):
    with pytest.raises(ValidationError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.csv")


def test_validate_dataset_detects_wrong_columns(tmp_path):
    output = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(output, index=False)
    df = load_dataset(output)

    with pytest.raises(ValidationError, match="Unexpected columns"):
        validate_dataset(df, expected_rows=1, spot_check_count=0)


def test_validate_dataset_detects_nans(tmp_path):
    output = tmp_path / "bad.csv"
    df = pd.DataFrame(
        {
            "unemployment_rate": [4.0],
            "interest_rate": [3.0],
            "housing_price_index": [100.0],
            "expected_credit_loss": [None],
        }
    )
    df.to_csv(output, index=False)

    with pytest.raises(ValidationError, match="NaN values found"):
        validate_dataset(load_dataset(output), expected_rows=1, spot_check_count=0)


def test_validate_dataset_detects_row_count_mismatch(tmp_path):
    output = tmp_path / "synthetic_ecl_dataset.csv"
    generate_dataset(n_samples=5, n_loans=10_000, output_path=output, seed=1)

    with pytest.raises(ValidationError, match="Row count mismatch"):
        validate_dataset(load_dataset(output), expected_rows=10, spot_check_count=0)
