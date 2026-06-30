"""Tests for synthetic dataset generation."""
from pathlib import Path

import pandas as pd
import pytest

from risk_engine.surrogate.generate_training_data import DATASET_COLUMNS, generate_dataset
from risk_engine.surrogate.sampling import FEATURE_NAMES
from risk_engine.monte_carlo.ecl_engine import compute_ecl

def test_generate_dataset_schema(tmp_path):
    output = tmp_path / "synthetic_ecl_dataset.csv"
    df = generate_dataset(
        n_samples=5,
        n_loans=10_000,
        output_path=output,
        seed=42,
    )

    assert list(df.columns) == DATASET_COLUMNS
    assert len(df) == 5
    assert output.is_file()

    loaded = pd.read_csv(output)
    assert list(loaded.columns) == DATASET_COLUMNS
    assert loaded.isna().sum().sum() == 0

def test_generate_dataset_labels_match_engine(tmp_path):
    output = tmp_path / "synthetic_ecl_dataset.csv"
    df = generate_dataset(
        n_samples=3,
        n_loans=20_000,
        output_path=output,
        seed=7,
    )

    for idx, row in df.iterrows():
        expected = compute_ecl(
            unemployment=row["unemployment_rate"],
            interest_rate=row["interest_rate"],
            hpi=row["housing_price_index"],
            n_loans=20_000,
            seed=7 + idx,
        )
        assert row["expected_credit_loss"] == expected

def test_generate_dataset_invalid_n_samples():
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        generate_dataset(n_samples=0, n_loans=1000, output_path=Path("unused.csv"))
