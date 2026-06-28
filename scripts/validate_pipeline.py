"""End-to-end validation: generate → train → evaluate → API smoke test."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
TESTS = PROJECT_ROOT / "tests"
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient

from ai_surrogate.app import app
from ai_surrogate.cache import ECLCache
from ai_surrogate.evaluate import EvalConfig, evaluate_model
from ai_surrogate.generate_training_data import generate_dataset
from ai_surrogate.train import TrainConfig, train_model
from conftest import FakeRedis


def main() -> int:
    os.environ.setdefault("LLM_MOCK", "true")

    print("=== ECL surrogate pipeline validation ===")

    with tempfile.TemporaryDirectory(prefix="ecl_pipeline_") as tmp_dir:
        tmp = Path(tmp_dir)
        dataset_path = tmp / "data.csv"
        model_path = tmp / "surrogate_v1.pt"
        scaler_path = tmp / "scaler_v1.pkl"

        print("[1/4] Generating synthetic dataset...")
        generate_dataset(
            n_samples=80,
            n_loans=5_000,
            output_path=dataset_path,
            seed=99,
        )

        print("[2/4] Training surrogate...")
        train_model(
            TrainConfig(
                dataset_path=dataset_path,
                model_path=model_path,
                scaler_path=scaler_path,
                epochs=60,
                batch_size=8,
                patience=15,
                seed=99,
            )
        )

        print("[3/4] Evaluating surrogate...")
        summary = evaluate_model(
            EvalConfig(
                dataset_path=dataset_path,
                model_path=model_path,
                scaler_path=scaler_path,
                seed=99,
                label_seed=99,
                n_loans=5_000,
            )
        )
        if not summary.passed:
            print("Evaluation FAILED validation gates.", file=sys.stderr)
            return 1
        print(
            f"  Evaluation passed (val MAE ratio {summary.val_mae_ratio:.2%})"
        )

        print("[4/4] API smoke test...")
        import ai_surrogate.app as app_module
        import ai_surrogate.inference as inference_module

        inference_module.DEFAULT_MODEL_PATH = model_path
        inference_module.DEFAULT_SCALER_PATH = scaler_path
        app_module.ECLCache.connect = lambda: ECLCache(
            enabled=True,
            ttl_seconds=3600,
            redis_client=FakeRedis(),
        )

        with TestClient(app) as client:
            predict = client.post(
                "/api/v2/predict",
                json={
                    "unemployment_rate": 6.5,
                    "interest_rate": 5.25,
                    "housing_price_index": 95.0,
                },
            )
            shock = client.post(
                "/api/v2/predict_shock",
                json={
                    "scenario_description": (
                        "Moderate slowdown with rising unemployment."
                    ),
                },
            )

        if predict.status_code != 200 or shock.status_code != 200:
            print("API smoke test FAILED.", file=sys.stderr)
            print("predict:", predict.status_code, predict.text, file=sys.stderr)
            print("predict_shock:", shock.status_code, shock.text, file=sys.stderr)
            return 1

        print(f"  /api/v2/predict ECL=${predict.json()['predicted_ecl']:,.0f}")
        print("  /api/v2/predict_shock returned executive summary")

    print("=== Pipeline validation PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
