"""Local inference helper for the trained ECL surrogate."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import MODELS_DIR
from ai_surrogate.model import ECLSurrogate
from ai_surrogate.scalers import load_scaler_artifact
from computations.ecl_engine import clip_macro_inputs

DEFAULT_MODEL_PATH = MODELS_DIR / "surrogate_v1.pt"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"


@dataclass
class SurrogatePredictor:
    """Loaded surrogate model and feature scaler."""

    model: ECLSurrogate
    feature_scaler: StandardScaler
    label_scaler: StandardScaler
    device: torch.device

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        scaler_path: Path | None = None,
        device: torch.device | None = None,
    ) -> "SurrogatePredictor":
        model_path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
        scaler_path = DEFAULT_SCALER_PATH if scaler_path is None else Path(scaler_path)
        device = torch.device("cpu") if device is None else device

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        model = ECLSurrogate()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        feature_scaler, label_scaler = load_scaler_artifact(scaler_path)
        return cls(
            model=model,
            feature_scaler=feature_scaler,
            label_scaler=label_scaler,
            device=device,
        )

    def predict_ecl(
        self,
        unemployment: float,
        interest_rate: float,
        hpi: float,
    ) -> float:
        unemployment, interest_rate, hpi = clip_macro_inputs(
            unemployment, interest_rate, hpi
        )
        features = np.array(
            [[unemployment, interest_rate, hpi]],
            dtype=np.float64,
        )
        scaled = self.feature_scaler.transform(features)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prediction = self.model(tensor)

        ecl = self.label_scaler.inverse_transform(prediction.cpu().numpy())
        return float(ecl[0, 0])


def load_predictor(
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> SurrogatePredictor:
    """Load model and scaler from disk."""
    return SurrogatePredictor.load(model_path=model_path, scaler_path=scaler_path)


def predict_ecl(
    unemployment: float,
    interest_rate: float,
    hpi: float,
    *,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    predictor: SurrogatePredictor | None = None,
) -> float:
    """
    Predict portfolio ECL for macro inputs.

    Clips inputs to configured bounds before inference.
    """
    if predictor is None:
        predictor = load_predictor(model_path=model_path, scaler_path=scaler_path)
    return predictor.predict_ecl(unemployment, interest_rate, hpi)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run local ECL surrogate inference.")
    parser.add_argument("--unemployment", type=float, required=True)
    parser.add_argument("--interest", type=float, required=True)
    parser.add_argument("--hpi", type=float, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ecl = predict_ecl(
        args.unemployment,
        args.interest,
        args.hpi,
        model_path=args.model,
        scaler_path=args.scaler,
    )
    print(f"Predicted ECL: ${ecl:,.2f}")


if __name__ == "__main__":
    main()
