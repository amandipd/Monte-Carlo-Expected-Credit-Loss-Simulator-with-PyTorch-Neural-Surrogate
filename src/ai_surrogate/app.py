"""FastAPI gateway for ECL surrogate inference."""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.agentic_translator import translate_scenario
from ai_surrogate.cache import ECLCache
from ai_surrogate.inference import SurrogatePredictor, load_predictor
from ai_surrogate.llm_client import LLMClientError
from ai_surrogate.report_synthesizer import synthesize_report
from ai_surrogate.schemas import (
    MacroCoordinates,
    PredictRequest,
    PredictResponse,
    PredictShockRequest,
    PredictShockResponse,
)
from computations.ecl_engine import clip_macro_inputs

APP_TITLE = "Distributed ECL Machine Learning Gateway"


def _predict_ecl_with_cache(
    predictor: SurrogatePredictor,
    cache: ECLCache,
    unemployment: float,
    interest_rate: float,
    hpi: float,
) -> tuple[float, float, bool]:
    """Return (predicted_ecl, inference_ms, cached)."""
    cached_ecl = cache.get(unemployment, interest_rate, hpi)
    if cached_ecl is not None:
        return cached_ecl, 0.0, True

    start = time.perf_counter()
    predicted_ecl = predictor.predict_ecl(unemployment, interest_rate, hpi)
    inference_ms = (time.perf_counter() - start) * 1000
    cache.set(unemployment, interest_rate, hpi, predicted_ecl)
    return predicted_ecl, inference_ms, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = load_predictor()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Surrogate model artifacts not found. Train the model before starting the API."
        ) from exc
    app.state.ecl_cache = ECLCache.connect()
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    cache: ECLCache = request.app.state.ecl_cache
    return {
        "status": "ok",
        "cache_enabled": cache.enabled,
        "cache_available": cache.available,
    }


@app.post("/api/v2/predict", response_model=PredictResponse)
async def predict(request_body: PredictRequest, request: Request) -> PredictResponse:
    predictor: SurrogatePredictor = request.app.state.predictor
    cache: ECLCache = request.app.state.ecl_cache

    unemployment, interest_rate, hpi = clip_macro_inputs(
        request_body.unemployment_rate,
        request_body.interest_rate,
        request_body.housing_price_index,
    )

    try:
        predicted_ecl, inference_ms, cached = _predict_ecl_with_cache(
            predictor,
            cache,
            unemployment,
            interest_rate,
            hpi,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(
        input_macro_coordinates=MacroCoordinates(
            unemployment_rate=unemployment,
            interest_rate=interest_rate,
            housing_price_index=hpi,
        ),
        predicted_ecl=predicted_ecl,
        inference_ms=inference_ms,
        cached=cached,
    )


@app.post("/api/v2/predict_shock", response_model=PredictShockResponse)
async def predict_shock(
    request_body: PredictShockRequest,
    request: Request,
) -> PredictShockResponse:
    predictor: SurrogatePredictor = request.app.state.predictor
    cache: ECLCache = request.app.state.ecl_cache

    try:
        unemployment, interest_rate, hpi = translate_scenario(
            request_body.scenario_description
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        predicted_ecl, inference_ms, cached = _predict_ecl_with_cache(
            predictor,
            cache,
            unemployment,
            interest_rate,
            hpi,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    try:
        executive_summary = synthesize_report(
            request_body.scenario_description,
            unemployment,
            interest_rate,
            hpi,
            predicted_ecl,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictShockResponse(
        input_macro_coordinates=MacroCoordinates(
            unemployment_rate=unemployment,
            interest_rate=interest_rate,
            housing_price_index=hpi,
        ),
        predicted_ecl=predicted_ecl,
        executive_summary=executive_summary,
        inference_ms=inference_ms,
        cached=cached,
    )
