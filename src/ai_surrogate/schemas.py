"""Pydantic schemas for the ECL surrogate API."""
from pydantic import BaseModel, Field

from config import MACRO_BOUNDS

_UNEMPLOYMENT_MIN, _UNEMPLOYMENT_MAX = MACRO_BOUNDS["unemployment_rate"]
_INTEREST_MIN, _INTEREST_MAX = MACRO_BOUNDS["interest_rate"]
_HPI_MIN, _HPI_MAX = MACRO_BOUNDS["housing_price_index"]


class PredictRequest(BaseModel):
    unemployment_rate: float = Field(
        ...,
        ge=_UNEMPLOYMENT_MIN,
        le=_UNEMPLOYMENT_MAX,
        description="Unemployment rate (percent).",
    )
    interest_rate: float = Field(
        ...,
        ge=_INTEREST_MIN,
        le=_INTEREST_MAX,
        description="Interest rate (percent).",
    )
    housing_price_index: float = Field(
        ...,
        ge=_HPI_MIN,
        le=_HPI_MAX,
        description="Housing price index level.",
    )


class MacroCoordinates(BaseModel):
    unemployment_rate: float
    interest_rate: float
    housing_price_index: float


class PredictResponse(BaseModel):
    input_macro_coordinates: MacroCoordinates
    predicted_ecl: float
    inference_ms: float
    cached: bool = Field(
        default=False,
        description="True when predicted_ecl was served from Redis cache.",
    )


class PredictShockRequest(BaseModel):
    scenario_description: str = Field(
        ...,
        min_length=1,
        description="Natural-language description of an economic crisis scenario.",
    )


class PredictShockResponse(BaseModel):
    input_macro_coordinates: MacroCoordinates
    predicted_ecl: float
    executive_summary: str
    inference_ms: float
    cached: bool = Field(
        default=False,
        description="True when predicted_ecl was served from Redis cache.",
    )
