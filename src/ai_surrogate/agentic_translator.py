"""Translate natural-language crisis scenarios into macro coordinates."""
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import (
    MACRO_BASELINE_HPI,
    MACRO_BASELINE_INTEREST,
    MACRO_BASELINE_UNEMPLOYMENT,
)
from ai_surrogate.llm_client import LLMClientError, OllamaClient
from ai_surrogate.prompts import TRANSLATION_SYSTEM_PROMPT, TRANSLATION_USER_PROMPT
from computations.ecl_engine import clip_macro_inputs


def mock_translate_scenario(scenario_description: str) -> tuple[float, float, float]:
    """Deterministic mock translator for tests and offline development."""
    text = scenario_description.lower()

    unemployment = MACRO_BASELINE_UNEMPLOYMENT
    interest = MACRO_BASELINE_INTEREST
    hpi = MACRO_BASELINE_HPI

    if any(term in text for term in ("unemployment", "jobless", "layoff")):
        unemployment = 10.0 if any(term in text for term in ("surge", "spike", "crisis", "high")) else 7.0
    if any(term in text for term in ("rate hike", "interest", "fed", "inflation")):
        interest = 8.0 if any(term in text for term in ("hike", "high", "surge", "spike")) else 6.0
    if any(term in text for term in ("housing", "hpi", "home price", "real estate")):
        hpi = 80.0 if any(term in text for term in ("crash", "decline", "fall", "drop")) else 90.0

    return clip_macro_inputs(unemployment, interest, hpi)


def _mock_enabled(mock: bool | None) -> bool:
    if mock is not None:
        return mock
    return os.getenv("LLM_MOCK", "false").lower() in {"1", "true", "yes"}


def translate_scenario(
    scenario_description: str,
    *,
    mock: bool | None = None,
    client: OllamaClient | None = None,
) -> tuple[float, float, float]:
    """
    Convert a crisis scenario description into macro coordinates.

    Uses Ollama locally by default (free). Set mock=True or LLM_MOCK=1 to skip LLM calls.
    """
    if _mock_enabled(mock):
        return mock_translate_scenario(scenario_description)

    ollama = client or OllamaClient()
    try:
        payload = ollama.chat_json(
            TRANSLATION_SYSTEM_PROMPT,
            TRANSLATION_USER_PROMPT.format(scenario=scenario_description.strip()),
        )
    except LLMClientError:
        raise

    try:
        unemployment = float(payload["unemployment_rate"])
        interest_rate = float(payload["interest_rate"])
        hpi = float(payload["housing_price_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMClientError(
            "Translation response missing required macro fields."
        ) from exc

    return clip_macro_inputs(unemployment, interest_rate, hpi)
