"""Tests for Ollama-based scenario translation."""
import sys
from pathlib import Path

import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.agentic_translator import mock_translate_scenario, translate_scenario
from ai_surrogate.llm_client import LLMClientError
from config import MACRO_BOUNDS


def test_mock_translate_scenario_returns_clipped_values():
    unemployment, interest, hpi = mock_translate_scenario(
        "A crisis with high unemployment, rate hikes, and a housing crash."
    )
    u_min, u_max = MACRO_BOUNDS["unemployment_rate"]
    i_min, i_max = MACRO_BOUNDS["interest_rate"]
    h_min, h_max = MACRO_BOUNDS["housing_price_index"]
    assert u_min <= unemployment <= u_max
    assert i_min <= interest <= i_max
    assert h_min <= hpi <= h_max


def test_translate_scenario_mock_mode():
    unemployment, interest, hpi = translate_scenario(
        "Unemployment surge and falling home prices.",
        mock=True,
    )
    assert unemployment == 10.0
    assert hpi == 80.0


def test_translate_scenario_requires_macro_fields_from_llm():
    class FakeClient:
        def chat_json(self, system_prompt, user_prompt):
            return {"unemployment_rate": 5.0}

    with pytest.raises(LLMClientError, match="required macro fields"):
        translate_scenario("Some scenario", mock=False, client=FakeClient())
