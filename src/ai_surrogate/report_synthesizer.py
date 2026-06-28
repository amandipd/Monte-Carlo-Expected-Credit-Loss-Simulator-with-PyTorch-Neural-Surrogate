"""Generate executive summaries from scenario, macro inputs, and predicted ECL."""
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.llm_client import LLMClientError, OllamaClient
from ai_surrogate.prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT


def mock_synthesize_report(
    scenario_description: str,
    unemployment_rate: float,
    interest_rate: float,
    housing_price_index: float,
    predicted_ecl: float,
) -> str:
    return (
        f"Under the scenario \"{scenario_description.strip()}\", macro stress sits at "
        f"{unemployment_rate:.1f}% unemployment, {interest_rate:.1f}% interest rates, "
        f"and an HPI of {housing_price_index:.1f}. The surrogate estimates portfolio ECL of "
        f"**${predicted_ecl:,.0f}**, indicating elevated credit loss exposure relative to baseline "
        f"conditions. Management should monitor unemployment and rate-sensitive segments closely."
    )


def _mock_enabled(mock: bool | None) -> bool:
    if mock is not None:
        return mock
    return os.getenv("LLM_MOCK", "false").lower() in {"1", "true", "yes"}


def synthesize_report(
    scenario_description: str,
    unemployment_rate: float,
    interest_rate: float,
    housing_price_index: float,
    predicted_ecl: float,
    *,
    mock: bool | None = None,
    client: OllamaClient | None = None,
) -> str:
    """Generate a short executive credit report markdown string."""
    if _mock_enabled(mock):
        return mock_synthesize_report(
            scenario_description,
            unemployment_rate,
            interest_rate,
            housing_price_index,
            predicted_ecl,
        )

    ollama = client or OllamaClient()
    prompt = REPORT_USER_PROMPT.format(
        scenario=scenario_description.strip(),
        unemployment_rate=unemployment_rate,
        interest_rate=interest_rate,
        housing_price_index=housing_price_index,
        predicted_ecl=predicted_ecl,
    )
    try:
        return ollama.chat(REPORT_SYSTEM_PROMPT, prompt).strip()
    except LLMClientError:
        raise
