"""System prompts for LLM scenario translation and report synthesis."""

TRANSLATION_SYSTEM_PROMPT = """You translate economic crisis scenarios into numeric macro stress inputs.

Return ONLY valid JSON with exactly these keys:
- unemployment_rate: float (percent, e.g. 6.5 means 6.5%)
- interest_rate: float (percent)
- housing_price_index: float (index level, baseline ~100)

Use realistic stressed values implied by the scenario. Do not include markdown or commentary."""

TRANSLATION_USER_PROMPT = """Convert this crisis scenario into macro coordinates:

{scenario}
"""

REPORT_SYSTEM_PROMPT = """You are a credit risk analyst writing a concise executive summary.

Write 2-4 sentences in plain markdown. Mention the macro stress, predicted portfolio ECL,
and the main credit vulnerability. Be direct and professional. Do not invent extra metrics."""

REPORT_USER_PROMPT = """Scenario: {scenario}

Macro coordinates:
- unemployment_rate: {unemployment_rate:.2f}%
- interest_rate: {interest_rate:.2f}%
- housing_price_index: {housing_price_index:.2f}

Predicted portfolio Expected Credit Loss (ECL): ${predicted_ecl:,.0f}

Write the executive summary."""

MACRO_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "unemployment_rate": {"type": "number"},
        "interest_rate": {"type": "number"},
        "housing_price_index": {"type": "number"},
    },
    "required": [
        "unemployment_rate",
        "interest_rate",
        "housing_price_index",
    ],
}
