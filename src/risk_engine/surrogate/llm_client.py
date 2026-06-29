"""Free local LLM client using Ollama."""
import json

import requests

from risk_engine.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

class LLMClientError(Exception):
    """Raised when the LLM client cannot complete a request."""

class OllamaClient:
    """Thin wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.timeout_seconds = (
            OLLAMA_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self.base_url}/api/chat"
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(
                "Could not reach Ollama. Install Ollama, run `ollama serve`, and pull a model "
                f"such as `ollama pull {self.model}`."
            ) from exc

        try:
            data = response.json()
            return str(data["message"]["content"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMClientError("Unexpected response format from Ollama.") from exc

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        content = self.chat(system_prompt, user_prompt, json_mode=True)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMClientError("Ollama returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError("Ollama JSON response must be an object.")
        return parsed
