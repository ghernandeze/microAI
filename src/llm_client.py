from __future__ import annotations

import json
from urllib import request, error

from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_enabled(self) -> bool:
        return self.settings.model_enabled and bool(self.settings.model_api_url)

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.is_enabled():
            raise RuntimeError(
                "El cliente LLM no está habilitado. Define MODEL_API_URL para activarlo."
            )

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
        }
        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            self.settings.model_api_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.model_api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "microcredito-agente"
            },
        )

        try:
            with request.urlopen(req, timeout=self.settings.model_timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Error HTTP {e.code}: {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"No fue posible conectar con el endpoint: {e}") from e

        return self._extract_text(body)

    @staticmethod
    def _extract_text(raw_body: str) -> str:
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body.strip()

        if isinstance(parsed, dict):
            if "output_text" in parsed and isinstance(parsed["output_text"], str):
                return parsed["output_text"].strip()
            if "content" in parsed and isinstance(parsed["content"], str):
                return parsed["content"].strip()
            if "choices" in parsed and parsed["choices"]:
                choice = parsed["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict) and "content" in message:
                        return str(message["content"]).strip()
            if "response" in parsed and isinstance(parsed["response"], str):
                return parsed["response"].strip()

        return raw_body.strip()
