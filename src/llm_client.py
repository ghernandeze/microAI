from __future__ import annotations

import json
from urllib import error, request

from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_enabled(self) -> bool:
        return self.settings.model_enabled and bool(self.settings.model_api_url)

    def call(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """
        Make an LLM API call.
        Returns a dict with optional keys:
          - 'text': str — the assistant's text response
          - 'tool_calls': list[dict] — parsed tool calls (name, arguments, id)
          - 'raw_assistant_message': dict — full message object (needed for multi-turn tool loops)
        """
        if not self.is_enabled():
            raise RuntimeError(
                "El cliente LLM no está habilitado. Define MODEL_API_URL para activarlo."
            )

        payload: dict = {
            "model": self.settings.model_name,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.settings.model_api_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.model_api_key}",
                "User-Agent": "Mozilla/5.0",
                "HTTP-Referer": "http://localhost",
                "X-Title": "microcredito-agente",
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

        return self._parse_response(body)

    def generate(self, messages: list[dict]) -> str:
        """Convenience wrapper — returns plain text (no tool calling)."""
        return self.call(messages).get("text", "")

    @staticmethod
    def _parse_response(raw_body: str) -> dict:
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return {"text": raw_body.strip()}

        if not isinstance(parsed, dict) or not parsed.get("choices"):
            return {"text": raw_body.strip()}

        message = parsed["choices"][0].get("message", {})
        result: dict = {}

        if message.get("content"):
            result["text"] = str(message["content"]).strip()

        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            result["raw_assistant_message"] = message
            result["tool_calls"] = []
            for tc in raw_tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                result["tool_calls"].append({
                    "id": tc.get("id", ""),
                    "name": tc["function"]["name"],
                    "arguments": args,
                })

        return result
