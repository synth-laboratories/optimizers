"""Chat completions for the ReAct policy. OpenRouter or OpenAI. No stub."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx


class Llm:
    def __init__(self) -> None:
        openrouter = os.environ.get("OPENROUTER_API_KEY") or ""
        openai = os.environ.get("OPENAI_API_KEY") or ""
        if openrouter:
            self.provider = "openrouter"
            self.api_key = openrouter
            self.base_url = os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
            self.model = os.environ.get("GAMEBENCH_LLM_MODEL") or "openai/gpt-4o-mini"
        elif openai:
            self.provider = "openai"
            self.api_key = openai
            self.base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            self.model = os.environ.get("GAMEBENCH_LLM_MODEL") or "gpt-4.1-nano"
        else:
            raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is required for the SpeedRunner harness")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], _attempts: int = 3) -> str:
        """Retry transient transport failures.

        A dropped connection mid-episode otherwise lands as reward 0.0 for the
        candidate, which is noise in the baseline every proposal is compared against.
        """
        last: Exception | None = None
        for attempt in range(_attempts):
            try:
                return self._once(messages)
            except (httpx.HTTPError, httpx.StreamError) as exc:
                last = exc
                if attempt + 1 < _attempts:
                    time.sleep(0.5 * (2**attempt))
        raise last  # type: ignore[misc]

    def _once(self, messages: list[dict[str, Any]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/synth-laboratories/optimizers"
            headers["X-Title"] = "gamebench-levers-speedrunner"
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 128,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        text = str(response.json()["choices"][0]["message"]["content"] or "")
        self.calls.append({"provider": self.provider, "model": self.model, "messages": messages, "text": text})
        return text
