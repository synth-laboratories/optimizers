"""HTTP env session injected into a ReAct/harness script as `env`."""

from __future__ import annotations

from typing import Any

import httpx


class EnvClient:
    def __init__(self, client: httpx.Client, env_url: str) -> None:
        self._client = client
        self._url = env_url.rstrip("/")
        self.session_id: str | None = None
        self.last_info: dict[str, Any] = {}
        self.last_step: dict[str, Any] = {}

    def reset(self, seed: int, max_steps: int = 16) -> dict[str, Any]:
        payload = self._client.post(
            f"{self._url}/reset",
            json={"seed": seed, "max_steps": max_steps},
        ).json()
        self.session_id = str(payload["session_id"])
        self.last_info = payload.get("info") or {}
        return payload["obs"]

    def step(self, action: str) -> dict[str, Any]:
        payload = self._client.post(
            f"{self._url}/step",
            json={"session_id": self.session_id, "action": action},
        ).json()
        self.last_step = payload
        self.last_info = payload.get("info") or {}
        return payload
