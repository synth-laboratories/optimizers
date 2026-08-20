"""HTTP env session handed to a harness script as `env`."""

from __future__ import annotations

from typing import Any

import httpx


class EnvClient:
    def __init__(self, client: httpx.Client, env_url: str, *, split: str = "train") -> None:
        self._client = client
        self._url = env_url.rstrip("/")
        self._split = split
        self.session_id: str | None = None
        self.last_info: dict[str, Any] = {}
        self.last_step: dict[str, Any] = {}
        self.steps = 0

    def reset(self, seed: int = 0, max_steps: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"seed": int(seed), "split": self._split}
        if max_steps is not None:
            body["max_steps"] = int(max_steps)
        payload = self._client.post(f"{self._url}/reset", json=body).json()
        self.session_id = str(payload["session_id"])
        self.last_info = payload.get("info") or {}
        self.steps = 0
        return payload["obs"]

    def step(self, action: Any) -> dict[str, Any]:
        payload = self._client.post(
            f"{self._url}/step",
            json={"session_id": self.session_id, "action": action},
        ).json()
        self.last_step = payload
        self.last_info = payload.get("info") or {}
        self.steps += 1
        return payload

    def spec(self) -> dict[str, Any]:
        return self._client.get(f"{self._url}/spec").json()
