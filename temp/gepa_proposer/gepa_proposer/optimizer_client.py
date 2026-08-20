from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from .episode import TERMINAL_STATUSES


def _is_transient_poll_error(exc: RuntimeError) -> bool:
    """Return whether a run-status read can safely be retried.

    The GEPA service and its worker briefly contend on sqlite while a lane is
    folding results.  A status read can therefore see SQLITE_BUSY even though
    the run itself is healthy.  Keep this deliberately narrow: mutation
    requests and unrelated 5xx responses must still fail immediately.
    """

    message = str(exc).lower()
    return "get /runs/" in message and "sqlite error: database is locked" in message


class OptimizerClient:
    """Talks to the GEPA service: create/fork, poll, pause, pin, export."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.environ.get("GEPA_SERVICE_URL") or "").rstrip("/")
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.base_url)

    def _request(self, method: str, path: str, json: Any | None = None) -> Any:
        if not self.live:
            raise RuntimeError("GEPA_SERVICE_URL is not set")
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method, f"{self.base_url}{path}", json=json)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"{method} {path} -> {response.status_code}: {response.text}"
                )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            return response.json()

    async def _arequest(self, method: str, path: str, json: Any | None = None) -> Any:
        if not self.live:
            raise RuntimeError("GEPA_SERVICE_URL is not set")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(method, f"{self.base_url}{path}", json=json)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"{method} {path} -> {response.status_code}: {response.text}"
                )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            return response.json()

    def create_run(self, body: dict[str, Any]) -> Any:
        return self._request("POST", "/runs", json=body)

    async def acreate_run(self, body: dict[str, Any]) -> Any:
        return await self._arequest("POST", "/runs", json=body)

    def get_run(self, run_id: str) -> Any:
        return self._request("GET", f"/runs/{run_id}")

    async def aget_run(self, run_id: str) -> Any:
        return await self._arequest("GET", f"/runs/{run_id}")

    def get_state(self, run_id: str) -> Any:
        return self._request("GET", f"/runs/{run_id}/state")

    async def aget_state(self, run_id: str) -> Any:
        return await self._arequest("GET", f"/runs/{run_id}/state")

    def wait_until_terminal(self, run_id: str, *, timeout_seconds: float, poll_seconds: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                latest = self.get_run(run_id) or {}
            except RuntimeError as exc:
                if not _is_transient_poll_error(exc):
                    raise
                time.sleep(poll_seconds)
                continue
            status = str(latest.get("status") or "")
            if status in TERMINAL_STATUSES:
                return latest
            time.sleep(poll_seconds)
        raise TimeoutError(f"GEPA run {run_id} did not finish within {timeout_seconds:.0f}s")

    async def await_until_terminal(
        self, run_id: str, *, timeout_seconds: float, poll_seconds: float = 2.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                latest = await self.aget_run(run_id) or {}
            except RuntimeError as exc:
                if not _is_transient_poll_error(exc):
                    raise
                await asyncio.sleep(poll_seconds)
                continue
            status = str(latest.get("status") or "")
            if status in TERMINAL_STATUSES:
                return latest
            await asyncio.sleep(poll_seconds)
        raise TimeoutError(f"GEPA run {run_id} did not finish within {timeout_seconds:.0f}s")

    def pause(self, run_id: str, timeout_seconds: int = 1800) -> Any:
        return self._request(
            "POST", f"/runs/{run_id}/pause", json={"timeout_seconds": timeout_seconds}
        )

    def resume(self, run_id: str) -> Any:
        return self._request("POST", f"/runs/{run_id}/resume")

    def pin(self, run_id: str, checkpoint_id: str | None = None) -> Any:
        body = {} if checkpoint_id is None else {"checkpoint_id": checkpoint_id}
        return self._request("POST", f"/runs/{run_id}/checkpoints/pin", json=body)

    def export(self, run_id: str, checkpoint_id: str) -> Any:
        return self._request("GET", f"/runs/{run_id}/checkpoints/{checkpoint_id}/export")
