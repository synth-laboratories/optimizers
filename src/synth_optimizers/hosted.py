from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


class HostedOptimizerError(RuntimeError):
    pass


class HostedOptimizerAuthError(HostedOptimizerError):
    pass


class HostedOptimizerHTTPError(HostedOptimizerError):
    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class HostedGepaConfig(Protocol):
    def to_toml_dict(self) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class HostedOptimizerClient:
    backend_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.backend_url = (
            self.backend_url
            or os.getenv("SYNTH_BACKEND_URL")
            or os.getenv("SYNTH_API_BASE_URL")
            or "https://api.usesynth.ai"
        ).rstrip("/")
        self.api_key = self.api_key if self.api_key is not None else os.getenv("SYNTH_API_KEY")

    def startup(
        self,
        *,
        client: str | None = None,
        version: str | None = None,
        algorithms: list[str] | None = None,
        runtime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        record: bool = False,
    ) -> dict[str, Any]:
        catalog = self._request_json("GET", "/api/v1/optimizers/startup")
        if record:
            self.record_startup(
                client=client,
                version=version,
                algorithms=algorithms,
                runtime=runtime,
                metadata=metadata,
            )
        return catalog

    def record_startup(
        self,
        *,
        client: str | None = None,
        version: str | None = None,
        algorithms: list[str] | None = None,
        runtime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/optimizers/startup",
            {
                "client": client,
                "version": version,
                "algorithms": list(algorithms or []),
                "runtime": runtime,
                "metadata": dict(metadata or {}),
            },
        )

    def submit_gepa(
        self,
        config: HostedGepaConfig | Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/optimizers/runs",
            {
                "algorithm": "gepa",
                "run_id": run_id,
                "idempotency_key": idempotency_key,
                "project_id": project_id,
                "config_json": _gepa_config_json(config),
                "container_pool": dict(container_pool) if container_pool is not None else None,
            },
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/optimizers/runs/{run_id}")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/api/v1/optimizers/runs/{run_id}/cancel")

    def get_artifact(self, run_id: str, artifact_name: str) -> bytes:
        return self._request_bytes(
            "GET",
            f"/api/v1/optimizers/runs/{run_id}/artifacts/{artifact_name}",
        )

    def events(self, run_id: str) -> Iterator[str]:
        request = self._build_request(
            "GET",
            f"/api/v1/optimizers/runs/{run_id}/events",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        yield line
        except HTTPError as exc:
            payload = exc.read()
            raise HostedOptimizerHTTPError(
                exc.code,
                payload.decode("utf-8", errors="replace") or str(exc),
            ) from exc
        except URLError as exc:
            raise HostedOptimizerError(f"GET events for {run_id} failed: {exc}") from exc

    def _request_json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._request_bytes(method, path, body)
        if not payload:
            return {}
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise HostedOptimizerError(f"{method} {path} did not return a JSON object")
        return decoded

    def _request_bytes(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> bytes:
        request = self._build_request(method, path, body)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            payload = exc.read()
            parsed: Any = None
            message = payload.decode("utf-8", errors="replace") or str(exc)
            try:
                parsed = json.loads(payload.decode("utf-8"))
                if isinstance(parsed, dict):
                    message = str(parsed.get("detail") or parsed.get("error") or message)
            except json.JSONDecodeError:
                pass
            raise HostedOptimizerHTTPError(exc.code, message, parsed) from exc
        except URLError as exc:
            raise HostedOptimizerError(f"{method} {path} failed: {exc}") from exc

    def _build_request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Request:
        api_key = self._api_key_for_request()
        data = None
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(_drop_none(body)).encode("utf-8")
        return Request(
            urljoin(f"{self.backend_url}/", path.lstrip("/")),
            data=data,
            headers=headers,
            method=method,
        )

    def _api_key_for_request(self) -> str | None:
        api_key = (self.api_key or "").strip()
        if api_key:
            return api_key
        if not _is_local_backend_url(str(self.backend_url or "")):
            raise HostedOptimizerAuthError(
                "SYNTH_API_KEY is required for hosted optimizer backend calls"
            )
        return None


def _is_local_backend_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _gepa_config_json(config: HostedGepaConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    to_toml_dict = getattr(config, "to_toml_dict", None)
    if not callable(to_toml_dict):
        raise HostedOptimizerError(
            "GEPA hosted config must be a mapping or expose to_toml_dict()"
        )
    payload = to_toml_dict()
    if not isinstance(payload, Mapping):
        raise HostedOptimizerError("GEPA hosted config to_toml_dict() must return a mapping")
    return dict(payload)


__all__ = [
    "HostedGepaConfig",
    "HostedOptimizerAuthError",
    "HostedOptimizerClient",
    "HostedOptimizerError",
    "HostedOptimizerHTTPError",
]
