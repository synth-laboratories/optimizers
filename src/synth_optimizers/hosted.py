from __future__ import annotations

import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlparse


class HostedOptimizerError(RuntimeError):
    pass


_PRIVATE_ERROR_KEYS = frozenset(
    {
        "access_token",
        "auth_json_b64",
        "authorization",
        "codex_auth_material",
        "credential_ref",
        "credential_refs",
        "id_token",
        "refresh_token",
        "service_token",
        "token_bundle",
        "worker_token",
        "x-api-key",
    }
)
_PRIVATE_PATH_MARKERS = (
    "/tmp/optimizers-beta",
    "/var/tmp/optimizers-beta",
    "/workspace/optimizers-beta",
    "/Users/",
    "/home/",
    "/root/",
    ".out/",
)
_PRIVATE_ERROR_KEY_RE = re.compile(
    r"(?i)(\b(?:"
    + "|".join(re.escape(key) for key in sorted(_PRIVATE_ERROR_KEYS))
    + r")\b\s*[:=]\s*)([\"'])(.*?)(\2)"
)
_PRIVATE_ERROR_UNQUOTED_KEY_RE = re.compile(
    r"(?i)(\b(?:"
    + "|".join(re.escape(key) for key in sorted(_PRIVATE_ERROR_KEYS))
    + r")\b\s*[:=]\s*)([^\s,}]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRETISH_RE = re.compile(r"(?i)\b(?:sk|sess|eyJ)[A-Za-z0-9._~+/=-]{16,}")


class OptimizerAlgorithmSlug(StrEnum):
    GEPA = "gepa"
    GELO = "go-ex"
    OHCO = "ohco"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def terminal(cls) -> frozenset[str]:
        return frozenset({cls.SUCCEEDED.value, cls.FAILED.value, cls.CANCELLED.value})


class ContainerTargetKind(StrEnum):
    DIRECT_URL = "direct_url"
    SYNTH_TUNNEL = "synth_tunnel"
    RHODES_POOL = "rhodes_pool"
    HOSTED_POOL_ENV = "hosted_pool_env"


class AlgorithmCatalogStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED_PRIVATE = "planned_private"


@dataclass(frozen=True, slots=True)
class AlgorithmCatalogEntry:
    algorithm: OptimizerAlgorithmSlug
    candidate_kinds: tuple[str, ...]
    status: AlgorithmCatalogStatus
    submit_supported: bool


@dataclass(frozen=True, slots=True)
class OptimizerBillingFeatureConfig:
    feature_id: str
    env_override: bool


@dataclass(frozen=True, slots=True)
class OptimizerStartupCatalog:
    available_algorithms: tuple[AlgorithmCatalogEntry, ...]
    org_id: str | None
    optimizers_beta_configured: bool
    billing_feature_ids: Mapping[str, OptimizerBillingFeatureConfig]
    billing_feature_ids_configured: Mapping[str, bool]

    @property
    def submit_supported(self) -> tuple[OptimizerAlgorithmSlug, ...]:
        return tuple(
            entry.algorithm for entry in self.available_algorithms if entry.submit_supported
        )


@dataclass(frozen=True, slots=True)
class ContainerPoolTarget:
    pool_id: str
    task_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"pool_id": self.pool_id}
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        return payload


@dataclass(frozen=True, slots=True)
class ContainerDirectTarget:
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    auth_bearer_env: str | None = None

    def to_config_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.url}
        if self.headers:
            payload["headers"] = dict(self.headers)
        if self.auth_bearer_env:
            payload["auth_bearer_env"] = self.auth_bearer_env
        return payload


@dataclass(frozen=True, slots=True)
class OptimizerRunSubmitResponse:
    run_id: str
    status: RunStatus
    events_url: str
    status_url: str
    artifact_base_url: str
    algorithm: OptimizerAlgorithmSlug | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerRunSubmitResponse":
        context = "optimizer submit response"
        algorithm = _algorithm_or_none(payload.get("algorithm"), context=context)
        return cls(
            run_id=_required_text(payload.get("run_id"), field="run_id", context=context),
            status=_run_status(payload.get("status"), context=context),
            events_url=_required_text(
                payload.get("events_url"),
                field="events_url",
                context=context,
            ),
            status_url=_required_text(
                payload.get("status_url"),
                field="status_url",
                context=context,
            ),
            artifact_base_url=_required_text(
                payload.get("artifact_base_url"),
                field="artifact_base_url",
                context=context,
            ),
            algorithm=algorithm,
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class OptimizerRunRecord:
    run_id: str
    status: RunStatus
    algorithm: OptimizerAlgorithmSlug | None
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None
    result: Mapping[str, Any] | None = None
    events_url: str | None = None
    status_url: str | None = None
    artifact_base_url: str | None = None
    finalize_state: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerRunRecord":
        context = "optimizer run response"
        return cls(
            run_id=_required_text(payload.get("run_id"), field="run_id", context=context),
            status=_run_status(payload.get("status"), context=context),
            algorithm=_algorithm_or_none(payload.get("algorithm"), context=context),
            created_at=_str_or_none(payload.get("created_at")),
            updated_at=_str_or_none(payload.get("updated_at")),
            error=_str_or_none(payload.get("error")),
            result=_mapping_or_none(payload.get("result")),
            events_url=_str_or_none(payload.get("events_url")),
            status_url=_str_or_none(payload.get("status_url")),
            artifact_base_url=_str_or_none(payload.get("artifact_base_url")),
            finalize_state=_str_or_none(payload.get("finalize_state")),
            raw=dict(payload),
        )


@dataclass(slots=True)
class SynthTunnelLease:
    lease_id: str
    public_url: str
    worker_token: str
    client: "HostedOptimizerClient | None" = field(default=None, repr=False)
    route_token: str | None = None
    agent_connect: Mapping[str, Any] | None = None
    expires_at: str | None = None

    def container_config(self) -> ContainerDirectTarget:
        return ContainerDirectTarget(
            url=self.public_url,
            headers={"authorization": f"Bearer {self.worker_token}"},
        )

    def refresh_worker_token(self) -> str:
        if self.client is None:
            raise HostedOptimizerError("SynthTunnelLease has no client for token refresh")
        payload = self.client._json_request(
            "POST", f"/api/v1/synthtunnel/leases/{self.lease_id}/token:refresh"
        )
        self.worker_token = str(payload.get("worker_token") or "")
        if not self.worker_token:
            raise HostedOptimizerError("SynthTunnel token refresh returned no worker_token")
        return self.worker_token

    def close(self) -> None:
        if self.client is not None:
            self.client._json_request(
                "DELETE",
                f"/api/v1/synthtunnel/leases/{self.lease_id}",
                context="Synth tunnel lease close response",
                allow_empty=True,
            )

    def __enter__(self) -> "SynthTunnelLease":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(slots=True)
class HostedOptimizerClient:
    backend_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.backend_url = (
            self.backend_url or os.environ.get("SYNTH_BACKEND_URL") or "https://api.usesynth.ai"
        ).rstrip("/")
        self.api_key = self.api_key or os.environ.get("SYNTH_API_KEY")
        if not self.api_key:
            raise HostedOptimizerError("SYNTH_API_KEY is required for hosted optimizer requests")

    def startup(self) -> OptimizerStartupCatalog:
        payload = self._json_request("GET", "/api/v1/optimizers/startup")
        startup_context = "optimizer startup catalog"
        raw_algorithms = payload.get("available_algorithms")
        if not isinstance(raw_algorithms, Sequence) or isinstance(raw_algorithms, str | bytes):
            raise HostedOptimizerError(
                "optimizer startup catalog missing available_algorithms list"
            )
        entries = []
        for index, raw in enumerate(raw_algorithms):
            context = f"optimizer startup available_algorithms[{index}]"
            if not isinstance(raw, Mapping):
                raise HostedOptimizerError(f"{context} is not an object")
            algorithm = _algorithm_slug(raw.get("algorithm"), context=context)
            status = _catalog_status(raw.get("status"), context=context)
            raw_candidate_kinds = raw.get("candidate_kinds")
            if not isinstance(raw_candidate_kinds, Sequence) or isinstance(
                raw_candidate_kinds, str | bytes
            ):
                raise HostedOptimizerError(f"{context} missing candidate_kinds list")
            entries.append(
                AlgorithmCatalogEntry(
                    algorithm=algorithm,
                    candidate_kinds=tuple(str(item) for item in raw_candidate_kinds),
                    status=status,
                    submit_supported=status == AlgorithmCatalogStatus.AVAILABLE
                    and algorithm != OptimizerAlgorithmSlug.OHCO,
                )
            )
        return OptimizerStartupCatalog(
            available_algorithms=tuple(entries),
            org_id=_str_or_none(payload.get("org_id")),
            optimizers_beta_configured=_required_bool(
                payload.get("optimizers_beta_configured"),
                field="optimizers_beta_configured",
                context=startup_context,
            ),
            billing_feature_ids=_billing_feature_ids(payload.get("billing_feature_ids")),
            billing_feature_ids_configured=_billing_feature_ids_configured(
                payload.get("billing_feature_ids_configured")
            ),
        )

    def submit_gepa(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        return self._submit(OptimizerAlgorithmSlug.GEPA, config, **kwargs)

    def submit_gepa_toml(
        self,
        config_toml: str,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: SynthTunnelLease | None = None,
    ) -> OptimizerRunSubmitResponse:
        config_toml = _required_text(
            config_toml,
            field="config_toml",
            context="optimizer GEPA submit",
        )
        if container_pool is not None and container_tunnel is not None:
            raise HostedOptimizerError(
                "container_pool and container_tunnel are mutually exclusive"
            )
        if container_tunnel is not None:
            try:
                config_json = tomllib.loads(config_toml)
            except tomllib.TOMLDecodeError as exc:
                raise HostedOptimizerError(f"invalid GEPA config_toml: {exc}") from exc
            config_json = _with_tunnel_container(config_json, container_tunnel)
            payload: dict[str, Any] = {
                "algorithm": OptimizerAlgorithmSlug.GEPA.value,
                "config_json": config_json,
            }
        else:
            payload = {
                "algorithm": OptimizerAlgorithmSlug.GEPA.value,
                "config_toml": config_toml,
            }
        self._add_submit_metadata(
            payload,
            run_id=run_id,
            idempotency_key=idempotency_key,
            project_id=project_id,
            container_pool=container_pool,
        )
        response = self._json_request("POST", "/api/v1/optimizers/runs", payload)
        return OptimizerRunSubmitResponse.from_payload(response)

    def submit_gepa_tunnel_toml(
        self,
        config_toml: str,
        *,
        container_tunnel: SynthTunnelLease,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
    ) -> OptimizerRunSubmitResponse:
        return self.submit_gepa_toml(
            config_toml,
            run_id=run_id,
            idempotency_key=idempotency_key,
            project_id=project_id,
            container_tunnel=container_tunnel,
        )

    def submit_gelo(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        return self._submit(OptimizerAlgorithmSlug.GELO, config, **kwargs)

    def get_run(self, run_id: str) -> OptimizerRunRecord:
        payload = self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}")
        return OptimizerRunRecord.from_payload(payload)

    def cancel_run(self, run_id: str) -> OptimizerRunRecord:
        payload = self._json_request("POST", f"/api/v1/optimizers/runs/{run_id}/cancel")
        return OptimizerRunRecord.from_payload(payload)

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_seconds: float = 2.0,
        timeout_seconds: float | None = None,
    ) -> OptimizerRunRecord:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            record = self.get_run(run_id)
            if record.status.value in RunStatus.terminal():
                return record
            if deadline is not None and time.monotonic() >= deadline:
                raise HostedOptimizerError(f"timed out waiting for optimizer run {run_id}")
            time.sleep(max(0.1, poll_seconds))

    def get_artifact(self, run_id: str, name: str) -> bytes:
        return self._bytes_request(
            "GET",
            f"/api/v1/optimizers/runs/{run_id}/artifacts/{name}",
            context="artifact",
        )

    def events(self, run_id: str) -> Iterator[Mapping[str, Any]]:
        for payload in self._sse_events(f"/api/v1/optimizers/runs/{run_id}/events"):
            yield payload

    def event_backfill(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
    ) -> Iterator[Mapping[str, Any]]:
        query = _event_query(after_seq=after_seq, limit=limit)
        query = f"{query}&stream=0"
        yield from self._ndjson_events(
            f"/api/v1/optimizers/runs/{run_id}/events?{query}",
            context="lifecycle event backfill",
        )

    def get_state(self, run_id: str) -> Mapping[str, Any]:
        return self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}/state")

    def get_state_slice(self, run_id: str, slice_name: str) -> Mapping[str, Any]:
        slice_path = quote(str(slice_name), safe="")
        return self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}/state/{slice_path}")

    def get_state_batch(
        self,
        run_id: str,
        slices: Sequence[str] | str,
    ) -> Mapping[str, Any]:
        slice_values = [slices] if isinstance(slices, str) else list(slices)
        slice_query = ",".join(str(item) for item in slice_values if str(item))
        if not slice_query:
            raise HostedOptimizerError("at least one state slice is required")
        query = urlencode({"slices": slice_query})
        return self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}/state/batch?{query}")

    def goex_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
    ) -> Iterator[Mapping[str, Any]]:
        query = _event_query(after_seq=after_seq, limit=limit)
        yield from self._ndjson_events(
            f"/api/v1/optimizers/runs/{run_id}/goex-events?{query}",
            context="Go-Ex event backfill",
        )

    def goex_event_stream(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
    ) -> Iterator[Mapping[str, Any]]:
        query = _event_query(after_seq=after_seq, limit=limit)
        yield from self._sse_events(f"/api/v1/optimizers/runs/{run_id}/goex-events/stream?{query}")

    def open_synth_tunnel(
        self,
        local_base_url: str,
        *,
        requested_ttl_seconds: int = 3600,
        metadata: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> SynthTunnelLease:
        parsed = urlparse(local_base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        payload = {
            "client_instance_id": f"synth-optimizers-{uuid.uuid4().hex[:16]}",
            "local_target": {"host": host, "port": port},
            "requested_ttl_seconds": requested_ttl_seconds,
            "metadata": dict(metadata or {}),
            "capabilities": dict(capabilities or {}),
        }
        response = self._json_request("POST", "/api/v1/synthtunnel/leases", payload)
        context = "Synth tunnel lease response"
        return SynthTunnelLease(
            lease_id=_required_text(response.get("lease_id"), field="lease_id", context=context),
            public_url=_required_text(
                response.get("public_url"),
                field="public_url",
                context=context,
            ),
            worker_token=_required_text(
                response.get("worker_token"),
                field="worker_token",
                context=context,
            ),
            client=self,
            route_token=_str_or_none(response.get("route_token")),
            agent_connect=_mapping_or_none(response.get("agent_connect")),
            expires_at=_str_or_none(response.get("expires_at")),
        )

    def _submit(
        self,
        algorithm: OptimizerAlgorithmSlug,
        config: Mapping[str, Any] | Any,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: SynthTunnelLease | None = None,
    ) -> OptimizerRunSubmitResponse:
        if container_pool is not None and container_tunnel is not None:
            raise HostedOptimizerError(
                "container_pool and container_tunnel are mutually exclusive"
            )
        config_json = _config_to_json(config)
        if container_tunnel is not None:
            config_json = _with_tunnel_container(config_json, container_tunnel)
        payload: dict[str, Any] = {
            "algorithm": algorithm.value,
            "config_json": config_json,
        }
        self._add_submit_metadata(
            payload,
            run_id=run_id,
            idempotency_key=idempotency_key,
            project_id=project_id,
            container_pool=container_pool,
        )
        response = self._json_request("POST", "/api/v1/optimizers/runs", payload)
        return OptimizerRunSubmitResponse.from_payload(response)

    def _add_submit_metadata(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None,
        idempotency_key: str | None,
        project_id: str | None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None,
    ) -> None:
        if run_id is not None:
            payload["run_id"] = run_id
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        if project_id is not None:
            payload["project_id"] = project_id
        if container_pool is not None:
            payload["container_pool"] = (
                container_pool.to_payload()
                if isinstance(container_pool, ContainerPoolTarget)
                else dict(container_pool)
            )

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        context: str = "hosted optimizer JSON response",
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            _api_endpoint(self.backend_url or "", path),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _public_error_detail(exc.read())
            raise HostedOptimizerError(
                f"hosted optimizer request failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HostedOptimizerError(f"hosted optimizer request failed: {exc}") from exc
        if not text:
            if not allow_empty:
                raise HostedOptimizerError(f"{context} was empty")
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HostedOptimizerError(f"{context} parse failed: {exc}") from exc
        if not isinstance(data, dict):
            raise HostedOptimizerError(f"{context} was not an object")
        return data

    def _bytes_request(self, method: str, path: str, *, context: str) -> bytes:
        request = urllib.request.Request(
            _api_endpoint(self.backend_url or "", path),
            headers={"Authorization": f"Bearer {self.api_key}"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = _public_error_detail(exc.read())
            raise HostedOptimizerError(
                f"hosted optimizer {context} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HostedOptimizerError(f"hosted optimizer {context} failed: {exc}") from exc

    def _ndjson_events(self, path: str, *, context: str) -> Iterator[Mapping[str, Any]]:
        body = self._bytes_request("GET", path, context=context).decode(
            "utf-8",
            errors="replace",
        )
        for line_number, line in enumerate(body.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HostedOptimizerError(
                    f"hosted optimizer {context} parse failed at line {line_number}: {exc}"
                ) from exc
            if not isinstance(data, Mapping):
                raise HostedOptimizerError(
                    f"hosted optimizer {context} line {line_number} is not an object"
                )
            yield dict(data)

    def _sse_events(self, path: str) -> Iterator[Mapping[str, Any]]:
        request = urllib.request.Request(
            _api_endpoint(self.backend_url or "", path),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="GET",
        )
        data_lines: list[str] = []
        event_index = 0
        try:
            with urllib.request.urlopen(
                request, timeout=max(self.timeout_seconds, 1200.0)
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").strip())
                        continue
                    if line or not data_lines:
                        continue
                    payload_text = "\n".join(data_lines)
                    data_lines.clear()
                    event_index += 1
                    yield _event_object_from_json(
                        payload_text,
                        context=f"hosted optimizer SSE event {event_index}",
                    )
                if data_lines:
                    event_index += 1
                    yield _event_object_from_json(
                        "\n".join(data_lines),
                        context=f"hosted optimizer SSE event {event_index}",
                    )
        except urllib.error.HTTPError as exc:
            detail = _public_error_detail(exc.read())
            raise HostedOptimizerError(
                f"hosted optimizer events failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HostedOptimizerError(f"hosted optimizer events failed: {exc}") from exc


def _api_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _public_error_detail(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    for marker in _PRIVATE_PATH_MARKERS:
        if marker in text:
            text = " ".join(
                "<path>" if any(marker in token for marker in _PRIVATE_PATH_MARKERS) else token
                for token in text.split()
            )
            break
    text = _PRIVATE_ERROR_KEY_RE.sub(r"\1\2<redacted>\4", text)
    text = _PRIVATE_ERROR_UNQUOTED_KEY_RE.sub(r"\1<redacted>", text)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    return _SECRETISH_RE.sub("<redacted>", text)


def _event_query(*, after_seq: int, limit: int) -> str:
    return urlencode(
        {
            "after_seq": max(0, int(after_seq)),
            "limit": max(1, min(5000, int(limit))),
        }
    )


def _config_to_json(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(config, "to_config_json"):
        config = config.to_config_json()
    if not isinstance(config, Mapping):
        raise TypeError("hosted optimizer config must be a mapping or expose to_config_json()")
    data = json.loads(json.dumps(config, default=_json_default))
    if not isinstance(data, dict):
        raise TypeError("hosted optimizer config must serialize to a JSON object")
    return data


def _with_tunnel_container(config_json: dict[str, Any], lease: SynthTunnelLease) -> dict[str, Any]:
    config_json = dict(config_json)
    raw_container = config_json.get("container")
    if raw_container is None:
        container: dict[str, Any] = {}
    elif isinstance(raw_container, Mapping):
        container = dict(raw_container)
    else:
        raise HostedOptimizerError("hosted optimizer config container must be an object")
    if container.get("pool") is not None:
        raise HostedOptimizerError("container_tunnel cannot be combined with config container.pool")
    if str(container.get("auth_bearer_env") or "").strip():
        raise HostedOptimizerError(
            "container_tunnel cannot be combined with config container.auth_bearer_env"
        )
    if container.get("auth_refresh") not in (None, {}):
        raise HostedOptimizerError(
            "container_tunnel cannot be combined with config container.auth_refresh"
        )
    raw_headers = container.get("headers") or {}
    if not isinstance(raw_headers, Mapping):
        raise HostedOptimizerError("hosted optimizer config container.headers must be an object")
    headers = {
        str(name): str(value)
        for name, value in raw_headers.items()
        if str(name).strip().lower() not in {"authorization", "x-api-key"}
    }
    container["url"] = _required_text(
        lease.public_url,
        field="public_url",
        context="Synth tunnel lease",
    )
    container["headers"] = headers
    container["auth_refresh"] = {
        "provider": "synth_tunnel",
        "lease_id": _required_text(
            lease.lease_id,
            field="lease_id",
            context="Synth tunnel lease",
        ),
        "refresh_interval_seconds": 900,
    }
    config_json["container"] = container
    return config_json


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_config_json"):
        return value.to_config_json()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _required_text(value: Any, *, field: str, context: str) -> str:
    if value is None:
        raise HostedOptimizerError(f"{context} missing {field}")
    text = str(value).strip()
    if not text:
        raise HostedOptimizerError(f"{context} has empty {field}")
    return text


def _required_bool(value: Any, *, field: str, context: str) -> bool:
    if not isinstance(value, bool):
        raise HostedOptimizerError(f"{context} field {field} must be a boolean")
    return value


def _run_status(value: Any, *, context: str) -> RunStatus:
    raw = _required_text(value, field="status", context=context)
    try:
        return RunStatus(raw)
    except ValueError as exc:
        raise HostedOptimizerError(f"{context} has unknown status {raw!r}") from exc


def _algorithm_slug(value: Any, *, context: str) -> OptimizerAlgorithmSlug:
    raw = _required_text(value, field="algorithm", context=context)
    try:
        return OptimizerAlgorithmSlug(raw)
    except ValueError as exc:
        raise HostedOptimizerError(f"{context} has unknown algorithm {raw!r}") from exc


def _algorithm_or_none(value: Any, *, context: str) -> OptimizerAlgorithmSlug | None:
    if value is None:
        return None
    return _algorithm_slug(value, context=context)


def _catalog_status(value: Any, *, context: str) -> AlgorithmCatalogStatus:
    raw = _required_text(value, field="status", context=context)
    try:
        return AlgorithmCatalogStatus(raw)
    except ValueError as exc:
        raise HostedOptimizerError(f"{context} has unknown catalog status {raw!r}") from exc


def _event_object_from_json(payload_text: str, *, context: str) -> dict[str, Any]:
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise HostedOptimizerError(f"{context} parse failed: {exc}") from exc
    if not isinstance(data, Mapping):
        raise HostedOptimizerError(f"{context} payload is not an object")
    return dict(data)


def _billing_feature_ids(value: Any) -> Mapping[str, OptimizerBillingFeatureConfig]:
    context = "optimizer startup billing_feature_ids"
    if not isinstance(value, Mapping):
        raise HostedOptimizerError(f"{context} must be an object")
    result: dict[str, OptimizerBillingFeatureConfig] = {}
    for algorithm in ("gepa", "go_ex"):
        raw_entry = value.get(algorithm)
        entry_context = f"{context}.{algorithm}"
        if not isinstance(raw_entry, Mapping):
            raise HostedOptimizerError(f"{entry_context} must be an object")
        result[algorithm] = OptimizerBillingFeatureConfig(
            feature_id=_required_text(
                raw_entry.get("feature_id"),
                field="feature_id",
                context=entry_context,
            ),
            env_override=_required_bool(
                raw_entry.get("env_override"),
                field="env_override",
                context=entry_context,
            ),
        )
    return result


def _billing_feature_ids_configured(value: Any) -> Mapping[str, bool]:
    context = "optimizer startup billing_feature_ids_configured"
    if not isinstance(value, Mapping):
        raise HostedOptimizerError(f"{context} must be an object")
    result: dict[str, bool] = {}
    for algorithm in ("gepa", "go_ex"):
        result[algorithm] = _required_bool(
            value.get(algorithm),
            field=algorithm,
            context=context,
        )
    return result


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None
