from __future__ import annotations

import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin

from .tunnels import (
    SynthTunnelLease,
    TunnelLease,
    TunnelProvider,
    attach_synth_tunnel_lease,
    create_tunnel_lease,
    tunnel_provider_value,
)
from .observability import (
    OptimizerEvent,
    OptimizerStateSlice,
    OptimizerStateSliceKind,
    state_slice_value,
)


class HostedOptimizerError(RuntimeError):
    pass


_PRIVATE_ERROR_KEYS = frozenset(
    {
        "access_token",
        "auth_json_b64",
        "authorization",
        "access_client_secret",
        "codex_auth_material",
        "cloudflare_token",
        "credential_ref",
        "credential_refs",
        "id_token",
        "ngrok_authtoken",
        "refresh_token",
        "service_token",
        "tunnel_token",
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
_PACKAGE_NAME = "synth-optimizers"
_USAGE_REGISTRATION_TIMEOUT_SECONDS = 2.0


class OptimizerAlgorithmSlug(StrEnum):
    GEPA = "gepa"
    GELO = "go-ex"
    SFT = "sft"
    MAPO = "mapo"
    OHCO = "ohco"
    ONLINE_REFLEXION = "online-reflexion"


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
    online_reflexion_release_evidence: Mapping[str, Any]

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
class HostedOptimizerClient:
    backend_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300.0
    register_usage: bool | None = None
    usage_registration_surface: str = "sdk"
    require_api_key: bool = True

    def __post_init__(self) -> None:
        self.backend_url = (
            self.backend_url or os.environ.get("SYNTH_BACKEND_URL") or "https://api.usesynth.ai"
        ).rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("SYNTH_API_KEY")
        if not self.api_key and self.require_api_key:
            raise HostedOptimizerError("SYNTH_API_KEY is required for hosted optimizer requests")
        if self.register_usage is None:
            self.register_usage = _usage_registration_default_enabled()

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
            online_reflexion_release_evidence=(
                _mapping_or_none(payload.get("online_reflexion_release_evidence")) or {}
            ),
        )

    def submit_gepa(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        return self._submit(OptimizerAlgorithmSlug.GEPA, config, **kwargs)

    def submit(
        self,
        config: Mapping[str, Any] | Any,
        *,
        algorithm: OptimizerAlgorithmSlug | str | None = None,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        resolved_algorithm = _resolve_submit_algorithm(config, algorithm)
        return self._submit(resolved_algorithm, config, **kwargs)

    def submit_gepa_toml(
        self,
        config_toml: str,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: TunnelLease | None = None,
    ) -> OptimizerRunSubmitResponse:
        config_toml = _required_text(
            config_toml,
            field="config_toml",
            context="optimizer GEPA submit",
        )
        if container_pool is not None and container_tunnel is not None:
            raise HostedOptimizerError("container_pool and container_tunnel are mutually exclusive")
        usage_registration_enabled = _usage_registration_enabled_from_toml(config_toml)
        usage_registration_models: list[dict[str, str]] = []
        try:
            usage_registration_models = _usage_registration_models_from_config(
                tomllib.loads(config_toml)
            )
        except tomllib.TOMLDecodeError:
            usage_registration_models = []
        if container_tunnel is not None:
            try:
                config_json = tomllib.loads(config_toml)
            except tomllib.TOMLDecodeError as exc:
                raise HostedOptimizerError(f"invalid GEPA config_toml: {exc}") from exc
            usage_registration_enabled = _usage_registration_enabled_from_config(config_json)
            usage_registration_models = _usage_registration_models_from_config(config_json)
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
        submit_response = OptimizerRunSubmitResponse.from_payload(response)
        if usage_registration_enabled:
            self.register_usage_submit(
                algorithm=OptimizerAlgorithmSlug.GEPA,
                models=usage_registration_models,
            )
        return submit_response

    def submit_gepa_tunnel_toml(
        self,
        config_toml: str,
        *,
        container_tunnel: TunnelLease,
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

    def submit_sft(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        """Submit a hosted SFT run executed by the Optimizers-beta backend."""
        return self._submit(OptimizerAlgorithmSlug.SFT, config, **kwargs)

    def submit_mapo(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        return self._submit(OptimizerAlgorithmSlug.MAPO, config, **kwargs)

    def submit_online_reflexion(
        self,
        config: Mapping[str, Any] | Any,
        **kwargs: Any,
    ) -> OptimizerRunSubmitResponse:
        return self._submit(OptimizerAlgorithmSlug.ONLINE_REFLEXION, config, **kwargs)

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
                if self.register_usage and record.algorithm is not None:
                    self.register_usage_complete(
                        algorithm=record.algorithm,
                        status=record.status.value,
                        uplift=_extract_uplift(record.result),
                    )
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

    def stream_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
        typed: bool = False,
    ) -> Iterator[Mapping[str, Any] | OptimizerEvent]:
        query = _event_query(after_seq=after_seq, limit=limit)
        for payload in self._sse_events(f"/api/v1/optimizers/runs/{run_id}/events?{query}"):
            yield OptimizerEvent.from_payload(payload) if typed else payload

    def event_backfill(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
        typed: bool = False,
    ) -> Iterator[Mapping[str, Any] | OptimizerEvent]:
        query = _event_query(after_seq=after_seq, limit=limit)
        query = f"{query}&stream=0"
        for payload in self._ndjson_events(
            f"/api/v1/optimizers/runs/{run_id}/events?{query}",
            context="lifecycle event backfill",
        ):
            yield OptimizerEvent.from_payload(payload) if typed else payload

    def get_state(self, run_id: str) -> Mapping[str, Any]:
        return self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}/state")

    def get_state_slice(
        self,
        run_id: str,
        slice_name: OptimizerStateSliceKind | str,
        *,
        typed: bool = False,
    ) -> Mapping[str, Any] | OptimizerStateSlice:
        slice_path = quote(state_slice_value(slice_name), safe="")
        payload = self._json_request("GET", f"/api/v1/optimizers/runs/{run_id}/state/{slice_path}")
        return OptimizerStateSlice.from_payload(payload) if typed else payload

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

    def algorithm_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
        typed: bool = False,
    ) -> Iterator[Mapping[str, Any] | OptimizerEvent]:
        query = _event_query(after_seq=after_seq, limit=limit)
        for payload in self._ndjson_events(
            f"/api/v1/optimizers/runs/{run_id}/algorithm-events?{query}",
            context="optimizer algorithm event backfill",
        ):
            yield OptimizerEvent.from_payload(payload) if typed else payload

    def goex_event_stream(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
    ) -> Iterator[Mapping[str, Any]]:
        query = _event_query(after_seq=after_seq, limit=limit)
        yield from self._sse_events(f"/api/v1/optimizers/runs/{run_id}/goex-events/stream?{query}")

    def algorithm_event_stream(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 500,
        typed: bool = False,
    ) -> Iterator[Mapping[str, Any] | OptimizerEvent]:
        query = _event_query(after_seq=after_seq, limit=limit)
        for payload in self._sse_events(
            f"/api/v1/optimizers/runs/{run_id}/algorithm-events/stream?{query}"
        ):
            yield OptimizerEvent.from_payload(payload) if typed else payload

    def online_reflexion_receipt(
        self,
        run_id: str,
        *,
        exposure_limit: int = 500,
        outcome_limit: int = 500,
    ) -> Mapping[str, Any]:
        query = urlencode(
            {
                "exposure_limit": max(1, min(5000, int(exposure_limit))),
                "outcome_limit": max(1, min(5000, int(outcome_limit))),
            }
        )
        return self._json_request(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/online-reflexion/receipt?{query}",
            context="online Reflexion receipt bundle",
        )

    def online_reflexion_receipt_audit(
        self,
        run_id: str,
        *,
        strict: bool = False,
    ) -> Mapping[str, Any]:
        query = urlencode({"strict": "true" if strict else "false"})
        return self._json_request(
            "GET",
            f"/api/v1/optimizers/runs/{quote(run_id, safe='')}/online-reflexion/receipt-audit?{query}",
            context="online Reflexion receipt audit",
        )

    def online_reflexion_receipt_audits(
        self,
        *,
        run_ids: Sequence[str] | None = None,
        layer_id: str | None = None,
        project_id: str | None = None,
        strict: bool = False,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        params: dict[str, str] = {
            "strict": "true" if strict else "false",
            "limit": str(max(1, min(100, int(limit)))),
        }
        clean_run_ids = [run_id.strip() for run_id in (run_ids or ()) if run_id.strip()]
        if clean_run_ids:
            params["run_ids"] = ",".join(clean_run_ids)
        if layer_id:
            params["layer_id"] = layer_id
        if project_id:
            params["project_id"] = project_id
        return self._json_request(
            "GET",
            f"/api/v1/optimizers/online-reflexion/receipt-audits?{urlencode(params)}",
            context="online Reflexion aggregate receipt audit",
        )

    def online_reflexion_evidence_packet(
        self,
        *,
        run_ids: Sequence[str] | None = None,
        layer_id: str | None = None,
        project_id: str | None = None,
        evidence_notes: Mapping[str, Any] | None = None,
        blog_decision_owner: str = "Josh",
        blog_approved_by_owner: bool = False,
        include_receipt_summaries: bool = True,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        audit = self.online_reflexion_receipt_audits(
            run_ids=run_ids,
            layer_id=layer_id,
            project_id=project_id,
            strict=False,
            limit=limit,
        )
        notes_review = validate_online_reflexion_evidence_notes(evidence_notes or {})
        audit_status = str(audit.get("status") or "").strip().lower()
        ready = (
            notes_review["status"] == "pass"
            and audit_status in {"pass", "passed", "ready", "complete", "completed"}
            and blog_approved_by_owner
        )
        return {
            "schema_version": "online_reflexion_evidence_packet.v1",
            "status": "ready" if ready else "attention_required",
            "audit": dict(audit),
            "evidence_notes_review": notes_review,
            "blog_decision_owner": blog_decision_owner,
            "blog_approved_by_owner": blog_approved_by_owner,
            "receipt_summaries_included": include_receipt_summaries,
        }

    def open_synth_tunnel(
        self,
        local_base_url: str,
        *,
        requested_ttl_seconds: int = 3600,
        metadata: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> SynthTunnelLease:
        lease = self.open_tunnel(
            local_base_url,
            provider=TunnelProvider.SYNTH_TUNNEL,
            requested_ttl_seconds=requested_ttl_seconds,
            metadata=metadata,
            capabilities=capabilities,
        )
        if not isinstance(lease, SynthTunnelLease):
            raise HostedOptimizerError("open_synth_tunnel did not return a SynthTunnel lease")
        return lease

    def attach_synth_tunnel(
        self,
        lease_id: str,
        local_base_url: str,
        *,
        heartbeat_extend_ttl_seconds: int = 3600,
        wait_ready: bool = True,
    ) -> SynthTunnelLease:
        """Reattach a local target to an existing SynthTunnel lease and URL."""

        try:
            return attach_synth_tunnel_lease(
                self,
                lease_id,
                local_base_url,
                heartbeat_extend_ttl_seconds=heartbeat_extend_ttl_seconds,
                wait_ready=wait_ready,
            )
        except Exception as exc:
            raise HostedOptimizerError(f"failed to attach SynthTunnel lease: {exc}") from exc

    def open_tunnel(
        self,
        local_base_url: str,
        *,
        provider: TunnelProvider | str = TunnelProvider.AUTO,
        requested_ttl_seconds: int = 3600,
        metadata: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        wait_ready: bool = True,
    ) -> TunnelLease:
        try:
            return create_tunnel_lease(
                self,
                local_base_url,
                provider=provider,
                requested_ttl_seconds=requested_ttl_seconds,
                metadata=metadata,
                capabilities=capabilities,
                wait_ready=wait_ready,
            )
        except RuntimeError as exc:
            raise HostedOptimizerError(_public_error_text(str(exc))) from exc

    def _submit(
        self,
        algorithm: OptimizerAlgorithmSlug,
        config: Mapping[str, Any] | Any,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: TunnelLease | None = None,
        billing_mode: str | None = None,
    ) -> OptimizerRunSubmitResponse:
        if container_pool is not None and container_tunnel is not None:
            raise HostedOptimizerError("container_pool and container_tunnel are mutually exclusive")
        config_json = _config_to_json(config)
        usage_registration_enabled = _usage_registration_enabled_from_config(config_json)
        usage_registration_models = _usage_registration_models_from_config(config_json)
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
            billing_mode=billing_mode,
        )
        response = self._json_request("POST", "/api/v1/optimizers/runs", payload)
        submit_response = OptimizerRunSubmitResponse.from_payload(response)
        if usage_registration_enabled:
            self.register_usage_submit(algorithm=algorithm, models=usage_registration_models)
        return submit_response

    def register_usage_submit(
        self,
        *,
        algorithm: OptimizerAlgorithmSlug,
        models: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        if not self.register_usage:
            return
        payload: dict[str, Any] = {
            "algorithm": algorithm.value,
            "event_name": "run_submit",
            "client_surface": _usage_registration_surface(self.usage_registration_surface),
            "package_name": _PACKAGE_NAME,
            "package_version": _package_version(),
            "internal": _usage_registration_internal(),
            "install_id": _usage_install_id(),
        }
        usage_models = _sanitize_usage_registration_models(models)
        if usage_models:
            payload["models"] = usage_models
        try:
            self._json_request(
                "POST",
                "/api/v1/optimizers/usage/registrations",
                payload,
                context="optimizer usage registration",
                include_auth=False,
                timeout_seconds=_USAGE_REGISTRATION_TIMEOUT_SECONDS,
            )
        except HostedOptimizerError:
            return

    def register_usage_complete(
        self,
        *,
        algorithm: OptimizerAlgorithmSlug,
        status: str,
        uplift: float | None = None,
        models: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        """Fire a run-completion ping: terminal status + uplift number (if present).
        Anonymous and best-effort, like the submit ping; never raises."""
        if not self.register_usage:
            return
        payload: dict[str, Any] = {
            "algorithm": algorithm.value,
            "event_name": "run_complete",
            "client_surface": _usage_registration_surface(self.usage_registration_surface),
            "package_name": _PACKAGE_NAME,
            "package_version": _package_version(),
            "status": status,
            "internal": _usage_registration_internal(),
            "install_id": _usage_install_id(),
        }
        if uplift is not None:
            payload["uplift"] = uplift
        usage_models = _sanitize_usage_registration_models(models)
        if usage_models:
            payload["models"] = usage_models
        try:
            self._json_request(
                "POST",
                "/api/v1/optimizers/usage/registrations",
                payload,
                context="optimizer usage registration",
                include_auth=False,
                timeout_seconds=_USAGE_REGISTRATION_TIMEOUT_SECONDS,
            )
        except HostedOptimizerError:
            return

    def _add_submit_metadata(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None,
        idempotency_key: str | None,
        project_id: str | None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None,
        billing_mode: str | None = None,
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
        if billing_mode is not None:
            payload["billing_mode"] = billing_mode

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        context: str = "hosted optimizer JSON response",
        allow_empty: bool = False,
        include_auth: bool = True,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if include_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            _api_endpoint(self.backend_url or "", path),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
            with urllib.request.urlopen(request, timeout=timeout) as response:
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
            headers=({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
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


def submit_mapo(
    config: Mapping[str, Any] | Any,
    *,
    client: HostedOptimizerClient | None = None,
    **kwargs: Any,
) -> OptimizerRunSubmitResponse:
    active_client = client or HostedOptimizerClient()
    return active_client.submit_mapo(config, **kwargs)


def submit_sft(
    config: Mapping[str, Any] | Any,
    *,
    client: HostedOptimizerClient | None = None,
    **kwargs: Any,
) -> OptimizerRunSubmitResponse:
    """Submit an SFT run through the public hosted Optimizers API."""
    active_client = client or HostedOptimizerClient()
    return active_client.submit_sft(config, **kwargs)


def submit_online_reflexion(
    config: Mapping[str, Any] | Any,
    *,
    client: HostedOptimizerClient | None = None,
    **kwargs: Any,
) -> OptimizerRunSubmitResponse:
    active_client = client or HostedOptimizerClient()
    return active_client.submit_online_reflexion(config, **kwargs)


def validate_online_reflexion_evidence_notes(
    evidence_notes: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the public release lanes without contacting the hosted service."""

    lanes = (
        "craftax_rotated_121_125",
        "alfworld_6x6_x3",
        "ebr_first_scale_compare",
        "harvey_lab_pilot",
        "hosted_staging_smoke",
    )
    complete_statuses = {"pass", "passed", "ready", "complete", "completed", "succeeded"}
    required_evidence: list[dict[str, Any]] = []
    remaining: list[str] = []
    for lane in lanes:
        raw = evidence_notes.get(lane)
        status = str(raw.get("status") or "").strip().lower() if isinstance(raw, Mapping) else ""
        complete = status in complete_statuses
        required_evidence.append(
            {
                "key": lane,
                "state": "complete" if complete else "attention_required",
                "evidence": dict(raw) if isinstance(raw, Mapping) else raw,
            }
        )
        if not complete:
            remaining.append(f"attach complete evidence for {lane}")

    release_raw = evidence_notes.get("release_blog_growth")
    release_status = (
        str(release_raw.get("status") or "").strip().lower()
        if isinstance(release_raw, Mapping)
        else ""
    )
    release_complete = release_status in complete_statuses
    if not release_complete:
        remaining.append("complete release/blog/growth readiness evidence")
    return {
        "schema_version": "online_reflexion_evidence_notes_review.v1",
        "status": "pass" if not remaining else "attention_required",
        "evidence_lanes_complete": all(item["state"] == "complete" for item in required_evidence),
        "release_gate_complete": release_complete,
        "release_gate": {
            "state": "complete" if release_complete else "attention_required",
            "evidence": dict(release_raw) if isinstance(release_raw, Mapping) else release_raw,
        },
        "required_evidence": required_evidence,
        "remaining": remaining,
    }


def _api_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _usage_registration_default_enabled() -> bool:
    disabled = os.environ.get("SYNTH_OPTIMIZERS_DISABLE_USAGE_REGISTRATION")
    if disabled is not None and _env_flag(disabled):
        return False
    enabled = os.environ.get("SYNTH_OPTIMIZERS_REGISTER_USAGE")
    if enabled is None:
        return True
    return _env_flag(enabled)


def _env_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _usage_registration_internal() -> bool:
    """True when this is our own testing (set SYNTH_OPTIMIZERS_INTERNAL=1 in our
    dev/CI envs) so the dashboard can separate internal runs from real OSS usage."""
    raw = os.environ.get("SYNTH_OPTIMIZERS_INTERNAL")
    return raw is not None and _env_flag(raw)


@lru_cache(maxsize=1)
def _usage_install_id() -> str:
    """Stable anonymous install id (a random UUID persisted under the state dir),
    so usage can be de-duplicated into DAU/WAU. It is NOT derived from hardware —
    just a random id — and is only ever created when usage reporting is enabled.
    Falls back to an ephemeral per-process id if the file can't be persisted."""
    raw_state_dir = os.environ.get("SYNTH_OPTIMIZERS_STATE_DIR")
    if raw_state_dir and raw_state_dir.strip():
        state_root = Path(raw_state_dir).expanduser()
    else:
        xdg_state = os.environ.get("XDG_STATE_HOME")
        state_root = (
            Path(xdg_state).expanduser()
            if xdg_state and xdg_state.strip()
            else Path.home() / ".local" / "state"
        )
    path = state_root / "synth-optimizers" / "install-id"
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    new_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id, encoding="utf-8")
    except OSError:
        pass  # read-only / sandboxed home: ephemeral anonymous id for this process
    return new_id


def _extract_uplift(result: Mapping[str, Any] | None) -> float | None:
    """Best-effort single uplift number from a run result (no other detail sent)."""
    if not isinstance(result, Mapping):
        return None
    for key in ("uplift", "improvement", "delta", "score_uplift"):
        v = result.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    best = result.get("best_score")
    base = result.get("baseline_score", result.get("baseline"))
    if isinstance(best, (int, float)) and isinstance(base, (int, float)):
        return float(best) - float(base)
    return None


def _usage_registration_surface(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized in {"sdk", "cli"}:
        return normalized
    return "unknown"


def _usage_registration_enabled_from_toml(config_toml: str) -> bool:
    try:
        config_json = tomllib.loads(config_toml)
    except tomllib.TOMLDecodeError:
        return True
    return _usage_registration_enabled_from_config(config_json)


def _usage_registration_enabled_from_config(config_json: Mapping[str, Any]) -> bool:
    raw_section = config_json.get("usage_registration")
    if not isinstance(raw_section, Mapping):
        return True
    raw_enabled = raw_section.get("enabled")
    if isinstance(raw_enabled, bool):
        return raw_enabled
    if isinstance(raw_enabled, str):
        return _env_flag(raw_enabled)
    return True


def _usage_registration_models_from_config(config_json: Mapping[str, Any]) -> list[dict[str, str]]:
    models: list[Mapping[str, str]] = []
    policy = config_json.get("policy")
    if isinstance(policy, Mapping):
        models.append(
            {
                "role": "policy",
                "provider": str(policy.get("provider") or ""),
                "model": str(policy.get("model") or ""),
            }
        )
    proposer = config_json.get("proposer")
    if isinstance(proposer, Mapping):
        models.append(
            {
                "role": "proposer",
                "provider": str(proposer.get("provider") or ""),
                "model": str(proposer.get("model") or ""),
            }
        )
    proposers = config_json.get("proposers")
    if isinstance(proposers, Mapping):
        defaults = proposers.get("defaults")
        default_provider = (
            str(defaults.get("provider") or "") if isinstance(defaults, Mapping) else ""
        )
        default_model = str(defaults.get("model") or "") if isinstance(defaults, Mapping) else ""
        for role, raw in proposers.items():
            if role == "defaults" or not isinstance(raw, Mapping):
                continue
            models.append(
                {
                    "role": str(raw.get("role") or role),
                    "provider": str(raw.get("provider") or default_provider),
                    "model": str(raw.get("model") or default_model),
                }
            )
    return _sanitize_usage_registration_models(models)


def _sanitize_usage_registration_models(
    models: Sequence[Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in list(models or [])[:16]:
        model = str(raw.get("model") or "").strip()[:160]
        if not model:
            continue
        provider = str(raw.get("provider") or "").strip()[:80]
        role = str(raw.get("role") or "").strip()[:64]
        key = (role, provider, model)
        if key in seen:
            continue
        seen.add(key)
        entry = {"model": model}
        if provider:
            entry["provider"] = provider
        if role:
            entry["role"] = role
        sanitized.append(entry)
    return sanitized


def _package_version() -> str | None:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def _public_error_detail(body: bytes) -> str:
    return _public_error_text(body.decode("utf-8", errors="replace"))


def _public_error_text(text: str) -> str:
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


def _resolve_submit_algorithm(
    config: Mapping[str, Any] | Any,
    algorithm: OptimizerAlgorithmSlug | str | None,
) -> OptimizerAlgorithmSlug:
    if algorithm is not None:
        return _algorithm_slug(algorithm, context="hosted optimizer submit")
    if isinstance(config, Mapping):
        raise HostedOptimizerError(
            "hosted optimizer submit requires algorithm=... for raw mapping configs"
        )
    return _algorithm_slug(
        getattr(config, "algorithm", None),
        context=f"{type(config).__name__} hosted optimizer config",
    )


def _with_tunnel_container(config_json: dict[str, Any], lease: TunnelLease) -> dict[str, Any]:
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
    target = _container_target_payload(lease.container_config())
    target_headers = target.get("headers") or {}
    if not isinstance(target_headers, Mapping):
        raise HostedOptimizerError("container_tunnel.container_config().headers must be an object")
    headers = {
        str(name): str(value)
        for name, value in {**dict(raw_headers), **dict(target_headers)}.items()
        if str(name).strip().lower() not in {"authorization", "x-api-key"}
    }
    container["url"] = _required_text(
        target.get("url"),
        field="public_url",
        context="tunnel lease",
    )
    if headers:
        container["headers"] = headers
    else:
        container.pop("headers", None)
    auth_bearer_env = str(target.get("auth_bearer_env") or "").strip()
    if auth_bearer_env:
        container["auth_bearer_env"] = auth_bearer_env
    if tunnel_provider_value(getattr(lease, "provider", None)) == TunnelProvider.SYNTH_TUNNEL.value:
        container["auth_refresh"] = {
            "provider": "synth_tunnel",
            "lease_id": _required_text(
                lease.lease_id,
                field="lease_id",
                context="Synth tunnel lease",
            ),
            "refresh_interval_seconds": 900,
        }
    else:
        container.pop("auth_refresh", None)
    config_json["container"] = container
    return config_json


def _container_target_payload(target: Any) -> Mapping[str, Any]:
    if hasattr(target, "to_config_json"):
        target = target.to_config_json()
    if not isinstance(target, Mapping):
        raise HostedOptimizerError("container_tunnel.container_config() must return an object")
    return dict(target)


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
