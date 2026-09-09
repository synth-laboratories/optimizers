"""Hosted training contracts and local rollout-plane preflight."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin, urlparse


TRAINING_EVENT_SCHEMA_VERSION = "training.event.v1"
ROLLOUT_CAPABILITIES_SCHEMA_VERSION = "training.rollout.capabilities.v1"
EXECUTOR_CONFIG_SCHEMA_VERSION = "training.executor.v1"


@dataclass(frozen=True, slots=True)
class BoundedTrainingCaps:
    steps: int
    wall_clock_seconds: int
    cost_usd: float

    def to_config_json(self) -> dict[str, int | float]:
        if (
            self.steps < 1
            or self.wall_clock_seconds < 1
            or not math.isfinite(self.cost_usd)
            or self.cost_usd <= 0
        ):
            raise TrainingClientError("training_bounded_caps_invalid")
        return {
            "steps": self.steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class RequestedTraining:
    sequence_cap: int
    max_sample_tokens: int
    batch_size: int
    checkpoint_every_steps: int
    total_training_steps: int
    evaluation_every_steps: int = 1

    def to_config_json(self) -> dict[str, int]:
        values = {
            "sequence_cap": self.sequence_cap,
            "max_sample_tokens": self.max_sample_tokens,
            "batch_size": self.batch_size,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "total_training_steps": self.total_training_steps,
            "evaluation_every_steps": self.evaluation_every_steps,
        }
        if any(isinstance(value, bool) or value < 1 for value in values.values()):
            raise TrainingClientError("requested_training_invalid")
        if self.max_sample_tokens > self.sequence_cap:
            raise TrainingClientError("requested_training_sample_exceeds_sequence_cap")
        return values


@dataclass(frozen=True, slots=True)
class HostedTrainingSpec:
    algorithm: str
    model_id: str
    model_revision: str
    rank: int
    requested_training: RequestedTraining
    bounded_run_caps: BoundedTrainingCaps
    algorithm_config: Mapping[str, Any] = field(default_factory=dict)
    repository_commits: Mapping[str, str] = field(default_factory=dict)
    maximum_policy_lag: int = 1
    environment_hold_seconds: int = 300


def build_hosted_training_config(
    spec: HostedTrainingSpec,
    *,
    provider: "ProviderTrainingPreflight",
    rollout: "ContainerTrainingPreflight",
) -> dict[str, Any]:
    implementation = {
        "cispo": ("slime-reference", "cispo.slime.v1"),
        "ppo": ("glm-5.3", "ppo.glm-5.3.v1"),
    }.get(spec.algorithm)
    if implementation is None:
        raise TrainingClientError("hosted_on_policy_algorithm_invalid")
    if provider.algorithm != spec.algorithm or provider.model_id != spec.model_id:
        raise TrainingClientError("training_provider_preflight_identity_mismatch")
    if (
        not spec.model_id.strip()
        or not spec.model_revision.strip()
        or spec.rank < 1
        or spec.maximum_policy_lag < 0
        or spec.environment_hold_seconds < 1
    ):
        raise TrainingClientError("hosted_training_spec_invalid")
    requested = spec.requested_training.to_config_json()
    rollout_config = rollout.rollout_config()
    rollout_config.update(
        {
            "connection_mode": "close",
            "maximum_policy_lag": spec.maximum_policy_lag,
            "environment_hold_seconds": spec.environment_hold_seconds,
        }
    )
    algorithm_config = dict(spec.algorithm_config)
    if spec.repository_commits:
        if "repository_commits" in algorithm_config:
            raise TrainingClientError("hosted_training_repository_commits_ambiguous")
        commits = {
            str(name).strip(): str(commit).strip()
            for name, commit in spec.repository_commits.items()
            if str(name).strip() and str(commit).strip()
        }
        if commits != dict(spec.repository_commits):
            raise TrainingClientError("hosted_training_repository_commits_invalid")
        algorithm_config["repository_commits"] = commits
    return {
        "schema_version": EXECUTOR_CONFIG_SCHEMA_VERSION,
        "run_id": "pending_backend_assignment",
        "attempt_id": "pending_backend_assignment",
        "algorithm": {
            "algorithm": spec.algorithm,
            "implementation": implementation[0],
            "implementation_version": implementation[1],
        },
        "model": {
            "id": spec.model_id,
            "revision": spec.model_revision,
            "rank": spec.rank,
        },
        "tinker": provider.provider_config(),
        "rollout": rollout_config,
        "container": {},
        "bounded_run_caps": spec.bounded_run_caps.to_config_json(),
        "requested_training": requested,
        "effective_training": dict(requested),
        "algorithm_config": algorithm_config,
    }


class TrainingClientError(RuntimeError):
    pass


class TrainingLifecycle(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    ENV_UNREACHABLE = "env_unreachable"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED_EVIDENCE = "failed_evidence"
    FAILED = "failed"
    INFRASTRUCTURE_LOST = "infrastructure_lost"
    CAP_REACHED = "cap_reached"

    @property
    def terminal(self) -> bool:
        return self in {
            self.CANCELLED,
            self.COMPLETED,
            self.DEGRADED,
            self.FAILED_EVIDENCE,
            self.FAILED,
            self.INFRASTRUCTURE_LOST,
            self.CAP_REACHED,
        }


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    event_id: str
    job_id: str
    attempt_id: str
    sequence: int
    occurred_at: str
    kind: str
    phase: str
    payload: Mapping[str, Any]
    producer: Mapping[str, str]
    schema_version: str = TRAINING_EVENT_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingEvent":
        event = cls(
            schema_version=_required_text(payload, "schema_version"),
            event_id=_required_text(payload, "event_id"),
            job_id=_required_text(payload, "job_id"),
            attempt_id=_required_text(payload, "attempt_id"),
            sequence=_positive_int(payload.get("sequence"), "sequence"),
            occurred_at=_required_text(payload, "occurred_at"),
            kind=_required_text(payload, "kind"),
            phase=_required_text(payload, "phase"),
            payload=_required_mapping(payload.get("payload"), "payload"),
            producer={
                key: _required_text(_required_mapping(payload.get("producer"), "producer"), key)
                for key in ("service", "version", "commit")
            },
        )
        if event.schema_version != TRAINING_EVENT_SCHEMA_VERSION:
            raise TrainingClientError("unsupported_training_event_schema")
        if not (event.occurred_at.endswith("Z") or event.occurred_at.endswith("+00:00")):
            raise TrainingClientError("training_event_timestamp_not_utc")
        return event


def reduce_training_lifecycle(
    current: TrainingLifecycle,
    event: TrainingEvent,
) -> TrainingLifecycle:
    if current.terminal:
        return current
    target = {
        "job.accepted": TrainingLifecycle.QUEUED,
        "validation.started": TrainingLifecycle.VALIDATING,
        "validation.succeeded": TrainingLifecycle.QUEUED,
        "provisioning.started": TrainingLifecycle.PROVISIONING,
        "training.started": TrainingLifecycle.RUNNING,
        "rollout.env_unreachable": TrainingLifecycle.ENV_UNREACHABLE,
        "rollout.env_reconnected": TrainingLifecycle.RUNNING,
        "checkpoint.writing": TrainingLifecycle.CHECKPOINTING,
        "evaluation.started": TrainingLifecycle.EVALUATING,
        "cancellation.requested": TrainingLifecycle.CANCELLING,
        "job.paused": TrainingLifecycle.PAUSED,
        "job.resumed": TrainingLifecycle.RUNNING,
        "job.completed": TrainingLifecycle.COMPLETED,
        "job.degraded": TrainingLifecycle.DEGRADED,
        "job.failed_evidence": TrainingLifecycle.FAILED_EVIDENCE,
        "job.failed": TrainingLifecycle.FAILED,
        "job.cancelled": TrainingLifecycle.CANCELLED,
        "job.infrastructure_lost": TrainingLifecycle.INFRASTRUCTURE_LOST,
        "job.cap_reached": TrainingLifecycle.CAP_REACHED,
    }.get(event.kind)
    if target is None:
        return current
    if target.terminal or target in {
        TrainingLifecycle.RUNNING,
        TrainingLifecycle.ENV_UNREACHABLE,
    }:
        return target
    order = list(TrainingLifecycle)
    return current if order.index(target) < order.index(current) else target


@dataclass(frozen=True, slots=True)
class TrainingRolloutRequirement:
    task_id: str
    min_concurrency: int = 1
    connection_mode: str = "close"
    required_operations: frozenset[str] = field(
        default_factory=lambda: frozenset({"rollout", "reward", "heartbeat"})
    )


@dataclass(frozen=True, slots=True)
class ContainerTrainingPreflight:
    local_base_url: str
    target_id: str
    task_id: str
    container_digest: str
    capability_hash: str
    capabilities: Mapping[str, Any]

    def rollout_config(self) -> dict[str, Any]:
        return {
            "protocol": "training.rollout.request.v1",
            "task_id": self.task_id,
            "container_id": self.target_id,
            "container_digest": self.container_digest,
            "container_capability_hash": self.capability_hash,
            "sampler_transport": "direct_https",
        }


@dataclass(frozen=True, slots=True)
class ProviderTrainingPreflight:
    provider: str
    model_id: str
    algorithm: str
    capability_hash: str
    capabilities: Mapping[str, Any]

    def provider_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "capability_hash": self.capability_hash,
            "capability_response": dict(self.capabilities),
            "spend_free_preflight": True,
        }


def validate_provider_training_capabilities(
    capabilities: Any,
    *,
    model_id: str,
    algorithm: str,
) -> ProviderTrainingPreflight:
    if not isinstance(capabilities, dict):
        raise TrainingClientError("training_provider_capabilities_invalid")
    if capabilities.get("schema_version") != "training.provider_capabilities.v1":
        raise TrainingClientError("training_provider_capability_schema_unsupported")
    offered_hash = _required_text(capabilities, "capability_hash")
    unhashed = {key: value for key, value in capabilities.items() if key != "capability_hash"}
    if offered_hash != _canonical_sha256(unhashed):
        raise TrainingClientError("training_provider_capability_hash_mismatch")
    if capabilities.get("supported") is not True:
        raise TrainingClientError(
            str(capabilities.get("reason") or "training_provider_combination_unsupported")
        )
    model = _required_mapping(capabilities.get("model"), "model")
    if (
        model.get("model_id") != model_id
        or capabilities.get("algorithm") != algorithm
        or capabilities.get("provider") != "tinker"
        or capabilities.get("spend_free") is not True
        or capabilities.get("creates_training_session") is not False
    ):
        raise TrainingClientError("training_provider_capability_requirement_unsatisfied")
    operations = capabilities.get("operations")
    if not isinstance(operations, list) or not {
        "sample",
        "forward",
        "train",
        "checkpoint_save",
        "checkpoint_resume",
        algorithm,
    }.issubset(set(operations)):
        raise TrainingClientError("training_provider_capability_requirement_unsatisfied")
    pricing = capabilities.get("pricing")
    if (
        not isinstance(pricing, Mapping)
        or pricing.get("schema_version") != "training.provider_pricing.v1"
        or pricing.get("currency") != "USD"
        or pricing.get("unit") != "per_million_tokens"
        or any(
            isinstance(pricing.get(name), bool)
            or not isinstance(pricing.get(name), int | float)
            or not math.isfinite(float(pricing[name]))
            or float(pricing[name]) <= 0
            for name in ("prefill", "cached_prefill", "sample", "train")
        )
    ):
        raise TrainingClientError("training_provider_pricing_snapshot_invalid")
    return ProviderTrainingPreflight(
        provider="tinker",
        model_id=model_id,
        algorithm=algorithm,
        capability_hash=offered_hash,
        capabilities=dict(capabilities),
    )


def apply_provider_preflight(
    config: Mapping[str, Any],
    preflight: ProviderTrainingPreflight,
) -> dict[str, Any]:
    updated = dict(config)
    if "tinker" in updated:
        raise TrainingClientError("training_config_tinker_already_set")
    updated["tinker"] = preflight.provider_config()
    return updated


def preflight_training_container(
    local_base_url: str,
    requirement: TrainingRolloutRequirement,
    *,
    timeout_seconds: float = 10.0,
) -> ContainerTrainingPreflight:
    parsed = urlparse(local_base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise TrainingClientError("training_container_must_be_local")
    endpoint = urljoin(local_base_url.rstrip("/") + "/", "training/capabilities")
    try:
        request = urllib.request.Request(
            endpoint, method="GET", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            raw = response.read(262_145)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TrainingClientError("training_container_preflight_unreachable") from exc
    if len(raw) > 262_144:
        raise TrainingClientError("training_container_capabilities_too_large")
    try:
        capabilities = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingClientError("training_container_capabilities_invalid") from exc
    return validate_training_capabilities(local_base_url, capabilities, requirement)


def validate_training_capabilities(
    local_base_url: str,
    capabilities: Any,
    requirement: TrainingRolloutRequirement,
) -> ContainerTrainingPreflight:
    if not isinstance(capabilities, dict):
        raise TrainingClientError("training_container_capabilities_invalid")
    if capabilities.get("schema_version") != ROLLOUT_CAPABILITIES_SCHEMA_VERSION:
        raise TrainingClientError("training_container_capability_schema_unsupported")
    offered_hash = _required_text(capabilities, "capability_hash")
    unhashed = {key: value for key, value in capabilities.items() if key != "capability_hash"}
    computed_hash = _canonical_sha256(unhashed)
    if offered_hash != computed_hash:
        raise TrainingClientError("training_container_capability_hash_mismatch")
    operations = capabilities.get("operations")
    protocols = capabilities.get("protocol_versions")
    modes = capabilities.get("connection_modes")
    if (
        capabilities.get("task_id") != requirement.task_id
        or not isinstance(operations, list)
        or not requirement.required_operations.issubset(set(operations))
        or not isinstance(protocols, list)
        or "training.rollout.request.v1" not in protocols
        or not isinstance(modes, list)
        or requirement.connection_mode not in modes
        or capabilities.get("supports_idempotency") is not True
        or capabilities.get("supports_sampler_https") is not True
        or _positive_int(capabilities.get("max_concurrency"), "max_concurrency")
        < requirement.min_concurrency
    ):
        raise TrainingClientError("training_container_capability_requirement_unsatisfied")
    return ContainerTrainingPreflight(
        local_base_url=local_base_url,
        target_id=_required_text(capabilities, "container_id"),
        task_id=_required_text(capabilities, "task_id"),
        container_digest=_required_text(capabilities, "container_digest"),
        capability_hash=offered_hash,
        capabilities=dict(capabilities),
    )


def apply_rollout_preflight(
    config: Mapping[str, Any],
    preflight: ContainerTrainingPreflight,
) -> dict[str, Any]:
    updated = dict(config)
    if "rollout" in updated:
        raise TrainingClientError("training_config_rollout_already_set")
    updated["rollout"] = preflight.rollout_config()
    return updated


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TrainingClientError(f"{field_name}_required")
    return value.strip()


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingClientError(f"{field_name}_object_required")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainingClientError(f"{field_name}_positive_integer_required")
    return value
