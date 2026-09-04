"""Append-only durable catalog for materialized policy artifacts.

Every materialized checkpoint is a durable, addressable record here. A
checkpoint that exists only as a path printed in a log does not exist. The
store is append-only: checkpoint records are immutable, and everything that
happens to a checkpoint afterwards -- publication transitions, policy-set
membership, lineage, evaluations -- is a new relation rather than a mutation.
Append-only is enforced by sqlite triggers, not by convention; the only
mutable table is ``aliases``, which exists precisely because human aliases are
declared mutable pointers that must resolve to immutable ids.

The two provider artifact roles are separate *types*, not a role string on one
type: a :class:`SamplerWeightsRef` cannot be stored, returned, or requested
where a :class:`TrainingStateRef` is required, and neither can stand in for the
other. That is the whole point of the split, so it is enforced by construction.

Nothing here names a task, a harness, an environment, or a model provider.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Iterator

from synth_optimizers.contracts.rl_records import (
    ARTIFACT_ROLES,
    RecordError,
    RendererProfile,
    digest,
)

CHECKPOINT_SCHEMA_VERSION = "cispo.checkpoint.v1"
LINEAGE_EDGE_SCHEMA_VERSION = "cispo.checkpoint_lineage_edge.v1"
SAVE_ATTEMPT_SCHEMA_VERSION = "cispo.checkpoint_save_attempt.v1"
EVALUATION_BINDING_SCHEMA_VERSION = "cispo.evaluation_binding.v1"

SAMPLER_ROLE = "sampler_weights"
TRAINING_STATE_ROLE = "training_state"

PUBLICATION_STATUSES = frozenset({"staged", "published", "orphaned", "superseded"})
REGISTRABLE_STATUSES = frozenset({"staged", "published"})
RESOLVABLE_STATUSES = frozenset({"published", "superseded"})
PUBLICATION_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "staged": frozenset({"published", "orphaned"}),
    "published": frozenset({"superseded"}),
    "orphaned": frozenset(),
    "superseded": frozenset(),
}

LINEAGE_RELATIONS = frozenset(
    {"parent", "policy_set_component", "match_set_trainee", "match_set_opponent"}
)
REVISION_KINDS = frozenset({"policy_set", "match_set"})
REVISION_TRANSITIONS = frozenset(
    {
        "created",
        "load",
        "ready",
        "health_check_failed",
        "attempt_open",
        "attempt_close",
        "retire",
        "superseded",
    }
)
EVALUATION_TARGET_KINDS = frozenset({"checkpoint", "policy_set", "match_set"})
SAVE_OUTCOMES = frozenset({"succeeded", "failed"})
ALIAS_TARGET_KINDS = EVALUATION_TARGET_KINDS

# Selectors that can change meaning under a running job. Never resolvable.
MUTABLE_SELECTOR_TOKENS = frozenset({"latest", "newest", "current", "head", "tip"})

_APPEND_ONLY_TABLES = (
    "checkpoints",
    "checkpoint_policy_types",
    "checkpoint_train_calls",
    "publication_events",
    "policy_set_memberships",
    "lineage_edges",
    "revisions",
    "revision_transitions",
    "save_attempts",
    "evaluation_bindings",
    "evaluation_metrics",
)


class CatalogError(RecordError):
    """A catalog write or read violated the catalog's own contract."""


class ImmutableRecordError(CatalogError):
    """An already-registered record was re-registered with different content."""


class UnknownRecordError(CatalogError):
    """A referenced checkpoint or revision is absent. Never guess a substitute."""


class ArtifactRoleError(CatalogError):
    """A sampler artifact was used as training state, or the reverse."""


class PublicationStatusError(CatalogError):
    """An illegal publication-status transition was attempted."""


class DuplicateSaveError(CatalogError):
    """Two live checkpoints for one parameter group in one published update."""


class BaselineMissingError(CatalogError):
    """Rollout admission was attempted before the baseline was catalogued."""


class LineageError(CatalogError):
    """A lineage edge named a record the catalog does not hold."""


def utc_now() -> str:
    """RFC3339 timestamp in UTC, the catalog's only time format."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{name} is required")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _require_digest(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if any(character.isspace() for character in text):
        raise CatalogError(f"{name} must not contain whitespace")
    algorithm, separator, remainder = text.partition(":")
    candidate = remainder if separator else algorithm
    if separator and not algorithm:
        raise CatalogError(f"{name} must name a digest algorithm before ':'")
    if len(candidate) < 8 or any(
        character not in "0123456789abcdefABCDEF" for character in candidate
    ):
        raise CatalogError(f"{name} must be a hex digest, optionally algorithm-prefixed")
    return text


def _require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogError(f"{name} must be a non-negative integer")
    return value


def _texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    return tuple(_require_text(value, name) for value in values)


@dataclass(frozen=True, slots=True)
class SamplerWeightsRef:
    """Immutable artifact a sampling client is created from. Never resumable."""

    ref: str
    digest: str
    role: ClassVar[str] = SAMPLER_ROLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _require_text(self.ref, "sampler_weights.ref"))
        object.__setattr__(self, "digest", _require_digest(self.digest, "sampler_weights.digest"))

    def to_payload(self) -> dict[str, Any]:
        return {"ref": self.ref, "digest": self.digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SamplerWeightsRef":
        return cls(ref=payload.get("ref", ""), digest=payload.get("digest", ""))


@dataclass(frozen=True, slots=True)
class TrainingStateRef:
    """Resumable training artifact. Never assumed to be directly sampleable."""

    ref: str
    digest: str
    role: ClassVar[str] = TRAINING_STATE_ROLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _require_text(self.ref, "training_state.ref"))
        object.__setattr__(self, "digest", _require_digest(self.digest, "training_state.digest"))

    def to_payload(self) -> dict[str, Any]:
        return {"ref": self.ref, "digest": self.digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingStateRef":
        return cls(ref=payload.get("ref", ""), digest=payload.get("digest", ""))


ArtifactRef = SamplerWeightsRef | TrainingStateRef


def assert_sampler_ref(value: object) -> SamplerWeightsRef:
    """Accept only a sampler artifact. A training state is a typed refusal."""

    if not isinstance(value, SamplerWeightsRef):
        raise ArtifactRoleError(f"expected a {SAMPLER_ROLE} reference, got {type(value).__name__}")
    return value


def assert_training_state_ref(value: object) -> TrainingStateRef:
    """Accept only a resumable artifact. A sampler ref is a typed refusal."""

    if not isinstance(value, TrainingStateRef):
        raise ArtifactRoleError(
            f"expected a {TRAINING_STATE_ROLE} reference, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True, slots=True)
class CheckpointArtifacts:
    """The two provider roles, separately addressed and never interchangeable."""

    sampler_weights: SamplerWeightsRef | None = None
    training_state: TrainingStateRef | None = None

    def __post_init__(self) -> None:
        if self.sampler_weights is None and self.training_state is None:
            raise CatalogError("a checkpoint must carry at least one artifact reference")
        if self.sampler_weights is not None:
            assert_sampler_ref(self.sampler_weights)
        if self.training_state is not None:
            assert_training_state_ref(self.training_state)
        if (
            self.sampler_weights is not None
            and self.training_state is not None
            and self.sampler_weights.ref == self.training_state.ref
        ):
            raise ArtifactRoleError(
                "one provider ref cannot serve both the sampler and the resumable role"
            )

    @property
    def sampler(self) -> SamplerWeightsRef:
        if self.sampler_weights is None:
            raise ArtifactRoleError("checkpoint has no sampler_weights artifact")
        return self.sampler_weights

    @property
    def resumable(self) -> TrainingStateRef:
        if self.training_state is None:
            raise ArtifactRoleError("checkpoint has no training_state artifact")
        return self.training_state

    def ref_for_role(self, role: str) -> ArtifactRef:
        if role == SAMPLER_ROLE:
            return self.sampler
        if role == TRAINING_STATE_ROLE:
            return self.resumable
        raise ArtifactRoleError(
            f"unknown artifact role {role!r}; expected one of {sorted(ARTIFACT_ROLES)}"
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.sampler_weights is not None:
            payload[SAMPLER_ROLE] = self.sampler_weights.to_payload()
        if self.training_state is not None:
            payload[TRAINING_STATE_ROLE] = self.training_state.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CheckpointArtifacts":
        sampler = payload.get(SAMPLER_ROLE)
        state = payload.get(TRAINING_STATE_ROLE)
        return cls(
            sampler_weights=SamplerWeightsRef.from_payload(sampler) if sampler else None,
            training_state=TrainingStateRef.from_payload(state) if state else None,
        )


@dataclass(frozen=True, slots=True)
class TrainingEvidence:
    """What the update that produced this checkpoint actually consumed."""

    groups: tuple[str, ...] = ()
    examples: int = 0
    tokens: int = 0
    provider_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", _texts(self.groups, "training_evidence.groups entry"))
        object.__setattr__(
            self, "examples", _require_count(self.examples, "training_evidence.examples")
        )
        object.__setattr__(self, "tokens", _require_count(self.tokens, "training_evidence.tokens"))
        if not isinstance(self.provider_cost, int | float) or self.provider_cost < 0:
            raise CatalogError("training_evidence.provider_cost must be a non-negative number")
        object.__setattr__(self, "provider_cost", float(self.provider_cost))

    def to_payload(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "examples": self.examples,
            "tokens": self.tokens,
            "provider_cost": self.provider_cost,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingEvidence":
        return cls(
            groups=tuple(payload.get("groups") or ()),
            examples=int(payload.get("examples", 0)),
            tokens=int(payload.get("tokens", 0)),
            provider_cost=float(payload.get("provider_cost", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    """What must still be true for this checkpoint's tokens to mean anything."""

    renderer_profile: str
    tokenizer: str
    container_contract_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "renderer_profile",
            _require_text(self.renderer_profile, "compatibility.renderer_profile"),
        )
        object.__setattr__(
            self, "tokenizer", _require_text(self.tokenizer, "compatibility.tokenizer")
        )
        object.__setattr__(
            self,
            "container_contract_hash",
            _require_digest(self.container_contract_hash, "compatibility.container_contract_hash"),
        )

    @classmethod
    def from_renderer_profile(
        cls, profile: RendererProfile, *, container_contract_hash: str
    ) -> "CheckpointCompatibility":
        return cls(
            renderer_profile=profile.profile_id,
            tokenizer=profile.tokenizer_id,
            container_contract_hash=container_contract_hash,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "renderer_profile": self.renderer_profile,
            "tokenizer": self.tokenizer,
            "container_contract_hash": self.container_contract_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CheckpointCompatibility":
        return cls(
            renderer_profile=payload.get("renderer_profile", ""),
            tokenizer=payload.get("tokenizer", ""),
            container_contract_hash=payload.get("container_contract_hash", ""),
        )


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """``cispo.checkpoint.v1``. Immutable once registered.

    ``publication_status`` and ``policy_set_revision_ids`` are the values *at
    registration*. Later publication transitions and policy-set publications
    are append-only relations; read the effective values through
    :meth:`CheckpointCatalog.describe_checkpoint`.
    """

    checkpoint_id: str
    run_id: str
    update_id: str
    train_call_ids: tuple[str, ...]
    parameter_group_id: str
    policy_type_ids: tuple[str, ...]
    policy_revision_id: str
    base_model: str
    artifacts: CheckpointArtifacts
    training_evidence: TrainingEvidence
    compatibility: CheckpointCompatibility
    created_at: str
    parent_checkpoint_id: str | None = None
    publication_status: str = "staged"
    policy_set_revision_ids: tuple[str, ...] = ()
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "run_id",
            "update_id",
            "parameter_group_id",
            "policy_revision_id",
            "base_model",
            "created_at",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "parent_checkpoint_id",
            _optional_text(self.parent_checkpoint_id, "parent_checkpoint_id"),
        )
        object.__setattr__(
            self, "train_call_ids", _texts(self.train_call_ids, "train_call_ids entry")
        )
        object.__setattr__(
            self, "policy_type_ids", _texts(self.policy_type_ids, "policy_type_ids entry")
        )
        object.__setattr__(
            self,
            "policy_set_revision_ids",
            _texts(self.policy_set_revision_ids, "policy_set_revision_ids entry"),
        )
        if not self.policy_type_ids:
            raise CatalogError("a checkpoint must name at least one policy type")
        if self.publication_status not in REGISTRABLE_STATUSES:
            raise PublicationStatusError(
                f"a checkpoint may only be registered as {sorted(REGISTRABLE_STATUSES)}, "
                f"not {self.publication_status!r}"
            )
        if self.parent_checkpoint_id == self.checkpoint_id:
            raise LineageError("a checkpoint cannot be its own parent")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CatalogError(f"unsupported checkpoint schema {self.schema_version!r}")

    @property
    def sampler_weights(self) -> SamplerWeightsRef:
        """The sampler artifact, or a typed refusal. Never the training state."""

        return self.artifacts.sampler

    @property
    def training_state(self) -> TrainingStateRef:
        """The resumable artifact, or a typed refusal. Never the sampler ref."""

        return self.artifacts.resumable

    @property
    def is_resumable(self) -> bool:
        return self.artifacts.training_state is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "update_id": self.update_id,
            "train_call_ids": list(self.train_call_ids),
            "parameter_group_id": self.parameter_group_id,
            "policy_type_ids": list(self.policy_type_ids),
            "policy_revision_id": self.policy_revision_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "base_model": self.base_model,
            "artifacts": self.artifacts.to_payload(),
            "publication_status": self.publication_status,
            "policy_set_revision_ids": list(self.policy_set_revision_ids),
            "training_evidence": self.training_evidence.to_payload(),
            "compatibility": self.compatibility.to_payload(),
            "created_at": self.created_at,
        }

    @property
    def record_digest(self) -> str:
        return digest(self.to_payload(), length=32)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CheckpointRecord":
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise CatalogError("checkpoint payload missing artifacts object")
        evidence = payload.get("training_evidence") or {}
        compatibility = payload.get("compatibility") or {}
        if not isinstance(evidence, Mapping) or not isinstance(compatibility, Mapping):
            raise CatalogError("checkpoint payload training_evidence/compatibility must be objects")
        return cls(
            checkpoint_id=payload.get("checkpoint_id", ""),
            run_id=payload.get("run_id", ""),
            update_id=payload.get("update_id", ""),
            train_call_ids=tuple(payload.get("train_call_ids") or ()),
            parameter_group_id=payload.get("parameter_group_id", ""),
            policy_type_ids=tuple(payload.get("policy_type_ids") or ()),
            policy_revision_id=payload.get("policy_revision_id", ""),
            base_model=payload.get("base_model", ""),
            artifacts=CheckpointArtifacts.from_payload(artifacts),
            training_evidence=TrainingEvidence.from_payload(evidence),
            compatibility=CheckpointCompatibility.from_payload(compatibility),
            created_at=payload.get("created_at", ""),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            publication_status=payload.get("publication_status", "staged"),
            policy_set_revision_ids=tuple(payload.get("policy_set_revision_ids") or ()),
            schema_version=payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class CheckpointView:
    """A checkpoint plus the append-only relations that accumulated on it."""

    record: CheckpointRecord
    publication_status: str
    policy_set_revision_ids: tuple[str, ...] = ()
    evaluation_ids: tuple[str, ...] = ()

    @property
    def checkpoint_id(self) -> str:
        return self.record.checkpoint_id

    def to_payload(self) -> dict[str, Any]:
        payload = self.record.to_payload()
        payload["publication_status"] = self.publication_status
        payload["policy_set_revision_ids"] = list(self.policy_set_revision_ids)
        payload["evaluation_ids"] = list(self.evaluation_ids)
        return payload


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """One directed relation between a checkpoint and what produced or used it."""

    child_checkpoint_id: str
    relation: str
    parent_checkpoint_id: str | None = None
    revision_id: str | None = None
    run_id: str | None = None
    update_id: str | None = None
    parameter_group_id: str | None = None
    train_call_ids: tuple[str, ...] = ()
    recorded_at: str = ""
    schema_version: str = LINEAGE_EDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "child_checkpoint_id",
            _require_text(self.child_checkpoint_id, "child_checkpoint_id"),
        )
        if self.relation not in LINEAGE_RELATIONS:
            raise LineageError(f"unknown lineage relation {self.relation!r}")
        if self.relation == "parent" and not self.parent_checkpoint_id:
            raise LineageError("a parent edge must name a parent checkpoint")
        if self.relation != "parent" and not self.revision_id:
            raise LineageError(f"a {self.relation} edge must name a revision")
        object.__setattr__(
            self, "train_call_ids", _texts(self.train_call_ids, "train_call_ids entry")
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "child_checkpoint_id": self.child_checkpoint_id,
            "relation": self.relation,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "revision_id": self.revision_id,
            "run_id": self.run_id,
            "update_id": self.update_id,
            "parameter_group_id": self.parameter_group_id,
            "train_call_ids": list(self.train_call_ids),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LineageEdge":
        return cls(
            child_checkpoint_id=payload.get("child_checkpoint_id", ""),
            relation=payload.get("relation", ""),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            revision_id=payload.get("revision_id"),
            run_id=payload.get("run_id"),
            update_id=payload.get("update_id"),
            parameter_group_id=payload.get("parameter_group_id"),
            train_call_ids=tuple(payload.get("train_call_ids") or ()),
            recorded_at=payload.get("recorded_at", ""),
        )


@dataclass(frozen=True, slots=True)
class SaveAttempt:
    """One attempt to materialize an artifact, successful or not."""

    run_id: str
    update_id: str
    parameter_group_id: str
    outcome: str
    checkpoint_id: str | None = None
    error: str | None = None
    packed_group_ids: tuple[str, ...] = ()
    provider_request_ids: tuple[str, ...] = ()
    recorded_at: str = ""
    schema_version: str = SAVE_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "update_id", "parameter_group_id"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        if self.outcome not in SAVE_OUTCOMES:
            raise CatalogError(f"unknown save outcome {self.outcome!r}")
        if self.outcome == "succeeded" and not self.checkpoint_id:
            raise CatalogError("a succeeded save attempt must name its checkpoint")
        if self.outcome == "failed" and not self.error:
            raise CatalogError("a failed save attempt must record why it failed")
        if self.outcome == "failed" and self.checkpoint_id:
            raise CatalogError("a failed save attempt cannot claim a checkpoint")
        object.__setattr__(
            self, "packed_group_ids", _texts(self.packed_group_ids, "packed_group_ids entry")
        )
        object.__setattr__(
            self,
            "provider_request_ids",
            _texts(self.provider_request_ids, "provider_request_ids entry"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "update_id": self.update_id,
            "parameter_group_id": self.parameter_group_id,
            "outcome": self.outcome,
            "checkpoint_id": self.checkpoint_id,
            "error": self.error,
            "packed_group_ids": list(self.packed_group_ids),
            "provider_request_ids": list(self.provider_request_ids),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SaveAttempt":
        return cls(
            run_id=payload.get("run_id", ""),
            update_id=payload.get("update_id", ""),
            parameter_group_id=payload.get("parameter_group_id", ""),
            outcome=payload.get("outcome", ""),
            checkpoint_id=payload.get("checkpoint_id"),
            error=payload.get("error"),
            packed_group_ids=tuple(payload.get("packed_group_ids") or ()),
            provider_request_ids=tuple(payload.get("provider_request_ids") or ()),
            recorded_at=payload.get("recorded_at", ""),
        )


@dataclass(frozen=True, slots=True)
class EvaluationBinding:
    """An append-only relation from an evaluation to exactly what it loaded."""

    evaluation_id: str
    target_kind: str
    target_id: str
    requested_selector: str
    resolved_checkpoint_ids: tuple[str, ...]
    loaded_refs: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    policy_set_revision_id: str | None = None
    match_set_revision_id: str | None = None
    recorded_at: str = ""
    schema_version: str = EVALUATION_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "target_id", "requested_selector"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        if self.target_kind not in EVALUATION_TARGET_KINDS:
            raise CatalogError(f"unknown evaluation target kind {self.target_kind!r}")
        object.__setattr__(
            self,
            "resolved_checkpoint_ids",
            _texts(self.resolved_checkpoint_ids, "resolved_checkpoint_ids entry"),
        )
        if not self.resolved_checkpoint_ids:
            raise CatalogError("an evaluation binding must name the checkpoints it resolved")
        object.__setattr__(self, "loaded_refs", _texts(self.loaded_refs, "loaded_refs entry"))
        metrics: dict[str, float] = {}
        for key, value in dict(self.metrics).items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise CatalogError(f"evaluation metric {key!r} must be a number")
            metrics[_require_text(key, "metric name")] = float(value)
        object.__setattr__(self, "metrics", metrics)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "requested_selector": self.requested_selector,
            "resolved_checkpoint_ids": list(self.resolved_checkpoint_ids),
            "loaded_refs": list(self.loaded_refs),
            "metrics": dict(self.metrics),
            "policy_set_revision_id": self.policy_set_revision_id,
            "match_set_revision_id": self.match_set_revision_id,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvaluationBinding":
        return cls(
            evaluation_id=payload.get("evaluation_id", ""),
            target_kind=payload.get("target_kind", ""),
            target_id=payload.get("target_id", ""),
            requested_selector=payload.get("requested_selector", ""),
            resolved_checkpoint_ids=tuple(payload.get("resolved_checkpoint_ids") or ()),
            loaded_refs=tuple(payload.get("loaded_refs") or ()),
            metrics=dict(payload.get("metrics") or {}),
            policy_set_revision_id=payload.get("policy_set_revision_id"),
            match_set_revision_id=payload.get("match_set_revision_id"),
            recorded_at=payload.get("recorded_at", ""),
        )


@dataclass(frozen=True, slots=True)
class RevisionRow:
    """A stored policy-set or match-set revision, as the catalog holds it."""

    revision_id: str
    revision_kind: str
    family_id: str
    payload: Mapping[str, Any]
    run_id: str | None = None
    update_id: str | None = None
    created_at: str = ""
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class RevisionTransition:
    """One recorded lifecycle transition of a revision."""

    revision_id: str
    transition: str
    attempt_id: str | None = None
    detail: str | None = None
    recorded_at: str = ""
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class AliasPointer:
    """A mutable human pointer. Only ever a route to an immutable id."""

    alias: str
    target_kind: str
    target_id: str
    updated_at: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    update_id TEXT NOT NULL,
    parameter_group_id TEXT NOT NULL,
    policy_revision_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    base_model TEXT NOT NULL,
    registered_status TEXT NOT NULL,
    has_sampler INTEGER NOT NULL,
    has_training_state INTEGER NOT NULL,
    renderer_profile TEXT NOT NULL,
    tokenizer TEXT NOT NULL,
    container_contract_hash TEXT NOT NULL,
    record_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS checkpoints_run ON checkpoints (run_id);
CREATE INDEX IF NOT EXISTS checkpoints_update ON checkpoints (run_id, update_id);
CREATE INDEX IF NOT EXISTS checkpoints_group ON checkpoints (parameter_group_id);
CREATE INDEX IF NOT EXISTS checkpoints_parent ON checkpoints (parent_checkpoint_id);
CREATE TABLE IF NOT EXISTS checkpoint_policy_types (
    checkpoint_id TEXT NOT NULL,
    policy_type_id TEXT NOT NULL,
    PRIMARY KEY (checkpoint_id, policy_type_id)
);
CREATE INDEX IF NOT EXISTS policy_types_by_type ON checkpoint_policy_types (policy_type_id);
CREATE TABLE IF NOT EXISTS checkpoint_train_calls (
    checkpoint_id TEXT NOT NULL,
    train_call_id TEXT NOT NULL,
    PRIMARY KEY (checkpoint_id, train_call_id)
);
CREATE INDEX IF NOT EXISTS train_calls_by_call ON checkpoint_train_calls (train_call_id);
CREATE TABLE IF NOT EXISTS publication_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS publication_by_checkpoint ON publication_events (checkpoint_id, seq);
CREATE TABLE IF NOT EXISTS policy_set_memberships (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    policy_set_revision_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memberships_by_checkpoint ON policy_set_memberships (checkpoint_id);
CREATE INDEX IF NOT EXISTS memberships_by_revision
    ON policy_set_memberships (policy_set_revision_id);
CREATE TABLE IF NOT EXISTS lineage_edges (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    child_checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    relation TEXT NOT NULL,
    revision_id TEXT,
    run_id TEXT,
    update_id TEXT,
    parameter_group_id TEXT,
    recorded_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lineage_by_child ON lineage_edges (child_checkpoint_id);
CREATE INDEX IF NOT EXISTS lineage_by_parent ON lineage_edges (parent_checkpoint_id);
CREATE TABLE IF NOT EXISTS revisions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id TEXT NOT NULL UNIQUE,
    revision_kind TEXT NOT NULL,
    family_id TEXT NOT NULL,
    run_id TEXT,
    update_id TEXT,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS revisions_by_family ON revisions (revision_kind, family_id, seq);
CREATE TABLE IF NOT EXISTS revision_transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id TEXT NOT NULL,
    transition TEXT NOT NULL,
    attempt_id TEXT,
    detail TEXT,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS transitions_by_revision ON revision_transitions (revision_id, seq);
CREATE TABLE IF NOT EXISTS save_attempts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    update_id TEXT NOT NULL,
    parameter_group_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    checkpoint_id TEXT,
    error TEXT,
    recorded_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS saves_by_update ON save_attempts (run_id, update_id);
CREATE TABLE IF NOT EXISTS evaluation_bindings (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL UNIQUE,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    requested_selector TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evaluations_by_target ON evaluation_bindings (target_kind, target_id);
CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
    evaluation_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    PRIMARY KEY (evaluation_id, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS evaluation_checkpoints_by_ckpt ON evaluation_checkpoints (checkpoint_id);
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_by_name ON evaluation_metrics (metric, value);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class CheckpointCatalog:
    """Durable append-only catalog over stdlib sqlite3.

    Open one per process on a run's receipt directory. Every write commits
    immediately unless it is nested in :meth:`transaction`, so an interrupted
    process leaves committed prefixes -- including staged components that never
    got published -- rather than losing them.
    """

    def __init__(self, path: str | Path, *, clock: Callable[[], str] = utc_now) -> None:
        self._path = str(path)
        self._clock = clock
        self._depth = 0
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._install_append_only_triggers()
        self._conn.execute(
            "INSERT OR IGNORE INTO catalog_meta (key, value) VALUES ('checkpoint_schema', ?)",
            (CHECKPOINT_SCHEMA_VERSION,),
        )

    # ------------------------------------------------------------- plumbing

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CheckpointCatalog":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _install_append_only_triggers(self) -> None:
        statements: list[str] = []
        for table in _APPEND_ONLY_TABLES:
            for verb in ("UPDATE", "DELETE"):
                statements.append(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{verb.lower()} "
                    f"BEFORE {verb} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, 'catalog table {table} is append-only'); END;"
                )
        self._conn.executescript("\n".join(statements))

    @contextmanager
    def transaction(self) -> Iterator["CheckpointCatalog"]:
        """Atomic unit of catalog writes: all of them, or none of them."""

        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self
        except BaseException:
            self._depth = 0
            self._conn.execute("ROLLBACK")
            raise
        self._depth = 0
        self._conn.execute("COMMIT")

    def now(self) -> str:
        return self._clock()

    # --------------------------------------------------------------- writes

    def register_checkpoint(self, record: CheckpointRecord) -> CheckpointRecord:
        """Append an immutable checkpoint record. Idempotent by record digest."""

        existing = self._checkpoint_row(record.checkpoint_id)
        if existing is not None:
            if existing["record_digest"] != record.record_digest:
                raise ImmutableRecordError(
                    f"checkpoint {record.checkpoint_id} is already catalogued "
                    "with different content"
                )
            return record
        if record.parent_checkpoint_id and not self.has_checkpoint(record.parent_checkpoint_id):
            raise LineageError(
                f"parent checkpoint {record.parent_checkpoint_id} is absent from the catalog"
            )
        live = self._live_save_for(record.run_id, record.update_id, record.parameter_group_id)
        if live is not None:
            raise DuplicateSaveError(
                f"parameter group {record.parameter_group_id} already has live checkpoint {live} "
                f"for {record.run_id}/{record.update_id}: save once per published update, "
                "not once per packed group"
            )
        timestamp = self.now()
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, run_id, update_id, parameter_group_id, policy_revision_id,
                    parent_checkpoint_id, base_model, registered_status, has_sampler,
                    has_training_state, renderer_profile, tokenizer, container_contract_hash,
                    record_digest, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.checkpoint_id,
                    record.run_id,
                    record.update_id,
                    record.parameter_group_id,
                    record.policy_revision_id,
                    record.parent_checkpoint_id,
                    record.base_model,
                    record.publication_status,
                    int(record.artifacts.sampler_weights is not None),
                    int(record.artifacts.training_state is not None),
                    record.compatibility.renderer_profile,
                    record.compatibility.tokenizer,
                    record.compatibility.container_contract_hash,
                    record.record_digest,
                    record.created_at,
                    json.dumps(record.to_payload(), sort_keys=True),
                ),
            )
            self._conn.executemany(
                "INSERT INTO checkpoint_policy_types (checkpoint_id, policy_type_id) VALUES (?, ?)",
                [(record.checkpoint_id, policy_type) for policy_type in record.policy_type_ids],
            )
            self._conn.executemany(
                "INSERT INTO checkpoint_train_calls (checkpoint_id, train_call_id) VALUES (?, ?)",
                [(record.checkpoint_id, call_id) for call_id in record.train_call_ids],
            )
            self._conn.execute(
                "INSERT INTO publication_events (checkpoint_id, status, reason, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (record.checkpoint_id, record.publication_status, "registered", timestamp),
            )
            if record.parent_checkpoint_id:
                self._append_edge(
                    LineageEdge(
                        child_checkpoint_id=record.checkpoint_id,
                        relation="parent",
                        parent_checkpoint_id=record.parent_checkpoint_id,
                        run_id=record.run_id,
                        update_id=record.update_id,
                        parameter_group_id=record.parameter_group_id,
                        train_call_ids=record.train_call_ids,
                        recorded_at=timestamp,
                    )
                )
            for revision_id in record.policy_set_revision_ids:
                self._append_membership(record.checkpoint_id, revision_id, timestamp)
        return record

    def register_baseline(
        self,
        record: CheckpointRecord,
        *,
        alias: str = "baseline",
        resumed: bool = False,
    ) -> CheckpointRecord:
        """Register the imported baseline and point ``baseline`` aliases at it.

        Rollout admission calls :meth:`assert_baseline_registered`, so this must
        happen before the first attempt is admitted.
        """

        if record.parent_checkpoint_id and not resumed:
            raise LineageError("the imported baseline has no parent checkpoint")
        if resumed and not record.parent_checkpoint_id:
            raise LineageError("a resumed baseline requires its exact parent checkpoint")
        if record.artifacts.sampler_weights is None:
            raise ArtifactRoleError("a baseline must carry a sampler_weights artifact")
        with self.transaction():
            self.register_checkpoint(record)
            if record.publication_status != "published":
                reason = "baseline_resume" if resumed else "baseline_import"
                self.record_publication(record.checkpoint_id, "published", reason=reason)
            self.put_alias(f"{alias}:{record.run_id}", "checkpoint", record.checkpoint_id)
            if self.alias(alias) is None:
                self.put_alias(alias, "checkpoint", record.checkpoint_id)
        return record

    def record_publication(
        self, checkpoint_id: str, status: str, *, reason: str | None = None
    ) -> str:
        """Append a publication transition. Illegal transitions are refused."""

        if status not in PUBLICATION_STATUSES:
            raise PublicationStatusError(f"unknown publication status {status!r}")
        current = self.publication_status(checkpoint_id)
        if status == current:
            return status
        if status not in PUBLICATION_TRANSITIONS[current]:
            raise PublicationStatusError(
                f"checkpoint {checkpoint_id} cannot go {current} -> {status}"
            )
        with self.transaction():
            self._conn.execute(
                "INSERT INTO publication_events (checkpoint_id, status, reason, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (checkpoint_id, status, reason, self.now()),
            )
        return status

    def record_save_attempt(self, attempt: SaveAttempt) -> SaveAttempt:
        """Append a save attempt, including the ones that failed."""

        if attempt.checkpoint_id and not self.has_checkpoint(attempt.checkpoint_id):
            raise UnknownRecordError(
                f"save attempt names checkpoint {attempt.checkpoint_id}, absent from the catalog"
            )
        recorded = attempt
        if not attempt.recorded_at:
            recorded = SaveAttempt(
                run_id=attempt.run_id,
                update_id=attempt.update_id,
                parameter_group_id=attempt.parameter_group_id,
                outcome=attempt.outcome,
                checkpoint_id=attempt.checkpoint_id,
                error=attempt.error,
                packed_group_ids=attempt.packed_group_ids,
                provider_request_ids=attempt.provider_request_ids,
                recorded_at=self.now(),
            )
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO save_attempts (
                    run_id, update_id, parameter_group_id, outcome, checkpoint_id, error,
                    recorded_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recorded.run_id,
                    recorded.update_id,
                    recorded.parameter_group_id,
                    recorded.outcome,
                    recorded.checkpoint_id,
                    recorded.error,
                    recorded.recorded_at,
                    json.dumps(recorded.to_payload(), sort_keys=True),
                ),
            )
        return recorded

    def record_lineage_edge(self, edge: LineageEdge) -> LineageEdge:
        """Append a lineage edge. Both endpoints must already be catalogued."""

        if not self.has_checkpoint(edge.child_checkpoint_id):
            raise LineageError(f"lineage child {edge.child_checkpoint_id} is absent")
        if edge.parent_checkpoint_id and not self.has_checkpoint(edge.parent_checkpoint_id):
            raise LineageError(f"lineage parent {edge.parent_checkpoint_id} is absent")
        recorded = edge if edge.recorded_at else _with_time(edge, self.now())
        with self.transaction():
            self._append_edge(recorded)
        return recorded

    def record_policy_set_membership(self, checkpoint_id: str, policy_set_revision_id: str) -> None:
        """Append the fact that a checkpoint is a component of a published set."""

        if not self.has_checkpoint(checkpoint_id):
            raise UnknownRecordError(f"checkpoint {checkpoint_id} is absent from the catalog")
        with self.transaction():
            self._append_membership(checkpoint_id, policy_set_revision_id, self.now())

    def put_revision(
        self,
        *,
        revision_id: str,
        revision_kind: str,
        family_id: str,
        payload: Mapping[str, Any],
        run_id: str | None = None,
        update_id: str | None = None,
        created_at: str | None = None,
    ) -> RevisionRow:
        """Append an immutable policy-set or match-set revision."""

        revision_id = _require_text(revision_id, "revision_id")
        if revision_kind not in REVISION_KINDS:
            raise CatalogError(f"unknown revision kind {revision_kind!r}")
        family_id = _require_text(family_id, "family_id")
        encoded = json.dumps(dict(payload), sort_keys=True)
        existing = self._conn.execute(
            "SELECT payload FROM revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        if existing is not None:
            if existing["payload"] != encoded:
                raise ImmutableRecordError(
                    f"revision {revision_id} is already catalogued with different content"
                )
            return self.get_revision(revision_id)
        timestamp = created_at or self.now()
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO revisions (
                    revision_id, revision_kind, family_id, run_id, update_id, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (revision_id, revision_kind, family_id, run_id, update_id, timestamp, encoded),
            )
            self._conn.execute(
                "INSERT INTO revision_transitions (revision_id, transition, attempt_id, detail, "
                "recorded_at) VALUES (?, 'created', NULL, NULL, ?)",
                (revision_id, timestamp),
            )
        return self.get_revision(revision_id)

    def record_revision_transition(
        self,
        revision_id: str,
        transition: str,
        *,
        attempt_id: str | None = None,
        detail: str | None = None,
    ) -> RevisionTransition:
        """Append one load / ready / attempt / retire transition."""

        if transition not in REVISION_TRANSITIONS:
            raise CatalogError(f"unknown revision transition {transition!r}")
        if not self.has_revision(revision_id):
            raise UnknownRecordError(f"revision {revision_id} is absent from the catalog")
        if transition in {"attempt_open", "attempt_close"} and not attempt_id:
            raise CatalogError(f"{transition} must name the attempt it counts")
        timestamp = self.now()
        with self.transaction():
            self._conn.execute(
                "INSERT INTO revision_transitions (revision_id, transition, attempt_id, detail, "
                "recorded_at) VALUES (?, ?, ?, ?, ?)",
                (revision_id, transition, attempt_id, detail, timestamp),
            )
        return RevisionTransition(
            revision_id=revision_id,
            transition=transition,
            attempt_id=attempt_id,
            detail=detail,
            recorded_at=timestamp,
        )

    def record_evaluation(self, binding: EvaluationBinding) -> EvaluationBinding:
        """Append an evaluation relation. Never mutates the checkpoint record."""

        for checkpoint_id in binding.resolved_checkpoint_ids:
            if not self.has_checkpoint(checkpoint_id):
                raise UnknownRecordError(
                    f"evaluation resolved checkpoint {checkpoint_id}, absent from the catalog"
                )
        recorded = binding
        if not binding.recorded_at:
            payload = binding.to_payload()
            payload["recorded_at"] = self.now()
            recorded = EvaluationBinding.from_payload(payload)
        encoded = json.dumps(recorded.to_payload(), sort_keys=True)
        existing = self._conn.execute(
            "SELECT payload FROM evaluation_bindings WHERE evaluation_id = ?",
            (recorded.evaluation_id,),
        ).fetchone()
        if existing is not None:
            if existing["payload"] != encoded:
                raise ImmutableRecordError(
                    f"evaluation {recorded.evaluation_id} is already catalogued differently"
                )
            return recorded
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO evaluation_bindings (
                    evaluation_id, target_kind, target_id, requested_selector, recorded_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    recorded.evaluation_id,
                    recorded.target_kind,
                    recorded.target_id,
                    recorded.requested_selector,
                    recorded.recorded_at,
                    encoded,
                ),
            )
            self._conn.executemany(
                "INSERT INTO evaluation_checkpoints (evaluation_id, checkpoint_id) VALUES (?, ?)",
                [
                    (recorded.evaluation_id, checkpoint_id)
                    for checkpoint_id in recorded.resolved_checkpoint_ids
                ],
            )
            self._conn.executemany(
                """
                INSERT INTO evaluation_metrics (
                    evaluation_id, target_kind, target_id, metric, value, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        recorded.evaluation_id,
                        recorded.target_kind,
                        recorded.target_id,
                        metric,
                        value,
                        recorded.recorded_at,
                    )
                    for metric, value in sorted(recorded.metrics.items())
                ],
            )
        return recorded

    def put_alias(self, alias: str, target_kind: str, target_id: str) -> AliasPointer:
        """Move a mutable human pointer. The target must be immutable and known."""

        alias = _require_text(alias, "alias")
        if target_kind not in ALIAS_TARGET_KINDS:
            raise CatalogError(f"unknown alias target kind {target_kind!r}")
        if alias.split(":", 1)[0] in MUTABLE_SELECTOR_TOKENS:
            raise CatalogError(f"alias {alias!r} collides with a forbidden mutable selector")
        if target_kind == "checkpoint" and not self.has_checkpoint(target_id):
            raise UnknownRecordError(f"alias target checkpoint {target_id} is absent")
        if target_kind != "checkpoint" and not self.has_revision(target_id):
            raise UnknownRecordError(f"alias target revision {target_id} is absent")
        timestamp = self.now()
        with self.transaction():
            self._conn.execute(
                "INSERT INTO aliases (alias, target_kind, target_id, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(alias) DO UPDATE SET target_kind = excluded.target_kind, "
                "target_id = excluded.target_id, updated_at = excluded.updated_at",
                (alias, target_kind, target_id, timestamp),
            )
        return AliasPointer(
            alias=alias, target_kind=target_kind, target_id=target_id, updated_at=timestamp
        )

    # ---------------------------------------------------------------- reads

    def has_checkpoint(self, checkpoint_id: str) -> bool:
        return self._checkpoint_row(checkpoint_id) is not None

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        row = self._checkpoint_row(checkpoint_id)
        if row is None:
            raise UnknownRecordError(f"checkpoint {checkpoint_id!r} is absent from the catalog")
        return CheckpointRecord.from_payload(json.loads(row["payload"]))

    def publication_status(self, checkpoint_id: str) -> str:
        row = self._conn.execute(
            "SELECT status FROM publication_events WHERE checkpoint_id = ? "
            "ORDER BY seq DESC LIMIT 1",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            raise UnknownRecordError(f"checkpoint {checkpoint_id!r} is absent from the catalog")
        return str(row["status"])

    def publication_history(self, checkpoint_id: str) -> tuple[tuple[str, str, str | None], ...]:
        rows = self._conn.execute(
            "SELECT status, recorded_at, reason FROM publication_events WHERE checkpoint_id = ? "
            "ORDER BY seq",
            (checkpoint_id,),
        ).fetchall()
        return tuple((row["status"], row["recorded_at"], row["reason"]) for row in rows)

    def describe_checkpoint(self, checkpoint_id: str) -> CheckpointView:
        record = self.get_checkpoint(checkpoint_id)
        return CheckpointView(
            record=record,
            publication_status=self.publication_status(checkpoint_id),
            policy_set_revision_ids=self.memberships_of(checkpoint_id),
            evaluation_ids=self.evaluation_ids_for(checkpoint_id),
        )

    def list_checkpoints(
        self,
        *,
        run_id: str | None = None,
        update_id: str | None = None,
        parameter_group_id: str | None = None,
        policy_type_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        publication_status: str | None = None,
        base_model: str | None = None,
        train_call_id: str | None = None,
        policy_set_revision_id: str | None = None,
        evaluation_metric: str | None = None,
        limit: int | None = None,
    ) -> tuple[CheckpointView, ...]:
        """Every declared index over the catalog, as one composable query."""

        if publication_status is not None and publication_status not in PUBLICATION_STATUSES:
            raise CatalogError(f"unknown publication status {publication_status!r}")
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("c.run_id = ?")
            params.append(run_id)
        if update_id is not None:
            clauses.append("c.update_id = ?")
            params.append(update_id)
        if parameter_group_id is not None:
            clauses.append("c.parameter_group_id = ?")
            params.append(parameter_group_id)
        if parent_checkpoint_id is not None:
            clauses.append("c.parent_checkpoint_id = ?")
            params.append(parent_checkpoint_id)
        if base_model is not None:
            clauses.append("c.base_model = ?")
            params.append(base_model)
        if policy_type_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM checkpoint_policy_types t "
                "WHERE t.checkpoint_id = c.checkpoint_id AND t.policy_type_id = ?)"
            )
            params.append(policy_type_id)
        if train_call_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM checkpoint_train_calls t "
                "WHERE t.checkpoint_id = c.checkpoint_id AND t.train_call_id = ?)"
            )
            params.append(train_call_id)
        if policy_set_revision_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM policy_set_memberships m "
                "WHERE m.checkpoint_id = c.checkpoint_id AND m.policy_set_revision_id = ?)"
            )
            params.append(policy_set_revision_id)
        if evaluation_metric is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluation_checkpoints e "
                "JOIN evaluation_metrics v ON v.evaluation_id = e.evaluation_id "
                "WHERE e.checkpoint_id = c.checkpoint_id AND v.metric = ?)"
            )
            params.append(evaluation_metric)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT checkpoint_id, payload, status FROM ("
            "SELECT c.checkpoint_id AS checkpoint_id, c.payload AS payload, c.seq AS seq, "
            "(SELECT status FROM publication_events e WHERE e.checkpoint_id = c.checkpoint_id "
            "ORDER BY e.seq DESC LIMIT 1) AS status "
            f"FROM checkpoints c {where}) "
        )
        if publication_status is not None:
            sql += "WHERE status = ? "
            params.append(publication_status)
        sql += "ORDER BY seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(_require_count(limit, "limit"))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        views: list[CheckpointView] = []
        for row in rows:
            record = CheckpointRecord.from_payload(json.loads(row["payload"]))
            views.append(
                CheckpointView(
                    record=record,
                    publication_status=str(row["status"]),
                    policy_set_revision_ids=self.memberships_of(record.checkpoint_id),
                    evaluation_ids=self.evaluation_ids_for(record.checkpoint_id),
                )
            )
        return tuple(views)

    def memberships_of(self, checkpoint_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT DISTINCT policy_set_revision_id FROM policy_set_memberships "
            "WHERE checkpoint_id = ? ORDER BY policy_set_revision_id",
            (checkpoint_id,),
        ).fetchall()
        return tuple(str(row["policy_set_revision_id"]) for row in rows)

    def policy_set_members(self, policy_set_revision_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT DISTINCT checkpoint_id FROM policy_set_memberships "
            "WHERE policy_set_revision_id = ? ORDER BY checkpoint_id",
            (policy_set_revision_id,),
        ).fetchall()
        return tuple(str(row["checkpoint_id"]) for row in rows)

    def evaluation_ids_for(self, checkpoint_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT evaluation_id FROM evaluation_checkpoints WHERE checkpoint_id = ? "
            "ORDER BY evaluation_id",
            (checkpoint_id,),
        ).fetchall()
        return tuple(str(row["evaluation_id"]) for row in rows)

    def lineage_edges(
        self,
        *,
        child_checkpoint_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        revision_id: str | None = None,
        relation: str | None = None,
    ) -> tuple[LineageEdge, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if child_checkpoint_id is not None:
            clauses.append("child_checkpoint_id = ?")
            params.append(child_checkpoint_id)
        if parent_checkpoint_id is not None:
            clauses.append("parent_checkpoint_id = ?")
            params.append(parent_checkpoint_id)
        if revision_id is not None:
            clauses.append("revision_id = ?")
            params.append(revision_id)
        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload FROM lineage_edges {where} ORDER BY seq", tuple(params)
        ).fetchall()
        return tuple(LineageEdge.from_payload(json.loads(row["payload"])) for row in rows)

    def ancestry(self, checkpoint_id: str) -> tuple[str, ...]:
        """Parent chain, nearest first. Stops at the imported baseline."""

        chain: list[str] = []
        seen = {checkpoint_id}
        cursor = self.get_checkpoint(checkpoint_id).parent_checkpoint_id
        while cursor:
            if cursor in seen:
                raise LineageError(f"lineage cycle through {cursor}")
            seen.add(cursor)
            chain.append(cursor)
            cursor = self.get_checkpoint(cursor).parent_checkpoint_id
        return tuple(chain)

    def has_revision(self, revision_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        return row is not None

    def revision_kind(self, revision_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT revision_kind FROM revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        return None if row is None else str(row["revision_kind"])

    def get_revision(self, revision_id: str) -> RevisionRow:
        row = self._conn.execute(
            "SELECT * FROM revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()
        if row is None:
            raise UnknownRecordError(f"revision {revision_id!r} is absent from the catalog")
        return RevisionRow(
            revision_id=str(row["revision_id"]),
            revision_kind=str(row["revision_kind"]),
            family_id=str(row["family_id"]),
            payload=json.loads(row["payload"]),
            run_id=row["run_id"],
            update_id=row["update_id"],
            created_at=str(row["created_at"]),
            sequence=int(row["seq"]),
        )

    def list_revisions(
        self,
        *,
        revision_kind: str | None = None,
        family_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[RevisionRow, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if revision_kind is not None:
            if revision_kind not in REVISION_KINDS:
                raise CatalogError(f"unknown revision kind {revision_kind!r}")
            clauses.append("revision_kind = ?")
            params.append(revision_kind)
        if family_id is not None:
            clauses.append("family_id = ?")
            params.append(family_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT revision_id FROM revisions {where} ORDER BY seq", tuple(params)
        ).fetchall()
        return tuple(self.get_revision(str(row["revision_id"])) for row in rows)

    def revision_transitions(self, revision_id: str) -> tuple[RevisionTransition, ...]:
        rows = self._conn.execute(
            "SELECT * FROM revision_transitions WHERE revision_id = ? ORDER BY seq",
            (revision_id,),
        ).fetchall()
        return tuple(
            RevisionTransition(
                revision_id=str(row["revision_id"]),
                transition=str(row["transition"]),
                attempt_id=row["attempt_id"],
                detail=row["detail"],
                recorded_at=str(row["recorded_at"]),
                sequence=int(row["seq"]),
            )
            for row in rows
        )

    def has_transition(self, revision_id: str, transition: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM revision_transitions WHERE revision_id = ? AND transition = ? LIMIT 1",
            (revision_id, transition),
        ).fetchone()
        return row is not None

    def active_attempts(self, revision_id: str) -> tuple[str, ...]:
        """Attempts that opened on this revision and have not closed."""

        opened = self._conn.execute(
            "SELECT DISTINCT attempt_id FROM revision_transitions "
            "WHERE revision_id = ? AND transition = 'attempt_open'",
            (revision_id,),
        ).fetchall()
        closed = self._conn.execute(
            "SELECT DISTINCT attempt_id FROM revision_transitions "
            "WHERE revision_id = ? AND transition = 'attempt_close'",
            (revision_id,),
        ).fetchall()
        closed_ids = {str(row["attempt_id"]) for row in closed}
        return tuple(
            sorted(
                str(row["attempt_id"]) for row in opened if str(row["attempt_id"]) not in closed_ids
            )
        )

    def active_revision_id(
        self, family_id: str, *, revision_kind: str = "policy_set"
    ) -> str | None:
        """The newest revision of a family that is neither superseded nor retired."""

        rows = self._conn.execute(
            "SELECT revision_id FROM revisions WHERE revision_kind = ? AND family_id = ? "
            "ORDER BY seq DESC",
            (revision_kind, family_id),
        ).fetchall()
        for row in rows:
            revision_id = str(row["revision_id"])
            if self.has_transition(revision_id, "retire"):
                continue
            if self.has_transition(revision_id, "superseded"):
                continue
            return revision_id
        return None

    def save_attempts(
        self,
        *,
        run_id: str | None = None,
        update_id: str | None = None,
        parameter_group_id: str | None = None,
        outcome: str | None = None,
    ) -> tuple[SaveAttempt, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if update_id is not None:
            clauses.append("update_id = ?")
            params.append(update_id)
        if parameter_group_id is not None:
            clauses.append("parameter_group_id = ?")
            params.append(parameter_group_id)
        if outcome is not None:
            if outcome not in SAVE_OUTCOMES:
                raise CatalogError(f"unknown save outcome {outcome!r}")
            clauses.append("outcome = ?")
            params.append(outcome)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload FROM save_attempts {where} ORDER BY seq", tuple(params)
        ).fetchall()
        return tuple(SaveAttempt.from_payload(json.loads(row["payload"])) for row in rows)

    def saves_for_update(self, run_id: str, update_id: str) -> Mapping[str, tuple[str, ...]]:
        """Live checkpoints per parameter group for one update: one each, or a bug."""

        result: dict[str, list[str]] = {}
        for view in self.list_checkpoints(run_id=run_id, update_id=update_id):
            if view.publication_status == "orphaned":
                continue
            result.setdefault(view.record.parameter_group_id, []).append(view.checkpoint_id)
        return {group: tuple(ids) for group, ids in sorted(result.items())}

    def evaluations(
        self,
        *,
        target_id: str | None = None,
        target_kind: str | None = None,
        checkpoint_id: str | None = None,
        metric: str | None = None,
    ) -> tuple[EvaluationBinding, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if target_id is not None:
            clauses.append("b.target_id = ?")
            params.append(target_id)
        if target_kind is not None:
            if target_kind not in EVALUATION_TARGET_KINDS:
                raise CatalogError(f"unknown evaluation target kind {target_kind!r}")
            clauses.append("b.target_kind = ?")
            params.append(target_kind)
        if checkpoint_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluation_checkpoints e "
                "WHERE e.evaluation_id = b.evaluation_id AND e.checkpoint_id = ?)"
            )
            params.append(checkpoint_id)
        if metric is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluation_metrics v "
                "WHERE v.evaluation_id = b.evaluation_id AND v.metric = ?)"
            )
            params.append(metric)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT b.payload AS payload FROM evaluation_bindings b {where} ORDER BY b.seq",
            tuple(params),
        ).fetchall()
        return tuple(EvaluationBinding.from_payload(json.loads(row["payload"])) for row in rows)

    def metric_rows(
        self, metric: str, *, target_kind: str | None = None, direction: str = "max"
    ) -> tuple[tuple[str, str, float], ...]:
        """``(target_kind, target_id, value)`` for one metric, best first."""

        if direction not in {"max", "min"}:
            raise CatalogError(f"unknown metric direction {direction!r}")
        clauses = ["metric = ?"]
        params: list[Any] = [_require_text(metric, "metric")]
        if target_kind is not None:
            clauses.append("target_kind = ?")
            params.append(target_kind)
        order = "DESC" if direction == "max" else "ASC"
        rows = self._conn.execute(
            f"SELECT target_kind, target_id, value FROM evaluation_metrics "
            f"WHERE {' AND '.join(clauses)} ORDER BY value {order}, seq ASC",
            tuple(params),
        ).fetchall()
        return tuple(
            (str(row["target_kind"]), str(row["target_id"]), float(row["value"])) for row in rows
        )

    def alias(self, alias: str) -> AliasPointer | None:
        row = self._conn.execute(
            "SELECT * FROM aliases WHERE alias = ?", (_require_text(alias, "alias"),)
        ).fetchone()
        if row is None:
            return None
        return AliasPointer(
            alias=str(row["alias"]),
            target_kind=str(row["target_kind"]),
            target_id=str(row["target_id"]),
            updated_at=str(row["updated_at"]),
        )

    def list_aliases(self) -> tuple[AliasPointer, ...]:
        rows = self._conn.execute("SELECT alias FROM aliases ORDER BY alias").fetchall()
        pointers = [self.alias(str(row["alias"])) for row in rows]
        return tuple(pointer for pointer in pointers if pointer is not None)

    def assert_baseline_registered(self, run_id: str) -> CheckpointRecord:
        """Rollout admission gate: no baseline in the catalog, no admission."""

        pointer = self.alias(f"baseline:{_require_text(run_id, 'run_id')}")
        if pointer is None:
            raise BaselineMissingError(
                f"run {run_id} has no baseline checkpoint in the catalog; "
                "register the imported baseline before admitting rollouts"
            )
        return self.get_checkpoint(pointer.target_id)

    # -------------------------------------------------------------- private

    def _checkpoint_row(self, checkpoint_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
            (_require_text(checkpoint_id, "checkpoint_id"),),
        ).fetchone()

    def _live_save_for(self, run_id: str, update_id: str, parameter_group_id: str) -> str | None:
        rows = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE run_id = ? AND update_id = ? "
            "AND parameter_group_id = ?",
            (run_id, update_id, parameter_group_id),
        ).fetchall()
        for row in rows:
            checkpoint_id = str(row["checkpoint_id"])
            if self.publication_status(checkpoint_id) != "orphaned":
                return checkpoint_id
        return None

    def _append_edge(self, edge: LineageEdge) -> None:
        self._conn.execute(
            """
            INSERT INTO lineage_edges (
                child_checkpoint_id, parent_checkpoint_id, relation, revision_id, run_id,
                update_id, parameter_group_id, recorded_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.child_checkpoint_id,
                edge.parent_checkpoint_id,
                edge.relation,
                edge.revision_id,
                edge.run_id,
                edge.update_id,
                edge.parameter_group_id,
                edge.recorded_at,
                json.dumps(edge.to_payload(), sort_keys=True),
            ),
        )

    def _append_membership(self, checkpoint_id: str, revision_id: str, timestamp: str) -> None:
        self._conn.execute(
            "INSERT INTO policy_set_memberships (checkpoint_id, policy_set_revision_id, "
            "recorded_at) VALUES (?, ?, ?)",
            (checkpoint_id, _require_text(revision_id, "policy_set_revision_id"), timestamp),
        )


def _with_time(edge: LineageEdge, timestamp: str) -> LineageEdge:
    payload = edge.to_payload()
    payload["recorded_at"] = timestamp
    return LineageEdge.from_payload(payload)


def checkpoint_id_for(
    *,
    run_id: str,
    update_id: str,
    parameter_group_id: str,
    policy_revision_id: str,
    role_salt: str = "",
) -> str:
    """Deterministic immutable id for a materialized checkpoint."""

    return "ckpt_" + digest(
        {
            "run_id": _require_text(run_id, "run_id"),
            "update_id": _require_text(update_id, "update_id"),
            "parameter_group_id": _require_text(parameter_group_id, "parameter_group_id"),
            "policy_revision_id": _require_text(policy_revision_id, "policy_revision_id"),
            "role_salt": role_salt,
        },
        length=24,
    )


def sequence_of(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize an id sequence, refusing blanks. Used by the publisher."""

    return _texts(values, "identifier")


__all__ = [
    "ALIAS_TARGET_KINDS",
    "AliasPointer",
    "ArtifactRef",
    "ArtifactRoleError",
    "BaselineMissingError",
    "CHECKPOINT_SCHEMA_VERSION",
    "CatalogError",
    "CheckpointArtifacts",
    "CheckpointCatalog",
    "CheckpointCompatibility",
    "CheckpointRecord",
    "CheckpointView",
    "DuplicateSaveError",
    "EVALUATION_BINDING_SCHEMA_VERSION",
    "EVALUATION_TARGET_KINDS",
    "EvaluationBinding",
    "ImmutableRecordError",
    "LINEAGE_RELATIONS",
    "LineageEdge",
    "LineageError",
    "MUTABLE_SELECTOR_TOKENS",
    "PUBLICATION_STATUSES",
    "PUBLICATION_TRANSITIONS",
    "PublicationStatusError",
    "REGISTRABLE_STATUSES",
    "RESOLVABLE_STATUSES",
    "REVISION_KINDS",
    "REVISION_TRANSITIONS",
    "RevisionRow",
    "RevisionTransition",
    "SAMPLER_ROLE",
    "SAVE_OUTCOMES",
    "SamplerWeightsRef",
    "SaveAttempt",
    "TRAINING_STATE_ROLE",
    "TrainingEvidence",
    "TrainingStateRef",
    "UnknownRecordError",
    "assert_sampler_ref",
    "assert_training_state_ref",
    "checkpoint_id_for",
    "sequence_of",
    "utc_now",
]
