"""Policy-set and match-set revisions, atomically published or not at all.

A multi-policy update produces one component checkpoint per parameter group.
Those components only mean something together: a rollout that samples a new
miner policy against an old scout policy is not a sample of either revision.
So publication is atomic -- every component or none -- and a one-sided failure
leaves the previously active revision live while still cataloguing whatever was
successfully materialized as a staged or orphaned artifact. Losing it would
lose the provider spend as well as the evidence.

A match-set revision closes the competitive case: it pins the trainee policy
set plus every non-trainable opponent's frozen checkpoint id, external model
identity, or scripted-baseline identity, because a reward earned against one
opponent set is not comparable to a reward earned against another.

Publication marks a revision ready only after its sampler artifact is
materialized and health-checked, and a revision may be retired only when its
active-attempt count reaches zero: unloading a revision an in-flight attempt is
still sampling from would make that attempt's evidence unusable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from synth_optimizers.rl.catalog import (
    MUTABLE_SELECTOR_TOKENS,
    CatalogError,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    LineageEdge,
    SamplerWeightsRef,
    SaveAttempt,
    utc_now,
)

POLICY_SET_SCHEMA_VERSION = "cispo.policy_set.v1"
MATCH_SET_SCHEMA_VERSION = "cispo.match_set.v1"

OPPONENT_BINDING_KINDS = frozenset({"pinned_checkpoint", "external_model", "scripted_baseline"})

ORPHAN_REASON = "one_sided_publication"
ORPHAN_RETENTION = "retain_for_run_receipt"


class PolicySetError(CatalogError):
    """A policy-set or match-set revision was invalid or unpublishable."""


class MissingComponentError(PolicySetError):
    """Publication fails closed: a component checkpoint is not catalogued."""


class PartialPublicationError(PolicySetError):
    """One component saved and another did not. Prior revision stays active."""

    def __init__(self, message: str, outcome: "AtomicPublication") -> None:
        super().__init__(message)
        self.outcome = outcome


class ReadinessError(PolicySetError):
    """A revision was used before it was loaded, health-checked, and ready."""


class HealthCheckError(PolicySetError):
    """A materialized sampler artifact failed its health check."""


class RetirementError(PolicySetError):
    """Retirement refused: something is still sampling from this revision."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicySetError(f"{name} is required")
    return value.strip()


def _reject_mutable_identity(value: str, name: str) -> str:
    text = _require_text(value, name)
    head = text.split(":", 1)[0].strip().lower()
    if head in MUTABLE_SELECTOR_TOKENS or text.lower().startswith("best:"):
        raise PolicySetError(
            f"{name} {text!r} is a mutable selector; pin an immutable identity instead"
        )
    return text


@dataclass(frozen=True, slots=True)
class PolicySetComponent:
    """One parameter group's contribution to a team revision."""

    policy_type_id: str
    parameter_group_id: str
    checkpoint_id: str
    policy_revision_id: str

    def __post_init__(self) -> None:
        for name in (
            "policy_type_id",
            "parameter_group_id",
            "checkpoint_id",
            "policy_revision_id",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        _reject_mutable_identity(self.checkpoint_id, "component checkpoint_id")

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_type_id": self.policy_type_id,
            "parameter_group_id": self.parameter_group_id,
            "checkpoint_id": self.checkpoint_id,
            "policy_revision_id": self.policy_revision_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PolicySetComponent":
        return cls(
            policy_type_id=payload.get("policy_type_id", ""),
            parameter_group_id=payload.get("parameter_group_id", ""),
            checkpoint_id=payload.get("checkpoint_id", ""),
            policy_revision_id=payload.get("policy_revision_id", ""),
        )


@dataclass(frozen=True, slots=True)
class PolicySetRevision:
    """Atomic manifest naming every component checkpoint of one team."""

    policy_set_revision_id: str
    policy_set_id: str
    run_id: str
    update_id: str
    components: tuple[PolicySetComponent, ...]
    created_at: str
    parent_policy_set_revision_id: str | None = None
    schema_version: str = POLICY_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "policy_set_revision_id",
            "policy_set_id",
            "run_id",
            "update_id",
            "created_at",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        _reject_mutable_identity(self.policy_set_revision_id, "policy_set_revision_id")
        if not self.components:
            raise PolicySetError("a policy-set revision must name at least one component")
        groups = [component.parameter_group_id for component in self.components]
        types = [component.policy_type_id for component in self.components]
        if len(set(groups)) != len(groups):
            raise PolicySetError("a policy-set revision names one component per parameter group")
        if len(set(types)) != len(types):
            raise PolicySetError("a policy-set revision names one component per policy type")
        if self.schema_version != POLICY_SET_SCHEMA_VERSION:
            raise PolicySetError(f"unsupported policy-set schema {self.schema_version!r}")

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(component.checkpoint_id for component in self.components)

    @property
    def parameter_group_ids(self) -> tuple[str, ...]:
        return tuple(component.parameter_group_id for component in self.components)

    def component_for_policy_type(self, policy_type_id: str) -> PolicySetComponent:
        for component in self.components:
            if component.policy_type_id == policy_type_id:
                return component
        raise PolicySetError(
            f"policy set {self.policy_set_revision_id} has no component for policy type "
            f"{policy_type_id!r}"
        )

    def component_for_group(self, parameter_group_id: str) -> PolicySetComponent:
        for component in self.components:
            if component.parameter_group_id == parameter_group_id:
                return component
        raise PolicySetError(
            f"policy set {self.policy_set_revision_id} has no component for parameter group "
            f"{parameter_group_id!r}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_set_revision_id": self.policy_set_revision_id,
            "policy_set_id": self.policy_set_id,
            "run_id": self.run_id,
            "update_id": self.update_id,
            "parent_policy_set_revision_id": self.parent_policy_set_revision_id,
            "components": [component.to_payload() for component in self.components],
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PolicySetRevision":
        components = payload.get("components") or ()
        if not isinstance(components, Sequence) or isinstance(components, str | bytes):
            raise PolicySetError("policy-set payload components must be a list")
        return cls(
            policy_set_revision_id=payload.get("policy_set_revision_id", ""),
            policy_set_id=payload.get("policy_set_id", ""),
            run_id=payload.get("run_id", ""),
            update_id=payload.get("update_id", ""),
            components=tuple(PolicySetComponent.from_payload(item) for item in components),
            created_at=payload.get("created_at", ""),
            parent_policy_set_revision_id=payload.get("parent_policy_set_revision_id"),
            schema_version=payload.get("schema_version", POLICY_SET_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class OpponentBinding:
    """One non-trainable instance, pinned to something that cannot move."""

    opponent_id: str
    binding_kind: str
    identity: str
    role_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "opponent_id", _require_text(self.opponent_id, "opponent_id"))
        if self.binding_kind not in OPPONENT_BINDING_KINDS:
            raise PolicySetError(
                f"unknown opponent binding kind {self.binding_kind!r}; "
                f"expected one of {sorted(OPPONENT_BINDING_KINDS)}"
            )
        object.__setattr__(self, "identity", _reject_mutable_identity(self.identity, "identity"))

    @property
    def is_pinned_checkpoint(self) -> bool:
        return self.binding_kind == "pinned_checkpoint"

    def to_payload(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "binding_kind": self.binding_kind,
            "identity": self.identity,
            "role_id": self.role_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OpponentBinding":
        return cls(
            opponent_id=payload.get("opponent_id", ""),
            binding_kind=payload.get("binding_kind", ""),
            identity=payload.get("identity", ""),
            role_id=payload.get("role_id"),
        )


@dataclass(frozen=True, slots=True)
class MatchSetRevision:
    """Trainee policy set plus every opponent's pinned identity."""

    match_set_revision_id: str
    match_set_id: str
    run_id: str
    policy_set_revision_id: str
    opponents: tuple[OpponentBinding, ...]
    created_at: str
    schema_version: str = MATCH_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "match_set_revision_id",
            "match_set_id",
            "run_id",
            "policy_set_revision_id",
            "created_at",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        _reject_mutable_identity(self.match_set_revision_id, "match_set_revision_id")
        _reject_mutable_identity(self.policy_set_revision_id, "policy_set_revision_id")
        ids = [opponent.opponent_id for opponent in self.opponents]
        if len(set(ids)) != len(ids):
            raise PolicySetError("a match set binds each opponent instance exactly once")
        if self.schema_version != MATCH_SET_SCHEMA_VERSION:
            raise PolicySetError(f"unsupported match-set schema {self.schema_version!r}")

    @property
    def pinned_checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(
            opponent.identity for opponent in self.opponents if opponent.is_pinned_checkpoint
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_set_revision_id": self.match_set_revision_id,
            "match_set_id": self.match_set_id,
            "run_id": self.run_id,
            "policy_set_revision_id": self.policy_set_revision_id,
            "opponents": [opponent.to_payload() for opponent in self.opponents],
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MatchSetRevision":
        opponents = payload.get("opponents") or ()
        if not isinstance(opponents, Sequence) or isinstance(opponents, str | bytes):
            raise PolicySetError("match-set payload opponents must be a list")
        return cls(
            match_set_revision_id=payload.get("match_set_revision_id", ""),
            match_set_id=payload.get("match_set_id", ""),
            run_id=payload.get("run_id", ""),
            policy_set_revision_id=payload.get("policy_set_revision_id", ""),
            opponents=tuple(OpponentBinding.from_payload(item) for item in opponents),
            created_at=payload.get("created_at", ""),
            schema_version=payload.get("schema_version", MATCH_SET_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class ComponentSaveAttempt:
    """The outcome of materializing one parameter group's artifacts."""

    parameter_group_id: str
    record: CheckpointRecord | None = None
    error: str | None = None
    packed_group_ids: tuple[str, ...] = ()
    provider_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_group_id", _require_text(self.parameter_group_id, "parameter_group_id")
        )
        if (self.record is None) == (self.error is None):
            raise PolicySetError(
                "a component save attempt is either a record or an error, never both or neither"
            )
        if self.record is not None and self.record.parameter_group_id != self.parameter_group_id:
            raise PolicySetError(
                f"save attempt for {self.parameter_group_id} carries a record for "
                f"{self.record.parameter_group_id}"
            )

    @property
    def succeeded(self) -> bool:
        return self.record is not None


@dataclass(frozen=True, slots=True)
class AtomicPublication:
    """What one publication attempt left behind, successful or not."""

    policy_set_revision_id: str | None
    active_policy_set_revision_id: str | None
    published_checkpoint_ids: tuple[str, ...] = ()
    staged_checkpoint_ids: tuple[str, ...] = ()
    orphaned_checkpoint_ids: tuple[str, ...] = ()
    superseded_checkpoint_ids: tuple[str, ...] = ()
    failed_parameter_groups: tuple[str, ...] = ()
    orphan_retention: str = ORPHAN_RETENTION

    @property
    def published(self) -> bool:
        return self.policy_set_revision_id is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy_set_revision_id": self.policy_set_revision_id,
            "active_policy_set_revision_id": self.active_policy_set_revision_id,
            "published_checkpoint_ids": list(self.published_checkpoint_ids),
            "staged_checkpoint_ids": list(self.staged_checkpoint_ids),
            "orphaned_checkpoint_ids": list(self.orphaned_checkpoint_ids),
            "superseded_checkpoint_ids": list(self.superseded_checkpoint_ids),
            "failed_parameter_groups": list(self.failed_parameter_groups),
            "orphan_retention": self.orphan_retention,
        }


@dataclass(frozen=True, slots=True)
class HealthCheckRequest:
    """What a readiness health check is given. No provider client leaks in."""

    revision_id: str
    checkpoint_id: str
    policy_type_id: str
    parameter_group_id: str
    sampler_weights: SamplerWeightsRef
    compatibility: CheckpointCompatibility


@dataclass(frozen=True, slots=True)
class HealthCheckOutcome:
    revision_id: str
    checkpoint_id: str
    healthy: bool
    detail: str | None = None


HealthCheck = Callable[[HealthCheckRequest], bool]


class PolicySetPublisher:
    """Atomic publication and the load/ready/active/retire lifecycle.

    The health check is injected: this module never talks to a provider, and a
    revision is never marked ready on the strength of a path existing.
    """

    def __init__(
        self,
        catalog: CheckpointCatalog,
        *,
        health_check: HealthCheck | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self._catalog = catalog
        self._health_check = health_check
        self._clock = clock

    @property
    def catalog(self) -> CheckpointCatalog:
        return self._catalog

    # -------------------------------------------------------------- staging

    def stage_component(
        self,
        record: CheckpointRecord,
        *,
        packed_group_ids: Sequence[str] = (),
        provider_request_ids: Sequence[str] = (),
    ) -> CheckpointRecord:
        """Catalogue a materialized component before any publication decision."""

        if record.publication_status != "staged":
            raise PolicySetError(
                f"component {record.checkpoint_id} must be registered staged, "
                f"not {record.publication_status!r}"
            )
        with self._catalog.transaction():
            self._catalog.register_checkpoint(record)
            self._catalog.record_save_attempt(
                SaveAttempt(
                    run_id=record.run_id,
                    update_id=record.update_id,
                    parameter_group_id=record.parameter_group_id,
                    outcome="succeeded",
                    checkpoint_id=record.checkpoint_id,
                    packed_group_ids=tuple(packed_group_ids) or record.training_evidence.groups,
                    provider_request_ids=tuple(provider_request_ids) or record.train_call_ids,
                )
            )
        return record

    def record_failed_save(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_group_id: str,
        error: str,
        packed_group_ids: Sequence[str] = (),
        provider_request_ids: Sequence[str] = (),
    ) -> SaveAttempt:
        """A save that did not materialize is still evidence about the run."""

        return self._catalog.record_save_attempt(
            SaveAttempt(
                run_id=run_id,
                update_id=update_id,
                parameter_group_id=parameter_group_id,
                outcome="failed",
                error=_require_text(error, "error"),
                packed_group_ids=tuple(packed_group_ids),
                provider_request_ids=tuple(provider_request_ids),
            )
        )

    # ----------------------------------------------------------- publishing

    def publish_round(
        self,
        revision: PolicySetRevision,
        attempts: Sequence[ComponentSaveAttempt],
    ) -> AtomicPublication:
        """One published round: every declared component, or none of them.

        Called once per published update, not once per packed training group.
        """

        declared = {component.parameter_group_id: component for component in revision.components}
        seen = [attempt.parameter_group_id for attempt in attempts]
        if len(set(seen)) != len(seen):
            raise PolicySetError("a published round saves each parameter group at most once")
        if set(seen) != set(declared):
            raise PolicySetError(
                "save attempts do not cover the declared components: "
                f"attempted={sorted(set(seen))} declared={sorted(declared)}"
            )
        staged: list[str] = []
        failed: list[str] = []
        for attempt in attempts:
            component = declared[attempt.parameter_group_id]
            if attempt.succeeded:
                record = attempt.record
                assert record is not None
                if record.checkpoint_id != component.checkpoint_id:
                    raise PolicySetError(
                        f"component {component.parameter_group_id} declares checkpoint "
                        f"{component.checkpoint_id} but the save produced "
                        f"{record.checkpoint_id}"
                    )
                self.stage_component(
                    record,
                    packed_group_ids=attempt.packed_group_ids,
                    provider_request_ids=attempt.provider_request_ids,
                )
                staged.append(record.checkpoint_id)
            else:
                self.record_failed_save(
                    run_id=revision.run_id,
                    update_id=revision.update_id,
                    parameter_group_id=attempt.parameter_group_id,
                    error=str(attempt.error),
                    packed_group_ids=attempt.packed_group_ids,
                    provider_request_ids=attempt.provider_request_ids,
                )
                failed.append(attempt.parameter_group_id)
        if failed:
            orphaned: list[str] = []
            for checkpoint_id in staged:
                self._catalog.record_publication(checkpoint_id, "orphaned", reason=ORPHAN_REASON)
                orphaned.append(checkpoint_id)
            outcome = AtomicPublication(
                policy_set_revision_id=None,
                active_policy_set_revision_id=self._catalog.active_revision_id(
                    revision.policy_set_id
                ),
                orphaned_checkpoint_ids=tuple(orphaned),
                failed_parameter_groups=tuple(sorted(failed)),
            )
            raise PartialPublicationError(
                "policy-set publication is atomic: parameter groups "
                f"{sorted(failed)} did not save, so revision "
                f"{revision.policy_set_revision_id} was not published",
                outcome,
            )
        return self.publish(revision)

    def publish(self, revision: PolicySetRevision) -> AtomicPublication:
        """Publish an already-materialized set. Fails closed on any absence."""

        missing = tuple(
            component.checkpoint_id
            for component in revision.components
            if not self._catalog.has_checkpoint(component.checkpoint_id)
        )
        if missing:
            raise MissingComponentError(
                f"cannot publish {revision.policy_set_revision_id}: component checkpoints "
                f"{list(missing)} are absent from the catalog"
            )
        for component in revision.components:
            self._assert_component_consistent(revision, component)
        prior = self._catalog.active_revision_id(revision.policy_set_id)
        published: list[str] = []
        superseded: list[str] = []
        with self._catalog.transaction():
            self._catalog.put_revision(
                revision_id=revision.policy_set_revision_id,
                revision_kind="policy_set",
                family_id=revision.policy_set_id,
                payload=revision.to_payload(),
                run_id=revision.run_id,
                update_id=revision.update_id,
                created_at=revision.created_at,
            )
            for component in revision.components:
                status = self._catalog.publication_status(component.checkpoint_id)
                if status in {"orphaned", "superseded"}:
                    raise PolicySetError(
                        f"component {component.checkpoint_id} is {status} and cannot be published"
                    )
                if status == "staged":
                    self._catalog.record_publication(
                        component.checkpoint_id,
                        "published",
                        reason=f"policy_set:{revision.policy_set_revision_id}",
                    )
                self._catalog.record_policy_set_membership(
                    component.checkpoint_id, revision.policy_set_revision_id
                )
                self._catalog.record_lineage_edge(
                    LineageEdge(
                        child_checkpoint_id=component.checkpoint_id,
                        relation="policy_set_component",
                        revision_id=revision.policy_set_revision_id,
                        run_id=revision.run_id,
                        update_id=revision.update_id,
                        parameter_group_id=component.parameter_group_id,
                    )
                )
                published.append(component.checkpoint_id)
            if prior is not None and prior != revision.policy_set_revision_id:
                self._catalog.record_revision_transition(
                    prior, "superseded", detail=revision.policy_set_revision_id
                )
                for checkpoint_id in self._catalog.policy_set_members(prior):
                    if checkpoint_id in published:
                        continue
                    if self._catalog.publication_status(checkpoint_id) != "published":
                        continue
                    self._catalog.record_publication(
                        checkpoint_id,
                        "superseded",
                        reason=f"policy_set:{revision.policy_set_revision_id}",
                    )
                    superseded.append(checkpoint_id)
        return AtomicPublication(
            policy_set_revision_id=revision.policy_set_revision_id,
            active_policy_set_revision_id=self._catalog.active_revision_id(revision.policy_set_id),
            published_checkpoint_ids=tuple(published),
            superseded_checkpoint_ids=tuple(superseded),
        )

    def publish_match_set(self, revision: MatchSetRevision) -> MatchSetRevision:
        """Pin a whole match. Every pinned identity must already be catalogued."""

        if self._catalog.revision_kind(revision.policy_set_revision_id) != "policy_set":
            raise MissingComponentError(
                f"match set {revision.match_set_revision_id} names trainee policy set "
                f"{revision.policy_set_revision_id}, which is not a published policy set"
            )
        missing = tuple(
            checkpoint_id
            for checkpoint_id in revision.pinned_checkpoint_ids
            if not self._catalog.has_checkpoint(checkpoint_id)
        )
        if missing:
            raise MissingComponentError(
                f"cannot publish {revision.match_set_revision_id}: pinned opponent checkpoints "
                f"{list(missing)} are absent from the catalog"
            )
        trainee = self.policy_set(revision.policy_set_revision_id)
        with self._catalog.transaction():
            self._catalog.put_revision(
                revision_id=revision.match_set_revision_id,
                revision_kind="match_set",
                family_id=revision.match_set_id,
                payload=revision.to_payload(),
                run_id=revision.run_id,
                created_at=revision.created_at,
            )
            for component in trainee.components:
                self._catalog.record_lineage_edge(
                    LineageEdge(
                        child_checkpoint_id=component.checkpoint_id,
                        relation="match_set_trainee",
                        revision_id=revision.match_set_revision_id,
                        run_id=revision.run_id,
                        parameter_group_id=component.parameter_group_id,
                    )
                )
            for opponent in revision.opponents:
                if not opponent.is_pinned_checkpoint:
                    continue
                self._catalog.record_lineage_edge(
                    LineageEdge(
                        child_checkpoint_id=opponent.identity,
                        relation="match_set_opponent",
                        revision_id=revision.match_set_revision_id,
                        run_id=revision.run_id,
                    )
                )
        return revision

    # ------------------------------------------------------------ lifecycle

    def policy_set(self, policy_set_revision_id: str) -> PolicySetRevision:
        row = self._catalog.get_revision(policy_set_revision_id)
        if row.revision_kind != "policy_set":
            raise PolicySetError(
                f"revision {policy_set_revision_id} is a {row.revision_kind}, not a policy set"
            )
        return PolicySetRevision.from_payload(row.payload)

    def match_set(self, match_set_revision_id: str) -> MatchSetRevision:
        row = self._catalog.get_revision(match_set_revision_id)
        if row.revision_kind != "match_set":
            raise PolicySetError(
                f"revision {match_set_revision_id} is a {row.revision_kind}, not a match set"
            )
        return MatchSetRevision.from_payload(row.payload)

    def mark_loaded(self, revision_id: str, *, detail: str | None = None) -> None:
        """The sampler artifacts are resident. Not yet usable for rollouts."""

        if self._catalog.has_transition(revision_id, "retire"):
            raise RetirementError(f"revision {revision_id} is retired and cannot be reloaded")
        self._catalog.record_revision_transition(revision_id, "load", detail=detail)

    def mark_ready(
        self, revision_id: str, *, health_check: HealthCheck | None = None
    ) -> tuple[HealthCheckOutcome, ...]:
        """Ready only after every sampler artifact is materialized and healthy."""

        check = health_check or self._health_check
        if check is None:
            raise ReadinessError(
                f"revision {revision_id} cannot be marked ready without a health check"
            )
        if not self._catalog.has_transition(revision_id, "load"):
            raise ReadinessError(f"revision {revision_id} must be loaded before it is ready")
        if self._catalog.has_transition(revision_id, "retire"):
            raise RetirementError(f"revision {revision_id} is retired")
        outcomes: list[HealthCheckOutcome] = []
        for checkpoint_id, policy_type_id, parameter_group_id in self._sampled_components(
            revision_id
        ):
            record = self._catalog.get_checkpoint(checkpoint_id)
            sampler = record.sampler_weights
            request = HealthCheckRequest(
                revision_id=revision_id,
                checkpoint_id=checkpoint_id,
                policy_type_id=policy_type_id,
                parameter_group_id=parameter_group_id,
                sampler_weights=sampler,
                compatibility=record.compatibility,
            )
            try:
                healthy = bool(check(request))
                detail = None
            except Exception as error:  # noqa: BLE001 - a raising probe is an unhealthy probe
                healthy = False
                detail = f"{type(error).__name__}: {error}"
            outcomes.append(
                HealthCheckOutcome(
                    revision_id=revision_id,
                    checkpoint_id=checkpoint_id,
                    healthy=healthy,
                    detail=detail,
                )
            )
            if not healthy:
                self._catalog.record_revision_transition(
                    revision_id,
                    "health_check_failed",
                    detail=f"{checkpoint_id}:{detail or 'unhealthy'}",
                )
                raise HealthCheckError(
                    f"revision {revision_id} component {checkpoint_id} failed its health check"
                    + (f": {detail}" if detail else "")
                )
        self._catalog.record_revision_transition(revision_id, "ready")
        return tuple(outcomes)

    def is_ready(self, revision_id: str) -> bool:
        return self._catalog.has_transition(revision_id, "ready") and not self.is_retired(
            revision_id
        )

    def is_retired(self, revision_id: str) -> bool:
        return self._catalog.has_transition(revision_id, "retire")

    def open_attempt(self, revision_id: str, attempt_id: str) -> None:
        """An attempt starts sampling from this revision. It now holds it open."""

        if self.is_retired(revision_id):
            raise RetirementError(
                f"revision {revision_id} is retired; an attempt cannot bind to it"
            )
        if not self._catalog.has_transition(revision_id, "ready"):
            raise ReadinessError(
                f"revision {revision_id} is not ready; no attempt may sample from it"
            )
        self._catalog.record_revision_transition(
            revision_id, "attempt_open", attempt_id=_require_text(attempt_id, "attempt_id")
        )

    def close_attempt(self, revision_id: str, attempt_id: str) -> None:
        attempt_id = _require_text(attempt_id, "attempt_id")
        if attempt_id not in self._catalog.active_attempts(revision_id):
            raise PolicySetError(
                f"attempt {attempt_id} is not sampling from revision {revision_id}"
            )
        self._catalog.record_revision_transition(
            revision_id, "attempt_close", attempt_id=attempt_id
        )

    def active_attempt_count(self, revision_id: str) -> int:
        return len(self._catalog.active_attempts(revision_id))

    def retire(self, revision_id: str, *, reason: str | None = None) -> None:
        """Retire only at zero active attempts. Otherwise evidence is destroyed."""

        if self.is_retired(revision_id):
            return
        active = self._catalog.active_attempts(revision_id)
        if active:
            raise RetirementError(
                f"revision {revision_id} still has {len(active)} active attempt(s) "
                f"{list(active)}; unloading it would make their evidence unusable"
            )
        self._catalog.record_revision_transition(revision_id, "retire", detail=reason)

    # -------------------------------------------------------------- private

    def _sampled_components(self, revision_id: str) -> tuple[tuple[str, str, str], ...]:
        kind = self._catalog.revision_kind(revision_id)
        if kind == "policy_set":
            revision = self.policy_set(revision_id)
            return tuple(
                (
                    component.checkpoint_id,
                    component.policy_type_id,
                    component.parameter_group_id,
                )
                for component in revision.components
            )
        if kind == "match_set":
            match = self.match_set(revision_id)
            trainee = self.policy_set(match.policy_set_revision_id)
            components = [
                (
                    component.checkpoint_id,
                    component.policy_type_id,
                    component.parameter_group_id,
                )
                for component in trainee.components
            ]
            for opponent in match.opponents:
                if not opponent.is_pinned_checkpoint:
                    continue
                record = self._catalog.get_checkpoint(opponent.identity)
                components.append(
                    (
                        opponent.identity,
                        record.policy_type_ids[0],
                        record.parameter_group_id,
                    )
                )
            return tuple(components)
        raise PolicySetError(f"revision {revision_id} is absent from the catalog")

    def _assert_component_consistent(
        self, revision: PolicySetRevision, component: PolicySetComponent
    ) -> None:
        record = self._catalog.get_checkpoint(component.checkpoint_id)
        if record.parameter_group_id != component.parameter_group_id:
            raise PolicySetError(
                f"component {component.checkpoint_id} belongs to parameter group "
                f"{record.parameter_group_id}, declared as {component.parameter_group_id}"
            )
        if component.policy_type_id not in record.policy_type_ids:
            raise PolicySetError(
                f"component {component.checkpoint_id} does not serve policy type "
                f"{component.policy_type_id!r}"
            )
        if record.policy_revision_id != component.policy_revision_id:
            raise PolicySetError(
                f"component {component.checkpoint_id} is policy revision "
                f"{record.policy_revision_id}, declared as {component.policy_revision_id}"
            )
        if record.artifacts.sampler_weights is None:
            raise PolicySetError(
                f"component {component.checkpoint_id} has no sampler artifact and cannot be "
                f"published into {revision.policy_set_revision_id}"
            )


__all__ = [
    "AtomicPublication",
    "ComponentSaveAttempt",
    "HealthCheck",
    "HealthCheckError",
    "HealthCheckOutcome",
    "HealthCheckRequest",
    "MATCH_SET_SCHEMA_VERSION",
    "MatchSetRevision",
    "MissingComponentError",
    "OPPONENT_BINDING_KINDS",
    "ORPHAN_REASON",
    "ORPHAN_RETENTION",
    "OpponentBinding",
    "POLICY_SET_SCHEMA_VERSION",
    "PartialPublicationError",
    "PolicySetComponent",
    "PolicySetError",
    "PolicySetPublisher",
    "PolicySetRevision",
    "ReadinessError",
    "RetirementError",
]
