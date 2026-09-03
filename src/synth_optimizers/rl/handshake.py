"""The two-sided readiness agreement, and the gate every attempt passes.

Discovery says what a container can do; the handshake says whether it can honor
this run. Nothing is negotiated after training starts, and no session, binding,
or paid request may precede acceptance.

A rejected mandatory clause stops the run before spend. A rejected or
unsupported optional clause records the fallback the run will use. A degraded
clause is acceptable only if the executor can satisfy it by lowering its own run
plan, and the lowered plan is re-handshaked rather than assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from ..contracts.rl_clauses import (
    HANDSHAKE_SCHEMA_VERSION,
    MANDATORY_CLAUSES,
    OPTIONAL_CLAUSES,
)
from ..contracts.rl_identity import PARTIAL_ROSTER_DISPOSITIONS, Horizon
from ..contracts.rl_records import RendererProfile, digest
from .capabilities import (
    CapabilityDocument,
    CapabilityDriftError,
    ClauseResult,
    ExecutorRequirements,
    merge_clause_results,
    rejected_mandatory,
)
from .contract import ContainerContract

OUTCOMES = ("admissible", "renegotiate", "refused")

# The only degraded clauses the executor can answer by lowering its own plan.
# A degraded mandatory clause outside this set stops the run: the executor has
# nothing to give up, so "degraded" would just mean "silently wrong".
LOWERABLE_CLAUSES: frozenset[str] = frozenset(
    {"lifecycle.concurrency", "lifecycle.lease_renewal"}
)


class HandshakeError(ValueError):
    """The readiness agreement is absent, stale, or not what was agreed."""


class ClauseRejected(HandshakeError):
    """A mandatory clause failed. Stop before session creation, name the clauses."""

    def __init__(self, results: Sequence[ClauseResult]) -> None:
        self.results = tuple(results)
        detail = "; ".join(
            f"{item.clause_id}={item.verdict}: {item.reason}" for item in self.results
        )
        super().__init__(f"handshake rejected {len(self.results)} mandatory clause(s): {detail}")

    @property
    def clause_ids(self) -> tuple[str, ...]:
        return tuple(item.clause_id for item in self.results)


class PlanNotLowerable(HandshakeError):
    """A degraded clause asked for a lower plan that the executor cannot form."""


class RenegotiationRequired(HandshakeError):
    """A lowered plan exists but has not been handshaked yet."""


class HandshakeExpired(HandshakeError):
    """The agreement's expiry has passed. Renew or re-handshake."""


class HandshakeRevoked(HandshakeError):
    """The container revoked this agreement. Drain, then re-handshake."""


class AgreementMismatch(HandshakeError):
    """An attempt or a resume names an agreement digest that is not the one agreed."""


class UnknownHandshake(HandshakeError):
    """An attempt names a handshake this executor never admitted."""


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_rfc3339(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise HandshakeError(f"{field_name} is required as an RFC3339 timestamp")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HandshakeError(f"{field_name} is not an RFC3339 timestamp: {text!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def format_rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class OptimizerIdentity:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    provider: str
    model_id: str
    transport: str


@dataclass(frozen=True, slots=True)
class TopologyExpectation:
    expected_topology_id: str
    trainable_teams: tuple[str, ...]
    partial_roster: str = "refuse"

    def __post_init__(self) -> None:
        if self.partial_roster not in PARTIAL_ROSTER_DISPOSITIONS:
            raise HandshakeError(f"unknown partial roster disposition {self.partial_roster!r}")


@dataclass(frozen=True, slots=True)
class RunPlan:
    """The dimensions the executor may lower, and nothing else."""

    group_size: int
    groups_per_step: int
    max_execution_slots: int
    maximum_policy_lag: int
    target_train_updates: int
    expected_horizon_seconds: float

    def __post_init__(self) -> None:
        if min(self.group_size, self.groups_per_step, self.max_execution_slots) < 1:
            raise HandshakeError("run plan sizes must be positive")
        if self.maximum_policy_lag < 0 or self.target_train_updates < 1:
            raise HandshakeError("run plan lag and update count are out of range")
        if self.expected_horizon_seconds <= 0:
            raise HandshakeError("run plan horizon must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "groups_per_step": self.groups_per_step,
            "max_execution_slots": self.max_execution_slots,
            "maximum_policy_lag": self.maximum_policy_lag,
            "target_train_updates": self.target_train_updates,
            "expected_horizon_seconds": self.expected_horizon_seconds,
        }

    def lowered(
        self,
        obligations: "Obligations",
        *,
        degraded_clauses: Sequence[str] = (),
    ) -> "RunPlan":
        """Lower only the dimensions the degraded clauses actually constrain."""

        clauses = set(degraded_clauses) or set(LOWERABLE_CLAUSES)
        group_size = self.group_size
        slots = self.max_execution_slots
        groups = self.groups_per_step
        horizon = self.expected_horizon_seconds
        if "lifecycle.concurrency" in clauses:
            ceiling = max(1, obligations.max_concurrency)
            slots = min(slots, ceiling)
            group_size = min(group_size, ceiling)
            groups = max(1, min(groups, ceiling // group_size))
        if "lifecycle.lease_renewal" in clauses and obligations.horizon is not None:
            horizon = min(horizon, float(obligations.horizon.value))
        lowered = RunPlan(
            group_size=group_size,
            groups_per_step=groups,
            max_execution_slots=slots,
            maximum_policy_lag=self.maximum_policy_lag,
            target_train_updates=self.target_train_updates,
            expected_horizon_seconds=horizon,
        )
        if lowered == self:
            raise PlanNotLowerable(
                "the container degraded "
                f"{sorted(clauses)} but the run plan is already at that bound"
            )
        return lowered


@dataclass(frozen=True, slots=True)
class TasksetRequest:
    taskset_id: str
    split: str
    task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise HandshakeError("handshake taskset must name at least one task id")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise HandshakeError("handshake taskset repeats a task id")


@dataclass(frozen=True, slots=True)
class ClockStamp:
    executor_time: str
    monotonic_source: str = "CLOCK_MONOTONIC"


@dataclass(frozen=True, slots=True)
class HandshakeRequest:
    """The executor's requirement document, sent in full, before any spend."""

    run_id: str
    optimizer: OptimizerIdentity
    policy: PolicyRequest
    renderer_profile: RendererProfile
    requirements: tuple[str, ...]
    topology: TopologyExpectation
    run_plan: RunPlan
    taskset: TasksetRequest
    clock: ClockStamp
    attempt: int = 1
    schema_version: str = HANDSHAKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        unknown = tuple(
            clause
            for clause in self.requirements
            if clause not in set(MANDATORY_CLAUSES) | set(OPTIONAL_CLAUSES)
        )
        if unknown:
            raise HandshakeError(f"requirement document names unknown clauses: {unknown}")
        missing = tuple(
            clause for clause in MANDATORY_CLAUSES if clause not in set(self.requirements)
        )
        if missing:
            raise HandshakeError(
                f"requirement document omits mandatory clauses: {missing}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "optimizer": {"name": self.optimizer.name, "version": self.optimizer.version},
            "policy": {
                "provider": self.policy.provider,
                "model_id": self.policy.model_id,
                "transport": self.policy.transport,
            },
            "renderer_profile": {
                "profile_id": self.renderer_profile.profile_id,
                "config_digest": self.renderer_profile.config_digest,
                "fingerprint": self.renderer_profile.fingerprint,
            },
            "requirements": list(self.requirements),
            "topology": {
                "expected_topology_id": self.topology.expected_topology_id,
                "trainable_teams": list(self.topology.trainable_teams),
                "partial_roster": self.topology.partial_roster,
            },
            "run_plan": self.run_plan.to_payload(),
            "taskset": {
                "taskset_id": self.taskset.taskset_id,
                "split": self.taskset.split,
                "task_ids": list(self.taskset.task_ids),
            },
            "clock": {
                "executor_time": self.clock.executor_time,
                "monotonic_source": self.clock.monotonic_source,
            },
        }

    @property
    def request_digest(self) -> str:
        return "sha256:" + digest(self.to_payload())

    def lower_run_plan(
        self,
        obligations: "Obligations",
        *,
        degraded_clauses: Sequence[str] = (),
        executor_time: str | None = None,
    ) -> "HandshakeRequest":
        """The next requirement document. A lowered plan is re-handshaked."""

        plan = self.run_plan.lowered(obligations, degraded_clauses=degraded_clauses)
        clock = ClockStamp(
            executor_time=executor_time or format_rfc3339(utc_now()),
            monotonic_source=self.clock.monotonic_source,
        )
        return replace(self, run_plan=plan, clock=clock, attempt=self.attempt + 1)


def build_request(
    *,
    run_id: str,
    optimizer: OptimizerIdentity,
    policy: PolicyRequest,
    requirements: ExecutorRequirements,
    topology: TopologyExpectation,
    run_plan: RunPlan,
    task_ids: Sequence[str],
    taskset_id: str,
    now: datetime | None = None,
) -> HandshakeRequest:
    """Build the requirement document from the executor's own requirement set."""

    return HandshakeRequest(
        run_id=run_id,
        optimizer=optimizer,
        policy=policy,
        renderer_profile=requirements.renderer_profile,
        requirements=requirements.declared_clauses(),
        topology=topology,
        run_plan=run_plan,
        taskset=TasksetRequest(
            taskset_id=taskset_id,
            split=requirements.split,
            task_ids=tuple(task_ids),
        ),
        clock=ClockStamp(executor_time=format_rfc3339(now or utc_now())),
    )


@dataclass(frozen=True, slots=True)
class TaskResolution:
    task_id: str
    content_digest: str
    topology_ref: str

    def __post_init__(self) -> None:
        if not self.content_digest.strip():
            raise HandshakeError(f"task {self.task_id} resolved without a content digest")


@dataclass(frozen=True, slots=True)
class Obligations:
    """What the container commits to for this run, and only this run."""

    max_concurrency: int
    lease_ttl_seconds: float
    deferred_scoring: bool
    quiescence: bool
    settlement_window_seconds: float = 0.0
    horizon: Horizon | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "max_concurrency": self.max_concurrency,
            "lease_ttl_seconds": self.lease_ttl_seconds,
            "deferred_scoring": self.deferred_scoring,
            "quiescence": self.quiescence,
            "settlement_window_seconds": self.settlement_window_seconds,
        }
        if self.horizon is not None:
            payload["horizon"] = {
                "horizon_kind": self.horizon.horizon_kind,
                "value_seconds": self.horizon.value,
                "time_dilation": self.horizon.time_dilation,
                "grace_seconds": self.horizon.grace_seconds,
            }
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "Obligations":
        if not isinstance(payload, Mapping):
            raise HandshakeError("handshake obligations must be an object")
        horizon_raw = payload.get("horizon")
        horizon: Horizon | None = None
        if isinstance(horizon_raw, Mapping):
            horizon = Horizon(
                horizon_kind=str(horizon_raw.get("horizon_kind", "")),
                value=float(horizon_raw.get("value_seconds", 0.0) or 0.0),
                time_dilation=float(horizon_raw.get("time_dilation", 1.0) or 1.0),
                grace_seconds=float(horizon_raw.get("grace_seconds", 0.0) or 0.0),
            )
        concurrency = payload.get("max_concurrency")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise HandshakeError("obligation max_concurrency must be a positive integer")
        return cls(
            max_concurrency=concurrency,
            lease_ttl_seconds=float(payload.get("lease_ttl_seconds", 0.0) or 0.0),
            deferred_scoring=bool(payload.get("deferred_scoring", False)),
            quiescence=bool(payload.get("quiescence", False)),
            settlement_window_seconds=float(payload.get("settlement_window_seconds", 0.0) or 0.0),
            horizon=horizon,
        )


@dataclass(frozen=True, slots=True)
class HandshakeVerdict:
    """The container's answer: per clause, with obligations and an expiry."""

    handshake_id: str
    accepted: bool
    clauses: tuple[ClauseResult, ...]
    obligations: Obligations
    taskset_resolution: tuple[TaskResolution, ...]
    capability_hash: str
    agreement_digest: str
    expires_at: datetime
    container_time: datetime
    measured_skew_seconds: float
    schema_version: str = HANDSHAKE_SCHEMA_VERSION
    raw: Mapping[str, Any] = field(default_factory=dict)

    def clause(self, clause_id: str) -> ClauseResult | None:
        for result in self.clauses:
            if result.clause_id == clause_id:
                return result
        return None

    @classmethod
    def from_payload(cls, payload: Any) -> "HandshakeVerdict":
        if not isinstance(payload, Mapping):
            raise HandshakeError("handshake verdict must be an object")
        if payload.get("schema_version") != HANDSHAKE_SCHEMA_VERSION:
            raise HandshakeError(
                f"handshake schema {payload.get('schema_version')!r} is unsupported; "
                f"expected {HANDSHAKE_SCHEMA_VERSION}"
            )
        handshake_id = payload.get("handshake_id")
        if not isinstance(handshake_id, str) or not handshake_id.strip():
            raise HandshakeError("handshake verdict carries no handshake_id")
        clauses_raw = payload.get("clauses")
        if not isinstance(clauses_raw, Sequence) or isinstance(clauses_raw, str | bytes):
            raise HandshakeError("handshake verdict clauses must be a list")
        clauses = tuple(
            ClauseResult(
                clause_id=str(item.get("clause_id", "")),
                verdict=str(item.get("verdict", "")),
                reason=str(item.get("reason") or item.get("note") or ""),
                source="container",
            )
            for item in clauses_raw
            if isinstance(item, Mapping)
        )
        seen = [result.clause_id for result in clauses]
        if len(set(seen)) != len(seen):
            raise HandshakeError("handshake verdict answers a clause twice")
        resolution_raw = payload.get("taskset_resolution") or ()
        if not isinstance(resolution_raw, Sequence) or isinstance(resolution_raw, str | bytes):
            raise HandshakeError("handshake taskset_resolution must be a list")
        clock_raw = payload.get("clock")
        clock = clock_raw if isinstance(clock_raw, Mapping) else {}
        capability_hash = payload.get("capability_hash")
        agreement_digest = payload.get("agreement_digest")
        if not isinstance(capability_hash, str) or not capability_hash.strip():
            raise HandshakeError("handshake verdict carries no capability_hash")
        if not isinstance(agreement_digest, str) or not agreement_digest.strip():
            raise HandshakeError("handshake verdict carries no agreement_digest")
        return cls(
            handshake_id=handshake_id.strip(),
            accepted=bool(payload.get("accepted", False)),
            clauses=clauses,
            obligations=Obligations.from_payload(payload.get("obligations")),
            taskset_resolution=tuple(
                TaskResolution(
                    task_id=str(item.get("task_id", "")),
                    content_digest=str(item.get("content_digest", "")),
                    topology_ref=str(item.get("topology_ref", "")),
                )
                for item in resolution_raw
                if isinstance(item, Mapping)
            ),
            capability_hash=capability_hash.strip(),
            agreement_digest=agreement_digest.strip(),
            expires_at=parse_rfc3339(payload.get("expires_at"), field_name="expires_at"),
            container_time=parse_rfc3339(
                clock.get("container_time"), field_name="clock.container_time"
            ),
            measured_skew_seconds=float(clock.get("measured_skew_seconds", 0.0) or 0.0),
            raw=dict(payload),
        )


def compute_agreement_digest(
    request: HandshakeRequest,
    *,
    handshake_id: str,
    capability_hash: str,
    renderer_fingerprint: str,
    taskset_resolution: Sequence[TaskResolution],
    obligations: Obligations,
    clauses: Sequence[ClauseResult],
) -> str:
    """Binds both documents, the capability hash, the renderer, tasks, obligations."""

    return "sha256:" + digest(
        {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "handshake_id": handshake_id,
            "request": request.to_payload(),
            "capability_hash": capability_hash,
            "renderer_fingerprint": renderer_fingerprint,
            "task_digests": sorted(
                [item.task_id, item.content_digest, item.topology_ref]
                for item in taskset_resolution
            ),
            "obligations": obligations.to_payload(),
            "clauses": sorted(
                [item.clause_id, item.verdict]
                for item in clauses
                if item.source == "container"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Fallback:
    """What the run will do instead, recorded because it changes the results."""

    clause_id: str
    verdict: str
    fallback: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Agreement:
    """An admitted handshake. Every attempt is gated against this object."""

    handshake_id: str
    run_id: str
    agreement_digest: str
    capability_hash: str
    contract_hash: str
    renderer_fingerprint: str
    request: HandshakeRequest
    obligations: Obligations
    taskset_resolution: tuple[TaskResolution, ...]
    clauses: tuple[ClauseResult, ...]
    fallbacks: tuple[Fallback, ...]
    expires_at: datetime
    admitted_at: datetime

    def task_digest(self, task_id: str) -> str:
        for item in self.taskset_resolution:
            if item.task_id == task_id:
                return item.content_digest
        raise AgreementMismatch(f"task {task_id!r} is not part of this agreement")

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "handshake_id": self.handshake_id,
            "run_id": self.run_id,
            "agreement_digest": self.agreement_digest,
            "capability_hash": self.capability_hash,
            "container_contract_hash": self.contract_hash,
            "renderer_fingerprint": self.renderer_fingerprint,
            "request": self.request.to_payload(),
            "obligations": self.obligations.to_payload(),
            "clauses": [item.to_payload() for item in self.clauses],
            "fallbacks": [
                {
                    "clause_id": item.clause_id,
                    "verdict": item.verdict,
                    "fallback": item.fallback,
                    "reason": item.reason,
                }
                for item in self.fallbacks
            ],
            "taskset_resolution": [
                {
                    "task_id": item.task_id,
                    "content_digest": item.content_digest,
                    "topology_ref": item.topology_ref,
                }
                for item in self.taskset_resolution
            ],
            "expires_at": format_rfc3339(self.expires_at),
            "admitted_at": format_rfc3339(self.admitted_at),
        }


@dataclass(frozen=True, slots=True)
class HandshakeDecision:
    """What the executor may do next, and nothing more."""

    outcome: str
    clauses: tuple[ClauseResult, ...]
    fallbacks: tuple[Fallback, ...] = ()
    agreement: Agreement | None = None
    next_request: HandshakeRequest | None = None
    degraded_clauses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise HandshakeError(f"unknown handshake outcome {self.outcome!r}")

    @property
    def admissible(self) -> bool:
        return self.outcome == "admissible"


def _skew_clause(
    capability: CapabilityDocument,
    verdict: HandshakeVerdict,
    tolerance: float,
) -> ClauseResult | None:
    # The horizon is the instant the reward is read, so for a wall-clock horizon
    # a skewed clock is a rejected clause rather than a logged warning.
    if capability.horizon.horizon_kind != "wall_clock":
        return None
    skew = abs(float(verdict.measured_skew_seconds))
    if skew <= tolerance:
        return None
    return ClauseResult(
        clause_id="reward.horizon_quiescence",
        verdict="rejected",
        reason=(
            f"measured clock skew {skew}s exceeds the declared tolerance {tolerance}s "
            "for a wall-clock horizon"
        ),
    )


def _resolution_clause(
    request: HandshakeRequest, verdict: HandshakeVerdict
) -> ClauseResult:
    resolved = [item.task_id for item in verdict.taskset_resolution]
    if len(set(resolved)) != len(resolved):
        return ClauseResult(
            clause_id="discovery.task_digests",
            verdict="rejected",
            reason="taskset resolution repeats a task id",
        )
    missing = tuple(task for task in request.taskset.task_ids if task not in set(resolved))
    if missing:
        return ClauseResult(
            clause_id="discovery.task_digests",
            verdict="rejected",
            reason=f"taskset resolution omits requested task ids: {missing}",
        )
    return ClauseResult(clause_id="discovery.task_digests", verdict="accepted")


def _unanswered_clauses(clauses: Sequence[ClauseResult]) -> list[ClauseResult]:
    answered = {item.clause_id for item in clauses}
    return [
        ClauseResult(
            clause_id=clause,
            verdict="rejected",
            reason="container returned no verdict for a mandatory clause",
        )
        for clause in MANDATORY_CLAUSES
        if clause not in answered
    ]


def _fallbacks(clauses: Sequence[ClauseResult], obligations: Obligations) -> tuple[Fallback, ...]:
    fallbacks: list[Fallback] = []
    for result in clauses:
        if result.clause_id not in OPTIONAL_CLAUSES:
            continue
        if result.verdict in {"rejected", "unsupported", "degraded"}:
            fallbacks.append(
                Fallback(
                    clause_id=result.clause_id,
                    verdict=result.verdict,
                    fallback=_FALLBACKS.get(result.clause_id, "clause_unused"),
                    reason=result.reason,
                )
            )
    if not obligations.quiescence:
        fallbacks.append(
            Fallback(
                clause_id="reward.horizon_quiescence",
                verdict="accepted",
                fallback="horizon_clipped_snapshot",
                reason="container attests no quiescence; reward reads a clipped snapshot",
            )
        )
    return tuple(fallbacks)


_FALLBACKS: dict[str, str] = {
    "evidence.tito": "message_in_capture_out",
    "evidence.artifact_reference": "inline_evidence_only",
    "reward.settlement_window": "single_read_at_horizon",
    "lifecycle.pause_resume": "cancel_and_replace",
    "topology.channels": "no_channel_completeness_check",
    "topology.minimum_roster": "declared_partial_roster_disposition",
    "topology.opponent_pinning": "no_pinned_opponent_set",
}


def evaluate_handshake(
    request: HandshakeRequest,
    verdict: HandshakeVerdict,
    *,
    capability: CapabilityDocument,
    contract: ContainerContract,
    executor_clauses: Sequence[ClauseResult] = (),
    skew_tolerance_seconds: float | None = None,
    now: datetime | None = None,
) -> HandshakeDecision:
    """Decide, before any session or paid request, what this run may do."""

    moment = now or utc_now()
    if verdict.capability_hash != capability.content_hash:
        raise CapabilityDriftError(
            "handshake was built on a different capability document: "
            f"{verdict.capability_hash} != {capability.content_hash}"
        )
    if verdict.expires_at <= moment:
        raise HandshakeExpired(
            f"handshake {verdict.handshake_id} expired at {format_rfc3339(verdict.expires_at)}"
        )
    local: list[ClauseResult] = [_resolution_clause(request, verdict)]
    if capability.renderer_profile.fingerprint != request.renderer_profile.fingerprint:
        local.append(
            ClauseResult(
                clause_id="policy.renderer_profile_match",
                verdict="rejected",
                reason=(
                    "renderer profile mismatch: container "
                    f"{capability.renderer_profile.fingerprint} != session "
                    f"{request.renderer_profile.fingerprint}"
                ),
            )
        )
    tolerance = (
        capability.clock_skew_tolerance_seconds
        if skew_tolerance_seconds is None
        else skew_tolerance_seconds
    )
    skew = _skew_clause(capability, verdict, tolerance)
    if skew is not None:
        local.append(skew)
    clauses = merge_clause_results(
        verdict.clauses,
        tuple(executor_clauses),
        tuple(local),
        tuple(_unanswered_clauses(verdict.clauses)),
    )
    blocking = rejected_mandatory(clauses)
    if blocking:
        raise ClauseRejected(blocking)
    degraded = tuple(
        result.clause_id
        for result in clauses
        if result.verdict == "degraded" and result.mandatory
    )
    fallbacks = _fallbacks(clauses, verdict.obligations)
    not_lowerable = tuple(clause for clause in degraded if clause not in LOWERABLE_CLAUSES)
    if not_lowerable:
        raise ClauseRejected(
            tuple(result for result in clauses if result.clause_id in set(not_lowerable))
        )
    if degraded:
        return HandshakeDecision(
            outcome="renegotiate",
            clauses=clauses,
            fallbacks=fallbacks,
            next_request=request.lower_run_plan(
                verdict.obligations,
                degraded_clauses=degraded,
                executor_time=format_rfc3339(moment),
            ),
            degraded_clauses=degraded,
        )
    if not verdict.accepted:
        raise ClauseRejected(
            (
                ClauseResult(
                    clause_id="contract.version",
                    verdict="rejected",
                    reason="container did not accept the handshake and named no failed clause",
                    source="container",
                ),
            )
        )
    expected_digest = compute_agreement_digest(
        request,
        handshake_id=verdict.handshake_id,
        capability_hash=capability.content_hash,
        renderer_fingerprint=capability.renderer_profile.fingerprint,
        taskset_resolution=verdict.taskset_resolution,
        obligations=verdict.obligations,
        # Only the verdict's own clause list: the container cannot know the
        # executor-side results, so the digest must be computable by both sides.
        clauses=verdict.clauses,
    )
    if expected_digest != verdict.agreement_digest:
        raise AgreementMismatch(
            "agreement digest disagreement: container "
            f"{verdict.agreement_digest} != executor {expected_digest}"
        )
    agreement = Agreement(
        handshake_id=verdict.handshake_id,
        run_id=request.run_id,
        agreement_digest=expected_digest,
        capability_hash=capability.content_hash,
        contract_hash=contract.contract_hash,
        renderer_fingerprint=capability.renderer_profile.fingerprint,
        request=request,
        obligations=verdict.obligations,
        taskset_resolution=verdict.taskset_resolution,
        clauses=clauses,
        fallbacks=fallbacks,
        expires_at=verdict.expires_at,
        admitted_at=moment,
    )
    return HandshakeDecision(
        outcome="admissible",
        clauses=clauses,
        fallbacks=fallbacks,
        agreement=agreement,
    )


class HandshakeLedger:
    """Admission, expiry, renewal, and revocation for one run's agreements.

    The records are frozen; only this ledger holds state, and every attempt
    passes ``assert_admissible`` before it is dispatched.
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._clock = clock or utc_now
        self._agreements: dict[str, Agreement] = {}
        self._revoked: dict[str, str] = {}

    @property
    def agreements(self) -> Mapping[str, Agreement]:
        return dict(self._agreements)

    def admit(self, decision: HandshakeDecision) -> Agreement:
        if decision.outcome == "renegotiate":
            raise RenegotiationRequired(
                "the container degraded "
                f"{list(decision.degraded_clauses)}; re-handshake the lowered run plan"
            )
        if decision.agreement is None or not decision.admissible:
            raise HandshakeError(f"handshake decision {decision.outcome!r} is not admissible")
        agreement = decision.agreement
        if agreement.handshake_id in self._revoked:
            raise HandshakeRevoked(
                f"handshake {agreement.handshake_id} was revoked: "
                f"{self._revoked[agreement.handshake_id]}"
            )
        self._agreements[agreement.handshake_id] = agreement
        return agreement

    def revoke(self, handshake_id: str, reason: str) -> None:
        """Stop admitting new attempts; in-flight work finishes or cancels."""

        self._revoked[handshake_id] = reason or "revoked by container"
        self._agreements.pop(handshake_id, None)

    def assert_admissible(
        self,
        handshake_id: str,
        agreement_digest: str,
        *,
        now: datetime | None = None,
    ) -> Agreement:
        """The gate every attempt, resume, and train dequeue passes."""

        if handshake_id in self._revoked:
            raise HandshakeRevoked(
                f"handshake {handshake_id} was revoked: {self._revoked[handshake_id]}"
            )
        agreement = self._agreements.get(handshake_id)
        if agreement is None:
            raise UnknownHandshake(f"handshake {handshake_id!r} was never admitted")
        moment = now or self._clock()
        if agreement.expires_at <= moment:
            raise HandshakeExpired(
                f"handshake {handshake_id} expired at {format_rfc3339(agreement.expires_at)}"
            )
        if agreement_digest != agreement.agreement_digest:
            raise AgreementMismatch(
                f"attempt names agreement {agreement_digest} but handshake "
                f"{handshake_id} agreed {agreement.agreement_digest}"
            )
        return agreement

    def renew(
        self,
        handshake_id: str,
        *,
        capability: CapabilityDocument,
        verdict: HandshakeVerdict,
        now: datetime | None = None,
    ) -> Agreement:
        """Re-read the capability document and fail closed on any change."""

        if handshake_id in self._revoked:
            raise HandshakeRevoked(
                f"handshake {handshake_id} was revoked: {self._revoked[handshake_id]}"
            )
        agreement = self._agreements.get(handshake_id)
        if agreement is None:
            raise UnknownHandshake(f"handshake {handshake_id!r} was never admitted")
        try:
            capability.assert_unchanged(agreement.capability_hash)
        except CapabilityDriftError:
            self.revoke(handshake_id, "capability document changed under a live handshake")
            raise
        if verdict.handshake_id != handshake_id:
            raise AgreementMismatch(
                f"renewal answers handshake {verdict.handshake_id!r}, not {handshake_id!r}"
            )
        if verdict.capability_hash != agreement.capability_hash:
            self.revoke(handshake_id, "renewal named a different capability hash")
            raise CapabilityDriftError(
                "renewal named a different capability hash: "
                f"{verdict.capability_hash} != {agreement.capability_hash}"
            )
        if verdict.agreement_digest != agreement.agreement_digest:
            raise AgreementMismatch(
                "renewal changed the agreement digest; re-handshake instead: "
                f"{verdict.agreement_digest} != {agreement.agreement_digest}"
            )
        moment = now or self._clock()
        if verdict.expires_at <= moment:
            raise HandshakeExpired(
                f"renewal of {handshake_id} expires at "
                f"{format_rfc3339(verdict.expires_at)}, already past"
            )
        renewed = replace(agreement, expires_at=verdict.expires_at, admitted_at=moment)
        self._agreements[handshake_id] = renewed
        return renewed
