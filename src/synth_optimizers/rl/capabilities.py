"""The hashed capability document and the requirements the executor demands.

Reading this document is discovery, not agreement: it says what a container can
do in general, and the handshake says whether it can honor one particular run.
Everything here happens before a training session exists and before one paid
request is issued.

The content hash is fail-closed. It covers the whole document, so any change at
all invalidates a prior preflight and every handshake built on it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.rl_clauses import (
    ALL_CLAUSES,
    MANDATORY_CLAUSES,
    OPTIONAL_CLAUSES,
    VERDICTS,
)
from ..contracts.rl_identity import (
    ACTUATION_MODELS,
    REWARD_RELATIONS,
    AgentInstance,
    CommunicationChannel,
    Horizon,
    Team,
    Topology,
    TopologyError,
)
from ..contracts.rl_records import (
    SAMPLING_TRANSPORTS,
    WIRE_APIS,
    RendererProfile,
)
from .contract import CISPO_CONTRACT_VERSION, ContainerContract

CAPABILITY_SCHEMA_VERSION = "cispo.capabilities.v1"

# Worse verdicts win when two checks answer the same clause.
VERDICT_SEVERITY: dict[str, int] = {"accepted": 0, "degraded": 1, "unsupported": 2, "rejected": 3}


class CapabilityError(ValueError):
    """The capability document is absent, malformed, or self-inconsistent."""


class CapabilityHashError(CapabilityError):
    """The document's content hash does not match the document."""


class CapabilityDriftError(CapabilityError):
    """The capability document changed under a live preflight. Fail closed."""


class PreflightRejected(CapabilityError):
    """A mandatory clause was rejected. No session, no paid request, no spend."""

    def __init__(self, results: Sequence["ClauseResult"]) -> None:
        self.results = tuple(results)
        detail = "; ".join(f"{item.clause_id}: {item.reason}" for item in self.results)
        super().__init__(f"capability preflight rejected {len(self.results)} clause(s): {detail}")

    @property
    def clause_ids(self) -> tuple[str, ...]:
        return tuple(item.clause_id for item in self.results)


@dataclass(frozen=True, slots=True)
class ClauseResult:
    """One clause, one verdict, one reason. Never a bare boolean."""

    clause_id: str
    verdict: str
    reason: str = ""
    source: str = "executor"

    def __post_init__(self) -> None:
        if self.clause_id not in ALL_CLAUSES:
            raise CapabilityError(f"unknown clause {self.clause_id!r}")
        if self.verdict not in VERDICTS:
            raise CapabilityError(f"unknown verdict {self.verdict!r} for {self.clause_id}")

    @property
    def mandatory(self) -> bool:
        return self.clause_id not in OPTIONAL_CLAUSES

    @property
    def severity(self) -> int:
        return VERDICT_SEVERITY[self.verdict]

    @property
    def blocks_run(self) -> bool:
        return self.mandatory and self.verdict in {"rejected", "unsupported"}

    def to_payload(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "source": self.source,
        }


def merge_clause_results(*groups: Sequence[ClauseResult]) -> tuple[ClauseResult, ...]:
    """Keep the worst verdict per clause, in canonical clause order."""

    worst: dict[str, ClauseResult] = {}
    for group in groups:
        for result in group:
            current = worst.get(result.clause_id)
            if current is None or result.severity > current.severity:
                worst[result.clause_id] = result
    return tuple(worst[clause] for clause in ALL_CLAUSES if clause in worst)


def rejected_mandatory(results: Sequence[ClauseResult]) -> tuple[ClauseResult, ...]:
    return tuple(result for result in results if result.blocks_run)


def assert_preflight_passed(results: Sequence[ClauseResult]) -> None:
    """Stop before session creation if any mandatory clause failed."""

    blocking = rejected_mandatory(results)
    if blocking:
        raise PreflightRejected(blocking)


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise CapabilityError(f"capability document field {name!r} must be an object")
    return value


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise CapabilityError(f"capability document field {name!r} is required")
    return value.strip()


def _flag(payload: Mapping[str, Any], name: str, *, default: bool | None = None) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise CapabilityError(f"capability document field {name!r} must be a boolean")
    return value


def _number(payload: Mapping[str, Any], name: str, *, default: float | None = None) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CapabilityError(f"capability document field {name!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CapabilityError(f"capability document field {name!r} must be finite")
    return number


def _positive_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CapabilityError(f"capability document field {name!r} must be a positive integer")
    return value


def canonical_capability_hash(document: Mapping[str, Any]) -> str:
    """Hash every field but the offered hash itself, as the existing preflight does."""

    unhashed = {key: value for key, value in document.items() if key != "capability_hash"}
    raw = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class DiscoveryCapability:
    taskset_id: str
    taskset_version: str
    splits: tuple[str, ...]
    task_content_digests: bool
    deterministic_lookup: bool
    duplicate_free: bool


@dataclass(frozen=True, slots=True)
class PolicyCapability:
    binding_transport: str
    wire_api: str
    session_scoped_sampler_origin: bool
    embeds_credentials: bool
    revision_immutable_after_admission: bool
    records_policy_revision: bool

    def __post_init__(self) -> None:
        if self.binding_transport not in SAMPLING_TRANSPORTS:
            raise CapabilityError(f"unknown binding_transport {self.binding_transport!r}")
        if self.wire_api not in WIRE_APIS:
            raise CapabilityError(f"unknown wire_api {self.wire_api!r}")


@dataclass(frozen=True, slots=True)
class LifecycleCapability:
    max_concurrency: int
    lease_ttl_seconds: float
    supports_idempotency: bool
    supports_cancellation: bool
    supports_lease_renewal: bool
    exactly_one_terminal_result: bool
    supports_pause_resume: bool = False
    straggler_grace_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EvidenceCapability:
    trace_v5: bool
    behavior_logprobs: bool
    strict_prefix: bool
    masking: bool
    wire_objects: bool
    artifact_reference: bool = False
    tokens_in_tokens_out: bool = False


@dataclass(frozen=True, slots=True)
class RewardCapability:
    authority: str
    binds_trace_digest: bool
    quiescence: bool
    horizon_clipping: bool
    channels: tuple[str, ...]
    reward_relation: str
    evaluation_plan_id: str
    settlement_window_seconds: float = 0.0
    deferred_scoring: bool = False

    def __post_init__(self) -> None:
        if self.reward_relation not in REWARD_RELATIONS:
            raise CapabilityError(f"unknown reward_relation {self.reward_relation!r}")


@dataclass(frozen=True, slots=True)
class RecoveryCapability:
    restart: bool
    stale_discard: bool


@dataclass(frozen=True, slots=True)
class CapabilityDocument:
    """The whole advertisement, typed, with its fail-closed content hash."""

    schema_version: str
    container_id: str
    container_image_digest: str
    contract_version: str
    renderer_profile: RendererProfile
    discovery: DiscoveryCapability
    policy: PolicyCapability
    lifecycle: LifecycleCapability
    evidence: EvidenceCapability
    reward: RewardCapability
    recovery: RecoveryCapability
    topology: Topology
    topology_ref: str
    horizon: Horizon
    clock_skew_tolerance_seconds: float
    content_hash: str
    raw: Mapping[str, Any] = field(default_factory=dict)

    def assert_unchanged(self, previous_hash: str) -> None:
        """Renewal and restart both re-read this document and fail closed."""

        if previous_hash and previous_hash != self.content_hash:
            raise CapabilityDriftError(
                "capability document changed under a live preflight: "
                f"{previous_hash} != {self.content_hash}"
            )

    @classmethod
    def from_payload(cls, payload: Any) -> "CapabilityDocument":
        if not isinstance(payload, Mapping):
            raise CapabilityError("capability document must be an object")
        if payload.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
            raise CapabilityError(
                f"capability schema {payload.get('schema_version')!r} is unsupported; "
                f"expected {CAPABILITY_SCHEMA_VERSION}"
            )
        offered = _text(payload, "capability_hash")
        computed = canonical_capability_hash(payload)
        if offered != computed:
            raise CapabilityHashError(
                f"capability hash mismatch: offered {offered}, computed {computed}"
            )
        discovery_raw = _mapping(payload, "discovery")
        splits = discovery_raw.get("splits")
        if not isinstance(splits, Sequence) or isinstance(splits, str | bytes) or not splits:
            raise CapabilityError("capability discovery.splits must be a non-empty list")
        policy_raw = _mapping(payload, "policy")
        lifecycle_raw = _mapping(payload, "lifecycle")
        evidence_raw = _mapping(payload, "evidence")
        reward_raw = _mapping(payload, "reward")
        recovery_raw = _mapping(payload, "recovery")
        topology_raw = _mapping(payload, "topology")
        clock_raw = payload.get("clock")
        clock = clock_raw if isinstance(clock_raw, Mapping) else {}
        channels = reward_raw.get("channels")
        if not isinstance(channels, Sequence) or isinstance(channels, str | bytes) or not channels:
            raise CapabilityError("capability reward.channels must be a non-empty list")
        topology = _parse_topology(topology_raw)
        if topology.horizon is None:
            raise CapabilityError("capability topology must declare a horizon")
        return cls(
            schema_version=CAPABILITY_SCHEMA_VERSION,
            container_id=_text(payload, "container_id"),
            container_image_digest=_text(payload, "container_image_digest"),
            contract_version=_text(payload, "contract_version"),
            renderer_profile=RendererProfile.from_payload(_mapping(payload, "renderer_profile")),
            discovery=DiscoveryCapability(
                taskset_id=_text(discovery_raw, "taskset_id"),
                taskset_version=_text(discovery_raw, "taskset_version"),
                splits=tuple(str(item) for item in splits),
                task_content_digests=_flag(discovery_raw, "task_content_digests"),
                deterministic_lookup=_flag(discovery_raw, "deterministic_lookup"),
                duplicate_free=_flag(discovery_raw, "duplicate_free"),
            ),
            policy=PolicyCapability(
                binding_transport=_text(policy_raw, "binding_transport"),
                wire_api=_text(policy_raw, "wire_api"),
                session_scoped_sampler_origin=_flag(policy_raw, "session_scoped_sampler_origin"),
                embeds_credentials=_flag(policy_raw, "embeds_credentials"),
                revision_immutable_after_admission=_flag(
                    policy_raw, "revision_immutable_after_admission"
                ),
                records_policy_revision=_flag(policy_raw, "records_policy_revision"),
            ),
            lifecycle=LifecycleCapability(
                max_concurrency=_positive_int(lifecycle_raw, "max_concurrency"),
                lease_ttl_seconds=_number(lifecycle_raw, "lease_ttl_seconds"),
                supports_idempotency=_flag(lifecycle_raw, "supports_idempotency"),
                supports_cancellation=_flag(lifecycle_raw, "supports_cancellation"),
                supports_lease_renewal=_flag(lifecycle_raw, "supports_lease_renewal"),
                exactly_one_terminal_result=_flag(lifecycle_raw, "exactly_one_terminal_result"),
                supports_pause_resume=_flag(
                    lifecycle_raw, "supports_pause_resume", default=False
                ),
                straggler_grace_seconds=_number(
                    lifecycle_raw, "straggler_grace_seconds", default=0.0
                ),
            ),
            evidence=EvidenceCapability(
                trace_v5=_flag(evidence_raw, "trace_v5"),
                behavior_logprobs=_flag(evidence_raw, "behavior_logprobs"),
                strict_prefix=_flag(evidence_raw, "strict_prefix"),
                masking=_flag(evidence_raw, "masking"),
                wire_objects=_flag(evidence_raw, "wire_objects"),
                artifact_reference=_flag(evidence_raw, "artifact_reference", default=False),
                tokens_in_tokens_out=_flag(evidence_raw, "tokens_in_tokens_out", default=False),
            ),
            reward=RewardCapability(
                authority=_text(reward_raw, "authority"),
                binds_trace_digest=_flag(reward_raw, "binds_trace_digest"),
                quiescence=_flag(reward_raw, "quiescence"),
                horizon_clipping=_flag(reward_raw, "horizon_clipping"),
                channels=tuple(str(item) for item in channels),
                reward_relation=_text(reward_raw, "reward_relation"),
                evaluation_plan_id=_text(reward_raw, "evaluation_plan_id"),
                settlement_window_seconds=_number(
                    reward_raw, "settlement_window_seconds", default=0.0
                ),
                deferred_scoring=_flag(reward_raw, "deferred_scoring", default=False),
            ),
            recovery=RecoveryCapability(
                restart=_flag(recovery_raw, "restart"),
                stale_discard=_flag(recovery_raw, "stale_discard"),
            ),
            topology=topology,
            topology_ref=topology.topology_id,
            horizon=topology.horizon,
            clock_skew_tolerance_seconds=_number(
                clock, "skew_tolerance_seconds", default=0.0
            ),
            content_hash=computed,
            raw=dict(payload),
        )


def _parse_topology(payload: Mapping[str, Any]) -> Topology:
    instances_raw = payload.get("agent_instances")
    teams_raw = payload.get("teams")
    if not isinstance(instances_raw, Sequence) or isinstance(instances_raw, str | bytes):
        raise CapabilityError("capability topology.agent_instances must be a list")
    if not isinstance(teams_raw, Sequence) or isinstance(teams_raw, str | bytes):
        raise CapabilityError("capability topology.teams must be a list")
    channels_raw = payload.get("communication_channels") or ()
    horizon_raw = payload.get("horizon")
    horizon: Horizon | None = None
    if isinstance(horizon_raw, Mapping):
        # The record names the magnitude `value`, because a step or tick horizon
        # has no seconds; `value_seconds` is accepted as the older spelling.
        magnitude = "value" if "value" in horizon_raw else "value_seconds"
        conversion = horizon_raw.get("seconds_per_unit")
        horizon = Horizon(
            horizon_kind=_text(horizon_raw, "horizon_kind"),
            value=_number(horizon_raw, magnitude),
            time_dilation=_number(horizon_raw, "time_dilation", default=1.0),
            grace_seconds=_number(horizon_raw, "grace_seconds", default=0.0),
            # Absent stays absent: a unit horizon with no declared conversion
            # must fail closed at lease sizing rather than default to one
            # second per unit.
            seconds_per_unit=(
                None if conversion is None else _number(horizon_raw, "seconds_per_unit")
            ),
        )
    parameter_groups_raw = payload.get("parameter_groups") or {}
    if not isinstance(parameter_groups_raw, Mapping):
        raise CapabilityError("capability topology.parameter_groups must be an object")
    try:
        return Topology(
            topology_id=_text(payload, "topology_id"),
            turn_model=_text(payload, "turn_model"),
            actuation_model=_text(payload, "actuation_model"),
            reward_relation=_text(payload, "reward_relation"),
            agent_instances=tuple(
                AgentInstance(
                    agent_instance_id=_text(item, "agent_instance_id"),
                    role_id=_text(item, "role_id"),
                    policy_type_id=_text(item, "policy_type_id"),
                    team_id=_text(item, "team_id"),
                    trainable=_flag(item, "trainable"),
                    pinned_identity=(
                        str(item["pinned_identity"]) if item.get("pinned_identity") else None
                    ),
                )
                for item in instances_raw
                if isinstance(item, Mapping)
            ),
            teams=tuple(
                Team(
                    team_id=_text(item, "team_id"),
                    trainable=_flag(item, "trainable"),
                    minimum_viable_roster=int(item.get("minimum_viable_roster", 0) or 0),
                )
                for item in teams_raw
                if isinstance(item, Mapping)
            ),
            communication_channels=tuple(
                CommunicationChannel(
                    channel_id=_text(item, "channel_id"),
                    scope=_text(item, "scope"),
                    trainable_for_author=_flag(item, "trainable_for_author", default=True),
                )
                for item in channels_raw
                if isinstance(item, Mapping)
            ),
            horizon=horizon,
            parameter_groups={str(k): str(v) for k, v in parameter_groups_raw.items()},
        )
    except TopologyError as exc:
        raise CapabilityError(f"capability topology is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExecutorRequirements:
    """What this run needs. The executor never lowers a mandatory requirement."""

    renderer_profile: RendererProfile
    min_concurrency: int
    horizon_seconds: float
    optimized_channel: str
    sampling_transport: str = "message_in_capture_out"
    wire_api: str = "chat_completions"
    split: str = "train"
    expected_taskset_id: str | None = None
    expected_topology_ref: str | None = None
    minimum_viable_roster: int = 1
    require_quiescence: bool = True
    require_artifact_reference: bool = False
    require_tito: bool = False
    require_pause_resume: bool = False
    require_channels: bool = False
    require_opponent_pinning: bool = False
    require_settlement_window: bool = False
    clock_skew_tolerance_seconds: float = 2.0
    accepted_actuation_models: frozenset[str] = field(
        default_factory=lambda: frozenset(ACTUATION_MODELS)
    )
    accepted_reward_relations: frozenset[str] = field(
        default_factory=lambda: frozenset(REWARD_RELATIONS)
    )

    def __post_init__(self) -> None:
        if self.min_concurrency < 1:
            raise CapabilityError("min_concurrency must be positive")
        if self.horizon_seconds <= 0:
            raise CapabilityError("horizon_seconds must be positive")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise CapabilityError(f"unknown sampling_transport {self.sampling_transport!r}")
        if self.wire_api not in WIRE_APIS:
            raise CapabilityError(f"unknown wire_api {self.wire_api!r}")

    def declared_clauses(self) -> tuple[str, ...]:
        """The clause ids sent in the handshake ``requirements`` list."""

        optional = []
        if self.require_artifact_reference:
            optional.append("evidence.artifact_reference")
        if self.require_tito:
            optional.append("evidence.tito")
        if self.require_pause_resume:
            optional.append("lifecycle.pause_resume")
        if self.require_channels:
            optional.append("topology.channels")
        if self.require_opponent_pinning:
            optional.append("topology.opponent_pinning")
        if self.require_settlement_window:
            optional.append("reward.settlement_window")
        if self.minimum_viable_roster > 1:
            optional.append("topology.minimum_roster")
        return tuple(MANDATORY_CLAUSES) + tuple(
            clause for clause in ALL_CLAUSES if clause in set(optional)
        )


def _clause(
    clause_id: str,
    ok: bool,
    reason: str,
    *,
    required: bool = True,
    degraded: bool = False,
) -> ClauseResult:
    if ok:
        return ClauseResult(clause_id=clause_id, verdict="accepted")
    if degraded:
        return ClauseResult(clause_id=clause_id, verdict="degraded", reason=reason)
    if not required and clause_id in OPTIONAL_CLAUSES:
        return ClauseResult(clause_id=clause_id, verdict="unsupported", reason=reason)
    return ClauseResult(clause_id=clause_id, verdict="rejected", reason=reason)


def _horizon_seconds(horizon: Horizon) -> float:
    """A horizon in seconds, whatever unit it was declared in.

    `Horizon.value` is a magnitude in the horizon's own units -- one step, a
    thousand ticks -- and a run plan asks for seconds. Comparing the two
    directly makes a one-step horizon look shorter than any plan, which
    degrades a clause that is in fact satisfied, and lowering the plan cannot
    fix it because the units never met.
    """

    try:
        return float(horizon.value) * horizon.declared_seconds_per_unit()
    except TopologyError:
        # A unit horizon that declared no conversion. It fails closed later,
        # where a lease is actually sized; here it simply covers nothing.
        return 0.0


def check_requirements(
    document: CapabilityDocument,
    requirements: ExecutorRequirements,
    *,
    contract: ContainerContract | None = None,
) -> tuple[ClauseResult, ...]:
    """Per-clause results, from the capability document alone, before spend."""

    results: list[ClauseResult] = []
    results.extend(_contract_clauses(document, contract))
    results.extend(_discovery_clauses(document, requirements))
    results.extend(_policy_clauses(document, requirements))
    results.extend(_lifecycle_clauses(document, requirements))
    results.extend(_evidence_clauses(document, requirements))
    results.extend(_reward_clauses(document, requirements))
    results.extend(_recovery_clauses(document))
    results.extend(_topology_clauses(document, requirements))
    return merge_clause_results(results)


def _contract_clauses(
    document: CapabilityDocument, contract: ContainerContract | None
) -> list[ClauseResult]:
    declared = contract.version if contract is not None else document.contract_version
    results = [
        _clause(
            "contract.version",
            declared == CISPO_CONTRACT_VERSION
            and document.contract_version == CISPO_CONTRACT_VERSION,
            f"container declares contract version {declared!r} and capability document "
            f"{document.contract_version!r}; executor speaks {CISPO_CONTRACT_VERSION!r}",
        )
    ]
    if contract is None:
        results.append(
            ClauseResult(
                clause_id="contract.routes",
                verdict="rejected",
                reason="no parsed route table was supplied to the capability preflight",
            )
        )
    else:
        results.append(ClauseResult(clause_id="contract.routes", verdict="accepted"))
    return results


def _discovery_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    discovery = document.discovery
    taskset_ok = requirements.split in discovery.splits and (
        requirements.expected_taskset_id is None
        or requirements.expected_taskset_id == discovery.taskset_id
    )
    topology_ok = (
        requirements.expected_topology_ref is None
        or requirements.expected_topology_ref == document.topology_ref
    ) and document.topology.actuation_model in requirements.accepted_actuation_models
    return [
        _clause(
            "discovery.taskset",
            taskset_ok,
            f"taskset {discovery.taskset_id!r} splits {discovery.splits} do not satisfy "
            f"split {requirements.split!r} / expected id {requirements.expected_taskset_id!r}",
        ),
        _clause(
            "discovery.task_digests",
            discovery.task_content_digests
            and discovery.deterministic_lookup
            and discovery.duplicate_free,
            "taskset rows must carry content digests and resolve deterministically "
            "and duplicate-free",
        ),
        _clause(
            "discovery.topology",
            topology_ok,
            f"topology {document.topology_ref!r} with actuation model "
            f"{document.topology.actuation_model!r} does not satisfy expected "
            f"{requirements.expected_topology_ref!r} / "
            f"{sorted(requirements.accepted_actuation_models)}",
        ),
    ]


def _policy_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    policy = document.policy
    transport_ok = (
        policy.binding_transport == requirements.sampling_transport
        and policy.wire_api == requirements.wire_api
    )
    renderer_ok = (
        document.renderer_profile.fingerprint == requirements.renderer_profile.fingerprint
    )
    return [
        _clause(
            "policy.binding_transport",
            transport_ok,
            f"container binds {policy.binding_transport!r} over {policy.wire_api!r}; "
            f"run requires {requirements.sampling_transport!r} over {requirements.wire_api!r}",
        ),
        _clause(
            "policy.renderer_profile_match",
            renderer_ok,
            "renderer profile mismatch: container "
            f"{document.renderer_profile.profile_id}@"
            f"{document.renderer_profile.fingerprint} != session "
            f"{requirements.renderer_profile.profile_id}@"
            f"{requirements.renderer_profile.fingerprint}",
        ),
        _clause(
            "policy.revision_immutability",
            policy.revision_immutable_after_admission and policy.records_policy_revision,
            "behavior policy identity must be immutable after admission and recorded "
            "on every trainable call",
        ),
        _clause(
            "policy.no_embedded_credentials",
            not policy.embeds_credentials,
            "container embeds raw sampler credentials in rollout requests",
        ),
        _clause(
            "policy.session_scoped_origin",
            policy.session_scoped_sampler_origin,
            "sampler origin must be session scoped, not global",
        ),
    ]


def _lifecycle_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    lifecycle = document.lifecycle
    horizon = document.horizon
    concurrency_ok = lifecycle.max_concurrency >= requirements.min_concurrency
    # A lease shorter than the horizon is fine only if it renews.
    lease_covers = (
        lifecycle.supports_lease_renewal and lifecycle.lease_ttl_seconds > 0
    ) or lifecycle.lease_ttl_seconds >= _horizon_seconds(horizon)
    horizon_covers_plan = _horizon_seconds(horizon) >= requirements.horizon_seconds
    if not lease_covers:
        lease = ClauseResult(
            clause_id="lifecycle.lease_renewal",
            verdict="rejected",
            reason=(
                f"lease ttl {lifecycle.lease_ttl_seconds}s does not cover the declared "
                f"{horizon.horizon_kind} horizon of {_horizon_seconds(horizon)}s and renewal "
                "is unsupported"
            ),
        )
    elif not horizon_covers_plan:
        lease = ClauseResult(
            clause_id="lifecycle.lease_renewal",
            verdict="degraded",
            reason=(
                f"declared horizon {_horizon_seconds(horizon)}s is shorter than the requested "
                f"{requirements.horizon_seconds}s; lower the run plan horizon"
            ),
        )
    else:
        lease = ClauseResult(clause_id="lifecycle.lease_renewal", verdict="accepted")
    return [
        _clause(
            "lifecycle.idempotency",
            lifecycle.supports_idempotency,
            "retrying an idempotency key must not create a second logical attempt",
        ),
        lease,
        _clause(
            "lifecycle.cancellation",
            lifecycle.supports_cancellation,
            "cancellation and terminal failure reporting are mandatory",
        ),
        _clause(
            "lifecycle.concurrency",
            concurrency_ok,
            f"advertised concurrency {lifecycle.max_concurrency} is below the requested "
            f"minimum {requirements.min_concurrency}",
            degraded=lifecycle.max_concurrency >= 1,
        ),
        _clause(
            "lifecycle.exactly_one_terminal",
            lifecycle.exactly_one_terminal_result,
            "exactly one terminal result per accepted attempt is mandatory",
        ),
        _clause(
            "lifecycle.pause_resume",
            lifecycle.supports_pause_resume,
            "container does not support pause and resume",
            required=requirements.require_pause_resume,
        ),
    ]


def _evidence_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    evidence = document.evidence
    tito_required = (
        requirements.require_tito
        or requirements.sampling_transport == "tokens_in_tokens_out"
    )
    return [
        _clause("evidence.trace_v5", evidence.trace_v5, "sealed Trace V5 evidence is mandatory"),
        _clause(
            "evidence.behavior_logprobs",
            evidence.behavior_logprobs,
            "per-token behavior logprobs from the sampling forward pass are mandatory",
        ),
        _clause(
            "evidence.strict_prefix",
            evidence.strict_prefix,
            "multi-turn stitching must follow the strict-prefix rule with branch records",
        ),
        _clause("evidence.masking", evidence.masking, "loss masks are mandatory"),
        _clause(
            "evidence.wire_objects",
            evidence.wire_objects,
            "the original wire objects must be persisted alongside token evidence",
        ),
        _clause(
            "evidence.artifact_reference",
            evidence.artifact_reference,
            "container inlines evidence and cannot store it by reference",
            required=requirements.require_artifact_reference,
        ),
        _clause(
            "evidence.tito",
            evidence.tokens_in_tokens_out,
            "container does not speak tokens-in tokens-out",
            required=tito_required,
        ),
    ]


def _reward_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    reward = document.reward
    channels_ok = (
        requirements.optimized_channel in reward.channels
        and reward.reward_relation in requirements.accepted_reward_relations
        and document.topology.reward_relation == reward.reward_relation
    )
    if reward.reward_relation in {"competitive_rank", "competitive_margin", "mixed"}:
        channels_ok = channels_ok and len(reward.channels) >= len(document.topology.teams)
    if reward.quiescence:
        quiescence = ClauseResult(clause_id="reward.horizon_quiescence", verdict="accepted")
    elif reward.horizon_clipping:
        # Clipping is the declared alternative to quiescence, not a lesser one.
        # The obligation records quiescence=false and the run records the fallback.
        quiescence = ClauseResult(
            clause_id="reward.horizon_quiescence",
            verdict="accepted",
            reason=(
                "container cannot quiesce at the horizon; a horizon-clipped snapshot "
                "is taken at the horizon instead"
            ),
        )
    else:
        quiescence = ClauseResult(
            clause_id="reward.horizon_quiescence",
            verdict="rejected",
            reason="container offers neither quiescence nor a horizon-clipped snapshot",
        )
    if requirements.require_quiescence and not reward.quiescence:
        quiescence = ClauseResult(
            clause_id="reward.horizon_quiescence",
            verdict="rejected",
            reason="run requires a quiescence attestation and the container cannot attest",
        )
    return [
        _clause(
            "reward.authority",
            reward.authority == "container" and bool(reward.evaluation_plan_id),
            f"reward authority {reward.authority!r} with plan "
            f"{reward.evaluation_plan_id!r} is not container-authoritative",
        ),
        _clause(
            "reward.binding_digest",
            reward.binds_trace_digest,
            "reward must be bound to the rollout id and the sealed trace digest",
        ),
        quiescence,
        _clause(
            "reward.settlement_window",
            reward.settlement_window_seconds > 0,
            "container declares no settlement window",
            required=requirements.require_settlement_window,
        ),
        _clause(
            "reward.channels",
            channels_ok,
            f"channels {reward.channels} under relation {reward.reward_relation!r} do not "
            f"carry the optimized channel {requirements.optimized_channel!r} for "
            f"{len(document.topology.teams)} team(s)",
        ),
    ]


def _recovery_clauses(document: CapabilityDocument) -> list[ClauseResult]:
    return [
        _clause(
            "recovery.restart",
            document.recovery.restart,
            "active work must have a recoverable lease across restart",
        ),
        _clause(
            "recovery.stale_discard",
            document.recovery.stale_discard,
            "stale queued work must be discardable before execution",
        ),
    ]


def _topology_clauses(
    document: CapabilityDocument, requirements: ExecutorRequirements
) -> list[ClauseResult]:
    topology = document.topology
    rosters = {team.team_id: team.minimum_viable_roster for team in topology.teams}
    minimum_ok = bool(rosters) and all(value >= 1 for value in rosters.values())
    if requirements.minimum_viable_roster > 1:
        minimum_ok = minimum_ok and all(
            value >= requirements.minimum_viable_roster for value in rosters.values()
        )
    opponents = topology.opponent_instances
    return [
        _clause(
            "topology.roster",
            bool(topology.agent_instances) and bool(topology.teams),
            "topology must declare its full instance roster and teams",
        ),
        _clause(
            "topology.channels",
            bool(topology.communication_channels),
            "topology declares no communication channels",
            required=requirements.require_channels,
        ),
        _clause(
            "topology.minimum_roster",
            minimum_ok,
            f"declared minimum viable rosters {rosters} do not meet "
            f"{requirements.minimum_viable_roster}",
            required=requirements.minimum_viable_roster > 1,
        ),
        _clause(
            "topology.opponent_pinning",
            all(instance.pinned_identity for instance in opponents),
            "a non-trainable instance did not pin an immutable identity",
            required=requirements.require_opponent_pinning or bool(opponents),
        ),
    ]


def preflight_capabilities(
    payload: Any,
    requirements: ExecutorRequirements,
    *,
    contract: ContainerContract | None = None,
    previous_hash: str = "",
) -> tuple[CapabilityDocument, tuple[ClauseResult, ...]]:
    """Parse, hash-check, drift-check, and clause-check before any session."""

    document = CapabilityDocument.from_payload(payload)
    document.assert_unchanged(previous_hash)
    results = check_requirements(document, requirements, contract=contract)
    assert_preflight_passed(results)
    return document, results
