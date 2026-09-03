"""Group identity, declared topology, and attempt identity.

A group is the unit of comparison, so anything that changes what a sample means
must be identical across its members. Topology is a container-declared fact:
nothing here infers a roster from an agent count, a role name, or a task name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .rl_records import TERMINAL_STATUSES, RecordError, digest

GROUP_PIN_SCHEMA_VERSION = "cispo.group_pin.v1"
TOPOLOGY_SCHEMA_VERSION = "cispo.topology.v1"
TASK_SPEC_SCHEMA_VERSION = "cispo.task_spec.v1"
ROLLOUT_RECEIPT_SCHEMA_VERSION = "cispo.rollout_receipt.v1"

TURN_MODELS = frozenset({"sequential", "concurrent_realtime"})
ACTUATION_MODELS = frozenset({"direct_action", "deferred_program"})
REWARD_RELATIONS = frozenset(
    {"cooperative", "competitive_rank", "competitive_margin", "mixed"}
)
PARTIAL_ROSTER_DISPOSITIONS = frozenset({"refuse", "drop_instance", "refuse_team"})
HORIZON_KINDS = frozenset({"wall_clock", "steps", "env_ticks"})
CHANNEL_SCOPES = frozenset({"intra_team", "cross_team", "private"})
ATTEMPT_STATES = (
    "queued",
    "running",
    "awaiting_score",
    "scored",
    "completed",
    "failed",
    "cancelled",
)
# One definition, so a reward receipt and an attempt receipt cannot disagree
# about what "terminal" means.
TERMINAL_ATTEMPT_STATES = TERMINAL_STATUSES


class MixedGroupError(RecordError):
    """A group's members disagree on a pinned field. Never average across it."""


class TopologyError(RecordError):
    """A declared topology was incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class GroupPin:
    """One group, one pin. Mixing any pinned field is a rejection."""

    group_id: str
    run_id: str
    algorithm_plan_hash: str
    behavior_fingerprint: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    policy_kind: str
    model_family: str
    container_image_digest: str
    container_contract_hash: str
    handshake_agreement_digest: str
    task_family: str
    cardinality: int
    policy_set_revision_id: str | None = None
    match_set_revision_id: str | None = None
    topology_id: str | None = None
    # The queue counts revisions; the catalog names them. Carry both so a pin
    # bridges to an immutable checkpoint identity without a lookup convention.
    policy_revision_id: str | None = None
    policy_span_count: int = 1
    schema_version: str = GROUP_PIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.cardinality < 1:
            raise RecordError("group cardinality must be positive")
        if self.policy_span_count != 1:
            raise RecordError(
                "policy_span_count must be 1: a group may not straddle two published revisions"
            )

    def mixing_key(self) -> tuple[Any, ...]:
        """Every field that makes two samples incomparable."""

        return (
            self.algorithm_plan_hash,
            self.behavior_fingerprint,
            self.policy_revision,
            self.wire_api,
            self.sampling_transport,
            self.policy_kind,
            self.model_family,
            self.container_image_digest,
            self.container_contract_hash,
            self.handshake_agreement_digest,
            self.task_family,
            self.policy_set_revision_id,
            self.match_set_revision_id,
            self.topology_id,
        )

    def mixing_fields(self) -> Mapping[str, Any]:
        names = (
            "algorithm_plan_hash",
            "behavior_fingerprint",
            "policy_revision",
            "wire_api",
            "sampling_transport",
            "policy_kind",
            "model_family",
            "container_image_digest",
            "container_contract_hash",
            "handshake_agreement_digest",
            "task_family",
            "policy_set_revision_id",
            "match_set_revision_id",
            "topology_id",
        )
        return dict(zip(names, self.mixing_key(), strict=True))

    @property
    def pin_digest(self) -> str:
        return digest(list(self.mixing_key()), length=32)


def assert_uniform_group(pins: Sequence[GroupPin]) -> GroupPin:
    """Reject a group whose members disagree, naming the offending field."""

    if not pins:
        raise MixedGroupError("group has no members")
    head = pins[0]
    expected = head.mixing_fields()
    for pin in pins[1:]:
        for name, value in pin.mixing_fields().items():
            if expected[name] != value:
                raise MixedGroupError(
                    f"group {head.group_id} mixes {name}: {expected[name]!r} != {value!r}"
                )
    return head


@dataclass(frozen=True, slots=True)
class AgentInstance:
    agent_instance_id: str
    role_id: str
    policy_type_id: str
    team_id: str
    trainable: bool
    pinned_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_instance_id.strip():
            raise TopologyError("agent_instance_id is required")
        if not self.trainable and not self.pinned_identity:
            raise TopologyError(
                f"non-trainable instance {self.agent_instance_id} must pin an immutable identity"
            )


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str
    trainable: bool
    minimum_viable_roster: int = 1


@dataclass(frozen=True, slots=True)
class CommunicationChannel:
    channel_id: str
    scope: str
    trainable_for_author: bool = True

    def __post_init__(self) -> None:
        if self.scope not in CHANNEL_SCOPES:
            raise TopologyError(f"unknown channel scope {self.scope!r}")


@dataclass(frozen=True, slots=True)
class Horizon:
    horizon_kind: str
    value: float
    time_dilation: float = 1.0
    grace_seconds: float = 0.0
    # A step or tick horizon carries no duration of its own, so a lease cannot
    # be derived from it without a declared conversion. Undeclared stays None so
    # it fails closed: silently reading 500 steps as 500 seconds is exactly the
    # guess the note forbids. Deriving the two clocks a lease needs — heartbeat
    # TTL and straggler deadline, which are not the same thing — belongs to the
    # lease sizing that also knows the quiescence and collection budgets.
    seconds_per_unit: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_kind not in HORIZON_KINDS:
            raise TopologyError(f"unknown horizon_kind {self.horizon_kind!r}")
        if self.value <= 0:
            raise TopologyError("horizon value must be positive")
        if self.seconds_per_unit is not None and self.seconds_per_unit <= 0:
            raise TopologyError("seconds_per_unit must be positive when declared")
        if self.time_dilation <= 0:
            raise TopologyError("time_dilation must be positive")

    def declared_seconds_per_unit(self) -> float:
        """The conversion, or a refusal. Never a default."""

        if self.horizon_kind == "wall_clock":
            return 1.0
        if self.seconds_per_unit is None:
            raise TopologyError(
                f"a {self.horizon_kind} horizon must declare seconds_per_unit; "
                "a lease may not be guessed from a unit with no duration"
            )
        return self.seconds_per_unit


@dataclass(frozen=True, slots=True)
class Topology:
    """A container-declared roster. The executor binds it; it never infers it."""

    topology_id: str
    turn_model: str
    actuation_model: str
    reward_relation: str
    agent_instances: tuple[AgentInstance, ...]
    teams: tuple[Team, ...]
    communication_channels: tuple[CommunicationChannel, ...] = ()
    horizon: Horizon | None = None
    parameter_groups: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = TOPOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.turn_model not in TURN_MODELS:
            raise TopologyError(f"unknown turn_model {self.turn_model!r}")
        if self.actuation_model not in ACTUATION_MODELS:
            raise TopologyError(f"unknown actuation_model {self.actuation_model!r}")
        if self.reward_relation not in REWARD_RELATIONS:
            raise TopologyError(f"unknown reward_relation {self.reward_relation!r}")
        if not self.agent_instances:
            raise TopologyError("topology declares no agent instances")
        ids = [instance.agent_instance_id for instance in self.agent_instances]
        if len(set(ids)) != len(ids):
            raise TopologyError("duplicate agent_instance_id in topology")
        team_ids = {team.team_id for team in self.teams}
        for instance in self.agent_instances:
            if instance.team_id not in team_ids:
                raise TopologyError(
                    f"instance {instance.agent_instance_id} names undeclared team "
                    f"{instance.team_id!r}"
                )
        if self.turn_model == "concurrent_realtime" and self.horizon is None:
            raise TopologyError("a concurrent real-time topology must declare a horizon")

    @property
    def is_multi_policy(self) -> bool:
        return len({instance.policy_type_id for instance in self.trainable_instances}) > 1

    @property
    def trainable_instances(self) -> tuple[AgentInstance, ...]:
        return tuple(instance for instance in self.agent_instances if instance.trainable)

    @property
    def opponent_instances(self) -> tuple[AgentInstance, ...]:
        return tuple(instance for instance in self.agent_instances if not instance.trainable)

    def parameter_group_for(self, agent_instance_id: str) -> str:
        for instance in self.agent_instances:
            if instance.agent_instance_id != agent_instance_id:
                continue
            if not instance.trainable:
                raise TopologyError(
                    f"instance {agent_instance_id} is not trainable and has no parameter group"
                )
            group = self.parameter_groups.get(instance.policy_type_id)
            if not group:
                raise TopologyError(
                    f"policy type {instance.policy_type_id!r} has no declared parameter group"
                )
            return group
        raise TopologyError(f"unknown agent instance {agent_instance_id!r}")

    def trainable_parameter_groups(self) -> tuple[str, ...]:
        seen: list[str] = []
        for instance in self.trainable_instances:
            group = self.parameter_groups.get(instance.policy_type_id)
            if group and group not in seen:
                seen.append(group)
        return tuple(seen)

    def roster_disposition(
        self, live_instance_ids: Iterable[str], *, disposition: str
    ) -> "RosterOutcome":
        """Apply the declared partial-roster disposition.

        ``refuse`` fails the episode on any absence. ``drop_instance`` trains on
        the survivors but fails a team that fell below its minimum viable
        roster. ``refuse_team`` excludes such a team and leaves the rest of the
        episode valid, which is the whole point of naming it separately.
        """

        if disposition not in PARTIAL_ROSTER_DISPOSITIONS:
            raise TopologyError(f"unknown partial roster disposition {disposition!r}")
        live = set(live_instance_ids)
        missing = tuple(
            instance.agent_instance_id
            for instance in self.agent_instances
            if instance.agent_instance_id not in live
        )
        if not missing:
            return RosterOutcome((), ())
        if disposition == "refuse":
            raise TopologyError(f"topology {self.topology_id} is missing instances: {missing}")
        below: list[str] = []
        for team in self.teams:
            surviving = sum(
                1
                for instance in self.agent_instances
                if instance.team_id == team.team_id and instance.agent_instance_id in live
            )
            if surviving < team.minimum_viable_roster:
                below.append(team.team_id)
        if below and disposition == "drop_instance":
            raise TopologyError(
                f"teams {tuple(below)} fell below their minimum viable roster; "
                "disposition 'drop_instance' cannot proceed"
            )
        return RosterOutcome(missing, tuple(below))

    def check_roster(
        self, live_instance_ids: Iterable[str], *, disposition: str
    ) -> tuple[str, ...]:
        """Missing instance ids under the declared disposition."""

        return self.roster_disposition(live_instance_ids, disposition=disposition).missing_instances


@dataclass(frozen=True, slots=True)
class RosterOutcome:
    """What survived a partial roster, and which teams were excluded."""

    missing_instances: tuple[str, ...]
    refused_teams: tuple[str, ...]

    @property
    def degraded(self) -> bool:
        return bool(self.missing_instances or self.refused_teams)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """What the curriculum writes. No tokens, no harness, no reward rule."""

    task_id: str
    split: str
    seed: int
    group_id: str
    task_family: str
    content_digest: str
    topology_ref: str | None = None
    tags: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TASK_SPEC_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RolloutReceipt:
    """What leaves the done boundary: identity plus digests, never raw tokens."""

    rollout_id: str
    proxy_request_id: str
    group_id: str
    sample_index: int
    policy_revision: int
    behavior_fingerprint: str
    terminal_status: str
    trace_digest: str
    evidence_digest: str
    reward_id: str | None = None
    handshake_id: str = ""
    # Per-attempt admission is auditable without reaching for the group pin.
    agreement_digest: str = ""
    agent_instance_id: str | None = None
    team_id: str | None = None
    probe: bool = False
    replaced_attempt_id: str | None = None
    # A receipt should be self-describing about why it exists, so cost is
    # attributable without joining the queue journal.
    replacement_index: int = 0
    replacement_reason: str | None = None
    # An attempt must be able to name the immutable artifacts it sampled from,
    # rather than leaving the run receipt to infer them from a fingerprint.
    checkpoint_id: str | None = None
    policy_set_revision_id: str | None = None
    match_set_revision_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ROLLOUT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.terminal_status not in TERMINAL_ATTEMPT_STATES:
            raise RecordError(f"receipt terminal_status {self.terminal_status!r} is not terminal")
