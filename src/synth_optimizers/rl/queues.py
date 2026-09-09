"""Four bounded durable queues, and the dequeue gate that guards training.

```text
 admit (per sample, never a whole group)
   │
   ▼
 ROLLOUT ──dispatch+lease──▶ (in flight) ──▶ SCORE ──▶ SCORED-RESULT
   ▲                              │                        │
   │  lease expiry recovers       │ straggler: cancel       │ accept
   │  the same logical attempt    │ and replace             ▼
   │                              │                    open groups (many)
   └──────────────────────────────┘                    prefer the oldest
                                                            │ complete
                                                            ▼
                                                      TRAIN-READY
                                                            │
                                                   DEQUEUE GATE: recheck
                                                   staleness here, not at
                                                   submit. Stale groups are
                                                   discarded or recycled.
```

The gate is the hard training guarantee: a group's staleness is rechecked when
it leaves the train-ready queue, because that is the moment its advantages are
about to become an update. Production never stops while scoring, training,
checkpointing and publication run — only the rollout queue and the open-group
count reject admission, and downstream fullness throttles dispatch instead of
refusing work that has already executed. Refusing an attempt that already ran
would break the one-terminal-result rule.

Every bound here is configuration: capacities, the staleness ceiling, the
disposition of a stale group, the advertised concurrency. This module chooses no
constant and branches on no algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..contracts.rl_identity import GroupPin, MixedGroupError, assert_uniform_group
from ..contracts.rl_records import RecordError
from .leases import LeaseBook
from .lifecycle import RunLifecycle
from .store import (
    GROUP_COMPLETE,
    GROUP_DISCARDED,
    GROUP_OPEN,
    GROUP_RECYCLED,
    GROUP_TRAIN_READY,
    GROUP_TRAINED,
    QUEUE_ROLLOUT,
    QUEUE_SCORE,
    QUEUE_SCORED_RESULT,
    QUEUE_TRAIN_READY,
    QUEUES,
    AttemptRow,
    GroupRow,
    JournalStore,
    LeaseRow,
    RecoverySnapshot,
)

STALE_DISCARD = "discard"
STALE_RECYCLE = "recycle"
#: What happens to a group the dequeue gate finds stale. Configuration.
STALE_DISPOSITIONS: frozenset[str] = frozenset({STALE_DISCARD, STALE_RECYCLE})


class QueueError(RecordError):
    """A queue refused an operation."""


class QueuePolicyError(QueueError):
    """The configured bounds are not self-consistent."""


class QueueFullError(QueueError):
    """A bounded queue is at capacity. This is backpressure, not a failure."""


class GroupAdmissionError(QueueError):
    """A sample does not fit the group it names."""


class DispatchRefused(QueueError):
    """Dispatch is refused: closed by lifecycle, or advertised concurrency reached."""


class StalenessError(QueueError):
    """A group's staleness is impossible, not merely too large."""


@dataclass(frozen=True, slots=True)
class QueueCapacities:
    """Depth of each bounded queue. The train-ready depth is the pipeline lag."""

    rollout: int
    score: int
    scored_result: int
    train_ready: int

    def __post_init__(self) -> None:
        for name in ("rollout", "score", "scored_result", "train_ready"):
            if getattr(self, name) < 1:
                raise QueuePolicyError(f"{name} capacity must be positive")

    def of(self, queue: str) -> int:
        if queue == QUEUE_ROLLOUT:
            return self.rollout
        if queue == QUEUE_SCORE:
            return self.score
        if queue == QUEUE_SCORED_RESULT:
            return self.scored_result
        if queue == QUEUE_TRAIN_READY:
            return self.train_ready
        raise QueueError(f"unknown queue {queue!r}")


@dataclass(frozen=True, slots=True)
class QueuePolicy:
    """Every bound the engine obeys, supplied by the caller."""

    capacities: QueueCapacities
    max_staleness: int
    max_in_flight: int
    max_open_groups: int
    stale_disposition: str = STALE_DISCARD
    prefer_oldest_open_group: bool = True

    def __post_init__(self) -> None:
        if self.max_staleness < 0:
            raise QueuePolicyError("max_staleness must be non-negative")
        if self.max_in_flight < 1:
            raise QueuePolicyError("max_in_flight must be positive")
        if self.max_open_groups < 1:
            raise QueuePolicyError("max_open_groups must be positive")
        if self.stale_disposition not in STALE_DISPOSITIONS:
            raise QueuePolicyError(f"unknown stale disposition {self.stale_disposition!r}")
        # The pipeline cannot hold more lag than the staleness bound tolerates,
        # or the gate is guaranteed to reject work the queue was told to hold.
        if self.capacities.train_ready - 1 > self.max_staleness:
            raise QueuePolicyError(
                f"train-ready capacity {self.capacities.train_ready} requires "
                f"max_staleness >= {self.capacities.train_ready - 1}, got {self.max_staleness}"
            )


@dataclass(frozen=True, slots=True)
class AttemptRequest:
    """One sample. Groups are never dispatched whole."""

    idempotency_key: str
    pin: GroupPin
    sample_index: int
    task_id: str
    seed: int
    attempt_id: str | None = None
    agent_instance_id: str | None = None
    team_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolved_attempt_id(self) -> str:
        return self.attempt_id or f"{self.pin.group_id}:s{self.sample_index}"


@dataclass(frozen=True, slots=True)
class RecycledSlot:
    """A slot the gate gave back, so the caller can re-admit it under a new pin."""

    sample_index: int
    task_id: str
    seed: int
    attempt_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class GateRejection:
    """A complete group the dequeue gate refused to train on."""

    group_id: str
    staleness: int
    disposition: str
    slots: tuple[RecycledSlot, ...]


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """One pass of the dequeue gate: at most one release, plus what it rejected."""

    released: GroupRow | None
    staleness: int | None
    members: tuple[AttemptRow, ...]
    rejected: tuple[GateRejection, ...]


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What a lease sweep did: recovered, cancelled, replaced."""

    recovered: tuple[str, ...] = ()
    cancelled: tuple[str, ...] = ()
    replacements: Mapping[str, str] = field(default_factory=dict)
    unfillable_groups: tuple[str, ...] = ()


class QueueEngine:
    """The durable queue engine for one run."""

    def __init__(
        self,
        store: JournalStore,
        *,
        run_id: str,
        policy: QueuePolicy,
        leases: LeaseBook,
        lifecycle: RunLifecycle,
    ) -> None:
        if lifecycle.run_id != run_id:
            raise QueuePolicyError("lifecycle and engine disagree on the run id")
        self.store = store
        self.run_id = run_id
        self.policy = policy
        self.leases = leases
        self.lifecycle = lifecycle

    # -- reads -------------------------------------------------------------

    def depth(self, queue: str) -> int:
        if queue not in QUEUES:
            raise QueueError(f"unknown queue {queue!r}")
        return self.store.queue_depth(queue, run_id=self.run_id)

    def capacity(self, queue: str) -> int:
        return self.policy.capacities.of(queue)

    def has_capacity(self, queue: str) -> bool:
        return self.depth(queue) < self.capacity(queue)

    def open_groups(self) -> tuple[GroupRow, ...]:
        return self.store.groups_in_state(GROUP_OPEN, run_id=self.run_id)

    def in_flight(self) -> tuple[AttemptRow, ...]:
        return self.store.in_flight(run_id=self.run_id)

    def group_progress(self, group_id: str) -> tuple[int, int]:
        """(filled slots, cardinality) for one group."""

        group = self.store.group(group_id)
        assert group is not None
        return len(self._filled_slots(group_id)), group.cardinality

    def unfillable_groups(self) -> tuple[str, ...]:
        """Open groups with a slot no live attempt holds any more."""

        unfillable: list[str] = []
        for group in self.open_groups():
            live = set(self._filled_slots(group.group_id))
            for attempt in self.store.attempts_in_group(group.group_id):
                if not attempt.is_terminal:
                    live.add(attempt.sample_index)
            if len(live) < group.cardinality:
                unfillable.append(group.group_id)
        return tuple(unfillable)

    # -- admission ---------------------------------------------------------

    def admit(self, request: AttemptRequest) -> AttemptRow:
        """Admit one sample. Retrying a key returns the same logical attempt."""

        existing = self.store.attempt_for_key(request.idempotency_key)
        if existing is not None:
            if (existing.group_id, existing.sample_index) != (
                request.pin.group_id,
                request.sample_index,
            ):
                raise GroupAdmissionError(
                    f"idempotency key {request.idempotency_key!r} already names "
                    f"{existing.attempt_id} in group {existing.group_id} "
                    f"sample {existing.sample_index}"
                )
            return existing
        self.lifecycle.assert_can_admit()
        return self._admit(request, replaced_attempt_id=None, replacement_index=0, bounded=True)

    def _admit(
        self,
        request: AttemptRequest,
        *,
        replaced_attempt_id: str | None,
        replacement_index: int,
        bounded: bool,
    ) -> AttemptRow:
        pin = request.pin
        group = self._group_for(pin)
        if group.state != GROUP_OPEN:
            raise GroupAdmissionError(
                f"group {group.group_id} is {group.state} and admits no further samples"
            )
        if not 0 <= request.sample_index < group.cardinality:
            raise GroupAdmissionError(
                f"sample index {request.sample_index} is outside group "
                f"{group.group_id} of cardinality {group.cardinality}"
            )
        occupant = self._slot_occupant(group.group_id, request.sample_index)
        if occupant is not None and occupant.attempt_id != replaced_attempt_id:
            raise GroupAdmissionError(
                f"group {group.group_id} sample {request.sample_index} is already held by "
                f"{occupant.attempt_id}"
            )
        if bounded and not self.has_capacity(QUEUE_ROLLOUT):
            raise QueueFullError(
                f"rollout queue is at capacity {self.capacity(QUEUE_ROLLOUT)}; "
                "hold production until it drains"
            )
        attempt_id = request.resolved_attempt_id()
        row, _created = self.store.admit_attempt(
            attempt_id=attempt_id,
            idempotency_key=request.idempotency_key,
            run_id=pin.run_id,
            group_id=group.group_id,
            sample_index=request.sample_index,
            task_id=request.task_id,
            seed=request.seed,
            policy_revision=pin.policy_revision,
            agent_instance_id=request.agent_instance_id,
            team_id=request.team_id,
            replaced_attempt_id=replaced_attempt_id,
            replacement_index=replacement_index,
            metadata=request.metadata,
        )
        return row

    def _group_for(self, pin: GroupPin) -> GroupRow:
        """Open the group on its first sample; every later sample must match it."""

        existing = self.store.group(pin.group_id, required=False)
        if existing is None:
            if pin.run_id != self.run_id:
                raise GroupAdmissionError(
                    f"pin names run {pin.run_id!r}, engine runs {self.run_id!r}"
                )
            if len(self.open_groups()) >= self.policy.max_open_groups:
                raise QueueFullError(
                    f"{self.policy.max_open_groups} groups are already open; "
                    "finish one before opening another"
                )
            return self.store.open_group(
                group_id=pin.group_id,
                run_id=pin.run_id,
                cardinality=pin.cardinality,
                pin=asdict(pin),
                pin_digest=pin.pin_digest,
                policy_revision=pin.policy_revision,
            )
        stored = GroupPin(**dict(existing.pin))
        if stored.run_id != pin.run_id:
            raise MixedGroupError(
                f"group {pin.group_id} mixes run_id: {stored.run_id!r} != {pin.run_id!r}"
            )
        if stored.cardinality != pin.cardinality:
            raise MixedGroupError(
                f"group {pin.group_id} mixes cardinality: "
                f"{stored.cardinality} != {pin.cardinality}"
            )
        assert_uniform_group((stored, pin))
        return existing

    def _filled_slots(self, group_id: str) -> tuple[int, ...]:
        filled: list[int] = []
        for member in self.store.group_members(group_id, active_only=True):
            attempt = self.store.attempt(member.attempt_id)
            if attempt.state == "completed":
                filled.append(attempt.sample_index)
        return tuple(sorted(set(filled)))

    def _slot_occupant(self, group_id: str, sample_index: int) -> AttemptRow | None:
        for member in self.store.group_members(group_id, active_only=True):
            if member.sample_index != sample_index:
                continue
            attempt = self.store.attempt(member.attempt_id)
            if not attempt.is_terminal or attempt.state == "completed":
                return attempt
        return None

    # -- dispatch ----------------------------------------------------------

    def next_dispatch(self, limit: int = 1) -> tuple[AttemptRow, ...]:
        """Queued samples to send now, oldest open group first.

        Returns nothing when the lifecycle closed dispatch, when advertised
        concurrency is exhausted, or when a downstream queue is full. Downstream
        fullness throttles here rather than rejecting executed work.
        """

        if limit < 1:
            raise QueueError("dispatch limit must be positive")
        if not self.lifecycle.gates.dispatch:
            return ()
        if not self.has_capacity(QUEUE_SCORE) or not self.has_capacity(QUEUE_SCORED_RESULT):
            return ()
        headroom = self.policy.max_in_flight - len(self.in_flight())
        if headroom < 1:
            return ()
        return self.store.dispatchable(
            limit=min(limit, headroom),
            run_id=self.run_id,
            oldest_group_first=self.policy.prefer_oldest_open_group,
        )

    def dispatch(self, attempt_id: str, *, holder: str) -> tuple[AttemptRow, LeaseRow]:
        """Send one sample to the container and take a lease sized for its horizon."""

        self.lifecycle.assert_can_dispatch()
        if len(self.in_flight()) >= self.policy.max_in_flight:
            raise DispatchRefused(
                f"advertised concurrency {self.policy.max_in_flight} is exhausted"
            )
        attempt = self.store.transition_attempt(attempt_id, "running", reason="dispatch")
        lease = self.leases.grant(attempt_id, holder=holder)
        return attempt, lease

    def heartbeat(self, attempt_id: str) -> LeaseRow:
        lease = self.leases.lease_for(attempt_id)
        if lease is None:
            raise QueueError(f"attempt {attempt_id} holds no active lease")
        return self.leases.heartbeat(lease.lease_id)

    # -- results -----------------------------------------------------------

    def report_awaiting_score(
        self, attempt_id: str, *, reason: str = "deferred_score"
    ) -> AttemptRow:
        """Staged scoring: rollout capacity is released before scoring begins."""

        self.lifecycle.assert_can_score()
        return self.store.transition_attempt(attempt_id, "awaiting_score", reason=reason)

    def report_scored(
        self, attempt_id: str, *, payload: Mapping[str, Any] | None = None
    ) -> AttemptRow:
        """Evidence and reward receipt are back; the attempt enters validation."""

        self.lifecycle.assert_can_score()
        return self.store.transition_attempt(
            attempt_id, "scored", reason="scored", detail=payload
        )

    def accept_evidence(
        self, attempt_id: str, *, payload: Mapping[str, Any] | None = None
    ) -> AttemptRow:
        """Validation passed: one episode result, and the group slot is filled."""

        self.lifecycle.assert_can_score()
        attempt = self.store.transition_attempt(
            attempt_id, "completed", reason="validated", result_payload=payload
        )
        self.store.close_leases_for(attempt_id, state="released", reason="terminal_result")
        self._maybe_complete_group(attempt.group_id)
        self.promote_ready_groups()
        return attempt

    def reject_evidence(self, attempt_id: str, *, reason: str) -> AttemptRow:
        """Validation failed. An absent reward is a failure, never a zero."""

        self.lifecycle.assert_can_score()
        return self.fail(attempt_id, reason=reason)

    def fail(self, attempt_id: str, *, reason: str) -> AttemptRow:
        attempt = self.store.transition_attempt(
            attempt_id, "failed", reason=reason, result_payload={"reason": reason}
        )
        self.store.close_leases_for(attempt_id, state="released", reason=reason)
        return attempt

    def retry_infrastructure(self, attempt_id: str, *, reason: str) -> AttemptRow | None:
        """Replace a failed, unscored slot without changing its task or policy pin."""
        attempt = self.fail(attempt_id, reason=reason)
        return self._replacement(attempt)

    def cancel(
        self, attempt_id: str, *, reason: str, route: str = "terminate"
    ) -> AttemptRow:
        """Cancel one attempt through the declared terminate route."""

        error = self.lifecycle.terminate(attempt_id, reason=reason)
        attempt = self.store.transition_attempt(
            attempt_id,
            "cancelled",
            reason=reason,
            result_payload={"reason": reason, "route": route, "terminate_error": error},
        )
        self.store.close_leases_for(attempt_id, state="cancelled", reason=reason)
        return attempt

    def _maybe_complete_group(self, group_id: str) -> GroupRow | None:
        group = self.store.group(group_id)
        assert group is not None
        if group.state != GROUP_OPEN:
            return None
        if len(self._filled_slots(group_id)) < group.cardinality:
            return None
        return self.store.transition_group(
            group_id, GROUP_COMPLETE, reason="all_slots_filled"
        )

    # -- the train-ready queue and its gate --------------------------------

    def promote_ready_groups(self) -> tuple[GroupRow, ...]:
        """Move complete groups into the bounded train-ready queue, oldest first."""

        promoted: list[GroupRow] = []
        for group in self.store.groups_in_state(GROUP_COMPLETE, run_id=self.run_id):
            if not self.has_capacity(QUEUE_TRAIN_READY):
                break
            promoted.append(
                self.store.transition_group(
                    group.group_id, GROUP_TRAIN_READY, reason="complete_group"
                )
            )
        return tuple(promoted)

    def train_dequeue(self, *, current_policy_revision: int) -> GateOutcome:
        """The dequeue gate. Staleness is rechecked here, not at submit.

        This check is the hard training guarantee: nothing leaves for a train
        step without it, and a group found stale is discarded or recycled per
        the configured disposition.
        """

        self.lifecycle.assert_can_train()
        rejected: list[GateRejection] = []
        while True:
            self.promote_ready_groups()
            queue = self.store.groups_in_state(GROUP_TRAIN_READY, run_id=self.run_id)
            if not queue:
                return GateOutcome(
                    released=None, staleness=None, members=(), rejected=tuple(rejected)
                )
            group = queue[0]
            staleness = current_policy_revision - group.policy_revision
            if staleness < 0:
                raise StalenessError(
                    f"group {group.group_id} carries policy revision "
                    f"{group.policy_revision}, ahead of the trainer's "
                    f"{current_policy_revision}"
                )
            if staleness > self.policy.max_staleness:
                rejected.append(self._reject_stale(group, staleness))
                continue
            released = self.store.transition_group(
                group.group_id,
                GROUP_TRAINED,
                reason="dequeue_gate_passed",
                detail={"staleness": staleness, "current_policy_revision": current_policy_revision},
            )
            return GateOutcome(
                released=released,
                staleness=staleness,
                members=self.group_members(group.group_id),
                rejected=tuple(rejected),
            )

    def _reject_stale(self, group: GroupRow, staleness: int) -> GateRejection:
        disposition = self.policy.stale_disposition
        target = GROUP_RECYCLED if disposition == STALE_RECYCLE else GROUP_DISCARDED
        slots = tuple(
            RecycledSlot(
                sample_index=attempt.sample_index,
                task_id=attempt.task_id,
                seed=attempt.seed,
                attempt_id=attempt.attempt_id,
                idempotency_key=attempt.idempotency_key,
            )
            for attempt in self.group_members(group.group_id)
        )
        self.store.transition_group(
            group.group_id,
            target,
            reason="stale_at_dequeue_gate",
            detail={
                "staleness": staleness,
                "max_staleness": self.policy.max_staleness,
                "disposition": disposition,
                "slots": [asdict(slot) for slot in slots],
            },
        )
        return GateRejection(
            group_id=group.group_id,
            staleness=staleness,
            disposition=disposition,
            slots=slots,
        )

    def group_members(self, group_id: str) -> tuple[AttemptRow, ...]:
        """The attempts that make up a group, in sample order."""

        members = []
        for member in self.store.group_members(group_id, active_only=True):
            members.append(self.store.attempt(member.attempt_id))
        return tuple(sorted(members, key=lambda row: row.sample_index))

    # -- lease sweeps ------------------------------------------------------

    def sweep(self, *, at: float | None = None) -> SweepReport:
        """Recover expired leases; cancel and replace stragglers."""

        recovered: list[str] = []
        cancelled: list[str] = []
        replacements: dict[str, str] = {}
        for lease in self.leases.stragglers(at=at):
            attempt = self.store.attempt(lease.attempt_id)
            replacement = self._cancel_straggler(attempt, lease)
            cancelled.append(attempt.attempt_id)
            if replacement is not None:
                replacements[attempt.attempt_id] = replacement.attempt_id
        for lease in self.leases.expired(at=at):
            attempt = self.store.attempt(lease.attempt_id)
            self.leases.mark_expired(lease.lease_id)
            self.store.transition_attempt(
                attempt.attempt_id,
                "queued",
                reason="lease_expired_recovery",
                detail={"lease_id": lease.lease_id, "dispatch_count": attempt.dispatch_count},
            )
            recovered.append(attempt.attempt_id)
        return SweepReport(
            recovered=tuple(recovered),
            cancelled=tuple(cancelled),
            replacements=replacements,
            unfillable_groups=self.unfillable_groups(),
        )

    def _cancel_straggler(self, attempt: AttemptRow, lease: LeaseRow) -> AttemptRow | None:
        policy = self.leases.straggler
        error = self.lifecycle.terminate(attempt.attempt_id, reason="straggler")
        self.leases.cancel(lease.lease_id, reason="straggler")
        self.store.transition_attempt(
            attempt.attempt_id,
            "cancelled",
            reason="straggler",
            result_payload={
                "reason": "straggler",
                "route": "terminate",
                "straggler_action": policy.action,
                "straggler_deadline": lease.straggler_deadline,
                "horizon_seconds": self.leases.horizon_seconds,
                "terminate_error": error,
            },
        )
        return self._replacement(attempt)

    def _replacement(self, attempt: AttemptRow) -> AttemptRow | None:
        if not self.leases.straggler.may_replace(attempt.replacement_index):
            return None
        index = attempt.replacement_index + 1
        group = self.store.group(attempt.group_id)
        assert group is not None
        pin = GroupPin(**dict(group.pin))
        request = AttemptRequest(
            idempotency_key=f"{attempt.idempotency_key}#r{index}",
            pin=pin,
            sample_index=attempt.sample_index,
            task_id=attempt.task_id,
            seed=attempt.seed,
            attempt_id=f"{attempt.attempt_id}#r{index}",
            agent_instance_id=attempt.agent_instance_id,
            team_id=attempt.team_id,
            metadata={**dict(attempt.metadata), "replaces": attempt.attempt_id},
        )
        return self._admit(
            request,
            replaced_attempt_id=attempt.attempt_id,
            replacement_index=index,
            bounded=False,
        )

    # -- recovery ----------------------------------------------------------

    def recover(self) -> RecoverySnapshot:
        """The durable snapshot a restarted process starts from."""

        return self.store.recover(self.run_id)


def engine_from(
    store: JournalStore,
    *,
    run_id: str,
    policy: QueuePolicy,
    leases: LeaseBook,
    lifecycle: RunLifecycle,
) -> QueueEngine:
    """Assemble an engine over an already-registered run."""

    store.lifecycle_state(run_id)
    return QueueEngine(
        store, run_id=run_id, policy=policy, leases=leases, lifecycle=lifecycle
    )


def queue_names() -> Sequence[str]:
    return QUEUES
