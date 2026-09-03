"""The four bounded queues, the dequeue gate, and lease recovery.

Time is injected everywhere: an hour-scale attempt is exercised by advancing a
manual clock, never by waiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from synth_optimizers.contracts.rl_identity import GroupPin, Horizon, MixedGroupError
from synth_optimizers.rl.leases import (
    STRAGGLER_CANCEL,
    LeaseBook,
    LeaseError,
    LeaseExpiredError,
    LeaseSizing,
    StragglerPolicy,
)
from synth_optimizers.rl.lifecycle import RunLifecycle
from synth_optimizers.rl.queues import (
    STALE_DISCARD,
    STALE_RECYCLE,
    AttemptRequest,
    DispatchRefused,
    GroupAdmissionError,
    QueueCapacities,
    QueueEngine,
    QueueFullError,
    QueuePolicy,
    QueuePolicyError,
    StalenessError,
)
from synth_optimizers.rl.store import (
    GROUP_COMPLETE,
    GROUP_DISCARDED,
    GROUP_OPEN,
    GROUP_RECYCLED,
    GROUP_TRAIN_READY,
    GROUP_TRAINED,
    LEASE_CANCELLED,
    LEASE_EXPIRED,
    QUEUE_ROLLOUT,
    QUEUE_SCORE,
    QUEUE_SCORED_RESULT,
    QUEUE_TRAIN_READY,
    JournalStore,
    ManualClock,
    RunIdentity,
    TerminalResultError,
)

RUN_ID = "run-1"
#: 30s heartbeat, two misses tolerated: silence is fatal after 90s.
HEARTBEAT_TTL = 90.0
#: One-hour horizon plus 90s of in-lease collection plus a 300s declared grace.
STRAGGLER_OFFSET = 3990.0


def make_pin(**overrides: object) -> GroupPin:
    payload: dict[str, object] = {
        "group_id": "g0",
        "run_id": RUN_ID,
        "algorithm_plan_hash": "sha256:plan",
        "behavior_fingerprint": "sha256:behavior",
        "policy_revision": 0,
        "wire_api": "chat_completions",
        "sampling_transport": "message_in_capture_out",
        "policy_kind": "declared-by-container",
        "model_family": "family-a",
        "container_image_digest": "sha256:image",
        "container_contract_hash": "sha256:contract",
        "handshake_agreement_digest": "sha256:agreement",
        "task_family": "taskset/seed-family-0",
        "cardinality": 2,
    }
    payload.update(overrides)
    return GroupPin(**payload)  # type: ignore[arg-type]


@dataclass(slots=True)
class Harness:
    clock: ManualClock
    store: JournalStore
    leases: LeaseBook
    lifecycle: RunLifecycle
    engine: QueueEngine
    terminated: list[str]


def build(
    tmp_path: Path,
    *,
    rollout: int = 8,
    score: int = 8,
    scored_result: int = 8,
    train_ready: int = 1,
    max_staleness: int = 0,
    max_in_flight: int = 8,
    max_open_groups: int = 4,
    stale_disposition: str = STALE_DISCARD,
    horizon: Horizon | None = None,
    sizing: LeaseSizing | None = None,
    straggler: StragglerPolicy | None = None,
) -> Harness:
    clock = ManualClock()
    store = JournalStore(tmp_path / "queue.sqlite3", clock=clock)
    store.register_run(
        RunIdentity(
            run_id=RUN_ID,
            container_contract_hash="sha256:contract",
            container_image_digest="sha256:image",
            algorithm_plan_hash="sha256:plan",
            renderer_fingerprint="sha256:renderer",
            handshake_agreement_digest="sha256:agreement",
            capability_hash="sha256:capabilities",
        )
    )
    terminated: list[str] = []
    lifecycle = RunLifecycle(store, RUN_ID, terminate=terminated.append, clock=clock)
    book = LeaseBook(
        store,
        horizon=horizon or Horizon(horizon_kind="wall_clock", value=3600.0, grace_seconds=300.0),
        sizing=sizing
        or LeaseSizing(
            heartbeat_interval_seconds=30.0,
            missed_heartbeats_allowed=2,
            quiescence_seconds=60.0,
            artifact_collection_seconds=30.0,
        ),
        straggler=straggler or StragglerPolicy(),
        clock=clock,
    )
    policy = QueuePolicy(
        capacities=QueueCapacities(
            rollout=rollout, score=score, scored_result=scored_result, train_ready=train_ready
        ),
        max_staleness=max_staleness,
        max_in_flight=max_in_flight,
        max_open_groups=max_open_groups,
        stale_disposition=stale_disposition,
    )
    engine = QueueEngine(store, run_id=RUN_ID, policy=policy, leases=book, lifecycle=lifecycle)
    return Harness(clock, store, book, lifecycle, engine, terminated)


def request(pin: GroupPin, index: int) -> AttemptRequest:
    return AttemptRequest(
        idempotency_key=f"key:{pin.group_id}:{index}",
        pin=pin,
        sample_index=index,
        task_id=f"row-{index}",
        seed=1000 + index,
    )


def admit_group(harness: Harness, pin: GroupPin) -> tuple[str, ...]:
    return tuple(
        harness.engine.admit(request(pin, index)).attempt_id
        for index in range(pin.cardinality)
    )


def finish(harness: Harness, attempt_id: str, *, reward: float = 1.0) -> None:
    harness.engine.dispatch(attempt_id, holder="worker-0")
    harness.engine.report_scored(attempt_id, payload={"reward": reward})
    harness.engine.accept_evidence(attempt_id, payload={"reward": reward})


def complete_group(harness: Harness, pin: GroupPin) -> tuple[str, ...]:
    attempts = admit_group(harness, pin)
    for attempt_id in attempts:
        finish(harness, attempt_id)
    return attempts


# -- configuration ---------------------------------------------------------


def test_pipeline_lag_may_not_exceed_the_staleness_bound() -> None:
    with pytest.raises(QueuePolicyError) as error:
        QueuePolicy(
            capacities=QueueCapacities(rollout=4, score=4, scored_result=4, train_ready=3),
            max_staleness=1,
            max_in_flight=4,
            max_open_groups=4,
        )
    assert "max_staleness >= 2" in str(error.value)
    QueuePolicy(
        capacities=QueueCapacities(rollout=4, score=4, scored_result=4, train_ready=3),
        max_staleness=2,
        max_in_flight=4,
        max_open_groups=4,
    )


def test_unknown_stale_disposition_is_refused() -> None:
    with pytest.raises(QueuePolicyError):
        QueuePolicy(
            capacities=QueueCapacities(rollout=1, score=1, scored_result=1, train_ready=1),
            max_staleness=0,
            max_in_flight=1,
            max_open_groups=1,
            stale_disposition="average_it_anyway",
        )


# -- admission -------------------------------------------------------------


def test_admission_is_per_sample_and_a_retry_returns_the_same_attempt(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    pin = make_pin(cardinality=4)
    first = harness.engine.admit(request(pin, 0))
    again = harness.engine.admit(request(pin, 0))
    assert again.attempt_id == first.attempt_id
    assert harness.engine.depth(QUEUE_ROLLOUT) == 1
    assert harness.engine.group_progress("g0") == (0, 4)
    with pytest.raises(GroupAdmissionError):
        harness.engine.admit(request(pin, 4))
    with pytest.raises(GroupAdmissionError):
        harness.engine.admit(
            AttemptRequest(
                idempotency_key="key:g0:0",
                pin=pin,
                sample_index=1,
                task_id="row-1",
                seed=1,
            )
        )
    with pytest.raises(GroupAdmissionError):
        harness.engine.admit(
            AttemptRequest(
                idempotency_key="another-key",
                pin=pin,
                sample_index=0,
                task_id="row-0",
                seed=0,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_revision", 7),
        ("behavior_fingerprint", "sha256:other-behavior"),
        ("container_image_digest", "sha256:other-image"),
        ("algorithm_plan_hash", "sha256:other-plan"),
        ("wire_api", "responses"),
        ("container_contract_hash", "sha256:other-contract"),
        ("task_family", "taskset/seed-family-1"),
        ("topology_id", "topology-b"),
    ],
)
def test_a_mixed_group_is_rejected_on_every_pinned_field(
    tmp_path: Path, field: str, value: object
) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    harness.engine.admit(request(pin, 0))
    with pytest.raises(MixedGroupError) as error:
        harness.engine.admit(request(make_pin(**{field: value}), 1))
    assert field in str(error.value)
    assert harness.engine.group_progress("g0") == (0, 2)


def test_a_group_may_not_mix_cardinality_or_run_id(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.engine.admit(request(make_pin(), 0))
    with pytest.raises(MixedGroupError) as error:
        harness.engine.admit(request(make_pin(cardinality=3), 1))
    assert "cardinality" in str(error.value)
    with pytest.raises(MixedGroupError) as error:
        harness.engine.admit(request(make_pin(run_id="run-2"), 1))
    assert "run_id" in str(error.value)


def test_a_bounded_rollout_queue_applies_backpressure(tmp_path: Path) -> None:
    harness = build(tmp_path, rollout=2)
    pin = make_pin(cardinality=4)
    harness.engine.admit(request(pin, 0))
    harness.engine.admit(request(pin, 1))
    with pytest.raises(QueueFullError) as error:
        harness.engine.admit(request(pin, 2))
    assert "rollout queue is at capacity 2" in str(error.value)
    harness.engine.dispatch("g0:s0", holder="worker-0")
    assert harness.engine.admit(request(pin, 2)).attempt_id == "g0:s2"


def test_the_open_group_bound_applies_backpressure(tmp_path: Path) -> None:
    harness = build(tmp_path, max_open_groups=2)
    harness.engine.admit(request(make_pin(group_id="g0"), 0))
    harness.engine.admit(request(make_pin(group_id="g1"), 0))
    with pytest.raises(QueueFullError) as error:
        harness.engine.admit(request(make_pin(group_id="g2"), 0))
    assert "2 groups are already open" in str(error.value)


# -- dispatch --------------------------------------------------------------


def test_partial_groups_are_preferred_to_completion(tmp_path: Path) -> None:
    harness = build(tmp_path, max_open_groups=3)
    older = make_pin(group_id="g0")
    newer = make_pin(group_id="g1")
    harness.engine.admit(request(older, 0))
    harness.engine.admit(request(newer, 0))
    harness.engine.admit(request(newer, 1))
    harness.engine.admit(request(older, 1))
    assert [row.attempt_id for row in harness.engine.next_dispatch(limit=4)] == [
        "g0:s0",
        "g0:s1",
        "g1:s0",
        "g1:s1",
    ]
    finish(harness, "g0:s0")
    assert harness.engine.group_progress("g0") == (1, 2)
    assert harness.store.group("g0").state == GROUP_OPEN
    # The oldest open group is still preferred while it has an unfilled slot.
    assert harness.engine.next_dispatch(limit=1)[0].attempt_id == "g0:s1"
    finish(harness, "g0:s1")
    assert harness.store.group("g0").state == GROUP_TRAIN_READY
    assert [row.attempt_id for row in harness.engine.next_dispatch(limit=2)] == [
        "g1:s0",
        "g1:s1",
    ]


def test_advertised_concurrency_throttles_dispatch(tmp_path: Path) -> None:
    harness = build(tmp_path, max_in_flight=1)
    pin = make_pin(cardinality=2)
    admit_group(harness, pin)
    harness.engine.dispatch("g0:s0", holder="worker-0")
    assert harness.engine.next_dispatch(limit=2) == ()
    with pytest.raises(DispatchRefused):
        harness.engine.dispatch("g0:s1", holder="worker-1")
    harness.engine.report_scored("g0:s0")
    assert [row.attempt_id for row in harness.engine.next_dispatch(limit=2)] == ["g0:s1"]


def test_a_full_score_queue_throttles_dispatch_rather_than_refusing_executed_work(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path, score=1, scored_result=1)
    pin = make_pin(cardinality=3)
    for index in range(3):
        harness.engine.admit(request(pin, index))
    harness.engine.dispatch("g0:s0", holder="worker-0")
    # The attempt already ran, so the score queue must accept it even at capacity.
    harness.engine.report_awaiting_score("g0:s0")
    assert harness.engine.depth(QUEUE_SCORE) == 1
    assert harness.engine.next_dispatch(limit=2) == ()
    harness.engine.report_scored("g0:s0")
    assert harness.engine.depth(QUEUE_SCORED_RESULT) == 1
    assert harness.engine.next_dispatch(limit=2) == ()
    harness.engine.accept_evidence("g0:s0", payload={"reward": 1.0})
    assert [row.attempt_id for row in harness.engine.next_dispatch(limit=1)] == ["g0:s1"]


# -- terminal results ------------------------------------------------------


def test_exactly_one_terminal_result_per_accepted_attempt(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    finish(harness, "g0:s0")
    with pytest.raises(TerminalResultError):
        harness.engine.reject_evidence("g0:s0", reason="absent_reward")
    with pytest.raises(TerminalResultError):
        harness.engine.cancel("g0:s0", reason="stop")
    assert harness.store.result("g0:s0").kind == "episode"
    harness.engine.dispatch("g0:s1", holder="worker-0")
    harness.engine.report_scored("g0:s1")
    harness.engine.reject_evidence("g0:s1", reason="absent_reward")
    assert harness.store.result("g0:s1").kind == "failure"
    # A group with a failed member cannot complete, and says so.
    assert harness.store.group("g0").state == GROUP_OPEN
    assert harness.engine.unfillable_groups() == ("g0",)


def test_a_group_completes_only_when_every_slot_is_filled(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    finish(harness, "g0:s0")
    assert harness.store.group("g0").state == GROUP_OPEN
    assert harness.engine.depth(QUEUE_TRAIN_READY) == 0
    finish(harness, "g0:s1")
    assert harness.store.group("g0").state == GROUP_TRAIN_READY
    assert harness.engine.depth(QUEUE_TRAIN_READY) == 1


# -- leases ----------------------------------------------------------------


def test_an_hour_scale_attempt_stays_alive_on_heartbeats(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    _attempt, lease = harness.engine.dispatch("g0:s0", holder="worker-0")
    assert lease.expires_at == HEARTBEAT_TTL
    assert lease.straggler_deadline == STRAGGLER_OFFSET
    for _tick in range(120):  # one hour of 30s heartbeats
        harness.clock.advance(30.0)
        harness.engine.heartbeat("g0:s0")
    assert harness.clock.now() == 3600.0
    assert harness.leases.expired() == ()
    assert harness.leases.stragglers() == ()
    assert harness.engine.sweep().recovered == ()
    assert harness.store.attempt("g0:s0").state == "running"


def test_a_step_horizon_is_sized_from_its_declared_conversion() -> None:
    sizing = LeaseSizing(heartbeat_interval_seconds=10.0)
    # The container's declaration is authoritative for a step or tick horizon.
    assert (
        sizing.horizon_seconds(
            Horizon(horizon_kind="steps", value=500.0, seconds_per_unit=4.0)
        )
        == 2000.0
    )
    assert sizing.horizon_seconds(Horizon(horizon_kind="wall_clock", value=3600.0)) == 3600.0
    # A caller that measured the substrate may override, but never infer.
    override = LeaseSizing(heartbeat_interval_seconds=10.0, seconds_per_unit=2.0)
    assert (
        override.horizon_seconds(
            Horizon(horizon_kind="steps", value=500.0, seconds_per_unit=4.0)
        )
        == 1000.0
    )
    assert (
        override.horizon_seconds(
            Horizon(horizon_kind="env_ticks", value=500.0, time_dilation=0.5)
        )
        == 500.0
    )
    with pytest.raises(LeaseError) as error:
        sizing.seconds_per_unit_for(_UndeclaredHorizon())
    assert "seconds_per_unit" in str(error.value)


@dataclass(frozen=True, slots=True)
class _UndeclaredHorizon:
    """A horizon that declares no conversion at all: lease sizing must refuse."""

    horizon_kind: str = "steps"
    value: float = 500.0
    time_dilation: float = 1.0
    grace_seconds: float = 0.0


def test_an_expired_lease_recovers_the_same_logical_attempt(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    _attempt, lease = harness.engine.dispatch("g0:s0", holder="worker-0")
    harness.clock.advance(HEARTBEAT_TTL + 10.0)
    with pytest.raises(LeaseExpiredError):
        harness.engine.heartbeat("g0:s0")
    report = harness.engine.sweep()
    assert report.recovered == ("g0:s0",)
    assert report.cancelled == ()
    recovered = harness.store.attempt("g0:s0")
    assert recovered.state == "queued"
    assert recovered.queue == QUEUE_ROLLOUT
    assert recovered.dispatch_count == 1
    assert harness.store.lease(lease.lease_id).state == LEASE_EXPIRED
    assert len(harness.store.attempts_in_group("g0")) == 2
    assert harness.store.result("g0:s0") is None
    # Redispatch is the same attempt with a second lease, not a second attempt.
    _again, second = harness.engine.dispatch("g0:s0", holder="worker-1")
    assert second.lease_id == "g0:s0#l2"
    assert harness.store.attempt("g0:s0").dispatch_count == 2


def test_a_straggler_is_cancelled_and_replaced_with_the_replacement_recorded(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    _attempt, lease = harness.engine.dispatch("g0:s0", holder="worker-0")
    harness.clock.advance(STRAGGLER_OFFSET + 1.0)
    report = harness.engine.sweep()
    assert report.cancelled == ("g0:s0",)
    assert report.recovered == ()
    assert report.replacements == {"g0:s0": "g0:s0#r1"}
    assert harness.terminated == ["g0:s0"]
    cancelled = harness.store.attempt("g0:s0")
    assert cancelled.state == "cancelled"
    result = harness.store.result("g0:s0")
    assert result.kind == "cancellation"
    assert result.payload["straggler_action"] == "cancel_and_replace"
    assert harness.store.lease(lease.lease_id).state == LEASE_CANCELLED
    replacement = harness.store.attempt("g0:s0#r1")
    assert replacement.replaced_attempt_id == "g0:s0"
    assert replacement.replacement_index == 1
    assert (replacement.sample_index, replacement.task_id, replacement.seed) == (0, "row-0", 1000)
    assert replacement.state == "queued"
    membership = [
        (row.attempt_id, row.active, row.disposition)
        for row in harness.store.group_members("g0")
        if row.sample_index == 0
    ]
    assert membership == [("g0:s0", False, "replaced"), ("g0:s0#r1", True, "held")]
    # The replacement finishes the group; membership keeps both, so cost is attributable.
    finish(harness, "g0:s0#r1")
    finish(harness, "g0:s1")
    assert harness.store.group("g0").state == GROUP_TRAIN_READY
    assert len(harness.store.group_members("g0")) == 3


def test_a_replacement_budget_of_zero_cancels_without_replacing(tmp_path: Path) -> None:
    harness = build(tmp_path, straggler=StragglerPolicy(action=STRAGGLER_CANCEL))
    pin = make_pin()
    admit_group(harness, pin)
    harness.engine.dispatch("g0:s0", holder="worker-0")
    harness.clock.advance(STRAGGLER_OFFSET + 1.0)
    report = harness.engine.sweep()
    assert report.cancelled == ("g0:s0",)
    assert report.replacements == {}
    assert report.unfillable_groups == ("g0",)
    assert harness.store.result("g0:s0").payload["straggler_action"] == STRAGGLER_CANCEL


def test_a_straggler_is_cancelled_even_while_heartbeating(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin()
    admit_group(harness, pin)
    harness.engine.dispatch("g0:s0", holder="worker-0")
    while harness.clock.now() < STRAGGLER_OFFSET:
        harness.clock.advance(30.0)
        harness.engine.heartbeat("g0:s0")
    assert harness.leases.expired() == ()
    assert [row.attempt_id for row in harness.leases.stragglers()] == ["g0:s0"]
    assert harness.engine.sweep().cancelled == ("g0:s0",)


# -- the dequeue gate ------------------------------------------------------


def test_a_stale_group_is_discarded_at_the_train_dequeue_while_a_fresh_one_passes(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path, train_ready=2, max_staleness=1, max_open_groups=4)
    complete_group(harness, make_pin(group_id="g0", policy_revision=0))
    complete_group(harness, make_pin(group_id="g1", policy_revision=2))
    assert harness.engine.depth(QUEUE_TRAIN_READY) == 2
    outcome = harness.engine.train_dequeue(current_policy_revision=2)
    assert [rejection.group_id for rejection in outcome.rejected] == ["g0"]
    assert outcome.rejected[0].staleness == 2
    assert outcome.rejected[0].disposition == STALE_DISCARD
    assert harness.store.group("g0").state == GROUP_DISCARDED
    assert outcome.released is not None
    assert outcome.released.group_id == "g1"
    assert outcome.staleness == 0
    assert [row.attempt_id for row in outcome.members] == ["g1:s0", "g1:s1"]
    assert harness.store.group("g1").state == GROUP_TRAINED
    assert harness.engine.depth(QUEUE_TRAIN_READY) == 0
    empty = harness.engine.train_dequeue(current_policy_revision=2)
    assert (empty.released, empty.rejected) == (None, ())


def test_a_stale_group_is_recycled_when_the_policy_says_so(tmp_path: Path) -> None:
    harness = build(tmp_path, max_staleness=0, stale_disposition=STALE_RECYCLE)
    complete_group(harness, make_pin(group_id="g0", policy_revision=0))
    outcome = harness.engine.train_dequeue(current_policy_revision=1)
    assert outcome.released is None
    rejection = outcome.rejected[0]
    assert rejection.disposition == STALE_RECYCLE
    assert harness.store.group("g0").state == GROUP_RECYCLED
    assert [(slot.sample_index, slot.task_id, slot.seed) for slot in rejection.slots] == [
        (0, "row-0", 1000),
        (1, "row-1", 1001),
    ]
    assert [slot.idempotency_key for slot in rejection.slots] == ["key:g0:0", "key:g0:1"]
    # The recycled slots come back under a fresh pin, as a new group.
    fresh = make_pin(group_id="g0-r1", policy_revision=1)
    for slot in rejection.slots:
        harness.engine.admit(
            AttemptRequest(
                idempotency_key=f"{slot.idempotency_key}#recycled",
                pin=fresh,
                sample_index=slot.sample_index,
                task_id=slot.task_id,
                seed=slot.seed,
            )
        )
    assert harness.engine.group_progress("g0-r1") == (0, 2)


def test_a_group_ahead_of_the_trainer_is_an_error_not_a_discard(tmp_path: Path) -> None:
    harness = build(tmp_path)
    complete_group(harness, make_pin(group_id="g0", policy_revision=3))
    with pytest.raises(StalenessError):
        harness.engine.train_dequeue(current_policy_revision=1)


def test_production_continues_while_scoring_and_training_run(tmp_path: Path) -> None:
    harness = build(tmp_path, train_ready=1, max_staleness=0, max_open_groups=4, rollout=8)
    complete_group(harness, make_pin(group_id="g0"))
    complete_group(harness, make_pin(group_id="g1"))
    # The train-ready queue is bounded, so the second complete group waits there
    # instead of stopping production.
    assert harness.engine.depth(QUEUE_TRAIN_READY) == 1
    assert harness.store.group("g1").state == GROUP_COMPLETE
    producing = make_pin(group_id="g2")
    admit_group(harness, producing)
    harness.engine.dispatch("g2:s0", holder="worker-0")
    outcome = harness.engine.train_dequeue(current_policy_revision=0)
    assert outcome.released.group_id == "g0"
    # Training released a slot; the waiting group promotes and production is untouched.
    assert harness.engine.promote_ready_groups()[0].group_id == "g1"
    assert harness.store.attempt("g2:s0").state == "running"
    assert [row.attempt_id for row in harness.engine.next_dispatch(limit=1)] == ["g2:s1"]
    harness.engine.dispatch("g2:s1", holder="worker-1")
    assert harness.engine.train_dequeue(current_policy_revision=0).released.group_id == "g1"


def test_restart_recovery_resumes_the_engine_from_the_journal(tmp_path: Path) -> None:
    harness = build(tmp_path, train_ready=2, max_staleness=1, max_open_groups=4)
    complete_group(harness, make_pin(group_id="g0", policy_revision=0))
    pin = make_pin(group_id="g1", policy_revision=1)
    admit_group(harness, pin)
    harness.engine.dispatch("g1:s0", holder="worker-0")
    snapshot = harness.engine.recover()
    assert [row.group_id for row in snapshot.train_ready] == ["g0"]
    assert [row.attempt_id for row in snapshot.active] == ["g1:s0"]
    assert [row.attempt_id for row in snapshot.queued] == ["g1:s1"]
    harness.store.close()

    resumed = build(tmp_path, train_ready=2, max_staleness=1, max_open_groups=4)
    after = resumed.engine.recover()
    assert [row.group_id for row in after.train_ready] == ["g0"]
    assert [row.attempt_id for row in after.active] == ["g1:s0"]
    assert [row.attempt_id for row in after.queued] == ["g1:s1"]
    assert [row.attempt_id for row in after.live_leases] == ["g1:s0"]
    outcome = resumed.engine.train_dequeue(current_policy_revision=1)
    assert outcome.released.group_id == "g0"
    assert outcome.staleness == 1
