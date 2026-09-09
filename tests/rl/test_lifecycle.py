"""Pause, drain, resume and stop, checked at every queue boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from synth_optimizers.contracts.rl_identity import GroupPin, Horizon
from synth_optimizers.rl.leases import LeaseBook, LeaseSizing, StragglerPolicy
from synth_optimizers.rl.lifecycle import (
    GATES,
    AdmissionClosed,
    DispatchClosed,
    LifecycleError,
    LifecycleTransitionError,
    ResumeRefused,
    RunLifecycle,
    ScoringClosed,
    TrainStepBlocked,
    gates_for,
)
from synth_optimizers.rl.queues import (
    AttemptRequest,
    QueueCapacities,
    QueueEngine,
    QueuePolicy,
)
from synth_optimizers.rl.store import (
    GROUP_ABANDONED,
    GROUP_TRAINED,
    LEASE_ACTIVE,
    LEASE_CANCELLED,
    LIFECYCLE_ADMITTING,
    LIFECYCLE_DRAINED,
    LIFECYCLE_DRAINING,
    LIFECYCLE_PAUSED,
    LIFECYCLE_STATES,
    LIFECYCLE_STOPPED,
    JournalStore,
    ManualClock,
    RunIdentity,
)

RUN_ID = "run-1"


def identity(**overrides: str) -> RunIdentity:
    payload = {
        "run_id": RUN_ID,
        "container_contract_hash": "sha256:contract",
        "container_image_digest": "sha256:image",
        "algorithm_plan_hash": "sha256:plan",
        "renderer_fingerprint": "sha256:renderer",
        "handshake_agreement_digest": "sha256:agreement",
        "capability_hash": "sha256:capabilities",
    }
    payload.update(overrides)
    return RunIdentity(**payload)


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
    lifecycle: RunLifecycle
    engine: QueueEngine
    terminated: list[str]


def build(
    tmp_path: Path,
    *,
    rehandshake: object | None = None,
    terminate: object | None = None,
    train_ready: int = 2,
    max_staleness: int = 1,
) -> Harness:
    clock = ManualClock()
    store = JournalStore(tmp_path / "queue.sqlite3", clock=clock)
    store.register_run(identity())
    terminated: list[str] = []

    def default_terminate(attempt_id: str) -> None:
        terminated.append(attempt_id)

    lifecycle = RunLifecycle(
        store,
        RUN_ID,
        terminate=terminate or default_terminate,  # type: ignore[arg-type]
        rehandshake=rehandshake,  # type: ignore[arg-type]
        clock=clock,
    )
    leases = LeaseBook(
        store,
        horizon=Horizon(horizon_kind="wall_clock", value=3600.0, grace_seconds=300.0),
        sizing=LeaseSizing(heartbeat_interval_seconds=30.0),
        straggler=StragglerPolicy(),
        clock=clock,
    )
    policy = QueuePolicy(
        capacities=QueueCapacities(
            rollout=8, score=8, scored_result=8, train_ready=train_ready
        ),
        max_staleness=max_staleness,
        max_in_flight=8,
        max_open_groups=4,
    )
    engine = QueueEngine(store, run_id=RUN_ID, policy=policy, leases=leases, lifecycle=lifecycle)
    return Harness(clock, store, lifecycle, engine, terminated)


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


def finish(harness: Harness, attempt_id: str) -> None:
    harness.engine.report_scored(attempt_id, payload={"reward": 1.0})
    harness.engine.accept_evidence(attempt_id, payload={"reward": 1.0})


# -- the gate matrix -------------------------------------------------------


def test_every_lifecycle_state_declares_its_gates() -> None:
    assert set(GATES) == set(LIFECYCLE_STATES)
    assert gates_for(LIFECYCLE_ADMITTING) == GATES[LIFECYCLE_ADMITTING]
    with pytest.raises(LifecycleError):
        gates_for("hibernating")


@pytest.mark.parametrize(
    ("state", "admit", "dispatch", "score", "train"),
    [
        (LIFECYCLE_ADMITTING, True, True, True, True),
        (LIFECYCLE_PAUSED, False, False, True, False),
        (LIFECYCLE_DRAINING, False, False, True, True),
        (LIFECYCLE_DRAINED, False, False, False, False),
        (LIFECYCLE_STOPPED, False, False, False, False),
    ],
)
def test_the_gates_at_each_boundary_are_declared(
    state: str, admit: bool, dispatch: bool, score: bool, train: bool
) -> None:
    gates = gates_for(state)
    assert (gates.admit, gates.dispatch, gates.score, gates.train) == (
        admit,
        dispatch,
        score,
        train,
    )


# -- pause -----------------------------------------------------------------


def test_pause_stops_admission_and_dispatch_but_keeps_leases_and_scoring(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    pin = make_pin(cardinality=3)
    for index in range(3):
        harness.engine.admit(request(pin, index))
    _attempt, lease = harness.engine.dispatch("g0:s0", holder="worker-0")

    assert harness.lifecycle.pause(reason="provider_quota") == LIFECYCLE_PAUSED
    assert harness.lifecycle.pause() == LIFECYCLE_PAUSED  # idempotent

    with pytest.raises(AdmissionClosed):
        harness.engine.admit(
            AttemptRequest(
                idempotency_key="key:g1:0",
                pin=make_pin(group_id="g1"),
                sample_index=0,
                task_id="row-0",
                seed=1,
            )
        )
    with pytest.raises(DispatchClosed):
        harness.engine.dispatch("g0:s1", holder="worker-1")
    assert harness.engine.next_dispatch(limit=2) == ()

    # The in-flight attempt keeps its lease and runs to a terminal result.
    assert harness.store.lease(lease.lease_id).state == LEASE_ACTIVE
    harness.clock.advance(30.0)
    harness.engine.heartbeat("g0:s0")
    finish(harness, "g0:s0")
    assert harness.store.result("g0:s0").kind == "episode"

    # A new train step is refused while paused.
    with pytest.raises(TrainStepBlocked):
        harness.engine.train_dequeue(current_policy_revision=0)

    # A retry of an already-admitted key is not new admission, so it still works.
    assert harness.engine.admit(request(pin, 1)).attempt_id == "g0:s1"


def test_pause_is_refused_from_a_stopped_or_draining_run(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.lifecycle.drain()
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.pause()
    harness.lifecycle.stop()
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.pause()


# -- drain -----------------------------------------------------------------


def test_drain_finishes_in_flight_work_trains_complete_groups_then_stops(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    full = make_pin(group_id="g0")
    partial = make_pin(group_id="g1")
    admit_group(harness, full)
    admit_group(harness, partial)
    for attempt_id in ("g0:s0", "g0:s1", "g1:s0"):
        harness.engine.dispatch(attempt_id, holder="worker-0")

    assert harness.lifecycle.drain(reason="cost_ceiling") == LIFECYCLE_DRAINING
    assert harness.lifecycle.drain() == LIFECYCLE_DRAINING  # idempotent
    with pytest.raises(AdmissionClosed):
        harness.engine.admit(request(make_pin(group_id="g2"), 0))
    with pytest.raises(LifecycleTransitionError) as error:
        harness.lifecycle.finish_drain()
    assert "in_flight" in str(error.value)

    finish(harness, "g0:s0")
    finish(harness, "g0:s1")
    finish(harness, "g1:s0")
    with pytest.raises(LifecycleTransitionError) as error:
        harness.lifecycle.finish_drain()
    assert "train_ready_groups" in str(error.value)

    released = harness.engine.train_dequeue(current_policy_revision=0)
    assert released.released.group_id == "g0"
    assert harness.lifecycle.outstanding_drain_work() == {
        "in_flight": (),
        "scored": (),
        "complete_groups": (),
        "train_ready_groups": (),
    }

    report = harness.lifecycle.finish_drain()
    assert harness.lifecycle.state == LIFECYCLE_DRAINED
    assert report.cancelled_attempts == ("g1:s1",)
    assert [group.group_id for group in report.abandoned_groups] == ["g1"]
    membership = report.abandoned_groups[0].membership
    assert [(row["attempt_id"], row["result_kind"]) for row in membership] == [
        ("g1:s0", "episode"),
        ("g1:s1", "cancellation"),
    ]
    assert harness.store.group("g0").state == GROUP_TRAINED
    assert harness.store.group("g1").state == GROUP_ABANDONED
    assert harness.store.attempts_without_result(run_id=RUN_ID) == ()
    with pytest.raises(ScoringClosed):
        harness.engine.report_scored("g1:s1")
    with pytest.raises(TrainStepBlocked):
        harness.engine.train_dequeue(current_policy_revision=0)
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.finish_drain()


def test_finish_drain_is_refused_when_the_run_is_not_draining(tmp_path: Path) -> None:
    harness = build(tmp_path)
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.finish_drain()


# -- resume ----------------------------------------------------------------


def test_resume_requires_a_rehandshake_hook(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.lifecycle.pause()
    with pytest.raises(LifecycleError) as error:
        harness.lifecycle.resume()
    assert "re-handshake hook" in str(error.value)
    assert harness.lifecycle.state == LIFECYCLE_PAUSED


def test_resume_is_only_legal_from_a_paused_run(tmp_path: Path) -> None:
    harness = build(tmp_path, rehandshake=identity)
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.resume()
    harness.lifecycle.drain()
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.resume()


def test_resume_re_admits_work_after_the_agreement_is_verified(tmp_path: Path) -> None:
    calls: list[int] = []

    def rehandshake() -> RunIdentity:
        calls.append(1)
        return identity()

    harness = build(tmp_path, rehandshake=rehandshake)
    pin = make_pin()
    admit_group(harness, pin)
    harness.lifecycle.pause()
    assert harness.lifecycle.resume() == LIFECYCLE_ADMITTING
    assert calls == [1]
    harness.engine.dispatch("g0:s0", holder="worker-0")
    assert harness.engine.admit(request(make_pin(group_id="g1"), 0)).attempt_id == "g1:s0"
    events = [
        (row.subject, row.to_state, row.reason)
        for row in harness.store.lifecycle_events(RUN_ID)
    ]
    assert events == [
        ("pause", LIFECYCLE_PAUSED, ""),
        ("resume", LIFECYCLE_ADMITTING, "rehandshake_verified"),
    ]


@pytest.mark.parametrize(
    "field",
    [
        "container_contract_hash",
        "container_image_digest",
        "algorithm_plan_hash",
        "renderer_fingerprint",
        "handshake_agreement_digest",
        "capability_hash",
    ],
)
def test_resume_is_refused_when_the_binding_changed(tmp_path: Path, field: str) -> None:
    def rehandshake() -> RunIdentity:
        return identity(**{field: "sha256:changed"})

    harness = build(tmp_path, rehandshake=rehandshake)
    harness.lifecycle.pause()
    with pytest.raises(ResumeRefused) as error:
        harness.lifecycle.resume()
    assert error.value.changed_fields == (field,)
    assert "lineage edge" in str(error.value)
    assert harness.lifecycle.state == LIFECYCLE_PAUSED
    with pytest.raises(AdmissionClosed):
        harness.engine.admit(request(make_pin(), 0))
    refusal = harness.store.lifecycle_events(RUN_ID)[-1]
    assert (refusal.subject, refusal.to_state, refusal.reason) == ("resume", None, "refused")
    assert refusal.detail["changed_fields"] == [field]


def test_resume_is_refused_when_the_run_id_changed(tmp_path: Path) -> None:
    harness = build(tmp_path, rehandshake=lambda: identity(run_id="run-2"))
    harness.lifecycle.pause()
    with pytest.raises(ResumeRefused) as error:
        harness.lifecycle.resume()
    assert error.value.changed_fields == ("run_id",)


# -- stop ------------------------------------------------------------------


def test_stop_cancels_through_the_terminate_route_and_leaves_receipts_complete(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    admit_group(harness, make_pin(group_id="g0"))
    admit_group(harness, make_pin(group_id="g1"))
    _attempt, lease = harness.engine.dispatch("g0:s0", holder="worker-0")
    harness.engine.dispatch("g1:s0", holder="worker-1")
    harness.engine.report_awaiting_score("g1:s0")

    report = harness.lifecycle.stop(reason="operator_stop")
    assert harness.lifecycle.state == LIFECYCLE_STOPPED
    assert report.from_state == LIFECYCLE_ADMITTING
    assert harness.terminated == ["g0:s0", "g1:s0"]
    assert set(report.cancelled_attempts) == {"g0:s0", "g0:s1", "g1:s0", "g1:s1"}
    assert lease.lease_id in report.closed_leases
    assert harness.store.lease(lease.lease_id).state == LEASE_CANCELLED
    assert harness.store.active_leases() == ()
    assert harness.store.attempts_without_result(run_id=RUN_ID) == ()
    for attempt_id in report.cancelled_attempts:
        assert harness.store.result(attempt_id).kind == "cancellation"
    assert {group.group_id for group in report.abandoned_groups} == {"g0", "g1"}
    assert harness.lifecycle.abandoned_groups() == ("g0", "g1")
    assert len(report.abandoned_groups[0].membership) == 2
    assert report.terminate_failures == {}
    with pytest.raises(AdmissionClosed):
        harness.engine.admit(request(make_pin(group_id="g2"), 0))
    with pytest.raises(LifecycleTransitionError):
        harness.lifecycle.stop()


def test_stop_records_a_failed_terminate_route_and_still_closes_the_receipts(
    tmp_path: Path,
) -> None:
    def terminate(attempt_id: str) -> None:
        raise TimeoutError(f"no route to {attempt_id}")

    harness = build(tmp_path, terminate=terminate)
    admit_group(harness, make_pin(group_id="g0"))
    harness.engine.dispatch("g0:s0", holder="worker-0")
    report = harness.lifecycle.stop()
    assert "no route to g0:s0" in report.terminate_failures["g0:s0"]
    assert harness.store.attempts_without_result(run_id=RUN_ID) == ()
    assert harness.store.result("g0:s0").payload["route"] == "terminate"
    failures = [
        row for row in harness.store.lifecycle_events(RUN_ID) if row.subject == "terminate_failed"
    ]
    assert failures[0].detail["attempt_id"] == "g0:s0"


def test_stop_after_a_pause_leaves_the_train_ready_group_abandoned(tmp_path: Path) -> None:
    harness = build(tmp_path)
    pin = make_pin(group_id="g0")
    admit_group(harness, pin)
    for attempt_id in ("g0:s0", "g0:s1"):
        harness.engine.dispatch(attempt_id, holder="worker-0")
        finish(harness, attempt_id)
    assert harness.store.group("g0").state == "train_ready"
    harness.lifecycle.pause()
    report = harness.lifecycle.stop(reason="cost_ceiling")
    assert [group.state_before for group in report.abandoned_groups] == ["train_ready"]
    assert harness.store.group("g0").state == GROUP_ABANDONED
    assert report.cancelled_attempts == ()
    assert harness.store.attempts_without_result(run_id=RUN_ID) == ()


def test_lifecycle_state_survives_a_restart(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.lifecycle.pause(reason="redeploy")
    harness.store.close()
    resumed = build(tmp_path, rehandshake=identity)
    assert resumed.lifecycle.state == LIFECYCLE_PAUSED
    with pytest.raises(AdmissionClosed):
        resumed.engine.admit(request(make_pin(), 0))
    assert resumed.lifecycle.resume() == LIFECYCLE_ADMITTING
    assert resumed.engine.admit(request(make_pin(), 0)).attempt_id == "g0:s0"
