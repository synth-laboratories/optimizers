"""The durable journal: idempotency, one terminal result, restart recovery.

Every clock here is injected. Nothing sleeps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_optimizers.rl.store import (
    GROUP_ABANDONED,
    GROUP_COMPLETE,
    GROUP_TRAIN_READY,
    LEASE_ACTIVE,
    LEASE_EXPIRED,
    LEASE_RELEASED,
    LIFECYCLE_ADMITTING,
    LIFECYCLE_PAUSED,
    MEMBERSHIP_HELD,
    MEMBERSHIP_REPLACED,
    QUEUE_ROLLOUT,
    QUEUE_SCORE,
    QUEUE_SCORED_RESULT,
    QUEUE_TRAIN_READY,
    IdempotencyError,
    JournalStore,
    LeaseStoreError,
    ManualClock,
    RunIdentity,
    StoreError,
    TerminalResultError,
    TransitionError,
    UnknownAttemptError,
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


def opened(store: JournalStore, group_id: str, cardinality: int, revision: int = 0) -> None:
    store.open_group(
        group_id=group_id,
        run_id=RUN_ID,
        cardinality=cardinality,
        pin={"group_id": group_id, "policy_revision": revision},
        pin_digest=f"pin:{group_id}",
        policy_revision=revision,
    )


def admit(store: JournalStore, group_id: str, index: int, **overrides: object) -> str:
    attempt_id = overrides.pop("attempt_id", f"{group_id}:s{index}")
    key = overrides.pop("idempotency_key", f"key:{group_id}:{index}")
    row, _created = store.admit_attempt(
        attempt_id=str(attempt_id),
        idempotency_key=str(key),
        run_id=RUN_ID,
        group_id=group_id,
        sample_index=index,
        task_id=f"row-{index}",
        seed=1000 + index,
        policy_revision=0,
        **overrides,  # type: ignore[arg-type]
    )
    return row.attempt_id


@pytest.fixture()
def store(tmp_path: Path) -> JournalStore:
    journal = JournalStore(tmp_path / "queue.sqlite3", clock=ManualClock())
    journal.register_run(identity())
    return journal


def test_registering_a_run_is_idempotent_and_rebinding_is_refused(store: JournalStore) -> None:
    store.register_run(identity())
    assert store.lifecycle_state(RUN_ID) == LIFECYCLE_ADMITTING
    with pytest.raises(StoreError) as error:
        store.register_run(identity(container_image_digest="sha256:other"))
    assert "already bound to a different identity" in str(error.value)


def test_retry_after_a_lost_response_yields_the_same_logical_attempt(
    store: JournalStore,
) -> None:
    opened(store, "g0", 2)
    first, created_first = store.admit_attempt(
        attempt_id="g0:s0",
        idempotency_key="key:g0:0",
        run_id=RUN_ID,
        group_id="g0",
        sample_index=0,
        task_id="row-0",
        seed=1000,
        policy_revision=0,
    )
    second, created_second = store.admit_attempt(
        attempt_id="a-different-id",
        idempotency_key="key:g0:0",
        run_id=RUN_ID,
        group_id="g0",
        sample_index=0,
        task_id="row-0",
        seed=1000,
        policy_revision=0,
    )
    assert created_first is True
    assert created_second is False
    assert second.attempt_id == first.attempt_id == "g0:s0"
    assert len(store.attempts_in_group("g0")) == 1


def test_one_key_may_not_name_two_logical_attempts(store: JournalStore) -> None:
    opened(store, "g0", 2)
    admit(store, "g0", 0)
    with pytest.raises(IdempotencyError):
        store.admit_attempt(
            attempt_id="g0:s1",
            idempotency_key="key:g0:0",
            run_id=RUN_ID,
            group_id="g0",
            sample_index=1,
            task_id="row-1",
            seed=1001,
            policy_revision=0,
        )


def test_the_queue_column_follows_the_state_machine(store: JournalStore) -> None:
    opened(store, "g0", 1)
    attempt_id = admit(store, "g0", 0)
    assert store.attempt(attempt_id).queue == QUEUE_ROLLOUT
    assert store.queue_depth(QUEUE_ROLLOUT, run_id=RUN_ID) == 1
    assert store.transition_attempt(attempt_id, "running").queue is None
    assert store.transition_attempt(attempt_id, "awaiting_score").queue == QUEUE_SCORE
    assert store.transition_attempt(attempt_id, "scored").queue == QUEUE_SCORED_RESULT
    completed = store.transition_attempt(attempt_id, "completed", result_payload={"reward": 1.0})
    assert completed.queue is None
    assert store.queue_depth(QUEUE_ROLLOUT, run_id=RUN_ID) == 0


def test_an_illegal_attempt_edge_is_refused(store: JournalStore) -> None:
    opened(store, "g0", 1)
    attempt_id = admit(store, "g0", 0)
    with pytest.raises(TransitionError):
        store.transition_attempt(attempt_id, "scored")
    with pytest.raises(UnknownAttemptError):
        store.transition_attempt("never-admitted", "running")


def test_exactly_one_terminal_result_per_accepted_attempt(store: JournalStore) -> None:
    opened(store, "g0", 1)
    attempt_id = admit(store, "g0", 0)
    store.transition_attempt(attempt_id, "running")
    store.transition_attempt(attempt_id, "scored")
    store.transition_attempt(attempt_id, "completed", result_payload={"reward": 0.5})
    result = store.result(attempt_id)
    assert result is not None
    assert result.kind == "episode"
    with pytest.raises(TerminalResultError):
        store.transition_attempt(attempt_id, "failed", result_payload={"reason": "second"})
    with pytest.raises(TerminalResultError):
        store.transition_attempt(attempt_id, "cancelled")
    assert store.result(attempt_id).kind == "episode"


def test_a_terminal_result_kind_is_derived_from_the_state(store: JournalStore) -> None:
    opened(store, "g0", 3)
    failed = admit(store, "g0", 0)
    cancelled = admit(store, "g0", 1)
    store.transition_attempt(failed, "failed", reason="absent_reward")
    store.transition_attempt(cancelled, "cancelled", reason="stop")
    assert store.result(failed).kind == "failure"
    assert store.result(cancelled).kind == "cancellation"


def test_group_transitions_are_bounded(store: JournalStore) -> None:
    opened(store, "g0", 1)
    store.transition_group("g0", GROUP_COMPLETE)
    store.transition_group("g0", GROUP_TRAIN_READY)
    with pytest.raises(TransitionError):
        store.transition_group("g0", GROUP_COMPLETE)
    store.transition_group("g0", GROUP_ABANDONED, reason="stop")
    with pytest.raises(TransitionError):
        store.transition_group("g0", GROUP_ABANDONED)


def test_membership_records_a_replacement_rather_than_swapping_it(
    store: JournalStore,
) -> None:
    opened(store, "g0", 1)
    original = admit(store, "g0", 0)
    store.transition_attempt(original, "running")
    store.transition_attempt(original, "cancelled", reason="straggler")
    replacement, created = store.admit_attempt(
        attempt_id=f"{original}#r1",
        idempotency_key="key:g0:0#r1",
        run_id=RUN_ID,
        group_id="g0",
        sample_index=0,
        task_id="row-0",
        seed=1000,
        policy_revision=0,
        replaced_attempt_id=original,
        replacement_index=1,
    )
    assert created is True
    assert replacement.replaced_attempt_id == original
    members = store.group_members("g0")
    assert [(row.attempt_id, row.active, row.disposition) for row in members] == [
        (original, False, MEMBERSHIP_REPLACED),
        (replacement.attempt_id, True, MEMBERSHIP_HELD),
    ]
    snapshot = store.membership_snapshot("g0")
    assert snapshot[0]["result_kind"] == "cancellation"
    assert snapshot[1]["state"] == "queued"
    kinds = [row.kind for row in store.journal_since(0) if row.subject == replacement.attempt_id]
    assert "attempt_replaced" in kinds


def test_leases_are_granted_renewed_and_closed_once(store: JournalStore) -> None:
    clock = ManualClock()
    store.clock = clock
    opened(store, "g0", 1)
    attempt_id = admit(store, "g0", 0)
    lease = store.grant_lease(
        attempt_id=attempt_id, holder="worker-0", expires_at=30.0, straggler_deadline=3600.0
    )
    assert lease.lease_id == f"{attempt_id}#l1"
    assert lease.state == LEASE_ACTIVE
    with pytest.raises(LeaseStoreError):
        store.grant_lease(
            attempt_id=attempt_id, holder="worker-1", expires_at=30.0, straggler_deadline=3600.0
        )
    renewed = store.renew_lease(lease.lease_id, expires_at=60.0)
    assert (renewed.expires_at, renewed.heartbeats) == (60.0, 1)
    assert store.active_leases(expires_at_or_before=59.0) == ()
    assert len(store.active_leases(expires_at_or_before=60.0)) == 1
    assert len(store.active_leases(deadline_at_or_before=3600.0)) == 1
    closed = store.close_lease(lease.lease_id, state=LEASE_EXPIRED, reason="heartbeat_lost")
    assert closed.state == LEASE_EXPIRED
    assert store.active_lease_for(attempt_id) is None
    # Closing a closed lease is a no-op, not a second journal entry.
    assert store.close_lease(lease.lease_id, state=LEASE_RELEASED).state == LEASE_EXPIRED
    with pytest.raises(LeaseStoreError):
        store.close_lease(lease.lease_id, state=LEASE_ACTIVE)


def test_the_journal_cursor_is_monotone_and_readable_from_any_point(
    store: JournalStore,
) -> None:
    opened(store, "g0", 1)
    mark = store.head_cursor()
    attempt_id = admit(store, "g0", 0)
    store.transition_attempt(attempt_id, "running", reason="dispatch")
    tail = store.journal_since(mark)
    cursors = [row.cursor for row in tail]
    assert cursors == sorted(cursors)
    assert [row.kind for row in tail] == ["attempt_admitted", "attempt_transition"]
    assert tail[-1].from_state == "queued"
    assert tail[-1].to_state == "running"
    assert store.journal_since(store.head_cursor()) == ()


def test_lifecycle_state_and_events_are_durable(store: JournalStore) -> None:
    store.record_lifecycle(RUN_ID, control="pause", to_state=LIFECYCLE_PAUSED, reason="quota")
    store.record_lifecycle(RUN_ID, control="resume", to_state=None, reason="refused")
    assert store.lifecycle_state(RUN_ID) == LIFECYCLE_PAUSED
    events = store.lifecycle_events(RUN_ID)
    assert [(row.subject, row.to_state) for row in events] == [
        ("pause", LIFECYCLE_PAUSED),
        ("resume", None),
    ]
    with pytest.raises(StoreError):
        store.record_lifecycle(RUN_ID, control="pause", to_state="hibernating")


def test_dispatchable_prefers_the_oldest_open_group(store: JournalStore) -> None:
    opened(store, "g0", 2)
    opened(store, "g1", 2)
    admit(store, "g1", 0)
    admit(store, "g1", 1)
    admit(store, "g0", 0)
    admit(store, "g0", 1)
    order = [row.attempt_id for row in store.dispatchable(limit=4, run_id=RUN_ID)]
    assert order == ["g0:s0", "g0:s1", "g1:s0", "g1:s1"]
    insertion = [row.attempt_id for row in store.dispatchable(limit=4, oldest_group_first=False)]
    assert insertion == ["g1:s0", "g1:s1", "g0:s0", "g0:s1"]
    store.transition_group("g0", GROUP_COMPLETE)
    assert [row.group_id for row in store.dispatchable(limit=4)] == ["g1", "g1"]


def test_restart_recovery_reports_queued_active_scored_and_train_ready(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.sqlite3"
    clock = ManualClock()
    first = JournalStore(path, clock=clock)
    first.register_run(identity())
    opened(first, "g0", 2)
    opened(first, "g1", 2)
    opened(first, "g2", 1)
    for index in (0, 1):
        attempt_id = admit(first, "g0", index)
        first.transition_attempt(attempt_id, "running")
        first.transition_attempt(attempt_id, "scored")
        first.transition_attempt(attempt_id, "completed", result_payload={"reward": 1.0})
    first.transition_group("g0", GROUP_COMPLETE)
    first.transition_group("g0", GROUP_TRAIN_READY)
    running = admit(first, "g1", 0)
    first.transition_attempt(running, "running")
    first.grant_lease(
        attempt_id=running, holder="worker-0", expires_at=3600.0, straggler_deadline=7200.0
    )
    admit(first, "g1", 1)
    scored = admit(first, "g2", 0)
    first.transition_attempt(scored, "running")
    first.transition_attempt(scored, "awaiting_score")
    first.transition_attempt(scored, "scored")
    first.record_lifecycle(RUN_ID, control="pause", to_state=LIFECYCLE_PAUSED)
    cursor_before = first.head_cursor()
    first.close()

    second = JournalStore(path, clock=ManualClock(clock.now()))
    snapshot = second.recover(RUN_ID)
    assert snapshot.lifecycle_state == LIFECYCLE_PAUSED
    assert snapshot.cursor == cursor_before
    assert [row.attempt_id for row in snapshot.queued] == ["g1:s1"]
    assert [row.attempt_id for row in snapshot.active] == ["g1:s0"]
    assert [row.attempt_id for row in snapshot.scored] == ["g2:s0"]
    assert [row.group_id for row in snapshot.train_ready] == ["g0"]
    assert [row.group_id for row in snapshot.open_groups] == ["g1", "g2"]
    assert snapshot.depth(QUEUE_ROLLOUT) == 1
    assert snapshot.depth(QUEUE_SCORED_RESULT) == 1
    assert snapshot.depth(QUEUE_TRAIN_READY) == 1
    assert [row.attempt_id for row in snapshot.live_leases] == ["g1:s0"]
    assert {row.attempt_id for row in snapshot.attempts_without_result} == {
        "g1:s0",
        "g1:s1",
        "g2:s0",
    }
    assert second.run_identity(RUN_ID).binding_digest == identity().binding_digest
    # The reopened journal keeps writing after the recovered cursor.
    recovered = admit(second, "g2", 0, idempotency_key="key:g2:0")
    assert recovered == "g2:s0"
    second.transition_attempt(
        "g2:s0", "completed", result_payload={"reward": 0.0}
    )
    assert second.head_cursor() > cursor_before
    second.close()
