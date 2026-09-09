"""Pause, drain, resume and stop, with defined semantics at every boundary.

These are first-class controls, not signals, and each one is defined per queue
boundary rather than per process:

* **Pause** stops admission and dispatch. In-flight attempts keep their leases
  and run to a terminal result; scoring, validation and catalog registration
  continue; a new train step is refused. A paused run is a legal resting state.
* **Drain** is pause plus completion: every in-flight attempt finishes and every
  complete group trains, then the run stops. Attempts still queued are cancelled
  and every partial group is recorded as abandoned with its membership, so its
  cost stays attributable.
* **Resume** re-handshakes first. A changed contract hash, image digest, plan
  hash or renderer fingerprint is refused: that is a new run with a lineage edge
  to this one, not a continuation.
* **Stop** cancels in-flight attempts through the declared terminate route,
  releases leases, and leaves receipts complete — exactly one terminal result
  per accepted attempt, including the ones that never ran.

The terminate route and the re-handshake are injected. This module owns the
semantics, not the transport.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.rl_records import RecordError
from .store import (
    GROUP_ABANDONED,
    GROUP_COMPLETE,
    GROUP_OPEN,
    GROUP_TRAIN_READY,
    LEASE_CANCELLED,
    LIFECYCLE_ADMITTING,
    LIFECYCLE_DRAINED,
    LIFECYCLE_DRAINING,
    LIFECYCLE_PAUSED,
    LIFECYCLE_STATES,
    LIFECYCLE_STOPPED,
    Clock,
    JournalStore,
    RunIdentity,
)

CONTROLS: tuple[str, ...] = ("pause", "drain", "resume", "stop")

#: Terminate route for one in-flight attempt. Raising is recorded, never fatal:
#: a container that cannot be reached still owes a local terminal result.
TerminateHook = Callable[[str], None]

#: Re-handshake performed before a resume re-admits work.
RehandshakeHook = Callable[[], RunIdentity]


class LifecycleError(RecordError):
    """A lifecycle control was refused."""


class LifecycleTransitionError(LifecycleError):
    """The control does not apply from the current lifecycle state."""


class AdmissionClosed(LifecycleError):
    """Admission is closed in this lifecycle state."""


class DispatchClosed(LifecycleError):
    """Dispatch to the container is closed in this lifecycle state."""


class ScoringClosed(LifecycleError):
    """Scoring and validation are closed in this lifecycle state."""


class TrainStepBlocked(LifecycleError):
    """A new train step may not start in this lifecycle state."""


class ResumeRefused(LifecycleError):
    """The run's binding changed, so resuming would silently change the dataset."""

    def __init__(self, message: str, *, changed_fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.changed_fields: tuple[str, ...] = tuple(changed_fields)


@dataclass(frozen=True, slots=True)
class LifecycleGates:
    """What each queue boundary may do in one lifecycle state."""

    admit: bool
    dispatch: bool
    score: bool
    train: bool


GATES: Mapping[str, LifecycleGates] = {
    LIFECYCLE_ADMITTING: LifecycleGates(admit=True, dispatch=True, score=True, train=True),
    LIFECYCLE_PAUSED: LifecycleGates(admit=False, dispatch=False, score=True, train=False),
    LIFECYCLE_DRAINING: LifecycleGates(admit=False, dispatch=False, score=True, train=True),
    LIFECYCLE_DRAINED: LifecycleGates(admit=False, dispatch=False, score=False, train=False),
    LIFECYCLE_STOPPED: LifecycleGates(admit=False, dispatch=False, score=False, train=False),
}


@dataclass(frozen=True, slots=True)
class AbandonedGroup:
    """A partial group whose cost stays attributable after it stops."""

    group_id: str
    state_before: str
    reason: str
    membership: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LifecycleReport:
    """What a drain or a stop actually did."""

    control: str
    from_state: str
    to_state: str
    cancelled_attempts: tuple[str, ...] = ()
    closed_leases: tuple[str, ...] = ()
    abandoned_groups: tuple[AbandonedGroup, ...] = ()
    terminate_failures: Mapping[str, str] = field(default_factory=dict)

    def as_detail(self) -> Mapping[str, Any]:
        return {
            "cancelled_attempts": list(self.cancelled_attempts),
            "closed_leases": list(self.closed_leases),
            "abandoned_groups": [
                {
                    "group_id": group.group_id,
                    "state_before": group.state_before,
                    "reason": group.reason,
                    "membership": list(group.membership),
                }
                for group in self.abandoned_groups
            ],
            "terminate_failures": dict(self.terminate_failures),
        }


class RunLifecycle:
    """The four controls over one run's queue boundaries."""

    def __init__(
        self,
        store: JournalStore,
        run_id: str,
        *,
        terminate: TerminateHook | None = None,
        rehandshake: RehandshakeHook | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.clock: Clock = clock or store.clock
        self._terminate = terminate
        self._rehandshake = rehandshake

    # -- state and gates ---------------------------------------------------

    @property
    def state(self) -> str:
        return self.store.lifecycle_state(self.run_id)

    @property
    def gates(self) -> LifecycleGates:
        state = self.state
        gates = GATES.get(state)
        if gates is None:  # pragma: no cover - guarded by LIFECYCLE_STATES
            raise LifecycleError(f"unknown lifecycle state {state!r}")
        return gates

    def assert_can_admit(self) -> None:
        state = self.state
        if not GATES[state].admit:
            raise AdmissionClosed(f"admission is closed while the run is {state}")

    def assert_can_dispatch(self) -> None:
        state = self.state
        if not GATES[state].dispatch:
            raise DispatchClosed(f"dispatch is closed while the run is {state}")

    def assert_can_score(self) -> None:
        state = self.state
        if not GATES[state].score:
            raise ScoringClosed(f"scoring is closed while the run is {state}")

    def assert_can_train(self) -> None:
        state = self.state
        if not GATES[state].train:
            raise TrainStepBlocked(f"a new train step may not start while the run is {state}")

    # -- controls ----------------------------------------------------------

    def pause(self, *, reason: str = "") -> str:
        state = self.state
        if state == LIFECYCLE_PAUSED:
            return state
        if state != LIFECYCLE_ADMITTING:
            raise LifecycleTransitionError(f"a {state} run cannot be paused")
        self.store.record_lifecycle(
            self.run_id, control="pause", to_state=LIFECYCLE_PAUSED, reason=reason
        )
        return LIFECYCLE_PAUSED

    def drain(self, *, reason: str = "") -> str:
        state = self.state
        if state == LIFECYCLE_DRAINING:
            return state
        if state not in (LIFECYCLE_ADMITTING, LIFECYCLE_PAUSED):
            raise LifecycleTransitionError(f"a {state} run cannot be drained")
        self.store.record_lifecycle(
            self.run_id, control="drain", to_state=LIFECYCLE_DRAINING, reason=reason
        )
        return LIFECYCLE_DRAINING

    def outstanding_drain_work(self) -> Mapping[str, tuple[str, ...]]:
        """What a drain is still waiting on before it may finish."""

        return {
            "in_flight": tuple(
                attempt.attempt_id for attempt in self.store.in_flight(run_id=self.run_id)
            ),
            "scored": tuple(
                attempt.attempt_id
                for attempt in self.store.attempts_in_state("scored", run_id=self.run_id)
            ),
            "complete_groups": tuple(
                group.group_id
                for group in self.store.groups_in_state(GROUP_COMPLETE, run_id=self.run_id)
            ),
            "train_ready_groups": tuple(
                group.group_id
                for group in self.store.groups_in_state(GROUP_TRAIN_READY, run_id=self.run_id)
            ),
        }

    def finish_drain(self, *, reason: str = "drain") -> LifecycleReport:
        """Close a drain: in-flight work is done and every complete group trained."""

        state = self.state
        if state != LIFECYCLE_DRAINING:
            raise LifecycleTransitionError(f"a {state} run is not draining")
        outstanding = {name: ids for name, ids in self.outstanding_drain_work().items() if ids}
        if outstanding:
            raise LifecycleTransitionError(
                f"drain is not finished; still outstanding: {outstanding}"
            )
        cancelled, closed = self._cancel_pending(reason=reason, route_terminate=False)
        abandoned = self._abandon_groups(reason=reason, states=(GROUP_OPEN,))
        report = LifecycleReport(
            control="drain",
            from_state=state,
            to_state=LIFECYCLE_DRAINED,
            cancelled_attempts=cancelled,
            closed_leases=closed,
            abandoned_groups=abandoned,
        )
        self.store.record_lifecycle(
            self.run_id,
            control="drain_finished",
            to_state=LIFECYCLE_DRAINED,
            reason=reason,
            detail=report.as_detail(),
        )
        return report

    def resume(self) -> str:
        """Re-handshake, verify the binding, then re-admit work. Never assume."""

        state = self.state
        if state != LIFECYCLE_PAUSED:
            raise LifecycleTransitionError(
                f"only a paused run may resume; this run is {state}"
            )
        if self._rehandshake is None:
            raise LifecycleError(
                "resume requires a re-handshake hook: the agreement must be re-verified "
                "before any work is re-admitted"
            )
        proposed = self._rehandshake()
        stored = self.store.run_identity(self.run_id)
        changed: list[str] = []
        if proposed.run_id != stored.run_id:
            changed.append("run_id")
        expected = stored.binding_fields()
        for name, value in proposed.binding_fields().items():
            if expected[name] != value:
                changed.append(name)
        if changed:
            self.store.record_lifecycle(
                self.run_id,
                control="resume",
                to_state=None,
                reason="refused",
                detail={
                    "changed_fields": changed,
                    "stored_binding_digest": stored.binding_digest,
                    "proposed_binding_digest": proposed.binding_digest,
                    "lineage": "a changed binding is a new run with a lineage edge to this one",
                },
            )
            raise ResumeRefused(
                f"resume refused: {', '.join(changed)} changed; that is a new run with a "
                "lineage edge to this one, not a continuation",
                changed_fields=changed,
            )
        self.store.record_lifecycle(
            self.run_id,
            control="resume",
            to_state=LIFECYCLE_ADMITTING,
            reason="rehandshake_verified",
            detail={"binding_digest": stored.binding_digest},
        )
        return LIFECYCLE_ADMITTING

    def stop(self, *, reason: str = "stop") -> LifecycleReport:
        """Cancel through the terminate route and leave the receipts complete."""

        state = self.state
        if state == LIFECYCLE_STOPPED:
            raise LifecycleTransitionError("the run is already stopped")
        cancelled, closed, failures = self._cancel_all(reason=reason)
        abandoned = self._abandon_groups(
            reason=reason, states=(GROUP_OPEN, GROUP_COMPLETE, GROUP_TRAIN_READY)
        )
        report = LifecycleReport(
            control="stop",
            from_state=state,
            to_state=LIFECYCLE_STOPPED,
            cancelled_attempts=cancelled,
            closed_leases=closed,
            abandoned_groups=abandoned,
            terminate_failures=failures,
        )
        self.store.record_lifecycle(
            self.run_id,
            control="stop",
            to_state=LIFECYCLE_STOPPED,
            reason=reason,
            detail=report.as_detail(),
        )
        return report

    def terminate(self, attempt_id: str, *, reason: str = "") -> str | None:
        """Route one cancellation through the declared terminate hook.

        Returns the recorded error text when the route failed. A container that
        cannot be reached still owes a local terminal result, so a failure here
        is journalled and the caller carries on cancelling.
        """

        if self._terminate is None:
            return None
        try:
            self._terminate(attempt_id)
        except Exception as error:  # noqa: BLE001 - recorded, never fatal
            text = f"{type(error).__name__}: {error}"
            self.store.record_lifecycle(
                self.run_id,
                control="terminate_failed",
                to_state=None,
                reason=reason,
                detail={"attempt_id": attempt_id, "error": text},
            )
            return text
        return None

    def abandoned_groups(self) -> tuple[str, ...]:
        return tuple(
            group.group_id
            for group in self.store.groups_in_state(GROUP_ABANDONED, run_id=self.run_id)
        )

    # -- internals ---------------------------------------------------------

    def _cancel_pending(
        self, *, reason: str, route_terminate: bool
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Cancel accepted-but-never-run attempts so their receipts close."""

        cancelled: list[str] = []
        closed: list[str] = []
        for attempt in self.store.attempts_in_state("queued", run_id=self.run_id):
            for lease in self.store.active_leases(attempt_id=attempt.attempt_id):
                self.store.close_lease(lease.lease_id, state=LEASE_CANCELLED, reason=reason)
                closed.append(lease.lease_id)
            self.store.transition_attempt(
                attempt.attempt_id,
                "cancelled",
                reason=reason,
                result_payload={
                    "reason": reason,
                    "route": "terminate" if route_terminate else "never_dispatched",
                },
            )
            cancelled.append(attempt.attempt_id)
        return tuple(cancelled), tuple(closed)

    def _cancel_all(
        self, *, reason: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, str]]:
        cancelled: list[str] = []
        closed: list[str] = []
        failures: dict[str, str] = {}
        for attempt in self.store.non_terminal_attempts(run_id=self.run_id):
            leases = self.store.active_leases(attempt_id=attempt.attempt_id)
            if leases:
                error = self.terminate(attempt.attempt_id, reason=reason)
                if error is not None:
                    failures[attempt.attempt_id] = error
            for lease in leases:
                self.store.close_lease(lease.lease_id, state=LEASE_CANCELLED, reason=reason)
                closed.append(lease.lease_id)
            self.store.transition_attempt(
                attempt.attempt_id,
                "cancelled",
                reason=reason,
                result_payload={"reason": reason, "route": "terminate"},
            )
            cancelled.append(attempt.attempt_id)
        return tuple(cancelled), tuple(closed), failures

    def _abandon_groups(
        self, *, reason: str, states: Sequence[str]
    ) -> tuple[AbandonedGroup, ...]:
        abandoned: list[AbandonedGroup] = []
        for state in states:
            for group in self.store.groups_in_state(state, run_id=self.run_id):
                membership = self.store.membership_snapshot(group.group_id)
                self.store.transition_group(
                    group.group_id,
                    GROUP_ABANDONED,
                    reason=reason,
                    detail={"membership": list(membership), "state_before": group.state},
                )
                abandoned.append(
                    AbandonedGroup(
                        group_id=group.group_id,
                        state_before=group.state,
                        reason=reason,
                        membership=membership,
                    )
                )
        return tuple(abandoned)


def gates_for(state: str) -> LifecycleGates:
    """The declared gates for one lifecycle state."""

    if state not in LIFECYCLE_STATES:
        raise LifecycleError(f"unknown lifecycle state {state!r}")
    return GATES[state]
