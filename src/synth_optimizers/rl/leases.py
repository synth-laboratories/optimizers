"""Leases sized from the container's advertised horizon.

Two clocks govern an attempt, and conflating them is what makes a queue declare
healthy work dead:

* the **heartbeat lease**, renewed while the container is alive. Missing it is
  evidence the holder is gone, so the attempt is recovered.
* the **straggler deadline**, fixed at grant from the advertised horizon plus
  the in-lease post-horizon work plus a declared grace. Passing it means the
  attempt is late even though it is still breathing, and the declared straggler
  policy cancels and replaces it.

An hour-scale attempt is the normal case here, not an anomaly: nothing derives a
timeout from a guess. A step- or tick-measured horizon is converted to wall
seconds only through a declared conversion — the container's own, or an explicit
sizing override — and refuses to be sized without one. Post-horizon quiescence
and artifact collection are inside the lease, never work done after the attempt
is considered complete.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.rl_identity import Horizon
from ..contracts.rl_records import RecordError
from .store import (
    LEASE_CANCELLED,
    LEASE_EXPIRED,
    LEASE_RELEASED,
    Clock,
    JournalStore,
    LeaseRow,
)

STRAGGLER_CANCEL = "cancel"
STRAGGLER_CANCEL_AND_REPLACE = "cancel_and_replace"
#: The declared straggler actions. Which one applies is configuration.
STRAGGLER_ACTIONS: frozenset[str] = frozenset({STRAGGLER_CANCEL, STRAGGLER_CANCEL_AND_REPLACE})


class LeaseError(RecordError):
    """A lease could not be sized, granted, or renewed."""


class LeaseExpiredError(LeaseError):
    """A heartbeat arrived after the lease had already lapsed."""


@dataclass(frozen=True, slots=True)
class LeaseSizing:
    """How an advertised horizon becomes a lease. Every number is declared."""

    heartbeat_interval_seconds: float
    missed_heartbeats_allowed: int = 2
    quiescence_seconds: float = 0.0
    artifact_collection_seconds: float = 0.0
    grace_seconds: float = 0.0
    #: Override for the horizon's own declared conversion. ``None`` means the
    #: container's declaration is used, so nothing here is ever a guess.
    seconds_per_unit: float | None = None

    def __post_init__(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            raise LeaseError("heartbeat interval must be positive")
        if self.missed_heartbeats_allowed < 1:
            raise LeaseError("a lease must tolerate at least one missed heartbeat")
        for name in ("quiescence_seconds", "artifact_collection_seconds", "grace_seconds"):
            if getattr(self, name) < 0:
                raise LeaseError(f"{name} must be non-negative")
        if self.seconds_per_unit is not None and self.seconds_per_unit <= 0:
            raise LeaseError("seconds_per_unit must be positive when declared")

    @property
    def heartbeat_ttl_seconds(self) -> float:
        """How long silence is tolerated before the lease is treated as lost."""

        return self.heartbeat_interval_seconds * (self.missed_heartbeats_allowed + 1)

    @property
    def in_lease_seconds(self) -> float:
        """Post-horizon work that belongs to the attempt, not to a later phase."""

        return self.quiescence_seconds + self.artifact_collection_seconds

    def seconds_per_unit_for(self, horizon: Horizon) -> float:
        """The declared conversion for a step- or tick-measured horizon.

        The container's declaration is authoritative; the sizing override exists
        for a caller that has measured the substrate more precisely. Neither is
        inferred from the horizon value.
        """

        if self.seconds_per_unit is not None:
            return self.seconds_per_unit
        declared = getattr(horizon, "seconds_per_unit", None)
        if declared is None:
            raise LeaseError(
                f"a {horizon.horizon_kind} horizon must declare seconds_per_unit; "
                "lease sizing is derived, never guessed"
            )
        return float(declared)

    def horizon_seconds(self, horizon: Horizon) -> float:
        """Wall seconds the container says one attempt may take."""

        if horizon.horizon_kind == "wall_clock":
            return float(horizon.value) * float(horizon.time_dilation)
        conversion = self.seconds_per_unit_for(horizon)
        return float(horizon.value) * conversion * float(horizon.time_dilation)

    def grace_for(self, horizon: Horizon) -> float:
        """The container's declared grace wins; the sizing's value is the fallback."""

        return float(horizon.grace_seconds) if horizon.grace_seconds > 0 else self.grace_seconds

    def straggler_offset_seconds(self, horizon: Horizon) -> float:
        """Horizon plus in-lease collection plus grace: when late becomes fatal."""

        return self.horizon_seconds(horizon) + self.in_lease_seconds + self.grace_for(horizon)


@dataclass(frozen=True, slots=True)
class StragglerPolicy:
    """Declared, not inferred: what happens to an attempt past its deadline."""

    action: str = STRAGGLER_CANCEL_AND_REPLACE
    max_replacements: int = 1

    def __post_init__(self) -> None:
        if self.action not in STRAGGLER_ACTIONS:
            raise LeaseError(f"unknown straggler action {self.action!r}")
        if self.max_replacements < 0:
            raise LeaseError("max_replacements must be non-negative")

    @property
    def replaces(self) -> bool:
        return self.action == STRAGGLER_CANCEL_AND_REPLACE

    def may_replace(self, replacement_index: int) -> bool:
        return self.replaces and replacement_index < self.max_replacements


class LeaseBook:
    """Grants, renews and expires the leases of one run's attempts."""

    def __init__(
        self,
        store: JournalStore,
        *,
        horizon: Horizon,
        sizing: LeaseSizing,
        straggler: StragglerPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self.horizon = horizon
        self.sizing = sizing
        self.straggler = straggler or StragglerPolicy()
        self.clock: Clock = clock or store.clock

    # -- sizing -----------------------------------------------------------

    @property
    def horizon_seconds(self) -> float:
        return self.sizing.horizon_seconds(self.horizon)

    @property
    def heartbeat_ttl_seconds(self) -> float:
        return self.sizing.heartbeat_ttl_seconds

    @property
    def straggler_offset_seconds(self) -> float:
        return self.sizing.straggler_offset_seconds(self.horizon)

    # -- lifecycle of one lease -------------------------------------------

    def grant(self, attempt_id: str, *, holder: str) -> LeaseRow:
        moment = self.clock.now()
        return self.store.grant_lease(
            attempt_id=attempt_id,
            holder=holder,
            expires_at=moment + self.heartbeat_ttl_seconds,
            straggler_deadline=moment + self.straggler_offset_seconds,
        )

    def heartbeat(self, lease_id: str) -> LeaseRow:
        """Renew from now. A heartbeat never moves the straggler deadline."""

        lease = self.store.lease(lease_id)
        moment = self.clock.now()
        if lease.expires_at <= moment:
            raise LeaseExpiredError(
                f"lease {lease_id} lapsed at {lease.expires_at} and cannot be renewed at {moment}"
            )
        return self.store.renew_lease(lease_id, expires_at=moment + self.heartbeat_ttl_seconds)

    def release(self, lease_id: str, *, reason: str = "terminal_result") -> LeaseRow:
        return self.store.close_lease(lease_id, state=LEASE_RELEASED, reason=reason)

    def cancel(self, lease_id: str, *, reason: str = "cancelled") -> LeaseRow:
        return self.store.close_lease(lease_id, state=LEASE_CANCELLED, reason=reason)

    def mark_expired(self, lease_id: str, *, reason: str = "heartbeat_lost") -> LeaseRow:
        return self.store.close_lease(lease_id, state=LEASE_EXPIRED, reason=reason)

    # -- sweeps -----------------------------------------------------------

    def stragglers(self, *, at: float | None = None) -> tuple[LeaseRow, ...]:
        """Active leases past horizon plus grace, heartbeating or not."""

        moment = self.clock.now() if at is None else at
        return self.store.active_leases(deadline_at_or_before=moment)

    def expired(self, *, at: float | None = None) -> tuple[LeaseRow, ...]:
        """Active leases whose heartbeat lapsed but whose deadline has not."""

        moment = self.clock.now() if at is None else at
        straggler_ids = {lease.lease_id for lease in self.stragglers(at=moment)}
        return tuple(
            lease
            for lease in self.store.active_leases(expires_at_or_before=moment)
            if lease.lease_id not in straggler_ids
        )

    def lease_for(self, attempt_id: str) -> LeaseRow | None:
        return self.store.active_lease_for(attempt_id)
