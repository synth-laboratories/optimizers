"""Durable journal for the container-first RL queue engine.

Every attempt admission, queue transition, lease, group-membership change and
lifecycle control is appended to one monotone log and applied to a small set of
derived tables inside the same transaction. A restarted process therefore reads
back exactly what was queued, active, scored and train-ready at the moment the
previous one died, without replaying anything by hand.

``sqlite3`` from the standard library is the whole dependency: the journal is a
file, not a service. Nothing here knows a task, a harness, an environment or an
algorithm. Bounds, capacities, horizons and dispositions all arrive as
configuration from the caller.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..contracts.rl_identity import ATTEMPT_STATES, TERMINAL_ATTEMPT_STATES
from ..contracts.rl_records import RecordError, digest

JOURNAL_SCHEMA_VERSION = "cispo.queue_journal.v1"

QUEUE_ROLLOUT = "rollout"
QUEUE_SCORE = "score"
QUEUE_SCORED_RESULT = "scored_result"
QUEUE_TRAIN_READY = "train_ready"
QUEUES: tuple[str, ...] = (QUEUE_ROLLOUT, QUEUE_SCORE, QUEUE_SCORED_RESULT, QUEUE_TRAIN_READY)

#: An attempt in one of these states is sitting in the named queue. ``running``
#: and every terminal state occupy no queue: a running attempt is held by its
#: lease, and a terminal attempt is held by its one result row.
QUEUE_FOR_STATE: Mapping[str, str | None] = {
    "queued": QUEUE_ROLLOUT,
    "running": None,
    "awaiting_score": QUEUE_SCORE,
    "scored": QUEUE_SCORED_RESULT,
    "completed": None,
    "failed": None,
    "cancelled": None,
}

#: The attempt state machine. ``running -> queued`` is lease recovery: the same
#: logical attempt returns to the rollout queue, it is never duplicated.
ATTEMPT_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "queued": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"awaiting_score", "scored", "queued", "failed", "cancelled"}),
    "awaiting_score": frozenset({"scored", "queued", "failed", "cancelled"}),
    "scored": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

GROUP_OPEN = "open"
GROUP_COMPLETE = "complete"
GROUP_TRAIN_READY = "train_ready"
GROUP_TRAINED = "trained"
GROUP_DISCARDED = "discarded"
GROUP_ABANDONED = "abandoned"
GROUP_RECYCLED = "recycled"
GROUP_STATES: tuple[str, ...] = (
    GROUP_OPEN,
    GROUP_COMPLETE,
    GROUP_TRAIN_READY,
    GROUP_TRAINED,
    GROUP_DISCARDED,
    GROUP_ABANDONED,
    GROUP_RECYCLED,
)

#: A group leaves ``train_ready`` exactly once: released to a train step, or
#: rejected by the dequeue gate as discarded or recycled.
GROUP_TRANSITIONS: Mapping[str, frozenset[str]] = {
    GROUP_OPEN: frozenset({GROUP_COMPLETE, GROUP_DISCARDED, GROUP_ABANDONED}),
    GROUP_COMPLETE: frozenset({GROUP_TRAIN_READY, GROUP_DISCARDED, GROUP_ABANDONED}),
    GROUP_TRAIN_READY: frozenset({GROUP_TRAINED, GROUP_DISCARDED, GROUP_RECYCLED, GROUP_ABANDONED}),
    GROUP_TRAINED: frozenset(),
    GROUP_DISCARDED: frozenset(),
    GROUP_ABANDONED: frozenset(),
    GROUP_RECYCLED: frozenset(),
}

#: One terminal result per accepted attempt, and the kind is derived from the
#: terminal state so a caller cannot record a failure as an episode.
RESULT_KIND_FOR_STATE: Mapping[str, str] = {
    "completed": "episode",
    "failed": "failure",
    "cancelled": "cancellation",
}
RESULT_KINDS: tuple[str, ...] = ("episode", "failure", "cancellation")

LEASE_ACTIVE = "active"
LEASE_RELEASED = "released"
LEASE_EXPIRED = "expired"
LEASE_CANCELLED = "cancelled"
LEASE_STATES: tuple[str, ...] = (LEASE_ACTIVE, LEASE_RELEASED, LEASE_EXPIRED, LEASE_CANCELLED)

MEMBERSHIP_HELD = "held"
MEMBERSHIP_REPLACED = "replaced"

LIFECYCLE_ADMITTING = "admitting"
LIFECYCLE_PAUSED = "paused"
LIFECYCLE_DRAINING = "draining"
LIFECYCLE_DRAINED = "drained"
LIFECYCLE_STOPPED = "stopped"
LIFECYCLE_STATES: tuple[str, ...] = (
    LIFECYCLE_ADMITTING,
    LIFECYCLE_PAUSED,
    LIFECYCLE_DRAINING,
    LIFECYCLE_DRAINED,
    LIFECYCLE_STOPPED,
)


class StoreError(RecordError):
    """The journal refused a write because it would break an invariant."""


class UnknownAttemptError(StoreError):
    """An attempt id that was never admitted."""


class UnknownGroupError(StoreError):
    """A group id that was never opened."""


class TransitionError(StoreError):
    """An illegal edge in the attempt or group state machine."""


class TerminalResultError(StoreError):
    """A second terminal result for one accepted attempt. Exactly one is legal."""


class IdempotencyError(StoreError):
    """One idempotency key was reused for a different logical attempt."""


class LeaseStoreError(StoreError):
    """A lease write that contradicts the lease already on record."""


class Clock(Protocol):
    """Time is injected. Nothing in the engine sleeps and nothing calls wall time."""

    def now(self) -> float:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ManualClock:
    """A clock the caller advances by hand, for tests and for replay."""

    time: float = 0.0

    def now(self) -> float:
        return self.time

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a clock may not run backwards")
        self.time += seconds
        return self.time

    def set_to(self, moment: float) -> float:
        if moment < self.time:
            raise ValueError("a clock may not run backwards")
        self.time = moment
        return self.time


@dataclass(slots=True)
class SystemClock:
    """Wall time, for a live run."""

    def now(self) -> float:
        return time.time()


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """What a run is bound to. Resume compares these field by field."""

    run_id: str
    container_contract_hash: str
    container_image_digest: str
    algorithm_plan_hash: str
    renderer_fingerprint: str
    handshake_agreement_digest: str = ""
    capability_hash: str = ""

    def binding_fields(self) -> Mapping[str, str]:
        """Every field whose change makes a resume a different run."""

        return {
            "container_contract_hash": self.container_contract_hash,
            "container_image_digest": self.container_image_digest,
            "algorithm_plan_hash": self.algorithm_plan_hash,
            "renderer_fingerprint": self.renderer_fingerprint,
            "handshake_agreement_digest": self.handshake_agreement_digest,
            "capability_hash": self.capability_hash,
        }

    @property
    def binding_digest(self) -> str:
        return digest(dict(self.binding_fields()), length=32)

    def to_payload(self) -> Mapping[str, Any]:
        return {"run_id": self.run_id, **self.binding_fields()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunIdentity":
        return cls(**{str(key): str(value) for key, value in payload.items()})


@dataclass(frozen=True, slots=True)
class AttemptRow:
    attempt_id: str
    idempotency_key: str
    run_id: str
    group_id: str
    sample_index: int
    task_id: str
    seed: int
    policy_revision: int
    state: str
    queue: str | None
    dispatch_count: int
    replacement_index: int
    replaced_attempt_id: str | None
    agent_instance_id: str | None
    team_id: str | None
    created_at: float
    updated_at: float
    sequence: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_ATTEMPT_STATES


@dataclass(frozen=True, slots=True)
class GroupRow:
    group_id: str
    run_id: str
    cardinality: int
    pin_digest: str
    policy_revision: int
    state: str
    opened_at: float
    closed_at: float | None
    sequence: int
    pin: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MembershipRow:
    group_id: str
    attempt_id: str
    sample_index: int
    active: bool
    disposition: str
    replaced_attempt_id: str | None
    recorded_at: float


@dataclass(frozen=True, slots=True)
class LeaseRow:
    lease_id: str
    attempt_id: str
    holder: str
    granted_at: float
    expires_at: float
    straggler_deadline: float
    heartbeats: int
    state: str
    closed_at: float | None


@dataclass(frozen=True, slots=True)
class ResultRow:
    attempt_id: str
    kind: str
    terminal_status: str
    recorded_at: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JournalRow:
    cursor: int
    kind: str
    run_id: str
    subject: str
    at: float
    from_state: str | None = None
    to_state: str | None = None
    from_queue: str | None = None
    to_queue: str | None = None
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    """What the queues held at the last durable write."""

    run_id: str
    lifecycle_state: str
    cursor: int
    queued: tuple[AttemptRow, ...]
    active: tuple[AttemptRow, ...]
    awaiting_score: tuple[AttemptRow, ...]
    scored: tuple[AttemptRow, ...]
    open_groups: tuple[GroupRow, ...]
    complete_groups: tuple[GroupRow, ...]
    train_ready: tuple[GroupRow, ...]
    live_leases: tuple[LeaseRow, ...]
    attempts_without_result: tuple[AttemptRow, ...]

    def depth(self, queue: str) -> int:
        if queue == QUEUE_ROLLOUT:
            return len(self.queued)
        if queue == QUEUE_SCORE:
            return len(self.awaiting_score)
        if queue == QUEUE_SCORED_RESULT:
            return len(self.scored)
        if queue == QUEUE_TRAIN_READY:
            return len(self.train_ready)
        raise StoreError(f"unknown queue {queue!r}")


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        identity TEXT NOT NULL,
        identity_digest TEXT NOT NULL,
        lifecycle_state TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attempt_groups (
        group_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        cardinality INTEGER NOT NULL,
        pin TEXT NOT NULL,
        pin_digest TEXT NOT NULL,
        policy_revision INTEGER NOT NULL,
        state TEXT NOT NULL,
        opened_at REAL NOT NULL,
        closed_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attempts (
        attempt_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL,
        group_id TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        task_id TEXT NOT NULL,
        seed INTEGER NOT NULL,
        policy_revision INTEGER NOT NULL,
        state TEXT NOT NULL,
        queue TEXT,
        dispatch_count INTEGER NOT NULL DEFAULT 0,
        replacement_index INTEGER NOT NULL DEFAULT 0,
        replaced_attempt_id TEXT,
        agent_instance_id TEXT,
        team_id TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS group_members (
        member_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        active INTEGER NOT NULL,
        disposition TEXT NOT NULL,
        replaced_attempt_id TEXT,
        recorded_at REAL NOT NULL,
        UNIQUE (group_id, attempt_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS leases (
        lease_id TEXT PRIMARY KEY,
        attempt_id TEXT NOT NULL,
        holder TEXT NOT NULL,
        granted_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        straggler_deadline REAL NOT NULL,
        heartbeats INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL,
        closed_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS results (
        attempt_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        terminal_status TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        recorded_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS journal (
        cursor INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        run_id TEXT NOT NULL,
        subject TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT,
        from_queue TEXT,
        to_queue TEXT,
        reason TEXT NOT NULL DEFAULT '',
        detail TEXT NOT NULL DEFAULT '{}',
        at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS attempts_by_state ON attempts (run_id, state)",
    "CREATE INDEX IF NOT EXISTS attempts_by_group ON attempts (group_id, sample_index)",
    "CREATE INDEX IF NOT EXISTS groups_by_state ON attempt_groups (run_id, state)",
    "CREATE INDEX IF NOT EXISTS leases_by_attempt ON leases (attempt_id, state)",
    "CREATE INDEX IF NOT EXISTS journal_by_run ON journal (run_id, cursor)",
)


def _encode(payload: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(payload or {}), sort_keys=True, default=str)


def _decode(text: str | None) -> Mapping[str, Any]:
    if not text:
        return {}
    loaded = json.loads(text)
    return loaded if isinstance(loaded, dict) else {"value": loaded}


class JournalStore:
    """The durable queue journal. One file, one monotone cursor."""

    def __init__(self, path: str | Path, *, clock: Clock | None = None) -> None:
        self.path = str(path)
        self.clock: Clock = clock or SystemClock()
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        with self._write() as cur:
            for statement in _SCHEMA:
                cur.execute(statement)
            cur.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (JOURNAL_SCHEMA_VERSION,),
            )
            cur.execute("INSERT OR IGNORE INTO meta (key,value) VALUES ('event_log_id',?)", (str(uuid.uuid4()),))

    # -- plumbing ---------------------------------------------------------

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "JournalStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Cursor]:
        cur = self._connection.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            yield cur
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
        finally:
            cur.close()

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        cur = self._connection.execute(sql, tuple(params))
        try:
            return cur.fetchall()
        finally:
            cur.close()

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def _append(
        self,
        cur: sqlite3.Cursor,
        kind: str,
        *,
        run_id: str,
        subject: str,
        from_state: str | None = None,
        to_state: str | None = None,
        from_queue: str | None = None,
        to_queue: str | None = None,
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> int:
        cur.execute(
            "INSERT INTO journal (kind, run_id, subject, from_state, to_state, from_queue, "
            "to_queue, reason, detail, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                run_id,
                subject,
                from_state,
                to_state,
                from_queue,
                to_queue,
                reason,
                _encode(detail),
                self.clock.now(),
            ),
        )
        return int(cur.lastrowid or 0)

    # -- runs and lifecycle state ----------------------------------------

    def register_run(self, identity: RunIdentity) -> RunIdentity:
        """Idempotent. Re-registering a different binding is a different run."""

        existing = self._one("SELECT * FROM runs WHERE run_id = ?", (identity.run_id,))
        if existing is not None:
            if existing["identity_digest"] != identity.binding_digest:
                raise StoreError(
                    f"run {identity.run_id} is already bound to a different identity; "
                    "a changed binding is a new run with a lineage edge, not this one"
                )
            return identity
        moment = self.clock.now()
        with self._write() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, identity, identity_digest, lifecycle_state, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identity.run_id,
                    _encode(identity.to_payload()),
                    identity.binding_digest,
                    LIFECYCLE_ADMITTING,
                    moment,
                    moment,
                ),
            )
            self._append(
                cur,
                "run_registered",
                run_id=identity.run_id,
                subject=identity.run_id,
                to_state=LIFECYCLE_ADMITTING,
                detail=dict(identity.to_payload()),
            )
        return identity

    def run_identity(self, run_id: str) -> RunIdentity:
        row = self._one("SELECT identity FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise StoreError(f"run {run_id!r} is not registered")
        return RunIdentity.from_payload(_decode(row["identity"]))

    def lifecycle_state(self, run_id: str) -> str:
        row = self._one("SELECT lifecycle_state FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise StoreError(f"run {run_id!r} is not registered")
        return str(row["lifecycle_state"])

    def record_lifecycle(
        self,
        run_id: str,
        *,
        control: str,
        to_state: str | None,
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> int:
        """Append a lifecycle event; ``to_state`` of ``None`` records a refusal."""

        current = self.lifecycle_state(run_id)
        if to_state is not None and to_state not in LIFECYCLE_STATES:
            raise StoreError(f"unknown lifecycle state {to_state!r}")
        with self._write() as cur:
            if to_state is not None:
                cur.execute(
                    "UPDATE runs SET lifecycle_state = ?, updated_at = ? WHERE run_id = ?",
                    (to_state, self.clock.now(), run_id),
                )
            return self._append(
                cur,
                "lifecycle",
                run_id=run_id,
                subject=control,
                from_state=current,
                to_state=to_state,
                reason=reason,
                detail=detail,
            )

    def lifecycle_events(self, run_id: str) -> tuple[JournalRow, ...]:
        rows = self._query(
            "SELECT * FROM journal WHERE run_id = ? AND kind = 'lifecycle' ORDER BY cursor",
            (run_id,),
        )
        return tuple(_journal(row) for row in rows)

    # -- groups -----------------------------------------------------------

    def open_group(
        self,
        *,
        group_id: str,
        run_id: str,
        cardinality: int,
        pin: Mapping[str, Any],
        pin_digest: str,
        policy_revision: int,
    ) -> GroupRow:
        """Idempotent: opening an open group returns the row already on record."""

        existing = self.group(group_id, required=False)
        if existing is not None:
            return existing
        if cardinality < 1:
            raise StoreError("group cardinality must be positive")
        moment = self.clock.now()
        with self._write() as cur:
            cur.execute(
                "INSERT INTO attempt_groups (group_id, run_id, cardinality, pin, pin_digest, "
                "policy_revision, state, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    group_id,
                    run_id,
                    int(cardinality),
                    _encode(pin),
                    pin_digest,
                    int(policy_revision),
                    GROUP_OPEN,
                    moment,
                ),
            )
            self._append(
                cur,
                "group_opened",
                run_id=run_id,
                subject=group_id,
                to_state=GROUP_OPEN,
                detail={"cardinality": int(cardinality), "pin_digest": pin_digest},
            )
        group = self.group(group_id)
        return group

    def group(self, group_id: str, *, required: bool = True) -> GroupRow | None:
        row = self._one(
            "SELECT *, rowid AS sequence FROM attempt_groups WHERE group_id = ?", (group_id,)
        )
        if row is None:
            if required:
                raise UnknownGroupError(f"group {group_id!r} was never opened")
            return None
        return _group(row)

    def groups_in_state(self, state: str, *, run_id: str | None = None) -> tuple[GroupRow, ...]:
        if state not in GROUP_STATES:
            raise StoreError(f"unknown group state {state!r}")
        rows = self._query(
            "SELECT *, rowid AS sequence FROM attempt_groups WHERE state = ? "
            "AND (? IS NULL OR run_id = ?) ORDER BY rowid",
            (state, run_id, run_id),
        )
        return tuple(_group(row) for row in rows)

    def transition_group(
        self,
        group_id: str,
        to_state: str,
        *,
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> GroupRow:
        group = self.group(group_id)
        assert group is not None
        if to_state not in GROUP_STATES:
            raise StoreError(f"unknown group state {to_state!r}")
        if to_state not in GROUP_TRANSITIONS[group.state]:
            raise TransitionError(
                f"group {group_id} may not move {group.state} -> {to_state}"
            )
        closed = None if to_state in (GROUP_OPEN, GROUP_COMPLETE) else self.clock.now()
        with self._write() as cur:
            cur.execute(
                "UPDATE attempt_groups SET state = ?, closed_at = COALESCE(?, closed_at) "
                "WHERE group_id = ?",
                (to_state, closed, group_id),
            )
            self._append(
                cur,
                "group_transition",
                run_id=group.run_id,
                subject=group_id,
                from_state=group.state,
                to_state=to_state,
                reason=reason,
                detail=detail,
            )
        moved = self.group(group_id)
        assert moved is not None
        return moved

    def group_members(
        self, group_id: str, *, active_only: bool = False
    ) -> tuple[MembershipRow, ...]:
        sql = "SELECT * FROM group_members WHERE group_id = ?"
        if active_only:
            sql += " AND active = 1"
        rows = self._query(sql + " ORDER BY member_seq", (group_id,))
        return tuple(_membership(row) for row in rows)

    def membership_snapshot(self, group_id: str) -> tuple[Mapping[str, Any], ...]:
        """Membership plus each member's terminal status, for an abandonment record."""

        rows = self._query(
            "SELECT m.attempt_id, m.sample_index, m.active, m.disposition, "
            "m.replaced_attempt_id, a.state, r.kind FROM group_members m "
            "JOIN attempts a ON a.attempt_id = m.attempt_id "
            "LEFT JOIN results r ON r.attempt_id = m.attempt_id "
            "WHERE m.group_id = ? ORDER BY m.member_seq",
            (group_id,),
        )
        return tuple(
            {
                "attempt_id": row["attempt_id"],
                "sample_index": int(row["sample_index"]),
                "active": bool(row["active"]),
                "disposition": row["disposition"],
                "replaced_attempt_id": row["replaced_attempt_id"],
                "state": row["state"],
                "result_kind": row["kind"],
            }
            for row in rows
        )

    # -- attempts ---------------------------------------------------------

    def admit_attempt(
        self,
        *,
        attempt_id: str,
        idempotency_key: str,
        run_id: str,
        group_id: str,
        sample_index: int,
        task_id: str,
        seed: int,
        policy_revision: int,
        agent_instance_id: str | None = None,
        team_id: str | None = None,
        replaced_attempt_id: str | None = None,
        replacement_index: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[AttemptRow, bool]:
        """Admit one sample. Retrying a key yields the same logical attempt.

        Returns the row and whether this call created it. A key reused for a
        different group or sample is a bug, not an idempotent retry, and raises.
        """

        existing = self._one(
            "SELECT *, rowid AS sequence FROM attempts WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        if existing is not None:
            row = _attempt(existing)
            if (row.group_id, row.sample_index, row.run_id) != (group_id, sample_index, run_id):
                raise IdempotencyError(
                    f"idempotency key {idempotency_key!r} already names attempt "
                    f"{row.attempt_id} in group {row.group_id} sample {row.sample_index}"
                )
            return row, False
        self.group(group_id)
        moment = self.clock.now()
        with self._write() as cur:
            cur.execute(
                "INSERT INTO attempts (attempt_id, idempotency_key, run_id, group_id, "
                "sample_index, task_id, seed, policy_revision, state, queue, dispatch_count, "
                "replacement_index, replaced_attempt_id, agent_instance_id, team_id, metadata, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, "
                "?, ?, ?)",
                (
                    attempt_id,
                    idempotency_key,
                    run_id,
                    group_id,
                    int(sample_index),
                    task_id,
                    int(seed),
                    int(policy_revision),
                    "queued",
                    QUEUE_ROLLOUT,
                    int(replacement_index),
                    replaced_attempt_id,
                    agent_instance_id,
                    team_id,
                    _encode(metadata),
                    moment,
                    moment,
                ),
            )
            if replaced_attempt_id is not None:
                cur.execute(
                    "UPDATE group_members SET active = 0, disposition = ? "
                    "WHERE group_id = ? AND attempt_id = ?",
                    (MEMBERSHIP_REPLACED, group_id, replaced_attempt_id),
                )
            cur.execute(
                "INSERT INTO group_members (group_id, attempt_id, sample_index, active, "
                "disposition, replaced_attempt_id, recorded_at) VALUES (?, ?, ?, 1, ?, ?, ?)",
                (
                    group_id,
                    attempt_id,
                    int(sample_index),
                    MEMBERSHIP_HELD,
                    replaced_attempt_id,
                    moment,
                ),
            )
            self._append(
                cur,
                "attempt_replaced" if replaced_attempt_id else "attempt_admitted",
                run_id=run_id,
                subject=attempt_id,
                to_state="queued",
                to_queue=QUEUE_ROLLOUT,
                reason="replacement" if replaced_attempt_id else "admission",
                detail={
                    "group_id": group_id,
                    "sample_index": int(sample_index),
                    "idempotency_key": idempotency_key,
                    "task_id": task_id,
                    "seed": int(seed),
                    "policy_revision": int(policy_revision),
                    "replaced_attempt_id": replaced_attempt_id,
                    "replacement_index": int(replacement_index),
                },
            )
        return self.attempt(attempt_id), True

    def attempt(self, attempt_id: str) -> AttemptRow:
        row = self._one(
            "SELECT *, rowid AS sequence FROM attempts WHERE attempt_id = ?", (attempt_id,)
        )
        if row is None:
            raise UnknownAttemptError(f"attempt {attempt_id!r} was never admitted")
        return _attempt(row)

    def attempt_for_key(self, idempotency_key: str) -> AttemptRow | None:
        row = self._one(
            "SELECT *, rowid AS sequence FROM attempts WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        return _attempt(row) if row is not None else None

    def transition_attempt(
        self,
        attempt_id: str,
        to_state: str,
        *,
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
        result_payload: Mapping[str, Any] | None = None,
    ) -> AttemptRow:
        """Move one attempt. A terminal target also writes its one result row."""

        attempt = self.attempt(attempt_id)
        if to_state not in ATTEMPT_STATES:
            raise StoreError(f"unknown attempt state {to_state!r}")
        if to_state in TERMINAL_ATTEMPT_STATES:
            prior = self.result(attempt_id)
            if prior is not None:
                raise TerminalResultError(
                    f"attempt {attempt_id} already has one terminal result "
                    f"({prior.kind}); exactly one is legal"
                )
        if to_state not in ATTEMPT_TRANSITIONS[attempt.state]:
            raise TransitionError(
                f"attempt {attempt_id} may not move {attempt.state} -> {to_state}"
            )
        queue = QUEUE_FOR_STATE[to_state]
        moment = self.clock.now()
        dispatch_count = attempt.dispatch_count + (1 if to_state == "running" else 0)
        with self._write() as cur:
            cur.execute(
                "UPDATE attempts SET state = ?, queue = ?, dispatch_count = ?, updated_at = ? "
                "WHERE attempt_id = ?",
                (to_state, queue, dispatch_count, moment, attempt_id),
            )
            if to_state in TERMINAL_ATTEMPT_STATES:
                kind = RESULT_KIND_FOR_STATE[to_state]
                cur.execute(
                    "INSERT INTO results (attempt_id, kind, terminal_status, payload, "
                    "recorded_at) VALUES (?, ?, ?, ?, ?)",
                    (attempt_id, kind, to_state, _encode(result_payload), moment),
                )
                self._append(
                    cur,
                    "result_recorded",
                    run_id=attempt.run_id,
                    subject=attempt_id,
                    to_state=to_state,
                    reason=kind,
                    detail={"group_id": attempt.group_id, "sample_index": attempt.sample_index},
                )
            self._append(
                cur,
                "attempt_transition",
                run_id=attempt.run_id,
                subject=attempt_id,
                from_state=attempt.state,
                to_state=to_state,
                from_queue=attempt.queue,
                to_queue=queue,
                reason=reason,
                detail=detail,
            )
        return self.attempt(attempt_id)

    def result(self, attempt_id: str) -> ResultRow | None:
        row = self._one("SELECT * FROM results WHERE attempt_id = ?", (attempt_id,))
        if row is None:
            return None
        return ResultRow(
            attempt_id=str(row["attempt_id"]),
            kind=str(row["kind"]),
            terminal_status=str(row["terminal_status"]),
            recorded_at=float(row["recorded_at"]),
            payload=_decode(row["payload"]),
        )

    def attempts_in_state(self, state: str, *, run_id: str | None = None) -> tuple[AttemptRow, ...]:
        if state not in ATTEMPT_STATES:
            raise StoreError(f"unknown attempt state {state!r}")
        rows = self._query(
            "SELECT *, rowid AS sequence FROM attempts WHERE state = ? "
            "AND (? IS NULL OR run_id = ?) ORDER BY rowid",
            (state, run_id, run_id),
        )
        return tuple(_attempt(row) for row in rows)

    def attempts_in_group(self, group_id: str) -> tuple[AttemptRow, ...]:
        rows = self._query(
            "SELECT *, rowid AS sequence FROM attempts WHERE group_id = ? "
            "ORDER BY sample_index, rowid",
            (group_id,),
        )
        return tuple(_attempt(row) for row in rows)

    def queue_depth(self, queue: str, *, run_id: str | None = None) -> int:
        if queue == QUEUE_TRAIN_READY:
            return len(self.groups_in_state(GROUP_TRAIN_READY, run_id=run_id))
        if queue not in QUEUES:
            raise StoreError(f"unknown queue {queue!r}")
        row = self._one(
            "SELECT COUNT(*) AS depth FROM attempts WHERE queue = ? AND (? IS NULL OR run_id = ?)",
            (queue, run_id, run_id),
        )
        return int(row["depth"]) if row is not None else 0

    def in_flight(self, *, run_id: str | None = None) -> tuple[AttemptRow, ...]:
        """Attempts a container is holding: running, or waiting on deferred scoring."""

        rows = self._query(
            "SELECT *, rowid AS sequence FROM attempts WHERE state IN ('running', "
            "'awaiting_score') AND (? IS NULL OR run_id = ?) ORDER BY rowid",
            (run_id, run_id),
        )
        return tuple(_attempt(row) for row in rows)

    def non_terminal_attempts(self, *, run_id: str | None = None) -> tuple[AttemptRow, ...]:
        placeholders = ", ".join("?" for _ in TERMINAL_ATTEMPT_STATES)
        terminal = tuple(sorted(TERMINAL_ATTEMPT_STATES))
        rows = self._query(
            f"SELECT *, rowid AS sequence FROM attempts WHERE state NOT IN ({placeholders}) "
            "AND (? IS NULL OR run_id = ?) ORDER BY rowid",
            (*terminal, run_id, run_id),
        )
        return tuple(_attempt(row) for row in rows)

    def attempts_without_result(self, *, run_id: str | None = None) -> tuple[AttemptRow, ...]:
        """Accepted attempts with no terminal result yet. Empty after a stop."""

        rows = self._query(
            "SELECT a.*, a.rowid AS sequence FROM attempts a "
            "LEFT JOIN results r ON r.attempt_id = a.attempt_id "
            "WHERE r.attempt_id IS NULL AND (? IS NULL OR a.run_id = ?) ORDER BY a.rowid",
            (run_id, run_id),
        )
        return tuple(_attempt(row) for row in rows)

    def dispatchable(
        self, *, limit: int, run_id: str | None = None, oldest_group_first: bool = True
    ) -> tuple[AttemptRow, ...]:
        """Queued attempts, preferring completion of the oldest still-open group."""

        order = "g.rowid, a.sample_index, a.rowid" if oldest_group_first else "a.rowid"
        rows = self._query(
            "SELECT a.*, a.rowid AS sequence FROM attempts a "
            "JOIN attempt_groups g ON g.group_id = a.group_id "
            "WHERE a.state = 'queued' AND g.state = ? AND (? IS NULL OR a.run_id = ?) "
            f"ORDER BY {order} LIMIT ?",
            (GROUP_OPEN, run_id, run_id, int(limit)),
        )
        return tuple(_attempt(row) for row in rows)

    # -- leases -----------------------------------------------------------

    def grant_lease(
        self,
        *,
        attempt_id: str,
        holder: str,
        expires_at: float,
        straggler_deadline: float,
        lease_id: str | None = None,
    ) -> LeaseRow:
        attempt = self.attempt(attempt_id)
        if self.active_lease_for(attempt_id) is not None:
            raise LeaseStoreError(f"attempt {attempt_id} already holds an active lease")
        count = self._one(
            "SELECT COUNT(*) AS n FROM leases WHERE attempt_id = ?", (attempt_id,)
        )
        index = (int(count["n"]) if count is not None else 0) + 1
        identifier = lease_id or f"{attempt_id}#l{index}"
        moment = self.clock.now()
        with self._write() as cur:
            cur.execute(
                "INSERT INTO leases (lease_id, attempt_id, holder, granted_at, expires_at, "
                "straggler_deadline, heartbeats, state) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    identifier,
                    attempt_id,
                    holder,
                    moment,
                    float(expires_at),
                    float(straggler_deadline),
                    LEASE_ACTIVE,
                ),
            )
            self._append(
                cur,
                "lease_granted",
                run_id=attempt.run_id,
                subject=identifier,
                to_state=LEASE_ACTIVE,
                detail={
                    "attempt_id": attempt_id,
                    "holder": holder,
                    "expires_at": float(expires_at),
                    "straggler_deadline": float(straggler_deadline),
                },
            )
        return self.lease(identifier)

    def renew_lease(self, lease_id: str, *, expires_at: float) -> LeaseRow:
        lease = self.lease(lease_id)
        if lease.state != LEASE_ACTIVE:
            raise LeaseStoreError(f"lease {lease_id} is {lease.state}, not active")
        attempt = self.attempt(lease.attempt_id)
        with self._write() as cur:
            cur.execute(
                "UPDATE leases SET expires_at = ?, heartbeats = heartbeats + 1 WHERE lease_id = ?",
                (float(expires_at), lease_id),
            )
            self._append(
                cur,
                "lease_renewed",
                run_id=attempt.run_id,
                subject=lease_id,
                to_state=LEASE_ACTIVE,
                detail={"attempt_id": lease.attempt_id, "expires_at": float(expires_at)},
            )
        return self.lease(lease_id)

    def close_lease(self, lease_id: str, *, state: str, reason: str = "") -> LeaseRow:
        if state not in LEASE_STATES or state == LEASE_ACTIVE:
            raise LeaseStoreError(f"{state!r} is not a closed lease state")
        lease = self.lease(lease_id)
        if lease.state != LEASE_ACTIVE:
            return lease
        attempt = self.attempt(lease.attempt_id)
        moment = self.clock.now()
        with self._write() as cur:
            cur.execute(
                "UPDATE leases SET state = ?, closed_at = ? WHERE lease_id = ?",
                (state, moment, lease_id),
            )
            self._append(
                cur,
                "lease_closed",
                run_id=attempt.run_id,
                subject=lease_id,
                from_state=LEASE_ACTIVE,
                to_state=state,
                reason=reason,
                detail={"attempt_id": lease.attempt_id},
            )
        return self.lease(lease_id)

    def close_leases_for(self, attempt_id: str, *, state: str, reason: str = "") -> int:
        closed = 0
        for lease in self.active_leases(attempt_id=attempt_id):
            self.close_lease(lease.lease_id, state=state, reason=reason)
            closed += 1
        return closed

    def lease(self, lease_id: str) -> LeaseRow:
        row = self._one("SELECT * FROM leases WHERE lease_id = ?", (lease_id,))
        if row is None:
            raise LeaseStoreError(f"lease {lease_id!r} is not on record")
        return _lease(row)

    def leases_for(self, attempt_id: str) -> tuple[LeaseRow, ...]:
        rows = self._query(
            "SELECT * FROM leases WHERE attempt_id = ? ORDER BY granted_at, lease_id",
            (attempt_id,),
        )
        return tuple(_lease(row) for row in rows)

    def active_lease_for(self, attempt_id: str) -> LeaseRow | None:
        leases = self.active_leases(attempt_id=attempt_id)
        return leases[0] if leases else None

    def active_leases(
        self,
        *,
        attempt_id: str | None = None,
        expires_at_or_before: float | None = None,
        deadline_at_or_before: float | None = None,
    ) -> tuple[LeaseRow, ...]:
        rows = self._query(
            "SELECT * FROM leases WHERE state = ? AND (? IS NULL OR attempt_id = ?) "
            "AND (? IS NULL OR expires_at <= ?) AND (? IS NULL OR straggler_deadline <= ?) "
            "ORDER BY granted_at, lease_id",
            (
                LEASE_ACTIVE,
                attempt_id,
                attempt_id,
                expires_at_or_before,
                expires_at_or_before,
                deadline_at_or_before,
                deadline_at_or_before,
            ),
        )
        return tuple(_lease(row) for row in rows)

    # -- journal and recovery ---------------------------------------------

    def head_cursor(self) -> int:
        row = self._one("SELECT COALESCE(MAX(cursor), 0) AS head FROM journal")
        return int(row["head"]) if row is not None else 0

    def journal_since(
        self, cursor: int = 0, *, limit: int | None = None, run_id: str | None = None
    ) -> tuple[JournalRow, ...]:
        sql = (
            "SELECT * FROM journal WHERE cursor > ? AND (? IS NULL OR run_id = ?) ORDER BY cursor"
        )
        params: list[Any] = [int(cursor), run_id, run_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return tuple(_journal(row) for row in self._query(sql, params))

    def event_page(self, run_id: str, cursor: int = 0, limit: int = 500) -> dict:
        """Compact event references over the same transactional queue journal."""
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError('invalid journal cursor or limit')
        log_id = self._query("SELECT value FROM meta WHERE key='event_log_id'", ())[0][0]
        rows = self.journal_since(cursor, limit=limit+1, run_id=run_id)
        events = [{'event_id': f'{log_id}:{r.cursor}', 'sequence': r.cursor,
            'event_type': 'runtime.'+r.kind, 'timestamp': None,
            'payload': {'segment_run_id': run_id, 'subject': r.subject,
                'from_state': r.from_state, 'to_state': r.to_state, 'source_clock_seconds': r.at,
                'from_queue': r.from_queue, 'to_queue': r.to_queue,
                'evidence_reference': {'journal': self.path, 'cursor': r.cursor}}} for r in rows[:limit]]
        return {'log_id': log_id, 'events': events, 'has_more': len(rows)>limit,
                'next_sequence': events[-1]['sequence'] if events else cursor}

    def recover(self, run_id: str) -> RecoverySnapshot:
        """What the queues held at the last durable write, after a restart."""

        return RecoverySnapshot(
            run_id=run_id,
            lifecycle_state=self.lifecycle_state(run_id),
            cursor=self.head_cursor(),
            queued=self.attempts_in_state("queued", run_id=run_id),
            active=self.attempts_in_state("running", run_id=run_id),
            awaiting_score=self.attempts_in_state("awaiting_score", run_id=run_id),
            scored=self.attempts_in_state("scored", run_id=run_id),
            open_groups=self.groups_in_state(GROUP_OPEN, run_id=run_id),
            complete_groups=self.groups_in_state(GROUP_COMPLETE, run_id=run_id),
            train_ready=self.groups_in_state(GROUP_TRAIN_READY, run_id=run_id),
            live_leases=self.active_leases(),
            attempts_without_result=self.attempts_without_result(run_id=run_id),
        )


def _attempt(row: sqlite3.Row) -> AttemptRow:
    return AttemptRow(
        attempt_id=str(row["attempt_id"]),
        idempotency_key=str(row["idempotency_key"]),
        run_id=str(row["run_id"]),
        group_id=str(row["group_id"]),
        sample_index=int(row["sample_index"]),
        task_id=str(row["task_id"]),
        seed=int(row["seed"]),
        policy_revision=int(row["policy_revision"]),
        state=str(row["state"]),
        queue=row["queue"],
        dispatch_count=int(row["dispatch_count"]),
        replacement_index=int(row["replacement_index"]),
        replaced_attempt_id=row["replaced_attempt_id"],
        agent_instance_id=row["agent_instance_id"],
        team_id=row["team_id"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        sequence=int(row["sequence"]),
        metadata=_decode(row["metadata"]),
    )


def _group(row: sqlite3.Row) -> GroupRow:
    return GroupRow(
        group_id=str(row["group_id"]),
        run_id=str(row["run_id"]),
        cardinality=int(row["cardinality"]),
        pin_digest=str(row["pin_digest"]),
        policy_revision=int(row["policy_revision"]),
        state=str(row["state"]),
        opened_at=float(row["opened_at"]),
        closed_at=row["closed_at"],
        sequence=int(row["sequence"]),
        pin=_decode(row["pin"]),
    )


def _membership(row: sqlite3.Row) -> MembershipRow:
    return MembershipRow(
        group_id=str(row["group_id"]),
        attempt_id=str(row["attempt_id"]),
        sample_index=int(row["sample_index"]),
        active=bool(row["active"]),
        disposition=str(row["disposition"]),
        replaced_attempt_id=row["replaced_attempt_id"],
        recorded_at=float(row["recorded_at"]),
    )


def _lease(row: sqlite3.Row) -> LeaseRow:
    return LeaseRow(
        lease_id=str(row["lease_id"]),
        attempt_id=str(row["attempt_id"]),
        holder=str(row["holder"]),
        granted_at=float(row["granted_at"]),
        expires_at=float(row["expires_at"]),
        straggler_deadline=float(row["straggler_deadline"]),
        heartbeats=int(row["heartbeats"]),
        state=str(row["state"]),
        closed_at=row["closed_at"],
    )


def _journal(row: sqlite3.Row) -> JournalRow:
    return JournalRow(
        cursor=int(row["cursor"]),
        kind=str(row["kind"]),
        run_id=str(row["run_id"]),
        subject=str(row["subject"]),
        at=float(row["at"]),
        from_state=row["from_state"],
        to_state=row["to_state"],
        from_queue=row["from_queue"],
        to_queue=row["to_queue"],
        reason=str(row["reason"]),
        detail=_decode(row["detail"]),
    )
