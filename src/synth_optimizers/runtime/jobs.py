"""Durable job store, append-only journal, and content-addressed keys."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts.training_schemas import LIFECYCLE_STATES, TERMINAL_STATES


RUNNER_VERSION = "synth-optimizers.training.v1"
PRODUCER_SERVICE = "synth-optimizers"
ATTEMPT_ID = "attempt-1"


def flatten_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    metrics = body.get("metrics")
    loss = body.get("train_loss")
    if loss is None:
        loss = body.get("loss")
    if loss is None and isinstance(metrics, Mapping):
        loss = metrics.get("loss")
    if loss is not None:
        body.setdefault("loss", loss)
        body.setdefault("train_loss", loss)
        body.setdefault("trainLoss", loss)
    if "step" not in body and body.get("update") is not None:
        body["step"] = body["update"]
    return body


class JobStoreError(RuntimeError):
    pass


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_payload(value: Mapping[str, Any] | str | bytes) -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def idempotency_key(
    *,
    algorithm_id: str,
    implementation_version: str,
    provider: str,
    model_id: str,
    dataset_digest: str,
    split_manifest_digest: str,
    renderer_version: str,
    training_config: Mapping[str, Any],
    reward_version: str,
    seed: int,
    runner_version: str,
    repeat_index: int,
) -> str:
    return digest_payload(
        {
            "algorithm_id": algorithm_id,
            "implementation_version": implementation_version,
            "provider": provider,
            "model_id": model_id,
            "dataset_digest": dataset_digest,
            "split_manifest_digest": split_manifest_digest,
            "renderer_version": renderer_version,
            "training_config": dict(training_config),
            "reward_version": reward_version,
            "seed": seed,
            "runner_version": runner_version,
            "repeat_index": repeat_index,
        }
    )


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class TrainingJob:
    job_id: str
    algorithm_id: str
    implementation_version: str
    provider: str
    model_id: str
    state: str
    idempotency_key: str
    config_json: str
    config_digest: str
    owner: str | None = None
    heartbeat_at: str | None = None
    resume_token: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self, path: str | Path) -> None:
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(database)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._events = threading.Condition(self._lock)
        self._ownership = threading.local()
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    algorithm_id TEXT NOT NULL,
                    implementation_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    config_json TEXT NOT NULL,
                    config_digest TEXT NOT NULL,
                    owner TEXT,
                    heartbeat_at TEXT,
                    resume_token TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS training_artifacts (
                    job_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    body BLOB NOT NULL,
                    PRIMARY KEY (job_id, name)
                );
                CREATE TABLE IF NOT EXISTS training_receipts (
                    job_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS reducer_checkpoints (
                    job_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL
                );
                """
            )
            self._db.commit()

    def persist_prepared(
        self,
        *,
        algorithm_id: str,
        implementation_version: str,
        provider: str,
        model_id: str,
        idempotency_key: str,
        config: Mapping[str, Any],
        job_id: str | None = None,
    ) -> TrainingJob:
        config_json = canonical_json(config)
        config_digest = digest_payload(config_json)
        with self._lock:
            existing = self._db.execute(
                "SELECT * FROM training_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["config_digest"] != config_digest:
                    raise JobStoreError("idempotency key reused with a different configuration")
                return self._job_from_row(existing)
            now = utcnow()
            resolved_id = job_id or f"{algorithm_id}_{uuid.uuid4().hex}"
            self._db.execute(
                """
                INSERT INTO training_jobs(
                    job_id, algorithm_id, implementation_version, provider, model_id, state,
                    idempotency_key, config_json, config_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    algorithm_id,
                    implementation_version,
                    provider,
                    model_id,
                    idempotency_key,
                    config_json,
                    config_digest,
                    now,
                    now,
                ),
            )
            self._db.commit()
            return self.require(resolved_id)

    def require(self, job_id: str) -> TrainingJob:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobStoreError(f"unknown training job {job_id}")
            return self._job_from_row(row)

    def lookup_idempotency(self, key: str) -> TrainingJob | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM training_jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            return None if row is None else self._job_from_row(row)

    @contextmanager
    def _write(self, job_id: str):
        """Serialize compare-and-write across connections and fence worker writes."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                owner = getattr(self._ownership, "owner", None)
                if owner is not None and self.require(job_id).owner != owner:
                    raise JobStoreError("stale worker fenced")
                yield
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @contextmanager
    def owned(self, owner: str):
        previous = getattr(self._ownership, "owner", None)
        self._ownership.owner = owner
        try:
            yield
        finally:
            self._ownership.owner = previous

    def release(self, job_id: str, owner: str) -> None:
        with self._write(job_id):
            self._db.execute(
                "UPDATE training_jobs SET owner = NULL, heartbeat_at = NULL WHERE job_id = ? AND owner = ?",
                (job_id, owner),
            )

    def claim(self, job_id: str, owner: str, *, stale_after_seconds: int = 30) -> TrainingJob:
        with self._write(job_id):
            job = self.require(job_id)
            if job.state in TERMINAL_STATES:
                return job
            if job.owner and not self._stale(job, stale_after_seconds):
                raise JobStoreError(f"job {job_id} already has an active owner")
            now = utcnow()
            self._db.execute(
                """UPDATE training_jobs SET owner = ?, heartbeat_at = ?,
                state = CASE WHEN state = 'prepared' THEN 'running' ELSE state END,
                updated_at = ? WHERE job_id = ?""", (owner, now, now, job_id),
            )
            return self.require(job_id)

    def heartbeat(self, job_id: str, owner: str) -> None:
        with self._write(job_id):
            job = self.require(job_id)
            if job.owner != owner:
                raise JobStoreError("heartbeat from non-owner")
            now = utcnow()
            self._db.execute(
                "UPDATE training_jobs SET heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (now, now, job_id),
            )

    def transition(self, job_id: str, state: str, *, error: str | None = None) -> TrainingJob:
        if state not in LIFECYCLE_STATES:
            raise JobStoreError(f"invalid lifecycle state {state}")
        with self._write(job_id):
            job = self.require(job_id)
            if job.state in TERMINAL_STATES:
                return job
            if job.state == "stop_requested" and state not in {"cancelled", "blocked_uncertain"}:
                return job
            if job.state == "pause_requested" and state in {"running", "evaluating", "materializing"}:
                return job
            now = utcnow()
            self._db.execute(
                "UPDATE training_jobs SET state = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (state, error, now, job_id),
            )
            self._insert_event(job_id, "training.lifecycle", {"state": state, "error": error}, state)
            self._events.notify_all()
            return self.require(job_id)

    def set_resume_token(self, job_id: str, token: str) -> None:
        with self._write(job_id):
            self._db.execute(
                "UPDATE training_jobs SET resume_token = ?, updated_at = ? WHERE job_id = ?",
                (token, utcnow(), job_id),
            )

    def _insert_event(self, job_id, kind, payload, phase):
        sequence = self._latest_sequence(job_id) + 1
        occurred_at, event_id = utcnow(), f"evt_{uuid.uuid4().hex}"
        self._db.execute(
            "INSERT INTO training_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, sequence, event_id, kind, phase, occurred_at, canonical_json(dict(payload))),
        )
        return self._public_event(
            job_id=job_id, algorithm_id=self.require(job_id).algorithm_id,
            event_id=event_id, sequence=sequence, kind=kind, phase=phase,
            occurred_at=occurred_at, payload=payload,
        )

    def append_event(
        self, job_id: str, kind: str, payload: Mapping[str, Any], *, phase: str
    ) -> dict[str, Any]:
        with self._write(job_id):
            event = self._insert_event(job_id, kind, payload, phase)
            self._events.notify_all()
            return event

    def append_event_once(self, job_id, kind, payload, *, phase):
        with self._write(job_id):
            row = self._db.execute(
                """SELECT e.*, j.algorithm_id FROM training_events e
                JOIN training_jobs j ON e.job_id=j.job_id
                WHERE e.job_id=? AND e.kind=? AND e.payload_json=? ORDER BY sequence LIMIT 1""",
                (job_id, kind, canonical_json(dict(payload))),
            ).fetchone()
            if row is not None:
                return self._event_from_row(row)
            event = self._insert_event(job_id, kind, payload, phase)
            self._events.notify_all()
            return event

    def wait_for_events(
        self, job_id: str, after_sequence: int, *, timeout: float = 1.0
    ) -> None:
        """Block until a later event, a terminal state, or timeout.

        The journal stays the record. Waiters are a live mirror; a timeout is
        a heartbeat opportunity, not a gap in the run.
        """

        with self._events:
            if self._latest_sequence(job_id) > after_sequence:
                return
            if self.require(job_id).state in TERMINAL_STATES:
                return
            self._events.wait(timeout=max(0.05, timeout))

    def _latest_sequence(self, job_id: str) -> int:
        last = self._db.execute(
            "SELECT MAX(sequence) FROM training_events WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        return int(last or 0)

    def events(self, job_id: str, *, after_sequence: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT e.*, j.algorithm_id AS algorithm_id
                FROM training_events e
                JOIN training_jobs j ON j.job_id = e.job_id
                WHERE e.job_id = ? AND e.sequence > ?
                ORDER BY e.sequence ASC
                LIMIT ?
                """,
                (job_id, after_sequence, max(1, min(5_000, limit))),
            ).fetchall()
            return [self._event_from_row(row) for row in rows]

    def status_events(self, job_id: str, *, byte_limit: int = 32768) -> list[dict[str, Any]]:
        """A bounded recent preview; the paged journal remains the complete source."""
        with self._lock:
            latest = self._latest_sequence(job_id)
        rows = self.events(job_id, after_sequence=max(0, latest - 100), limit=100)
        selected, size = [], 2
        for row in reversed(rows):
            encoded = json.dumps(row).encode()
            if len(encoded) > byte_limit // 2:
                row = {**row, "payload": {"source_sequence": row["sequence"],
                       "source_url": f"/v1/runs/{job_id}/optimizer-events?after_sequence={row['sequence']-1}&limit=1",
                       "payload_digest": digest_payload(row["payload"]), "omitted_from_preview": True}}
                encoded = json.dumps(row).encode()
            if size + len(encoded) + 2 > byte_limit:
                break
            selected.append(row)
            size += len(encoded) + 2
        return list(reversed(selected))

    def put_artifact(self, job_id: str, name: str, body: bytes, *, content_type: str) -> str:
        digest = digest_payload(body)
        with self._write(job_id):
            self._db.execute(
                """
                INSERT OR REPLACE INTO training_artifacts(job_id, name, content_type, digest, body)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, name, content_type, digest, body),
            )
            self._insert_event(job_id, "training.artifact", {"name": name, "digest": digest,
                               "content_type": content_type}, self.require(job_id).state)
        return digest

    def artifact(self, job_id: str, name: str) -> tuple[bytes, str, str]:
        with self._lock:
            row = self._db.execute(
                "SELECT body, content_type, digest FROM training_artifacts WHERE job_id = ? AND name = ?",
                (job_id, name),
            ).fetchone()
            if row is None:
                raise JobStoreError(f"unknown artifact {name}")
            return bytes(row["body"]), str(row["content_type"]), str(row["digest"])

    def artifacts(self, job_id: str) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT name, content_type, digest FROM training_artifacts WHERE job_id = ? ORDER BY name",
                (job_id,),
            ).fetchall()
            return [
                {"name": row["name"], "content_type": row["content_type"], "digest": row["digest"]}
                for row in rows
            ]

    def put_receipt(self, job_id: str, request_id: str, payload: Mapping[str, Any]) -> None:
        with self._write(job_id):
            self._db.execute(
                """
                INSERT OR REPLACE INTO training_receipts(job_id, request_id, payload_json)
                VALUES (?, ?, ?)
                """,
                (job_id, request_id, canonical_json(payload)),
            )

            self._insert_event(job_id, "training.receipt", dict(payload), self.require(job_id).state)

    def receipts(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload_json FROM training_receipts WHERE job_id = ? ORDER BY request_id",
                (job_id,),
            ).fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    def save_reducer(self, job_id: str, sequence: int, snapshot: Mapping[str, Any]) -> None:
        with self._write(job_id):
            self._db.execute(
                """
                INSERT OR REPLACE INTO reducer_checkpoints(job_id, sequence, snapshot_json)
                VALUES (?, ?, ?)
                """,
                (job_id, sequence, canonical_json(snapshot)),
            )

    def load_reducer(self, job_id: str) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT sequence, snapshot_json FROM reducer_checkpoints WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return int(row["sequence"]), json.loads(row["snapshot_json"])

    def request_pause(self, job_id):
        job = self.require(job_id)
        if job.state in TERMINAL_STATES or job.state in {"stop_requested", "paused"}:
            return job
        return self.transition(job_id, "pause_requested")

    def resume_prepared(self, job_id):
        with self._write(job_id):
            job = self.require(job_id)
            if job.owner and not self._stale(job, 30):
                return job
            if job.state == "paused":
                self._db.execute("UPDATE training_jobs SET state='prepared', updated_at=? WHERE job_id=?",
                                 (utcnow(), job_id))
                self._insert_event(job_id, "training.lifecycle", {"state": "prepared", "error": None}, "prepared")
            return self.require(job_id)

    def cancellation_requested(self, job_id: str) -> bool:
        return self.require(job_id).state in {"stop_requested", "cancelled"}

    def request_cancel(self, job_id: str) -> TrainingJob:
        with self._write(job_id):
            job = self.require(job_id)
            if job.state in TERMINAL_STATES:
                return job
            # An unowned prepared job has no admitted provider work to drain.
            state = "cancelled" if job.state == "prepared" and job.owner is None else "stop_requested"
            self._db.execute(
                "UPDATE training_jobs SET state = ?, updated_at = ? WHERE job_id = ?",
                (state, utcnow(), job_id),
            )
            self._insert_event(job_id, "training.lifecycle", {"state": state, "error": None}, state)
            self._events.notify_all()
            return self.require(job_id)

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _stale(job: TrainingJob, stale_after_seconds: int) -> bool:
        if not job.heartbeat_at:
            return True
        try:
            heartbeat = datetime.fromisoformat(job.heartbeat_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        return (datetime.now(UTC) - heartbeat).total_seconds() > stale_after_seconds

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> TrainingJob:
        return TrainingJob(
            job_id=row["job_id"],
            algorithm_id=row["algorithm_id"],
            implementation_version=row["implementation_version"],
            provider=row["provider"],
            model_id=row["model_id"],
            state=row["state"],
            idempotency_key=row["idempotency_key"],
            config_json=row["config_json"],
            config_digest=row["config_digest"],
            owner=row["owner"],
            heartbeat_at=row["heartbeat_at"],
            resume_token=row["resume_token"],
            error=row["error"],
        )

    @classmethod
    def _event_from_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._public_event(
            job_id=row["job_id"],
            algorithm_id=row["algorithm_id"],
            event_id=row["event_id"],
            sequence=row["sequence"],
            kind=row["kind"],
            phase=row["phase"],
            occurred_at=row["occurred_at"],
            payload=json.loads(row["payload_json"]),
        )

    @staticmethod
    def _public_event(
        *,
        job_id: str,
        algorithm_id: str,
        event_id: str,
        sequence: int,
        kind: str,
        phase: str,
        occurred_at: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        from .jobs import ATTEMPT_ID, flatten_metric_payload

        return {
            "schema_version": "training.event.v1",
            "event_id": event_id,
            "job_id": job_id,
            "optimizer_run_id": job_id,
            "algorithm_id": algorithm_id,
            "attempt_id": ATTEMPT_ID,
            "sequence": sequence,
            "sequence_number": sequence,
            "event_type": kind,
            "kind": kind,
            "type": kind,
            "phase": phase,
            "occurred_at": occurred_at,
            "payload": flatten_metric_payload(payload),
            "producer": {
                "service": PRODUCER_SERVICE,
                "version": RUNNER_VERSION,
                "commit": "local",
            },
        }
