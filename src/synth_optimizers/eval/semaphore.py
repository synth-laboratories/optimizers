"""The global trial semaphore.

A semaphore is an internal concurrency primitive, not an algorithm and not a
product surface. There is exactly one lease store per `eval` home, shared by
every local run in every worker process, so the concurrency ceiling is a
property of the machine rather than of whichever run happened to start first.

Leases are files guarded by an exclusive lock. A lease whose owning process
died, or whose heartbeat lapsed past the configured TTL, is reclaimed by the
next acquirer: a crashed worker cannot strand capacity.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path


class SemaphoreTimeout(RuntimeError):
    """No token became free inside the caller's budget."""


@dataclass(frozen=True, slots=True)
class Lease:
    id: str
    path: Path
    run_id: str
    trial_id: str
    acquired_at: float


class TrialSemaphore:
    def __init__(self, directory: Path, *, capacity: int, ttl_seconds: int) -> None:
        if capacity < 1:
            raise ValueError("semaphore capacity must be at least 1")
        self.directory = directory
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.directory / ".lock"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _live_leases(self) -> list[dict[str, object]]:
        """Read the store, deleting leases whose owner is gone or stale."""

        now = time.time()
        live: list[dict[str, object]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                continue
            expires_at = float(record.get("expires_at", 0.0))
            pid = int(record.get("pid", 0))
            if expires_at < now or not _process_alive(pid):
                path.unlink(missing_ok=True)
                continue
            live.append(record)
        return live

    def snapshot(self) -> dict[str, object]:
        with self._locked():
            live = self._live_leases()
        return {
            "capacity": self.capacity,
            "leased": len(live),
            "available": max(0, self.capacity - len(live)),
            "leases": [
                {"run_id": item.get("run_id"), "trial_id": item.get("trial_id")} for item in live
            ],
        }

    def acquire(
        self,
        *,
        run_id: str,
        trial_id: str,
        timeout_seconds: float | None = None,
        should_abort: Callable[[], bool] | None = None,
        poll_seconds: float = 0.2,
    ) -> Lease:
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds
        while True:
            if should_abort is not None and should_abort():
                raise SemaphoreTimeout("cancelled while waiting for a semaphore token")
            with self._locked():
                if len(self._live_leases()) < self.capacity:
                    lease_id = f"lease_{uuid.uuid4().hex[:12]}"
                    path = self.directory / f"{lease_id}.json"
                    now = time.time()
                    path.write_text(
                        json.dumps(
                            {
                                "lease_id": lease_id,
                                "run_id": run_id,
                                "trial_id": trial_id,
                                "pid": os.getpid(),
                                "acquired_at": now,
                                "expires_at": now + self.ttl_seconds,
                            }
                        ),
                        encoding="utf-8",
                    )
                    return Lease(
                        id=lease_id,
                        path=path,
                        run_id=run_id,
                        trial_id=trial_id,
                        acquired_at=now,
                    )
            if deadline is not None and time.time() >= deadline:
                raise SemaphoreTimeout(
                    f"no eval semaphore token available within {timeout_seconds}s "
                    f"(capacity {self.capacity})"
                )
            time.sleep(poll_seconds)

    def heartbeat(self, lease: Lease) -> None:
        """Keep a long trial's token alive without widening the TTL for others."""

        with self._locked():
            if not lease.path.is_file():
                return
            try:
                record = json.loads(lease.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            record["expires_at"] = time.time() + self.ttl_seconds
            lease.path.write_text(json.dumps(record), encoding="utf-8")

    def release(self, lease: Lease) -> None:
        with self._locked():
            lease.path.unlink(missing_ok=True)

    def release_run(self, run_id: str) -> int:
        """Drop every lease a run still holds. Used on resume and on cancel."""

        removed = 0
        with self._locked():
            for path in sorted(self.directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    path.unlink(missing_ok=True)
                    continue
                if record.get("run_id") == run_id:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
