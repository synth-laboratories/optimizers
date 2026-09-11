"""The global trial semaphore.

A semaphore is an internal concurrency primitive, not an algorithm and not a
product surface. There is exactly one lease store per `eval` home, shared by
every local run in every worker process, so the concurrency ceiling is a
property of the machine rather than of whichever run happened to start first.

Leases are files guarded by an exclusive lock. An expired lease with a bound
provider intent is quarantined: worker death is not proof of container death.
Unbound leases can be reclaimed because they have not authorized a launch.
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

from .models import write_json


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
        """Retain uncertain provider capacity when an owner disappears."""

        now = time.time()
        live: list[dict[str, object]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "unreadable eval lease; reconcile before admitting work"
                ) from error
            if record.get("quarantined") is True:
                live.append(record)
                continue
            expires_at = float(record.get("expires_at", 0.0))
            pid = int(record.get("pid", 0))
            if expires_at < now or not _process_alive(pid):
                if record.get("resource"):
                    record.update({"quarantined": True, "quarantine_reason": "owner_lost"})
                    write_json(path, record)
                    live.append(record)
                else:
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
                {
                    "run_id": item.get("run_id"),
                    "trial_id": item.get("trial_id"),
                    "quarantined": item.get("quarantined", False),
                    "resource": item.get("resource"),
                }
                for item in live
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
                    write_json(
                        path,
                        {
                            "lease_id": lease_id,
                            "run_id": run_id,
                            "trial_id": trial_id,
                            "pid": os.getpid(),
                            "acquired_at": now,
                            "expires_at": now + self.ttl_seconds,
                        },
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
            record = json.loads(lease.path.read_text(encoding="utf-8"))
            if record.get("quarantined") is True:
                raise RuntimeError("eval lease is quarantined; stop execution")
            record["expires_at"] = time.time() + self.ttl_seconds
            write_json(lease.path, record)

    def release(self, lease: Lease) -> None:
        with self._locked():
            if lease.path.is_file():
                record = json.loads(lease.path.read_text(encoding="utf-8"))
                if record.get("quarantined") is True:
                    return
            lease.path.unlink(missing_ok=True)

    def bind_resource(self, lease: Lease, resource: dict[str, object]) -> None:
        """Persist launch identity before execution can create a remote resource."""
        if not resource.get("runtime") or not resource.get("container_id"):
            raise ValueError("eval launch intent requires runtime and container identity")
        with self._locked():
            record = json.loads(lease.path.read_text(encoding="utf-8"))
            if record.get("quarantined") is True or record.get("resource"):
                raise RuntimeError("eval lease cannot bind a second resource")
            record["resource"] = dict(resource)
            write_json(lease.path, record)

    def quarantine(self, lease: Lease, resource: dict[str, object]) -> None:
        """Retain capacity after an unconfirmed stop, including across restart."""
        with self._locked():
            record = json.loads(lease.path.read_text(encoding="utf-8"))
            record.update({"quarantined": True, "resource": resource})
            write_json(lease.path, record)

    def release_run(self, run_id: str) -> int:
        """Drop every lease a run still holds. Used on resume and on cancel."""

        removed = 0
        with self._locked():
            for path in sorted(self.directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError("unreadable eval lease; reconcile before release") from error
                if record.get("run_id") != run_id or record.get("quarantined") is True:
                    continue
                if record.get("resource"):
                    record.update(
                        {"quarantined": True, "quarantine_reason": "run_release_without_stop"}
                    )
                    write_json(path, record)
                else:
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
