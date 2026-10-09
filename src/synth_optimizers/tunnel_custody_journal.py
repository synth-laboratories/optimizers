"""Fsynced local custody receipts, never authority or automatic effect replay.

See backend notes/specifications/tanha/current/systems/tunnels/v2_catalog.md.
Receipts may contain grant route tokens; files are private and are not logs.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from uuid import UUID, uuid4

from .tunnels import TunnelError

MAX_RECEIPT_BYTES = 16384
MAX_RECEIPTS = 4096


class TunnelCustodyInDoubt(TunnelError):
    """A retained receipt needs lookup; retrying origin/submission is not implied."""

    def __init__(self, path: Path, phase: str):
        self.receipt_path = path
        self.phase = phase
        super().__init__(f"tunnel custody {phase} is unresolved; receipt={path}")


class FileLeaseCustodyJournal:
    def __init__(self, directory: Path, lease: UUID):
        if lease.int == 0:
            raise TunnelError("nil custody lease identity")
        self.directory = directory.absolute()
        self.path = self.directory / f"{lease}.json"
        self.lease = lease
        self.revision = 0
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        lock = None
        acquired = False
        try:
            lock = os.open(
                ".custody.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory
            )
            for _ in range(20):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            if not acquired:
                raise TunnelError("custody directory lock deadline exceeded")
            yield directory
        finally:
            if lock is not None:
                if acquired:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                os.close(lock)
            os.close(directory)

    def _read(self, directory: int) -> dict | None:
        try:
            descriptor = os.open(
                self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_RECEIPT_BYTES
                or metadata.st_mode & 0o077
            ):
                raise TunnelError("custody receipt is oversized or not private")
            encoded = source.read(MAX_RECEIPT_BYTES + 1)
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise TunnelError("custody receipt exceeds byte capacity")
        payload = json.loads(encoded)
        if (
            not isinstance(payload, dict)
            or (type(payload.get("version")) is not int or payload["version"] != 1)
            or payload.get("lease") != str(self.lease)
        ):
            raise TunnelError("custody receipt identity/version mismatch")
        if (
            type(payload.get("revision")) is not int
            or not 1 <= payload["revision"] <= 9007199254740991
        ):
            raise TunnelError("custody receipt revision invalid")
        return payload

    def read(self) -> dict:
        with self._locked() as directory:
            payload = self._read(directory)
            if payload is None:
                raise TunnelError("custody receipt is missing")
            self.revision = payload["revision"]
            return payload

    def save(self, payload: dict) -> dict:
        """CAS one local receipt; no network I/O occurs under the directory lock."""
        with self._locked() as directory:
            previous = self._read(directory)
            revision = previous["revision"] if previous is not None else 0
            if revision != self.revision:
                raise TunnelError("custody receipt revision conflict")
            if previous is None:
                with os.scandir(directory) as entries:
                    if sum(1 for _ in islice(entries, MAX_RECEIPTS + 1)) >= MAX_RECEIPTS:
                        raise TunnelError("custody receipt directory capacity exceeded")
            if revision >= 9007199254740991:
                raise TunnelError("custody receipt revision exhausted")
            saved = {**payload, "version": 1, "revision": revision + 1, "lease": str(self.lease)}
            encoded = json.dumps(saved, sort_keys=True, separators=(",", ":")).encode()
            if len(encoded) > MAX_RECEIPT_BYTES:
                raise TunnelError("custody receipt exceeds byte capacity")
            temporary = f".{self.lease}.{uuid4()}.tmp"
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path.name, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            except OSError as error:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
                except OSError as cleanup:
                    raise TunnelError("custody commit and temporary cleanup failed") from cleanup
                raise TunnelCustodyInDoubt(self.path, "local_commit") from error
            self.revision = saved["revision"]
            return saved
