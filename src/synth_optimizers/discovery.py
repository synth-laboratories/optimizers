"""Local GEPA service/run discovery primitives.

The board treats ``GEPA_HOME`` as the local discovery contract:

- ``services/*.json``: live service heartbeats.
- ``index.jsonl``: append-only global run index, one line per run start.

This module is intentionally read-only for the board. Writers live in the Rust
runtime so standalone and service-managed runs share the same registry surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

DEFAULT_SERVICE_STALE_SECONDS = 30.0
GEPA_HOME_ENV = "GEPA_HOME"


@dataclass(frozen=True)
class ServiceHeartbeat:
    source_id: str
    path: Path
    service_url: str
    payload: dict
    last_seen: datetime | None


@dataclass(frozen=True)
class RunIndexEntry:
    run_id: str
    run_dir: Path
    event_feed_path: Path | None
    run_registry_path: Path | None
    pid: int | None
    started_at: datetime | None
    owning_service_url: str | None


def gepa_home(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get(GEPA_HOME_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".gepa"


def services_dir(home: str | Path | None = None) -> Path:
    return gepa_home(home) / "services"


def index_path(home: str | Path | None = None) -> Path:
    return gepa_home(home) / "index.jsonl"


def read_service_heartbeats(
    home: str | Path | None = None,
    *,
    stale_after_seconds: float = DEFAULT_SERVICE_STALE_SECONDS,
) -> list[ServiceHeartbeat]:
    now = _now()
    heartbeats: list[ServiceHeartbeat] = []
    root = services_dir(home)
    if not root.exists():
        return []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        pid = _int_or_none(payload.get("pid"))
        if pid is not None and not pid_alive(pid):
            continue
        last_seen = parse_ts(payload.get("last_seen") or payload.get("started_at"))
        if (
            last_seen is not None
            and (now - last_seen).total_seconds() > stale_after_seconds
        ):
            continue
        service_url = service_url_from_payload(payload)
        if not service_url:
            continue
        heartbeats.append(
            ServiceHeartbeat(
                source_id=str(payload.get("source_id") or path.stem),
                path=path,
                service_url=service_url,
                payload=payload,
                last_seen=last_seen,
            )
        )
    return heartbeats


def service_url_from_payload(payload: dict) -> str | None:
    explicit = payload.get("service_url")
    if isinstance(explicit, str) and explicit:
        return explicit.rstrip("/")
    bind = payload.get("bind")
    if not isinstance(bind, str) or not bind:
        return None
    host, sep, port = bind.rpartition(":")
    if not sep:
        return None
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}".rstrip("/")


def read_run_index(home: str | Path | None = None) -> list[RunIndexEntry]:
    path = index_path(home)
    if not path.exists():
        return []
    entries: list[RunIndexEntry] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or not raw.get("run_id") or not raw.get("run_dir"):
            continue
        entries.append(
            RunIndexEntry(
                run_id=str(raw["run_id"]),
                run_dir=Path(str(raw["run_dir"])).expanduser(),
                event_feed_path=_opt_path(raw.get("event_feed_path")),
                run_registry_path=_opt_path(raw.get("run_registry_path")),
                pid=_int_or_none(raw.get("pid")),
                started_at=parse_ts(raw.get("started_at")),
                owning_service_url=(
                    str(raw["owning_service_url"]).rstrip("/")
                    if raw.get("owning_service_url")
                    else None
                ),
            )
        )
    return entries


def latest_run_index(entries: Iterable[RunIndexEntry]) -> dict[str, RunIndexEntry]:
    latest: dict[str, RunIndexEntry] = {}
    for entry in entries:
        current = latest.get(entry.run_id)
        if current is None or _ts_key(entry.started_at) >= _ts_key(current.started_at):
            latest[entry.run_id] = entry
    return latest


def registry_roots_from_index(entries: Iterable[RunIndexEntry]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        if entry.run_registry_path:
            root = entry.run_registry_path.parent
        else:
            root = entry.run_dir.parent
        resolved = root.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(root)
    return roots


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_key(ts: datetime | None) -> float:
    return ts.timestamp() if ts else 0.0


def _opt_path(value: object) -> Path | None:
    return Path(str(value)).expanduser() if value else None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
