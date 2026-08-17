from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SLOT = "slot1"
DEFAULT_MAX_EVENTS = 5000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def project_gepa_run_started(*, run_id: str, config_path: str, output_dir: str | Path) -> None:
    _post_vl(
        {
            "_time": utc_now(),
            "_msg": f"gepa run.start {run_id}",
            "level": "info",
            "logger": "synth_optimizers.gepa",
            "slot": _vl_slot(),
            "service": "gepa",
            "event_domain": "local_optimizer",
            "event_type": "run.start",
            "phase": "started",
            "run_id": run_id,
            "gepa_config": str(config_path),
            "output_dir": str(output_dir),
        }
    )


def project_gepa_run_artifacts(manifest_path: str | Path) -> int:
    if os.environ.get("SYNTH_OPTIMIZERS_VL_PROJECT") == "0":
        return 0
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        _warn_or_raise(f"GEPA manifest not found for VL projection: {manifest_file}")
        return 0
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn_or_raise(f"GEPA manifest read failed for VL projection: {exc}")
        return 0
    event_feed = Path(str(manifest.get("event_feed_path") or manifest_file.parent / "events.jsonl"))
    if not event_feed.is_file():
        _warn_or_raise(f"GEPA event feed not found for VL projection: {event_feed}")
        return 0
    max_events = _positive_int(os.environ.get("SYNTH_OPTIMIZERS_VL_MAX_EVENTS"), DEFAULT_MAX_EVENTS)
    emitted = 0
    for line in event_feed.read_text(encoding="utf-8", errors="replace").splitlines():
        if emitted >= max_events:
            break
        raw = _json_object(line)
        if raw is None:
            continue
        _post_vl(_gepa_event_document(raw, manifest_file))
        emitted += 1
    run_id = _run_id_from_manifest(manifest, manifest_file)
    _post_vl(
        {
            "_time": utc_now(),
            "_msg": f"gepa run.terminal {run_id}",
            "level": "info",
            "logger": "synth_optimizers.gepa",
            "slot": _vl_slot(),
            "service": "gepa",
            "event_domain": "local_optimizer",
            "event_type": "run.terminal",
            "phase": "terminal",
            "run_id": run_id,
            "manifest_path": str(manifest_file),
            "event_count_projected": emitted,
            "best_candidate_id": _best_candidate_id(manifest),
            "cost_usd": _number_or_none(manifest.get("cost_usd")),
        }
    )
    return emitted + 1


def _gepa_event_document(raw: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    fields = raw.get("fields") if isinstance(raw.get("fields"), dict) else {}
    event_type = str(raw.get("type") or "gepa.event")
    run_id = str(fields.get("run_id") or _run_id_from_manifest_path(manifest_path))
    document: dict[str, Any] = {
        "_time": str(raw.get("ts") or fields.get("at") or utc_now()),
        "_msg": str(raw.get("message") or fields.get("message") or event_type),
        "level": _level_for(event_type, fields),
        "logger": "synth_optimizers.gepa",
        "slot": _vl_slot(),
        "service": "gepa",
        "event_domain": "local_optimizer",
        "event_type": event_type,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
    }
    for key in (
        "generation",
        "candidate_id",
        "parent_id",
        "phase",
        "state",
        "trigger",
        "from",
        "to",
        "status",
        "exit_code",
        "reward",
        "train_reward",
        "heldout_reward",
        "proposer_model",
        "policy_model",
    ):
        value = fields.get(key)
        if isinstance(value, str | int | float | bool):
            document[key] = value
    if event_type == "gepa.run.started":
        document["phase"] = "started"
    elif "generation" in event_type and "start" in event_type:
        document["phase"] = "generation.start"
    elif "generation" in event_type and ("end" in event_type or "complete" in event_type):
        document["phase"] = "generation.end"
    return document


def _post_vl(document: dict[str, Any]) -> None:
    if os.environ.get("SYNTH_OPTIMIZERS_VL_PROJECT") == "0":
        return
    if not str(document.get("event_domain") or "").strip():
        _warn_or_raise("VictoriaLogs event missing event_domain for synth-optimizers")
        return
    insert_url = _vl_insert_url()
    if not insert_url:
        _warn_or_raise("VictoriaLogs write URL not configured for synth-optimizers")
        return
    data = (json.dumps(document, separators=(",", ":")) + "\n").encode("utf-8")
    headers = {"content-type": "application/stream+json"}
    token = os.environ.get("VICTORIA_LOGS_WRITE_BEARER_TOKEN") or os.environ.get(
        "STACK_VICTORIA_LOGS_WRITE_BEARER_TOKEN"
    )
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(insert_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=2):
            return
    except urllib.error.URLError as exc:
        _warn_or_raise(f"VictoriaLogs insert failed for synth-optimizers: {exc}")


def _vl_insert_url() -> str | None:
    write_url = os.environ.get("VICTORIA_LOGS_WRITE_URL") or os.environ.get(
        "STACK_VICTORIA_LOGS_WRITE_URL"
    )
    if not write_url:
        port = _slot_victorialogs_port(_vl_slot())
        if port is None:
            return None
        write_url = f"http://127.0.0.1:{port}"
    base = write_url.rstrip("/")
    separator = "&" if "?" in base else "?"
    if "/insert/" in base:
        if "?_stream_fields=" in base or "&_stream_fields=" in base:
            return base
        return f"{base}{separator}_stream_fields=slot,service,event_domain"
    return f"{base}/insert/jsonline?_stream_fields=slot,service,event_domain"


def _slot_victorialogs_port(slot: str) -> int | None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidate = root.parent / "synth-dev" / "config" / "slots" / f"{_safe_slot(slot)}.toml"
        if candidate.is_file():
            match = re.search(
                r"^\s*victorialogs\s*=\s*(\d+)\s*$", candidate.read_text(), re.MULTILINE
            )
            if match:
                return int(match.group(1))
    return None


def _vl_slot() -> str:
    return _safe_slot(
        os.environ.get("STACK_VL_SLOT")
        or os.environ.get("SYNTH_OPTIMIZERS_VL_SLOT")
        or DEFAULT_SLOT
    )


def _safe_slot(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_.-]+", value.strip()) else DEFAULT_SLOT


def _warn_or_raise(message: str) -> None:
    if os.environ.get("SYNTH_OPTIMIZERS_REQUIRE_VL", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(message)
    print(f"[telemetry-warning] [synth-optimizers] warning: {message}; VL event skipped", file=sys.stderr)


def _json_object(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _level_for(event_type: str, fields: dict[str, Any]) -> str:
    text = f"{event_type} {fields.get('status') or ''}".lower()
    if "error" in text or "fail" in text:
        return "error"
    if "warn" in text:
        return "warning"
    return "info"


def _run_id_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> str:
    for key in ("run_id",):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return _run_id_from_manifest_path(manifest_path)


def _run_id_from_manifest_path(manifest_path: Path) -> str:
    return manifest_path.parent.name


def _best_candidate_id(manifest: dict[str, Any]) -> str | None:
    best = manifest.get("best_candidate")
    if isinstance(best, dict):
        value = best.get("candidate_id") or best.get("id")
        return str(value) if value else None
    return None


def _number_or_none(value: object) -> int | float | None:
    return value if isinstance(value, int | float) else None


def _positive_int(value: str | None, fallback: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback
