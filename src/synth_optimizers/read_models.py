"""Bounded Workshop read models for SFT and CISPO.

Workshop must not reconstruct optimizer state from the raw journal. These
reducers emit a summary plus keyset-paginated collections with byte bounds.
"""

from __future__ import annotations

import json
import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict, replace
from typing import Any

from .runtime import JobStore, digest_payload


SHARED_SUMMARY_SCHEMA = "optimizer.summary.v1"
SFT_READ_SCHEMA = "optimizer.sft.read.v1"
CISPO_READ_SCHEMA = "optimizer.cispo.read.v1"
DEFAULT_BYTE_LIMIT = 65_536


@dataclass(frozen=True, slots=True)
class Page:
    items: tuple[Mapping[str, Any], ...]
    next_key: str | None
    truncated: bool
    schema_version: str
    projected_at_sequence: int
    bytes: int


def _page(
    items: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    projected_at_sequence: int,
    after_key: str | None,
    key_field: str,
    byte_limit: int,
    cursor_context: tuple[str, str] | None = None,
) -> Page:
    keys = [str(item[key_field]) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate projection ordering key")
    if after_key is not None and after_key not in keys:
        raise ValueError("unknown or stale projection cursor")
    start = 0 if after_key is None else keys.index(after_key) + 1

    def make_page(rows, more):
        next_key = str(rows[-1][key_field]) if more and rows else None
        if next_key is not None and cursor_context is not None:
            next_key = _encode_cursor(*cursor_context, projected_at_sequence, next_key)
        page = Page(tuple(rows), next_key,
                    more, schema_version, projected_at_sequence, 0)
        while True:
            size = len(json.dumps(asdict(page), ensure_ascii=True).encode("utf-8"))
            if size == page.bytes:
                return page
            page = replace(page, bytes=size)

    empty = make_page([], False)
    if empty.bytes > byte_limit:
        raise ValueError("projection byte limit cannot fit page envelope")
    remaining = items[start:]
    if not remaining:
        return empty
    maximum = min(100, len(remaining))
    candidate = make_page(remaining[:maximum], maximum < len(remaining))
    if candidate.bytes <= byte_limit:
        return candidate
    # All prefixes below maximum carry a continuation cursor. Binary search
    # their exact wire envelopes rather than serializing every growing prefix.
    low, high = 1, maximum - 1
    best = None
    while low <= high:
        count = (low + high) // 2
        candidate = make_page(remaining[:count], True)
        if candidate.bytes <= byte_limit:
            best = candidate
            low = count + 1
        else:
            high = count - 1
    if best is None:
        raise ValueError("projection row exceeds byte limit; use source artifact")
    return best


def _encode_cursor(job_id, collection, sequence, key):
    payload = json.dumps([1, job_id, collection, sequence, key], separators=(",", ":")).encode()
    return "pc1." + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(store, job_id, collection, cursor, at_sequence):
    with store._lock:
        latest = store._latest_sequence(job_id)
    if cursor is None:
        return None, latest if at_sequence is None else at_sequence
    try:
        if len(cursor) > 8192 or not cursor.startswith("pc1."):
            raise ValueError()
        payload = cursor[4:]
        version, run, scope, sequence, key = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if version != 1 or run != job_id or scope != collection or type(sequence) is not int or not 0 <= sequence <= latest or not isinstance(key, str):
            raise ValueError()
        if at_sequence is not None and at_sequence != sequence:
            raise ValueError()
        return key, sequence
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("unknown or stale projection cursor") from exc


def reduce_summary(store: JobStore, job_id: str, *, at_sequence: int | None = None) -> dict[str, Any]:
    from .runtime.projections import summary_at
    job = store.require(job_id)
    with store._lock:
        latest = store._latest_sequence(job_id)
    bound = latest if at_sequence is None else at_sequence
    if not 0 <= bound <= latest:
        raise ValueError("unknown projection sequence")
    snapshot = summary_at(store, job_id, bound)
    state = snapshot["state"]
    best = snapshot["checkpoint"]
    missing = snapshot["receipt_count"] == 0 or snapshot["missing_cost_count"] > 0
    usage = {**snapshot["usage"], "cost_usd": None if missing else snapshot["usage"]["cost_usd"], "cost_missing": missing}
    config = json.loads(job.config_json)
    return {
        "schema_version": SHARED_SUMMARY_SCHEMA, "job_id": job.job_id, "state": state,
        "progress": {"completed_units": snapshot["steps"], "state": state},
        "algorithm_id": job.algorithm_id, "implementation_version": job.implementation_version,
        "model_id": job.model_id, "checkpoint": best,
        "dataset": config.get("dataset_manifest") or config.get("dataset"),
        "current_metric": snapshot["metric"],
        "best_metric": None if best is None else best.get("calibration_accuracy"),
        "usage": usage, "failure_reason": snapshot["error"],
        "stale": state not in {"completed", "failed", "cancelled"} and job.heartbeat_at is None,
        "projected_at_sequence": bound,
    }


def _indexed_page(store, job_id, collection, schema, after_key, byte_limit, at_sequence, transform):
    from .runtime.projections import collection_rows
    key, bound = _decode_cursor(store, job_id, collection, after_key, at_sequence)
    if not 0 <= bound <= store._latest_sequence(job_id):
        raise ValueError("unknown projection sequence")
    rows = collection_rows(store, job_id, collection, bound, key)
    items = [{**(transform(row) if transform is not None else row), "item_id": item_key} for row, item_key in rows]
    return _page(items, schema_version=schema, projected_at_sequence=bound, after_key=None,
                 key_field="item_id", byte_limit=byte_limit, cursor_context=(job_id, collection))


def sft_collections(
    store: JobStore,
    job_id: str,
    *,
    collection: str,
    after_key: str | None = None,
    byte_limit: int = DEFAULT_BYTE_LIMIT,
    at_sequence: int | None = None,
    transform=None,
) -> Page:
    if collection not in {"training_metrics", "metric_points", "checkpoints", "candidates", "checkpoint_evaluations",
                           "evaluations", "per_intent", "dataset_errors", "child_evaluations", "rollouts", "evidence_refs", "artifacts", "receipts"}:
        raise KeyError(collection)
    return _indexed_page(store, job_id, collection, SFT_READ_SCHEMA, after_key, byte_limit, at_sequence, transform)


def cispo_collections(
    store: JobStore,
    job_id: str,
    *,
    collection: str,
    after_key: str | None = None,
    byte_limit: int = DEFAULT_BYTE_LIMIT,
    at_sequence: int | None = None,
    transform=None,
) -> Page:
    if collection not in {"iterations", "metric_points", "rollout_groups", "rollouts", "reward_distributions", "advantage_distributions",
                           "importance_ratios", "zero_advantage_groups", "checkpoints", "candidates", "checkpoint_evaluations", "evaluations",
                           "per_intent", "artifacts", "receipts"}:
        raise KeyError(collection)
    return _indexed_page(store, job_id, collection, CISPO_READ_SCHEMA, after_key, byte_limit, at_sequence, transform)


def replay_equals_read_model(store: JobStore, job_id: str) -> bool:
    live = reduce_summary(store, job_id)
    store.save_reducer(job_id, int(live["projected_at_sequence"]), dict(live))
    replayed = reduce_summary(store, job_id, at_sequence=int(live["projected_at_sequence"]))
    return digest_payload(live) == digest_payload(replayed)


def _events_through(store: JobStore, job_id: str, at_sequence: int | None) -> list[dict[str, Any]]:
    # Freeze the upper bound before paging so concurrent appends cannot extend a read.
    with store._lock:
        latest = store._latest_sequence(job_id)
    bound = latest if at_sequence is None else at_sequence
    if bound < 0 or bound > latest:
        raise ValueError("unknown projection sequence")
    events = []
    cursor = 0
    while cursor < bound:
        batch = store.events(job_id, after_sequence=cursor, limit=min(5000, bound - cursor))
        if not batch:
            raise ValueError("projection journal contains a gap")
        events.extend(event for event in batch if event["sequence"] <= bound)
        cursor = batch[-1]["sequence"]
    return events


def _historical_rows(events, kind, key):
    rows = {}
    for event in events:
        if event["kind"] == kind:
            row = event["payload"]
            rows[row[key]] = row
    return [rows[key] for key in sorted(rows)]


def _latest(events: Sequence[Mapping[str, Any]], kinds: set[str]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if event["kind"] in kinds:
            return event
    return None


def _best_checkpoint(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    promoted = _latest(events, {"sft.checkpoint.promoted", "sft.checkpoint.selected", "cispo.checkpoint.promoted"})
    return None if promoted is None else promoted.get("payload")


def _progress(job: Any, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = sum(1 for event in events if event["kind"] in {"sft.step.metrics", "cispo.update.completed"})
    return {"completed_units": steps, "state": job.state}


def _collection_item(event: Mapping[str, Any], key_field: str) -> dict[str, Any]:
    from .runtime.jobs import flatten_metric_payload

    payload = flatten_metric_payload(event.get("payload") or {})
    if key_field == "event_id":
        payload["event_id"] = event["event_id"]
    if key_field == "update" and "update" not in payload:
        payload["update"] = payload.get("iteration") or event["sequence"]
    payload.setdefault(key_field, payload.get(key_field) or event["event_id"])
    payload["sequence"] = event["sequence"]
    payload.setdefault("details", dict(payload))
    return payload
