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

    selected = []
    page = make_page([], False)
    if page.bytes > byte_limit:
        raise ValueError("projection byte limit cannot fit page envelope")
    for index in range(start, len(items)):
        candidate = make_page([*selected, items[index]], index + 1 < len(items))
        if candidate.bytes > byte_limit or len(selected) == 100:
            if not selected:
                raise ValueError("projection row exceeds byte limit; use source artifact")
            page = make_page(selected, True)
            if page.bytes > byte_limit:
                raise ValueError("projection cursor exceeds byte limit")
            return page
        selected.append(items[index])
        page = candidate
    return page


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
    job = store.require(job_id)
    events = _events_through(store, job_id, at_sequence)
    latest_metric = _latest(events, {"sft.step.metrics", "cispo.update.completed"})
    best = _best_checkpoint(events)
    if at_sequence is not None:
        state_event = _latest(events, {"training.lifecycle"})
        state = state_event["payload"]["state"] if state_event else (events[-1]["phase"] if events else "prepared")
        job = replace(job, state=state, error=state_event["payload"].get("error") if state_event else None)
    receipts = (store.receipts(job_id) if at_sequence is None else
                _historical_rows(events, "training.receipt", "request_id"))
    cost_values = [row.get("cost_usd") for row in receipts]
    cost_missing = not receipts or any(row.get("cost_missing", row.get("cost_usd") is None) for row in receipts)
    usage = {
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in receipts),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in receipts),
        "training_tokens": sum(int(row.get("training_tokens") or 0) for row in receipts),
        "cost_usd": None if cost_missing else sum(float(value) for value in cost_values if value is not None),
        "cost_missing": cost_missing,
    }
    config = json.loads(job.config_json)
    return {
        "schema_version": SHARED_SUMMARY_SCHEMA,
        "job_id": job.job_id,
        "state": job.state,
        "progress": _progress(job, events),
        "algorithm_id": job.algorithm_id,
        "implementation_version": job.implementation_version,
        "model_id": job.model_id,
        "checkpoint": best,
        "dataset": config.get("dataset_manifest") or config.get("dataset"),
        "current_metric": None if latest_metric is None else latest_metric.get("payload"),
        "best_metric": None if best is None else best.get("calibration_accuracy"),
        "usage": usage,
        "failure_reason": job.error,
        "stale": job.state not in {"completed", "failed", "cancelled"} and job.heartbeat_at is None,
        "projected_at_sequence": events[-1]["sequence"] if events else 0,
    }


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
    after_key, at_sequence = _decode_cursor(store, job_id, collection, after_key, at_sequence)
    events = _events_through(store, job_id, at_sequence)
    sequence = events[-1]["sequence"] if events else 0
    mapping = {
        "training_metrics": ("sft.step.metrics", "step", SFT_READ_SCHEMA),
        "metric_points": ("sft.step.metrics", "step", SFT_READ_SCHEMA),
        "checkpoints": ("sft.checkpoint.created", "checkpoint_id", SFT_READ_SCHEMA),
        "candidates": ("sft.checkpoint.created", "checkpoint_id", SFT_READ_SCHEMA),
        "checkpoint_evaluations": ("sft.checkpoint_eval.completed", "checkpoint_id", SFT_READ_SCHEMA),
        "evaluations": ("sft.checkpoint_eval.completed", "checkpoint_id", SFT_READ_SCHEMA),
        "per_intent": ("sft.heldout_eval.completed", "event_id", SFT_READ_SCHEMA),
        "dataset_errors": ("sft.dataset.validated", "event_id", SFT_READ_SCHEMA),
    }
    if collection in {"evaluations", "checkpoint_evaluations"}:
        items = []
        for event in events:
            if event["kind"] == "sft.child_eval.completed":
                result = event["payload"]
                item = {key: value for key, value in result.items() if key not in {"rollouts", "evidence_refs"}}
                item.update(item_id=result["eval_job_id"], evaluation_id=result["eval_job_id"],
                            checkpointId=result["checkpoint_id"], phase=result["role"],
                            score=result.get("value"), evaluator=result["evaluator_id"],
                            metric=result.get("metric_ref"))
            elif event["kind"] == "sft.checkpoint_eval.completed":
                item = _collection_item(event, "checkpoint_id")
                item["item_id"] = event["event_id"]
            else:
                continue
            if transform is not None:
                item = {**transform(item), "item_id": item["item_id"]}
            items.append(item)
        return _page(items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
                     after_key=after_key, key_field="item_id", byte_limit=byte_limit,
                     cursor_context=(job_id, collection))
    if collection in {"child_evaluations", "rollouts", "evidence_refs"}:
        items = []
        for event in events:
            if event["kind"] != "sft.child_eval.completed":
                continue
            result = event["payload"]
            provenance = {key: result[key] for key in ("eval_job_id", "checkpoint_id", "evaluator_id", "role")}
            if collection == "child_evaluations":
                items.append({**{key: value for key, value in result.items() if key not in {"rollouts", "evidence_refs"}},
                              "item_id": result["eval_job_id"]})
            else:
                for index, row in enumerate(result.get(collection, [])):
                    items.append({**provenance, "reference": row,
                                  "item_id": f"{result['eval_job_id']}:{collection}:{index}"})
        return _page(items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
                     after_key=after_key, key_field="item_id", byte_limit=byte_limit,
                     cursor_context=(job_id, collection))
    if collection == "artifacts":
        items = [{"name": row["name"], "digest": row["digest"]} for row in (store.artifacts(job_id) if at_sequence is None else _historical_rows(events, "training.artifact", "name"))]
        return _page(
            items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="name", byte_limit=byte_limit, cursor_context=(job_id, collection),
        )
    if collection == "receipts":
        items = [{"request_id": row["request_id"], **row} for row in (store.receipts(job_id) if at_sequence is None else _historical_rows(events, "training.receipt", "request_id"))]
        return _page(
            items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="request_id", byte_limit=byte_limit, cursor_context=(job_id, collection),
        )
    if collection not in mapping:
        raise KeyError(collection)
    kind, key_field, schema = mapping[collection]
    items = [_collection_item(event, key_field) for event in events if event["kind"] == kind]
    if transform is not None:
        items = [transform(item) for item in items]
    return _page(
        items, schema_version=schema, projected_at_sequence=sequence,
        after_key=after_key, key_field=key_field, byte_limit=byte_limit, cursor_context=(job_id, collection),
    )


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
    after_key, at_sequence = _decode_cursor(store, job_id, collection, after_key, at_sequence)
    events = _events_through(store, job_id, at_sequence)
    sequence = events[-1]["sequence"] if events else 0
    mapping = {
        "iterations": ("cispo.update.completed", "update", CISPO_READ_SCHEMA),
        "metric_points": ("cispo.update.completed", "update", CISPO_READ_SCHEMA),
        "rollout_groups": ("cispo.rollout_group.completed", "group_id", CISPO_READ_SCHEMA),
        "rollouts": ("cispo.rollout_group.completed", "group_id", CISPO_READ_SCHEMA),
        "reward_distributions": ("cispo.rollout_group.completed", "group_id", CISPO_READ_SCHEMA),
        "advantage_distributions": ("cispo.group_advantage.computed", "group_id", CISPO_READ_SCHEMA),
        "importance_ratios": ("cispo.importance_ratio.measured", "update", CISPO_READ_SCHEMA),
        "zero_advantage_groups": ("cispo.zero_advantage.detected", "group_id", CISPO_READ_SCHEMA),
        "checkpoints": ("cispo.checkpoint.created", "checkpoint_id", CISPO_READ_SCHEMA),
        "candidates": ("cispo.checkpoint.created", "checkpoint_id", CISPO_READ_SCHEMA),
        "checkpoint_evaluations": ("cispo.checkpoint_eval.completed", "checkpoint_id", CISPO_READ_SCHEMA),
        "evaluations": ("cispo.checkpoint_eval.completed", "checkpoint_id", CISPO_READ_SCHEMA),
        "per_intent": ("cispo.heldout_eval.completed", "event_id", CISPO_READ_SCHEMA),
    }
    if collection == "artifacts":
        items = [{"name": row["name"], "digest": row["digest"]} for row in (store.artifacts(job_id) if at_sequence is None else _historical_rows(events, "training.artifact", "name"))]
        return _page(
            items, schema_version=CISPO_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="name", byte_limit=byte_limit, cursor_context=(job_id, collection),
        )
    if collection == "receipts":
        items = [{"request_id": row["request_id"], **row} for row in (store.receipts(job_id) if at_sequence is None else _historical_rows(events, "training.receipt", "request_id"))]
        return _page(
            items, schema_version=CISPO_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="request_id", byte_limit=byte_limit, cursor_context=(job_id, collection),
        )
    if collection not in mapping:
        raise KeyError(collection)
    kind, key_field, schema = mapping[collection]
    items = [_collection_item(event, key_field) for event in events if event["kind"] == kind]
    if transform is not None:
        items = [transform(item) for item in items]
    return _page(
        items, schema_version=schema, projected_at_sequence=sequence,
        after_key=after_key, key_field=key_field, byte_limit=byte_limit, cursor_context=(job_id, collection),
    )


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
