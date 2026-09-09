"""Bounded Workshop read models for SFT and CISPO.

Workshop must not reconstruct optimizer state from the raw journal. These
reducers emit a summary plus keyset-paginated collections with byte bounds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
) -> Page:
    selected: list[Mapping[str, Any]] = []
    size = 2
    truncated = False
    next_key = None
    started = after_key is None
    for item in items:
        key = str(item[key_field])
        if not started:
            if key == after_key:
                started = True
            continue
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"))
        if selected and size + len(encoded) > byte_limit:
            truncated = True
            next_key = str(selected[-1][key_field])
            break
        selected.append(item)
        size += len(encoded) + 1
    return Page(
        items=tuple(selected),
        next_key=next_key,
        truncated=truncated,
        schema_version=schema_version,
        projected_at_sequence=projected_at_sequence,
        bytes=size,
    )


def reduce_summary(store: JobStore, job_id: str, *, at_sequence: int | None = None) -> dict[str, Any]:
    job = store.require(job_id)
    events = _events_through(store, job_id, at_sequence)
    latest_metric = _latest(events, {"sft.step.metrics", "cispo.update.completed"})
    best = _best_checkpoint(events)
    receipts = store.receipts(job_id)
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
) -> Page:
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
    if collection == "artifacts":
        items = [{"name": row["name"], "digest": row["digest"]} for row in store.artifacts(job_id)]
        return _page(
            items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="name", byte_limit=byte_limit,
        )
    if collection == "receipts":
        items = [{"request_id": row["request_id"], **row} for row in store.receipts(job_id)]
        return _page(
            items, schema_version=SFT_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="request_id", byte_limit=byte_limit,
        )
    if collection not in mapping:
        raise KeyError(collection)
    kind, key_field, schema = mapping[collection]
    items = [_collection_item(event, key_field) for event in events if event["kind"] == kind]
    return _page(
        items, schema_version=schema, projected_at_sequence=sequence,
        after_key=after_key, key_field=key_field, byte_limit=byte_limit,
    )


def cispo_collections(
    store: JobStore,
    job_id: str,
    *,
    collection: str,
    after_key: str | None = None,
    byte_limit: int = DEFAULT_BYTE_LIMIT,
    at_sequence: int | None = None,
) -> Page:
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
        items = [{"name": row["name"], "digest": row["digest"]} for row in store.artifacts(job_id)]
        return _page(
            items, schema_version=CISPO_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="name", byte_limit=byte_limit,
        )
    if collection == "receipts":
        items = [{"request_id": row["request_id"], **row} for row in store.receipts(job_id)]
        return _page(
            items, schema_version=CISPO_READ_SCHEMA, projected_at_sequence=sequence,
            after_key=after_key, key_field="request_id", byte_limit=byte_limit,
        )
    if collection not in mapping:
        raise KeyError(collection)
    kind, key_field, schema = mapping[collection]
    items = [_collection_item(event, key_field) for event in events if event["kind"] == kind]
    return _page(
        items, schema_version=schema, projected_at_sequence=sequence,
        after_key=after_key, key_field=key_field, byte_limit=byte_limit,
    )


def replay_equals_read_model(store: JobStore, job_id: str) -> bool:
    live = reduce_summary(store, job_id)
    store.save_reducer(job_id, int(live["projected_at_sequence"]), dict(live))
    replayed = reduce_summary(store, job_id, at_sequence=int(live["projected_at_sequence"]))
    return digest_payload(live) == digest_payload(replayed)


def _events_through(store: JobStore, job_id: str, at_sequence: int | None) -> list[dict[str, Any]]:
    events = store.events(job_id, after_sequence=0, limit=5_000)
    if at_sequence is None:
        return events
    return [event for event in events if int(event["sequence"]) <= at_sequence]


def _latest(events: Sequence[Mapping[str, Any]], kinds: set[str]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if event["kind"] in kinds:
            return event
    return None


def _best_checkpoint(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    promoted = _latest(events, {"sft.checkpoint.promoted", "cispo.checkpoint.promoted"})
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
