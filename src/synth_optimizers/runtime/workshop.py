"""Workshop projection-first surfaces for public SFT and CISPO.

The sqlite journal stays the record. These helpers publish the GEPA-shaped
page and collection names Workshop already consumes: `optimizer_event_page.v1`
plus `metric_points` / `candidates` / `evaluations` / `rollouts`. Visuals must
not rebuild charts from the raw journal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.training_schemas import TERMINAL_STATES
from ..read_models import Page, cispo_collections, reduce_summary, sft_collections
from .jobs import JobStore, flatten_metric_payload


EVENT_PAGE_SCHEMA = "optimizer_event_page.v1"


def optimizer_event_page(
    store: JobStore,
    job_id: str,
    *,
    after_sequence: int = 0,
    limit: int = 500,
) -> dict[str, Any]:
    job = store.require(job_id)
    events = store.events(job_id, after_sequence=after_sequence, limit=limit)
    next_sequence = int(events[-1]["sequence"]) if events else after_sequence
    return {
        "schema_version": EVENT_PAGE_SCHEMA,
        "run_id": job_id,
        "log_id": job_id,
        "after_sequence": after_sequence,
        "next_sequence": next_sequence,
        "terminal": job.state in TERMINAL_STATES,
        "events": events,
    }


def state_batch(
    store: JobStore,
    job_id: str,
    slices: str,
    *,
    algorithm_id: str,
) -> dict[str, Any]:
    summary = reduce_summary(store, job_id)
    payload: dict[str, Any] = {"run_id": job_id, "summary": summary}
    for raw in slices.split(","):
        name = raw.strip()
        if not name or name == "summary":
            continue
        page = workshop_collection(store, job_id, collection=name, algorithm_id=algorithm_id)
        payload[name] = {"items": [dict(item) for item in page.items]}
    return payload


def workshop_collection(
    store: JobStore,
    job_id: str,
    *,
    collection: str,
    algorithm_id: str,
    after_key: str | None = None,
    byte_limit: int = 65_536,
):
    if algorithm_id == "cispo":
        resolved = _CISPO_ALIASES.get(collection, collection)
        page = cispo_collections(
            store, job_id, collection=resolved, after_key=after_key, byte_limit=byte_limit,
            transform=(lambda item: _workshop_row(item, collection)) if collection in _CISPO_ALIASES else None
        )
        return page
    if collection == "proposer_calls" and algorithm_id != "cispo":
        return _empty_page(store, job_id)
    resolved = _SFT_ALIASES.get(collection, collection)
    page = sft_collections(
        store, job_id, collection=resolved, after_key=after_key, byte_limit=byte_limit,
        transform=(lambda item: _workshop_row(item, collection)) if collection in _SFT_ALIASES else None
    )
    return page


def _wrap_workshop_page(page: Page, collection: str) -> Page:
    return Page(
        items=tuple(_workshop_row(item, collection) for item in page.items),
        next_key=page.next_key,
        truncated=page.truncated,
        schema_version=page.schema_version,
        projected_at_sequence=page.projected_at_sequence,
        bytes=page.bytes,
    )


def _empty_page(store: JobStore, job_id: str) -> Page:
    events = store.events(job_id, after_sequence=0, limit=1)
    sequence = int(events[-1]["sequence"]) if events else 0
    return Page(
        items=(),
        next_key=None,
        truncated=False,
        schema_version="optimizer.summary.v1",
        projected_at_sequence=sequence,
        bytes=2,
    )


def _workshop_row(item: Mapping[str, Any], collection: str) -> dict[str, Any]:
    details = flatten_metric_payload(item)
    key = (
        details.get("checkpoint_id")
        or details.get("step")
        or details.get("update")
        or details.get("group_id")
        or details.get("event_id")
        or details.get("name")
        or details.get("sequence")
    )
    return {
        "item_id": str(key),
        "kind": collection,
        "details": details,
        **details,
    }


_SFT_ALIASES = {
    "metric_points": "training_metrics",
    "candidates": "checkpoints",
    "evaluations": "checkpoint_evaluations",
}

_CISPO_ALIASES = {
    "metric_points": "iterations",
    "candidates": "checkpoints",
    "evaluations": "checkpoint_evaluations",
    "rollouts": "rollout_groups",
}
