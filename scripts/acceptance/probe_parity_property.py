#!/usr/bin/env python3
"""Gate 5 diagnosis: what parity property do the normalized event feeds have?

`synth-optimizers events compare` is whole-file byte equality over
`events.normalized.jsonl` (rust/crates/synth_optimizer_platform/src/events.rs
`compare_normalized_event_feeds`). This probe measures, over the feeds the
acceptance harness just produced, which exclusions are needed before the fresh,
cached, and readonly feeds agree -- and reports the residual for each step, so
the claim RELEASE.md should make can be stated exactly.

Run `run_offline_gates.py` and `probe_like_for_like.py` first; this reads their
output directories under accept-run/optimizers/gepa/runs.

    .venv/bin/python scripts/acceptance/probe_parity_property.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "accept-run" / "optimizers" / "gepa" / "runs"

# Event types that carry only runtime/execution telemetry. Their presence, count,
# or field values track how the work was scheduled, not what the optimizer decided.
RUNTIME_EVENT_TYPES = {
    "optimizer.rollout_queue.updated",  # worker-pool admission counters
    "optimizer.limit.estimate_updated",  # budget forecast, wall-clock stamped
    "runtime.job.completed",  # wall seconds, latency percentiles, cache hit/miss
}
# Field names that carry only runtime telemetry, at any depth.
RUNTIME_FIELD_KEYS = {
    "active_workers",
    "semaphore_size",
    "queued_rollouts",
    "generated_at",
    "sample_count",
    "runtime_summary",
}
# A fresh rollout id is minted per execution and is embedded inside nested
# resource refs (`child_resource_ref.id`, `.attributes.stream_id`,
# `.attributes.reward_url`), where the top-level `rollout_id` volatile-key strip
# in cache.rs `is_volatile_key` does not reach it.
ROLLOUT_ID = re.compile(r"gepa_[0-9a-f]{16,}")


def load(name: str) -> list[dict]:
    path = RUNS / name / "events.normalized.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in RUNTIME_FIELD_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return ROLLOUT_ID.sub("{ROLLOUT_ID}", value)
    return value


def project(events: list[dict]) -> list[dict]:
    """Decision/outcome projection: drop runtime telemetry, keep everything else."""
    out = []
    for event in events:
        if event.get("type") in RUNTIME_EVENT_TYPES:
            continue
        fields = event.get("fields")
        # `partial: true` marks the worker-pool progress copy of an evaluation
        # result, emitted by emit_runtime_rollout_progress only for rollouts that
        # actually executed. The canonical non-partial copy is emitted for every
        # evaluation regardless of cache, and is kept.
        if isinstance(fields, dict) and fields.get("partial") is True:
            continue
        projected = dict(event)
        projected.pop("sequence_number", None)  # shifts whenever any count shifts
        out.append(scrub(projected))
    return out


def digest(events: list[dict]) -> str:
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()


def main() -> int:
    available = [n for n in ("fresh", "cached", "readonly", "offA", "offB") if (RUNS / n).is_dir()]
    missing = {"fresh", "cached", "readonly"} - set(available)
    if missing:
        raise SystemExit(f"Missing required evidence feeds: {sorted(missing)}")
    feeds = {name: load(name) for name in available}
    print("raw normalized feed lengths (what `events compare` sees):")
    for name, events in feeds.items():
        print(f"    {name:9s} {len(events):4d} events")
    projected = {name: project(events) for name, events in feeds.items()}
    print("\ndecision/outcome projection:")
    for name, events in projected.items():
        print(f"    {name:9s} {len(events):4d} events  sha256={digest(events)}")
    print()
    failures = 0
    for left, right in (("fresh", "cached"), ("fresh", "readonly"), ("cached", "readonly")):
        if left not in projected or right not in projected:
            continue
        equal = projected[left] == projected[right]
        print(f"    {left} vs {right}: {'EQUAL' if equal else 'DIFFERS'}")
        failures += 0 if equal else 1
    if "offA" in projected and "offB" in projected:
        equal = projected["offA"] == projected["offB"]
        print(
            f"    offA vs offB (two cache-off runs): {'EQUAL' if equal else 'DIFFERS'}"
            "   [harness artifact: the offline proposer embeds a digest of its own"
            " request, which contains per-execution rollout ids]"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
