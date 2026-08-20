#!/usr/bin/env python3
"""Mint archive-native generation_start cursor fixtures, and backfill real usage.

Two jobs, both offline over existing GEPA workspace archives. Nothing here
starts an optimizer run.

1. ``mint`` — export a ``generation_start`` cursor straight out of a real
   ``workspace.sqlite`` as a ``gepa_cursor_fixture.v1`` file. This is
   ``export_checkpoints.py`` generalised: that script only knows the three
   Banking77 generations of ``b77_gepa_eval_mfg_20260819`` and only accepts
   checkpoints whose ``metadata.retain`` is set, which is a flag the outer
   proposer-eval fork path writes and a plain cookbook run does not. The real
   requirement for a fixture is only that the checkpoint is a
   ``generation_start`` and is **not** a compacted ``checkpoint_summary.v1``
   (those are forbidden as fixtures), so ``retain`` is optional here.

2. ``backfill`` — write real ``usage`` totals into fixtures that were
   reconstructed with ``usage: {}``. The totals are the source archive's own
   checkpoint usage at the point the fixture was reconstructed from, so they
   are frozen archive numbers, not estimates. Fixtures with no archive behind
   them (seed-only TAU2 / MiniGrid / OfficeQA) get explicit zeros instead of
   ``{}``: no rollouts happened in those cursors, so zero is the true total.

Usage:
    python mint_checkpoint_fixtures.py            # mint + backfill
    python mint_checkpoint_fixtures.py --mint-only
    python mint_checkpoint_fixtures.py --backfill-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"

COOKBOOK_RUNS = Path(
    "/Users/joshuapurtell/Documents/GitHub/synth-cookbooks-public/cookbooks/optimizers/gepa/runs"
)
OPTIMIZER_RUNS = Path("/Users/joshuapurtell/Documents/GitHub/optimizers/runs")
PROPOSER_RUNS = HERE / "generated"

USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "rollout_calls",
    "proposer_calls",
)


# --------------------------------------------------------------------------- mint


@dataclass(frozen=True)
class Mint:
    filename: str
    task_id: str
    label: str
    maturity: str
    workspace: Path
    generation: int
    description: str


MINTS: tuple[Mint, ...] = (
    # Banking77, lineage A: the same b77_gepa_eval_mfg_20260819 archive that
    # train:0/1/2 came from, carried two generations further by a proposer-eval
    # fork that completed. 19 candidates is the deepest Banking77 cursor here.
    Mint(
        filename="banking77_gen3.json",
        task_id="train:3",
        label="banking77-gen3",
        maturity="deep",
        workspace=OPTIMIZER_RUNS / "gepa_24b32fd4c2e74e96aed7ba747dcd5c55" / "workspace.sqlite",
        generation=3,
        description=(
            "Generation 3 generation_start cursor, 19 candidates, from a completed "
            "proposer-eval fork of the b77_gepa_eval_mfg_20260819 lineage. Deeper "
            "than train:2 (13 candidates): exploration starts from a fuller frontier."
        ),
    ),
    # Banking77, lineage B: an independent async cookbook run with a different
    # seed prompt and a much weaker seed (0.54 full-train vs 0.68), so the
    # proposer has real headroom instead of a near-ceiling parent.
    Mint(
        filename="banking77_async_fresh.json",
        task_id="train:4",
        label="banking77-async-fresh",
        maturity="fresh",
        workspace=COOKBOOK_RUNS / "banking77_gepa_async_t50_mb20_h100_735a9c29" / "workspace.sqlite",
        generation=0,
        description=(
            "Generation 0 generation_start cursor from the independent async "
            "Banking77 lineage (seed full-train 0.54, vs 0.68 for train:0). Second "
            "seed prompt for the same inner container."
        ),
    ),
    Mint(
        filename="banking77_async_first.json",
        task_id="train:5",
        label="banking77-async-first-checkpoint",
        maturity="first",
        workspace=COOKBOOK_RUNS / "banking77_gepa_async_t50_mb20_h100_735a9c29" / "workspace.sqlite",
        generation=1,
        description=(
            "Generation 1 generation_start cursor (7 candidates) from the async "
            "Banking77 lineage. Same shape as train:1 on an independent run."
        ),
    ),
    # Crafter: archive-native counterparts to the compacted crafter:0/1/2
    # reconstructions. Same run, but the cursor is the archive's own snapshot
    # (state_history, sensor frames, program, rollout_task_id all intact).
    Mint(
        filename="crafter_archive_fresh.json",
        task_id="crafter:3",
        label="crafter-archive-fresh",
        maturity="fresh",
        workspace=COOKBOOK_RUNS / "crafter_gepa_public_0fbad055" / "workspace.sqlite",
        generation=0,
        description=(
            "Generation 0 generation_start cursor taken verbatim from the "
            "crafter_gepa_public_0fbad055 archive (crafter:0 is the compacted "
            "reconstruction of the same point)."
        ),
    ),
    Mint(
        filename="crafter_archive_mature.json",
        task_id="crafter:4",
        label="crafter-archive-mature",
        maturity="mature",
        workspace=COOKBOOK_RUNS / "crafter_gepa_public_0fbad055" / "workspace.sqlite",
        generation=2,
        description=(
            "Generation 2 generation_start cursor (4 candidates) taken verbatim "
            "from the crafter_gepa_public_0fbad055 archive."
        ),
    ),
    # TAU2: the inner GEPA smoke that actually improved (train 0.050 -> 0.200,
    # heldout 0 -> 0.125) left a full 7-generation archive. tau2:0 stays the
    # seed-only fixture; these two are real checkpointed cursors.
    Mint(
        filename="tau2_first.json",
        task_id="tau2:1",
        label="tau2-retail-first-checkpoint",
        maturity="first",
        workspace=PROPOSER_RUNS / "tau2_gepa_runs" / "tau2_retail_gepa_20260819_long" / "workspace.sqlite",
        generation=1,
        description=(
            "Generation 1 generation_start cursor (3 candidates) from the "
            "tau2_retail_gepa_20260819_long inner GEPA run. 20 train / 8 heldout."
        ),
    ),
    Mint(
        filename="tau2_mature.json",
        task_id="tau2:2",
        label="tau2-retail-mature",
        maturity="mature",
        workspace=PROPOSER_RUNS / "tau2_gepa_runs" / "tau2_retail_gepa_20260819_long" / "workspace.sqlite",
        generation=6,
        description=(
            "Generation 6 generation_start cursor (17 candidates) from the "
            "completed tau2_retail_gepa_20260819_long run. 20 train / 8 heldout."
        ),
    ),
    # MiniGrid: same story, the Empty-5x5 smoke went 0.353 -> 0.842 train.
    Mint(
        filename="minigrid_first.json",
        task_id="minigrid:1",
        label="minigrid-empty-first-checkpoint",
        maturity="first",
        workspace=PROPOSER_RUNS / "minigrid_gepa_runs" / "minigrid_empty_gepa_20260819" / "workspace.sqlite",
        generation=1,
        description=(
            "Generation 1 generation_start cursor (3 candidates) from the "
            "minigrid_empty_gepa_20260819 inner GEPA run. 8 train / 4 heldout seeds."
        ),
    ),
    Mint(
        filename="minigrid_mature.json",
        task_id="minigrid:2",
        label="minigrid-empty-mature",
        maturity="mature",
        workspace=PROPOSER_RUNS / "minigrid_gepa_runs" / "minigrid_empty_gepa_20260819" / "workspace.sqlite",
        generation=4,
        description=(
            "Generation 4 generation_start cursor (12 candidates) from the completed "
            "minigrid_empty_gepa_20260819 run. 8 train / 4 heldout seeds."
        ),
    ),
)


def _is_compacted(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if metadata.get("storage_compacted"):
        return True
    snapshot = record.get("snapshot") or {}
    if snapshot.get("compacted"):
        return True
    schema = str(snapshot.get("schema") or "")
    return "checkpoint_summary" in schema


def load_generation_starts(workspace_sqlite: Path) -> dict[int, dict[str, Any]]:
    """First non-compacted ``generation_start`` cursor per generation.

    First, not last: a resumed run can write a second, thinner generation_start
    for the same generation after storage compaction of its sensor frames.
    """
    conn = sqlite3.connect(f"file:{workspace_sqlite}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT generation, run_state, checkpoint_json
            FROM checkpoints
            WHERE checkpoint_kind = 'gepa_cursor'
            ORDER BY sequence_number ASC
            """
        ).fetchall()
    finally:
        conn.close()
    by_generation: dict[int, dict[str, Any]] = {}
    for generation, run_state, checkpoint_json in rows:
        if run_state != "generation_start" or generation is None:
            continue
        record = json.loads(checkpoint_json)
        if _is_compacted(record):
            continue
        by_generation.setdefault(int(generation), record)
    return by_generation


def _usage_totals(raw: Any) -> dict[str, int]:
    source = raw if isinstance(raw, dict) else {}
    totals = {key: 0 for key in USAGE_KEYS}
    for key in USAGE_KEYS:
        try:
            totals[key] = int(source.get(key) or 0)
        except (TypeError, ValueError):
            totals[key] = 0
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def mint_one(spec: Mint) -> dict[str, Any]:
    if not spec.workspace.is_file():
        return {"task_id": spec.task_id, "status": "missing_workspace", "workspace": str(spec.workspace)}
    records = load_generation_starts(spec.workspace)
    record = records.get(spec.generation)
    if record is None:
        return {
            "task_id": spec.task_id,
            "status": "missing_generation",
            "workspace": str(spec.workspace),
            "available_generations": sorted(records),
        }
    snapshot = record.get("snapshot") or {}
    candidates = snapshot.get("candidates") or []
    train_rows = snapshot.get("train_rows") or []
    heldout_rows = snapshot.get("heldout_rows") or []
    if not candidates or not train_rows or not heldout_rows:
        return {"task_id": spec.task_id, "status": "incomplete_snapshot"}

    usage = _usage_totals(snapshot.get("usage") or record.get("usage"))
    snapshot["usage"] = usage
    record["snapshot"] = snapshot
    record["usage"] = _usage_totals(record.get("usage") or usage)

    snapshot_bytes = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    payload = {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": spec.task_id,
        "label": spec.label,
        "maturity": spec.maturity,
        "description": spec.description,
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": snapshot.get("run_id") or "",
        "source_checkpoint_id": record.get("checkpoint_id") or "",
        "generation": record.get("generation"),
        "snapshot_sha256": digest,
        "usage_source": (
            f"{snapshot.get('run_id') or spec.workspace.parent.name}:"
            f"{record.get('checkpoint_id') or f'generation_{spec.generation}'}"
        ),
        # No embedded `downstream`: these families all resolve through
        # fixtures._infer_downstream, so the inner policy has one home.
        "cursor": snapshot,
        "checkpoint": record,
    }
    path = FIXTURES_DIR / spec.filename
    path.write_text(json.dumps(payload) + "\n")
    return {
        "task_id": spec.task_id,
        "status": "written",
        "path": str(path),
        "source_run_id": payload["source_run_id"],
        "generation": payload["generation"],
        "candidates": len(candidates),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "usage": usage,
        "bytes": path.stat().st_size,
    }


# ----------------------------------------------------------------------- backfill


@dataclass(frozen=True)
class Backfill:
    filename: str
    workspace: Path | None
    sequence_number: int | None
    note: str


BACKFILLS: tuple[Backfill, ...] = (
    Backfill(
        "crafter_fresh.json",
        COOKBOOK_RUNS / "crafter_gepa_public_0fbad055" / "workspace.sqlite",
        12,
        "reconstructed from checkpoint 12",
    ),
    Backfill(
        "crafter_first.json",
        COOKBOOK_RUNS / "crafter_gepa_public_0fbad055" / "workspace.sqlite",
        43,
        "reconstructed from checkpoint 43",
    ),
    Backfill(
        "crafter_mature.json",
        COOKBOOK_RUNS / "crafter_gepa_public_0fbad055" / "workspace.sqlite",
        56,
        "reconstructed from checkpoint 56 (completed)",
    ),
    Backfill(
        "healthbench_fresh.json",
        COOKBOOK_RUNS / "healthbench_groq_gepa_aug13i" / "workspace.sqlite",
        68,
        "seed rebuild; gen 0 generation_start totals",
    ),
    Backfill(
        "healthbench_first.json",
        COOKBOOK_RUNS / "healthbench_groq_gepa_aug13i" / "workspace.sqlite",
        355,
        "first scored children; gen 1 generation_start totals",
    ),
    Backfill(
        "healthbench_mature.json",
        COOKBOOK_RUNS / "healthbench_groq_gepa_aug13i" / "workspace.sqlite",
        402,
        "reconstructed from checkpoint 402 (last checkpoint of the run)",
    ),
    Backfill(
        "healthbench_accepted.json",
        COOKBOOK_RUNS / "healthbench_groq_gepa_aug13i" / "workspace.sqlite",
        402,
        "seed + accepted frontier at the end of the run",
    ),
    Backfill(
        "healthbench_openai.json",
        COOKBOOK_RUNS / "healthbench2_eval_smoke_openai_v2" / "workspace.sqlite",
        22,
        "completed OpenAI smoke; final checkpoint totals",
    ),
    # No archive behind these: seed-only cursors where nothing was rolled out.
    # Zero is the true total; `{}` only reads as unknown.
    Backfill("tau2_fresh.json", None, None, "seed-only fixture; no rollouts in this cursor"),
    Backfill("minigrid_fresh.json", None, None, "seed-only fixture; no rollouts in this cursor"),
    Backfill("officeqa_fresh.json", None, None, "seed-only fixture; no rollouts in this cursor"),
)


def _archive_usage(workspace: Path, sequence_number: int) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{workspace}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT usage_json FROM checkpoints WHERE sequence_number = ? LIMIT 1",
            (sequence_number,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"{workspace}: no checkpoint at sequence {sequence_number}")
    return _usage_totals(json.loads(row[0]))


def backfill_one(spec: Backfill) -> dict[str, Any]:
    path = FIXTURES_DIR / spec.filename
    if not path.is_file():
        return {"fixture": spec.filename, "status": "missing"}
    if spec.workspace is None:
        usage = _usage_totals({})
        source = "zeros"
    else:
        if not spec.workspace.is_file():
            return {"fixture": spec.filename, "status": "missing_workspace"}
        usage = _archive_usage(spec.workspace, int(spec.sequence_number))
        source = f"{spec.workspace.parent.name}:checkpoint_{spec.sequence_number}"

    payload = json.loads(path.read_text())
    before = json.dumps((payload.get("checkpoint") or {}).get("usage"), sort_keys=True)
    for holder in (payload.get("cursor"), payload.get("checkpoint")):
        if isinstance(holder, dict):
            holder["usage"] = dict(usage)
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("snapshot"), dict):
        checkpoint["snapshot"]["usage"] = dict(usage)
    payload["usage_source"] = source
    path.write_text(json.dumps(payload) + "\n")
    return {
        "fixture": spec.filename,
        "status": "unchanged" if before == json.dumps(usage, sort_keys=True) else "backfilled",
        "usage_source": source,
        "note": spec.note,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mint-only", action="store_true")
    parser.add_argument("--backfill-only", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if not args.backfill_only:
        report["minted"] = [mint_one(spec) for spec in MINTS]
    if not args.mint_only:
        report["backfilled"] = [backfill_one(spec) for spec in BACKFILLS]
    print(json.dumps(report, indent=2))
    failures = [
        row
        for row in report.get("minted", []) + report.get("backfilled", [])
        if row.get("status") not in {"written", "backfilled", "unchanged"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
