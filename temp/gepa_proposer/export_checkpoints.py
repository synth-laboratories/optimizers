#!/usr/bin/env python3
"""Export retained generation_start cursors from a GEPA workspace.sqlite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

TASK_MAP = (
    (0, "train:0", "banking77-fresh", "fresh", "banking77_fresh.json"),
    (1, "train:1", "banking77-first-checkpoint", "first", "banking77_first.json"),
    (2, "train:2", "banking77-mature", "mature", "banking77_mature.json"),
)


def _is_retained(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if metadata.get("storage_compacted"):
        return False
    return bool(metadata.get("retain"))


def load_retained_generation_starts(workspace_sqlite: Path) -> dict[int, dict[str, Any]]:
    conn = sqlite3.connect(workspace_sqlite)
    rows = conn.execute(
        """
        SELECT generation, run_state, checkpoint_kind, checkpoint_json, sequence_number
        FROM checkpoints
        WHERE checkpoint_kind = 'gepa_cursor'
        ORDER BY generation ASC, sequence_number ASC
        """
    ).fetchall()
    by_generation: dict[int, dict[str, Any]] = {}
    for generation, run_state, _kind, checkpoint_json, _seq in rows:
        if run_state != "generation_start" or generation is None:
            continue
        record = json.loads(checkpoint_json)
        if not _is_retained(record):
            continue
        snapshot = record.get("snapshot") or {}
        if snapshot.get("compacted") or snapshot.get("schema") == "checkpoint_summary.v1":
            continue
        by_generation[int(generation)] = record
    return by_generation


def fixture_payload(
    *,
    task_id: str,
    label: str,
    maturity: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    snapshot = record["snapshot"]
    snapshot_bytes = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    return {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": task_id,
        "label": label,
        "maturity": maturity,
        "description": f"Retained generation_start cursor at generation {record.get('generation')}.",
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": snapshot.get("run_id") or "",
        "source_checkpoint_id": record.get("checkpoint_id") or "",
        "generation": record.get("generation"),
        "snapshot_sha256": digest,
        "cursor": snapshot,
        "checkpoint": record,
    }


def export(workspace_sqlite: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_generation = load_retained_generation_starts(workspace_sqlite)
    written: dict[str, Any] = {"workspace": str(workspace_sqlite), "tasks": []}
    for generation, task_id, label, maturity, filename in TASK_MAP:
        record = by_generation.get(generation)
        if record is None:
            written["tasks"].append(
                {"task_id": task_id, "generation": generation, "status": "missing"}
            )
            continue
        payload = fixture_payload(
            task_id=task_id, label=label, maturity=maturity, record=record
        )
        path = out_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written["tasks"].append(
            {
                "task_id": task_id,
                "generation": generation,
                "status": "written",
                "path": str(path),
                "fixture_id": payload["fixture_id"],
                "candidate_count": len((payload["cursor"].get("candidates") or [])),
                "checkpoint_id": payload["source_checkpoint_id"],
            }
        )
    written["available_generations"] = sorted(by_generation)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = export(Path(args.workspace), Path(args.out_dir))
    print(json.dumps(report, indent=2))
    missing = [row for row in report["tasks"] if row["status"] == "missing"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
