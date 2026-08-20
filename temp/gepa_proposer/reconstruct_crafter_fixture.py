#!/usr/bin/env python3
"""Export restorable Crafter GEPA fixtures from crafter_gepa_public_0fbad055."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(
    "/Users/joshuapurtell/Documents/GitHub/synth-cookbooks-public/"
    "cookbooks/optimizers/gepa/runs/crafter_gepa_public_0fbad055/workspace.sqlite"
)
OUT_DIR = Path(__file__).resolve().parent / "fixtures"
MINIBATCH_SIZE = 4
DOWNSTREAM = {
    "id": "crafter",
    "image_id": "crafter",
    "url_env": "CRAFTER_URL",
    "url_pool_env": "CRAFTER_URLS",
    "candidate_field": "react_system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 256,
        "env_var": "OPENAI_API_KEY",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_score(score: dict) -> dict:
    example_id = str(score.get("example_id") or score.get("task_id") or "").strip()
    compact = {
        "example_id": example_id,
        "task_id": str(score.get("task_id") or example_id),
        "reward": score.get("reward"),
    }
    if score.get("seed") is not None:
        compact["seed"] = score.get("seed")
    return compact


def _compact_row(row: dict) -> dict:
    example_id = str(row.get("task_id") or row.get("example_id") or "").strip()
    compact = {
        "task_id": example_id,
        "example_id": example_id,
        "split": row.get("split"),
        "seed": row.get("seed"),
    }
    return compact


def _compact_candidate(raw: dict) -> dict:
    train_scores = [_compact_score(item) for item in (raw.get("train_scores") or [])]
    minibatch_scores = [_compact_score(item) for item in (raw.get("minibatch_scores") or [])]
    payload = raw.get("payload") or {}
    return {
        "candidate_id": raw.get("candidate_id"),
        "parent_id": raw.get("parent_id"),
        "source": raw.get("source") or ("seed" if raw.get("parent_id") is None else "reflector:parent_variation"),
        "status": raw.get("status"),
        "payload": {"react_system_prompt": payload.get("react_system_prompt")} if isinstance(payload, dict) else payload,
        "lever_bundle": raw.get("lever_bundle"),
        "train_reward": raw.get("train_reward"),
        "minibatch_reward": raw.get("minibatch_reward"),
        "heldout_reward": raw.get("heldout_reward"),
        "train_scores": train_scores,
        "minibatch_scores": minibatch_scores[:MINIBATCH_SIZE] or train_scores[:MINIBATCH_SIZE],
        "acceptance_metadata": {},
    }


def _cursor(*, run_id: str, generation: int, snapshot: dict, reconstructed_from: str) -> dict:
    candidates = [_compact_candidate(row) for row in snapshot.get("candidates") or []]
    train_rows = [_compact_row(row) for row in snapshot.get("train_rows") or []]
    heldout_rows = [_compact_row(row) for row in snapshot.get("heldout_rows") or []]
    minibatch_rows = [_compact_row(row) for row in snapshot.get("minibatch_rows") or []] or train_rows[:MINIBATCH_SIZE]
    best_id = snapshot.get("best_candidate_id") or (candidates[0].get("candidate_id") if candidates else None)
    return {
        "schema_version": "gepa_cursor.v1",
        "run_id": run_id,
        "phase": "generation_start",
        "generation": generation,
        "proposal_index": 0,
        "proposal_queue": [],
        "heldout_candidate_index": 0,
        "pending_job_id": None,
        "pending_effect_id": None,
        "pending_reservation_ids": [],
        "active_evaluation": None,
        "candidates": candidates,
        "best_candidate_id": best_id,
        "rollout_task_id": None,
        "rollout_count": 0,
        "cost_usd": 0.0,
        "usage": {},
        "usage_ledger": [],
        "stopper_states": [],
        "stopper_sequence": 0,
        "checkpoint_sequence": 1,
        "train_rows": train_rows,
        "minibatch_rows": minibatch_rows,
        "reflection_rows": minibatch_rows,
        "heldout_rows": heldout_rows,
        "program": snapshot.get("program"),
        "objective_set": None,
        "state_history": [],
        "terminal_summary": None,
        "error_summary": None,
        "metadata": {"retain": True, "reconstructed_from": reconstructed_from},
    }


def _write_fixture(*, path: Path, task_id: str, label: str, maturity: str, description: str, cursor: dict, source_run_id: str) -> dict:
    snapshot_bytes = json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    checkpoint_id = f"checkpoint_{digest[:16]}"
    checkpoint = {
        "best_candidate_id": cursor["best_candidate_id"],
        "candidate_count": len(cursor["candidates"]),
        "candidate_id": cursor["best_candidate_id"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "gepa_cursor",
        "cost_usd": 0.0,
        "created_at": _utc(),
        "evaluation_stage": "generation_start",
        "frontier_count": len(cursor["candidates"]),
        "generation": cursor["generation"],
        "metadata": {"retain": True},
        "reason": "reconstructed_generation_start",
        "rollout_count": 0,
        "run_state": "generation_start",
        "schema_version": "checkpoint_record.v1",
        "sequence_number": 1,
        "snapshot": cursor,
        "status": "retained",
        "usage": {},
    }
    payload = {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": task_id,
        "label": label,
        "maturity": maturity,
        "description": description,
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": source_run_id,
        "source_checkpoint_id": checkpoint_id,
        "generation": cursor["generation"],
        "snapshot_sha256": digest,
        "downstream": DOWNSTREAM,
        "cursor": cursor,
        "checkpoint": checkpoint,
    }
    path.write_text(json.dumps(payload) + "\n")
    return {
        "path": str(path),
        "task_id": task_id,
        "candidates": len(cursor["candidates"]),
        "bytes": path.stat().st_size,
    }


def _snapshot(conn: sqlite3.Connection, sequence_number: int) -> dict:
    row = conn.execute(
        "SELECT checkpoint_json FROM checkpoints WHERE sequence_number = ?",
        (sequence_number,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"missing Crafter checkpoint {sequence_number}")
    snapshot = json.loads(row[0]).get("snapshot") or {}
    if not snapshot.get("candidates") or not snapshot.get("train_rows") or not snapshot.get("heldout_rows"):
        raise SystemExit(f"Crafter checkpoint {sequence_number} is missing archive rows")
    return snapshot


def main() -> int:
    conn = sqlite3.connect(WORKSPACE)
    run_id = "crafter_gepa_public_0fbad055"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        _write_fixture(
            path=OUT_DIR / "crafter_fresh.json",
            task_id="crafter:0",
            label="crafter-fresh",
            maturity="fresh",
            description="Generation 0 generation_start cursor from crafter_gepa_public_0fbad055.",
            cursor=_cursor(
                run_id=run_id,
                generation=0,
                snapshot=_snapshot(conn, 12),
                reconstructed_from=f"{run_id}:checkpoint_12",
            ),
            source_run_id=run_id,
        ),
        _write_fixture(
            path=OUT_DIR / "crafter_first.json",
            task_id="crafter:1",
            label="crafter-first-checkpoint",
            maturity="first",
            description="Generation 2 generation_start cursor after the first accepted Crafter child.",
            cursor=_cursor(
                run_id=run_id,
                generation=2,
                snapshot=_snapshot(conn, 43),
                reconstructed_from=f"{run_id}:checkpoint_43",
            ),
            source_run_id=run_id,
        ),
        _write_fixture(
            path=OUT_DIR / "crafter_mature.json",
            task_id="crafter:2",
            label="crafter-mature",
            maturity="mature",
            description=(
                "Completed Crafter archive reconstructed as generation_start so eval_uplift "
                "has a real heldout baseline."
            ),
            cursor=_cursor(
                run_id=run_id,
                generation=2,
                snapshot=_snapshot(conn, 56),
                reconstructed_from=f"{run_id}:checkpoint_56",
            ),
            source_run_id=run_id,
        ),
    ]
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
