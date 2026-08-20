#!/usr/bin/env python3
"""Rebuild restorable HealthBench GEPA fixtures from a compacted workspace."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(
    "/Users/joshuapurtell/Documents/GitHub/synth-cookbooks-public/"
    "cookbooks/optimizers/gepa/runs/healthbench_groq_gepa_aug13i/workspace.sqlite"
)
OUT_DIR = Path(__file__).resolve().parent / "fixtures"
KEEP_STATUSES = {"full_train_evaluated", "accepted", "rejected_full_train"}
MINIBATCH_SIZE = 20
DOWNSTREAM = {
    "id": "healthbench2",
    "image_id": "healthbench2",
    "url_env": "HEALTHBENCH_URL",
    "url_pool_env": "HEALTHBENCH_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "api_family": "chat_completions",
        "base_url": "https://api.groq.com/openai/v1",
        "max_tokens": 1536,
        "env_var": "GROQ_API_KEY",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scores(conn: sqlite3.Connection, candidate_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT example_id, task_id, reward, evaluation_stage
        FROM candidate_seed_rewards
        WHERE candidate_id = ?
        ORDER BY example_id
        """,
        (candidate_id,),
    ).fetchall()
    return [
        {
            "example_id": example_id,
            "task_id": task_id or example_id,
            "reward": float(reward or 0.0),
            "evaluation_stage": stage,
        }
        for example_id, task_id, reward, stage in rows
        if example_id
    ]


def _compact_candidate(record: dict, scores: list[dict], *, status: str) -> dict:
    train_scores = [item for item in scores if str(item.get("example_id") or "").startswith("train:")]
    payload = record.get("payload") or {}
    return {
        "candidate_id": record.get("candidate_id"),
        "parent_id": record.get("parent_id"),
        "source": record.get("source") or ("seed" if record.get("parent_id") is None else "reflector:parent_variation"),
        "status": status,
        "payload": payload,
        "lever_bundle": record.get("lever_bundle"),
        "train_reward": record.get("train_reward"),
        "minibatch_reward": record.get("minibatch_reward"),
        "heldout_reward": record.get("heldout_reward"),
        "train_scores": train_scores,
        "minibatch_scores": train_scores[:MINIBATCH_SIZE],
        "acceptance_metadata": {},
    }


def _cursor(*, run_id: str, generation: int, candidates: list[dict], train_rows: list, heldout_rows: list, program: dict, reconstructed_from: str) -> dict:
    best = max(
        candidates,
        key=lambda row: float(row.get("train_reward") or row.get("minibatch_reward") or 0.0),
    )
    minibatch_rows = train_rows[:MINIBATCH_SIZE]
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
        "best_candidate_id": best.get("candidate_id"),
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
        "program": program,
        "objective_set": None,
        "state_history": [],
        "terminal_summary": None,
        "error_summary": None,
        "metadata": {"retain": True, "reconstructed_from": reconstructed_from},
    }


OPENAI_WORKSPACE = Path(
    "/Users/joshuapurtell/Documents/GitHub/synth-cookbooks-public/"
    "cookbooks/optimizers/gepa/runs/healthbench2_eval_smoke_openai_v2/workspace.sqlite"
)
OPENAI_DOWNSTREAM = {
    "id": "healthbench2",
    "image_id": "healthbench2",
    "url_env": "HEALTHBENCH_URL",
    "url_pool_env": "HEALTHBENCH_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 1536,
        "env_var": "OPENAI_API_KEY",
    },
}


def _compact_row(row: dict) -> dict:
    task_id = str(row.get("task_id") or row.get("example_id") or "").strip()
    compact = {"task_id": task_id, "split": row.get("split"), "seed": row.get("seed")}
    if row.get("example_id"):
        compact["example_id"] = row.get("example_id")
    return compact


def _write_fixture(
    *,
    path: Path,
    task_id: str,
    label: str,
    maturity: str,
    description: str,
    cursor: dict,
    source_run_id: str,
    downstream: dict | None = None,
) -> dict:
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
        "downstream": downstream or DOWNSTREAM,
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


def main() -> int:
    conn = sqlite3.connect(WORKSPACE)
    train_rows = [{"task_id": f"train:{index}", "split": "train", "seed": index} for index in range(60)]
    heldout_rows = [
        {"task_id": f"heldout:{100 + index}", "split": "heldout", "seed": 100 + index}
        for index in range(50)
    ]
    program = json.loads(
        conn.execute("SELECT program_json FROM prompt_program_snapshots LIMIT 1").fetchone()[0]
    )
    table_rows = list(conn.execute("SELECT candidate_id, status, record_json FROM candidates ORDER BY updated_at"))
    by_id: dict[str, dict] = {}
    for candidate_id, status, record_json in table_rows:
        record = json.loads(record_json)
        if not record.get("payload"):
            payload_row = conn.execute(
                "SELECT payload_json FROM candidate_payloads WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if payload_row:
                record["payload"] = json.loads(payload_row[0])
        record["candidate_id"] = record.get("candidate_id") or candidate_id
        by_id[str(record["candidate_id"])] = _compact_candidate(
            record, _scores(conn, candidate_id), status=status
        )

    snap_row = conn.execute(
        "SELECT checkpoint_json FROM checkpoints WHERE sequence_number = 402"
    ).fetchone()
    snap_candidates = []
    if snap_row:
        snapshot = json.loads(snap_row[0]).get("snapshot") or {}
        for raw in snapshot.get("candidates") or []:
            cid = str(raw.get("candidate_id") or "")
            compact = by_id.get(cid)
            if compact is None:
                compact = _compact_candidate(raw, [], status=str(raw.get("status") or "registered"))
            else:
                compact = dict(compact)
                compact["status"] = raw.get("status") or compact["status"]
                compact["source"] = raw.get("source") or compact["source"]
            snap_candidates.append(compact)

    seed = next((row for row in by_id.values() if row.get("parent_id") is None), None)
    if seed is None:
        raise SystemExit("no HealthBench seed candidate")
    scored = [row for row in by_id.values() if row["status"] in KEEP_STATUSES]
    if not scored:
        raise SystemExit("no scored HealthBench candidates")
    mature = snap_candidates or list(by_id.values())
    run_id = "healthbench_groq_gepa_aug13i"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        _write_fixture(
            path=OUT_DIR / "healthbench_fresh.json",
            task_id="healthbench:0",
            label="healthbench-fresh",
            maturity="fresh",
            description="Seed-only generation_start cursor reconstructed from healthbench_groq_gepa_aug13i.",
            cursor=_cursor(
                run_id=run_id,
                generation=0,
                candidates=[seed],
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                program=program,
                reconstructed_from=f"{run_id}:seed",
            ),
            source_run_id=run_id,
        ),
        _write_fixture(
            path=OUT_DIR / "healthbench_first.json",
            task_id="healthbench:1",
            label="healthbench-first-checkpoint",
            maturity="first",
            description="Reconstructed generation_start cursor after the first scored HealthBench children.",
            cursor=_cursor(
                run_id=run_id,
                generation=1,
                candidates=scored,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                program=program,
                reconstructed_from=run_id,
            ),
            source_run_id=run_id,
        ),
        _write_fixture(
            path=OUT_DIR / "healthbench_mature.json",
            task_id="healthbench:2",
            label="healthbench-mature",
            maturity="mature",
            description="Latest uncompacted HealthBench archive reconstructed as generation_start.",
            cursor=_cursor(
                run_id=run_id,
                generation=1,
                candidates=mature,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                program=program,
                reconstructed_from=f"{run_id}:checkpoint_402",
            ),
            source_run_id=run_id,
        ),
        _write_fixture(
            path=OUT_DIR / "healthbench_accepted.json",
            task_id="healthbench:4",
            label="healthbench-accepted-frontier",
            maturity="first",
            description="Seed plus the accepted child from healthbench_groq_gepa_aug13i.",
            cursor=_cursor(
                run_id=run_id,
                generation=1,
                candidates=[
                    row
                    for row in scored
                    if row.get("parent_id") is None or row.get("status") == "accepted"
                ],
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                program=program,
                reconstructed_from=f"{run_id}:accepted",
            ),
            source_run_id=run_id,
        ),
    ]
    written.append(_write_openai_fixture())
    print(json.dumps(written, indent=2))
    return 0


def _write_openai_fixture() -> dict:
    conn = sqlite3.connect(OPENAI_WORKSPACE)
    program = json.loads(
        conn.execute("SELECT program_json FROM prompt_program_snapshots LIMIT 1").fetchone()[0]
    )
    snap_row = conn.execute(
        "SELECT checkpoint_json FROM checkpoints ORDER BY sequence_number DESC LIMIT 1"
    ).fetchone()
    if snap_row is None:
        raise SystemExit("no OpenAI HealthBench checkpoint")
    snapshot = json.loads(snap_row[0]).get("snapshot") or {}
    raw = (snapshot.get("candidates") or [None])[0]
    if not isinstance(raw, dict):
        raise SystemExit("OpenAI HealthBench checkpoint has no candidate")
    scores = [
        {
            "example_id": item.get("example_id") or item.get("task_id"),
            "task_id": item.get("task_id") or item.get("example_id"),
            "reward": item.get("reward"),
        }
        for item in (raw.get("train_scores") or [])
    ]
    candidate = _compact_candidate(raw, scores, status=str(raw.get("status") or "full_train_evaluated"))
    candidate["heldout_reward"] = raw.get("heldout_reward")
    run_id = "healthbench2_eval_smoke_openai_v2"
    train_rows = [_compact_row(row) for row in snapshot.get("train_rows") or []]
    heldout_rows = [_compact_row(row) for row in snapshot.get("heldout_rows") or []]
    if not train_rows or not heldout_rows:
        raise SystemExit("OpenAI HealthBench checkpoint is missing train/heldout rows")
    return _write_fixture(
        path=OUT_DIR / "healthbench_openai.json",
        task_id="healthbench:3",
        label="healthbench-openai-scored-seed",
        maturity="fresh",
        description=(
            "Scored seed plus heldout from healthbench2_eval_smoke_openai_v2 "
            "(OpenAI gpt-4.1-nano inner policy, 2/2 train/heldout)."
        ),
        cursor=_cursor(
            run_id=run_id,
            generation=0,
            candidates=[candidate],
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            program=program,
            reconstructed_from=f"{run_id}:completed",
        ),
        source_run_id=run_id,
        downstream=OPENAI_DOWNSTREAM,
    )


if __name__ == "__main__":
    raise SystemExit(main())
