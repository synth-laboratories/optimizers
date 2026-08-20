#!/usr/bin/env python3
"""Write a seed-only MiniGrid DoorKey GEPA cursor fixture (fresh)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "minigrid_container"))
from tasks import DEFAULT_SYSTEM_PROMPT, HELDOUT_SEEDS, TRAIN_SEEDS, rows_for  # noqa: E402

OUT = HERE / "fixtures" / "minigrid_fresh.json"
DOWNSTREAM = {
    "id": "minigrid",
    "image_id": "minigrid_doorkey",
    "url_env": "MINIGRID_URL",
    "url_pool_env": "MINIGRID_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 64,
        "env_var": "OPENAI_API_KEY",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    train_rows = rows_for("train")
    heldout_rows = rows_for("heldout")
    candidate_id = "minigrid_doorkey_seed"
    cursor = {
        "schema_version": "gepa_cursor.v1",
        "run_id": "minigrid_fresh_seed",
        "phase": "generation_start",
        "generation": 0,
        "proposal_index": 0,
        "proposal_queue": [],
        "heldout_candidate_index": 0,
        "pending_job_id": None,
        "pending_effect_id": None,
        "pending_reservation_ids": [],
        "active_evaluation": None,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "parent_id": None,
                "source": "seed",
                "status": "full_train_evaluated",
                "payload": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
                "lever_bundle": {
                    "schema_version": "lever_bundle.v1",
                    "bundle_id": candidate_id,
                    "parent_ids": [],
                    "mutated_lever_ids": ["system_prompt"],
                    "values": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
                    "metadata": {},
                },
                "train_reward": None,
                "minibatch_reward": None,
                "heldout_reward": None,
                "train_scores": [],
                "minibatch_scores": [],
                "acceptance_metadata": {},
            }
        ],
        "best_candidate_id": candidate_id,
        "rollout_task_id": None,
        "rollout_count": 0,
        "cost_usd": 0.0,
        "usage": {},
        "usage_ledger": [],
        "stopper_states": [],
        "stopper_sequence": 0,
        "checkpoint_sequence": 1,
        "train_rows": train_rows,
        "minibatch_rows": train_rows[:4],
        "reflection_rows": train_rows[:4],
        "heldout_rows": heldout_rows,
        "program": {
            "version": "prompt_program.v1",
            "program_id": "minigrid_system_prompt_gepa",
            "modules": [
                {
                    "module_id": "system_prompt",
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                    "mutable": True,
                    "candidate_field": "system_prompt",
                    "template_variables": [],
                }
            ],
            "target_modules": [
                {
                    "module_id": "system_prompt",
                    "candidate_field": "system_prompt",
                    "objective": "outcome_reward",
                }
            ],
            "seed_candidate": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
        },
        "objective_set": None,
        "state_history": [],
        "terminal_summary": None,
        "error_summary": None,
        "metadata": {
            "retain": True,
            "benchmark": "MiniGrid Empty-5x5",
            "dataset": "MiniGrid-Empty-5x5-v0",
            "train_seeds": TRAIN_SEEDS,
            "heldout_seeds": HELDOUT_SEEDS,
        },
    }
    digest = hashlib.sha256(json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint_id = f"checkpoint_{digest[:16]}"
    payload = {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": "minigrid:0",
        "label": "minigrid-empty-fresh",
        "maturity": "fresh",
        "description": (
            "Seed-only MiniGrid Empty-5x5 GEPA fixture. Inner container runs live "
            "gymnasium episodes; GEPA mutates system_prompt. Set MINIGRID_ENV_ID to "
            "MiniGrid-DoorKey-5x5-v0 for the locked-door variant. "
            f"Train seeds {TRAIN_SEEDS[0]}..{TRAIN_SEEDS[-1]} ({len(TRAIN_SEEDS)}), "
            f"heldout seeds {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]} ({len(HELDOUT_SEEDS)})."
        ),
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": "minigrid_fresh_seed",
        "source_checkpoint_id": checkpoint_id,
        "generation": 0,
        "snapshot_sha256": digest,
        "downstream": DOWNSTREAM,
        "cursor": cursor,
        "checkpoint": {
            "best_candidate_id": candidate_id,
            "candidate_count": 1,
            "candidate_id": candidate_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_kind": "gepa_cursor",
            "cost_usd": 0.0,
            "created_at": _utc(),
            "evaluation_stage": "generation_start",
            "frontier_count": 1,
            "generation": 0,
            "metadata": {"retain": True},
            "reason": "minigrid_doorkey_seed",
            "rollout_count": 0,
            "run_state": "generation_start",
            "schema_version": "checkpoint_record.v1",
            "sequence_number": 1,
            "snapshot": cursor,
            "status": "retained",
            "usage": {},
        },
    }
    OUT.write_text(json.dumps(payload) + "\n")
    print(
        json.dumps(
            {
                "path": str(OUT),
                "bytes": OUT.stat().st_size,
                "train": len(TRAIN_SEEDS),
                "heldout": len(HELDOUT_SEEDS),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
