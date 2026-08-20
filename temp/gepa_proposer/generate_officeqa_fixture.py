#!/usr/bin/env python3
"""Write a seed-only OfficeQA GEPA cursor fixture (fresh)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "officeqa_fresh.json"
SEED = (
    "You answer questions about U.S. Treasury documents. Use only the supplied "
    "source text. Be exact on figures, years, and units. Return only the final "
    "answer string, with no explanation."
)
DOWNSTREAM = {
    "id": "officeqa",
    "image_id": "officeqa",
    "url_env": "OFFICEQA_URL",
    "url_pool_env": "OFFICEQA_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1",
        "api_family": "chat_completions",
        "max_tokens": 256,
        "env_var": "OPENAI_API_KEY",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    train_rows = [{"task_id": f"train:{index}", "split": "train", "seed": index} for index in range(24)]
    heldout_rows = [{"task_id": f"heldout:{index}", "split": "heldout", "seed": index} for index in range(16)]
    candidate_id = "officeqa_seed"
    cursor = {
        "schema_version": "gepa_cursor.v1",
        "run_id": "officeqa_fresh_seed",
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
                "payload": {"system_prompt": SEED},
                "lever_bundle": {
                    "schema_version": "lever_bundle.v1",
                    "bundle_id": candidate_id,
                    "parent_ids": [],
                    "mutated_lever_ids": ["system_prompt"],
                    "values": {"system_prompt": SEED},
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
        "minibatch_rows": train_rows[:8],
        "reflection_rows": train_rows[:8],
        "heldout_rows": heldout_rows,
        "program": {
            "version": "prompt_program.v1",
            "program_id": "officeqa_system_prompt",
            "modules": [
                {
                    "module_id": "system_prompt",
                    "role": "system",
                    "content": SEED,
                    "mutable": True,
                    "candidate_field": "system_prompt",
                    "template_variables": [],
                }
            ],
            "target_modules": [
                {"module_id": "system_prompt", "candidate_field": "system_prompt", "objective": "outcome_reward"}
            ],
            "seed_candidate": {"system_prompt": SEED},
        },
        "objective_set": None,
        "state_history": [],
        "terminal_summary": None,
        "error_summary": None,
        "metadata": {
            "retain": True,
            "benchmark": "OfficeQA Full (easy train / hard heldout when CSV is mounted)",
            "dataset": "databricks/officeqa",
        },
    }
    digest = hashlib.sha256(json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint_id = f"checkpoint_{digest[:16]}"
    payload = {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": "officeqa:0",
        "label": "officeqa-fresh",
        "maturity": "fresh",
        "description": (
            "Seed-only OfficeQA GEPA fixture. Inner container is Databricks OfficeQA "
            "(Treasury Bulletins). Mount OFFICEQA_CSV from huggingface.co/datasets/databricks/officeqa."
        ),
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": "officeqa_fresh_seed",
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
            "reason": "officeqa_seed",
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
    print(json.dumps({"path": str(OUT), "bytes": OUT.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
