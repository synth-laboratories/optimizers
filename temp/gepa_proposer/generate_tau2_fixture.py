#!/usr/bin/env python3
"""Write a seed-only τ²-bench retail GEPA cursor fixture (fresh)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tau2_container"))
from tasks import HELDOUT_IDS, TRAIN_IDS, rows_for  # noqa: E402

OUT = HERE / "fixtures" / "tau2_fresh.json"
POLICY = (HERE / "tau2_container" / "policy.md").read_text(encoding="utf-8")
DOWNSTREAM = {
    "id": "tau2",
    "image_id": "tau2_retail",
    "url_env": "TAU2_URL",
    "url_pool_env": "TAU2_URLS",
    "candidate_field": "domain_policy",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 512,
        "env_var": "OPENAI_API_KEY",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    train_rows = rows_for("train")
    heldout_rows = rows_for("heldout")
    candidate_id = "tau2_retail_seed"
    cursor = {
        "schema_version": "gepa_cursor.v1",
        "run_id": "tau2_fresh_seed",
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
                "payload": {"domain_policy": POLICY},
                "lever_bundle": {
                    "schema_version": "lever_bundle.v1",
                    "bundle_id": candidate_id,
                    "parent_ids": [],
                    "mutated_lever_ids": ["domain_policy"],
                    "values": {"domain_policy": POLICY},
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
            "program_id": "tau2_retail_domain_policy",
            "modules": [
                {
                    "module_id": "domain_policy",
                    "role": "system",
                    "content": POLICY,
                    "mutable": True,
                    "candidate_field": "domain_policy",
                    "template_variables": [],
                }
            ],
            "target_modules": [
                {"module_id": "domain_policy", "candidate_field": "domain_policy", "objective": "outcome_reward"}
            ],
            "seed_candidate": {"domain_policy": POLICY},
        },
        "objective_set": None,
        "state_history": [],
        "terminal_summary": None,
        "error_summary": None,
        "metadata": {
            "retain": True,
            "benchmark": "τ²-bench retail",
            "dataset": "sierra-research/tau2-bench",
            "official_train_ids": TRAIN_IDS,
            "official_heldout_ids": HELDOUT_IDS,
        },
    }
    digest = hashlib.sha256(json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint_id = f"checkpoint_{digest[:16]}"
    payload = {
        "schema": "gepa_cursor_fixture.v1",
        "task_id": "tau2:0",
        "label": "tau2-retail-fresh",
        "maturity": "fresh",
        "description": (
            "Seed-only τ²-bench retail GEPA fixture. Inner container wraps "
            "sierra-research/tau2-bench; GEPA mutates domain_policy. "
            f"Train ids {TRAIN_IDS[0]}..{TRAIN_IDS[-1]} ({len(TRAIN_IDS)}), "
            f"heldout ids {HELDOUT_IDS[0]}..{HELDOUT_IDS[-1]} ({len(HELDOUT_IDS)})."
        ),
        "fixture_id": f"gepa_fixture_{digest[:16]}",
        "source_run_id": "tau2_fresh_seed",
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
            "reason": "tau2_retail_seed",
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
    print(json.dumps({"path": str(OUT), "bytes": OUT.stat().st_size, "train": len(TRAIN_IDS), "heldout": len(HELDOUT_IDS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
