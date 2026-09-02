#!/usr/bin/env python3
"""Bounded paid Tinker canary: gpt-oss-20b Banking77 SFT then cispo.slime.v1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_key(path: Path | None) -> None:
    if os.environ.get("TINKER_API_KEY", "").strip():
        return
    if path is None or not path.is_file():
        raise SystemExit("TINKER_API_KEY is unset and --env-file was not found")
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = raw.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit("TINKER_API_KEY is missing from the env file")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-cispo", action="store_true")
    parser.add_argument(
        "--sft-events",
        type=Path,
        default=None,
        help="Reuse a completed SFT canary's events instead of paying for another SFT step",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory already exists")
    load_key(args.env_file)
    args.output_dir.mkdir(parents=True)

    from synth_optimizers.cispo_executor import TinkerCispoExecutor
    from synth_optimizers.providers.tinker.validation import write_receipt
    from synth_optimizers.recipes.banking77 import cispo_recipe, sft_recipe
    from synth_optimizers.runtime import JobStore
    from synth_optimizers.sft_executor import TinkerSftExecutor

    store = JobStore(args.output_dir / "jobs.sqlite")
    if args.sft_events is not None:
        sft_events = json.loads(args.sft_events.read_text(encoding="utf-8"))
        created = next(
            event for event in sft_events if event["event_type"] == "sft.checkpoint.created"
        )
        write_json(args.output_dir / "sft.reused.json", {"sft_events": str(args.sft_events), "status": "reused"})
        sft_status = "reused"
    else:
        sft = TinkerSftExecutor.local(store)
        sft_request = sft_recipe(steps=1).request
        sft_request["training"]["batch_size"] = 2
        sft_request["rank"] = 8
        sft_result = sft.submit(sft_request, job_id="sft_canary")
        write_json(args.output_dir / "sft.status.json", {k: v for k, v in sft_result.items() if k != "events"})
        write_json(args.output_dir / "sft.events.json", sft_result["events"])
        if sft_result["status"] != "completed":
            raise SystemExit(f"SFT canary failed: {sft_result.get('error')}")
        created = next(
            event for event in sft_result["events"] if event["event_type"] == "sft.checkpoint.created"
        )
        sft_status = sft_result["status"]
    if args.skip_cispo:
        store.close()
        return
    cispo = TinkerCispoExecutor.local(store, allow_unvalidated_canary=True)
    cispo_request = cispo_recipe(mode="learning_signal", updates=1).request
    cispo_request["allow_unvalidated_canary"] = True
    cispo_request["training"]["max_sample_tokens"] = 32
    cispo_request["parent_checkpoint"] = {
        "checkpoint_id": created["payload"]["training_checkpoint_id"],
        "provider_reference": created["payload"]["training_provider_reference"],
        "resume_token": created["payload"]["resume_token"],
        "kind": "training",
        "step": created["payload"]["step"],
        "digest": created["payload"]["digest"],
    }
    cispo_result = cispo.submit(cispo_request, job_id="cispo_canary")
    write_json(
        args.output_dir / "cispo.status.json",
        {k: v for k, v in cispo_result.items() if k != "events"},
    )
    write_json(args.output_dir / "cispo.events.json", cispo_result["events"])
    paid = any(event["event_type"] == "cispo.importance_ratio.measured" for event in cispo_result["events"])
    receipt = write_receipt(
        args.output_dir / "cispo.slime.v1.receipt.json",
        {
            "model_id": "openai/gpt-oss-20b",
            "validated": cispo_result["status"] == "completed" and paid,
            "paid_update": paid,
            "sft_job_id": "sft_canary",
            "cispo_job_id": "cispo_canary",
            "renderer_version": cispo_request["renderer_version"],
            "cost_usd": None,
            "cost_missing": True,
        },
    )
    write_json(args.output_dir / "summary.json", {"sft": sft_status, "cispo": cispo_result["status"], "paid_update": paid, "receipt": receipt})
    store.close()
    if cispo_result["status"] != "completed":
        raise SystemExit(f"CISPO canary failed: {cispo_result.get('error')}")
    if not paid:
        raise SystemExit("CISPO canary completed without a paid update; cispo.slime.v1 stays unvalidated")


if __name__ == "__main__":
    main()
