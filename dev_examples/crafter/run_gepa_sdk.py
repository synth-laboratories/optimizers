"""Run the Crafter GEPA dev example through typed Python SDK config."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from synth_containers import Container
from synth_optimizers import (
    BudgetConfig,
    CacheConfig,
    GepaBudgetConfig,
    GepaConfig,
    GepaPipeline,
    ObjectiveConfig,
    OptimizerRun,
    ProposerConfig,
    ProposerPromptConfig,
    RunSettings,
    TasksetSelection,
)


def _build_container(service: Any) -> Container:
    container = Container(
        "crafter-gepa-sdk",
        runtime_id="crafter_gepa_dev",
        description="Crafter-style achievement policy optimization container.",
        policy_ready=True,
    )

    @container.task_info
    async def task_info() -> dict[str, Any]:
        return await service.task_info()

    @container.program
    async def program() -> dict[str, Any]:
        return await service.program()

    @container.taskset
    async def taskset() -> dict[str, Any]:
        return await service.taskset()

    @container.taskset_tasks
    def taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        split = str(payload.get("split") or "train")
        task_ids = [str(task_id) for task_id in payload.get("task_ids") or []]
        return {
            "tasks": [
                service._public_row(service._task_for_id(split=split, task_id=task_id))
                for task_id in task_ids
            ]
        }

    @container.rollout
    def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        return service.rollout(payload)

    return container


def _short_prompt(best_candidate: dict[str, Any]) -> str:
    payload = best_candidate.get("payload")
    if not isinstance(payload, dict):
        return ""
    prompt = str(payload.get("react_system_prompt") or "")
    if len(prompt) > 240:
        return prompt[:240] + "..."
    return prompt


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Crafter GEPA via synth_optimizers SDK.")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--policy-model",
        default=os.environ.get("GEPA_POLICY_MODEL", "gemini-3.1-flash-lite"),
    )
    parser.add_argument(
        "--policy-api-key-env",
        default=os.environ.get("GEPA_POLICY_API_KEY_ENV", "GEMINI_API_KEY"),
    )
    parser.add_argument(
        "--policy-base-url",
        default=os.environ.get(
            "GEPA_POLICY_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
    parser.add_argument(
        "--proposer-model",
        default=os.environ.get("GEPA_PROPOSER_MODEL", "gpt-5.4-nano"),
    )
    parser.add_argument(
        "--proposer-api-key-env",
        default=os.environ.get("GEPA_PROPOSER_API_KEY_ENV", "OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=int(os.environ.get("GEPA_TRAIN_SIZE", "3")),
    )
    parser.add_argument(
        "--heldout-size",
        type=int,
        default=int(os.environ.get("GEPA_HELDOUT_SIZE", "2")),
    )
    parser.add_argument(
        "--train-task-id-start",
        type=int,
        default=int(os.environ.get("GEPA_TRAIN_TASK_ID_START", "11")),
    )
    parser.add_argument(
        "--heldout-task-id-start",
        type=int,
        default=int(os.environ.get("GEPA_HELDOUT_TASK_ID_START", "101")),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("GEPA_CRAFTER_MAX_TURNS", "12")),
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=int(os.environ.get("GEPA_MAX_GENERATIONS", "2")),
    )
    parser.add_argument(
        "--proposals-per-generation",
        type=int,
        default=int(os.environ.get("GEPA_PROPOSALS_PER_GENERATION", "3")),
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=int(os.environ.get("GEPA_MINIBATCH_SIZE", "3")),
    )
    parser.add_argument(
        "--train-rollouts",
        type=int,
        default=int(os.environ.get("GEPA_MAX_TRAIN_ROLLOUTS", "48")),
    )
    parser.add_argument(
        "--heldout-rollouts",
        type=int,
        default=int(os.environ.get("GEPA_MAX_HELDOUT_ROLLOUTS", "10")),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=int(os.environ.get("GEPA_ROLLOUT_WORKERS", "8")),
    )
    parser.add_argument(
        "--max-in-flight-candidates",
        type=int,
        default=int(os.environ.get("GEPA_MAX_IN_FLIGHT_CANDIDATES", "3")),
    )
    parser.add_argument(
        "--rollout-async-timeout-seconds",
        type=int,
        default=int(os.environ.get("GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS", "120")),
    )
    args = parser.parse_args()

    if not os.environ.get(args.policy_api_key_env):
        raise SystemExit(f"{args.policy_api_key_env} is required for Crafter policy rollouts.")
    if not os.environ.get(args.proposer_api_key_env):
        raise SystemExit(f"{args.proposer_api_key_env} is required for the Codex proposer.")

    os.environ["CRAFTER_POLICY_MODEL"] = args.policy_model
    os.environ["CRAFTER_POLICY_API_KEY_ENV"] = args.policy_api_key_env
    os.environ["CRAFTER_POLICY_BASE_URL"] = args.policy_base_url
    os.environ["CRAFTER_MAX_TURNS"] = str(args.max_turns)

    sys.path.insert(0, str(example_dir))
    import synth_service_app as service

    run_id = args.run_id or f"crafter_gepa_sdk_{uuid.uuid4().hex[:8]}"
    run_dir = example_dir / "runs" / run_id
    train_ids = [
        service._row_for_seed(split="train", seed=args.train_task_id_start + offset)["task_id"]
        for offset in range(args.train_size)
    ]
    heldout_ids = [
        service._row_for_seed(split="test", seed=args.heldout_task_id_start + offset)["task_id"]
        for offset in range(args.heldout_size)
    ]
    prompt = ProposerPromptConfig.from_defaults()
    prompt.best_practices += (
        "\n\n## Crafter-specific proposer guidance\n"
        "- Use actionable_side_info.missing_achievements and last_events to propose "
        "achievement precondition rules.\n"
        "- Protect JSON action validity while improving survival and crafting order heuristics.\n"
    )

    container = _build_container(service)
    print(f"GEPA run_id: {run_id}")
    print(f"GEPA output_dir: {run_dir}")
    print(f"GEPA taskset: train={len(train_ids)} heldout={len(heldout_ids)}")
    print("GEPA objectives: achievement_unlock_rate maximize, turn_count minimize")
    print("GEPA rollout_transport: async")

    with container.serve(startup_timeout_seconds=60) as handle:
        config = GepaConfig(
            container=handle.connection(),
            run=RunSettings(run_id=run_id, output_dir=run_dir, seed=0),
            taskset=TasksetSelection(
                train_split="train",
                heldout_split="test",
                train_ids=train_ids,
                heldout_ids=heldout_ids,
            ),
            program=None,
            objectives=ObjectiveConfig(
                objective_keys=["achievement_unlock_rate", "turn_count"],
                objective_directions={
                    "achievement_unlock_rate": "maximize",
                    "turn_count": "minimize",
                },
                selection_objective="achievement_unlock_rate",
                protected_objectives=["achievement_unlock_rate"],
                frontier_type="per_objective",
            ),
            policy=None,
            proposer=ProposerConfig(
                model=args.proposer_model,
                auth_mode="api_key",
                api_key_env=args.proposer_api_key_env,
                copy_host_auth=False,
                prompt=prompt,
            ),
            budgets=GepaBudgetConfig(
                max_generations=args.max_generations,
                proposals_per_generation=args.proposals_per_generation,
                minibatch_size=args.minibatch_size,
                max_total_rollouts=args.train_rollouts + args.heldout_rollouts,
                max_train_rollouts=args.train_rollouts,
                max_heldout_rollouts=args.heldout_rollouts,
            ),
            budget=BudgetConfig(max_cost_usd=0.0),
            pipeline=GepaPipeline(
                rollout_transport="async",
                rollout_timeout_seconds=args.rollout_async_timeout_seconds,
                rollout_concurrency=args.rollout_workers,
                candidate_concurrency=args.max_in_flight_candidates,
            ),
            cache=CacheConfig(
                mode="readwrite",
                path=run_dir / f"{run_id}_cache.sqlite",
                namespace=run_id,
            ),
        )
        result = OptimizerRun(config).execute()

    best_candidate = result.best_candidate or {}
    print("Crafter GEPA SDK run complete")
    print(f"manifest: {result.manifest_path}")
    print(f"cost_usd: {result.cost_usd:.4f}")
    if isinstance(best_candidate, dict):
        print(f"best_candidate: {best_candidate.get('candidate_id', '?')}")
        prompt_text = _short_prompt(best_candidate)
        if prompt_text:
            print("best_prompt:")
            print(prompt_text)
        print("best_candidate_json:")
        print(json.dumps(best_candidate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
