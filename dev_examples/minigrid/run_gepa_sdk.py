"""Run the MiniGrid GEPA dev example through typed Python SDK config."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
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
    RunSettings,
    TasksetSelection,
)


class ProfileInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "default"
    description: str = ""


class BaseRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str = "minigrid_gepa_public"


class PolicyRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "gpt-4.1-nano"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class ProposerRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-5.4-nano"
    auth_mode: str = "api_key"
    api_key_env: str = "OPENAI_API_KEY"
    copy_host_auth: bool = False


class TaskRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    max_steps: int = 48


class TasksetRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_size: int = 8
    heldout_size: int = 8
    train_task_id_start: int | str = "random"
    heldout_task_id_start: int | str = "random"


class BudgetsRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_rollouts: int = 80
    heldout_rollouts: int = 16


class PipelineRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rollout_workers: int = 10
    max_in_flight_candidates: int = 3
    proposals_per_generation: int = 3
    max_generations: int = 2
    minibatch_size: int = 4


class TimeoutsRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rollout_async_seconds: int = 120


class MiniGridRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile: ProfileInfo = Field(default_factory=ProfileInfo)
    base: BaseRunProfile = Field(default_factory=BaseRunProfile)
    policy: PolicyRunProfile = Field(default_factory=PolicyRunProfile)
    proposer: ProposerRunProfile = Field(default_factory=ProposerRunProfile)
    task: TaskRunProfile = Field(default_factory=TaskRunProfile)
    taskset: TasksetRunProfile = Field(default_factory=TasksetRunProfile)
    budgets: BudgetsRunProfile = Field(default_factory=BudgetsRunProfile)
    pipeline: PipelineRunProfile = Field(default_factory=PipelineRunProfile)
    timeouts: TimeoutsRunProfile = Field(default_factory=TimeoutsRunProfile)


def _resolve_profile(example_dir: Path, raw: str) -> Path:
    if raw.endswith(".toml") or "/" in raw:
        return Path(raw).resolve()
    return example_dir / "run_profiles" / f"{raw}.toml"


def _task_id_start(raw: int | str) -> int:
    if isinstance(raw, int):
        return raw
    if raw in {"random", "auto"}:
        return int.from_bytes(os.urandom(4), "big")
    return int(raw)


def _build_container(service: Any) -> Container:
    container = Container(
        "minigrid-gepa-sdk",
        runtime_id="minigrid_gepa_live",
        description="MiniGrid live gymnasium policy optimization container.",
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
            "tasks": [service._task_for_id(split=split, task_id=task_id) for task_id in task_ids]
        }

    @container.rollout
    def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        return service.rollout(payload)

    return container


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run MiniGrid GEPA through the Python SDK.")
    parser.add_argument("profile", nargs="?", default="default")
    parser.add_argument("--profile", dest="profile_name")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    profile_dir = example_dir / "run_profiles"
    if args.list:
        for path in sorted(profile_dir.glob("*.toml")):
            print(path.stem)
        return

    profile_name = args.profile_name if args.profile_name is not None else args.profile
    profile_path = _resolve_profile(example_dir, profile_name)
    if not profile_path.exists():
        raise SystemExit(f"GEPA profile not found: {profile_path}")
    profile = MiniGridRunProfile.model_validate(tomllib.loads(profile_path.read_text()))
    if "GEPA_MINIGRID_ENV_ID" in os.environ:
        profile.task.env_id = os.environ["GEPA_MINIGRID_ENV_ID"]
    if "GEPA_MINIGRID_MAX_STEPS" in os.environ:
        profile.task.max_steps = int(os.environ["GEPA_MINIGRID_MAX_STEPS"])
    if "GEPA_TRAIN_SIZE" in os.environ:
        profile.taskset.train_size = int(os.environ["GEPA_TRAIN_SIZE"])
    if "GEPA_HELDOUT_SIZE" in os.environ:
        profile.taskset.heldout_size = int(os.environ["GEPA_HELDOUT_SIZE"])
    if "GEPA_TRAIN_TASK_ID_START" in os.environ:
        profile.taskset.train_task_id_start = os.environ["GEPA_TRAIN_TASK_ID_START"]
    if "GEPA_HELDOUT_TASK_ID_START" in os.environ:
        profile.taskset.heldout_task_id_start = os.environ["GEPA_HELDOUT_TASK_ID_START"]
    if "GEPA_MAX_TRAIN_ROLLOUTS" in os.environ:
        profile.budgets.train_rollouts = int(os.environ["GEPA_MAX_TRAIN_ROLLOUTS"])
    if "GEPA_MAX_HELDOUT_ROLLOUTS" in os.environ:
        profile.budgets.heldout_rollouts = int(os.environ["GEPA_MAX_HELDOUT_ROLLOUTS"])
    if "GEPA_MAX_GENERATIONS" in os.environ:
        profile.pipeline.max_generations = int(os.environ["GEPA_MAX_GENERATIONS"])
    if "GEPA_PROPOSALS_PER_GENERATION" in os.environ:
        profile.pipeline.proposals_per_generation = int(os.environ["GEPA_PROPOSALS_PER_GENERATION"])
    if "GEPA_MINIBATCH_SIZE" in os.environ:
        profile.pipeline.minibatch_size = int(os.environ["GEPA_MINIBATCH_SIZE"])
    if "GEPA_ROLLOUT_WORKERS" in os.environ:
        profile.pipeline.rollout_workers = int(os.environ["GEPA_ROLLOUT_WORKERS"])
    if "GEPA_MAX_IN_FLIGHT_CANDIDATES" in os.environ:
        profile.pipeline.max_in_flight_candidates = int(os.environ["GEPA_MAX_IN_FLIGHT_CANDIDATES"])
    if "GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS" in os.environ:
        profile.timeouts.rollout_async_seconds = int(
            os.environ["GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS"]
        )
    policy_model = os.environ.get("GEPA_POLICY_MODEL", profile.policy.model)
    proposer_model = os.environ.get("GEPA_PROPOSER_MODEL", profile.proposer.model)
    policy_api_key_env = os.environ.get("GEPA_POLICY_API_KEY_ENV", profile.policy.api_key_env)
    policy_base_url = os.environ.get("GEPA_POLICY_BASE_URL", profile.policy.base_url or "")
    proposer_api_key_env = os.environ.get(
        "GEPA_PROPOSER_API_KEY_ENV",
        profile.proposer.api_key_env,
    )
    if not os.environ.get(policy_api_key_env):
        raise SystemExit(f"{policy_api_key_env} is required for MiniGrid policy rollouts.")
    if not os.environ.get(proposer_api_key_env):
        raise SystemExit(f"{proposer_api_key_env} is required for the Codex proposer.")

    os.environ["MINIGRID_POLICY_MODEL"] = policy_model
    os.environ["MINIGRID_POLICY_API_KEY_ENV"] = policy_api_key_env
    if policy_base_url:
        os.environ["MINIGRID_POLICY_BASE_URL"] = policy_base_url
    os.environ["MINIGRID_MAX_STEPS"] = str(profile.task.max_steps)
    os.environ["MINIGRID_ENV_ID"] = profile.task.env_id

    sys.path.insert(0, str(example_dir))
    import synth_service_app as service

    run_id = f"{profile.base.run_id}_{uuid.uuid4().hex[:8]}"
    run_dir = example_dir / "runs" / run_id
    train_start = _task_id_start(profile.taskset.train_task_id_start)
    heldout_start = _task_id_start(profile.taskset.heldout_task_id_start)
    train_ids = [
        service._row_for_seed(split="train", seed=train_start + offset)["task_id"]
        for offset in range(profile.taskset.train_size)
    ]
    heldout_ids = [
        service._row_for_seed(split="test", seed=heldout_start + offset)["task_id"]
        for offset in range(profile.taskset.heldout_size)
    ]

    print(f"GEPA profile: {profile_name}")
    print(f"GEPA profile_toml: {profile_path}")
    print(f"GEPA run_id: {run_id}")
    print(f"GEPA output_dir: {run_dir}")
    print(f"GEPA taskset: train={len(train_ids)} heldout={len(heldout_ids)}")
    print(f"GEPA task: env_id={profile.task.env_id} max_steps={profile.task.max_steps}")
    print("GEPA objectives: task_success maximize, episode_steps minimize")
    print("GEPA rollout_transport: async")

    container = _build_container(service)
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
                objective_keys=["task_success", "episode_steps"],
                objective_directions={
                    "task_success": "maximize",
                    "episode_steps": "minimize",
                },
                selection_objective="task_success",
                protected_objectives=["task_success"],
                frontier_type="per_objective",
            ),
            policy=None,
            proposer=ProposerConfig(
                model=proposer_model,
                auth_mode=profile.proposer.auth_mode,
                api_key_env=proposer_api_key_env,
                copy_host_auth=profile.proposer.copy_host_auth,
            ),
            budgets=GepaBudgetConfig(
                max_generations=profile.pipeline.max_generations,
                proposals_per_generation=profile.pipeline.proposals_per_generation,
                minibatch_size=profile.pipeline.minibatch_size,
                max_total_rollouts=profile.budgets.train_rollouts
                + profile.budgets.heldout_rollouts,
                max_train_rollouts=profile.budgets.train_rollouts,
                max_heldout_rollouts=profile.budgets.heldout_rollouts,
            ),
            budget=BudgetConfig(max_cost_usd=0.0),
            pipeline=GepaPipeline(
                rollout_transport="async",
                rollout_timeout_seconds=profile.timeouts.rollout_async_seconds,
                candidate_concurrency=profile.pipeline.max_in_flight_candidates,
                rollout_concurrency=profile.pipeline.rollout_workers,
            ),
            cache=CacheConfig(
                mode="readwrite",
                path=run_dir / f"{run_id}_cache.sqlite",
                namespace=run_id,
            ),
        )
        result = OptimizerRun(config).execute()

    print("MiniGrid GEPA SDK run complete")
    print(f"manifest: {result.manifest_path}")
    print(f"cost_usd: {result.cost_usd:.4f}")
    print(f"best_candidate: {result.best_candidate.get('candidate_id', '?')}")


if __name__ == "__main__":
    main()
