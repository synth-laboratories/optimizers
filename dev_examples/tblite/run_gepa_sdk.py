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


class BaseRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str = "tblite_gepa_public"


class PolicyRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-4.1-nano"


class ProposerRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-5.4-nano"
    auth_mode: str = "api_key"
    api_key_env: str = "OPENAI_API_KEY"
    copy_host_auth: bool = False


class TaskRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    test_timeout_seconds: int = 30


class TasksetRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_size: int = 3
    heldout_size: int = 2
    train_task_id_start: int = 0
    heldout_task_id_start: int = 100


class BudgetsRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_rollouts: int = 16
    heldout_rollouts: int = 4


class PipelineRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rollout_workers: int = 10
    max_in_flight_candidates: int = 1
    proposals_per_generation: int = 1
    max_generations: int = 1
    minibatch_size: int | None = None


class TimeoutsRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rollout_async_seconds: int = 60


class TBLiteRunProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base: BaseRunProfile = Field(default_factory=BaseRunProfile)
    policy: PolicyRunProfile = Field(default_factory=PolicyRunProfile)
    proposer: ProposerRunProfile = Field(default_factory=ProposerRunProfile)
    task: TaskRunProfile = Field(default_factory=TaskRunProfile)
    taskset: TasksetRunProfile = Field(default_factory=TasksetRunProfile)
    budgets: BudgetsRunProfile = Field(default_factory=BudgetsRunProfile)
    pipeline: PipelineRunProfile = Field(default_factory=PipelineRunProfile)
    timeouts: TimeoutsRunProfile = Field(default_factory=TimeoutsRunProfile)


class TBLiteEnvOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_model: str | None = None
    proposer_model: str | None = None
    test_timeout_seconds: int | None = None
    train_size: int | None = None
    heldout_size: int | None = None
    train_task_id_start: int | None = None
    heldout_task_id_start: int | None = None
    train_rollouts: int | None = None
    heldout_rollouts: int | None = None
    rollout_workers: int | None = None
    max_in_flight_candidates: int | None = None
    proposals_per_generation: int | None = None
    max_generations: int | None = None
    minibatch_size: int | None = None
    rollout_async_seconds: int | None = None

    @classmethod
    def from_env(cls) -> "TBLiteEnvOverrides":
        payload: dict[str, object] = {}
        if "GEPA_POLICY_MODEL" in os.environ:
            payload["policy_model"] = os.environ["GEPA_POLICY_MODEL"]
        if "GEPA_PROPOSER_MODEL" in os.environ:
            payload["proposer_model"] = os.environ["GEPA_PROPOSER_MODEL"]
        if "GEPA_TBLITE_TEST_TIMEOUT_SECONDS" in os.environ:
            payload["test_timeout_seconds"] = int(os.environ["GEPA_TBLITE_TEST_TIMEOUT_SECONDS"])
        if "GEPA_TRAIN_SIZE" in os.environ:
            payload["train_size"] = int(os.environ["GEPA_TRAIN_SIZE"])
        if "GEPA_HELDOUT_SIZE" in os.environ:
            payload["heldout_size"] = int(os.environ["GEPA_HELDOUT_SIZE"])
        if "GEPA_TRAIN_TASK_ID_START" in os.environ:
            payload["train_task_id_start"] = int(os.environ["GEPA_TRAIN_TASK_ID_START"])
        if "GEPA_HELDOUT_TASK_ID_START" in os.environ:
            payload["heldout_task_id_start"] = int(os.environ["GEPA_HELDOUT_TASK_ID_START"])
        if "GEPA_MAX_TRAIN_ROLLOUTS" in os.environ:
            payload["train_rollouts"] = int(os.environ["GEPA_MAX_TRAIN_ROLLOUTS"])
        if "GEPA_MAX_HELDOUT_ROLLOUTS" in os.environ:
            payload["heldout_rollouts"] = int(os.environ["GEPA_MAX_HELDOUT_ROLLOUTS"])
        if "GEPA_ROLLOUT_WORKERS" in os.environ:
            payload["rollout_workers"] = int(os.environ["GEPA_ROLLOUT_WORKERS"])
        if "GEPA_MAX_IN_FLIGHT_CANDIDATES" in os.environ:
            payload["max_in_flight_candidates"] = int(os.environ["GEPA_MAX_IN_FLIGHT_CANDIDATES"])
        if "GEPA_PROPOSALS_PER_GENERATION" in os.environ:
            payload["proposals_per_generation"] = int(os.environ["GEPA_PROPOSALS_PER_GENERATION"])
        if "GEPA_MAX_GENERATIONS" in os.environ:
            payload["max_generations"] = int(os.environ["GEPA_MAX_GENERATIONS"])
        if "GEPA_MINIBATCH_SIZE" in os.environ:
            payload["minibatch_size"] = int(os.environ["GEPA_MINIBATCH_SIZE"])
        if "GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS" in os.environ:
            payload["rollout_async_seconds"] = int(os.environ["GEPA_ROLLOUT_ASYNC_TIMEOUT_SECONDS"])
        return cls.model_validate(payload)


class TBLiteRunSettings(BaseModel):
    policy_model: str
    proposer_model: str
    test_timeout_seconds: int
    train_size: int
    heldout_size: int
    train_task_id_start: int
    heldout_task_id_start: int
    train_rollouts: int
    heldout_rollouts: int
    rollout_workers: int
    max_in_flight_candidates: int
    proposals_per_generation: int
    max_generations: int
    minibatch_size: int
    rollout_async_seconds: int
    proposer_auth_mode: str
    proposer_api_key_env: str
    proposer_copy_host_auth: bool

    @classmethod
    def from_sources(
        cls, profile: TBLiteRunProfile, overrides: TBLiteEnvOverrides
    ) -> "TBLiteRunSettings":
        train_size = (
            overrides.train_size if overrides.train_size is not None else profile.taskset.train_size
        )
        minibatch_size = profile.pipeline.minibatch_size
        if overrides.minibatch_size is not None:
            minibatch_size = overrides.minibatch_size
        if minibatch_size is None:
            minibatch_size = train_size
        return cls(
            policy_model=(
                overrides.policy_model
                if overrides.policy_model is not None
                else profile.policy.model
            ),
            proposer_model=(
                overrides.proposer_model
                if overrides.proposer_model is not None
                else profile.proposer.model
            ),
            test_timeout_seconds=(
                overrides.test_timeout_seconds
                if overrides.test_timeout_seconds is not None
                else profile.task.test_timeout_seconds
            ),
            train_size=train_size,
            heldout_size=(
                overrides.heldout_size
                if overrides.heldout_size is not None
                else profile.taskset.heldout_size
            ),
            train_task_id_start=(
                overrides.train_task_id_start
                if overrides.train_task_id_start is not None
                else profile.taskset.train_task_id_start
            ),
            heldout_task_id_start=(
                overrides.heldout_task_id_start
                if overrides.heldout_task_id_start is not None
                else profile.taskset.heldout_task_id_start
            ),
            train_rollouts=(
                overrides.train_rollouts
                if overrides.train_rollouts is not None
                else profile.budgets.train_rollouts
            ),
            heldout_rollouts=(
                overrides.heldout_rollouts
                if overrides.heldout_rollouts is not None
                else profile.budgets.heldout_rollouts
            ),
            rollout_workers=(
                overrides.rollout_workers
                if overrides.rollout_workers is not None
                else profile.pipeline.rollout_workers
            ),
            max_in_flight_candidates=(
                overrides.max_in_flight_candidates
                if overrides.max_in_flight_candidates is not None
                else profile.pipeline.max_in_flight_candidates
            ),
            proposals_per_generation=(
                overrides.proposals_per_generation
                if overrides.proposals_per_generation is not None
                else profile.pipeline.proposals_per_generation
            ),
            max_generations=(
                overrides.max_generations
                if overrides.max_generations is not None
                else profile.pipeline.max_generations
            ),
            minibatch_size=minibatch_size,
            rollout_async_seconds=(
                overrides.rollout_async_seconds
                if overrides.rollout_async_seconds is not None
                else profile.timeouts.rollout_async_seconds
            ),
            proposer_auth_mode=profile.proposer.auth_mode,
            proposer_api_key_env=profile.proposer.api_key_env,
            proposer_copy_host_auth=profile.proposer.copy_host_auth,
        )


class GepaResultCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str = "?"


def _seed_range(start: int, count: int) -> list[int]:
    return [start + idx for idx in range(count)]


def _load_profile(path: Path) -> TBLiteRunProfile:
    return TBLiteRunProfile.model_validate(tomllib.loads(path.read_text()))


def _resolve_profile(example_dir: Path, raw: str) -> Path:
    if raw.endswith(".toml"):
        return Path(raw).resolve()
    if "/" in raw:
        return Path(raw).resolve()
    return example_dir / "run_profiles" / f"{raw}.toml"


def _build_container(service: Any) -> Container:
    container = Container(
        "tblite-gepa-sdk",
        runtime_id="tblite_gepa_live",
        description="Terminal-Bench-Lite Python function implementation container.",
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
    async def taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        request = service.TasksetTasksRequest.model_validate(payload)
        return await service.taskset_tasks(request)

    @container.rollout
    def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        return service.rollout(service.RolloutRequest.model_validate(payload))

    return container


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run TBLite GEPA through the Python SDK.")
    parser.add_argument("profile", nargs="?", default="default")
    parser.add_argument("--profile", dest="profile_name")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    profile_dir = example_dir / "run_profiles"
    if args.list:
        for path in sorted(profile_dir.glob("*.toml")):
            print(path.stem)
        return

    profile_name = args.profile
    if args.profile_name is not None:
        profile_name = args.profile_name
    profile_path = _resolve_profile(example_dir, profile_name)
    if not profile_path.exists():
        raise SystemExit(f"GEPA profile not found: {profile_path}")
    profile = _load_profile(profile_path)
    settings = TBLiteRunSettings.from_sources(profile, TBLiteEnvOverrides.from_env())

    os.environ["TBLITE_POLICY_MODEL"] = settings.policy_model
    os.environ["TBLITE_TEST_TIMEOUT_SECONDS"] = str(settings.test_timeout_seconds)
    sys.path.insert(0, str(example_dir))
    import synth_service_app as service

    run_id = f"{profile.base.run_id}_{uuid.uuid4().hex[:8]}"
    run_dir = example_dir / "runs" / run_id

    container = _build_container(service)
    print(f"GEPA profile: {profile_name}")
    print(f"GEPA profile_toml: {profile_path}")
    print(f"GEPA run_id: {run_id}")
    print(f"GEPA output_dir: {run_dir}")
    print("GEPA rollout_transport: async")

    train_ids = [
        service._row_for_seed(split="train", seed=seed)["task_id"]
        for seed in _seed_range(settings.train_task_id_start, settings.train_size)
    ]
    heldout_ids = [
        service._row_for_seed(split="test", seed=seed)["task_id"]
        for seed in _seed_range(settings.heldout_task_id_start, settings.heldout_size)
    ]

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
                objective_keys=["correctness", "completion_time_seconds"],
                objective_directions={
                    "correctness": "maximize",
                    "completion_time_seconds": "minimize",
                },
                selection_objective="correctness",
                protected_objectives=["correctness"],
                frontier_type="per_objective",
            ),
            policy=None,
            proposer=ProposerConfig(
                model=settings.proposer_model,
                auth_mode=settings.proposer_auth_mode,
                api_key_env=settings.proposer_api_key_env,
                copy_host_auth=settings.proposer_copy_host_auth,
            ),
            budgets=GepaBudgetConfig(
                max_generations=settings.max_generations,
                proposals_per_generation=settings.proposals_per_generation,
                minibatch_size=settings.minibatch_size,
                max_total_rollouts=settings.train_rollouts + settings.heldout_rollouts,
                max_train_rollouts=settings.train_rollouts,
                max_heldout_rollouts=settings.heldout_rollouts,
            ),
            budget=BudgetConfig(max_cost_usd=0.0),
            pipeline=GepaPipeline(
                rollout_transport="async",
                rollout_timeout_seconds=settings.rollout_async_seconds,
                candidate_concurrency=settings.max_in_flight_candidates,
                rollout_concurrency=settings.rollout_workers,
            ),
            cache=CacheConfig(
                mode="readwrite",
                path=run_dir / f"{run_id}_cache.sqlite",
                namespace=run_id,
            ),
        )
        result = OptimizerRun(config).execute()

    best = GepaResultCandidate.model_validate(result.best_candidate)
    print("TBLite GEPA SDK run complete")
    print(f"manifest: {result.manifest_path}")
    print(f"cost_usd: {result.cost_usd:.4f}")
    print(f"best_candidate: {best.candidate_id}")


if __name__ == "__main__":
    main()
