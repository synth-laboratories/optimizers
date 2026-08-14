from __future__ import annotations

import http.client
import json
import os
import tomllib
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field
from synth_containers import ContainerConnection, PromptProgram

from ._synth_optimizers import GepaRun as _NativeGepaRun
from ._synth_optimizers import GepaRunResult
from ._synth_optimizers import default_proposer_best_practices as _default_proposer_best_practices
from .hosted import OptimizerAlgorithmSlug


class PolicyType(StrEnum):
    DAG = "dag"
    REACT = "react"
    CODEX = "codex"


class RolloutTransport(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class GepaPipelineMode(StrEnum):
    SYNC_SERIAL = "sync_serial"
    ASYNC_PIPELINED = "async_pipelined"
    FLASH_EVOLVE = "flash_evolve"


class GepaStalenessPolicy(StrEnum):
    FULL = "full"
    GUARDED = "guarded"
    REFLECTIVE = "reflective"


class ContainerTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    auth_bearer_env: str | None = None
    pool: "ContainerPoolTomlSection | None" = None
    command: list[str] = Field(default_factory=list)
    cwd: str | Path | None = None
    startup_timeout_seconds: int | None = None

    def to_connection(self) -> ContainerConnection:
        return ContainerConnection(url=self.resolved_url())

    def resolved_url(self) -> str:
        if self.pool is None:
            url = (self.url or "").strip()
            if not url:
                raise ValueError("container.url or container.pool.pool_id is required")
            return url
        return self.pool.resolved_url()

    def resolved_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.auth_bearer_env:
            headers.setdefault("authorization", _bearer_header_from_env(self.auth_bearer_env))
        elif self.pool is not None and not _has_authorization_header(headers):
            headers["authorization"] = _bearer_header_from_env(self.pool.api_key_env)
        return headers

    def resolved_auth_bearer_env(self) -> str | None:
        if self.auth_bearer_env:
            return self.auth_bearer_env
        if self.pool is not None and not _has_authorization_header(self.headers):
            return self.pool.api_key_env
        return None


class ContainerPoolTomlSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_id: str = Field(min_length=1)
    task_id: str | None = None
    backend_base_url: str | None = None
    api_key_env: str = "SYNTH_API_KEY"

    def resolved_url(self) -> str:
        pool_id = _path_segment(self.pool_id, "container.pool.pool_id")
        backend_base = _normalize_backend_base_url(
            self.backend_base_url or _backend_base_url_from_env() or "https://api.usesynth.ai"
        )
        if self.task_id:
            task_id = _path_segment(self.task_id, "container.pool.task_id")
            return f"{backend_base}/v1/pools/{pool_id}/tasks/{task_id}/container"
        return f"{backend_base}/v1/pools/{pool_id}/container"


class RunSettingsTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str = "gepa_sdk_run"
    output_dir: str | Path = "runs"
    seed: int = 0

    def to_domain(self) -> "RunSettings":
        return RunSettings(
            run_id=self.run_id,
            output_dir=self.output_dir,
            seed=self.seed,
        )


class UsageRegistrationTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True

    def to_domain(self) -> "UsageRegistrationConfig":
        return UsageRegistrationConfig(enabled=self.enabled)


class TasksetTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_split: str = "train"
    heldout_split: str = "test"
    train_ids: list[str] = Field(default_factory=lambda: ["train:0"])
    heldout_ids: list[str] = Field(default_factory=lambda: ["heldout:0"])
    filters: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> "TasksetSelection":
        return TasksetSelection(
            train_split=self.train_split,
            heldout_split=self.heldout_split,
            train_ids=list(self.train_ids),
            heldout_ids=list(self.heldout_ids),
            filters=dict(self.filters),
        )


class ProposerPromptTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    best_practices: str | None = None
    best_practices_path: str | Path | None = None

    def to_domain(self, base_dir: Path) -> "ProposerPromptConfig | None":
        if self.best_practices is None and self.best_practices_path is None:
            return None
        best_practices_path = self.best_practices_path
        if best_practices_path is not None:
            path = Path(best_practices_path)
            best_practices_path = path if path.is_absolute() else base_dir / path
        return ProposerPromptConfig(
            best_practices=self.best_practices,
            best_practices_path=best_practices_path,
        )


class ProposerDockerTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image: str | None = None
    workspace_mount_path: str = "/workspace"
    network: str = "bridge"
    extra_env: dict[str, str] = Field(default_factory=dict)

    def to_domain(self) -> "ProposerDockerConfig | None":
        if self.image is None:
            return None
        return ProposerDockerConfig(
            image=self.image,
            workspace_mount_path=self.workspace_mount_path,
            network=self.network,
            extra_env=dict(self.extra_env),
        )


class NanoCodexTomlSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: str = "live"
    max_turns_per_session: int = 16
    record_dir: str | Path | None = None
    replay_dir: str | Path | None = None
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["search", "read", "apply_patch", "exec"]
    )

    def to_domain(self, base_dir: Path) -> "NanoCodexConfig":
        def resolve(path_value: str | Path | None) -> Path | None:
            if path_value is None:
                return None
            path = Path(path_value)
            return path if path.is_absolute() else base_dir / path

        return NanoCodexConfig(
            enabled=self.enabled,
            mode=self.mode,
            max_turns_per_session=self.max_turns_per_session,
            record_dir=resolve(self.record_dir),
            replay_dir=resolve(self.replay_dir),
            allowed_tools=list(self.allowed_tools),
        )


class ProposerTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    backend: str = "codex_app_server"
    runtime_substrate: str = "local"
    execution_mode: str = "local_process"
    provider: str = "openai"
    api_family: str = "chat_completions"
    base_url: str | None = None
    model: str | None = "gpt-5.4-mini"
    reasoning_effort: str | None = "medium"
    service_tier: str | None = None
    auth_mode: str = "api_key"
    api_key_env: str | None = "OPENAI_API_KEY"
    copy_host_auth: bool = False
    codex_home: str | Path | None = None
    timeout_seconds: int = 900
    sandbox_mode: str | None = "workspace-write"
    approval_policy: str | None = "never"
    command: list[str] = Field(default_factory=list)
    prompt: ProposerPromptTomlSection = Field(default_factory=ProposerPromptTomlSection)
    docker: ProposerDockerTomlSection | None = None
    nano_codex: NanoCodexTomlSection = Field(default_factory=NanoCodexTomlSection)

    def to_domain(self, base_dir: Path) -> "ProposerConfig":
        codex_home = None
        if self.codex_home is not None:
            path = Path(self.codex_home)
            codex_home = path if path.is_absolute() else base_dir / path
        # ChatGPT-subscription auth forbids api_key_env. Since this field defaults
        # to "OPENAI_API_KEY" and TOML cannot express null to override it, null it
        # out explicitly for chatgpt auth (mirrors the service.rs chatgpt path).
        api_key_env = self.api_key_env
        if str(self.auth_mode).strip().lower() == "chatgpt":
            api_key_env = None
        return ProposerConfig(
            backend=self.backend,
            runtime_substrate=self.runtime_substrate,
            execution_mode=self.execution_mode,
            provider=self.provider,
            api_family=self.api_family,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            service_tier=self.service_tier,
            auth_mode=self.auth_mode,
            api_key_env=api_key_env,
            copy_host_auth=self.copy_host_auth,
            codex_home=codex_home,
            timeout_seconds=self.timeout_seconds,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
            command=list(self.command),
            prompt=self.prompt.to_domain(base_dir),
            docker=self.docker.to_domain() if self.docker is not None else None,
            nano_codex=self.nano_codex.to_domain(base_dir),
        )


class PolicyTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4.1-nano"
    api_key_env: str | None = "OPENAI_API_KEY"
    policy_type: PolicyType | str = PolicyType.DAG
    api_family: str = "chat_completions"
    base_url: str | None = None
    inference_url: str | None = None
    max_tokens: int | None = None
    disable_reasoning: str = "auto"
    tool_call_style: str = "none"
    proxy_mode: str = "proxy_only"
    credential_mode: str = "byok"
    config: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> "PolicyConfig | None":
        if not self.enabled:
            return None
        return PolicyConfig(
            provider=self.provider,
            model=self.model,
            api_key_env=self.api_key_env,
            policy_type=self.policy_type,
            api_family=self.api_family,
            base_url=self.base_url,
            inference_url=self.inference_url,
            max_tokens=self.max_tokens,
            disable_reasoning=self.disable_reasoning,
            tool_call_style=self.tool_call_style,
            proxy_mode=self.proxy_mode,
            credential_mode=self.credential_mode,
            config=dict(self.config),
        )


class GepaPipelineWorkersTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    propose: int = 1
    rollout: int = 8
    evaluate: int = 1


class GepaSpeculativeCompletionTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    alpha: float = 0.25


class GepaAdaptiveStageWorkersTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    min: int = 1
    max: int = 128
    backlog_threshold: int = 2
    stale_gap_threshold: int = 2


class GepaPipelineTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: GepaPipelineMode | str = GepaPipelineMode.SYNC_SERIAL
    staleness_policy: GepaStalenessPolicy | str = GepaStalenessPolicy.FULL
    delta_max: int = 2
    max_in_flight_candidates: int = 1
    workers: GepaPipelineWorkersTomlSection = Field(default_factory=GepaPipelineWorkersTomlSection)
    speculative_completion: GepaSpeculativeCompletionTomlSection = Field(
        default_factory=GepaSpeculativeCompletionTomlSection
    )
    adaptive_stage_workers: GepaAdaptiveStageWorkersTomlSection = Field(
        default_factory=GepaAdaptiveStageWorkersTomlSection
    )


class ObjectiveAcceptanceTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    protected_objectives: list[str] = Field(default_factory=list)
    min_objective_delta: float | None = None
    objective_regression_tolerance: float | None = None


class GepaTaskPoolsTomlSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pareto: list[str] = Field(default_factory=list)
    minibatch: list[str] = Field(default_factory=list)
    reflection: list[str] = Field(default_factory=list)
    heldout: list[str] = Field(default_factory=list)

    def to_domain(self) -> "GepaTaskPools":
        return GepaTaskPools(
            pareto=list(self.pareto),
            minibatch=list(self.minibatch),
            reflection=list(self.reflection),
            heldout=list(self.heldout),
        )


class GepaTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_cost_usd: float = 0.0
    max_time_seconds: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    proposer_estimated_cost_usd: float | None = None
    proposer_estimated_prompt_tokens: int | None = None
    proposer_estimated_completion_tokens: int | None = None
    proposer_estimated_total_tokens: int | None = None
    rollout_estimated_cost_usd: float | None = None
    rollout_estimated_prompt_tokens: int | None = None
    rollout_estimated_completion_tokens: int | None = None
    rollout_estimated_total_tokens: int | None = None
    rollout_estimated_wall_seconds: int | None = None
    max_generations: int = 1
    proposals_per_generation: int = 1
    minibatch_size: int = 1
    max_total_rollouts: int = 16
    max_train_rollouts: int | None = None
    max_heldout_rollouts: int | None = None
    minibatch_accept_margin: float = 0.0
    rollout_failure_rate_tolerance: float = 0.25
    rollout_submission_mode: RolloutTransport | str = RolloutTransport.ASYNC
    rollout_async_timeout_seconds: int = 600
    rollout_chunk_size: int | None = None
    pipeline: GepaPipelineTomlSection = Field(default_factory=GepaPipelineTomlSection)
    objective_keys: list[str] = Field(default_factory=list)
    objective_directions: dict[str, str] = Field(default_factory=dict)
    selection_objective: str | None = None
    frontier_type: str = "per_example"
    acceptance_criterion: str = "primary_improvement"
    objective_acceptance: ObjectiveAcceptanceTomlSection = Field(
        default_factory=ObjectiveAcceptanceTomlSection
    )
    task_pools: GepaTaskPoolsTomlSection = Field(default_factory=GepaTaskPoolsTomlSection)

    def budget_config(self) -> "BudgetConfig":
        return BudgetConfig(
            max_cost_usd=self.max_cost_usd,
            max_time_seconds=self.max_time_seconds,
            max_prompt_tokens=self.max_prompt_tokens,
            max_completion_tokens=self.max_completion_tokens,
            max_total_tokens=self.max_total_tokens,
        )

    def gepa_budget_config(self) -> "GepaBudgetConfig":
        return GepaBudgetConfig(
            max_generations=self.max_generations,
            proposals_per_generation=self.proposals_per_generation,
            minibatch_size=self.minibatch_size,
            max_total_rollouts=self.max_total_rollouts,
            max_train_rollouts=self.max_train_rollouts,
            max_heldout_rollouts=self.max_heldout_rollouts,
            minibatch_accept_margin=self.minibatch_accept_margin,
            rollout_failure_rate_tolerance=self.rollout_failure_rate_tolerance,
            rollout_chunk_size=self.rollout_chunk_size,
            proposer_estimated_cost_usd=self.proposer_estimated_cost_usd,
            proposer_estimated_prompt_tokens=self.proposer_estimated_prompt_tokens,
            proposer_estimated_completion_tokens=self.proposer_estimated_completion_tokens,
            proposer_estimated_total_tokens=self.proposer_estimated_total_tokens,
            rollout_estimated_cost_usd=self.rollout_estimated_cost_usd,
            rollout_estimated_prompt_tokens=self.rollout_estimated_prompt_tokens,
            rollout_estimated_completion_tokens=self.rollout_estimated_completion_tokens,
            rollout_estimated_total_tokens=self.rollout_estimated_total_tokens,
            rollout_estimated_wall_seconds=self.rollout_estimated_wall_seconds,
        )

    def pipeline_config(self) -> "GepaPipeline":
        return GepaPipeline(
            mode=self.pipeline.mode,
            staleness_policy=self.pipeline.staleness_policy,
            staleness_delta_max=self.pipeline.delta_max,
            rollout_transport=self.rollout_submission_mode,
            rollout_timeout_seconds=self.rollout_async_timeout_seconds,
            candidate_concurrency=self.pipeline.max_in_flight_candidates,
            proposer_concurrency=self.pipeline.workers.propose,
            rollout_concurrency=self.pipeline.workers.rollout,
            evaluator_concurrency=self.pipeline.workers.evaluate,
            speculative_alpha=(
                self.pipeline.speculative_completion.alpha
                if self.pipeline.speculative_completion.enabled
                else None
            ),
            adaptive_stage_workers=self.pipeline.adaptive_stage_workers.enabled,
            adaptive_stage_workers_max=self.pipeline.adaptive_stage_workers.max,
            adaptive_stage_workers_backlog_threshold=(
                self.pipeline.adaptive_stage_workers.backlog_threshold
            ),
            adaptive_stage_workers_stale_gap_threshold=(
                self.pipeline.adaptive_stage_workers.stale_gap_threshold
            ),
        )

    def objective_config(self) -> "ObjectiveConfig":
        return ObjectiveConfig(
            objective_keys=list(self.objective_keys),
            objective_directions=dict(self.objective_directions),
            selection_objective=self.selection_objective,
            protected_objectives=list(self.objective_acceptance.protected_objectives),
            frontier_type=self.frontier_type,
            acceptance_criterion=self.acceptance_criterion,
            min_objective_delta=self.objective_acceptance.min_objective_delta,
            objective_regression_tolerance=(
                self.objective_acceptance.objective_regression_tolerance
            ),
        )


class CacheTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "readwrite"
    path: str | Path | None = None
    namespace: str | None = None

    def to_domain(self) -> "CacheConfig":
        return CacheConfig(
            mode=self.mode,
            path=self.path,
            namespace=self.namespace,
        )


class CandidateTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_modules: list[str] = Field(default_factory=list)


class JesterkyWorkflowTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    spec: str = "examples/gepa_trace_annotate.json"
    command: str = "jesterky"
    actor: str = "codex"
    model: str | None = None
    concurrency: int = 4
    timeout_seconds: int = 600
    fail_closed: bool = True

    def to_domain(self) -> "JesterkyWorkflowConfig":
        return JesterkyWorkflowConfig(
            enabled=bool(self.enabled),
            spec=str(self.spec or "examples/gepa_trace_annotate.json"),
            command=str(self.command or "jesterky"),
            actor=str(self.actor or "codex"),
            model=self.model,
            concurrency=int(self.concurrency),
            timeout_seconds=int(self.timeout_seconds),
            fail_closed=bool(self.fail_closed),
        )


class GepaTomlDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    container: ContainerTomlSection
    run: RunSettingsTomlSection = Field(default_factory=RunSettingsTomlSection)
    taskset: TasksetTomlSection = Field(default_factory=TasksetTomlSection)
    proposer: ProposerTomlSection = Field(default_factory=ProposerTomlSection)
    policy: PolicyTomlSection = Field(default_factory=PolicyTomlSection)
    gepa: GepaTomlSection = Field(default_factory=GepaTomlSection)
    jesterky_workflow: JesterkyWorkflowTomlSection = Field(
        default_factory=JesterkyWorkflowTomlSection
    )
    cache: CacheTomlSection = Field(default_factory=CacheTomlSection)
    usage_registration: UsageRegistrationTomlSection = Field(
        default_factory=UsageRegistrationTomlSection
    )
    candidate: CandidateTomlSection = Field(default_factory=CandidateTomlSection)
    seed_candidate: dict[str, str] = Field(default_factory=dict)

    def to_config(self, source_path: Path) -> "GepaConfig":
        return GepaConfig(
            container=self.container.to_connection(),
            container_headers=dict(self.container.headers),
            container_auth_bearer_env=self.container.resolved_auth_bearer_env(),
            container_command=list(self.container.command),
            container_cwd=self.container.cwd,
            container_startup_timeout_seconds=self.container.startup_timeout_seconds,
            taskset=self.taskset.to_domain(),
            run=self.run.to_domain(),
            objectives=self.gepa.objective_config(),
            policy=self.policy.to_domain(),
            proposer=self.proposer.to_domain(source_path.parent),
            task_pools=self.gepa.task_pools.to_domain(),
            budgets=self.gepa.gepa_budget_config(),
            pipeline=self.gepa.pipeline_config(),
            budget=self.gepa.budget_config(),
            jesterky_workflow=self.jesterky_workflow.to_domain(),
            cache=self.cache.to_domain(),
            usage_registration=self.usage_registration.to_domain(),
            target_modules=list(self.candidate.target_modules),
            seed_candidate=dict(self.seed_candidate),
            _source_path=source_path,
        )


class ContainerCapabilityMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    policy_ready: bool


class ContainerCapabilitiesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: ContainerCapabilityMetadataPayload


class ContainerMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capabilities: ContainerCapabilitiesPayload


class ProgramTargetModulePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    module_id: str = Field(min_length=1)
    candidate_field: str = ""
    objective: str = ""


class ProgramPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed_candidate: dict[str, str] = Field(min_length=1)
    target_modules: list[ProgramTargetModulePayload] = Field(min_length=1)


@dataclass(slots=True)
class RunSettings:
    run_id: str = "gepa_sdk_run"
    output_dir: str | Path = "runs"
    seed: int = 0

    def to_toml(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "seed": int(self.seed),
        }


@dataclass(slots=True)
class UsageRegistrationConfig:
    enabled: bool = True

    def to_toml(self) -> dict[str, Any]:
        return {"enabled": bool(self.enabled)}


@dataclass(slots=True)
class TasksetSelection:
    train_split: str = "train"
    heldout_split: str = "test"
    train_ids: list[str] = field(default_factory=lambda: ["train:0"])
    heldout_ids: list[str] = field(default_factory=lambda: ["heldout:0"])
    filters: dict[str, Any] = field(default_factory=dict)

    def to_toml(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "train_split": self.train_split,
            "heldout_split": self.heldout_split,
            "train_ids": list(self.train_ids),
            "heldout_ids": list(self.heldout_ids),
        }
        if self.filters:
            payload["filters"] = dict(self.filters)
        return payload


@dataclass(slots=True)
class GepaTaskPools:
    pareto: list[str]
    minibatch: list[str]
    reflection: list[str]
    heldout: list[str]

    def validate(self) -> None:
        for name, values in {
            "pareto": self.pareto,
            "minibatch": self.minibatch,
            "reflection": self.reflection,
            "heldout": self.heldout,
        }.items():
            if not values:
                raise ValueError(f"GepaTaskPools.{name} must not be empty")
            if any(not value.strip() for value in values):
                raise ValueError(f"GepaTaskPools.{name} entries must not be empty")
        minibatch = set(self.minibatch)
        reflection = set(self.reflection)
        missing = sorted(minibatch - reflection)
        if missing:
            raise ValueError(
                "GepaTaskPools.minibatch must be a subset of GepaTaskPools.reflection; "
                f"missing from reflection: {missing}"
            )
        heldout = set(self.heldout)
        search_ids = set(self.pareto) | minibatch | reflection
        overlaps = sorted(heldout & search_ids)
        if overlaps:
            raise ValueError(
                "GepaTaskPools.heldout must be disjoint from pareto/minibatch/reflection; "
                f"overlaps: {overlaps}"
            )

    def validate_against_taskset(self, train_ids: list[str], heldout_ids: list[str]) -> None:
        """Pools are split-local: search pools draw from train, heldout from heldout."""
        unknown_search = sorted(
            (set(self.pareto) | set(self.minibatch) | set(self.reflection)) - set(train_ids)
        )
        if unknown_search:
            raise ValueError(
                "GepaTaskPools pareto/minibatch/reflection ids must come from "
                f"taskset.train_ids; unknown: {unknown_search}"
            )
        unknown_heldout = sorted(set(self.heldout) - set(heldout_ids))
        if unknown_heldout:
            raise ValueError(
                "GepaTaskPools.heldout ids must come from taskset.heldout_ids; "
                f"unknown: {unknown_heldout}"
            )

    def to_toml(self) -> dict[str, Any]:
        self.validate()
        return {
            "pareto": list(self.pareto),
            "minibatch": list(self.minibatch),
            "reflection": list(self.reflection),
            "heldout": list(self.heldout),
        }


@dataclass(slots=True)
class ProposerPromptConfig:
    best_practices: str | None = None
    best_practices_path: str | Path | None = None

    @classmethod
    def from_defaults(cls) -> "ProposerPromptConfig":
        return cls(best_practices=GepaDefaults.proposer_best_practices())

    @classmethod
    def from_path(cls, path: str | Path) -> "ProposerPromptConfig":
        return cls(best_practices=Path(path).read_text())

    def to_toml(self) -> dict[str, Any]:
        if self.best_practices is not None and self.best_practices_path is not None:
            raise ValueError(
                "ProposerPromptConfig accepts at most one of best_practices or best_practices_path"
            )
        if self.best_practices is not None and not self.best_practices.strip():
            raise ValueError("ProposerPromptConfig.best_practices must be non-empty when set")
        return _drop_none(
            {
                "best_practices": self.best_practices,
                "best_practices_path": (
                    str(self.best_practices_path) if self.best_practices_path is not None else None
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ProposerDefaults:
    best_practices_md: str


@dataclass(frozen=True, slots=True)
class GepaDefaults:
    proposer: ProposerDefaults

    @staticmethod
    def current() -> "GepaDefaults":
        return GepaDefaults(
            proposer=ProposerDefaults(best_practices_md=GepaDefaults.proposer_best_practices())
        )

    @staticmethod
    def proposer_best_practices() -> str:
        return _default_proposer_best_practices()

    @staticmethod
    def write_proposer_best_practices(path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(GepaDefaults.proposer_best_practices())
        return output_path

    @staticmethod
    def proposer_config() -> "ProposerConfig":
        return ProposerConfig()

    @staticmethod
    def config_template(*, container_url: str) -> "GepaConfig":
        return GepaConfig(
            container=ContainerConnection(url=container_url),
            taskset=TasksetSelection(),
            task_pools=GepaTaskPools(
                pareto=["train:0"],
                minibatch=["train:0"],
                reflection=["train:0"],
                heldout=["heldout:0"],
            ),
            policy=None,
        )


@dataclass(slots=True)
class ProposerDockerConfig:
    image: str
    workspace_mount_path: str = "/workspace"
    network: str = "bridge"
    extra_env: dict[str, str] = field(default_factory=dict)

    def to_toml(self) -> dict[str, Any]:
        return _drop_none(
            {
                "image": self.image,
                "workspace_mount_path": self.workspace_mount_path,
                "network": self.network,
                "extra_env": dict(self.extra_env),
            }
        )


@dataclass(slots=True)
class NanoCodexConfig:
    enabled: bool = False
    mode: str = "live"
    max_turns_per_session: int = 16
    record_dir: str | Path | None = None
    replay_dir: str | Path | None = None
    allowed_tools: list[str] = field(
        default_factory=lambda: ["search", "read", "apply_patch", "exec"]
    )

    def validate(self) -> None:
        mode = self.mode.strip().lower().replace("-", "_")
        if mode not in {"live", "replay"}:
            raise ValueError("NanoCodexConfig.mode must be live or replay")
        if self.max_turns_per_session <= 0:
            raise ValueError("NanoCodexConfig.max_turns_per_session must be positive")
        if mode == "replay" and self.replay_dir is None:
            raise ValueError("NanoCodexConfig replay mode requires replay_dir")
        if (
            mode == "replay"
            and self.record_dir is not None
            and Path(self.record_dir).resolve() == Path(self.replay_dir).resolve()
        ):
            raise ValueError(
                "NanoCodexConfig replay record_dir must differ from replay_dir"
            )
        permitted = {"search", "read", "apply_patch", "exec"}
        normalized = [tool.strip().lower().replace("-", "_") for tool in self.allowed_tools]
        if not normalized or any(tool not in permitted for tool in normalized):
            raise ValueError(
                "NanoCodexConfig.allowed_tools must contain only search, read, apply_patch, exec"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("NanoCodexConfig.allowed_tools must not contain duplicates")

    def to_toml(self) -> dict[str, Any]:
        self.validate()
        return _drop_none(
            {
                "enabled": bool(self.enabled),
                "mode": self.mode,
                "max_turns_per_session": int(self.max_turns_per_session),
                "record_dir": str(self.record_dir) if self.record_dir is not None else None,
                "replay_dir": str(self.replay_dir) if self.replay_dir is not None else None,
                "allowed_tools": list(self.allowed_tools),
            }
        )


@dataclass(slots=True)
class ProposerConfig:
    backend: str = "codex_app_server"
    runtime_substrate: str = "local"
    execution_mode: str = "local_process"
    provider: str = "openai"
    api_family: str = "chat_completions"
    base_url: str | None = None
    model: str | None = "gpt-5.4-mini"
    reasoning_effort: str | None = "medium"
    service_tier: str | None = None
    auth_mode: str = "api_key"
    api_key_env: str | None = "OPENAI_API_KEY"
    copy_host_auth: bool = False
    codex_home: str | Path | None = None
    timeout_seconds: int = 900
    sandbox_mode: str | None = "workspace-write"
    approval_policy: str | None = "never"
    command: list[str] = field(default_factory=list)
    prompt: ProposerPromptConfig | None = None
    docker: ProposerDockerConfig | None = None
    nano_codex: NanoCodexConfig = field(default_factory=NanoCodexConfig)

    @classmethod
    def local(cls, **kwargs: Any) -> "ProposerConfig":
        return cls(runtime_substrate="local", **kwargs)

    @classmethod
    def docker_substrate(cls, *, image: str, **kwargs: Any) -> "ProposerConfig":
        return cls(
            runtime_substrate="docker",
            docker=ProposerDockerConfig(image=image),
            **kwargs,
        )

    def to_toml(self) -> dict[str, Any]:
        auth_mode = str(self.auth_mode).strip().lower().replace("-", "_")
        if self.nano_codex.enabled:
            if auth_mode not in {"chatgpt", "host"}:
                raise ValueError(
                    "nano_codex requires proposer auth_mode='chatgpt' or 'host'"
                )
            if not self.copy_host_auth or self.api_key_env is not None:
                raise ValueError(
                    "nano_codex requires copy_host_auth=True and api_key_env=None"
                )
            if str(self.runtime_substrate).strip().lower() != "local":
                raise ValueError("nano_codex currently requires runtime_substrate='local'")
        api_key_env = self.api_key_env
        if auth_mode in {"chatgpt", "host"}:
            api_key_env = None
        payload = _drop_none(
            {
                "backend": self.backend,
                "runtime_substrate": self.runtime_substrate,
                "execution_mode": self.execution_mode,
                "provider": self.provider,
                "api_family": self.api_family,
                "base_url": self.base_url,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "service_tier": self.service_tier,
                "auth_mode": self.auth_mode,
                "api_key_env": api_key_env,
                "copy_host_auth": bool(self.copy_host_auth),
                "codex_home": str(self.codex_home) if self.codex_home is not None else None,
                "timeout_seconds": int(self.timeout_seconds),
                "sandbox_mode": self.sandbox_mode,
                "approval_policy": self.approval_policy,
                "command": list(self.command),
            }
        )
        if self.prompt is not None:
            prompt = self.prompt.to_toml()
            if prompt:
                payload["prompt"] = prompt
        if self.docker is not None:
            payload["docker"] = self.docker.to_toml()
        payload["nano_codex"] = self.nano_codex.to_toml()
        return payload


@dataclass(slots=True)
class PolicyConfig:
    provider: str = "openai"
    model: str = "gpt-4.1-nano"
    api_key_env: str | None = "OPENAI_API_KEY"
    policy_type: PolicyType | str = PolicyType.DAG
    api_family: str = "chat_completions"
    base_url: str | None = None
    inference_url: str | None = None
    max_tokens: int | None = None
    disable_reasoning: str = "auto"
    tool_call_style: str = "none"
    proxy_mode: str = "proxy_only"
    credential_mode: str = "byok"
    config: dict[str, Any] = field(default_factory=dict)

    def to_toml(self) -> dict[str, Any]:
        return _drop_none(
            {
                "enabled": True,
                "provider": self.provider,
                "model": self.model,
                "api_key_env": self.api_key_env,
                "policy_type": str(self.policy_type),
                "api_family": self.api_family,
                "base_url": self.base_url,
                "inference_url": self.inference_url,
                "max_tokens": self.max_tokens,
                "disable_reasoning": self.disable_reasoning,
                "tool_call_style": self.tool_call_style,
                "proxy_mode": self.proxy_mode,
                "credential_mode": self.credential_mode,
                "config": dict(self.config),
            }
        )


@dataclass(slots=True)
class ObjectiveConfig:
    objective_keys: list[str] = field(default_factory=list)
    objective_directions: dict[str, str] = field(default_factory=dict)
    selection_objective: str | None = None
    protected_objectives: list[str] = field(default_factory=list)
    frontier_type: str = "per_example"
    acceptance_criterion: str = "primary_improvement"
    min_objective_delta: float | None = None
    objective_regression_tolerance: float | None = None

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        if self.objective_keys:
            gepa["objective_keys"] = list(self.objective_keys)
        if self.objective_directions:
            gepa["objective_directions"] = dict(self.objective_directions)
        if self.selection_objective is not None:
            gepa["selection_objective"] = self.selection_objective
        gepa["frontier_type"] = self.frontier_type
        gepa["acceptance_criterion"] = self.acceptance_criterion
        objective_acceptance: dict[str, Any] = {}
        if self.protected_objectives:
            objective_acceptance["protected_objectives"] = list(self.protected_objectives)
        if self.min_objective_delta is not None:
            objective_acceptance["min_objective_delta"] = self.min_objective_delta
        if self.objective_regression_tolerance is not None:
            objective_acceptance["objective_regression_tolerance"] = (
                self.objective_regression_tolerance
            )
        if objective_acceptance:
            gepa["objective_acceptance"] = objective_acceptance


@dataclass(slots=True)
class BudgetConfig:
    max_cost_usd: float = 0.0
    max_time_seconds: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        gepa["max_cost_usd"] = float(self.max_cost_usd)
        if self.max_time_seconds is not None:
            gepa["max_time_seconds"] = int(self.max_time_seconds)
        if self.max_prompt_tokens is not None:
            gepa["max_prompt_tokens"] = int(self.max_prompt_tokens)
        if self.max_completion_tokens is not None:
            gepa["max_completion_tokens"] = int(self.max_completion_tokens)
        if self.max_total_tokens is not None:
            gepa["max_total_tokens"] = int(self.max_total_tokens)


@dataclass(slots=True)
class GepaBudgetConfig:
    max_generations: int = 1
    proposals_per_generation: int = 1
    minibatch_size: int = 1
    max_total_rollouts: int = 16
    max_train_rollouts: int | None = None
    max_heldout_rollouts: int | None = None
    minibatch_accept_margin: float = 0.0
    rollout_failure_rate_tolerance: float = 0.25
    rollout_chunk_size: int | None = None
    proposer_estimated_cost_usd: float | None = None
    proposer_estimated_prompt_tokens: int | None = None
    proposer_estimated_completion_tokens: int | None = None
    proposer_estimated_total_tokens: int | None = None
    rollout_estimated_cost_usd: float | None = None
    rollout_estimated_prompt_tokens: int | None = None
    rollout_estimated_completion_tokens: int | None = None
    rollout_estimated_total_tokens: int | None = None
    rollout_estimated_wall_seconds: int | None = None

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        gepa.update(
            {
                "max_generations": int(self.max_generations),
                "proposals_per_generation": int(self.proposals_per_generation),
                "minibatch_size": int(self.minibatch_size),
                "minibatch_accept_margin": float(self.minibatch_accept_margin),
                "max_total_rollouts": int(self.max_total_rollouts),
                "rollout_failure_rate_tolerance": float(self.rollout_failure_rate_tolerance),
            }
        )
        if self.max_train_rollouts is not None:
            gepa["max_train_rollouts"] = int(self.max_train_rollouts)
        if self.max_heldout_rollouts is not None:
            gepa["max_heldout_rollouts"] = int(self.max_heldout_rollouts)
        if self.rollout_chunk_size is not None:
            gepa["rollout_chunk_size"] = int(self.rollout_chunk_size)
        if self.proposer_estimated_cost_usd is not None:
            gepa["proposer_estimated_cost_usd"] = float(self.proposer_estimated_cost_usd)
        if self.proposer_estimated_prompt_tokens is not None:
            gepa["proposer_estimated_prompt_tokens"] = int(self.proposer_estimated_prompt_tokens)
        if self.proposer_estimated_completion_tokens is not None:
            gepa["proposer_estimated_completion_tokens"] = int(
                self.proposer_estimated_completion_tokens
            )
        if self.proposer_estimated_total_tokens is not None:
            gepa["proposer_estimated_total_tokens"] = int(self.proposer_estimated_total_tokens)
        if self.rollout_estimated_cost_usd is not None:
            gepa["rollout_estimated_cost_usd"] = float(self.rollout_estimated_cost_usd)
        if self.rollout_estimated_prompt_tokens is not None:
            gepa["rollout_estimated_prompt_tokens"] = int(self.rollout_estimated_prompt_tokens)
        if self.rollout_estimated_completion_tokens is not None:
            gepa["rollout_estimated_completion_tokens"] = int(
                self.rollout_estimated_completion_tokens
            )
        if self.rollout_estimated_total_tokens is not None:
            gepa["rollout_estimated_total_tokens"] = int(self.rollout_estimated_total_tokens)
        if self.rollout_estimated_wall_seconds is not None:
            gepa["rollout_estimated_wall_seconds"] = int(self.rollout_estimated_wall_seconds)


DEFAULT_PROPOSER_ESTIMATED_COST_USD = 0.05
DEFAULT_ROLLOUT_ESTIMATED_COST_USD = 0.01


def _apply_default_budget_estimates(gepa: dict[str, Any]) -> None:
    if float(gepa.get("max_cost_usd") or 0.0) <= 0.0:
        return
    gepa.setdefault("proposer_estimated_cost_usd", DEFAULT_PROPOSER_ESTIMATED_COST_USD)
    gepa.setdefault("rollout_estimated_cost_usd", DEFAULT_ROLLOUT_ESTIMATED_COST_USD)


@dataclass(slots=True)
class GepaPipeline:
    mode: GepaPipelineMode | str = GepaPipelineMode.SYNC_SERIAL
    staleness_policy: GepaStalenessPolicy | str = GepaStalenessPolicy.FULL
    staleness_delta_max: int = 2
    rollout_transport: RolloutTransport | str = RolloutTransport.ASYNC
    rollout_timeout_seconds: int = 600
    proposer_concurrency: int = 1
    rollout_concurrency: int = 8
    evaluator_concurrency: int = 1
    candidate_concurrency: int = 1
    speculative_alpha: float | None = None
    adaptive_stage_workers: bool = False
    adaptive_stage_workers_max: int = 128
    adaptive_stage_workers_backlog_threshold: int = 2
    adaptive_stage_workers_stale_gap_threshold: int = 2

    @classmethod
    def sync_serial(
        cls,
        *,
        rollout_transport: RolloutTransport | str = RolloutTransport.ASYNC,
        rollout_timeout_seconds: int = 600,
    ) -> "GepaPipeline":
        return cls(
            mode=GepaPipelineMode.SYNC_SERIAL,
            rollout_transport=rollout_transport,
            rollout_timeout_seconds=rollout_timeout_seconds,
            candidate_concurrency=1,
            rollout_concurrency=1,
        )

    @classmethod
    def async_pipelined(
        cls,
        *,
        candidate_concurrency: int = 4,
        rollout_concurrency: int = 8,
        rollout_transport: RolloutTransport | str = RolloutTransport.ASYNC,
        rollout_timeout_seconds: int = 600,
    ) -> "GepaPipeline":
        return cls(
            mode=GepaPipelineMode.ASYNC_PIPELINED,
            staleness_policy=GepaStalenessPolicy.FULL,
            rollout_transport=rollout_transport,
            rollout_timeout_seconds=rollout_timeout_seconds,
            candidate_concurrency=candidate_concurrency,
            rollout_concurrency=rollout_concurrency,
        )

    @classmethod
    def flash_evolve(
        cls,
        *,
        candidate_concurrency: int = 8,
        rollout_concurrency: int = 8,
        staleness_policy: GepaStalenessPolicy | str = GepaStalenessPolicy.GUARDED,
        staleness_delta_max: int = 2,
        speculative_alpha: float | None = None,
        adaptive_stage_workers: bool = False,
        rollout_transport: RolloutTransport | str = RolloutTransport.ASYNC,
        rollout_timeout_seconds: int = 600,
    ) -> "GepaPipeline":
        return cls(
            mode=GepaPipelineMode.FLASH_EVOLVE,
            staleness_policy=staleness_policy,
            staleness_delta_max=staleness_delta_max,
            speculative_alpha=speculative_alpha,
            adaptive_stage_workers=adaptive_stage_workers,
            rollout_transport=rollout_transport,
            rollout_timeout_seconds=rollout_timeout_seconds,
            candidate_concurrency=candidate_concurrency,
            rollout_concurrency=rollout_concurrency,
        )

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        rollout_transport = str(self.rollout_transport)
        gepa["rollout_submission_mode"] = rollout_transport
        gepa["rollout_poll_interval_ms"] = 250
        gepa["rollout_async_timeout_seconds"] = int(self.rollout_timeout_seconds)
        gepa["pipeline"] = {
            "mode": str(self.mode),
            "staleness_policy": str(self.staleness_policy),
            "delta_max": int(self.staleness_delta_max),
            "max_in_flight_candidates": int(self.candidate_concurrency),
            "workers": {
                "propose": int(self.proposer_concurrency),
                "rollout": int(self.rollout_concurrency),
                "evaluate": int(self.evaluator_concurrency),
            },
            "adaptive_rollout_concurrency": {
                "enabled": True,
                "initial": int(self.rollout_concurrency),
                "min": 1,
                "max": int(self.rollout_concurrency),
                "increase_step": 5,
                "decrease_step": 5,
                "increase_after_successes": 20,
            },
            "speculative_completion": {
                "enabled": self.speculative_alpha is not None,
                "alpha": float(self.speculative_alpha or 0.25),
            },
            "adaptive_stage_workers": {
                "enabled": bool(self.adaptive_stage_workers),
                "min": 1,
                "max": int(self.adaptive_stage_workers_max),
                "backlog_threshold": int(self.adaptive_stage_workers_backlog_threshold),
                "stale_gap_threshold": int(self.adaptive_stage_workers_stale_gap_threshold),
            },
        }


@dataclass(slots=True)
class CacheConfig:
    mode: str = "readwrite"
    path: str | Path | None = None
    namespace: str | None = None

    def to_toml(self) -> dict[str, Any]:
        return _drop_none(
            {
                "mode": self.mode,
                "path": str(self.path) if self.path is not None else None,
                "namespace": self.namespace,
            }
        )


@dataclass(slots=True)
class JesterkyWorkflowConfig:
    """Per-run toggle for jesterky trace annotate inside GEPA."""

    enabled: bool = False
    spec: str = "examples/gepa_trace_annotate.json"
    command: str = "jesterky"
    actor: str = "codex"
    model: str | None = None
    concurrency: int = 4
    timeout_seconds: int = 600
    fail_closed: bool = True

    def to_toml(self) -> dict[str, Any]:
        return _drop_none(
            {
                "enabled": bool(self.enabled),
                "spec": str(self.spec),
                "command": str(self.command),
                "actor": str(self.actor),
                "model": self.model,
                "concurrency": int(self.concurrency),
                "timeout_seconds": int(self.timeout_seconds),
                "fail_closed": bool(self.fail_closed),
            }
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not str(self.spec or "").strip():
            raise ValueError("jesterky_workflow.spec must be non-empty when enabled")
        if not str(self.command or "").strip():
            raise ValueError("jesterky_workflow.command must be non-empty when enabled")
        actor = str(self.actor or "").strip()
        if actor not in {"fake", "codex"}:
            raise ValueError(
                "jesterky_workflow.actor must be fake or codex when enabled, "
                f"got {self.actor!r}"
            )
        if int(self.concurrency) <= 0:
            raise ValueError("jesterky_workflow.concurrency must be > 0 when enabled")
        if int(self.timeout_seconds) <= 0:
            raise ValueError(
                "jesterky_workflow.timeout_seconds must be > 0 when enabled"
            )


@dataclass(slots=True)
class OutputConfig:
    output_dir: str | Path | None = None


@dataclass(slots=True)
class GepaConfig:
    algorithm: ClassVar[OptimizerAlgorithmSlug] = OptimizerAlgorithmSlug.GEPA

    container: ContainerConnection
    taskset: TasksetSelection
    task_pools: GepaTaskPools
    container_headers: dict[str, str] = field(default_factory=dict)
    container_auth_bearer_env: str | None = None
    container_command: list[str] = field(default_factory=list)
    container_cwd: str | Path | None = None
    container_startup_timeout_seconds: int | None = None
    run: RunSettings = field(default_factory=RunSettings)
    program: PromptProgram | None = None
    objectives: ObjectiveConfig | None = None
    policy: PolicyConfig | None = None
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    budgets: GepaBudgetConfig = field(default_factory=GepaBudgetConfig)
    pipeline: GepaPipeline = field(default_factory=GepaPipeline)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    jesterky_workflow: JesterkyWorkflowConfig = field(
        default_factory=JesterkyWorkflowConfig
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    usage_registration: UsageRegistrationConfig = field(
        default_factory=UsageRegistrationConfig
    )
    output: OutputConfig | None = None
    target_modules: list[str] | None = None
    seed_candidate: dict[str, str] | None = None
    _source_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.output and self.output.output_dir is not None:
            self.run.output_dir = self.output.output_dir
        self.validate()

    @classmethod
    def from_toml(cls, path: str | Path) -> "GepaConfig":
        source_path = Path(path)
        document = GepaTomlDocument.model_validate(tomllib.loads(source_path.read_text()))
        return document.to_config(source_path)

    def validate(self) -> None:
        if not self.container.url.strip():
            raise ValueError("GepaConfig.container.url is required")
        if not self.taskset.train_ids:
            raise ValueError("GepaConfig.taskset.train_ids must not be empty")
        if not self.taskset.heldout_ids:
            raise ValueError("GepaConfig.taskset.heldout_ids must not be empty")
        self.task_pools.validate()
        self.task_pools.validate_against_taskset(
            self.taskset.train_ids, self.taskset.heldout_ids
        )
        self.jesterky_workflow.validate()
        if self.policy is not None:
            if not self.target_modules:
                raise ValueError(
                    "GEPA policy runs require candidate.target_modules so "
                    "prompt delivery assertions can be bound to candidate fields"
                )
            proxy_mode = str(self.policy.proxy_mode or "").strip().lower()
            if proxy_mode == "allow_direct":
                raise ValueError(
                    "policy.proxy_mode='allow_direct' is forbidden for GEPA "
                    "policy runs; use 'proxy_only' or 'assert_proxy'"
                )
            if proxy_mode not in {"proxy_only", "assert_proxy"}:
                raise ValueError(
                    "policy.proxy_mode must be 'proxy_only' or 'assert_proxy' "
                    "for GEPA policy runs"
                )
        elif self.target_modules:
            raise ValueError(
                "GEPA prompt-overlay runs require an explicit policy with "
                "proxy_mode='proxy_only' or 'assert_proxy'"
            )

    def to_toml_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        payload["run"] = self.run.to_toml()
        container_payload: dict[str, Any] = {"url": self.container.url.rstrip("/")}
        if self.container_headers:
            container_payload["headers"] = dict(self.container_headers)
        if self.container_auth_bearer_env is not None:
            container_payload["auth_bearer_env"] = self.container_auth_bearer_env
        if self.container_command:
            container_payload["command"] = list(self.container_command)
        if self.container_cwd is not None:
            container_payload["cwd"] = str(self.container_cwd)
        if self.container_startup_timeout_seconds is not None:
            container_payload["startup_timeout_seconds"] = int(
                self.container_startup_timeout_seconds
            )
        payload["container"] = container_payload
        payload["taskset"] = self.taskset.to_toml()
        if self.target_modules is not None:
            payload["candidate"] = {"target_modules": list(self.target_modules)}
        else:
            payload["candidate"] = {"target_modules": []}
        if self.seed_candidate is not None:
            payload["seed_candidate"] = dict(self.seed_candidate)
        else:
            payload["seed_candidate"] = {}
        if self.policy is None:
            payload["policy"] = {"enabled": False}
        else:
            payload["policy"] = self.policy.to_toml()
        payload["proposer"] = self.proposer.to_toml()
        gepa: dict[str, Any] = {}
        self.budgets.apply_to_gepa(gepa)
        self.budget.apply_to_gepa(gepa)
        self.pipeline.apply_to_gepa(gepa)
        if self.objectives is not None:
            self.objectives.apply_to_gepa(gepa)
        _apply_default_budget_estimates(gepa)
        gepa["task_pools"] = self.task_pools.to_toml()
        payload["gepa"] = gepa
        payload["jesterky_workflow"] = self.jesterky_workflow.to_toml()
        payload["cache"] = self.cache.to_toml()
        return payload

    def to_config_json(self) -> dict[str, Any]:
        return self.to_toml_dict()

    def write_toml(self, path: str | Path | None = None) -> Path:
        if path is None:
            if self._source_path is not None:
                path = self._source_path.with_name(
                    f"{self._source_path.stem}.{self.run.run_id}.sdk.toml"
                )
            else:
                output_dir = Path(self.run.output_dir)
                path = output_dir / "_sdk_configs" / f"{self.run.run_id}.toml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_toml_dumps(self.to_toml_dict()))
        return path

    def execute(self) -> GepaRunResult:
        self._register_usage_submit()
        if not self.container_command:
            self._preflight_container_capabilities()
        config_path = self.write_toml()
        try:
            result = _NativeGepaRun.from_toml(str(config_path)).execute()
        except Exception:
            self._register_usage_complete(status="failed")
            raise
        self._register_usage_complete(status="succeeded")
        return result

    def _register_usage_submit(self) -> None:
        if not self.usage_registration.enabled:
            return
        from .hosted import (
            HostedOptimizerClient,
            HostedOptimizerError,
            OptimizerAlgorithmSlug,
        )

        try:
            client = HostedOptimizerClient(
                api_key="",
                register_usage=None,
                require_api_key=False,
            )
            client.register_usage_submit(
                algorithm=OptimizerAlgorithmSlug.GEPA,
                models=self._usage_registration_models(),
            )
        except HostedOptimizerError:
            return

    def _register_usage_complete(self, *, status: str) -> None:
        if not self.usage_registration.enabled:
            return
        from .hosted import (
            HostedOptimizerClient,
            HostedOptimizerError,
            OptimizerAlgorithmSlug,
        )

        try:
            client = HostedOptimizerClient(
                api_key="",
                register_usage=None,
                require_api_key=False,
            )
            client.register_usage_complete(
                algorithm=OptimizerAlgorithmSlug.GEPA,
                status=status,
                models=self._usage_registration_models(),
            )
        except HostedOptimizerError:
            return

    def _usage_registration_models(self) -> list[dict[str, str]]:
        models: list[dict[str, str]] = []
        if self.proposer.model:
            models.append(
                {
                    "role": "proposer",
                    "provider": self.proposer.provider,
                    "model": self.proposer.model,
                }
            )
        if self.policy is not None:
            models.append(
                {
                    "role": "policy",
                    "provider": self.policy.provider,
                    "model": self.policy.model,
                }
            )
        return models

    def _preflight_container_capabilities(self) -> None:
        metadata = ContainerMetadataPayload.model_validate(
            _http_json(
                self.container.url,
                "/metadata",
                headers=self.container_headers,
                auth_bearer_env=self.container_auth_bearer_env,
            )
        )
        if self.policy is None and not metadata.capabilities.metadata.policy_ready:
            raise ValueError(
                "policy=None requires the container to advertise policy readiness in "
                "metadata.capabilities.metadata.policy_ready"
            )
        if self.program is None:
            ProgramPayload.model_validate(
                _http_json(
                    self.container.url,
                    "/program",
                    headers=self.container_headers,
                    auth_bearer_env=self.container_auth_bearer_env,
                )
            )


class GepaRun:
    def __init__(self, config: GepaConfig) -> None:
        self.config = config
        self.config_path = "" if config._source_path is None else str(config._source_path)

    @staticmethod
    def from_toml(path: str | Path) -> "GepaRun":
        return GepaRun(GepaConfig.from_toml(path))

    def execute(self) -> GepaRunResult:
        return self.config.execute()


def _http_json(
    base_url: str,
    path: str,
    *,
    timeout_seconds: float = 10.0,
    headers: Mapping[str, str] | None = None,
    auth_bearer_env: str | None = None,
) -> dict[str, Any]:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported container URL scheme: {parsed.scheme!r}")
    connection_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    request_path = path
    if parsed.path and parsed.path != "/":
        request_path = f"{parsed.path.rstrip('/')}{path}"
    with closing(connection_cls(parsed.netloc, timeout=timeout_seconds)) as connection:
        connection.request(
            "GET",
            request_path,
            headers=_http_headers(headers or {}, auth_bearer_env),
        )
        response = connection.getresponse()
        body = response.read()
    if response.status >= 400:
        raise ValueError(f"container GET {path} failed with HTTP {response.status}")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"container GET {path} did not return a JSON object")
    return payload


def _http_headers(headers: Mapping[str, str], auth_bearer_env: str | None) -> dict[str, str]:
    resolved = dict(headers)
    if auth_bearer_env and not _has_authorization_header(resolved):
        resolved["authorization"] = _bearer_header_from_env(auth_bearer_env)
    return resolved


def _has_authorization_header(headers: Mapping[str, str]) -> bool:
    return any(name.strip().lower() == "authorization" for name in headers)


def _bearer_header_from_env(env_name: str) -> str:
    token = os.getenv(env_name, "").strip()
    if not token:
        raise ValueError(f"container auth references missing environment variable {env_name!r}")
    return f"Bearer {token}"


def _backend_base_url_from_env() -> str | None:
    for name in (
        "SYNTH_BACKEND_URL_OVERRIDE",
        "SYNTH_BACKEND_URL",
        "SYNTH_API_URL",
        "DEV_SYNTH_BACKEND_URL",
        "DEV_BACKEND_URL",
        "PROD_SYNTH_BACKEND_URL",
        "PROD_BACKEND_URL",
        "BACKEND_URL",
    ):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _normalize_backend_base_url(url: str) -> str:
    base = url.strip().rstrip("/")
    for suffix in ("/v1", "/api"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _path_segment(value: str, field: str) -> str:
    segment = value.strip()
    if not segment:
        raise ValueError(f"{field} is required")
    if any(char in segment for char in "/?#"):
        raise ValueError(f"{field} must be a single URL path segment")
    return segment


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != []}


def _toml_dumps(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    scalars: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            tables[key] = value
        elif value is not None:
            scalars[key] = value
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        _write_toml_table(lines, key, value)
    return "\n".join(lines).rstrip() + "\n"


def _write_toml_table(lines: list[str], prefix: str, payload: Mapping[str, Any]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            table_items.append((key, value))
        else:
            scalar_items.append((key, value))
    if scalar_items:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{prefix}]")
        for key, value in scalar_items:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in table_items:
        _write_toml_table(lines, f"{prefix}.{key}", value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, StrEnum):
        return json.dumps(str(value))
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))
