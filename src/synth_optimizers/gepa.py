from __future__ import annotations

import http.client
import json
import tomllib
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field
from synth_containers import ContainerConnection, PromptProgram

from ._synth_optimizers import GepaRun as _NativeGepaRun
from ._synth_optimizers import GepaRunResult
from ._synth_optimizers import default_proposer_best_practices as _default_proposer_best_practices


class PolicyType(StrEnum):
    DAG = "dag"
    REACT = "react"
    CODEX = "codex"


class RolloutTransport(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class ContainerTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1)

    def to_connection(self) -> ContainerConnection:
        return ContainerConnection(url=self.url)


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


class ProposerTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    backend: str = "codex_app_server"
    runtime_substrate: str = "local"
    execution_mode: str = "local_process"
    provider: str = "openai"
    api_family: str = "chat_completions"
    model: str | None = "gpt-5.4-nano"
    reasoning_effort: str | None = "medium"
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

    def to_domain(self, base_dir: Path) -> "ProposerConfig":
        codex_home = None
        if self.codex_home is not None:
            path = Path(self.codex_home)
            codex_home = path if path.is_absolute() else base_dir / path
        return ProposerConfig(
            backend=self.backend,
            runtime_substrate=self.runtime_substrate,
            execution_mode=self.execution_mode,
            provider=self.provider,
            api_family=self.api_family,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            auth_mode=self.auth_mode,
            api_key_env=self.api_key_env,
            copy_host_auth=self.copy_host_auth,
            codex_home=codex_home,
            timeout_seconds=self.timeout_seconds,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
            command=list(self.command),
            prompt=self.prompt.to_domain(base_dir),
            docker=self.docker.to_domain() if self.docker is not None else None,
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
    proxy_mode: str = "allow_direct"
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

    rollout: int = 8


class GepaPipelineTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_in_flight_candidates: int = 1
    workers: GepaPipelineWorkersTomlSection = Field(default_factory=GepaPipelineWorkersTomlSection)


class ObjectiveAcceptanceTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    protected_objectives: list[str] = Field(default_factory=list)
    min_objective_delta: float | None = None
    objective_regression_tolerance: float | None = None


class GepaTomlSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_cost_usd: float = 0.0
    max_time_seconds: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
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
    pipeline: GepaPipelineTomlSection = Field(default_factory=GepaPipelineTomlSection)
    objective_keys: list[str] = Field(default_factory=list)
    objective_directions: dict[str, str] = Field(default_factory=dict)
    selection_objective: str | None = None
    frontier_type: str = "per_example"
    acceptance_criterion: str = "primary_improvement"
    objective_acceptance: ObjectiveAcceptanceTomlSection = Field(
        default_factory=ObjectiveAcceptanceTomlSection
    )

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
        )

    def pipeline_config(self) -> "GepaPipeline":
        return GepaPipeline(
            rollout_transport=self.rollout_submission_mode,
            rollout_timeout_seconds=self.rollout_async_timeout_seconds,
            candidate_concurrency=self.pipeline.max_in_flight_candidates,
            rollout_concurrency=self.pipeline.workers.rollout,
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


class GepaTomlDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    container: ContainerTomlSection
    run: RunSettingsTomlSection = Field(default_factory=RunSettingsTomlSection)
    taskset: TasksetTomlSection = Field(default_factory=TasksetTomlSection)
    proposer: ProposerTomlSection = Field(default_factory=ProposerTomlSection)
    policy: PolicyTomlSection = Field(default_factory=PolicyTomlSection)
    gepa: GepaTomlSection = Field(default_factory=GepaTomlSection)
    cache: CacheTomlSection = Field(default_factory=CacheTomlSection)
    candidate: CandidateTomlSection = Field(default_factory=CandidateTomlSection)
    seed_candidate: dict[str, str] = Field(default_factory=dict)

    def to_config(self, source_path: Path) -> "GepaConfig":
        return GepaConfig(
            container=self.container.to_connection(),
            taskset=self.taskset.to_domain(),
            run=self.run.to_domain(),
            objectives=self.gepa.objective_config(),
            policy=self.policy.to_domain(),
            proposer=self.proposer.to_domain(source_path.parent),
            budgets=self.gepa.gepa_budget_config(),
            pipeline=self.gepa.pipeline_config(),
            budget=self.gepa.budget_config(),
            cache=self.cache.to_domain(),
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
class ProposerConfig:
    backend: str = "codex_app_server"
    runtime_substrate: str = "local"
    execution_mode: str = "local_process"
    provider: str = "openai"
    api_family: str = "chat_completions"
    base_url: str | None = None
    model: str | None = "gpt-5.4-nano"
    reasoning_effort: str | None = "medium"
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
                "auth_mode": self.auth_mode,
                "api_key_env": self.api_key_env,
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
    proxy_mode: str = "allow_direct"
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

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        gepa.update(
            {
                "max_generations": int(self.max_generations),
                "proposals_per_generation": int(self.proposals_per_generation),
                "minibatch_size": int(self.minibatch_size),
                "minibatch_accept_margin": float(self.minibatch_accept_margin),
                "max_total_rollouts": int(self.max_total_rollouts),
                "rollout_failure_rate_tolerance": float(
                    self.rollout_failure_rate_tolerance
                ),
            }
        )
        if self.max_train_rollouts is not None:
            gepa["max_train_rollouts"] = int(self.max_train_rollouts)
        if self.max_heldout_rollouts is not None:
            gepa["max_heldout_rollouts"] = int(self.max_heldout_rollouts)


@dataclass(slots=True)
class GepaPipeline:
    rollout_transport: RolloutTransport | str = RolloutTransport.ASYNC
    rollout_timeout_seconds: int = 600
    rollout_concurrency: int = 8
    candidate_concurrency: int = 1

    def apply_to_gepa(self, gepa: dict[str, Any]) -> None:
        rollout_transport = str(self.rollout_transport)
        gepa["rollout_submission_mode"] = rollout_transport
        gepa["rollout_poll_interval_ms"] = 250
        gepa["rollout_async_timeout_seconds"] = int(self.rollout_timeout_seconds)
        gepa["pipeline"] = {
            "mode": "async_pipelined",
            "max_in_flight_candidates": int(self.candidate_concurrency),
            "workers": {
                "propose": 1,
                "rollout": int(self.rollout_concurrency),
                "evaluate": 1,
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
class OutputConfig:
    output_dir: str | Path | None = None


@dataclass(slots=True)
class GepaConfig:
    container: ContainerConnection
    taskset: TasksetSelection
    run: RunSettings = field(default_factory=RunSettings)
    program: PromptProgram | None = None
    objectives: ObjectiveConfig | None = None
    policy: PolicyConfig | None = None
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    budgets: GepaBudgetConfig = field(default_factory=GepaBudgetConfig)
    pipeline: GepaPipeline = field(default_factory=GepaPipeline)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
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

    def to_toml_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        payload["run"] = self.run.to_toml()
        payload["container"] = {"url": self.container.url.rstrip("/")}
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
        payload["gepa"] = gepa
        payload["cache"] = self.cache.to_toml()
        return payload

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
        self._preflight_container_capabilities()
        config_path = self.write_toml()
        return _NativeGepaRun.from_toml(str(config_path)).execute()

    def _preflight_container_capabilities(self) -> None:
        metadata = ContainerMetadataPayload.model_validate(
            _http_json(self.container.url, "/metadata")
        )
        if self.policy is None and not metadata.capabilities.metadata.policy_ready:
            raise ValueError(
                "policy=None requires the container to advertise policy readiness in "
                "metadata.capabilities.metadata.policy_ready"
            )
        if self.program is None:
            ProgramPayload.model_validate(_http_json(self.container.url, "/program"))


class GepaRun:
    def __init__(self, config: GepaConfig) -> None:
        self.config = config
        self.config_path = "" if config._source_path is None else str(config._source_path)

    @staticmethod
    def from_toml(path: str | Path) -> "GepaRun":
        return GepaRun(GepaConfig.from_toml(path))

    def execute(self) -> GepaRunResult:
        return self.config.execute()


def _http_json(base_url: str, path: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
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
        connection.request("GET", request_path)
        response = connection.getresponse()
        body = response.read()
    if response.status >= 400:
        raise ValueError(f"container GET {path} failed with HTTP {response.status}")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"container GET {path} did not return a JSON object")
    return payload


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
