"""Canonical prompt-learning config models for local offline prompt-opt."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from ..utils import load_toml
from .shared import ExtraModel


class PromptLearningPolicyConfig(ExtraModel):
    provider: str | None = None
    inference_url: str | None = None
    inference_mode: str | None = None
    temperature: float = 0.0
    max_completion_tokens: int = 512
    policy_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    context_override: dict[str, Any] | None = None


class MessagePatternConfig(ExtraModel):
    role: str
    pattern: str
    order: int = 0


class PromptStageConfig(ExtraModel):
    id: str | None = None
    name: str | None = None
    messages: list[MessagePatternConfig] = Field(default_factory=list)
    wildcards: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptCandidateConfig(ExtraModel):
    stages: list[PromptStageConfig] = Field(default_factory=list)


class TrainPoolsConfig(ExtraModel):
    reflection_seeds: list[int] = Field(default_factory=list)
    pareto_seeds: list[int] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)


class TaskDataConfig(ExtraModel):
    split: str | None = None
    train_pools: TrainPoolsConfig | dict[str, Any] | None = None
    validation_seeds: list[int] = Field(default_factory=list)
    validation_examples: list[dict[str, Any]] = Field(default_factory=list)
    train_examples: list[dict[str, Any]] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)
    examples_by_seed: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("train_pools", mode="before")
    @classmethod
    def _coerce_train_pools(cls, value: Any) -> Any:
        if value is None or isinstance(value, TrainPoolsConfig):
            return value
        if isinstance(value, Mapping):
            return TrainPoolsConfig.model_validate(dict(value))
        raise ValueError("train_pools must be a mapping when provided")


class PopulationConfig(ExtraModel):
    initial_size: int = 1
    num_generations: int = 1
    children_per_generation: int = 4


class TerminationConditionsConfig(ExtraModel):
    total_rollouts: int | None = None
    max_iterations: int | None = None
    patience: int | None = None


class GEPAConfig(ExtraModel):
    env_name: str | None = None
    initial_candidate: PromptCandidateConfig | dict[str, Any] | None = None
    termination_conditions: TerminationConditionsConfig | dict[str, Any] | None = None
    mode: str | None = None
    execution_mode: str | None = None
    proposer_backend: str | None = None
    population: PopulationConfig | dict[str, Any] | None = None
    modules: list[dict[str, Any]] | None = None

    @field_validator("initial_candidate", mode="before")
    @classmethod
    def _coerce_initial_candidate(cls, value: Any) -> Any:
        if value is None or isinstance(value, PromptCandidateConfig):
            return value
        if isinstance(value, Mapping):
            return PromptCandidateConfig.model_validate(dict(value))
        raise ValueError("initial_candidate must be a mapping when provided")

    @field_validator("termination_conditions", mode="before")
    @classmethod
    def _coerce_termination(cls, value: Any) -> Any:
        if value is None or isinstance(value, TerminationConditionsConfig):
            return value
        if isinstance(value, Mapping):
            return TerminationConditionsConfig.model_validate(dict(value))
        raise ValueError("termination_conditions must be a mapping when provided")

    @field_validator("population", mode="before")
    @classmethod
    def _coerce_population(cls, value: Any) -> Any:
        if value is None or isinstance(value, PopulationConfig):
            return value
        if isinstance(value, Mapping):
            return PopulationConfig.model_validate(dict(value))
        raise ValueError("population must be a mapping when provided")


class MIPROAlgorithmConfig(ExtraModel):
    initial_candidate: PromptCandidateConfig | dict[str, Any] | None = None
    execution_mode: str | None = None
    mode: str | None = None
    num_candidates: int = 8
    max_iterations: int = 8
    early_stop_rounds: int = 3
    min_improvement: float = 1e-6
    proposer_backend: str = "single_prompt"
    seed: int = 0
    termination_conditions: TerminationConditionsConfig | dict[str, Any] | None = None

    @field_validator("initial_candidate", mode="before")
    @classmethod
    def _coerce_initial_candidate(cls, value: Any) -> Any:
        if value is None or isinstance(value, PromptCandidateConfig):
            return value
        if isinstance(value, Mapping):
            return PromptCandidateConfig.model_validate(dict(value))
        raise ValueError("initial_candidate must be a mapping when provided")

    @field_validator("termination_conditions", mode="before")
    @classmethod
    def _coerce_termination(cls, value: Any) -> Any:
        if value is None or isinstance(value, TerminationConditionsConfig):
            return value
        if isinstance(value, Mapping):
            return TerminationConditionsConfig.model_validate(dict(value))
        raise ValueError("termination_conditions must be a mapping when provided")


class PromptLearningConfig(ExtraModel):
    algorithm: str
    job_kind: str | None = None
    algorithm_name: str | None = None
    execution_mode: str | None = None
    config_schema_version: str | None = None
    container_url: str | None = None
    container_id: str | None = None
    task_data: TaskDataConfig | dict[str, Any] | None = None
    policy: PromptLearningPolicyConfig | dict[str, Any] | None = None
    gepa: GEPAConfig | None = None
    mipro: MIPROAlgorithmConfig | dict[str, Any] | None = None
    verifier: dict[str, Any] | None = None
    proxy_models: dict[str, Any] | None = None
    env_config: dict[str, Any] | None = None
    use_byok: bool | None = None

    @field_validator("task_data", mode="before")
    @classmethod
    def _coerce_task_data(cls, value: Any) -> Any:
        if value is None or isinstance(value, TaskDataConfig):
            return value
        if isinstance(value, Mapping):
            return TaskDataConfig.model_validate(dict(value))
        raise ValueError("task_data must be a mapping when provided")

    @field_validator("policy", mode="before")
    @classmethod
    def _coerce_policy(cls, value: Any) -> Any:
        if value is None or isinstance(value, PromptLearningPolicyConfig):
            return value
        if isinstance(value, Mapping):
            return PromptLearningPolicyConfig.model_validate(dict(value))
        raise ValueError("policy must be a mapping when provided")

    @field_validator("mipro", mode="before")
    @classmethod
    def _coerce_mipro(cls, value: Any) -> Any:
        if value is None or isinstance(value, MIPROAlgorithmConfig):
            return value
        if isinstance(value, Mapping):
            return MIPROAlgorithmConfig.model_validate(dict(value))
        raise ValueError("mipro must be a mapping when provided")

    @model_validator(mode="after")
    def _normalize_execution_mode(self) -> "PromptLearningConfig":
        if self.algorithm_name is None:
            self.algorithm_name = self.algorithm
        if self.job_kind is None:
            self.job_kind = "optimization"

        algorithm_mode = None
        if self.algorithm == "gepa" and self.gepa is not None:
            algorithm_mode = self.gepa.execution_mode or self.gepa.mode
        elif self.algorithm == "mipro" and self.mipro is not None:
            algorithm_mode = self.mipro.execution_mode or self.mipro.mode

        effective = self.execution_mode or algorithm_mode or "offline"
        if effective == "proxied":
            effective = "retrieved"
        self.execution_mode = effective

        if self.algorithm == "gepa" and self.gepa is not None:
            self.gepa.execution_mode = effective
        if self.algorithm == "mipro" and self.mipro is not None:
            self.mipro.execution_mode = effective
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PromptLearningConfig":
        payload = dict(data)
        if "prompt_learning" in payload and isinstance(payload["prompt_learning"], Mapping):
            payload = dict(payload["prompt_learning"])
        return cls.model_validate(payload)

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptLearningConfig":
        return cls.from_mapping(load_toml(path))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)
