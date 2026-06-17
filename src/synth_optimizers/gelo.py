from __future__ import annotations

import json
import tomllib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from .hosted import ContainerPoolTarget, OptimizerAlgorithmSlug
from .tunnels import TunnelProvider, tunnel_provider_value


class GeloCacheMode(StrEnum):
    OFF = "off"
    READWRITE = "readwrite"
    READONLY = "readonly"


class GeloProposerRole(StrEnum):
    CORE = "core_proposer"
    AUX_HILL_CLIMB = "aux_hill_climb_proposer"
    AUX_DATA_MINER = "aux_data_miner_proposer"
    AUX_CONSOLIDATE = "aux_consolidate_proposer"
    AUX_CONSOLIDATE_HC = "aux_consolidate_hill_climb_proposer"
    THEME_VERIFIER = "theme_verifier_agent"
    TERMINATOR = "terminator_agent"


class GeloRewardMode(StrEnum):
    ACHIEVEMENT = "achievement"
    PROGRESS = "progress"
    DUNGEONGRID_PROGRESS = "dungeongrid_progress_reward"


class GeloCheckpointSemantics(StrEnum):
    TRUE_ENV_SNAPSHOT = "true_environment_snapshot"
    REQUEST_SNAPSHOT = "request_snapshot"


class GeloPluginKind(StrEnum):
    SFT = "sft"
    RLVR = "rlvr"
    OPSD = "opsd"


class GeloPresetName(StrEnum):
    CRAFTER = "crafter"
    CRAFTER_SMOKE = "crafter_smoke"
    SOKOBAN_SMOKE = "sokoban_smoke"
    NETHACK_SMOKE = "nethack_smoke"
    DUNGEONGRID_PLUS_PICO = "dungeongrid_plus_pico"


class GeloMaterializeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeloRunSection:
    run_id: str
    output_dir: str | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GeloContainerSection:
    url: str | None = None
    pool: ContainerPoolTarget | Mapping[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    auth_bearer_env: str | None = None
    auth_refresh: Mapping[str, Any] = field(default_factory=dict)
    startup_timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class GeloTasksetSection:
    train_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    profile: str | None = None
    backend: str | None = None
    target_achievement: str | None = None
    reward_mode: GeloRewardMode | str | None = None
    checkpoint_semantics: GeloCheckpointSemantics | str | None = None
    env_config: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeloPolicySection:
    model: str
    provider: str
    api_key_env: str
    base_url: str | None = None
    inference_url: str | None = None
    max_tokens: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeloEngineSection:
    max_rollouts: int
    proposer_rounds: int
    base_react_system_prompt: str | None = None
    fresh_rollouts_per_round: int | None = None
    resume_rollouts_per_round: int | None = None
    regression_rollouts_per_round: int | None = None
    segment_steps: int | None = None
    resume_segment_steps: int | None = None
    candidates_per_proposer: int | None = None
    heldout_measurement_rollouts: int | None = None
    holdout_consolidate_k: int | None = None
    bootstrap_train_rollout_count: int | None = None
    reserved_search_rollout_budget: int | None = None
    target_new_candidate_count: int | None = None
    min_new_candidate_count: int | None = None
    max_initial_rollouts_per_candidate: int | None = None
    min_non_baseline_candidate_fresh_rollouts: int | None = None
    auto_consolidate_min_mature_themes: int | None = None
    auto_consolidate_theme_count: int | None = None
    auto_consolidate_min_score: float | None = None
    auto_aux_consolidate_hill_climb_enabled: bool | None = None
    max_llm_turns: int | None = None
    max_actions_per_turn: int | None = None
    submission_mode: str | None = None
    execute_live_proposers: bool | None = None
    request_timeout_seconds: float | None = None
    container_connect_timeout_seconds: float = 30.0
    allow_resume_fallback_to_fresh: bool | None = None
    rollout_state_poll_seconds: float | None = None
    rollout_terminator_poll_seconds: float | None = None
    rollout_stall_timeout_seconds: float | None = None
    full_rollout_lane_enabled: bool = True
    full_rollout_budget_per_round: int | None = None
    full_rollout_initial_budget: int | None = None
    full_rollout_cadence: int | None = None
    preserve_search_measurement_split: bool = True
    theme_start_score_band: tuple[float, float] = (0.0, 1.0)
    frontier_prune_enabled: bool | None = None
    frontier_prune_soft_cap: int | None = None
    frontier_prune_retain_tail: int | None = None
    frontier_prune_hard_delete: bool | None = None
    rollout_evidence_compact_enabled: bool | None = None
    rollout_evidence_compact_retain_tail: int | None = None
    rollout_evidence_compact_mid_rollout_keep: int | None = None
    rollout_evidence_record_warn_threshold: int | None = None
    fresh_rollouts_per_parent: int | None = None
    resume_rollouts_per_parent: int | None = None
    full_rollout_concurrency: int | None = None
    theme_rollout_concurrency: int | None = None
    closeout_heldout_concurrency: int | None = None
    closeout_heldout_candidate_parallelism: int | None = None
    heldout_measurement_concurrency: int | None = None
    heldout_measurement_candidate_parallelism: int | None = None
    full_rollout_checkpoint_cadence: str | None = None
    full_rollout_checkpoint_budget: int | None = None
    data_miner_min_new_checkpoints: int | None = None
    data_miner_rollouts_per_job: int | None = None
    agent_result_max_retries: int | None = None
    agent_dispatch_concurrency: int | None = None
    agent_model_concurrency: Mapping[str, int] = field(default_factory=dict)
    agent_dispatch_stall_ceiling_seconds: int | None = None
    data_miner_cadence: str | None = None
    theme_finalize_min_checkpoints: int | None = None
    theme_partials_per_candidate: int | None = None
    theme_saturation_threshold: float | None = None
    theme_saturation_min_rollouts: int | None = None
    max_tentative_themes: int | None = None
    tentative_theme_max_age_rounds: int | None = None
    theme_proposal_round_budget: int | None = None
    theme_aux_budget_per_theme: int | None = None
    theme_no_progress_rounds: int | None = None
    theme_aux_rounds_per_staircase: int | None = None
    terminator_default: str | None = None
    promotion_min_seeds: int | None = None
    promotion_margin: float | None = None
    consolidation_budget_per_round: int | None = None
    consolidation_max_themes: int | None = None
    allow_single_theme_consolidation: bool | None = None
    min_tentative_themes_before_activation: int | None = None
    data_miner_while_active_theme_climbing: bool | None = None
    data_miner_authority: bool | None = None
    all_candidate_holdout_seed_count: int | None = None
    auto_aux_hill_climb_calls_per_round: int | None = None
    theme_eval_checkpoints: int | None = None
    agent_turn_message_stall_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class GeloSeedCandidateSection:
    react_system_prompt: str
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class GeloCacheSection:
    mode: GeloCacheMode | str = GeloCacheMode.OFF
    path: str | None = None
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class GeloDiskBudgetSection:
    enabled: bool = True
    soft_limit_gb: float = 5.0
    hard_limit_gb: float = 10.0
    path: str | None = None


@dataclass(frozen=True, slots=True)
class GeloPluginSection:
    kind: GeloPluginKind | str
    status: str = "beta"
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeloPluginsSection:
    lanes: tuple[GeloPluginSection, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GeloProposerSection:
    model: str
    provider: str
    role: str | None = None
    output_schema: str | None = None
    backend: str = "codex_app_server"
    auth_mode: str = "auto"
    timeout_seconds: int = 300
    reasoning_effort: str | None = None
    service_tier: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    runtime_substrate: str | None = None
    allow_unverified_model: bool | None = None


@dataclass(frozen=True, slots=True)
class GeloHostedConfig:
    algorithm: ClassVar[OptimizerAlgorithmSlug] = OptimizerAlgorithmSlug.GELO

    run: GeloRunSection
    container: GeloContainerSection
    taskset: GeloTasksetSection
    policy: GeloPolicySection
    go_ex: GeloEngineSection
    seed_candidate: GeloSeedCandidateSection
    proposers: Mapping[GeloProposerRole, GeloProposerSection]
    cache: GeloCacheSection | None = None
    disk_budget: GeloDiskBudgetSection | None = None
    plugins: GeloPluginsSection | Mapping[str, Any] | None = None

    def to_config_json(self) -> dict[str, Any]:
        payload = _clean_config_value(self)
        _validate_materialized_config(payload)
        return payload


_ROLE_OUTPUT_SCHEMAS: Mapping[GeloProposerRole, str] = {
    GeloProposerRole.CORE: "goex.core_proposer.output.v1",
    GeloProposerRole.AUX_HILL_CLIMB: "goex.aux_hill_climb.output.v1",
    GeloProposerRole.AUX_DATA_MINER: "goex.aux_data_miner.v1",
    GeloProposerRole.AUX_CONSOLIDATE: "goex.aux_consolidate.output.v1",
    GeloProposerRole.AUX_CONSOLIDATE_HC: "goex.aux_consolidate_hill_climb.output.v1",
    GeloProposerRole.THEME_VERIFIER: "goex_theme_verifier_result.v1",
    GeloProposerRole.TERMINATOR: "goex_terminator_decision.v1",
}

_PROPOSER_ROLES: tuple[GeloProposerRole, ...] = (
    GeloProposerRole.CORE,
    GeloProposerRole.AUX_HILL_CLIMB,
    GeloProposerRole.AUX_DATA_MINER,
    GeloProposerRole.AUX_CONSOLIDATE,
    GeloProposerRole.AUX_CONSOLIDATE_HC,
)

_AGENT_ROLES: tuple[GeloProposerRole, ...] = (
    GeloProposerRole.THEME_VERIFIER,
    GeloProposerRole.TERMINATOR,
)

_ALL_ROLES: tuple[GeloProposerRole, ...] = _PROPOSER_ROLES + _AGENT_ROLES

_GO_EX_FLAT_KEYS: frozenset[str] = frozenset(
    field.name for field in fields(GeloEngineSection)
)

_TASKSET_KEYS: frozenset[str] = frozenset(
    {
        "profile",
        "train_seeds",
        "heldout_seeds",
        "env_config",
        "reward_mode",
        "target_achievement",
        "backend",
        "context",
    }
)

_POLICY_KEYS: Mapping[str, str] = {
    "policy_model": "model",
    "policy_provider": "provider",
    "policy_base_url": "base_url",
    "policy_inference_url": "inference_url",
    "policy_api_key_env": "api_key_env",
    "policy_max_completion_tokens": "max_tokens",
    "policy_max_tokens": "max_tokens",
    "policy_config": "config",
}

_ROLE_FIELD_KEYS: Mapping[str, str] = {
    "model": "{role}_model",
    "provider": "{role}_provider",
    "reasoning_effort": "{role}_reasoning_effort",
    "timeout_seconds": "{role}_timeout_seconds",
    "backend": "{role}_backend",
    "api_key_env": "{role}_api_key_env",
    "base_url": "{role}_base_url",
    "auth_mode": "{role}_auth_mode",
    "service_tier": "{role}_service_tier",
    "allow_unverified_model": "{role}_allow_unverified_model",
}

_LEGACY_BACKEND_ALIASES: frozenset[str] = frozenset({"codex_workspace", "workspace"})


@dataclass(frozen=True, slots=True)
class GeloPreset:
    name: GeloPresetName
    proposer_rounds: int = 3
    train_seed_count: int = 8
    heldout_seed_count: int = 8
    max_rollouts: int = 32
    policy_model: str = "gemini-3.1-flash-lite"
    policy_provider: str = "gemini"
    policy_api_key_env: str = "GEMINI_API_KEY"

    @classmethod
    def from_name(cls, name: GeloPresetName | str, **overrides: Any) -> "GeloPreset":
        try:
            preset_name = GeloPresetName(str(name))
        except ValueError as exc:
            raise GeloMaterializeError(f"unknown GELO preset {name!r}") from exc
        if preset_name == GeloPresetName.CRAFTER_SMOKE:
            defaults = {
                "proposer_rounds": 1,
                "train_seed_count": 4,
                "heldout_seed_count": 2,
                "max_rollouts": 80,
            }
        elif preset_name == GeloPresetName.CRAFTER:
            defaults = {
                "proposer_rounds": 3,
                "train_seed_count": 8,
                "heldout_seed_count": 8,
                "max_rollouts": 6000,
            }
        elif preset_name == GeloPresetName.SOKOBAN_SMOKE:
            # GameBench rust gold Sokoban (tasks/sokoban-singleplayer). Mirrors the
            # proven local config goex_sokoban_gpt_oss_120b_budget.json, scaled for a
            # hosted smoke. Policy runs Groq gpt-oss-120b inside the react container;
            # proposers are Synth-managed (same as crafter_smoke).
            defaults = {
                "proposer_rounds": 1,
                "train_seed_count": 4,
                "heldout_seed_count": 2,
                "max_rollouts": 48,
                "policy_model": "openai/gpt-oss-120b",
                "policy_provider": "groq",
                "policy_api_key_env": "GROQ_API_KEY",
            }
        else:
            raise GeloMaterializeError(
                f"preset {preset_name.value!r} is not available in the public package yet"
            )
        defaults.update(overrides)
        return cls(name=preset_name, **defaults)

    @classmethod
    def crafter_smoke(cls, **overrides: Any) -> "GeloPreset":
        return cls.from_name(GeloPresetName.CRAFTER_SMOKE, **overrides)

    @classmethod
    def crafter(cls, **overrides: Any) -> "GeloPreset":
        return cls.from_name(GeloPresetName.CRAFTER, **overrides)

    @classmethod
    def sokoban_smoke(cls, **overrides: Any) -> "GeloPreset":
        return cls.from_name(GeloPresetName.SOKOBAN_SMOKE, **overrides)

    def to_config(
        self,
        *,
        container_url: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: Any | None = None,
        run_id: str | None = None,
    ) -> GeloHostedConfig:
        _validate_container_inputs(
            container_url=container_url,
            container_pool=container_pool,
            container_tunnel=container_tunnel,
        )
        if container_tunnel is not None:
            if not hasattr(container_tunnel, "container_config"):
                raise GeloMaterializeError("container_tunnel must expose container_config()")
            container = _container_section_from_tunnel(container_tunnel)
        else:
            container = GeloContainerSection(url=container_url, pool=container_pool)
        if not container.url and _container_pool_payload(container.pool) is None:
            raise GeloMaterializeError(
                "container_url, container_pool, or container_tunnel is required"
            )
        run = GeloRunSection(run_id=run_id or _default_run_id(self.name))
        if self.name == GeloPresetName.SOKOBAN_SMOKE:
            config = _sokoban_config(self, run=run, container=container)
        else:
            config = GeloHostedConfig(
                run=run,
                container=container,
                taskset=GeloTasksetSection(
                    train_seeds=_seed_tuple(3001, self.train_seed_count),
                    heldout_seeds=_seed_tuple(7001, self.heldout_seed_count),
                    profile="crafter_react",
                    reward_mode=GeloRewardMode.ACHIEVEMENT,
                    env_config={"task_family": "crafter_react"},
                ),
                policy=GeloPolicySection(
                    model=self.policy_model,
                    provider=self.policy_provider,
                    api_key_env=self.policy_api_key_env,
                    inference_url=(
                        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                        if self.policy_provider == "gemini"
                        else None
                    ),
                ),
                go_ex=_crafter_engine(self),
                seed_candidate=GeloSeedCandidateSection(
                    react_system_prompt=(
                        "You are a Crafter ReAct policy. Explore safely, collect resources, "
                        "craft tools when prerequisites are present, and use only valid "
                        "crafter_interact actions."
                    )
                ),
                proposers=_default_crafter_roles(),
                cache=GeloCacheSection(mode=GeloCacheMode.OFF),
                disk_budget=GeloDiskBudgetSection(
                    enabled=True, soft_limit_gb=5.0, hard_limit_gb=10.0
                ),
            )
        _validate_materialized_config(config.to_config_json())
        return config

    def materialize(
        self,
        *,
        container_url: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: Any | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return self.to_config(
            container_url=container_url,
            container_pool=container_pool,
            container_tunnel=container_tunnel,
            run_id=run_id,
        ).to_config_json()


@dataclass(frozen=True, slots=True)
class GeloMaterializer:
    config: Mapping[str, Any]

    @staticmethod
    def from_paths(
        toml: Path | str,
        overlay: Path | str | None = None,
    ) -> "GeloMaterializer":
        base_path = Path(toml)
        base = _load_config_document(base_path)
        overlay_payload = _load_config_document(Path(overlay)) if overlay is not None else None
        if _is_launcher_config(base):
            overlay_config = (
                _normalize_config_document(overlay_payload, source=Path(overlay))
                if overlay_payload is not None
                else {}
            )
            launcher_config = _structured_from_launcher(base, base_path)
            config = _deep_merge(overlay_config, launcher_config)
        else:
            config = _normalize_config_document(base, source=base_path)
            if overlay_payload is not None:
                config = _deep_merge(
                    config,
                    _normalize_config_document(overlay_payload, source=Path(overlay)),
                )
        return GeloMaterializer(config=config)

    def materialize(
        self,
        *,
        container_url: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: Any | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        _validate_container_inputs(
            container_url=container_url,
            container_pool=container_pool,
            container_tunnel=container_tunnel,
        )
        config = dict(self.config)
        if run_id is not None:
            run = dict(_mapping(config.get("run")))
            run["run_id"] = run_id
            config["run"] = run
        if container_tunnel is not None:
            if not hasattr(container_tunnel, "container_config"):
                raise GeloMaterializeError("container_tunnel must expose container_config()")
            container = _clean_config_value(
                _container_section_from_tunnel(container_tunnel)
            )
            config["container"] = _deep_merge(_mapping(config.get("container")), container)
        if container_url is not None:
            container = dict(_mapping(config.get("container")))
            container["url"] = container_url
            config["container"] = container
        if container_pool is not None:
            container = dict(_mapping(config.get("container")))
            container["pool"] = (
                container_pool.to_payload()
                if isinstance(container_pool, ContainerPoolTarget)
                else dict(container_pool)
            )
            config["container"] = container
        _validate_materialized_config(config)
        return _clean_config_value(config)


def _seed_tuple(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + max(1, int(count))))


def _default_run_id(name: GeloPresetName) -> str:
    return f"goex_{name.value}_{uuid.uuid4().hex[:12]}"


def _crafter_engine(preset: GeloPreset) -> GeloEngineSection:
    smoke = preset.name == GeloPresetName.CRAFTER_SMOKE
    return GeloEngineSection(
        max_rollouts=preset.max_rollouts,
        proposer_rounds=preset.proposer_rounds,
        submission_mode="sync",
        execute_live_proposers=True,
        bootstrap_train_rollout_count=2,
        reserved_search_rollout_budget=2 if smoke else 4,
        heldout_measurement_rollouts=max(1, preset.heldout_seed_count),
        all_candidate_holdout_seed_count=max(1, preset.heldout_seed_count),
        closeout_heldout_concurrency=8 if smoke else 50,
        closeout_heldout_candidate_parallelism=2 if smoke else 6,
        heldout_measurement_concurrency=8 if smoke else 50,
        heldout_measurement_candidate_parallelism=2 if smoke else 6,
        agent_dispatch_concurrency=2 if smoke else 16,
        agent_model_concurrency={"gpt-5.4-mini": 2, "deepseek-v4-flash": 4}
        if smoke
        else {"deepseek-v4-flash": 16},
        request_timeout_seconds=180.0,
        rollout_state_poll_seconds=0.5,
        rollout_terminator_poll_seconds=0.5,
        rollout_stall_timeout_seconds=240.0,
        max_llm_turns=6,
        max_actions_per_turn=10,
        segment_steps=100,
        resume_segment_steps=60,
        full_rollout_lane_enabled=True,
        data_miner_authority=False if smoke else True,
        full_rollout_initial_budget=8 if smoke else 50,
        full_rollout_budget_per_round=0 if smoke else 45,
        full_rollout_cadence=1,
        fresh_rollouts_per_parent=0,
        full_rollout_concurrency=8 if smoke else 50,
        theme_rollout_concurrency=8 if smoke else 50,
        allow_resume_fallback_to_fresh=False,
        resume_rollouts_per_parent=1 if smoke else 2,
        full_rollout_checkpoint_cadence="per_llm_call",
        full_rollout_checkpoint_budget=200 if smoke else 1200,
        frontier_prune_enabled=True,
        frontier_prune_soft_cap=200 if smoke else 800,
        frontier_prune_retain_tail=80 if smoke else 200,
        frontier_prune_hard_delete=False,
        data_miner_min_new_checkpoints=1,
        data_miner_rollouts_per_job=4 if smoke else 15,
        data_miner_cadence="after_full_rollout_phase",
        theme_finalize_min_checkpoints=2 if smoke else 3,
        max_tentative_themes=3 if smoke else 5,
        tentative_theme_max_age_rounds=1 if smoke else 3,
        theme_start_score_band=(0.0, 0.9),
        theme_eval_checkpoints=0,
        theme_partials_per_candidate=1 if smoke else 2,
        theme_saturation_threshold=0.9,
        theme_saturation_min_rollouts=2,
        theme_aux_rounds_per_staircase=1 if smoke else 2,
        theme_proposal_round_budget=1 if smoke else 2,
        theme_aux_budget_per_theme=1 if smoke else 4,
        theme_no_progress_rounds=1 if smoke else 2,
        terminator_default="agent",
        promotion_min_seeds=2,
        promotion_margin=0.05,
        holdout_consolidate_k=2 if smoke else 4,
        auto_aux_hill_climb_calls_per_round=1 if smoke else 2,
        auto_consolidate_min_mature_themes=2,
        auto_consolidate_theme_count=2 if smoke else 5,
        auto_consolidate_min_score=0.0,
        consolidation_budget_per_round=1 if smoke else 4,
        consolidation_max_themes=2 if smoke else 5,
        allow_single_theme_consolidation=False,
        min_tentative_themes_before_activation=1,
        data_miner_while_active_theme_climbing=True,
        target_new_candidate_count=2,
        candidates_per_proposer=3 if smoke else 5,
    )


_SOKOBAN_SEED_PROMPT = (
    "You are playing Sokoban. Reply with exactly one move as JSON: "
    '{"action":"up"}, {"action":"down"}, {"action":"left"}, or {"action":"right"}. '
    "Push boxes onto targets. Avoid pushing boxes into walls or corners unless that "
    "move places the box on a target."
)

# Mirrors goex_sokoban_gpt_oss_120b_budget.json taskset.context.milestone_ladder.
_SOKOBAN_MILESTONE_LADDER: tuple[Mapping[str, Any], ...] = (
    {"milestone_id": "first_push", "title": "Make the first useful box push", "reward": 0.2, "region": {}},
    {
        "milestone_id": "box_on_target",
        "title": "Move a box onto a target",
        "reward": 0.5,
        "region": {"requires_milestone": "first_push"},
    },
    {
        "milestone_id": "level_complete",
        "title": "Solve the Sokoban puzzle",
        "reward": 1.0,
        "region": {"requires_milestone": "box_on_target"},
    },
)


def _sokoban_engine(preset: "GeloPreset") -> GeloEngineSection:
    """GameBench Sokoban smoke engine — scaled-down mirror of the proven budget config."""
    return GeloEngineSection(
        max_rollouts=preset.max_rollouts,
        proposer_rounds=preset.proposer_rounds,
        base_react_system_prompt=_SOKOBAN_SEED_PROMPT,
        submission_mode="sync",
        execute_live_proposers=True,
        bootstrap_train_rollout_count=0,
        fresh_rollouts_per_round=4,
        resume_rollouts_per_round=0,
        heldout_measurement_rollouts=max(1, preset.heldout_seed_count),
        all_candidate_holdout_seed_count=max(1, preset.heldout_seed_count),
        full_rollout_lane_enabled=True,
        full_rollout_initial_budget=6,
        full_rollout_budget_per_round=4,
        full_rollout_cadence=1,
        fresh_rollouts_per_parent=1,
        full_rollout_concurrency=3,
        theme_rollout_concurrency=3,
        full_rollout_checkpoint_cadence="per_llm_call",
        full_rollout_checkpoint_budget=160,
        data_miner_authority=True,
        data_miner_min_new_checkpoints=1,
        data_miner_rollouts_per_job=2,
        data_miner_cadence="after_full_rollout_phase",
        theme_finalize_min_checkpoints=1,
        max_tentative_themes=5,
        tentative_theme_max_age_rounds=3,
        theme_start_score_band=(0.0, 0.95),
        theme_partials_per_candidate=3,
        theme_saturation_threshold=0.85,
        theme_saturation_min_rollouts=1,
        theme_aux_rounds_per_staircase=3,
        theme_proposal_round_budget=3,
        theme_aux_budget_per_theme=3,
        theme_no_progress_rounds=3,
        terminator_default="agent",
        promotion_min_seeds=1,
        promotion_margin=0.01,
        holdout_consolidate_k=2,
        auto_aux_hill_climb_calls_per_round=3,
        auto_consolidate_min_mature_themes=1,
        auto_consolidate_theme_count=2,
        auto_consolidate_min_score=0.0,
        allow_single_theme_consolidation=True,
        consolidation_budget_per_round=2,
        consolidation_max_themes=3,
        target_new_candidate_count=4,
        candidates_per_proposer=3,
        segment_steps=20,
        resume_segment_steps=10,
        max_llm_turns=20,
        max_actions_per_turn=1,
        request_timeout_seconds=240.0,
        rollout_state_poll_seconds=0.25,
        rollout_terminator_poll_seconds=0.25,
        rollout_stall_timeout_seconds=300.0,
        container_connect_timeout_seconds=15.0,
    )


def _sokoban_config(
    preset: "GeloPreset",
    *,
    run: GeloRunSection,
    container: GeloContainerSection,
) -> GeloHostedConfig:
    return GeloHostedConfig(
        run=run,
        container=container,
        taskset=GeloTasksetSection(
            train_seeds=_seed_tuple(101, preset.train_seed_count),
            heldout_seeds=_seed_tuple(201, preset.heldout_seed_count),
            profile="sokoban_singleplayer_agent",
            reward_mode="sokoban_sparse_shaped",
            checkpoint_semantics=GeloCheckpointSemantics.TRUE_ENV_SNAPSHOT,
            # Target is conveyed via context.target_objective_label + the milestone
            # ladder (matches the proven goex_sokoban_gpt_oss_120b_budget.json, which
            # sets no top-level target_achievement — that field is crafter-shaped).
            env_config={
                "task_path": "tasks/difficulty/gold_08_double_push.json",
                "max_steps": 20,
            },
            context={
                "task_family": "sokoban_singleplayer",
                "checkpoint_restore_semantics": "true_environment_snapshot",
                "allow_request_snapshot_resume": False,
                "require_real_rewards": True,
                "required_candidate_kind": "prompt",
                "prompt_only_candidate_authoring": True,
                "forbid_code_policy_candidates": True,
                "target_objective_label": "level_complete",
                "milestone_ladder": [dict(m) for m in _SOKOBAN_MILESTONE_LADDER],
            },
        ),
        policy=GeloPolicySection(
            model=preset.policy_model,
            provider=preset.policy_provider,
            api_key_env=preset.policy_api_key_env,
            inference_url="https://api.groq.com/openai/v1/chat/completions",
            max_tokens=512,
            config={
                "use_lm": True,
                "policy_id": "greedy_distance_v1",
                "temperature": 0.0,
                "max_steps": 20,
            },
        ),
        go_ex=_sokoban_engine(preset),
        seed_candidate=GeloSeedCandidateSection(react_system_prompt=_SOKOBAN_SEED_PROMPT),
        proposers=_default_crafter_roles(),
        cache=GeloCacheSection(mode=GeloCacheMode.OFF),
        disk_budget=GeloDiskBudgetSection(enabled=True, soft_limit_gb=5.0, hard_limit_gb=10.0),
    )


def _default_crafter_roles() -> Mapping[GeloProposerRole, GeloProposerSection]:
    roles: dict[GeloProposerRole, GeloProposerSection] = {}
    for role in _PROPOSER_ROLES:
        roles[role] = GeloProposerSection(
            role=role.value,
            output_schema=_ROLE_OUTPUT_SCHEMAS[role],
            model="gpt-5.4-mini",
            provider="openai",
            backend="codex_app_server",
            auth_mode="auto",
            reasoning_effort="high",
            timeout_seconds=300,
        )
    roles[GeloProposerRole.THEME_VERIFIER] = GeloProposerSection(
        role=GeloProposerRole.THEME_VERIFIER.value,
        output_schema=_ROLE_OUTPUT_SCHEMAS[GeloProposerRole.THEME_VERIFIER],
        model="deepseek-v4-flash",
        provider="deepseek",
        backend="deepseek_chat",
        auth_mode="api_key",
        reasoning_effort="low",
        timeout_seconds=120,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    )
    roles[GeloProposerRole.TERMINATOR] = GeloProposerSection(
        role=GeloProposerRole.TERMINATOR.value,
        output_schema=_ROLE_OUTPUT_SCHEMAS[GeloProposerRole.TERMINATOR],
        model="deepseek-v4-flash",
        provider="deepseek",
        backend="deepseek_chat",
        auth_mode="api_key",
        reasoning_effort="low",
        timeout_seconds=60,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    )
    return roles


def _load_config_document(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise GeloMaterializeError(f"cannot read GELO config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeloMaterializeError(f"GELO config {path} must contain an object")
    return payload


def _is_launcher_config(payload: Mapping[str, Any]) -> bool:
    return "launcher" in payload or "materialize" in payload


def _normalize_config_document(
    payload: Mapping[str, Any] | None,
    *,
    source: Path | None,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if "extras" in payload and isinstance(payload.get("extras"), Mapping):
        return _structured_from_flat(dict(payload["extras"]))
    if _is_launcher_config(payload):
        return _structured_from_launcher(payload, source)
    return _clean_config_value(payload)


def _structured_from_launcher(payload: Mapping[str, Any], source: Path | None) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    launcher = _mapping(payload.get("launcher"))
    materialize = _mapping(payload.get("materialize"))
    if launcher.get("service_url"):
        flat["service_url"] = launcher["service_url"]
    for key in (
        "bootstrap_train_rollout_count",
        "proposer_rounds",
        "heldout_measurement_rollouts",
    ):
        if key in materialize:
            flat[key] = materialize[key]
    for count_key, seeds_key, start in (
        ("train_seed_count", "train_seeds", 3001),
        ("heldout_seed_count", "heldout_seeds", 7001),
    ):
        if count_key in materialize and seeds_key not in flat:
            flat[seeds_key] = list(_seed_tuple(start, int(materialize[count_key])))
    policy = _mapping(payload.get("policy"))
    if policy:
        for key, value in policy.items():
            flat[f"policy_{key}"] = value
    proposers = _mapping(payload.get("proposers"))
    defaults = _mapping(proposers.get("defaults"))
    for role in _ALL_ROLES:
        role_payload = dict(defaults if role in _PROPOSER_ROLES else {})
        role_payload.update(_mapping(proposers.get(role.value)))
        for key, value in role_payload.items():
            flat[f"{role.value}_{key}"] = value
    try:
        return _structured_from_flat(flat)
    except GeloMaterializeError as exc:
        location = f" in {source}" if source is not None else ""
        raise GeloMaterializeError(f"invalid GELO launcher config{location}: {exc}") from exc


def _structured_from_flat(flat: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    run = {
        key: flat[key]
        for key in ("run_id", "output_dir", "seed")
        if key in flat
    }
    if run:
        result["run"] = run
    service_url = flat.get("service_url") or flat.get("container_url")
    if service_url:
        result["container"] = {"url": service_url}
    taskset = {key: flat[key] for key in _TASKSET_KEYS if key in flat}
    if taskset:
        result["taskset"] = taskset
    policy: dict[str, Any] = {}
    for flat_key, structured_key in _POLICY_KEYS.items():
        if flat_key in flat:
            policy[structured_key] = flat[flat_key]
    if policy:
        result["policy"] = policy
    go_ex = {}
    for key in _GO_EX_FLAT_KEYS:
        if key in flat:
            value = flat[key]
            if key == "data_miner_cadence" and isinstance(value, int):
                value = "every_tick" if value <= 0 else "after_full_rollout_phase"
            go_ex[key] = value
    base_prompt = flat.get("base_prompt")
    if base_prompt is not None and "base_react_system_prompt" not in go_ex:
        go_ex["base_react_system_prompt"] = base_prompt
    if go_ex:
        result["go_ex"] = go_ex
    seed_candidate = flat.get("seed_candidate")
    if isinstance(seed_candidate, Mapping):
        result["seed_candidate"] = dict(seed_candidate)
    else:
        seed_values = {
            key: flat[key]
            for key in ("react_system_prompt", "system_prompt")
            if isinstance(flat.get(key), str)
        }
        if seed_values:
            result["seed_candidate"] = seed_values
    proposers = {}
    for role in _ALL_ROLES:
        proposers[role.value] = _role_from_flat(flat, role)
    if proposers:
        result["proposers"] = proposers
    for key in ("plugins", "cache", "disk_budget"):
        value = flat.get(key)
        if isinstance(value, Mapping):
            result[key] = dict(value)
    return result


def _role_from_flat(flat: Mapping[str, Any], role: GeloProposerRole) -> dict[str, Any]:
    config: dict[str, Any] = {
        "role": role.value,
        "output_schema": _ROLE_OUTPUT_SCHEMAS[role],
    }
    for source_key, pattern in _ROLE_FIELD_KEYS.items():
        value = flat.get(pattern.format(role=role.value))
        if value is None and role in _PROPOSER_ROLES:
            value = flat.get(pattern.format(role="proposer"))
            if value is None:
                value = flat.get(pattern.format(role="codex_proposer"))
        if value is None:
            continue
        if source_key == "backend" and str(value).strip() in _LEGACY_BACKEND_ALIASES:
            raise GeloMaterializeError(
                f"GELO proposer {role.value} backend {value!r} is retired; "
                "use 'codex_app_server'"
            )
        config[source_key] = value
    return config


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(_mapping(merged[key]), value)
        else:
            merged[key] = _clean_config_value(value)
    return merged


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_mapping(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _container_payload_from_target(target: Any) -> Mapping[str, Any]:
    if hasattr(target, "to_config_json"):
        payload = target.to_config_json()
    else:
        payload = _clean_config_value(target)
    if not isinstance(payload, Mapping):
        raise GeloMaterializeError("container_tunnel.container_config() must serialize to an object")
    return payload


def _container_section_from_target(target: Any) -> GeloContainerSection:
    payload = _container_payload_from_target(target)
    raw_pool = payload.get("pool")
    pool = _container_pool_payload(raw_pool)
    raw_auth_refresh = payload.get("auth_refresh")
    if raw_auth_refresh is not None and not isinstance(raw_auth_refresh, Mapping):
        raise GeloMaterializeError("container auth_refresh must be an object")
    startup_timeout = payload.get("startup_timeout_seconds")
    try:
        timeout = int(startup_timeout) if startup_timeout is not None else 30
    except (TypeError, ValueError) as exc:
        raise GeloMaterializeError("container startup_timeout_seconds must be an integer") from exc
    return GeloContainerSection(
        url=_text_or_none(payload.get("url")),
        pool=pool,
        headers=_string_mapping(payload.get("headers")),
        auth_bearer_env=_text_or_none(payload.get("auth_bearer_env")),
        auth_refresh=dict(raw_auth_refresh) if isinstance(raw_auth_refresh, Mapping) else {},
        startup_timeout_seconds=timeout,
    )


def _container_section_from_tunnel(container_tunnel: Any) -> GeloContainerSection:
    section = _container_section_from_target(container_tunnel.container_config())
    provider = tunnel_provider_value(getattr(container_tunnel, "provider", None))
    if provider != TunnelProvider.SYNTH_TUNNEL.value:
        return section
    headers = {
        name: value
        for name, value in section.headers.items()
        if name.strip().lower() not in {"authorization", "x-api-key", "x-api-keys"}
    }
    lease_id = _text_or_none(getattr(container_tunnel, "lease_id", None))
    if lease_id is None:
        raise GeloMaterializeError("SynthTunnel container_tunnel must expose lease_id")
    return replace(
        section,
        headers=headers,
        auth_refresh={
            "provider": "synth_tunnel",
            "lease_id": lease_id,
            "refresh_interval_seconds": 900,
        },
    )


def _container_pool_payload(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ContainerPoolTarget):
        return value.to_payload()
    if isinstance(value, Mapping):
        return dict(value)
    raise GeloMaterializeError("container pool must be an object")


def _validate_container_inputs(
    *,
    container_url: str | None,
    container_pool: ContainerPoolTarget | Mapping[str, Any] | None,
    container_tunnel: Any | None,
) -> None:
    sources = [
        bool(str(container_url or "").strip()),
        container_pool is not None,
        container_tunnel is not None,
    ]
    if sum(1 for present in sources if present) > 1:
        raise GeloMaterializeError(
            "container_url, container_pool, and container_tunnel are mutually exclusive"
        )


_GELO_SUPPORTED_PLUGIN_KINDS = frozenset({GeloPluginKind.SFT.value})
_GELO_FUTURE_PLUGIN_KINDS = frozenset(
    {GeloPluginKind.RLVR.value, GeloPluginKind.OPSD.value}
)
_GELO_PLUGIN_KIND_KEYS = frozenset({"kind", "type", "plugin_kind"})
_GELO_PLUGIN_COLLECTION_KEYS = frozenset({"lanes", "items"})
_GELO_PLUGIN_METADATA_KEYS = frozenset({"enabled", "status", "version", "config"})


def _plugin_key(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _validate_plugin_kind(kind: Any, *, path: str) -> None:
    normalized = _plugin_key(kind)
    if normalized in _GELO_SUPPORTED_PLUGIN_KINDS:
        return
    if normalized in _GELO_FUTURE_PLUGIN_KINDS:
        raise GeloMaterializeError(
            f"GELO plugin lane {normalized!r} is not supported yet; "
            "only 'sft' is accepted as a beta plugin lane"
        )
    raise GeloMaterializeError(
        f"GELO plugin lane {kind!r} at {path} is unsupported; "
        "only 'sft' is accepted as a beta plugin lane"
    )


def _validate_plugin_declaration(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise GeloMaterializeError(f"GELO plugin declaration at {path} must be an object")
    for key in _GELO_PLUGIN_KIND_KEYS:
        if key in value:
            _validate_plugin_kind(value[key], path=f"{path}.{key}")
            return
    raise GeloMaterializeError(
        f"GELO plugin declaration at {path} must include kind='sft'"
    )


def _validate_plugin_collection(value: Any, *, path: str) -> None:
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            _validate_plugin_declaration(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, Mapping):
        raise GeloMaterializeError(f"GELO plugins section at {path} must be an object")
    if any(key in value for key in _GELO_PLUGIN_KIND_KEYS):
        _validate_plugin_declaration(value, path=path)
        return
    for key, item in value.items():
        normalized_key = _plugin_key(key)
        if normalized_key in _GELO_SUPPORTED_PLUGIN_KINDS | _GELO_FUTURE_PLUGIN_KINDS:
            _validate_plugin_kind(normalized_key, path=f"{path}.{key}")
        elif normalized_key in _GELO_PLUGIN_COLLECTION_KEYS:
            _validate_plugin_collection(item, path=f"{path}.{key}")
        elif normalized_key in _GELO_PLUGIN_METADATA_KEYS:
            continue
        else:
            raise GeloMaterializeError(
                f"GELO plugin key {key!r} at {path} is unsupported; "
                "use plugins.lanes=[{kind='sft'}] or plugins.sft"
            )


def _validate_plugin_lanes(config: Mapping[str, Any]) -> None:
    plugins = config.get("plugins")
    if plugins is None:
        return
    _validate_plugin_collection(plugins, path="plugins")


def _validate_materialized_config(config: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in ("run", "container", "taskset", "policy", "go_ex", "seed_candidate", "proposers")
        if not isinstance(config.get(key), Mapping)
    ]
    if missing:
        raise GeloMaterializeError(f"GELO config missing object section(s): {', '.join(missing)}")
    run = _mapping(config.get("run"))
    if not str(run.get("run_id") or "").strip():
        raise GeloMaterializeError("GELO config requires run.run_id")
    container = _mapping(config.get("container"))
    has_url = bool(str(container.get("url") or "").strip())
    pool = _container_pool_payload(container.get("pool"))
    has_pool = bool(str(_mapping(pool).get("pool_id") or "").strip())
    if has_url and has_pool:
        raise GeloMaterializeError("GELO config must not set both container.url and container.pool")
    if not has_url and not has_pool:
        raise GeloMaterializeError("GELO config requires container.url or container.pool.pool_id")
    raw_auth_refresh = container.get("auth_refresh")
    if raw_auth_refresh is not None:
        if not isinstance(raw_auth_refresh, Mapping):
            raise GeloMaterializeError("GELO config container.auth_refresh must be an object")
        if str(container.get("auth_bearer_env") or "").strip():
            raise GeloMaterializeError(
                "GELO config must not combine container.auth_refresh and auth_bearer_env"
            )
        provider = str(raw_auth_refresh.get("provider") or "").strip()
        if provider != "synth_tunnel":
            raise GeloMaterializeError(
                "GELO config container.auth_refresh.provider must be synth_tunnel"
            )
        if not str(raw_auth_refresh.get("lease_id") or "").strip():
            raise GeloMaterializeError(
                "GELO config container.auth_refresh.lease_id is required"
            )
    taskset = _mapping(config.get("taskset"))
    if not taskset.get("train_seeds") or not taskset.get("heldout_seeds"):
        raise GeloMaterializeError("GELO config requires taskset.train_seeds and heldout_seeds")
    go_ex = _mapping(config.get("go_ex"))
    if int(go_ex.get("max_rollouts") or 0) <= 0:
        raise GeloMaterializeError("GELO config requires go_ex.max_rollouts > 0")
    if int(go_ex.get("proposer_rounds") or 0) <= 0:
        raise GeloMaterializeError("GELO config requires go_ex.proposer_rounds > 0")
    _validate_plugin_lanes(config)
    proposers = _mapping(config.get("proposers"))
    for role_name, role_config in proposers.items():
        if not isinstance(role_config, Mapping):
            raise GeloMaterializeError(f"GELO proposer {role_name} config must be an object")
        backend = str(role_config.get("backend") or "").strip()
        if backend in _LEGACY_BACKEND_ALIASES:
            raise GeloMaterializeError(
                f"GELO proposer {role_name} backend {backend!r} is retired; "
                "use 'codex_app_server'"
            )


def _clean_config_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, ContainerPoolTarget):
        return value.to_payload()
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for dataclass_field in fields(value):
            item = getattr(value, dataclass_field.name)
            if item is None:
                continue
            cleaned = _clean_config_value(item)
            if cleaned == {} or cleaned == []:
                continue
            result[dataclass_field.name] = cleaned
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned = _clean_config_value(item)
            if cleaned == {} or cleaned == []:
                continue
            result[str(_clean_config_value(key))] = cleaned
        return result
    if isinstance(value, tuple | list):
        return [_clean_config_value(item) for item in value]
    return value
