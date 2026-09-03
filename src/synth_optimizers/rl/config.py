"""The run configuration: one TOML document, validated before anything runs.

This is the ``cispo.container.v1`` surface from the design note. It carries the
container connection, the taskset selection, the model identity, the algorithm
plan, the pipeline bounds, the declared topology binding, the opponent set, the
reward channel, the evaluation shape, the lifecycle rules, the offline mode and
the artifact policy. Nothing else.

Three rules make this file load-bearing rather than decorative:

* **An unknown key is refused, never ignored.** A typo in a bound is a run that
  quietly does something else, which is the failure mode this plane exists to
  eliminate.
* **A container concern is refused by name.** Environment, harness,
  renderer-selection and reward-mode fields belong to the container's own
  declaration. An executor that could pick a renderer could disagree with the
  container about token identity, and the handshake would have nothing to
  compare.
* **Startup invariants are checked here, not at the first failure.** The
  pipeline may not hold more lag than the staleness bound tolerates, and a run
  may not ask for more provider train calls than the plan's step ceiling
  allows.

The algorithm arrives as ``[plan]``: a preset name plus explicit dimension
overrides. There is no section named after an algorithm, and no field in this
file selects a code path by algorithm name -- ``preset = "cispo"`` and
``preset = "gspo"`` differ only in the plan they expand to.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import plan as plan_module
from .plan import AlgorithmPlan

CONFIG_SCHEMA_VERSION = "cispo.container.v1"

#: Field names that name a container concern. The executor never selects an
#: environment, a harness, a renderer or a reward rule: it binds what the
#: container declares, and the handshake is where the two are compared.
CONTAINER_CONCERN_FIELDS: Mapping[str, str] = {
    "environment": "the container declares its environment; the executor binds it",
    "environment_id": "the container declares its environment; the executor binds it",
    "env": "the container declares its environment; the executor binds it",
    "env_id": "the container declares its environment; the executor binds it",
    "harness": "the harness lives inside the container",
    "harness_id": "the harness lives inside the container",
    "agent": "the harness lives inside the container",
    "renderer": "the renderer profile is declared by the container and matched, not chosen",
    "renderer_id": "the renderer profile is declared by the container and matched, not chosen",
    "renderer_profile": "the renderer profile is declared by the container and matched",
    "renderer_selection": "the renderer profile is declared by the container and matched",
    "reward_mode": "the reward rule is the container's authority",
    "reward_kind": "the reward rule is the container's authority",
    "reward_fn": "the reward rule is the container's authority",
    "scorer": "the reward rule is the container's authority",
    "task": "tasks are rows in the container's taskset, named by id",
    "image": "a launcher resolves an image into a URL; the executor sees the URL",
}

#: Dimension keys ``[plan]`` may override, mirroring :mod:`.plan`.
PLAN_DIMENSIONS: tuple[str, ...] = (
    "rollout",
    "credit",
    "objective",
    "correction",
    "reducer",
    "schedule",
)

PIPELINE_MODES: tuple[str, ...] = ("async_queued", "synchronous")
OFFLINE_MODES: tuple[str, ...] = ("off", "replay")
PARTIAL_ROSTER: tuple[str, ...] = ("refuse", "drop_instance", "refuse_team")
STALE_DISPOSITIONS: tuple[str, ...] = ("discard", "recycle")


class ConfigError(ValueError):
    """The configuration was refused. Loading never repairs a document."""


# --------------------------------------------------------------------------- #
# Reading helpers -- every one of them refuses rather than defaults silently
# --------------------------------------------------------------------------- #


def _reject_container_concerns(section: str, payload: Mapping[str, Any]) -> None:
    for key in payload:
        reason = CONTAINER_CONCERN_FIELDS.get(str(key).lower())
        if reason is not None:
            raise ConfigError(
                f"[{section}] declares {key!r}, which is a container concern: {reason}"
            )


class _Reader:
    """One section, read exhaustively. Whatever is left over is an error."""

    def __init__(self, section: str, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ConfigError(f"[{section}] must be a table")
        _reject_container_concerns(section, payload)
        self.section = section
        self._payload = dict(payload)
        self._seen: set[str] = set()

    def _take(self, name: str, default: Any) -> Any:
        self._seen.add(name)
        return self._payload.get(name, default)

    def text(self, name: str, default: str | None = None) -> str:
        value = self._take(name, default)
        if value is None or not isinstance(value, str) or not value.strip():
            raise ConfigError(f"[{self.section}] {name} must be a non-empty string")
        return value.strip()

    def optional_text(self, name: str, default: str | None = None) -> str | None:
        value = self._take(name, default)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"[{self.section}] {name} must be a non-empty string when present")
        return value.strip()

    def choice(self, name: str, allowed: Sequence[str], default: str | None = None) -> str:
        value = self.text(name, default)
        if value not in allowed:
            raise ConfigError(
                f"[{self.section}] {name}={value!r} is not one of {sorted(allowed)}"
            )
        return value

    def flag(self, name: str, default: bool) -> bool:
        value = self._take(name, default)
        if not isinstance(value, bool):
            raise ConfigError(f"[{self.section}] {name} must be true or false")
        return value

    def count(self, name: str, default: int | None = None, *, minimum: int = 1) -> int:
        value = self._take(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"[{self.section}] {name} must be an integer")
        if value < minimum:
            raise ConfigError(f"[{self.section}] {name} must be at least {minimum}")
        return value

    def optional_count(self, name: str, *, minimum: int = 1) -> int | None:
        value = self._take(name, None)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"[{self.section}] {name} must be an integer when present")
        if value < minimum:
            raise ConfigError(f"[{self.section}] {name} must be at least {minimum}")
        return value

    def number(self, name: str, default: float | None = None, *, minimum: float = 0.0) -> float:
        value = self._take(name, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"[{self.section}] {name} must be a number")
        if float(value) < minimum:
            raise ConfigError(f"[{self.section}] {name} must be at least {minimum}")
        return float(value)

    def optional_number(self, name: str) -> float | None:
        value = self._take(name, None)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"[{self.section}] {name} must be a number when present")
        return float(value)

    def strings(self, name: str) -> tuple[str, ...]:
        value = self._take(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ConfigError(f"[{self.section}] {name} must be a list of strings")
        rows = tuple(item.strip() for item in value)
        if any(not item for item in rows):
            raise ConfigError(f"[{self.section}] {name} contains an empty entry")
        if len(set(rows)) != len(rows):
            raise ConfigError(f"[{self.section}] {name} repeats an entry")
        return rows

    def free_table(self, name: str) -> Mapping[str, str]:
        """A caller-defined map: keys are data, so they are not schema-checked."""

        value = self._take(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"[{self.section}] {name} must be a table")
        out: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(item, str):
                raise ConfigError(f"[{self.section}] {name}.{key} must be a string")
            out[str(key)] = item
        return out

    def table(self, name: str) -> Mapping[str, Any] | None:
        value = self._take(name, None)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ConfigError(f"[{self.section}] {name} must be a table")
        return dict(value)

    def raw(self, name: str) -> Any:
        return self._take(name, None)

    def done(self) -> None:
        unknown = sorted(set(self._payload) - self._seen)
        if unknown:
            raise ConfigError(
                f"[{self.section}] declares unknown keys {unknown}; "
                "an unrecognized bound is refused, never ignored"
            )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContainerConnection:
    """Where the container is and how to reach it. Never what it contains."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    auth_bearer_env: str | None = None
    timeout_seconds: float = 30.0

    def resolved_headers(self, environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
        """Headers with the bearer token read from the environment, if declared."""

        source = os.environ if environ is None else environ
        headers = dict(self.headers)
        if self.auth_bearer_env:
            token = source.get(self.auth_bearer_env)
            if not token:
                raise ConfigError(
                    f"[container] auth_bearer_env names {self.auth_bearer_env!r} "
                    "but that variable is unset"
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def redacted(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "headers": {name: "<redacted>" for name in sorted(self.headers)},
            "auth_bearer_env": self.auth_bearer_env,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class TasksetSelection:
    train_split: str = "train"
    evaluation_split: str = "heldout"
    train_ids: tuple[str, ...] = ()
    evaluation_ids: tuple[str, ...] = ()
    taskset_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "train_ids": list(self.train_ids),
            "evaluation_ids": list(self.evaluation_ids),
            "taskset_id": self.taskset_id,
        }


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """The policy identity the provider will train and the container will sample."""

    provider: str
    id: str
    family: str
    rank: int = 8
    policy_kind: str = "declared_policy"
    wire_api: str = "chat_completions"
    sampling_transport: str = "message_in_capture_out"

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "id": self.id,
            "family": self.family,
            "rank": self.rank,
            "policy_kind": self.policy_kind,
            "wire_api": self.wire_api,
            "sampling_transport": self.sampling_transport,
        }


@dataclass(frozen=True, slots=True)
class PlanSelection:
    """A preset, its explicit dimension overrides, and this run's sizing.

    ``group_size``, ``groups_per_step`` and ``steps_per_round`` are plan fields
    written where an operator expects to find them; they are folded into the
    expanded plan, not read anywhere else. ``target_train_updates`` and
    ``maximum_sampled_groups`` bound the run rather than the algorithm.
    """

    preset: str
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    group_size: int | None = None
    groups_per_step: int | None = None
    steps_per_round: int | None = None
    target_train_updates: int = 1
    maximum_sampled_groups: int | None = None

    def expand(self) -> AlgorithmPlan:
        """Preset plus overrides plus sizing -> one immutable, hashed plan."""

        overlay: dict[str, Any] = {"preset": self.preset}
        for key, patch in self.overrides.items():
            overlay[key] = dict(patch)
        if self.group_size is not None:
            rollout = dict(overlay.get("rollout") or {})
            rollout["cardinality"] = self.group_size
            overlay["rollout"] = rollout
        schedule = dict(overlay.get("schedule") or {})
        if self.groups_per_step is not None:
            schedule["groups_per_step"] = self.groups_per_step
        if self.steps_per_round is not None:
            schedule["max_steps_per_round"] = self.steps_per_round
        if schedule:
            overlay["schedule"] = schedule
        return plan_module.expand(overlay)

    def to_payload(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "overrides": {key: dict(value) for key, value in self.overrides.items()},
            "group_size": self.group_size,
            "groups_per_step": self.groups_per_step,
            "steps_per_round": self.steps_per_round,
            "target_train_updates": self.target_train_updates,
            "maximum_sampled_groups": self.maximum_sampled_groups,
        }


@dataclass(frozen=True, slots=True)
class PipelineBounds:
    """Every queue bound the engine obeys. Capacities are depths, not rates."""

    mode: str = "async_queued"
    max_execution_slots: int = 1
    rollout_queue_capacity: int = 8
    score_queue_capacity: int = 8
    scored_result_queue_capacity: int = 8
    train_ready_capacity: int = 1
    maximum_policy_lag: int = 0
    rollout_retries: int = 0
    score_retries: int = 0
    max_open_groups: int = 1
    stale_disposition: str = "discard"
    heartbeat_interval_seconds: float = 30.0
    missed_heartbeats_allowed: int = 2
    quiescence_seconds: float = 0.0
    artifact_collection_seconds: float = 0.0
    straggler_max_replacements: int = 1
    expected_horizon_seconds: float | None = None
    poll_interval_seconds: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_execution_slots": self.max_execution_slots,
            "rollout_queue_capacity": self.rollout_queue_capacity,
            "score_queue_capacity": self.score_queue_capacity,
            "scored_result_queue_capacity": self.scored_result_queue_capacity,
            "train_ready_capacity": self.train_ready_capacity,
            "maximum_policy_lag": self.maximum_policy_lag,
            "rollout_retries": self.rollout_retries,
            "score_retries": self.score_retries,
            "max_open_groups": self.max_open_groups,
            "stale_disposition": self.stale_disposition,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "missed_heartbeats_allowed": self.missed_heartbeats_allowed,
            "quiescence_seconds": self.quiescence_seconds,
            "artifact_collection_seconds": self.artifact_collection_seconds,
            "straggler_max_replacements": self.straggler_max_replacements,
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True, slots=True)
class TopologyBinding:
    """The container declares the topology; this says which part is ours."""

    expected_topology_id: str | None = None
    trainable_teams: tuple[str, ...] = ()
    partial_roster: str = "refuse"
    same_policy_reduction: str = "token_weighted_mean"
    policy_types: Mapping[str, str] = field(default_factory=dict)
    minimum_viable_roster: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "expected_topology_id": self.expected_topology_id,
            "trainable_teams": list(self.trainable_teams),
            "partial_roster": self.partial_roster,
            "same_policy_reduction": self.same_policy_reduction,
            "policy_types": dict(self.policy_types),
            "minimum_viable_roster": self.minimum_viable_roster,
        }


@dataclass(frozen=True, slots=True)
class OpponentBinding:
    match_set_revision: str | None = None
    allow_alias_resolution: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "match_set_revision": self.match_set_revision,
            "allow_alias_resolution": self.allow_alias_resolution,
        }


@dataclass(frozen=True, slots=True)
class RewardBinding:
    """Which declared channel this run optimizes. Not how it is computed."""

    optimized_channel: str
    horizon_grace_seconds: float = 0.0
    require_quiescence: bool = True
    require_settlement_window: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "optimized_channel": self.optimized_channel,
            "horizon_grace_seconds": self.horizon_grace_seconds,
            "require_quiescence": self.require_quiescence,
            "require_settlement_window": self.require_settlement_window,
        }


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    paired: bool = False
    baseline_samples: int = 0
    trained_samples: int = 0
    fixed_match_set: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "paired": self.paired,
            "baseline_samples": self.baseline_samples,
            "trained_samples": self.trained_samples,
            "fixed_match_set": self.fixed_match_set,
        }


@dataclass(frozen=True, slots=True)
class LifecycleRules:
    resume_requires_rehandshake: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {"resume_requires_rehandshake": self.resume_requires_rehandshake}


@dataclass(frozen=True, slots=True)
class OfflineMode:
    mode: str = "off"
    source_run_ids: tuple[str, ...] = ()
    accepted_staleness: int = 0

    @property
    def replaying(self) -> bool:
        return self.mode == "replay"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_run_ids": list(self.source_run_ids),
            "accepted_staleness": self.accepted_staleness,
        }


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    checkpoint_every_published_update: bool = True
    retain_training_state: bool = True
    catalog: str = "runs/checkpoints.sqlite3"
    directory: str = "runs"

    def to_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_every_published_update": self.checkpoint_every_published_update,
            "retain_training_state": self.retain_training_state,
            "catalog": self.catalog,
            "directory": self.directory,
        }


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One validated ``cispo.container.v1`` document."""

    schema_version: str
    container: ContainerConnection
    taskset: TasksetSelection
    model: ModelBinding
    plan: PlanSelection
    pipeline: PipelineBounds
    topology: TopologyBinding
    opponents: OpponentBinding
    reward: RewardBinding
    evaluation: EvaluationPlan
    lifecycle: LifecycleRules
    offline: OfflineMode
    artifacts: ArtifactPolicy
    run_id: str = "run"

    def expanded_plan(self) -> AlgorithmPlan:
        return self.plan.expand()

    @property
    def group_size(self) -> int:
        return self.expanded_plan().rollout.cardinality

    @property
    def maximum_sampled_groups(self) -> int:
        """Groups this run may ever sample, replacements included."""

        declared = self.plan.maximum_sampled_groups
        if declared is not None:
            return declared
        return self.plan.target_train_updates * self.expanded_plan().groups_per_step

    def redacted_payload(self) -> dict[str, Any]:
        """The effective configuration, safe to write into a receipt."""

        expanded = self.expanded_plan()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "container": self.container.redacted(),
            "taskset": self.taskset.to_payload(),
            "model": self.model.to_payload(),
            "plan": self.plan.to_payload(),
            "pipeline": self.pipeline.to_payload(),
            "topology": self.topology.to_payload(),
            "opponents": self.opponents.to_payload(),
            "reward": self.reward.to_payload(),
            "evaluation": self.evaluation.to_payload(),
            "lifecycle": self.lifecycle.to_payload(),
            "offline": self.offline.to_payload(),
            "artifacts": self.artifacts.to_payload(),
            "expanded_plan": expanded.to_dict(),
            "plan_hash": expanded.plan_hash,
            "maximum_sampled_groups": self.maximum_sampled_groups,
        }

    def with_plan_overrides(self, **sizing: Any) -> "RunConfig":
        """A sibling config differing only in run sizing. Used by tests and CLIs."""

        return replace(self, plan=replace(self.plan, **sizing))


SECTIONS: tuple[str, ...] = (
    "container",
    "taskset",
    "model",
    "plan",
    "pipeline",
    "topology",
    "opponents",
    "reward",
    "evaluation",
    "lifecycle",
    "offline",
    "artifacts",
)


def _plan_section(payload: Mapping[str, Any] | None) -> PlanSelection:
    reader = _Reader("plan", payload)
    preset = reader.text("preset")
    overrides: dict[str, Mapping[str, Any]] = {}
    for dimension in PLAN_DIMENSIONS:
        value = reader.raw(dimension)
        if value is None:
            continue
        if isinstance(value, str):
            # The note writes a dimension override as a bare kind name. It means
            # the same thing as ``{kind = "..."}`` and expands identically.
            overrides[dimension] = {"kind": value}
        elif isinstance(value, Mapping):
            overrides[dimension] = dict(value)
        else:
            raise ConfigError(
                f"[plan] {dimension} must be a dimension kind or a table of dimension fields"
            )
    selection = PlanSelection(
        preset=preset,
        overrides=overrides,
        group_size=reader.optional_count("group_size"),
        groups_per_step=reader.optional_count("groups_per_step"),
        steps_per_round=reader.optional_count("steps_per_round"),
        target_train_updates=reader.count("target_train_updates", 1),
        maximum_sampled_groups=reader.optional_count("maximum_sampled_groups"),
    )
    reader.done()
    return selection


def _container_section(payload: Mapping[str, Any] | None) -> ContainerConnection:
    reader = _Reader("container", payload)
    connection = ContainerConnection(
        url=reader.text("url"),
        headers=reader.free_table("headers"),
        auth_bearer_env=reader.optional_text("auth_bearer_env"),
        timeout_seconds=reader.number("timeout_seconds", 30.0, minimum=0.001),
    )
    reader.done()
    return connection


def _taskset_section(payload: Mapping[str, Any] | None) -> TasksetSelection:
    reader = _Reader("taskset", payload)
    selection = TasksetSelection(
        train_split=reader.text("train_split", "train"),
        evaluation_split=reader.text("evaluation_split", "heldout"),
        train_ids=reader.strings("train_ids"),
        evaluation_ids=reader.strings("evaluation_ids"),
        taskset_id=reader.optional_text("taskset_id"),
    )
    reader.done()
    return selection


def _model_section(payload: Mapping[str, Any] | None) -> ModelBinding:
    reader = _Reader("model", payload)
    binding = ModelBinding(
        provider=reader.text("provider"),
        id=reader.text("id"),
        family=reader.text("family"),
        rank=reader.count("rank", 8),
        policy_kind=reader.text("policy_kind", "declared_policy"),
        wire_api=reader.text("wire_api", "chat_completions"),
        sampling_transport=reader.text("sampling_transport", "message_in_capture_out"),
    )
    reader.done()
    return binding


def _pipeline_section(payload: Mapping[str, Any] | None) -> PipelineBounds:
    reader = _Reader("pipeline", payload)
    bounds = PipelineBounds(
        mode=reader.choice("mode", PIPELINE_MODES, "async_queued"),
        max_execution_slots=reader.count("max_execution_slots", 1),
        rollout_queue_capacity=reader.count("rollout_queue_capacity", 8),
        score_queue_capacity=reader.count("score_queue_capacity", 8),
        scored_result_queue_capacity=reader.count("scored_result_queue_capacity", 8),
        train_ready_capacity=reader.count("train_ready_capacity", 1),
        maximum_policy_lag=reader.count("maximum_policy_lag", 0, minimum=0),
        rollout_retries=reader.count("rollout_retries", 0, minimum=0),
        score_retries=reader.count("score_retries", 0, minimum=0),
        max_open_groups=reader.count("max_open_groups", 1),
        stale_disposition=reader.choice("stale_disposition", STALE_DISPOSITIONS, "discard"),
        heartbeat_interval_seconds=reader.number(
            "heartbeat_interval_seconds", 30.0, minimum=0.001
        ),
        missed_heartbeats_allowed=reader.count("missed_heartbeats_allowed", 2),
        quiescence_seconds=reader.number("quiescence_seconds", 0.0),
        artifact_collection_seconds=reader.number("artifact_collection_seconds", 0.0),
        straggler_max_replacements=reader.count("straggler_max_replacements", 1, minimum=0),
        expected_horizon_seconds=reader.optional_number("expected_horizon_seconds"),
        poll_interval_seconds=reader.number("poll_interval_seconds", 1.0, minimum=0.0),
    )
    reader.done()
    return bounds


def _topology_section(payload: Mapping[str, Any] | None) -> TopologyBinding:
    reader = _Reader("topology", payload)
    binding = TopologyBinding(
        expected_topology_id=reader.optional_text("expected_topology_id"),
        trainable_teams=reader.strings("trainable_teams"),
        partial_roster=reader.choice("partial_roster", PARTIAL_ROSTER, "refuse"),
        same_policy_reduction=reader.text("same_policy_reduction", "token_weighted_mean"),
        policy_types=reader.free_table("policy_types"),
        minimum_viable_roster=reader.count("minimum_viable_roster", 1),
    )
    reader.done()
    return binding


def _opponents_section(payload: Mapping[str, Any] | None) -> OpponentBinding:
    reader = _Reader("opponents", payload)
    binding = OpponentBinding(
        match_set_revision=reader.optional_text("match_set_revision"),
        allow_alias_resolution=reader.flag("allow_alias_resolution", False),
    )
    reader.done()
    return binding


def _reward_section(payload: Mapping[str, Any] | None) -> RewardBinding:
    reader = _Reader("reward", payload)
    binding = RewardBinding(
        optimized_channel=reader.text("optimized_channel"),
        horizon_grace_seconds=reader.number("horizon_grace_seconds", 0.0),
        require_quiescence=reader.flag("require_quiescence", True),
        require_settlement_window=reader.flag("require_settlement_window", False),
    )
    reader.done()
    return binding


def _evaluation_section(payload: Mapping[str, Any] | None) -> EvaluationPlan:
    reader = _Reader("evaluation", payload)
    plan = EvaluationPlan(
        paired=reader.flag("paired", False),
        baseline_samples=reader.count("baseline_samples", 0, minimum=0),
        trained_samples=reader.count("trained_samples", 0, minimum=0),
        fixed_match_set=reader.flag("fixed_match_set", True),
    )
    reader.done()
    return plan


def _lifecycle_section(payload: Mapping[str, Any] | None) -> LifecycleRules:
    reader = _Reader("lifecycle", payload)
    rules = LifecycleRules(
        resume_requires_rehandshake=reader.flag("resume_requires_rehandshake", True)
    )
    reader.done()
    return rules


def _offline_section(payload: Mapping[str, Any] | None) -> OfflineMode:
    reader = _Reader("offline", payload)
    mode = OfflineMode(
        mode=reader.choice("mode", OFFLINE_MODES, "off"),
        source_run_ids=reader.strings("source_run_ids"),
        accepted_staleness=reader.count("accepted_staleness", 0, minimum=0),
    )
    reader.done()
    if mode.replaying and not mode.source_run_ids:
        raise ConfigError("[offline] replay mode needs at least one source_run_id")
    if not mode.replaying and mode.source_run_ids:
        raise ConfigError("[offline] source_run_ids are only meaningful in replay mode")
    return mode


def _artifacts_section(payload: Mapping[str, Any] | None) -> ArtifactPolicy:
    reader = _Reader("artifacts", payload)
    policy = ArtifactPolicy(
        checkpoint_every_published_update=reader.flag(
            "checkpoint_every_published_update", True
        ),
        retain_training_state=reader.flag("retain_training_state", True),
        catalog=reader.text("catalog", "runs/checkpoints.sqlite3"),
        directory=reader.text("directory", "runs"),
    )
    reader.done()
    return policy


def _assert_startup_invariants(config: RunConfig) -> None:
    """The bounds the note requires to hold before the first attempt is admitted."""

    plan = config.expanded_plan()
    plan_module.require_implemented(plan)
    pipeline = config.pipeline
    lag = pipeline.train_ready_capacity - 1
    if lag > pipeline.maximum_policy_lag:
        raise ConfigError(
            f"[pipeline] train_ready_capacity {pipeline.train_ready_capacity} holds "
            f"{lag} revisions of lag but maximum_policy_lag is "
            f"{pipeline.maximum_policy_lag}; the dequeue gate would reject work the "
            "queue was told to hold"
        )
    ceiling = plan.max_steps_per_round
    if config.plan.target_train_updates > ceiling:
        raise ConfigError(
            f"[plan] target_train_updates {config.plan.target_train_updates} exceeds the "
            f"plan's step ceiling of {ceiling} per round; raise steps_per_round "
            "explicitly if the workspace really allows it"
        )
    if config.maximum_sampled_groups < plan.groups_per_step * config.plan.target_train_updates:
        raise ConfigError(
            f"[plan] maximum_sampled_groups {config.maximum_sampled_groups} cannot supply "
            f"{plan.groups_per_step} groups for each of "
            f"{config.plan.target_train_updates} updates"
        )
    if config.reward.require_quiescence and config.reward.horizon_grace_seconds < 0:
        raise ConfigError("[reward] horizon_grace_seconds must be non-negative")
    if config.evaluation.paired and not (
        config.evaluation.baseline_samples and config.evaluation.trained_samples
    ):
        raise ConfigError(
            "[evaluation] a paired evaluation needs both baseline_samples and trained_samples"
        )


def from_mapping(payload: Mapping[str, Any], *, run_id: str = "run") -> RunConfig:
    """Validate one already-parsed document. Unknown keys are refused."""

    if not isinstance(payload, Mapping):
        raise ConfigError("a run configuration must be a table")
    version = payload.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version {version!r} is unsupported; expected {CONFIG_SCHEMA_VERSION!r}"
        )
    declared_run_id = payload.get("run_id")
    if declared_run_id is not None and (
        not isinstance(declared_run_id, str) or not declared_run_id.strip()
    ):
        raise ConfigError("run_id must be a non-empty string when present")
    unknown = sorted(set(payload) - set(SECTIONS) - {"schema_version", "run_id"})
    if unknown:
        for name in unknown:
            reason = CONTAINER_CONCERN_FIELDS.get(name.lower())
            if reason is not None:
                raise ConfigError(f"[{name}] is a container concern: {reason}")
        raise ConfigError(
            f"unknown top-level sections {unknown}; known sections are {list(SECTIONS)}"
        )
    config = RunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        container=_container_section(payload.get("container")),
        taskset=_taskset_section(payload.get("taskset")),
        model=_model_section(payload.get("model")),
        plan=_plan_section(payload.get("plan")),
        pipeline=_pipeline_section(payload.get("pipeline")),
        topology=_topology_section(payload.get("topology")),
        opponents=_opponents_section(payload.get("opponents")),
        reward=_reward_section(payload.get("reward")),
        evaluation=_evaluation_section(payload.get("evaluation")),
        lifecycle=_lifecycle_section(payload.get("lifecycle")),
        offline=_offline_section(payload.get("offline")),
        artifacts=_artifacts_section(payload.get("artifacts")),
        run_id=str(declared_run_id).strip() if declared_run_id else run_id,
    )
    _assert_startup_invariants(config)
    return config


def loads(text: str, *, run_id: str = "run") -> RunConfig:
    """Parse and validate a TOML document."""

    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"configuration is not valid TOML: {error}") from error
    return from_mapping(payload, run_id=run_id)


def load(path: str | Path, *, run_id: str | None = None) -> RunConfig:
    """Read and validate a configuration file."""

    location = Path(path)
    text = location.read_text(encoding="utf-8")
    return loads(text, run_id=run_id or location.stem)
