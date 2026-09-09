"""Declarative configuration for the fake CISPO container.

Every conformance-relevant behavior of a fake is a flag here. Nothing in this
module knows how to serve a request; it only says what a container claims and
how it is allowed to misbehave. No task, harness, or environment name appears.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from synth_optimizers.contracts.rl_clauses import ALL_CLAUSES, VERDICTS
from synth_optimizers.contracts.rl_identity import Horizon, Topology
from synth_optimizers.contracts.rl_records import RecordError, RendererProfile, digest

CONTRACT_VERSION = "synth_optimizers.cispo.v1"

#: The declared route table. The client formats these from ``/metadata`` and
#: calls nothing else, so a fake that omits a mandatory route fails loudly.
DECLARED_ROUTES: Mapping[str, str] = {
    "health_route": "/health",
    "capabilities_route": "/training/capabilities",
    "handshake_route": "/training/handshake",
    "taskset_route": "/taskset",
    "taskset_tasks_route": "/taskset/tasks",
    "topology_route": "/topologies/{topology_id}",
    "policy_bind_route": "/policy-configs",
    "policy_set_bind_route": "/policy-sets",
    "rollout_route": "/rollout",
    "rollout_state_route": "/rollouts/{rollout_id}",
    "rollout_events_route": "/rollouts/{rollout_id}/events",
    "rollout_renew_route": "/rollouts/{rollout_id}/renew",
    "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
    "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
    "trace_route": "/rollouts/{rollout_id}/trace",
    "artifacts_route": "/rollouts/{rollout_id}/artifacts",
    "reward_route": "/reward",
}

#: Correlation fields the container must preserve verbatim.
CORRELATION_FIELDS: tuple[str, ...] = (
    "run_id",
    "group_id",
    "sample_index",
    "seed",
    "policy_revision",
    "agent_instance_id",
    "team_id",
    "policy_set_revision",
    "match_set_revision_id",
)

PROMPT_BUDGET_POLICIES = frozenset({"refuse", "truncate", "compact"})
REWARD_KINDS = frozenset({"environment", "rubric_judge", "deferred_verifier", "rank"})

#: Opponent references that are not an immutable identity.
ALIAS_REFS = frozenset({"latest", "head", "stable", "current", "main"})

_EPOCH = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)


class ContainerError(RuntimeError):
    """The container refused a request. Carries the HTTP status and payload."""

    def __init__(self, status: int, payload: Mapping[str, Any]) -> None:
        self.status = status
        self.payload = dict(payload)
        self.reason = str(payload.get("reason") or payload.get("error") or "")
        super().__init__(f"HTTP {status}: {self.reason or json.dumps(self.payload)}")


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #


class Clock:
    """An injected monotone clock. Nothing in the fakes ever sleeps."""

    __slots__ = ("_offset", "_lock")

    def __init__(self, offset_seconds: float = 0.0) -> None:
        self._offset = float(offset_seconds)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._offset

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a monotone clock cannot go backwards")
        with self._lock:
            self._offset += float(seconds)
            return self._offset

    def rfc3339(self, extra_seconds: float = 0.0) -> str:
        return (_EPOCH + timedelta(seconds=self.now() + extra_seconds)).isoformat()


@dataclass(frozen=True, slots=True)
class EvidenceDefects:
    """Deliberate non-conformance. Every field defaults to conformant.

    Each flag exists to produce exactly one typed failure downstream; the
    docstring of each names the error a validator must raise.
    """

    #: ``InferenceCall.validate_for_training`` -> ``EvidenceError`` (length).
    omit_logprobs: bool = False
    #: ``InferenceCall.validate_for_training`` -> ``EvidenceError`` (sentinel).
    sentinel_logprobs: bool = False
    #: ``InferenceCall.validate_for_training`` -> ``EvidenceError`` (all zero).
    zero_logprobs: bool = False
    #: ``InferenceCall.validate_for_training`` -> ``EvidenceError`` (length).
    logprob_length_delta: int = 0
    #: ``RewardRecord.validate`` -> ``EvidenceError`` (absent is not zero).
    absent_reward: bool = False
    #: ``assert_declared_channels_present`` -> ``EvidenceError``.
    dropped_channel_id: str | None = None
    #: ``assert_strict_prefix`` -> ``EvidenceError`` (unexplained divergence).
    rerender_turn: int | None = None
    #: ``assert_no_flattened_wire`` -> ``EvidenceError``.
    flatten_wire: bool = False
    #: ``topology_from_payload`` -> ``TopologyError`` (alias resolution).
    opponent_alias: str | None = None
    #: ``assert_instance_trajectories`` -> ``TopologyError`` under ``refuse``.
    missing_instance_id: str | None = None
    #: ``assert_probe_evidence_marked`` -> ``EvidenceError``.
    probe_indistinguishable: bool = False
    #: ``HorizonEvidence.validate`` -> ``EvidenceError`` (no attestation,
    #: no clipping) and ``assert_effects_within_horizon`` -> ``EvidenceError``.
    unquiesced_deferred_program: bool = False
    #: ``assert_uniform_group`` -> ``MixedGroupError`` (pin drift).
    match_set_drift: str | None = None


@dataclass(frozen=True, slots=True)
class ContainerConfig:
    """One fake container, declared rather than coded.

    ``topology`` and ``renderer_profile`` are container-declared facts: the
    executor binds what is here and never infers a roster from an agent count
    or a task name.
    """

    container_id: str
    image_digest: str
    renderer_profile: RendererProfile
    topology: Topology
    taskset_id: str
    task_ids: tuple[str, ...]
    task_family: str = "family_a"
    splits: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    taskset_version: str = "1"
    strong_task_digests: bool = False
    model_id: str = "openai/gpt-oss-20b"
    model_family: str = "gpt_oss"
    policy_kind: str = "declared_policy"
    wire_api: str = "chat_completions"
    sampling_transport: str = "message_in_capture_out"

    # --- lifecycle capability flags ---------------------------------------- #
    #: The container's own advertised obligation. A lease is never derived from
    #: the horizon: the horizon says how long an episode runs, the TTL says how
    #: long one grant survives without a heartbeat, and they are different
    #: clocks.
    lease_ttl_seconds: float = 300.0
    #: When false, a TTL shorter than the declared horizon cannot be extended,
    #: which is a ``lifecycle.lease_renewal`` rejection rather than a warning.
    lease_renewable: bool = True
    #: Declared conversion for the fallback ``steps`` horizon, so a step
    #: horizon never reads as seconds by accident.
    step_seconds_per_unit: float = 30.0
    advertised_concurrency: int = 8
    polls_until_terminal: int = 1
    handshake_ttl_seconds: float = 600.0
    partial_roster_disposition: str = "refuse"

    # --- evidence capability flags ----------------------------------------- #
    turns: int = 1
    tito_supported: bool = False
    probe_binding_supported: bool = True
    artifact_by_reference: bool = False
    judge_spans: bool = False
    declared_compaction_turn: int | None = None
    compaction_authored_by_policy: bool = False
    max_prompt_tokens: int | None = None
    prompt_budget_policy: str = "refuse"
    finish_reason: str = "stop_token"

    # --- reward capability flags ------------------------------------------- #
    reward_kind: str = "environment"
    deferred_scoring: bool = False
    quiescence_supported: bool = True
    settlement_window_seconds: float = 0.0
    #: The measure the container's reward contract emits when nothing varies it.
    reward_value: float = 1.0
    #: Per-attempt measure. A single constant makes every attempt in a group
    #: tie, and a group with no ordering carries no credit, so a fake that only
    #: ever declares one value cannot drive a run as far as a train call. This
    #: is either a callable ``(task_id, sample_index) -> float`` or a mapping
    #: keyed by ``(task_id, sample_index)`` or by ``sample_index`` alone;
    #: whatever it does not answer falls back to ``reward_value``.
    reward_value_by_sample: (
        Mapping[Any, float] | Callable[[str, int], float] | None
    ) = None
    optimized_team_id: str | None = None
    evaluation_plan_id: str = "eval_plan_v1"
    judge_model_id: str = "judge/model-a"

    # --- handshake shaping -------------------------------------------------- #
    clock_skew_seconds: float = 0.0
    skew_tolerance_seconds: float = 1.0
    #: Which clause a skew breach is reported under. The note requires "a
    #: rejected clause" without naming one; see the report.
    skew_clause_id: str = "reward.horizon_quiescence"
    clause_overrides: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    seed: int = 0
    defects: EvidenceDefects = field(default_factory=EvidenceDefects)

    def __post_init__(self) -> None:
        if self.prompt_budget_policy not in PROMPT_BUDGET_POLICIES:
            raise RecordError(f"unknown prompt_budget_policy {self.prompt_budget_policy!r}")
        if self.reward_kind not in REWARD_KINDS:
            raise RecordError(f"unknown reward_kind {self.reward_kind!r}")
        if not self.task_ids:
            raise RecordError("a container must declare at least one task row")
        if self.turns < 1:
            raise RecordError("turns must be positive")
        for clause_id, (verdict, _reason) in self.clause_overrides.items():
            if clause_id not in ALL_CLAUSES:
                raise RecordError(f"unknown clause override {clause_id!r}")
            if verdict not in VERDICTS:
                raise RecordError(f"unknown verdict {verdict!r}")

    def reward_for(self, task_id: str, sample_index: int) -> float:
        """The measure this container declares for one attempt of one row."""

        source = self.reward_value_by_sample
        if source is None:
            return float(self.reward_value)
        if callable(source):
            return float(source(task_id, sample_index))
        for key in ((task_id, sample_index), sample_index):
            if key in source:
                return float(source[key])
        return float(self.reward_value)

    @property
    def declared_splits(self) -> dict[str, list[str]]:
        """The splits the container advertises, from one place only."""

        splits = dict(self.splits) or {
            "train": list(self.task_ids),
            "eval": list(self.task_ids[:1]),
        }
        return {name: list(rows) for name, rows in splits.items()}

    @property
    def reward_channel_ids(self) -> tuple[str, ...]:
        """The channels this container's reward contract will actually emit."""

        teams = self.topology.teams
        if self.topology.reward_relation in {"competitive_rank", "competitive_margin"}:
            ordered = sorted(teams, key=lambda team: (not team.trainable, team.team_id))
            return tuple(f"score::{team.team_id}" for team in ordered)
        return ("score",)

    @property
    def contract_hash(self) -> str:
        return digest({"contract": CONTRACT_VERSION, "routes": dict(DECLARED_ROUTES)}, length=32)

    @property
    def horizon(self) -> Horizon:
        return self.topology.horizon or Horizon(
            horizon_kind="steps",
            value=float(self.turns),
            seconds_per_unit=self.step_seconds_per_unit,
        )

    @property
    def horizon_seconds(self) -> float:
        """Wall-clock duration the declared horizon actually covers.

        Raises ``TopologyError`` through ``declared_seconds_per_unit`` when a
        unit horizon declared no conversion: a lease may not be guessed from a
        unit with no duration.
        """

        horizon = self.horizon
        return horizon.value * horizon.declared_seconds_per_unit()
