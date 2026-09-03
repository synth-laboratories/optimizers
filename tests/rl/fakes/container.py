"""An in-process fake CISPO container plus the client that drives it.

The fake serves the full declared route surface over stdlib ``http.server`` on
``127.0.0.1`` at an ephemeral port. It is a *contract* fake, not a simulator:
it has no environment, no model, and no task logic. Everything a conformance
test needs to vary is a flag on :class:`ContainerConfig`, and everything the
fake emits is derived deterministically from ``ContainerConfig.seed`` plus the
requested task row, so a replay test can reproduce an attempt bit-for-bit.

Two rules the fake never breaks, because they are the reason it exists:

* it round-trips the executor's opaque correlation metadata untouched; and
* it emits exactly one terminal result per accepted attempt.

Everything else -- including well-formedness of the evidence -- is a flag, so
that the deliberately non-conformant scenarios in :mod:`fakes.scenarios`
are the *same code path* as the conformant ones with one bit flipped.

No real time passes anywhere: leases, horizons, and handshake expiry all read
an injected :class:`Clock`.
"""

from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

from synth_optimizers.contracts.rl_clauses import (
    ALL_CLAUSES,
    HANDSHAKE_SCHEMA_VERSION,
    MANDATORY_CLAUSES,
    VERDICTS,
)
from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    CommunicationChannel,
    GroupPin,
    Horizon,
    Team,
    Topology,
    TopologyError,
)
from synth_optimizers.contracts.rl_records import (
    LOGPROB_SENTINEL,
    BehaviorFingerprint,
    CompactionProvenance,
    EvidenceError,
    HorizonEvidence,
    InferenceCall,
    RecordError,
    RendererProfile,
    RewardChannel,
    RewardRecord,
    SamplingProfile,
    TrainableEpisode,
    TrainableSegment,
    digest,
)

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


# --------------------------------------------------------------------------- #
# Configuration -- the whole conformance matrix lives here
# --------------------------------------------------------------------------- #


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
    model_id: str = "openai/gpt-oss-20b"
    model_family: str = "gpt_oss"
    policy_kind: str = "declared_policy"
    wire_api: str = "chat_completions"
    sampling_transport: str = "message_in_capture_out"

    # --- lifecycle capability flags ---------------------------------------- #
    lease_ttl_seconds: float = 300.0
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
    reward_value: float = 1.0
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

    @property
    def contract_hash(self) -> str:
        return digest({"contract": CONTRACT_VERSION, "routes": dict(DECLARED_ROUTES)}, length=32)

    @property
    def horizon(self) -> Horizon:
        return self.topology.horizon or Horizon(horizon_kind="steps", value=float(self.turns))


# --------------------------------------------------------------------------- #
# Deterministic synthesis helpers
# --------------------------------------------------------------------------- #


def _rng(*parts: Any) -> random.Random:
    """Seeded from a canonical digest, so it never depends on hash ordering."""

    return random.Random(digest(list(parts)))


def _tokens(count: int, *parts: Any) -> tuple[int, ...]:
    rng = _rng("tokens", *parts)
    return tuple(rng.randrange(1000, 90000) for _ in range(count))


def _logprob_vector(count: int, *parts: Any) -> tuple[float, ...]:
    rng = _rng("logprobs", *parts)
    return tuple(-round(rng.random() * 1.8 + 0.02, 6) for _ in range(count))


# --------------------------------------------------------------------------- #
# Evidence synthesis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Attempt:
    rollout_id: str
    idempotency_key: str
    task_id: str
    correlation: dict[str, Any]
    handshake_id: str
    policy_config: dict[str, Any]
    admitted_at: float


def _behavior(cfg: ContainerConfig, revision: int, transport: str) -> BehaviorFingerprint:
    return BehaviorFingerprint(
        renderer_profile=cfg.renderer_profile,
        model_family=cfg.model_family,
        model_id=cfg.model_id,
        policy_revision=revision,
        wire_api=cfg.wire_api,
        sampling_transport=transport,
        sampling=SamplingProfile(temperature=1.0, top_p=1.0, seed=cfg.seed),
    )


def _wire_objects(
    cfg: ContainerConfig, *, turn: int, instance_id: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The wire object is the semantic record; tokens are the training record.

    ``flatten_wire`` declares the responses wire and then persists chat
    messages, which is the prohibited flattening.
    """

    declared = cfg.wire_api
    persisted = "chat_completions" if cfg.defects.flatten_wire else declared
    key = {"turn": turn, "instance": instance_id}
    if persisted == "responses":
        return (
            {"wire": "responses", "input": [{"type": "message", "role": "user"}], **key},
            {"wire": "responses", "output": [{"type": "message", "role": "assistant"}]},
        )
    return (
        {"wire": "chat_completions", "messages": [{"role": "user"}], **key},
        {"wire": "chat_completions", "choices": [{"message": {"role": "assistant"}}]},
    )


def _apply_logprob_defects(
    cfg: ContainerConfig, values: tuple[float, ...]
) -> tuple[float, ...]:
    defects = cfg.defects
    if defects.omit_logprobs:
        return ()
    if defects.sentinel_logprobs:
        return (LOGPROB_SENTINEL,) + values[1:]
    if defects.zero_logprobs:
        return tuple(0.0 for _ in values)
    if defects.logprob_length_delta:
        delta = defects.logprob_length_delta
        if delta > 0:
            return values + values[:delta]
        return values[: max(0, len(values) + delta)]
    return values


def _prompt_budget(
    cfg: ContainerConfig, prompt: tuple[int, ...], *, turn: int
) -> tuple[tuple[int, ...], CompactionProvenance | None]:
    """Declared, never improvised: refuse, truncate, or compact."""

    budget = cfg.max_prompt_tokens
    if budget is None or len(prompt) <= budget:
        return prompt, None
    if cfg.prompt_budget_policy == "refuse":
        raise _Refusal("prompt_budget_refused", f"rendered prompt {len(prompt)} > {budget}")
    if cfg.prompt_budget_policy == "truncate":
        return prompt[-budget:], CompactionProvenance(
            rule="prompt_budget_truncate_head",
            divergence_index=0,
            removed_message_indices=(0,),
            authored_by_policy=False,
        )
    return prompt[:1] + prompt[-(budget - 1) :], CompactionProvenance(
        rule="prompt_budget_compact_middle",
        divergence_index=1,
        removed_message_indices=(1, 2),
        authored_by_policy=False,
    )


class _Refusal(Exception):
    """The container itself refuses the attempt (not an evidence failure)."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(reason)


def _instance_calls(
    cfg: ContainerConfig,
    *,
    attempt: _Attempt,
    instance: AgentInstance | None,
    transport: str,
    probe: bool,
) -> list[InferenceCall]:
    """Synthesize one instance's per-turn calls.

    Rendered prompt pieces derive from the task row, the renderer profile, the
    instance, and the turn -- never from the transport or the container seed --
    so a TiTo container and a message-in container reach byte-identical prompt
    token ids for the same row and profile. Sampled generations and their
    logprobs additionally derive from ``ContainerConfig.seed``, which is what
    makes a whole attempt reproducible bit-for-bit under a seed.
    """

    instance_id = instance.agent_instance_id if instance else None
    trainable = bool(instance.trainable) if instance else True
    if probe and not cfg.defects.probe_indistinguishable:
        trainable = False
    revision = int(attempt.correlation.get("policy_revision") or 0)
    fingerprint = _behavior(cfg, revision, transport).value
    profile_key = cfg.renderer_profile.fingerprint
    system = _tokens(4, "system", profile_key)
    opening = _tokens(8, "task", attempt.task_id, profile_key, instance_id)

    if probe:
        provenance = "engine_meta" if cfg.defects.probe_indistinguishable else "probe_synthetic"
    elif trainable:
        provenance = "engine_meta"
    else:
        provenance = "wire_derived"

    calls: list[InferenceCall] = []
    previous: InferenceCall | None = None
    branch_id = "root"
    for turn in range(cfg.turns):
        compaction: CompactionProvenance | None = None
        parent_branch: str | None = None
        forks = cfg.declared_compaction_turn == turn
        rerenders = cfg.defects.rerender_turn == turn
        if previous is None:
            prompt = system + opening
        elif forks or rerenders:
            prompt = system + _tokens(6, "summary", attempt.task_id, instance_id, turn)
            if forks:
                compaction = CompactionProvenance(
                    rule=(
                        "policy_sampled_summary"
                        if cfg.compaction_authored_by_policy
                        else "harness_deterministic_compaction"
                    ),
                    divergence_index=len(system),
                    removed_message_indices=(1, 2),
                    authored_by_policy=cfg.compaction_authored_by_policy,
                )
                parent_branch = branch_id
                branch_id = f"branch_{turn}"
        else:
            prompt = previous.full_sequence + _tokens(
                5, "observation", attempt.task_id, instance_id, turn
            )
        prompt, budget_provenance = _prompt_budget(cfg, prompt, turn=turn)
        if budget_provenance is not None and compaction is None:
            compaction = budget_provenance
            if previous is not None:
                parent_branch = branch_id
                branch_id = f"budget_{turn}"

        generated = _tokens(
            6, "generation", cfg.seed, attempt.task_id, instance_id, turn, profile_key
        )
        logprobs = _apply_logprob_defects(
            cfg,
            _logprob_vector(len(generated), cfg.seed, attempt.task_id, instance_id, turn),
        )
        sampled_mask = tuple(1 if trainable else 0 for _ in generated)
        wire_request, wire_response = _wire_objects(cfg, turn=turn, instance_id=instance_id)

        effect_start: int | None = None
        effect_end: int | None = None
        if cfg.topology.actuation_model == "deferred_program":
            effect_start = turn * 10
            effect_end = turn * 10 + 8
            last_turn = turn == cfg.turns - 1
            if last_turn and cfg.defects.unquiesced_deferred_program:
                effect_end = int(cfg.horizon.value) + 40

        call = InferenceCall(
            call_id=f"{attempt.rollout_id}:{instance_id or 'solo'}:{turn}",
            proxy_request_id=f"prid_{digest([attempt.rollout_id, instance_id, turn], length=12)}",
            rollout_id=attempt.rollout_id,
            group_id=str(attempt.correlation.get("group_id") or ""),
            sample_index=int(attempt.correlation.get("sample_index") or 0),
            behavior_fingerprint=fingerprint,
            policy_revision=revision,
            wire_api=cfg.wire_api,
            sampling_transport=transport,
            token_capture_provenance=provenance,
            prompt_token_ids=prompt,
            generation_token_ids=generated,
            generation_logprobs=logprobs,
            sampled_mask=sampled_mask,
            finish_reason=cfg.finish_reason if turn == cfg.turns - 1 else "stop_token",
            stop_token_ids=cfg.renderer_profile.stop_token_ids,
            trainable=trainable,
            branch_id=branch_id,
            parent_branch_id=parent_branch,
            compaction=compaction,
            agent_instance_id=instance_id,
            team_id=instance.team_id if instance else None,
            role_id=instance.role_id if instance else None,
            policy_type_id=instance.policy_type_id if instance else None,
            parameter_group_id=(
                cfg.topology.parameter_groups.get(instance.policy_type_id)
                if instance and instance.trainable
                else None
            ),
            policy_set_revision_id=attempt.correlation.get("policy_set_revision"),
            effect_tick_start=effect_start,
            effect_tick_end=effect_end,
            wire_request=wire_request,
            wire_response=wire_response,
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(generated)},
            created_at=f"tick:{turn}",
        )
        calls.append(call)
        previous = call
    return calls


def _judge_call(cfg: ContainerConfig, attempt: _Attempt) -> InferenceCall:
    """A rubric judge's span: recorded with its author named, never trainable."""

    generated = _tokens(5, "judge", cfg.seed, attempt.task_id)
    return InferenceCall(
        call_id=f"{attempt.rollout_id}:judge:0",
        proxy_request_id=f"prid_{digest([attempt.rollout_id, 'judge'], length=12)}",
        rollout_id=attempt.rollout_id,
        group_id=str(attempt.correlation.get("group_id") or ""),
        sample_index=int(attempt.correlation.get("sample_index") or 0),
        behavior_fingerprint="judge",
        policy_revision=0,
        wire_api=cfg.wire_api,
        sampling_transport=cfg.sampling_transport,
        token_capture_provenance="wire_derived",
        prompt_token_ids=_tokens(6, "judge_prompt", attempt.task_id),
        generation_token_ids=generated,
        generation_logprobs=_logprob_vector(
            len(generated), "judge", cfg.seed, attempt.task_id
        ),
        sampled_mask=tuple(0 for _ in generated),
        finish_reason="stop_token",
        stop_token_ids=cfg.renderer_profile.stop_token_ids,
        trainable=False,
        role_id="judge",
        policy_type_id="judge",
        wire_request={
            "wire": cfg.wire_api,
            "author": "judge",
            "judge_model_id": cfg.judge_model_id,
        },
        wire_response={"wire": cfg.wire_api, "author": "judge"},
        usage={"author": "judge"},
        created_at="tick:score",
    )


def _stitch(calls: Sequence[InferenceCall]) -> TrainableSegment:
    """One contiguous trainer sequence. Retained prefixes are loss-masked."""

    final = calls[-1]
    total = final.full_sequence
    mask = [0] * len(total)
    logprobs = [0.0] * len(total)
    for call in calls:
        start = len(call.prompt_token_ids)
        if start + len(call.generation_token_ids) > len(total):
            continue
        flags = call.sampled_mask or tuple(1 for _ in call.generation_token_ids)
        for offset, flag in enumerate(flags):
            mask[start + offset] = int(bool(flag))
        for offset, value in enumerate(call.generation_logprobs):
            if start + offset < len(logprobs):
                logprobs[start + offset] = value
    return TrainableSegment(
        token_ids=total,
        loss_mask=tuple(mask),
        behavior_logprobs=tuple(logprobs),
        branch_id=final.branch_id,
        parameter_group_id=final.parameter_group_id,
        agent_instance_id=final.agent_instance_id,
        call_ids=tuple(call.call_id for call in calls),
    )


def _episodes(
    cfg: ContainerConfig,
    attempt: _Attempt,
    calls: Sequence[InferenceCall],
    *,
    trace_digest: str,
) -> list[TrainableEpisode]:
    trainable = [call for call in calls if call.trainable]
    by_instance: dict[str | None, list[InferenceCall]] = {}
    for call in trainable:
        by_instance.setdefault(call.agent_instance_id, []).append(call)
    episodes: list[TrainableEpisode] = []
    for instance_id, instance_calls in by_instance.items():
        branches: dict[str, list[InferenceCall]] = {}
        for call in instance_calls:
            branches.setdefault(call.branch_id, []).append(call)
        segments = tuple(_stitch(group) for group in branches.values())
        head = instance_calls[0]
        episodes.append(
            TrainableEpisode(
                rollout_id=attempt.rollout_id,
                task_id=attempt.task_id,
                seed=int(attempt.correlation.get("seed") or 0),
                policy_revision=head.policy_revision,
                behavior_fingerprint=head.behavior_fingerprint,
                segments=segments,
                terminal_status="completed",
                usage={
                    "calls": len(instance_calls),
                    "prompt_tokens": sum(len(c.prompt_token_ids) for c in instance_calls),
                    "completion_tokens": sum(
                        len(c.generation_token_ids) for c in instance_calls
                    ),
                    "provider_request_ids": [c.proxy_request_id for c in instance_calls],
                },
                agent_instance_id=instance_id,
                team_id=head.team_id,
                policy_set_revision_id=head.policy_set_revision_id,
                trace_digest=trace_digest,
            )
        )
    return episodes


def _reward(
    cfg: ContainerConfig, attempt: _Attempt, *, trace_digest: str, scored_offset: float
) -> RewardRecord | None:
    if cfg.defects.absent_reward:
        return RewardRecord(
            reward_id=f"reward_{attempt.rollout_id}",
            rollout_id=attempt.rollout_id,
            trace_digest=trace_digest,
            channels=(),
            optimized_channel="absent",
            terminal_status="completed",
            evaluation_plan_id=cfg.evaluation_plan_id,
            metadata={"missing_evidence": "verifier produced no measure"},
        )
    teams = cfg.topology.teams or (Team(team_id="solo", trainable=True),)
    competitive = cfg.topology.reward_relation in {"competitive_rank", "competitive_margin"}
    channels: list[RewardChannel] = []
    if competitive:
        ordered = sorted(teams, key=lambda team: (not team.trainable, team.team_id))
        for rank, team in enumerate(ordered, start=1):
            measure = round(cfg.reward_value / rank, 6)
            channels.append(
                RewardChannel(
                    channel_id=f"score::{team.team_id}",
                    team_id=team.team_id,
                    measure=measure,
                    rank=rank,
                )
            )
    else:
        team = teams[0]
        channels.append(
            RewardChannel(
                channel_id="score",
                team_id=team.team_id if len(teams) > 1 else None,
                measure=cfg.reward_value,
            )
        )
    wanted_team = cfg.optimized_team_id
    optimized = channels[0].channel_id
    if wanted_team is not None:
        for channel in channels:
            if channel.team_id == wanted_team:
                optimized = channel.channel_id
    quiesced = cfg.quiescence_supported and not cfg.defects.unquiesced_deferred_program
    clipped = (not cfg.quiescence_supported) and not cfg.defects.unquiesced_deferred_program
    horizon = HorizonEvidence(
        horizon_kind=cfg.horizon.horizon_kind,
        horizon_value=cfg.horizon.value,
        scored_at_offset_seconds=scored_offset,
        clipped=clipped,
        quiescence_attested=quiesced,
        settlement_window_seconds=cfg.settlement_window_seconds,
        credited_settlement_seconds=min(scored_offset, cfg.settlement_window_seconds),
    )
    metadata: dict[str, Any] = {"reward_kind": cfg.reward_kind}
    if cfg.reward_kind == "rubric_judge" or cfg.judge_spans:
        metadata["judge_model_id"] = cfg.judge_model_id
        metadata["judge_spans_trainable"] = False
    if cfg.deferred_scoring:
        metadata["deferred_scoring"] = True
    return RewardRecord(
        reward_id=f"reward_{attempt.rollout_id}",
        rollout_id=attempt.rollout_id,
        trace_digest=trace_digest,
        channels=tuple(channels),
        optimized_channel=optimized,
        terminal_status="completed",
        evaluation_plan_id=cfg.evaluation_plan_id,
        horizon=horizon,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Payload codecs
# --------------------------------------------------------------------------- #


def _renderer_payload(profile: RendererProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "package": profile.package,
        "package_version": profile.package_version,
        "config_digest": profile.config_digest,
        "tokenizer_id": profile.tokenizer_id,
        "tokenizer_digest": profile.tokenizer_digest,
        "stop_token_ids": list(profile.stop_token_ids),
        "modalities": list(profile.modalities),
        "add_generation_prompt": profile.add_generation_prompt,
    }


def _compaction_payload(provenance: CompactionProvenance | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    return {
        "rule": provenance.rule,
        "divergence_index": provenance.divergence_index,
        "removed_message_indices": list(provenance.removed_message_indices),
        "authored_by_policy": provenance.authored_by_policy,
    }


def _call_payload(call: InferenceCall) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "proxy_request_id": call.proxy_request_id,
        "rollout_id": call.rollout_id,
        "group_id": call.group_id,
        "sample_index": call.sample_index,
        "behavior_fingerprint": call.behavior_fingerprint,
        "policy_revision": call.policy_revision,
        "wire_api": call.wire_api,
        "sampling_transport": call.sampling_transport,
        "token_capture_provenance": call.token_capture_provenance,
        "prompt_token_ids": list(call.prompt_token_ids),
        "generation_token_ids": list(call.generation_token_ids),
        "generation_logprobs": list(call.generation_logprobs),
        "sampled_mask": list(call.sampled_mask),
        "content_mask": list(call.content_mask),
        "finish_reason": call.finish_reason,
        "stop_token_ids": list(call.stop_token_ids),
        "trainable": call.trainable,
        "branch_id": call.branch_id,
        "parent_branch_id": call.parent_branch_id,
        "compaction": _compaction_payload(call.compaction),
        "agent_instance_id": call.agent_instance_id,
        "team_id": call.team_id,
        "role_id": call.role_id,
        "policy_type_id": call.policy_type_id,
        "parameter_group_id": call.parameter_group_id,
        "policy_set_revision_id": call.policy_set_revision_id,
        "effect_tick_start": call.effect_tick_start,
        "effect_tick_end": call.effect_tick_end,
        "wire_request": dict(call.wire_request),
        "wire_response": dict(call.wire_response),
        "usage": dict(call.usage),
        "created_at": call.created_at,
        "schema_version": call.schema_version,
    }


def inference_call_from_payload(payload: Mapping[str, Any]) -> InferenceCall:
    """Rebuild the shared record from the wire. Lossless round trip."""

    compaction = payload.get("compaction")
    return InferenceCall(
        call_id=str(payload["call_id"]),
        proxy_request_id=str(payload["proxy_request_id"]),
        rollout_id=str(payload["rollout_id"]),
        group_id=str(payload.get("group_id") or ""),
        sample_index=int(payload.get("sample_index") or 0),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        policy_revision=int(payload["policy_revision"]),
        wire_api=str(payload["wire_api"]),
        sampling_transport=str(payload["sampling_transport"]),
        token_capture_provenance=str(payload["token_capture_provenance"]),
        prompt_token_ids=tuple(int(item) for item in payload["prompt_token_ids"]),
        generation_token_ids=tuple(int(item) for item in payload["generation_token_ids"]),
        generation_logprobs=tuple(float(item) for item in payload["generation_logprobs"]),
        sampled_mask=tuple(int(item) for item in payload.get("sampled_mask") or ()),
        finish_reason=str(payload["finish_reason"]),
        stop_token_ids=tuple(int(item) for item in payload.get("stop_token_ids") or ()),
        content_mask=tuple(int(item) for item in payload.get("content_mask") or ()),
        trainable=bool(payload.get("trainable", True)),
        branch_id=str(payload.get("branch_id") or "root"),
        parent_branch_id=payload.get("parent_branch_id"),
        compaction=(
            CompactionProvenance(
                rule=str(compaction["rule"]),
                divergence_index=int(compaction["divergence_index"]),
                removed_message_indices=tuple(
                    int(item) for item in compaction.get("removed_message_indices") or ()
                ),
                authored_by_policy=bool(compaction.get("authored_by_policy")),
            )
            if compaction
            else None
        ),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        role_id=payload.get("role_id"),
        policy_type_id=payload.get("policy_type_id"),
        parameter_group_id=payload.get("parameter_group_id"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        effect_tick_start=payload.get("effect_tick_start"),
        effect_tick_end=payload.get("effect_tick_end"),
        wire_request=dict(payload.get("wire_request") or {}),
        wire_response=dict(payload.get("wire_response") or {}),
        usage=dict(payload.get("usage") or {}),
        created_at=str(payload.get("created_at") or ""),
    )


def _episode_payload(episode: TrainableEpisode) -> dict[str, Any]:
    return {
        "rollout_id": episode.rollout_id,
        "task_id": episode.task_id,
        "seed": episode.seed,
        "policy_revision": episode.policy_revision,
        "behavior_fingerprint": episode.behavior_fingerprint,
        "terminal_status": episode.terminal_status,
        "usage": dict(episode.usage),
        "agent_instance_id": episode.agent_instance_id,
        "team_id": episode.team_id,
        "policy_set_revision_id": episode.policy_set_revision_id,
        "trace_digest": episode.trace_digest,
        "segments": [
            {
                "token_ids": list(segment.token_ids),
                "loss_mask": list(segment.loss_mask),
                "behavior_logprobs": list(segment.behavior_logprobs),
                "branch_id": segment.branch_id,
                "parameter_group_id": segment.parameter_group_id,
                "agent_instance_id": segment.agent_instance_id,
                "call_ids": list(segment.call_ids),
            }
            for segment in episode.segments
        ],
    }


def trainable_episode_from_payload(payload: Mapping[str, Any]) -> TrainableEpisode:
    segments = tuple(
        TrainableSegment(
            token_ids=tuple(int(item) for item in raw["token_ids"]),
            loss_mask=tuple(int(item) for item in raw["loss_mask"]),
            behavior_logprobs=tuple(float(item) for item in raw["behavior_logprobs"]),
            branch_id=str(raw.get("branch_id") or "root"),
            parameter_group_id=raw.get("parameter_group_id"),
            agent_instance_id=raw.get("agent_instance_id"),
            call_ids=tuple(str(item) for item in raw.get("call_ids") or ()),
        )
        for raw in payload.get("segments") or ()
    )
    return TrainableEpisode(
        rollout_id=str(payload["rollout_id"]),
        task_id=str(payload["task_id"]),
        seed=int(payload.get("seed") or 0),
        policy_revision=int(payload["policy_revision"]),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        segments=segments,
        terminal_status=str(payload["terminal_status"]),
        usage=dict(payload.get("usage") or {}),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        trace_digest=str(payload.get("trace_digest") or ""),
    )


def _reward_payload(record: RewardRecord) -> dict[str, Any]:
    horizon = record.horizon
    return {
        "reward_id": record.reward_id,
        "rollout_id": record.rollout_id,
        "trace_digest": record.trace_digest,
        "optimized_channel": record.optimized_channel,
        "terminal_status": record.terminal_status,
        "evaluation_plan_id": record.evaluation_plan_id,
        "metadata": dict(record.metadata),
        "channels": [
            {
                "channel_id": channel.channel_id,
                "team_id": channel.team_id,
                "measure": channel.measure,
                "rank": channel.rank,
            }
            for channel in record.channels
        ],
        "horizon": (
            None
            if horizon is None
            else {
                "horizon_kind": horizon.horizon_kind,
                "horizon_value": horizon.horizon_value,
                "scored_at_offset_seconds": horizon.scored_at_offset_seconds,
                "clipped": horizon.clipped,
                "quiescence_attested": horizon.quiescence_attested,
                "settlement_window_seconds": horizon.settlement_window_seconds,
                "credited_settlement_seconds": horizon.credited_settlement_seconds,
            }
        ),
    }


def reward_record_from_payload(payload: Mapping[str, Any]) -> RewardRecord:
    horizon = payload.get("horizon")
    return RewardRecord(
        reward_id=str(payload["reward_id"]),
        rollout_id=str(payload["rollout_id"]),
        trace_digest=str(payload.get("trace_digest") or ""),
        channels=tuple(
            RewardChannel(
                channel_id=str(raw["channel_id"]),
                team_id=raw.get("team_id"),
                measure=float(raw["measure"]),
                rank=raw.get("rank"),
            )
            for raw in payload.get("channels") or ()
        ),
        optimized_channel=str(payload["optimized_channel"]),
        terminal_status=str(payload["terminal_status"]),
        evaluation_plan_id=str(payload["evaluation_plan_id"]),
        horizon=(
            None
            if not horizon
            else HorizonEvidence(
                horizon_kind=str(horizon["horizon_kind"]),
                horizon_value=float(horizon["horizon_value"]),
                scored_at_offset_seconds=float(horizon["scored_at_offset_seconds"]),
                clipped=bool(horizon["clipped"]),
                quiescence_attested=bool(horizon["quiescence_attested"]),
                settlement_window_seconds=float(horizon.get("settlement_window_seconds") or 0.0),
                credited_settlement_seconds=float(
                    horizon.get("credited_settlement_seconds") or 0.0
                ),
            )
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def _topology_payload(cfg: ContainerConfig) -> dict[str, Any]:
    topology = cfg.topology
    alias = cfg.defects.opponent_alias
    return {
        "topology_id": topology.topology_id,
        "turn_model": topology.turn_model,
        "actuation_model": topology.actuation_model,
        "reward_relation": topology.reward_relation,
        "parameter_groups": dict(topology.parameter_groups),
        "partial_roster_disposition": cfg.partial_roster_disposition,
        "agent_instances": [
            {
                "agent_instance_id": instance.agent_instance_id,
                "role_id": instance.role_id,
                "policy_type_id": instance.policy_type_id,
                "team_id": instance.team_id,
                "trainable": instance.trainable,
                "policy_ref": (
                    alias
                    if (alias and not instance.trainable)
                    else instance.pinned_identity
                ),
            }
            for instance in topology.agent_instances
        ],
        "teams": [
            {
                "team_id": team.team_id,
                "trainable": team.trainable,
                "minimum_viable_roster": team.minimum_viable_roster,
            }
            for team in topology.teams
        ],
        "communication_channels": [
            {
                "channel_id": channel.channel_id,
                "scope": channel.scope,
                "trainable_for_author": channel.trainable_for_author,
            }
            for channel in topology.communication_channels
        ],
        "horizon": {
            "horizon_kind": topology.horizon.horizon_kind,
            "value": topology.horizon.value,
            "time_dilation": topology.horizon.time_dilation,
            "grace_seconds": topology.horizon.grace_seconds,
            "seconds_per_unit": topology.horizon.seconds_per_unit,
            "lease_seconds": topology.horizon.lease_seconds,
        }
        if topology.horizon
        else None,
    }


def topology_from_payload(payload: Mapping[str, Any]) -> Topology:
    """Decode a declared topology.

    A non-trainable instance whose ``policy_ref`` is an alias rather than an
    immutable identity is refused here with a ``TopologyError``: an opponent
    resolved as ``latest`` is not a reproducible sample.
    """

    instances: list[AgentInstance] = []
    for raw in payload.get("agent_instances") or ():
        trainable = bool(raw.get("trainable"))
        ref = raw.get("policy_ref")
        if not trainable and isinstance(ref, str) and ref.strip().lower() in ALIAS_REFS:
            raise TopologyError(
                f"opponent {raw.get('agent_instance_id')!r} resolves alias {ref!r}; "
                "a non-trainable instance must pin an immutable identity"
            )
        instances.append(
            AgentInstance(
                agent_instance_id=str(raw["agent_instance_id"]),
                role_id=str(raw["role_id"]),
                policy_type_id=str(raw["policy_type_id"]),
                team_id=str(raw["team_id"]),
                trainable=trainable,
                pinned_identity=ref,
            )
        )
    horizon = payload.get("horizon")
    return Topology(
        topology_id=str(payload["topology_id"]),
        turn_model=str(payload["turn_model"]),
        actuation_model=str(payload["actuation_model"]),
        reward_relation=str(payload["reward_relation"]),
        agent_instances=tuple(instances),
        teams=tuple(
            Team(
                team_id=str(raw["team_id"]),
                trainable=bool(raw.get("trainable")),
                minimum_viable_roster=int(raw.get("minimum_viable_roster") or 1),
            )
            for raw in payload.get("teams") or ()
        ),
        communication_channels=tuple(
            CommunicationChannel(
                channel_id=str(raw["channel_id"]),
                scope=str(raw["scope"]),
                trainable_for_author=bool(raw.get("trainable_for_author", True)),
            )
            for raw in payload.get("communication_channels") or ()
        ),
        horizon=(
            None
            if not horizon
            else Horizon(
                horizon_kind=str(horizon["horizon_kind"]),
                value=float(horizon["value"]),
                time_dilation=float(horizon.get("time_dilation") or 1.0),
                grace_seconds=float(horizon.get("grace_seconds") or 0.0),
                seconds_per_unit=float(horizon.get("seconds_per_unit") or 1.0),
            )
        ),
        parameter_groups=dict(payload.get("parameter_groups") or {}),
    )


# --------------------------------------------------------------------------- #
# Server state
# --------------------------------------------------------------------------- #


class _State:
    """All mutable container state. Guarded by one lock; no background work."""

    def __init__(self, cfg: ContainerConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self.lock = threading.Lock()
        self.capability_epoch = 0
        self.handshakes: dict[str, dict[str, Any]] = {}
        self.revoked: set[str] = set()
        self.policy_configs: dict[str, dict[str, Any]] = {}
        self.policy_sets: dict[str, dict[str, Any]] = {}
        self.attempts: dict[str, _Attempt] = {}
        self.idempotency: dict[str, str] = {}
        self.states: dict[str, str] = {}
        self.polls: dict[str, int] = {}
        self.leases: dict[str, float] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.terminals: dict[str, int] = {}
        self.traces: dict[str, dict[str, Any]] = {}
        self.rewards: dict[str, dict[str, Any] | None] = {}
        self.refusals: dict[str, str] = {}
        self.counter = 0
        self.request_log: list[tuple[str, str]] = []

    # -- capabilities --------------------------------------------------- #

    def capability_document(self) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "schema_version": "training.rollout.capabilities.v1",
            "contract_version": CONTRACT_VERSION,
            "capability_epoch": self.capability_epoch,
            "container_id": cfg.container_id,
            "image_digest": cfg.image_digest,
            "contract_hash": cfg.contract_hash,
            "lifecycle": {
                "asynchronous_submission": True,
                "idempotent_attempt_ids": True,
                "cancellation": True,
                "lease_ttl_seconds": cfg.lease_ttl_seconds,
                "max_concurrency": cfg.advertised_concurrency,
                "exactly_one_terminal": True,
                "pause_resume": True,
                "correlation_fields": list(CORRELATION_FIELDS),
            },
            "evidence": {
                "trace_v5": True,
                "behavior_logprobs": True,
                "strict_prefix_stitching": True,
                "masking_convention": "renderer_sampled_mask_x_policy_authorship",
                "wire_objects_persisted": True,
                "artifact_by_reference": cfg.artifact_by_reference,
                "tito": cfg.tito_supported,
                "sampling_transports": (
                    ["message_in_capture_out", "tokens_in_tokens_out"]
                    if cfg.tito_supported
                    else ["message_in_capture_out"]
                ),
                "wire_api": cfg.wire_api,
                "probe_binding": cfg.probe_binding_supported,
                "prompt_budget_policy": cfg.prompt_budget_policy,
                "max_prompt_tokens": cfg.max_prompt_tokens,
            },
            "reward": {
                "authority": "container",
                "reward_kind": cfg.reward_kind,
                "deferred_scoring": cfg.deferred_scoring,
                "quiescence": cfg.quiescence_supported,
                "settlement_window_seconds": cfg.settlement_window_seconds,
                "evaluation_plan_id": cfg.evaluation_plan_id,
                "channels_per_team": cfg.topology.reward_relation != "cooperative",
            },
            "topology": _topology_payload(cfg),
            "horizon": {
                "horizon_kind": cfg.horizon.horizon_kind,
                "value": cfg.horizon.value,
                "time_dilation": cfg.horizon.time_dilation,
                "grace_seconds": cfg.horizon.grace_seconds,
                "seconds_per_unit": cfg.horizon.seconds_per_unit,
                "lease_seconds": cfg.horizon.lease_seconds,
            },
            "renderer_profile": _renderer_payload(cfg.renderer_profile),
            "advertised_concurrency": cfg.advertised_concurrency,
        }

    def capability_hash(self) -> str:
        return digest(self.capability_document(), length=32)

    # -- events --------------------------------------------------------- #

    def emit(self, rollout_id: str, kind: str, **extra: Any) -> None:
        log = self.events.setdefault(rollout_id, [])
        log.append(
            {
                "cursor": len(log) + 1,
                "kind": kind,
                "at": self.clock.rfc3339(),
                "rollout_id": rollout_id,
                **extra,
            }
        )
        if kind in {"episode", "failure", "cancellation"}:
            self.terminals[rollout_id] = self.terminals.get(rollout_id, 0) + 1


# --------------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------------- #


def _clause_verdicts(
    state: _State, request: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = state.cfg
    plan = dict(request.get("run_plan") or {})
    requested_slots = int(plan.get("max_execution_slots") or plan.get("group_size") or 1)
    verdicts: dict[str, tuple[str, str]] = {clause: ("accepted", "") for clause in ALL_CLAUSES}

    if not cfg.tito_supported:
        verdicts["evidence.tito"] = ("unsupported", "container does not speak token in/out")
    if not cfg.artifact_by_reference:
        verdicts["evidence.artifact_reference"] = ("unsupported", "traces are always inline")
    if cfg.settlement_window_seconds <= 0:
        verdicts["reward.settlement_window"] = ("unsupported", "scored state does not lag")
    if not cfg.quiescence_supported:
        verdicts["reward.horizon_quiescence"] = (
            "degraded",
            "cannot kill agent-authored background processes; "
            "serves a horizon-clipped state snapshot instead",
        )
    if requested_slots > cfg.advertised_concurrency:
        verdicts["lifecycle.concurrency"] = (
            "degraded",
            f"{cfg.advertised_concurrency} leases available, {requested_slots} requested",
        )
    if len(cfg.topology.agent_instances) < 2:
        verdicts["topology.channels"] = ("unsupported", "single-instance topology")
        verdicts["topology.opponent_pinning"] = ("unsupported", "no opponent instances")
        verdicts["topology.minimum_roster"] = ("unsupported", "single-instance topology")
    if not cfg.topology.opponent_instances:
        verdicts["topology.opponent_pinning"] = ("unsupported", "no opponent instances")

    requested_profile = dict(request.get("renderer_profile") or {})
    if requested_profile:
        declared = _renderer_payload(cfg.renderer_profile)
        mismatched = [
            name
            for name in ("profile_id", "config_digest", "tokenizer_digest")
            if name in requested_profile and requested_profile[name] != declared[name]
        ]
        if mismatched:
            verdicts["policy.renderer_profile_match"] = (
                "rejected",
                f"renderer profile differs on {sorted(mismatched)}",
            )

    if abs(cfg.clock_skew_seconds) > cfg.skew_tolerance_seconds:
        verdicts[cfg.skew_clause_id] = (
            "rejected",
            f"measured skew {cfg.clock_skew_seconds}s exceeds tolerance "
            f"{cfg.skew_tolerance_seconds}s and the horizon is when reward is read",
        )

    for clause_id, (verdict, reason) in cfg.clause_overrides.items():
        verdicts[clause_id] = (verdict, reason)

    clauses = [
        {"clause_id": clause_id, "verdict": verdict, "reason": reason}
        for clause_id, (verdict, reason) in verdicts.items()
    ]
    obligations = {
        "max_concurrency": cfg.advertised_concurrency,
        "lease_ttl_seconds": cfg.lease_ttl_seconds,
        "deferred_scoring": cfg.deferred_scoring,
        "quiescence": cfg.quiescence_supported,
        "settlement_window_seconds": cfg.settlement_window_seconds,
        "partial_roster": cfg.partial_roster_disposition,
        "probe_binding": cfg.probe_binding_supported,
        "prompt_budget_policy": cfg.prompt_budget_policy,
        "horizon": {
            "horizon_kind": cfg.horizon.horizon_kind,
            "value_seconds": cfg.horizon.value,
            "time_dilation": cfg.horizon.time_dilation,
        },
    }
    return clauses, obligations


def _handshake(state: _State, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    cfg = state.cfg
    renew_of = request.get("renew_of")
    capability_hash = state.capability_hash()
    if renew_of:
        prior = state.handshakes.get(str(renew_of))
        if prior is None:
            return 404, {"error": "unknown_handshake", "reason": str(renew_of)}
        if prior["capability_hash"] != capability_hash:
            return 409, {
                "error": "capability_document_changed",
                "reason": "capability hash changed since acceptance; fail closed",
                "prior_capability_hash": prior["capability_hash"],
                "capability_hash": capability_hash,
            }
    clauses, obligations = _clause_verdicts(state, request)
    by_id = {clause["clause_id"]: clause for clause in clauses}
    rejected_mandatory = [
        clause_id
        for clause_id in MANDATORY_CLAUSES
        if by_id[clause_id]["verdict"] == "rejected"
    ]
    degraded = [clause["clause_id"] for clause in clauses if clause["verdict"] == "degraded"]
    requested_ids = tuple(dict.fromkeys(request.get("taskset", {}).get("task_ids") or ()))
    rows = requested_ids or cfg.task_ids
    resolution = [
        {
            "task_id": task_id,
            "content_digest": digest([cfg.taskset_id, cfg.taskset_version, task_id], length=32),
            "topology_ref": cfg.topology.topology_id,
            "task_family": cfg.task_family,
        }
        for task_id in rows
        if task_id in cfg.task_ids
    ]
    accept_degraded = set(request.get("accept_degraded") or ())
    unaccepted_degraded = [clause for clause in degraded if clause not in accept_degraded]
    accepted = not rejected_mandatory and not unaccepted_degraded
    state.counter += 1
    handshake_id = f"hs_{digest([cfg.container_id, state.counter, request], length=16)}"
    agreement_digest = digest(
        {
            "request": request,
            "capability_hash": capability_hash,
            "clauses": clauses,
            "obligations": obligations,
            "taskset_resolution": resolution,
            "renderer_profile": _renderer_payload(cfg.renderer_profile),
        },
        length=32,
    )
    record = {
        "schema_version": HANDSHAKE_SCHEMA_VERSION,
        "handshake_id": handshake_id,
        "accepted": accepted,
        "clauses": clauses,
        "rejected_mandatory_clauses": rejected_mandatory,
        "degraded_clauses": degraded,
        "unaccepted_degraded_clauses": unaccepted_degraded,
        "obligations": obligations,
        "taskset_resolution": resolution,
        "capability_hash": capability_hash,
        "agreement_digest": agreement_digest,
        "expires_at": state.clock.rfc3339(cfg.handshake_ttl_seconds),
        "expires_at_offset": state.clock.now() + cfg.handshake_ttl_seconds,
        "clock": {
            "container_time": state.clock.rfc3339(cfg.clock_skew_seconds),
            "measured_skew_seconds": cfg.clock_skew_seconds,
            "tolerance_seconds": cfg.skew_tolerance_seconds,
        },
    }
    if accepted:
        state.handshakes[handshake_id] = record
    return 200, record


def _check_handshake(state: _State, body: Mapping[str, Any]) -> dict[str, Any] | None:
    handshake_id = body.get("handshake_id")
    if not handshake_id:
        raise _Refused(400, "handshake_absent", "no handshake_id on the attempt")
    record = state.handshakes.get(str(handshake_id))
    if record is None:
        raise _Refused(403, "handshake_unknown", f"unknown handshake {handshake_id!r}")
    if str(handshake_id) in state.revoked:
        raise _Refused(403, "handshake_revoked", f"handshake {handshake_id!r} was revoked")
    if state.clock.now() > float(record["expires_at_offset"]):
        raise _Refused(403, "handshake_expired", f"handshake {handshake_id!r} expired")
    supplied = body.get("agreement_digest")
    if supplied is not None and supplied != record["agreement_digest"]:
        raise _Refused(
            409,
            "agreement_digest_mismatch",
            "attempt agreement digest does not match its handshake",
        )
    return record


class _Refused(Exception):
    def __init__(self, status: int, code: str, reason: str) -> None:
        self.status = status
        self.code = code
        self.reason = reason
        super().__init__(reason)


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #


def _seal(state: _State, attempt: _Attempt) -> None:
    """Build the sealed trace and the reward once, deterministically."""

    cfg = state.cfg
    probe = bool(attempt.policy_config.get("probe"))
    transport = str(attempt.policy_config.get("transport") or cfg.sampling_transport)
    topology = cfg.topology
    missing = cfg.defects.missing_instance_id
    calls: list[InferenceCall] = []
    instances: list[AgentInstance | None]
    if len(topology.agent_instances) == 1 and not topology.opponent_instances:
        instances = [topology.agent_instances[0]]
    else:
        instances = list(topology.agent_instances)
    live_ids: list[str] = []
    for instance in instances:
        if instance is not None and instance.agent_instance_id == missing:
            continue
        if instance is not None:
            live_ids.append(instance.agent_instance_id)
        calls.extend(
            _instance_calls(
                cfg, attempt=attempt, instance=instance, transport=transport, probe=probe
            )
        )
    if cfg.judge_spans:
        calls.append(_judge_call(cfg, attempt))

    channels = []
    for channel in topology.communication_channels:
        dropped = channel.channel_id == cfg.defects.dropped_channel_id
        channels.append(
            {
                "channel_id": channel.channel_id,
                "scope": channel.scope,
                "trainable_for_author": channel.trainable_for_author,
                "message_count": 0 if dropped else 2 * len(live_ids),
            }
        )

    call_payloads = [_call_payload(call) for call in calls]
    trace_digest = digest(
        {
            "rollout_id": attempt.rollout_id,
            "calls": call_payloads,
            "channels": channels,
            "container": cfg.container_id,
        },
        length=32,
    )
    episodes = _episodes(cfg, attempt, calls, trace_digest=trace_digest)
    trace = {
        "rollout_id": attempt.rollout_id,
        "task_id": attempt.task_id,
        "sealed": True,
        "schema_version": "trace.v5",
        "trace_digest": trace_digest,
        "renderer_profile": _renderer_payload(cfg.renderer_profile),
        "wire_api": cfg.wire_api,
        "sampling_transport": transport,
        "probe": probe,
        "turn_model": topology.turn_model,
        "actuation_model": topology.actuation_model,
        "declared_channels": channels,
        "correlation": dict(attempt.correlation),
        "instances": [
            {
                "agent_instance_id": instance.agent_instance_id,
                "role_id": instance.role_id,
                "policy_type_id": instance.policy_type_id,
                "team_id": instance.team_id,
                "trainable": instance.trainable,
                "pinned_identity": instance.pinned_identity,
                "present": instance.agent_instance_id in live_ids,
                "absent_at_tick": None if instance.agent_instance_id in live_ids else 0,
                "last_live_tick": None if instance.agent_instance_id in live_ids else 0,
            }
            for instance in topology.agent_instances
        ],
        "calls": call_payloads,
        "episodes": [_episode_payload(episode) for episode in episodes],
    }
    state.traces[attempt.rollout_id] = trace
    scored_offset = max(0.0, state.clock.now() - attempt.admitted_at)
    record = _reward(cfg, attempt, trace_digest=trace_digest, scored_offset=scored_offset)
    state.rewards[attempt.rollout_id] = None if record is None else _reward_payload(record)


def _group_pin_fields(state: _State, attempt: _Attempt) -> dict[str, Any]:
    cfg = state.cfg
    revision = int(attempt.correlation.get("policy_revision") or 0)
    transport = str(attempt.policy_config.get("transport") or cfg.sampling_transport)
    handshake = state.handshakes.get(attempt.handshake_id, {})
    match_set = cfg.defects.match_set_drift or attempt.correlation.get("match_set_revision_id")
    return {
        "behavior_fingerprint": _behavior(cfg, revision, transport).value,
        "policy_revision": revision,
        "wire_api": cfg.wire_api,
        "sampling_transport": transport,
        "policy_kind": cfg.policy_kind,
        "model_family": cfg.model_family,
        "container_image_digest": cfg.image_digest,
        "container_contract_hash": cfg.contract_hash,
        "handshake_agreement_digest": handshake.get("agreement_digest", ""),
        "task_family": cfg.task_family,
        "topology_id": cfg.topology.topology_id,
        "policy_set_revision_id": attempt.correlation.get("policy_set_revision"),
        "match_set_revision_id": match_set,
    }


def group_pin_from_fields(
    fields: Mapping[str, Any],
    *,
    group_id: str,
    run_id: str,
    algorithm_plan_hash: str,
    cardinality: int,
) -> GroupPin:
    """Build the executor-side pin from the container's contributed fields.

    The container contributes image digest, contract hash, agreement digest,
    wire, transport, policy kind, model family, task family, topology, and the
    policy-set / match-set revisions it actually resolved. The executor
    contributes the group identity and the plan hash.
    """

    return GroupPin(
        group_id=group_id,
        run_id=run_id,
        algorithm_plan_hash=algorithm_plan_hash,
        behavior_fingerprint=str(fields["behavior_fingerprint"]),
        policy_revision=int(fields["policy_revision"]),
        wire_api=str(fields["wire_api"]),
        sampling_transport=str(fields["sampling_transport"]),
        policy_kind=str(fields["policy_kind"]),
        model_family=str(fields["model_family"]),
        container_image_digest=str(fields["container_image_digest"]),
        container_contract_hash=str(fields["container_contract_hash"]),
        handshake_agreement_digest=str(fields["handshake_agreement_digest"]),
        task_family=str(fields["task_family"]),
        cardinality=cardinality,
        policy_set_revision_id=fields.get("policy_set_revision_id"),
        match_set_revision_id=fields.get("match_set_revision_id"),
        topology_id=fields.get("topology_id"),
    )


def _submit(state: _State, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    cfg = state.cfg
    record = _check_handshake(state, body)
    assert record is not None
    key = str(body.get("idempotency_key") or "")
    if not key:
        raise _Refused(400, "idempotency_key_required", "an attempt must carry a key")
    correlation = {
        name: body.get("correlation", {}).get(name)
        for name in CORRELATION_FIELDS
        if name in (body.get("correlation") or {})
    }
    if key in state.idempotency:
        rollout_id = state.idempotency[key]
        attempt = state.attempts[rollout_id]
        return 202, {
            "rollout_id": rollout_id,
            "state": state.states[rollout_id],
            "idempotent_replay": True,
            "lease_expires_at_offset": state.leases.get(rollout_id, 0.0),
            "correlation": dict(attempt.correlation),
        }
    active = sum(
        1 for rid, value in state.states.items() if value in {"queued", "running", "awaiting_score"}
    )
    if active >= cfg.advertised_concurrency:
        raise _Refused(
            429,
            "concurrency_exhausted",
            f"{active} active attempts at advertised concurrency {cfg.advertised_concurrency}",
        )
    task_id = str(body.get("task_id") or "")
    if task_id not in cfg.task_ids:
        raise _Refused(404, "unknown_task", f"task {task_id!r} is not in the taskset")
    config_id = str(body.get("policy_config_id") or body.get("policy_set_id") or "")
    policy_config = state.policy_configs.get(config_id) or state.policy_sets.get(config_id)
    if policy_config is None:
        raise _Refused(409, "policy_not_bound", f"policy binding {config_id!r} is not bound")
    state.counter += 1
    rollout_id = f"ro_{digest([cfg.container_id, key, state.counter], length=16)}"
    attempt = _Attempt(
        rollout_id=rollout_id,
        idempotency_key=key,
        task_id=task_id,
        correlation=dict(correlation),
        handshake_id=str(body["handshake_id"]),
        policy_config=dict(policy_config),
        admitted_at=state.clock.now(),
    )
    state.attempts[rollout_id] = attempt
    state.idempotency[key] = rollout_id
    state.states[rollout_id] = "running"
    state.polls[rollout_id] = 0
    state.leases[rollout_id] = state.clock.now() + cfg.lease_ttl_seconds
    state.emit(rollout_id, "admitted", correlation=dict(correlation))
    return 202, {
        "rollout_id": rollout_id,
        "state": "running",
        "accepted": True,
        "idempotent_replay": False,
        "lease_expires_at": state.clock.rfc3339(cfg.lease_ttl_seconds),
        "lease_expires_at_offset": state.leases[rollout_id],
        "handshake_id": attempt.handshake_id,
        "correlation": dict(attempt.correlation),
        "group_pin_fields": _group_pin_fields(state, attempt),
    }


def _advance(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    current = state.states[rollout_id]
    if current in {"completed", "failed", "cancelled"}:
        return _state_payload(state, rollout_id)
    state.polls[rollout_id] += 1
    if state.clock.now() > state.leases[rollout_id]:
        state.states[rollout_id] = "failed"
        state.refusals[rollout_id] = "lease_expired"
        state.emit(rollout_id, "failure", code="lease_expired")
        return _state_payload(state, rollout_id)
    if state.polls[rollout_id] >= cfg.polls_until_terminal and current == "running":
        try:
            _seal(state, attempt)
        except _Refusal as refusal:
            state.states[rollout_id] = "failed"
            state.refusals[rollout_id] = refusal.code
            state.emit(rollout_id, "failure", code=refusal.code, reason=refusal.reason)
            return _state_payload(state, rollout_id)
        if cfg.deferred_scoring:
            state.states[rollout_id] = "awaiting_score"
            state.emit(rollout_id, "horizon_reached")
        else:
            state.states[rollout_id] = "scored"
            state.emit(rollout_id, "scored")
    return _state_payload(state, rollout_id)


def _state_payload(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    trace = state.traces.get(rollout_id)
    rows = (trace or {}).get("instances", [])
    present = {row["agent_instance_id"] for row in rows if row["present"]}
    return {
        "rollout_id": rollout_id,
        "state": state.states[rollout_id],
        "terminal": state.states[rollout_id] in {"completed", "failed", "cancelled"},
        "terminal_count": state.terminals.get(rollout_id, 0),
        "failure_code": state.refusals.get(rollout_id),
        "lease_expires_at": state.clock.rfc3339(state.leases[rollout_id] - state.clock.now()),
        "lease_expires_at_offset": state.leases[rollout_id],
        "lease_expired": state.clock.now() > state.leases[rollout_id],
        "handshake_id": attempt.handshake_id,
        "correlation": dict(attempt.correlation),
        "group_pin_fields": _group_pin_fields(state, attempt),
        "instance_liveness": [
            {
                "agent_instance_id": instance.agent_instance_id,
                "live": (
                    instance.agent_instance_id in present
                    if trace
                    else instance.agent_instance_id != cfg.defects.missing_instance_id
                ),
            }
            for instance in cfg.topology.agent_instances
        ],
    }


def _finalize(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
        payload = _state_payload(state, rollout_id)
        payload["already_terminal"] = True
        return payload
    if rollout_id not in state.traces:
        _seal(state, attempt)
    state.states[rollout_id] = "completed"
    state.emit(rollout_id, "episode", trace_digest=state.traces[rollout_id]["trace_digest"])
    payload = _state_payload(state, rollout_id)
    payload["already_terminal"] = False
    quiesced = cfg.quiescence_supported and not cfg.defects.unquiesced_deferred_program
    payload["snapshot"] = {
        "horizon_kind": cfg.horizon.horizon_kind,
        "horizon_value": cfg.horizon.value,
        "clipped": (not cfg.quiescence_supported)
        and not cfg.defects.unquiesced_deferred_program,
        "quiescence_attested": quiesced,
        "agent_authored_programs_killed": quiesced,
        "taken_at_offset": state.clock.now() - attempt.admitted_at,
    }
    payload["trace_digest"] = state.traces[rollout_id]["trace_digest"]
    return payload


def _handle(
    state: _State,
    method: str,
    path: str,
    query: Mapping[str, list[str]],
    body: Any,
) -> tuple[int, Any]:
    cfg = state.cfg
    body = body if isinstance(body, dict) else {}
    segments = [part for part in path.split("/") if part]

    if method == "GET" and path == "/metadata":
        return 200, {
            "metadata": {
                "optimizer_contracts": {
                    "cispo": {"version": CONTRACT_VERSION, **dict(DECLARED_ROUTES)}
                }
            }
        }
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "container_id": cfg.container_id,
            "container_version": cfg.taskset_version,
            "image_digest": cfg.image_digest,
            "contract_version": CONTRACT_VERSION,
        }
    if method == "GET" and path == "/training/capabilities":
        document = state.capability_document()
        return 200, {"capabilities": document, "capability_hash": state.capability_hash()}
    if method == "POST" and path == "/training/handshake":
        return _handshake(state, body)
    if method == "GET" and path == "/taskset":
        splits = dict(cfg.splits) or {"train": list(cfg.task_ids), "eval": list(cfg.task_ids[:1])}
        return 200, {
            "taskset_id": cfg.taskset_id,
            "version": cfg.taskset_version,
            "splits": {name: list(rows) for name, rows in splits.items()},
            "task_family": cfg.task_family,
        }
    if method == "GET" and path == "/taskset/tasks":
        requested = tuple(dict.fromkeys(_csv(query.get("ids")) or cfg.task_ids))
        unknown = [task_id for task_id in requested if task_id not in cfg.task_ids]
        if unknown:
            return 404, {"error": "unknown_task", "reason": f"{unknown}"}
        return 200, {
            "rows": [
                {
                    "task_id": task_id,
                    "topology_ref": cfg.topology.topology_id,
                    "task_family": cfg.task_family,
                    "seed": index,
                    "content_digest": digest(
                        [cfg.taskset_id, cfg.taskset_version, task_id], length=32
                    ),
                }
                for index, task_id in enumerate(requested)
            ]
        }
    if method == "GET" and len(segments) == 2 and segments[0] == "topologies":
        if segments[1] != cfg.topology.topology_id:
            return 404, {"error": "unknown_topology", "reason": segments[1]}
        payload = _topology_payload(cfg)
        payload["minimum_viable_roster"] = {
            team.team_id: team.minimum_viable_roster for team in cfg.topology.teams
        }
        return 200, payload
    if method == "POST" and path == "/policy-configs":
        kind = str(body.get("kind") or "trainable")
        if kind == "probe" and not cfg.probe_binding_supported:
            return 409, {"error": "probe_unsupported", "reason": "no probe binding kind"}
        transport = str(body.get("transport") or cfg.sampling_transport)
        if transport == "tokens_in_tokens_out" and not cfg.tito_supported:
            return 409, {"error": "transport_unsupported", "reason": transport}
        if any(name in body for name in ("api_key", "bearer_token", "credential")):
            return 400, {"error": "embedded_credential", "reason": "credentials never inline"}
        state.counter += 1
        config_id = f"pc_{digest([cfg.container_id, state.counter, body], length=16)}"
        record = {
            "config_id": config_id,
            "kind": kind,
            "probe": kind == "probe",
            "transport": transport,
            "policy_revision": int(body.get("policy_revision") or 0),
            "renderer_profile": _renderer_payload(cfg.renderer_profile),
            "sampler_ready": True,
            "sampler_origin": f"/samplers/{config_id}",
            "immutable": True,
        }
        state.policy_configs[config_id] = record
        return 200, record
    if method == "POST" and path == "/policy-sets":
        bindings = list(body.get("bindings") or [])
        declared = {
            instance.agent_instance_id: instance for instance in cfg.topology.agent_instances
        }
        bound_ids = {str(item.get("agent_instance_id")) for item in bindings}
        if bound_ids != set(declared):
            return 409, {
                "error": "partial_roster_binding",
                "reason": "no episode may start with a half-bound roster",
                "missing": sorted(set(declared) - bound_ids),
                "unknown": sorted(bound_ids - set(declared)),
            }
        for item in bindings:
            instance = declared[str(item["agent_instance_id"])]
            ref = str(item.get("policy_ref") or "")
            if not instance.trainable and ref.strip().lower() in ALIAS_REFS:
                return 409, {
                    "error": "alias_opponent_binding",
                    "reason": f"opponent {instance.agent_instance_id} may not resolve {ref!r}",
                }
        state.counter += 1
        set_id = f"ps_{digest([cfg.container_id, state.counter, body], length=16)}"
        record = {
            "config_id": set_id,
            "policy_set_id": set_id,
            "policy_set_revision_id": str(body.get("policy_set_revision_id") or set_id),
            "kind": str(body.get("kind") or "trainable"),
            "probe": str(body.get("kind") or "") == "probe",
            "transport": str(body.get("transport") or cfg.sampling_transport),
            "policy_revision": int(body.get("policy_revision") or 0),
            "atomic": True,
            "bindings": [
                {
                    "agent_instance_id": instance.agent_instance_id,
                    "trainable": instance.trainable,
                    "parameter_group_id": cfg.topology.parameter_groups.get(
                        instance.policy_type_id
                    ),
                    "pinned_identity": instance.pinned_identity,
                    "renderer_profile": _renderer_payload(cfg.renderer_profile),
                }
                for instance in cfg.topology.agent_instances
            ],
        }
        state.policy_sets[set_id] = record
        return 200, record
    if method == "POST" and path == "/rollout":
        return _submit(state, body)

    if segments[:1] == ["rollouts"] and len(segments) >= 2:
        rollout_id = segments[1]
        if rollout_id not in state.attempts:
            return 404, {"error": "unknown_rollout", "reason": rollout_id}
        tail = segments[2:]
        if method == "GET" and not tail:
            return 200, _advance(state, rollout_id)
        if method == "GET" and tail == ["events"]:
            cursor = int((query.get("cursor") or ["0"])[0])
            log = state.events.get(rollout_id, [])
            return 200, {
                "events": [event for event in log if event["cursor"] > cursor],
                "next_cursor": log[-1]["cursor"] if log else cursor,
            }
        if method == "POST" and tail == ["renew"]:
            if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
                return 409, {"error": "not_renewable", "reason": state.states[rollout_id]}
            if state.clock.now() > state.leases[rollout_id]:
                return 409, {"error": "lease_expired", "reason": "renew after expiry"}
            state.leases[rollout_id] = state.clock.now() + cfg.lease_ttl_seconds
            state.emit(rollout_id, "lease_renewed")
            return 200, {
                "rollout_id": rollout_id,
                "lease_expires_at": state.clock.rfc3339(cfg.lease_ttl_seconds),
                "lease_expires_at_offset": state.leases[rollout_id],
            }
        if method == "POST" and tail == ["finalize"]:
            return 200, _finalize(state, rollout_id)
        if method == "POST" and tail == ["terminate"]:
            if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
                payload = _state_payload(state, rollout_id)
                payload["already_terminal"] = True
                return 200, payload
            state.states[rollout_id] = "cancelled"
            state.refusals[rollout_id] = str(body.get("reason") or "cancelled")
            state.emit(rollout_id, "cancellation", reason=state.refusals[rollout_id])
            payload = _state_payload(state, rollout_id)
            payload["already_terminal"] = False
            return 200, payload
        if method == "GET" and tail == ["trace"]:
            trace = state.traces.get(rollout_id)
            if trace is None:
                return 409, {"error": "trace_not_sealed", "reason": state.states[rollout_id]}
            if cfg.artifact_by_reference:
                return 200, {
                    "rollout_id": rollout_id,
                    "inline": False,
                    "trace_digest": trace["trace_digest"],
                    "trace_ref": f"/rollouts/{rollout_id}/trace/body",
                }
            return 200, {"inline": True, **trace}
        if method == "GET" and tail == ["trace", "body"]:
            trace = state.traces.get(rollout_id)
            if trace is None:
                return 409, {"error": "trace_not_sealed", "reason": state.states[rollout_id]}
            return 200, {"inline": True, **trace}
        if method == "GET" and tail == ["artifacts"]:
            trace = state.traces.get(rollout_id)
            inventory = [
                {
                    "artifact_id": f"{rollout_id}:trace",
                    "role": "trace",
                    "digest": (trace or {}).get("trace_digest", ""),
                    "bytes": len(json.dumps(trace or {})),
                    "fetch_handle": f"/rollouts/{rollout_id}/trace/body",
                    "by_reference": cfg.artifact_by_reference,
                }
            ]
            if cfg.artifact_by_reference:
                inventory.append(
                    {
                        "artifact_id": f"{rollout_id}:recording",
                        "role": "recording",
                        "digest": digest([rollout_id, "recording"], length=32),
                        "bytes": 734003200,
                        "fetch_handle": f"/rollouts/{rollout_id}/trace/body",
                        "by_reference": True,
                    }
                )
            return 200, {"rollout_id": rollout_id, "artifacts": inventory}
        return 405, {"error": "route_not_declared", "reason": path}

    if path == "/reward" and method in {"GET", "POST"}:
        rollout_id = str(body.get("rollout_id") or (query.get("rollout_id") or [""])[0])
        if rollout_id not in state.attempts:
            return 404, {"error": "unknown_rollout", "reason": rollout_id}
        if cfg.deferred_scoring and state.states[rollout_id] != "completed":
            return 202, {
                "rollout_id": rollout_id,
                "state": "pending",
                "reason": "deferred verifier has not settled",
            }
        payload = state.rewards.get(rollout_id)
        if payload is None:
            if rollout_id not in state.traces:
                return 409, {"error": "reward_not_ready", "reason": state.states[rollout_id]}
            return 422, {
                "error": "missing_evidence",
                "reason": "no reward for a sealed trace; absent is not zero",
            }
        return 200, payload

    return 404, {"error": "route_not_declared", "reason": path}


def _csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    for value in values:
        out.extend(part for part in value.split(",") if part)
    return tuple(out)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _State

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        return

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "bad_json", "reason": "body is not JSON"})
            return
        with self.state.lock:
            self.state.request_log.append((method, parsed.path))
            try:
                status, payload = _handle(self.state, method, parsed.path, query, body)
            except _Refused as refused:
                status, payload = refused.status, {
                    "error": refused.code,
                    "reason": refused.reason,
                }
            except _Refusal as refusal:
                status, payload = 422, {"error": refusal.code, "reason": refusal.reason}
            except (RecordError, TopologyError) as error:
                status, payload = 500, {"error": "record_error", "reason": str(error)}
        self._respond(status, payload)

    def _respond(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """One attempt driven end to end through the declared routes."""

    rollout_id: str
    submit: Mapping[str, Any]
    states: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    finalize: Mapping[str, Any]
    trace: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    reward_payload: Mapping[str, Any] | None
    calls: tuple[InferenceCall, ...]
    episodes: tuple[TrainableEpisode, ...]

    @property
    def trainable_calls(self) -> tuple[InferenceCall, ...]:
        return tuple(call for call in self.calls if call.trainable)

    @property
    def reward(self) -> RewardRecord:
        if self.reward_payload is None:
            raise EvidenceError(
                f"rollout {self.rollout_id} produced no reward record; absent is not zero"
            )
        return reward_record_from_payload(self.reward_payload)

    @property
    def trace_digest(self) -> str:
        return str(self.trace.get("trace_digest") or "")

    def calls_for(self, agent_instance_id: str) -> tuple[InferenceCall, ...]:
        return tuple(
            call for call in self.calls if call.agent_instance_id == agent_instance_id
        )

    def episode_for(self, agent_instance_id: str) -> TrainableEpisode:
        for episode in self.episodes:
            if episode.agent_instance_id == agent_instance_id:
                return episode
        raise EvidenceError(
            f"rollout {self.rollout_id} has no trajectory for instance {agent_instance_id!r}"
        )


class ContainerClient:
    """Calls only routes the container declared in ``/metadata``."""

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.routes: dict[str, str] = {}
        self.handshake_id: str = ""
        self.agreement_digest: str = ""
        self._load_routes()

    # -- transport ------------------------------------------------------ #

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            payload = json.loads(error.read() or b"{}")
            return error.code, payload

    def call(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> Any:
        status, payload = self.request(method, path, body)
        if status >= 400:
            raise ContainerError(status, payload if isinstance(payload, dict) else {})
        return payload

    def route(self, key: str, **params: Any) -> str:
        if key not in self.routes:
            raise ContainerError(404, {"error": "route_not_declared", "reason": key})
        return self.routes[key].format(**params)

    def _load_routes(self) -> None:
        payload = self.call("GET", "/metadata")
        contract = payload["metadata"]["optimizer_contracts"]["cispo"]
        self.contract_version = contract["version"]
        self.routes = {key: value for key, value in contract.items() if key.endswith("_route")}

    # -- declared routes ------------------------------------------------ #

    def health(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("health_route"))

    def capabilities(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("capabilities_route"))

    def handshake(self, document: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = self.call("POST", self.route("handshake_route"), document)
        if payload.get("accepted"):
            self.handshake_id = payload["handshake_id"]
            self.agreement_digest = payload["agreement_digest"]
        return payload

    def taskset(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("taskset_route"))

    def taskset_tasks(self, ids: Iterable[str]) -> Mapping[str, Any]:
        query = urllib.parse.urlencode({"ids": ",".join(ids)})
        return self.call("GET", f"{self.route('taskset_tasks_route')}?{query}")

    def topology(self, topology_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("topology_route", topology_id=topology_id))

    def bind_policy(self, **body: Any) -> Mapping[str, Any]:
        return self.call("POST", self.route("policy_bind_route"), body)

    def bind_policy_set(self, **body: Any) -> Mapping[str, Any]:
        return self.call("POST", self.route("policy_set_bind_route"), body)

    def submit(self, **body: Any) -> Mapping[str, Any]:
        body.setdefault("handshake_id", self.handshake_id)
        body.setdefault("agreement_digest", self.agreement_digest)
        return self.call("POST", self.route("rollout_route"), body)

    def state(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("rollout_state_route", rollout_id=rollout_id))

    def events(self, rollout_id: str, cursor: int = 0) -> Mapping[str, Any]:
        path = self.route("rollout_events_route", rollout_id=rollout_id)
        return self.call("GET", f"{path}?cursor={cursor}")

    def renew(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("POST", self.route("rollout_renew_route", rollout_id=rollout_id), {})

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("POST", self.route("rollout_finalize_route", rollout_id=rollout_id), {})

    def terminate(self, rollout_id: str, reason: str = "cancelled") -> Mapping[str, Any]:
        path = self.route("rollout_terminate_route", rollout_id=rollout_id)
        return self.call("POST", path, {"reason": reason})

    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        payload = self.call("GET", self.route("trace_route", rollout_id=rollout_id))
        if payload.get("inline"):
            return payload
        body = self.call("GET", str(payload["trace_ref"]))
        if body["trace_digest"] != payload["trace_digest"]:
            raise EvidenceError("trace reference digest does not match its inventory entry")
        return body

    def artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("artifacts_route", rollout_id=rollout_id))

    def reward(self, rollout_id: str) -> tuple[int, Any]:
        return self.request("GET", f"{self.route('reward_route')}?rollout_id={rollout_id}")

    # -- convenience drivers -------------------------------------------- #

    def task_ids(self) -> tuple[str, ...]:
        rows = self.taskset_tasks(())["rows"]
        return tuple(str(row["task_id"]) for row in rows)

    def requirement_document(self, **overrides: Any) -> dict[str, Any]:
        capabilities = self.capabilities()["capabilities"]
        document: dict[str, Any] = {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "run_id": "run_fake",
            "optimizer": {"name": "synth_optimizers.cispo", "version": "0.0.0-test"},
            "policy": {
                "provider": "fake",
                "model_id": capabilities["renderer_profile"]["tokenizer_id"],
                "transport": "message_in_capture_out",
            },
            "renderer_profile": {
                "profile_id": capabilities["renderer_profile"]["profile_id"],
                "config_digest": capabilities["renderer_profile"]["config_digest"],
                "tokenizer_digest": capabilities["renderer_profile"]["tokenizer_digest"],
            },
            "requirements": list(MANDATORY_CLAUSES),
            "topology": {
                "expected_topology_id": capabilities["topology"]["topology_id"],
                "trainable_teams": [
                    team["team_id"]
                    for team in capabilities["topology"]["teams"]
                    if team["trainable"]
                ],
                "partial_roster": capabilities["topology"]["partial_roster_disposition"],
            },
            "run_plan": {
                "group_size": 2,
                "groups_per_step": 1,
                "max_execution_slots": 2,
                "maximum_policy_lag": 1,
                "target_train_updates": 1,
                "expected_horizon_seconds": capabilities["horizon"]["value"],
            },
            "taskset": {"taskset_id": self.taskset()["taskset_id"], "split": "train"},
            "clock": {"executor_time": "2026-09-02T12:00:00+00:00"},
        }
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(document.get(key), dict):
                document[key] = {**document[key], **value}
            else:
                document[key] = value
        return document

    def preflight(self, **overrides: Any) -> Mapping[str, Any]:
        """Health, metadata, capabilities, handshake -- in the declared order."""

        self.health()
        self.capabilities()
        return self.handshake(self.requirement_document(**overrides))

    def negotiate(self, **overrides: Any) -> tuple[Mapping[str, Any], ...]:
        """Preflight, then re-handshake once per degradation. Never assume.

        A degraded concurrency clause is satisfied by *lowering the run plan*
        and re-handshaking; any other degradation is re-handshaked with the
        fallback named explicitly in ``accept_degraded``. Returns every
        handshake exchange in order, so a receipt can name them all.
        """

        exchanges = [self.preflight(**overrides)]
        latest = exchanges[-1]
        if latest.get("accepted"):
            return tuple(exchanges)
        if latest["rejected_mandatory_clauses"]:
            return tuple(exchanges)
        degraded = list(latest["unaccepted_degraded_clauses"])
        lowered = dict(overrides)
        if "lifecycle.concurrency" in degraded:
            ceiling = int(latest["obligations"]["max_concurrency"])
            plan = dict(lowered.get("run_plan") or {})
            plan.update({"max_execution_slots": ceiling, "group_size": ceiling})
            lowered["run_plan"] = plan
            degraded = [clause for clause in degraded if clause != "lifecycle.concurrency"]
        lowered["accept_degraded"] = degraded
        exchanges.append(self.handshake(self.requirement_document(**lowered)))
        return tuple(exchanges)

    def bind(
        self, *, probe: bool = False, policy_revision: int = 0, **extra: Any
    ) -> Mapping[str, Any]:
        """Bind one policy, or a whole roster when the topology is joint."""

        capabilities = self.capabilities()["capabilities"]
        instances = capabilities["topology"]["agent_instances"]
        kind = "probe" if probe else "trainable"
        if len(instances) < 2:
            return self.bind_policy(kind=kind, policy_revision=policy_revision, **extra)
        return self.bind_policy_set(
            kind=kind,
            policy_revision=policy_revision,
            policy_set_revision_id=extra.pop("policy_set_revision_id", "policy-set-1"),
            bindings=[
                {
                    "agent_instance_id": instance["agent_instance_id"],
                    "policy_ref": instance["policy_ref"] or f"ckpt::rev{policy_revision}",
                }
                for instance in instances
            ],
            **extra,
        )

    def run_attempt(
        self,
        *,
        task_id: str,
        idempotency_key: str | None = None,
        correlation: Mapping[str, Any] | None = None,
        binding: Mapping[str, Any] | None = None,
        probe: bool = False,
        polls: int = 4,
        renew: bool = True,
    ) -> AttemptResult:
        """Submit, poll, renew, finalize, read trace and reward. One attempt."""

        if not self.handshake_id:
            self.preflight()
        record = binding or self.bind(probe=probe)
        key = idempotency_key or f"key::{task_id}::{record['config_id']}"
        payload = dict(correlation or {})
        payload.setdefault("run_id", "run_fake")
        payload.setdefault("group_id", "group_fake")
        payload.setdefault("sample_index", 0)
        payload.setdefault("seed", 7)
        payload.setdefault("policy_revision", int(record.get("policy_revision") or 0))
        if "policy_set_revision_id" in record:
            payload.setdefault("policy_set_revision", record["policy_set_revision_id"])
        submit = self.submit(
            task_id=task_id,
            idempotency_key=key,
            policy_config_id=record["config_id"],
            correlation=payload,
        )
        rollout_id = str(submit["rollout_id"])
        if renew:
            self.renew(rollout_id)
        seen: list[Mapping[str, Any]] = []
        for _ in range(polls):
            snapshot = self.state(rollout_id)
            seen.append(snapshot)
            if snapshot["state"] in {"scored", "awaiting_score"} or snapshot["terminal"]:
                break
        finalize = self.finalize(rollout_id)
        trace = self.trace(rollout_id)
        artifacts = self.artifacts(rollout_id)
        status, reward_payload = self.reward(rollout_id)
        events = self.events(rollout_id)["events"]
        return AttemptResult(
            rollout_id=rollout_id,
            submit=submit,
            states=tuple(seen),
            events=tuple(events),
            finalize=finalize,
            trace=trace,
            artifacts=artifacts,
            reward_payload=reward_payload if status == 200 else None,
            calls=tuple(inference_call_from_payload(row) for row in trace["calls"]),
            episodes=tuple(
                trainable_episode_from_payload(row) for row in trace.get("episodes") or ()
            ),
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


class RunningContainer:
    """A live fake. ``base_url`` plus ``shutdown()`` is the whole contract."""

    __slots__ = ("config", "clock", "_state", "_server", "_thread")

    def __init__(self, config: ContainerConfig, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self._state = _State(config, clock)
        handler = type("_BoundHandler", (_Handler,), {"state": self._state})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def declared_routes(self) -> Mapping[str, str]:
        return dict(DECLARED_ROUTES)

    @property
    def requested_paths(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._state.request_log)

    def client(self, **kwargs: Any) -> ContainerClient:
        return ContainerClient(self.base_url, **kwargs)

    def revoke_handshake(self, handshake_id: str) -> None:
        """The container may revoke a handshake when it degrades."""

        with self._state.lock:
            self._state.revoked.add(handshake_id)

    def bump_capability_epoch(self) -> str:
        """Change the capability document so a renewal must fail closed."""

        with self._state.lock:
            self._state.capability_epoch += 1
            return self._state.capability_hash()

    @property
    def attempt_count(self) -> int:
        """How many logical attempts the container actually admitted."""

        with self._state.lock:
            return len(self._state.attempts)

    def terminal_count(self, rollout_id: str) -> int:
        with self._state.lock:
            return self._state.terminals.get(rollout_id, 0)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "RunningContainer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()


def serve(config: ContainerConfig, *, clock: Clock | None = None) -> RunningContainer:
    """Start one fake container.

    The whole construction API: pass a configuration, get back a running
    server with ``base_url``, ``client()``, and ``shutdown()``. Bound to
    ``127.0.0.1`` on an ephemeral port; no outbound network, no real sleeping.
    """

    return RunningContainer(config, clock or Clock())
