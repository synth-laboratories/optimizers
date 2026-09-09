"""Deterministic evidence synthesis.

Rendered prompt pieces derive from the task row, the renderer profile, the
agent instance, and the turn. Sampled generations and their logprobs derive
additionally from ``ContainerConfig.seed``, which is what makes one attempt
reproducible bit-for-bit. Nothing here touches the network or the clock.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synth_optimizers.contracts.rl_identity import AgentInstance, Team
from synth_optimizers.contracts.rl_records import (
    LOGPROB_SENTINEL,
    BehaviorFingerprint,
    CompactionProvenance,
    HorizonEvidence,
    InferenceCall,
    RewardChannel,
    RewardRecord,
    SamplingProfile,
    TrainableEpisode,
    TrainableSegment,
    digest,
)

from .config import ContainerConfig

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
    #: A straggler replacement is recorded as such rather than silently
    #: changing group membership.
    replaced_attempt_id: str | None = None
    replacement_index: int = 0
    replacement_reason: str | None = None


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
    #: Foreign authorship is declared, never implied by a zero mask. An
    #: opponent instance authored its own tokens, so the policy under training
    #: never sampled them.
    authored_by_policy = instance.trainable if instance else True
    trainable = authored_by_policy
    if probe and not cfg.defects.probe_indistinguishable:
        trainable = False
    revision = int(attempt.correlation.get("policy_revision") or 0)
    fingerprint = _behavior(cfg, revision, transport).value
    profile_key = cfg.renderer_profile.fingerprint
    system = _tokens(4, "system", profile_key)
    opening = _tokens(8, "task", attempt.task_id, profile_key, instance_id)

    if probe:
        provenance = "engine_meta" if cfg.defects.probe_indistinguishable else "probe_synthetic"
    elif authored_by_policy:
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
        sampled_mask = tuple(1 if authored_by_policy else 0 for _ in generated)
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
            renderer_profile_fingerprint=profile_key,
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
        renderer_profile_fingerprint=cfg.renderer_profile.fingerprint,
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


def _author_kind(call: InferenceCall) -> str:
    """Who sampled these tokens. Declared, never inferred from a zero mask."""

    if call.role_id == "judge" or call.policy_type_id == "judge":
        return "judge"
    if call.role_id == "verifier":
        return "verifier"
    if not call.trainable and call.token_capture_provenance == "wire_derived":
        return "opponent"
    return "policy"


def _stitch(calls: Sequence[InferenceCall], *, author_kind: str = "policy") -> TrainableSegment:
    """One contiguous trainer sequence. Retained prefixes are loss-masked.

    A segment authored by anything but the policy is emitted with a fully zero
    mask: the shared record refuses foreign authorship that carries trainable
    tokens, and that refusal is the point.
    """

    final = calls[-1]
    total = final.full_sequence
    mask = [0] * len(total)
    logprobs = [0.0] * len(total)
    policy_authored = author_kind == "policy"
    for call in calls:
        start = len(call.prompt_token_ids)
        if start + len(call.generation_token_ids) > len(total):
            continue
        flags = call.sampled_mask or tuple(1 for _ in call.generation_token_ids)
        for offset, flag in enumerate(flags):
            mask[start + offset] = int(bool(flag)) if policy_authored else 0
        for offset, value in enumerate(call.generation_logprobs):
            if start + offset < len(logprobs):
                logprobs[start + offset] = value
    starts = [c.effect_tick_start for c in calls if c.effect_tick_start is not None]
    ends = [c.effect_tick_end for c in calls if c.effect_tick_end is not None]
    return TrainableSegment(
        token_ids=total,
        loss_mask=tuple(mask),
        behavior_logprobs=tuple(logprobs),
        branch_id=final.branch_id,
        parameter_group_id=final.parameter_group_id,
        agent_instance_id=final.agent_instance_id,
        call_ids=tuple(call.call_id for call in calls),
        author_kind=author_kind,
        role_id=final.role_id,
        policy_type_id=final.policy_type_id,
        team_id=final.team_id,
        policy_revision=final.policy_revision,
        policy_set_revision_id=final.policy_set_revision_id,
        effect_tick_start=min(starts) if starts else None,
        effect_tick_end=max(ends) if ends else None,
    )


def _segments_by_branch(
    calls: Sequence[InferenceCall], *, author_kind: str
) -> tuple[TrainableSegment, ...]:
    branches: dict[str, list[InferenceCall]] = {}
    for call in calls:
        branches.setdefault(call.branch_id, []).append(call)
    return tuple(
        _stitch(group, author_kind=author_kind) for group in branches.values()
    )


def _episodes(
    cfg: ContainerConfig,
    attempt: _Attempt,
    calls: Sequence[InferenceCall],
    *,
    trace_digest: str,
    probe: bool = False,
) -> list[TrainableEpisode]:
    """One trajectory per policy-authored agent instance.

    A probe attempt still produces episodes -- marking them ``probe`` is what
    keeps them out of a group, and an empty list would hide the fact that the
    container produced evidence at all.
    """

    policy_calls = [call for call in calls if _author_kind(call) == "policy"]
    by_instance: dict[str | None, list[InferenceCall]] = {}
    for call in policy_calls:
        by_instance.setdefault(call.agent_instance_id, []).append(call)
    episodes: list[TrainableEpisode] = []
    for instance_id, instance_calls in by_instance.items():
        head = instance_calls[0]
        episodes.append(
            TrainableEpisode(
                rollout_id=attempt.rollout_id,
                task_id=attempt.task_id,
                seed=int(attempt.correlation.get("seed") or 0),
                policy_revision=head.policy_revision,
                behavior_fingerprint=head.behavior_fingerprint,
                segments=_segments_by_branch(instance_calls, author_kind="policy"),
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
                root_rollout_id=attempt.rollout_id,
                trace_digest=trace_digest,
                probe=probe,
            )
        )
    return episodes


def _context_segments(calls: Sequence[InferenceCall]) -> list[TrainableSegment]:
    """Foreign-authored spans, recorded as untrainable context.

    An opponent's, another instance's, a verifier's, or a judge's tokens are
    never trainable for the policy under training, and their author is named
    rather than left to be guessed from a zero mask.
    """

    grouped: dict[tuple[str, str | None], list[InferenceCall]] = {}
    for call in calls:
        author = _author_kind(call)
        if author == "policy":
            continue
        grouped.setdefault((author, call.agent_instance_id), []).append(call)
    segments: list[TrainableSegment] = []
    for (author, _instance_id), group in grouped.items():
        segments.extend(_segments_by_branch(group, author_kind=author))
    return segments


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
    # The declared measure for *this* attempt: one constant for every attempt
    # would leave every group tied and no group with an ordering to credit.
    value = cfg.reward_for(
        attempt.task_id, int(attempt.correlation.get("sample_index") or 0)
    )
    channels: list[RewardChannel] = []
    if competitive:
        ordered = sorted(teams, key=lambda team: (not team.trainable, team.team_id))
        for rank, team in enumerate(ordered, start=1):
            measure = round(value / rank, 6)
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
                measure=value,
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
