"""Wire codecs: shared records in, JSON out, and back again losslessly.

The decoders are the half a caller needs: they rebuild the shared record types
from a container response, and they refuse what no record may express -- an
opponent resolved by alias rather than by immutable identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    CommunicationChannel,
    GroupPin,
    Horizon,
    RolloutReceipt,
    Team,
    Topology,
    TopologyError,
)
from synth_optimizers.contracts.rl_records import (
    CompactionProvenance,
    HorizonEvidence,
    InferenceCall,
    RendererProfile,
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
)

from .config import ALIAS_REFS, ContainerConfig

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
        "renderer_profile_fingerprint": call.renderer_profile_fingerprint,
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
        renderer_profile_fingerprint=str(payload.get("renderer_profile_fingerprint") or ""),
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


def _segment_payload(segment: TrainableSegment) -> dict[str, Any]:
    """One trainer sequence. ``author_kind`` is explicit, never implied."""

    return {
        "token_ids": list(segment.token_ids),
        "loss_mask": list(segment.loss_mask),
        "behavior_logprobs": list(segment.behavior_logprobs),
        "branch_id": segment.branch_id,
        "parameter_group_id": segment.parameter_group_id,
        "agent_instance_id": segment.agent_instance_id,
        "call_ids": list(segment.call_ids),
        "author_kind": segment.author_kind,
        "role_id": segment.role_id,
        "policy_type_id": segment.policy_type_id,
        "team_id": segment.team_id,
        "policy_revision": segment.policy_revision,
        "policy_set_revision_id": segment.policy_set_revision_id,
        "effect_tick_start": segment.effect_tick_start,
        "effect_tick_end": segment.effect_tick_end,
    }


def segment_from_payload(payload: Mapping[str, Any]) -> TrainableSegment:
    return TrainableSegment(
        token_ids=tuple(int(item) for item in payload["token_ids"]),
        loss_mask=tuple(int(item) for item in payload["loss_mask"]),
        behavior_logprobs=tuple(float(item) for item in payload["behavior_logprobs"]),
        branch_id=str(payload.get("branch_id") or "root"),
        parameter_group_id=payload.get("parameter_group_id"),
        agent_instance_id=payload.get("agent_instance_id"),
        call_ids=tuple(str(item) for item in payload.get("call_ids") or ()),
        author_kind=str(payload.get("author_kind") or "policy"),
        role_id=payload.get("role_id"),
        policy_type_id=payload.get("policy_type_id"),
        team_id=payload.get("team_id"),
        policy_revision=payload.get("policy_revision"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        effect_tick_start=payload.get("effect_tick_start"),
        effect_tick_end=payload.get("effect_tick_end"),
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
        "root_rollout_id": episode.root_rollout_id,
        "trace_digest": episode.trace_digest,
        "probe": episode.probe,
        "segments": [_segment_payload(segment) for segment in episode.segments],
    }


def trainable_episode_from_payload(payload: Mapping[str, Any]) -> TrainableEpisode:
    return TrainableEpisode(
        rollout_id=str(payload["rollout_id"]),
        task_id=str(payload["task_id"]),
        seed=int(payload.get("seed") or 0),
        policy_revision=int(payload["policy_revision"]),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        segments=tuple(
            segment_from_payload(raw) for raw in payload.get("segments") or ()
        ),
        terminal_status=str(payload["terminal_status"]),
        usage=dict(payload.get("usage") or {}),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        root_rollout_id=payload.get("root_rollout_id"),
        trace_digest=str(payload.get("trace_digest") or ""),
        probe=bool(payload.get("probe")),
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


def _receipt_payload(receipt: RolloutReceipt) -> dict[str, Any]:
    return {
        "rollout_id": receipt.rollout_id,
        "proxy_request_id": receipt.proxy_request_id,
        "group_id": receipt.group_id,
        "sample_index": receipt.sample_index,
        "policy_revision": receipt.policy_revision,
        "behavior_fingerprint": receipt.behavior_fingerprint,
        "terminal_status": receipt.terminal_status,
        "trace_digest": receipt.trace_digest,
        "evidence_digest": receipt.evidence_digest,
        "reward_id": receipt.reward_id,
        "handshake_id": receipt.handshake_id,
        "agreement_digest": receipt.agreement_digest,
        "agent_instance_id": receipt.agent_instance_id,
        "team_id": receipt.team_id,
        "probe": receipt.probe,
        "replaced_attempt_id": receipt.replaced_attempt_id,
        "replacement_index": receipt.replacement_index,
        "replacement_reason": receipt.replacement_reason,
        "metadata": dict(receipt.metadata),
    }


def rollout_receipt_from_payload(payload: Mapping[str, Any]) -> RolloutReceipt:
    """What leaves the done boundary: identity plus digests, never raw tokens."""

    return RolloutReceipt(
        rollout_id=str(payload["rollout_id"]),
        proxy_request_id=str(payload["proxy_request_id"]),
        group_id=str(payload.get("group_id") or ""),
        sample_index=int(payload.get("sample_index") or 0),
        policy_revision=int(payload.get("policy_revision") or 0),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        terminal_status=str(payload["terminal_status"]),
        trace_digest=str(payload.get("trace_digest") or ""),
        evidence_digest=str(payload.get("evidence_digest") or ""),
        reward_id=payload.get("reward_id"),
        handshake_id=str(payload.get("handshake_id") or ""),
        agreement_digest=str(payload.get("agreement_digest") or ""),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        probe=bool(payload.get("probe")),
        replaced_attempt_id=payload.get("replaced_attempt_id"),
        replacement_index=int(payload.get("replacement_index") or 0),
        replacement_reason=payload.get("replacement_reason"),
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
                seconds_per_unit=(
                    None
                    if horizon.get("seconds_per_unit") is None
                    else float(horizon["seconds_per_unit"])
                ),
            )
        ),
        parameter_groups=dict(payload.get("parameter_groups") or {}),
    )


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
