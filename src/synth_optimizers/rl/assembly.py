"""Batch assembly: validated evidence in, a provider-ready batch out.

A batch is built from ``TrainableEpisode`` segments and ``RewardRecord``
receipts and from nothing else. Every record is validated before it is used,
group uniformity is asserted rather than assumed, and a span that this
parameter group did not author never reaches this parameter group's batch.

Nothing here branches on which algorithm is running. The plan supplies the
credit estimator, the zero-advantage policy, the staleness bound, the
same-policy reduction, and the packing, and this module reads those fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.rl_identity import GroupPin, Topology, assert_uniform_group
from ..contracts.rl_records import (
    LOGPROB_SENTINEL,
    EvidenceError,
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
    digest,
)
from .credit import (
    CreditSample,
    InstanceStream,
    SamePolicyReduction,
    TeamAdvantageFanout,
    estimate,
    fan_out_team_advantage,
    reduce_same_policy,
)
from .plan import AlgorithmPlan

#: The parameter group a single-policy run trains. A solo run still names its
#: parameter group, so a solo batch and a joint batch have the same shape.
SOLO_PARAMETER_GROUP = "actor"

DROP_FOREIGN_AUTHOR = "foreign_author"
DROP_UNATTRIBUTED_AUTHOR = "unattributed_author"
DROP_NO_TRAINABLE_TOKENS = "no_trainable_tokens"
DROP_STALE = "staleness_bound"


class AssemblyError(EvidenceError):
    """Evidence that cannot be assembled. Never degraded to zero reward."""


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One scored attempt: its episode, its reward receipt, and its group pin."""

    group_id: str
    sample_index: int
    pin: GroupPin
    episode: TrainableEpisode
    reward: RewardRecord
    #: The environment attempt this episode's branches belong to. Several
    #: branches of one attempt must not multiply that attempt's weight.
    root_rollout_id: str | None = None
    #: Container-declared roster. Absent for a single-policy run.
    topology: Topology | None = None
    #: Published revisions between the sampling policy and the training policy.
    staleness_steps: int = 0
    source_run_id: str = ""

    def __post_init__(self) -> None:
        if self.pin.group_id != self.group_id:
            raise AssemblyError(
                f"bundle group {self.group_id!r} does not match its pin "
                f"{self.pin.group_id!r}"
            )
        if self.sample_index < 0:
            raise AssemblyError("sample_index must be non-negative")
        if self.staleness_steps < 0:
            raise AssemblyError("staleness_steps must be non-negative")

    @property
    def root_id(self) -> str:
        return self.root_rollout_id or self.episode.rollout_id


@dataclass(frozen=True, slots=True)
class DroppedSpan:
    """A span that did not enter a batch, and the reason it did not."""

    rollout_id: str
    group_id: str
    segment_index: int
    reason: str
    agent_instance_id: str | None = None
    parameter_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class SpanBatchItem:
    """One trainable span, ready for one provider step."""

    parameter_group_id: str
    group_id: str
    rollout_id: str
    root_rollout_id: str
    sample_index: int
    branch_id: str
    agent_instance_id: str | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    advantage: float
    #: ``1 / branches`` of this attempt inside this parameter group.
    root_rollout_weight: float
    #: Share of the parameter group's update, from the same-policy reduction.
    same_policy_weight: float
    trainable_tokens: int
    policy_revision: int
    staleness_steps: int
    call_ids: tuple[str, ...] = ()

    def identity(self) -> dict[str, Any]:
        """Everything that makes this item this item, for composition digests."""

        return {
            "parameter_group_id": self.parameter_group_id,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "root_rollout_id": self.root_rollout_id,
            "sample_index": self.sample_index,
            "branch_id": self.branch_id,
            "agent_instance_id": self.agent_instance_id,
            "token_ids": list(self.token_ids),
            "loss_mask": list(self.loss_mask),
            "behavior_logprobs": list(self.behavior_logprobs),
            "advantage": repr(self.advantage),
            "root_rollout_weight": repr(self.root_rollout_weight),
            "same_policy_weight": repr(self.same_policy_weight),
            "trainable_tokens": self.trainable_tokens,
            "policy_revision": self.policy_revision,
            "staleness_steps": self.staleness_steps,
        }


@dataclass(frozen=True, slots=True)
class GroupAdvantageProvenance:
    """Per-group advantage provenance: how each advantage came to exist."""

    group_id: str
    plan_hash: str
    pin_digest: str
    credit_kind: str
    optimized_channel: str
    resolved_channel: str
    topology_id: str | None
    rollout_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    lengths: tuple[int, ...]
    advantages: tuple[float, ...]
    zero_variance: bool
    skipped: bool
    fanout_parameter_groups: tuple[str, ...]
    same_policy_reduction: str
    reward_ids: tuple[str, ...] = ()
    trace_digests: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "plan_hash": self.plan_hash,
            "pin_digest": self.pin_digest,
            "credit_kind": self.credit_kind,
            "optimized_channel": self.optimized_channel,
            "resolved_channel": self.resolved_channel,
            "topology_id": self.topology_id,
            "rollout_ids": list(self.rollout_ids),
            "rewards": [repr(value) for value in self.rewards],
            "lengths": list(self.lengths),
            "advantages": [repr(value) for value in self.advantages],
            "zero_variance": self.zero_variance,
            "skipped": self.skipped,
            "fanout_parameter_groups": list(self.fanout_parameter_groups),
            "same_policy_reduction": self.same_policy_reduction,
            "reward_ids": list(self.reward_ids),
            "trace_digests": list(self.trace_digests),
        }


@dataclass(frozen=True, slots=True)
class ProviderStep:
    """One provider training call. It may carry several groups."""

    step_index: int
    parameter_group_id: str
    group_ids: tuple[str, ...]
    items: tuple[SpanBatchItem, ...]

    @property
    def trainable_tokens(self) -> int:
        return sum(item.trainable_tokens for item in self.items)


@dataclass(frozen=True, slots=True)
class ParameterGroupBatch:
    """Everything one trainable component trains on this round."""

    parameter_group_id: str
    steps: tuple[ProviderStep, ...]
    same_policy: SamePolicyReduction

    @property
    def items(self) -> tuple[SpanBatchItem, ...]:
        return tuple(item for step in self.steps for item in step.items)

    @property
    def group_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for step in self.steps:
            for group_id in step.group_ids:
                if group_id not in seen:
                    seen.append(group_id)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """A provider-ready round, with the provenance of every advantage in it."""

    plan_hash: str
    round_index: int
    parameter_groups: tuple[ParameterGroupBatch, ...]
    provenance: tuple[GroupAdvantageProvenance, ...]
    dropped_spans: tuple[DroppedSpan, ...] = ()
    dropped_bundles: tuple[DroppedSpan, ...] = ()
    off_policy: bool = False
    source_run_ids: tuple[str, ...] = ()
    accepted_staleness: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> tuple[ProviderStep, ...]:
        return tuple(step for batch in self.parameter_groups for step in batch.steps)

    @property
    def items(self) -> tuple[SpanBatchItem, ...]:
        return tuple(item for batch in self.parameter_groups for item in batch.items)

    def batch_for(self, parameter_group_id: str) -> ParameterGroupBatch:
        for batch in self.parameter_groups:
            if batch.parameter_group_id == parameter_group_id:
                return batch
        raise AssemblyError(f"batch has no parameter group {parameter_group_id!r}")

    def provenance_for(self, group_id: str) -> GroupAdvantageProvenance:
        for record in self.provenance:
            if record.group_id == group_id:
                return record
        raise AssemblyError(f"batch has no provenance for group {group_id!r}")

    def advantage_payload(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.provenance]

    def composition_payload(self) -> list[dict[str, Any]]:
        """Batch composition, excluding whether the batch was produced online."""

        return [
            {
                "parameter_group_id": batch.parameter_group_id,
                "same_policy": batch.same_policy.receipt(),
                "steps": [
                    {
                        "step_index": step.step_index,
                        "group_ids": list(step.group_ids),
                        "items": [item.identity() for item in step.items],
                    }
                    for step in batch.steps
                ],
            }
            for batch in self.parameter_groups
        ]

    @property
    def advantage_digest(self) -> str:
        return digest(
            {"plan_hash": self.plan_hash, "advantages": self.advantage_payload()}, length=32
        )

    @property
    def composition_digest(self) -> str:
        return digest(
            {"plan_hash": self.plan_hash, "composition": self.composition_payload()}, length=32
        )


# --- Validation helpers ------------------------------------------------------


def _validate_segment_logprobs(segment: TrainableSegment, rollout_id: str, index: int) -> None:
    """Sentinel and malformed logprobs are refused before a batch exists."""

    if segment.trainable_tokens == 0:
        # Nothing here can enter a batch; the span is dropped with a reason.
        return
    for position, (flag, value) in enumerate(
        zip(segment.loss_mask, segment.behavior_logprobs, strict=True)
    ):
        if not flag:
            continue
        if math.isnan(value) or math.isinf(value):
            raise AssemblyError(
                f"episode {rollout_id} segment {index} logprob {position} is not finite"
            )
        if value == LOGPROB_SENTINEL:
            raise AssemblyError(
                f"episode {rollout_id} segment {index} logprob {position} is the provider "
                f"sentinel {LOGPROB_SENTINEL}; its presence cannot prove a real logprob"
            )
    if all(
        value == 0.0
        for flag, value in zip(segment.loss_mask, segment.behavior_logprobs, strict=True)
        if flag
    ):
        raise AssemblyError(
            f"episode {rollout_id} segment {index} has identically zero behavior logprobs"
        )


def _resolve_channel(reward: RewardRecord, team_id: str | None) -> RewardChannel:
    """Which declared channel the group is optimized on. Recorded, not guessed."""

    by_id = {channel.channel_id: channel for channel in reward.channels}
    optimized = by_id.get(reward.optimized_channel)
    if optimized is None:
        raise AssemblyError(
            f"reward {reward.reward_id} names channel {reward.optimized_channel!r} "
            "which it does not carry"
        )
    if team_id is None or optimized.team_id == team_id or optimized.team_id is None:
        # A channel that names no team is the run's single measure, and it
        # applies to whichever team stamped the trajectory. A team becomes a
        # comparison key only where the reward actually separates teams; a
        # cooperative container that stamps its one team on every episode and
        # reports one untargeted measure is the common case, not an error.
        return optimized
    candidates = [channel for channel in reward.channels if channel.team_id == team_id]
    if len(candidates) == 1:
        return candidates[0]
    raise AssemblyError(
        f"reward {reward.reward_id} has {len(candidates)} channels for team {team_id!r}; "
        "the optimized channel for a team must be unambiguous"
    )


def _staleness_admits(plan: AlgorithmPlan, bundle: EvidenceBundle) -> bool:
    bound = plan.correction.max_weight_staleness
    if plan.correction.kind == "staleness_drop" and plan.correction.enabled:
        return bundle.staleness_steps <= bound
    if bundle.staleness_steps > bound:
        raise AssemblyError(
            f"episode {bundle.episode.rollout_id} is {bundle.staleness_steps} revisions stale "
            f"but correction {plan.correction.kind!r} admits at most {bound}"
        )
    return True


def _trainee_instances(topology: Topology) -> frozenset[str]:
    return frozenset(
        instance.agent_instance_id for instance in topology.trainable_instances
    )


def _span_parameter_group(
    bundle: EvidenceBundle, segment: TrainableSegment, index: int
) -> tuple[str | None, str | None]:
    """Resolve a span's parameter group, or the reason it is not trainable here."""

    episode = bundle.episode
    topology = bundle.topology
    if topology is None:
        author = segment.agent_instance_id
        if author is not None and episode.agent_instance_id is not None:
            if author != episode.agent_instance_id:
                return None, DROP_FOREIGN_AUTHOR
        return segment.parameter_group_id or SOLO_PARAMETER_GROUP, None
    author = segment.agent_instance_id
    if author is None:
        return None, DROP_UNATTRIBUTED_AUTHOR
    if author not in _trainee_instances(topology):
        return None, DROP_FOREIGN_AUTHOR
    declared = topology.parameter_group_for(author)
    if segment.parameter_group_id is not None and segment.parameter_group_id != declared:
        raise AssemblyError(
            f"episode {episode.rollout_id} segment {index} claims parameter group "
            f"{segment.parameter_group_id!r} but the topology declares {declared!r} for "
            f"instance {author!r}"
        )
    return declared, None


# --- Assembly ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Span:
    bundle: EvidenceBundle
    segment: TrainableSegment
    segment_index: int
    parameter_group_id: str


def _validated_bundles(
    plan: AlgorithmPlan, bundles: Sequence[EvidenceBundle]
) -> tuple[list[EvidenceBundle], list[DroppedSpan]]:
    admitted: list[EvidenceBundle] = []
    dropped: list[DroppedSpan] = []
    for bundle in bundles:
        episode = bundle.episode
        episode.validate()
        bundle.reward.validate(episode_trace_digest=episode.trace_digest)
        if bundle.pin.algorithm_plan_hash != plan.plan_hash:
            raise AssemblyError(
                f"group {bundle.group_id} was produced under plan hash "
                f"{bundle.pin.algorithm_plan_hash!r} but this batch runs {plan.plan_hash!r}"
            )
        if bundle.pin.behavior_fingerprint != episode.behavior_fingerprint:
            raise AssemblyError(
                f"episode {episode.rollout_id} behavior fingerprint does not match its pin"
            )
        for index, segment in enumerate(episode.segments):
            _validate_segment_logprobs(segment, episode.rollout_id, index)
        if not _staleness_admits(plan, bundle):
            dropped.append(
                DroppedSpan(
                    rollout_id=episode.rollout_id,
                    group_id=bundle.group_id,
                    segment_index=-1,
                    reason=DROP_STALE,
                )
            )
            continue
        admitted.append(bundle)
    return admitted, dropped


def _split_spans(
    bundles: Sequence[EvidenceBundle],
) -> tuple[list[_Span], list[DroppedSpan]]:
    spans: list[_Span] = []
    dropped: list[DroppedSpan] = []
    for bundle in bundles:
        for index, segment in enumerate(bundle.episode.segments):
            parameter_group, reason = _span_parameter_group(bundle, segment, index)
            if parameter_group is None:
                dropped.append(
                    DroppedSpan(
                        rollout_id=bundle.episode.rollout_id,
                        group_id=bundle.group_id,
                        segment_index=index,
                        reason=str(reason),
                        agent_instance_id=segment.agent_instance_id,
                        parameter_group_id=segment.parameter_group_id,
                    )
                )
                continue
            if segment.trainable_tokens == 0:
                dropped.append(
                    DroppedSpan(
                        rollout_id=bundle.episode.rollout_id,
                        group_id=bundle.group_id,
                        segment_index=index,
                        reason=DROP_NO_TRAINABLE_TOKENS,
                        agent_instance_id=segment.agent_instance_id,
                        parameter_group_id=parameter_group,
                    )
                )
                continue
            spans.append(
                _Span(
                    bundle=bundle,
                    segment=segment,
                    segment_index=index,
                    parameter_group_id=parameter_group,
                )
            )
    return spans, dropped


def _group_credit(
    plan: AlgorithmPlan,
    group_id: str,
    members: Sequence[EvidenceBundle],
    spans: Sequence[_Span],
) -> tuple[TeamAdvantageFanout, GroupAdvantageProvenance]:
    pin = assert_uniform_group([bundle.pin for bundle in members])
    tokens_by_rollout: dict[str, int] = {}
    groups_by_rollout: dict[str, list[str]] = {}
    for span in spans:
        rollout_id = span.bundle.episode.rollout_id
        tokens_by_rollout[rollout_id] = (
            tokens_by_rollout.get(rollout_id, 0) + span.segment.trainable_tokens
        )
        bucket = groups_by_rollout.setdefault(rollout_id, [])
        if span.parameter_group_id not in bucket:
            bucket.append(span.parameter_group_id)

    samples: list[CreditSample] = []
    resolved_channels: set[str] = set()
    optimized_channels: set[str] = set()
    for bundle in members:
        channel = _resolve_channel(bundle.reward, bundle.episode.team_id)
        resolved_channels.add(channel.channel_id)
        optimized_channels.add(bundle.reward.optimized_channel)
        rollout_id = bundle.episode.rollout_id
        samples.append(
            CreditSample(
                sample_key=rollout_id,
                reward=channel.measure,
                length=tokens_by_rollout.get(rollout_id, 0),
                reward_channel_id=channel.channel_id,
                team_id=bundle.episode.team_id,
                parameter_groups=tuple(sorted(groups_by_rollout.get(rollout_id, []))),
            )
        )
    if len(resolved_channels) != 1:
        raise AssemblyError(
            f"group {group_id} resolves to {len(resolved_channels)} reward channels; "
            "a group is one comparison on one channel"
        )
    credit = estimate(plan.credit, samples)
    fanout_targets = sorted({span.parameter_group_id for span in spans})
    if not fanout_targets:
        raise AssemblyError(f"group {group_id} has no trainable spans after filtering")
    fanout = fan_out_team_advantage(credit, fanout_targets)
    topology_ids = {
        bundle.topology.topology_id for bundle in members if bundle.topology is not None
    }
    provenance = GroupAdvantageProvenance(
        group_id=group_id,
        plan_hash=plan.plan_hash,
        pin_digest=pin.pin_digest,
        credit_kind=credit.kind,
        optimized_channel=sorted(optimized_channels)[0],
        resolved_channel=credit.reward_channel_id,
        topology_id=sorted(topology_ids)[0] if topology_ids else pin.topology_id,
        rollout_ids=credit.sample_keys,
        rewards=credit.rewards,
        lengths=credit.lengths,
        advantages=credit.advantages,
        zero_variance=credit.zero_variance,
        skipped=credit.skipped,
        fanout_parameter_groups=fanout.parameter_groups,
        same_policy_reduction=plan.credit.same_policy_reduction,
        reward_ids=tuple(bundle.reward.reward_id for bundle in members),
        trace_digests=tuple(bundle.episode.trace_digest for bundle in members),
    )
    return fanout, provenance


def _pack(
    plan: AlgorithmPlan,
    parameter_group_id: str,
    items_by_group: Mapping[str, list[SpanBatchItem]],
) -> tuple[ProviderStep, ...]:
    """Several groups per provider step, retaining each group's advantages."""

    per_step = plan.schedule.groups_per_step
    ordered = sorted(items_by_group)
    steps: list[ProviderStep] = []
    for index in range(0, len(ordered), per_step):
        chunk = ordered[index : index + per_step]
        steps.append(
            ProviderStep(
                step_index=len(steps),
                parameter_group_id=parameter_group_id,
                group_ids=tuple(chunk),
                items=tuple(item for group_id in chunk for item in items_by_group[group_id]),
            )
        )
    if len(steps) > plan.schedule.max_steps_per_round:
        raise AssemblyError(
            f"parameter group {parameter_group_id!r} needs {len(steps)} provider steps but the "
            f"plan allows {plan.schedule.max_steps_per_round} per round"
        )
    return tuple(steps)


def assemble(
    plan: AlgorithmPlan,
    bundles: Sequence[EvidenceBundle],
    *,
    round_index: int = 0,
    off_policy: bool = False,
    source_run_ids: Sequence[str] = (),
    accepted_staleness: int = 0,
) -> TrainingBatch:
    """Build one round's provider-ready batch, or raise saying why not."""

    if not bundles:
        raise AssemblyError("a batch needs at least one scored attempt")
    admitted, dropped_bundles = _validated_bundles(plan, bundles)
    if not admitted:
        raise AssemblyError("every attempt was refused by the staleness policy")
    spans, dropped_spans = _split_spans(admitted)
    if not spans:
        raise AssemblyError("no trainable spans survived authorship and mask filtering")

    by_group: dict[str, list[EvidenceBundle]] = {}
    for bundle in sorted(admitted, key=lambda item: (item.sample_index, item.episode.rollout_id)):
        by_group.setdefault(bundle.group_id, []).append(bundle)
    spans_by_group: dict[str, list[_Span]] = {}
    for span in spans:
        spans_by_group.setdefault(span.bundle.group_id, []).append(span)

    provenance: list[GroupAdvantageProvenance] = []
    fanouts: dict[str, TeamAdvantageFanout] = {}
    for group_id in sorted(by_group):
        fanout, record = _group_credit(
            plan, group_id, by_group[group_id], spans_by_group.get(group_id, ())
        )
        provenance.append(record)
        if not record.skipped:
            fanouts[group_id] = fanout

    trainable_spans = [span for span in spans if span.bundle.group_id in fanouts]

    # One vote per environment attempt inside a parameter group: several
    # branches of one attempt must not multiply that attempt's weight.
    branch_counts: dict[tuple[str, str], set[str]] = {}
    for span in trainable_spans:
        key = (span.parameter_group_id, span.bundle.root_id)
        branch_counts.setdefault(key, set()).add(span.segment.branch_id)

    streams_by_pg: dict[str, list[InstanceStream]] = {}
    stream_index: dict[tuple[str, str, str], int] = {}
    for span in sorted(
        trainable_spans,
        key=lambda item: (
            item.parameter_group_id,
            item.bundle.group_id,
            item.bundle.sample_index,
            item.bundle.episode.rollout_id,
            item.segment_index,
        ),
    ):
        instance = span.segment.agent_instance_id or span.bundle.episode.rollout_id
        key = (span.parameter_group_id, span.bundle.episode.rollout_id, instance)
        streams = streams_by_pg.setdefault(span.parameter_group_id, [])
        if key in stream_index:
            existing = streams[stream_index[key]]
            streams[stream_index[key]] = InstanceStream(
                sample_key=existing.sample_key,
                agent_instance_id=existing.agent_instance_id,
                trainable_tokens=existing.trainable_tokens + span.segment.trainable_tokens,
            )
        else:
            stream_index[key] = len(streams)
            streams.append(
                InstanceStream(
                    sample_key=span.bundle.episode.rollout_id,
                    agent_instance_id=instance,
                    trainable_tokens=span.segment.trainable_tokens,
                )
            )

    reductions = {
        parameter_group_id: reduce_same_policy(
            plan.credit.same_policy_reduction, parameter_group_id, streams
        )
        for parameter_group_id, streams in streams_by_pg.items()
    }

    items_by_pg: dict[str, dict[str, list[SpanBatchItem]]] = {}
    for span in sorted(
        trainable_spans,
        key=lambda item: (
            item.parameter_group_id,
            item.bundle.group_id,
            item.bundle.sample_index,
            item.bundle.episode.rollout_id,
            item.segment_index,
        ),
    ):
        bundle = span.bundle
        parameter_group_id = span.parameter_group_id
        instance = span.segment.agent_instance_id or bundle.episode.rollout_id
        branches = branch_counts[(parameter_group_id, bundle.root_id)]
        reduction = reductions[parameter_group_id]
        item = SpanBatchItem(
            parameter_group_id=parameter_group_id,
            group_id=bundle.group_id,
            rollout_id=bundle.episode.rollout_id,
            root_rollout_id=bundle.root_id,
            sample_index=bundle.sample_index,
            branch_id=span.segment.branch_id,
            agent_instance_id=span.segment.agent_instance_id,
            token_ids=tuple(span.segment.token_ids),
            loss_mask=tuple(span.segment.loss_mask),
            behavior_logprobs=tuple(span.segment.behavior_logprobs),
            advantage=fanouts[bundle.group_id].advantage_for(
                bundle.episode.rollout_id, parameter_group_id
            ),
            root_rollout_weight=1.0 / len(branches),
            same_policy_weight=reduction.weight_for(bundle.episode.rollout_id, instance),
            trainable_tokens=span.segment.trainable_tokens,
            policy_revision=bundle.episode.policy_revision,
            staleness_steps=bundle.staleness_steps,
            call_ids=tuple(span.segment.call_ids),
        )
        items_by_pg.setdefault(parameter_group_id, {}).setdefault(bundle.group_id, []).append(item)

    parameter_groups = tuple(
        ParameterGroupBatch(
            parameter_group_id=parameter_group_id,
            steps=_pack(plan, parameter_group_id, items_by_pg[parameter_group_id]),
            same_policy=reductions[parameter_group_id],
        )
        for parameter_group_id in sorted(items_by_pg)
    )
    if not parameter_groups:
        raise AssemblyError(
            "every group was skipped as zero-advantage; there is nothing to train on"
        )
    runs = tuple(sorted({bundle.source_run_id for bundle in admitted if bundle.source_run_id}))
    return TrainingBatch(
        plan_hash=plan.plan_hash,
        round_index=round_index,
        parameter_groups=parameter_groups,
        provenance=tuple(provenance),
        dropped_spans=tuple(dropped_spans),
        dropped_bundles=tuple(dropped_bundles),
        off_policy=off_policy,
        source_run_ids=tuple(source_run_ids) or runs,
        accepted_staleness=accepted_staleness,
    )
