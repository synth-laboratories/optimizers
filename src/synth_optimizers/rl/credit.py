"""The credit dimension: reward receipts to per-sample advantage.

Credit is not reward. Every estimator here is keyed by the plan's ``CREDITS``
vocabulary and selected from a table, never from a conditional on an algorithm
name. The estimator bodies match ``tito_train.credit`` exactly so a replay of
one plane's evidence reproduces the other plane's advantages.

Two things beyond the single-agent case live here:

* Zero-variance group detection. A group whose members all tie has no ordering,
  so it carries no evidence; the plan decides whether such a group is skipped.
* The joint-episode path. One group-relative team advantage per episode is
  fanned out to every trainee parameter group, and the same-policy reduction is
  receipted so a chatty low-throughput role cannot take the parameter group's
  update by token count alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .plan import GROUP_RELATIVE_CREDITS, CreditEstimator


class CreditError(ValueError):
    """A credit estimator was given incomparable or malformed samples."""


# --- Estimators, keyed by the plan vocabulary --------------------------------


def length_weighted_leave_one_out(
    rewards: Sequence[float], lengths: Sequence[int]
) -> list[float]:
    """Leave-one-out baseline weighted by each other member's rewarded tokens."""

    if len(rewards) != len(lengths):
        raise CreditError("rewards and lengths must align")
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    advantages: list[float] = []
    for index, reward in enumerate(rewards):
        weight_sum = sum(lengths[j] for j in range(len(rewards)) if j != index)
        if weight_sum <= 0:
            baseline = sum(rewards[j] for j in range(len(rewards)) if j != index) / (
                len(rewards) - 1
            )
        else:
            baseline = (
                sum(lengths[j] * rewards[j] for j in range(len(rewards)) if j != index)
                / weight_sum
            )
        advantages.append(float(reward) - float(baseline))
    return advantages


def length_weighted_leave_one_out_standardized(
    rewards: Sequence[float], lengths: Sequence[int]
) -> list[float]:
    """Leave-one-out, rescaled by the group's own reward spread.

    A raw reward difference is the right credit when rewards are O(1) and the
    wrong one when they are not: a sparse aggregate can put a whole group's
    advantages at 1e-3 and produce a gradient too small to move an adapter.
    Dividing by the group's reward standard deviation makes credit scale-free,
    so what is learned is the ordering inside the group.

    A group whose rewards all tie has no ordering, so this returns exactly zero
    rather than dividing by an epsilon and amplifying float noise into a
    gradient. Such a group is not evidence.
    """

    base = length_weighted_leave_one_out(rewards, lengths)
    if len(rewards) < 2:
        return base
    mean = sum(rewards) / len(rewards)
    variance = sum((float(reward) - mean) ** 2 for reward in rewards) / len(rewards)
    if variance <= 0.0:
        return [0.0 for _ in base]
    deviation = variance**0.5
    return [advantage / deviation for advantage in base]


def leave_one_out(rewards: Sequence[float]) -> list[float]:
    return length_weighted_leave_one_out(rewards, [1] * len(rewards))


def group_mean(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(reward) - mean for reward in rewards]


def raw_reward(rewards: Sequence[float]) -> list[float]:
    return [float(reward) for reward in rewards]


ESTIMATORS: Mapping[str, Callable[[Sequence[float], Sequence[int]], list[float]]] = {
    "length_weighted_leave_one_out": length_weighted_leave_one_out,
    "length_weighted_leave_one_out_standardized": length_weighted_leave_one_out_standardized,
    "leave_one_out": lambda rewards, _lengths: leave_one_out(rewards),
    "group_mean": lambda rewards, _lengths: group_mean(rewards),
    "raw_reward": lambda rewards, _lengths: raw_reward(rewards),
}


def is_zero_variance_group(advantages: Sequence[float], *, atol: float = 1e-8) -> bool:
    """Every member indistinguishable from every other. Not evidence."""

    return all(abs(float(value)) <= atol for value in advantages)


# --- Group credit ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CreditSample:
    """One comparable sample of a group: one episode's optimized measure."""

    sample_key: str
    reward: float
    length: int
    reward_channel_id: str = ""
    team_id: str | None = None
    parameter_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_key.strip():
            raise CreditError("credit sample needs a stable key")
        if self.length < 0:
            raise CreditError(f"sample {self.sample_key} has negative length")


@dataclass(frozen=True, slots=True)
class GroupCredit:
    """Per-sample advantage plus everything needed to re-derive it."""

    kind: str
    sample_keys: tuple[str, ...]
    rewards: tuple[float, ...]
    lengths: tuple[int, ...]
    advantages: tuple[float, ...]
    zero_variance: bool
    skipped: bool
    atol: float
    reward_channel_id: str = ""

    def advantage_for(self, sample_key: str) -> float:
        for key, advantage in zip(self.sample_keys, self.advantages, strict=True):
            if key == sample_key:
                return advantage
        raise CreditError(f"no credit for sample {sample_key!r}")

    def receipt(self) -> dict[str, Any]:
        return {
            "credit_kind": self.kind,
            "reward_channel_id": self.reward_channel_id,
            "sample_keys": list(self.sample_keys),
            "rewards": list(self.rewards),
            "lengths": list(self.lengths),
            "advantages": list(self.advantages),
            "zero_variance": self.zero_variance,
            "skipped": self.skipped,
            "zero_advantage_atol": self.atol,
        }


def estimator_for(kind: str) -> Callable[[Sequence[float], Sequence[int]], list[float]]:
    """Table lookup. A new credit kind is a row, never a branch."""

    if kind not in ESTIMATORS:
        raise CreditError(
            f"credit estimator {kind!r} expands as a plan dimension but has no table "
            f"entry in this plane; implemented: {sorted(ESTIMATORS)}"
        )
    return ESTIMATORS[kind]


def estimate(credit: CreditEstimator, samples: Sequence[CreditSample]) -> GroupCredit:
    """One group in, one receipted advantage vector out."""

    if not samples:
        raise CreditError("a group with no samples has no credit")
    channels = {sample.reward_channel_id for sample in samples}
    if len(channels) > 1:
        raise CreditError(
            f"group mixes reward channels {sorted(channels)}; the optimized channel is one channel"
        )
    rewards = tuple(float(sample.reward) for sample in samples)
    lengths = tuple(int(sample.length) for sample in samples)
    advantages = tuple(estimator_for(credit.kind)(rewards, lengths))
    if len(advantages) != len(samples):
        raise CreditError(f"credit estimator {credit.kind!r} returned the wrong arity")
    zero_variance = credit.kind in GROUP_RELATIVE_CREDITS and is_zero_variance_group(
        advantages, atol=credit.zero_advantage_atol
    )
    return GroupCredit(
        kind=credit.kind,
        sample_keys=tuple(sample.sample_key for sample in samples),
        rewards=rewards,
        lengths=lengths,
        advantages=advantages,
        zero_variance=zero_variance,
        skipped=bool(credit.skip_zero_advantage and zero_variance),
        atol=credit.zero_advantage_atol,
        reward_channel_id=next(iter(channels)),
    )


# --- Joint episodes: team advantage, fan-out, same-policy reduction ----------


@dataclass(frozen=True, slots=True)
class TeamAdvantageFanout:
    """One episode's team advantage, replicated to every trainee group.

    The advantage is computed across episodes and never across teams inside one
    episode: a trainee team measured against a frozen opponent team in the same
    episode is not a group.
    """

    credit: GroupCredit
    parameter_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parameter_groups:
            raise CreditError("fan-out needs at least one trainee parameter group")
        if len(set(self.parameter_groups)) != len(self.parameter_groups):
            raise CreditError("duplicate parameter group in fan-out")

    def advantage_for(self, sample_key: str, parameter_group_id: str) -> float:
        if parameter_group_id not in self.parameter_groups:
            raise CreditError(
                f"parameter group {parameter_group_id!r} is not a fan-out target; "
                f"targets: {list(self.parameter_groups)}"
            )
        return self.credit.advantage_for(sample_key)

    def receipt(self) -> dict[str, Any]:
        payload = self.credit.receipt()
        payload["fanout_parameter_groups"] = list(self.parameter_groups)
        return payload


def fan_out_team_advantage(
    credit: GroupCredit, parameter_groups: Sequence[str]
) -> TeamAdvantageFanout:
    """One group-relative team advantage per episode, for every trainee group."""

    ordered: list[str] = []
    for group in parameter_groups:
        if group not in ordered:
            ordered.append(group)
    return TeamAdvantageFanout(credit=credit, parameter_groups=tuple(ordered))


@dataclass(frozen=True, slots=True)
class InstanceStream:
    """One agent instance's trainable tokens inside one episode."""

    sample_key: str
    agent_instance_id: str
    trainable_tokens: int

    def __post_init__(self) -> None:
        if self.trainable_tokens < 0:
            raise CreditError(
                f"instance {self.agent_instance_id} reports negative trainable tokens"
            )


@dataclass(frozen=True, slots=True)
class StreamWeight:
    sample_key: str
    agent_instance_id: str
    trainable_tokens: int
    #: Share of the parameter group's update this stream carries.
    weight: float
    #: Share per trainable token, i.e. ``weight / trainable_tokens``.
    per_token_weight: float


@dataclass(frozen=True, slots=True)
class SamePolicyReduction:
    """The receipt: which normalization was applied and what share each got.

    ``naive_shares`` is what plain flattening would have handed each instance —
    its token count over the batch's token count. ``applied_shares`` is what the
    declared reduction actually hands it. When a low-throughput role emits most
    of a parameter group's tokens, these differ, and the difference is the whole
    point of declaring the reduction.
    """

    kind: str
    parameter_group_id: str
    streams: tuple[StreamWeight, ...]

    @property
    def total_tokens(self) -> int:
        return sum(stream.trainable_tokens for stream in self.streams)

    def _by_instance(self, values: Mapping[str, float]) -> dict[str, float]:
        return dict(sorted(values.items()))

    @property
    def naive_shares(self) -> dict[str, float]:
        total = self.total_tokens
        shares: dict[str, float] = {}
        for stream in self.streams:
            share = (stream.trainable_tokens / total) if total else 0.0
            shares[stream.agent_instance_id] = shares.get(stream.agent_instance_id, 0.0) + share
        return self._by_instance(shares)

    @property
    def applied_shares(self) -> dict[str, float]:
        shares: dict[str, float] = {}
        for stream in self.streams:
            shares[stream.agent_instance_id] = (
                shares.get(stream.agent_instance_id, 0.0) + stream.weight
            )
        return self._by_instance(shares)

    def weight_for(self, sample_key: str, agent_instance_id: str) -> float:
        for stream in self.streams:
            if stream.sample_key == sample_key and stream.agent_instance_id == agent_instance_id:
                return stream.weight
        raise CreditError(
            f"no same-policy weight for instance {agent_instance_id!r} in {sample_key!r}"
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "same_policy_reduction": self.kind,
            "parameter_group_id": self.parameter_group_id,
            "streams": [
                {
                    "sample_key": stream.sample_key,
                    "agent_instance_id": stream.agent_instance_id,
                    "trainable_tokens": stream.trainable_tokens,
                    "weight": stream.weight,
                    "per_token_weight": stream.per_token_weight,
                }
                for stream in self.streams
            ],
            "naive_shares": self.naive_shares,
            "applied_shares": self.applied_shares,
        }


def _shares_none(streams: Sequence[InstanceStream]) -> list[float]:
    """Plain flattening: every token weighs the same, so token count decides."""

    total = sum(stream.trainable_tokens for stream in streams)
    if total <= 0:
        return [0.0 for _ in streams]
    return [stream.trainable_tokens / total for stream in streams]


def _shares_token_weighted_mean(streams: Sequence[InstanceStream]) -> list[float]:
    """One vote per agent instance; token-weighted inside the instance."""

    per_instance: dict[str, int] = {}
    for stream in streams:
        per_instance[stream.agent_instance_id] = (
            per_instance.get(stream.agent_instance_id, 0) + stream.trainable_tokens
        )
    live = {name: total for name, total in per_instance.items() if total > 0}
    if not live:
        return [0.0 for _ in streams]
    shares: list[float] = []
    for stream in streams:
        instance_total = live.get(stream.agent_instance_id, 0)
        if instance_total <= 0:
            shares.append(0.0)
            continue
        shares.append(stream.trainable_tokens / instance_total / len(live))
    return shares


def _shares_episode_uniform(streams: Sequence[InstanceStream]) -> list[float]:
    """One vote per episode; token-weighted inside the episode."""

    per_episode: dict[str, int] = {}
    for stream in streams:
        per_episode[stream.sample_key] = (
            per_episode.get(stream.sample_key, 0) + stream.trainable_tokens
        )
    live = {name: total for name, total in per_episode.items() if total > 0}
    if not live:
        return [0.0 for _ in streams]
    shares: list[float] = []
    for stream in streams:
        episode_total = live.get(stream.sample_key, 0)
        if episode_total <= 0:
            shares.append(0.0)
            continue
        shares.append(stream.trainable_tokens / episode_total / len(live))
    return shares


SAME_POLICY_REDUCERS: Mapping[str, Callable[[Sequence[InstanceStream]], list[float]]] = {
    "none": _shares_none,
    "token_weighted_mean": _shares_token_weighted_mean,
    "episode_uniform": _shares_episode_uniform,
}


def reduce_same_policy(
    kind: str, parameter_group_id: str, streams: Sequence[InstanceStream]
) -> SamePolicyReduction:
    """Apply the declared same-policy reduction and receipt what it did."""

    if kind not in SAME_POLICY_REDUCERS:
        raise CreditError(
            f"same-policy reduction {kind!r} has no table entry; "
            f"implemented: {sorted(SAME_POLICY_REDUCERS)}"
        )
    shares = SAME_POLICY_REDUCERS[kind](streams)
    weighted = tuple(
        StreamWeight(
            sample_key=stream.sample_key,
            agent_instance_id=stream.agent_instance_id,
            trainable_tokens=stream.trainable_tokens,
            weight=share,
            per_token_weight=(share / stream.trainable_tokens)
            if stream.trainable_tokens
            else 0.0,
        )
        for stream, share in zip(streams, shares, strict=True)
    )
    return SamePolicyReduction(
        kind=kind, parameter_group_id=parameter_group_id, streams=weighted
    )
