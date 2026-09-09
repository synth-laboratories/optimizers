"""The credit dimension: legacy parity, zero variance, fan-out, and reduction."""

from __future__ import annotations

import pytest

from synth_optimizers.cispo import group_advantages, is_zero_advantage_group
from synth_optimizers.rl.credit import (
    CreditError,
    CreditSample,
    InstanceStream,
    estimate,
    estimator_for,
    fan_out_team_advantage,
    group_mean,
    is_zero_variance_group,
    leave_one_out,
    length_weighted_leave_one_out,
    length_weighted_leave_one_out_standardized,
    reduce_same_policy,
)
from synth_optimizers.rl.plan import PRESETS, expand

CISPO = PRESETS["cispo"]


def _samples(rewards: list[float], lengths: list[int]) -> list[CreditSample]:
    return [
        CreditSample(
            sample_key=f"roll-{index}",
            reward=reward,
            length=length,
            reward_channel_id="outcome",
        )
        for index, (reward, length) in enumerate(zip(rewards, lengths, strict=True))
    ]


# --- Parity with the existing CISPO advantage/skip logic ---------------------


@pytest.mark.parametrize(
    "rewards",
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.25, 0.5, 0.75, 1.0],
    ],
)
def test_group_mean_credit_is_the_legacy_unnormalized_advantage(rewards: list[float]) -> None:
    """Bit-for-bit: both centre on the same mean, computed the same way."""

    lengths = [10] * len(rewards)
    assert tuple(group_mean(rewards)) == group_advantages(rewards, normalize=False)
    credit = estimate(
        expand({"preset": "cispo", "credit": {"kind": "group_mean"}}).credit,
        _samples(rewards, lengths),
    )
    assert credit.advantages == group_advantages(rewards, normalize=False)


@pytest.mark.parametrize(
    "rewards",
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.25, 0.5, 0.75, 1.0],
    ],
)
def test_cispo_preset_skip_decision_matches_the_legacy_skip_decision(
    rewards: list[float],
) -> None:
    """The tie/no-tie verdict is identical, which is what the skip turns on."""

    credit = estimate(CISPO.credit, _samples(rewards, [10] * len(rewards)))
    legacy_zero = is_zero_advantage_group(group_advantages(rewards, normalize=False))
    assert credit.zero_variance is legacy_zero
    assert credit.skipped is legacy_zero
    assert is_zero_variance_group(credit.advantages) is legacy_zero


def test_cispo_preset_credit_is_the_legacy_centering_rescaled_by_the_loo_factor() -> None:
    """Equal lengths reduce the plan's estimator to n/(n-1) times centering.

    The `cispo` preset's credit kind is `length_weighted_leave_one_out`, whose
    baseline excludes the sample itself; the legacy helper centres on a mean
    that includes it. On equal-length members the two differ by exactly that
    factor and never by sign or ordering.
    """

    rewards = [1.0, 0.0, 0.0, 0.0]
    credit = estimate(CISPO.credit, _samples(rewards, [10] * len(rewards)))
    legacy = group_advantages(rewards, normalize=False)
    factor = (len(rewards) - 1) / len(rewards)
    assert [value * factor for value in credit.advantages] == pytest.approx(
        list(legacy), rel=1e-12, abs=1e-15
    )
    signs = [(a > 0) == (b > 0) for a, b in zip(credit.advantages, legacy, strict=True)]
    assert all(signs)


# --- Estimator table ---------------------------------------------------------


def test_length_weighted_baseline_follows_tokens_not_headcount() -> None:
    rewards = [1.0, 0.0, 0.0]
    heavy = length_weighted_leave_one_out(rewards, [1, 100, 1])
    uniform = leave_one_out(rewards)
    # Member 1 carries 100 of the group's 102 tokens, so it dominates the
    # baseline of every other member. Member 2's baseline is dragged from the
    # headcount's 0.5 down to 1/101, and its penalty nearly vanishes.
    assert uniform == pytest.approx([1.0, -0.5, -0.5])
    assert heavy == pytest.approx([1.0, -0.5, -1.0 / 101.0])
    assert abs(heavy[2]) < abs(uniform[2]) / 10
    # Member 1's own baseline excludes itself, so its credit is unchanged.
    assert heavy[1] == uniform[1]


def test_standardized_variant_returns_exact_zero_on_a_tie() -> None:
    assert length_weighted_leave_one_out_standardized([0.5, 0.5, 0.5], [3, 3, 3]) == [
        0.0,
        0.0,
        0.0,
    ]


def test_standardized_variant_makes_a_tiny_reward_scale_visible() -> None:
    rewards = [0.0104, 0.0, 0.0, 0.0]
    lengths = [7, 7, 7, 7]
    raw = length_weighted_leave_one_out(rewards, lengths)
    scaled = length_weighted_leave_one_out_standardized(rewards, lengths)
    assert max(abs(value) for value in raw) < 0.02
    assert max(abs(value) for value in scaled) > 1.0


def test_unimplemented_credit_kinds_raise_rather_than_branch() -> None:
    with pytest.raises(CreditError, match="no table entry"):
        estimator_for("gae")
    with pytest.raises(CreditError, match="no table entry"):
        estimate(
            expand({"preset": "ppo", "credit": {"kind": "gae"}}).credit,
            _samples([1.0, 0.0], [4, 4]),
        )


def test_a_group_may_not_mix_reward_channels() -> None:
    samples = [
        CreditSample(sample_key="a", reward=1.0, length=4, reward_channel_id="rank"),
        CreditSample(sample_key="b", reward=0.0, length=4, reward_channel_id="margin"),
    ]
    with pytest.raises(CreditError, match="mixes reward channels"):
        estimate(CISPO.credit, samples)


def test_credit_receipt_carries_everything_needed_to_re_derive_it() -> None:
    credit = estimate(CISPO.credit, _samples([1.0, 0.0], [4, 6]))
    receipt = credit.receipt()
    assert receipt["credit_kind"] == "length_weighted_leave_one_out"
    assert receipt["rewards"] == [1.0, 0.0]
    assert receipt["lengths"] == [4, 6]
    assert receipt["zero_advantage_atol"] == 1e-8
    assert receipt["skipped"] is False


# --- Joint episodes ----------------------------------------------------------


def test_one_team_advantage_fans_out_to_every_trainee_parameter_group() -> None:
    credit = estimate(CISPO.credit, _samples([1.0, 0.0, 0.0, 0.0], [12, 12, 12, 12]))
    fanout = fan_out_team_advantage(credit, ["pg_beta", "pg_alpha", "pg_beta"])
    assert fanout.parameter_groups == ("pg_beta", "pg_alpha")
    for key in credit.sample_keys:
        assert fanout.advantage_for(key, "pg_alpha") == fanout.advantage_for(key, "pg_beta")
        assert fanout.advantage_for(key, "pg_alpha") == credit.advantage_for(key)
    with pytest.raises(CreditError, match="not a fan-out target"):
        fanout.advantage_for("roll-0", "pg_gamma")
    assert fanout.receipt()["fanout_parameter_groups"] == ["pg_beta", "pg_alpha"]


def test_same_policy_reduction_stops_token_count_domination() -> None:
    """A chatty low-throughput instance must not own the parameter group.

    One instance emits 100 of the group's 104 trainable tokens. Plain
    flattening hands it 96% of the update; the declared reduction hands each
    instance one vote and shrinks the chatty instance's per-token weight
    instead.
    """

    streams = [
        InstanceStream(sample_key="ep-1", agent_instance_id="inst_quiet", trainable_tokens=4),
        InstanceStream(sample_key="ep-1", agent_instance_id="inst_chatty", trainable_tokens=100),
    ]

    naive = reduce_same_policy("none", "pg_alpha", streams)
    assert naive.applied_shares == pytest.approx(
        {"inst_chatty": 100 / 104, "inst_quiet": 4 / 104}
    )
    assert naive.applied_shares["inst_chatty"] / naive.applied_shares["inst_quiet"] == (
        pytest.approx(25.0)
    )

    reduced = reduce_same_policy("token_weighted_mean", "pg_alpha", streams)
    assert reduced.naive_shares == naive.applied_shares
    assert reduced.applied_shares == pytest.approx({"inst_chatty": 0.5, "inst_quiet": 0.5})
    per_token = {
        stream.agent_instance_id: stream.per_token_weight for stream in reduced.streams
    }
    assert per_token["inst_quiet"] == pytest.approx(0.125)
    assert per_token["inst_chatty"] == pytest.approx(0.005)

    receipt = reduced.receipt()
    assert receipt["same_policy_reduction"] == "token_weighted_mean"
    assert receipt["naive_shares"] != receipt["applied_shares"]


def test_same_policy_reduction_spreads_an_instance_over_its_episodes() -> None:
    streams = [
        InstanceStream(sample_key="ep-1", agent_instance_id="inst_a", trainable_tokens=3),
        InstanceStream(sample_key="ep-2", agent_instance_id="inst_a", trainable_tokens=9),
        InstanceStream(sample_key="ep-1", agent_instance_id="inst_b", trainable_tokens=12),
    ]
    reduced = reduce_same_policy("token_weighted_mean", "pg_alpha", streams)
    assert reduced.applied_shares == pytest.approx({"inst_a": 0.5, "inst_b": 0.5})
    assert reduced.weight_for("ep-1", "inst_a") == pytest.approx(0.125)
    assert reduced.weight_for("ep-2", "inst_a") == pytest.approx(0.375)


def test_episode_uniform_reduction_gives_each_episode_one_vote() -> None:
    streams = [
        InstanceStream(sample_key="ep-1", agent_instance_id="inst_a", trainable_tokens=1),
        InstanceStream(sample_key="ep-2", agent_instance_id="inst_b", trainable_tokens=99),
    ]
    reduced = reduce_same_policy("episode_uniform", "pg_alpha", streams)
    assert reduced.applied_shares == pytest.approx({"inst_a": 0.5, "inst_b": 0.5})


def test_unknown_same_policy_reduction_raises() -> None:
    with pytest.raises(CreditError, match="no table entry"):
        reduce_same_policy(
            "median_of_roles",
            "pg_alpha",
            [InstanceStream(sample_key="ep-1", agent_instance_id="a", trainable_tokens=1)],
        )
