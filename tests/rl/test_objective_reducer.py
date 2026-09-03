"""The objective and reducer dimensions: table entries, not engine branches."""

from __future__ import annotations

import math

import pytest

from synth_optimizers.cispo import CispoConfig
from synth_optimizers.cispo import objective as legacy_objective
from synth_optimizers.rl.objective import (
    OBJECTIVE_KERNELS,
    ObjectiveError,
    broadcast_advantage,
    cispo,
    evaluate,
    importance_ratios,
    kernel_for,
)
from synth_optimizers.rl.plan import PRESETS, expand
from synth_optimizers.rl.reducer import (
    REDUCER_KERNELS,
    ReducerError,
    branch_aware_root_coefficients,
    coefficients,
    reduce_loss,
)

BEHAVIOR = (-0.5, -1.5, -0.2, -2.0, -0.8)
CURRENT = (-0.4, -3.0, -0.2, -0.1, -1.4)
MASK = (1, 1, 0, 1, 1)


# --- Objective ---------------------------------------------------------------


def test_cispo_matches_the_pinned_reference_kernel_token_for_token() -> None:
    """Same clipping, same stop-gradded weight, same per-token loss."""

    advantages = (0.75, 0.75, 0.75, 0.75, 0.75)
    plan = PRESETS["cispo"]
    ours = evaluate(
        plan.objective,
        current_logprobs=CURRENT,
        behavior_logprobs=BEHAVIOR,
        advantages=advantages,
        mask=MASK,
    )
    ppo_kl = tuple(
        -(current - behavior) for current, behavior in zip(CURRENT, BEHAVIOR, strict=True)
    )
    legacy = legacy_objective(
        ppo_kl,
        CURRENT,
        advantages,
        tuple(bool(flag) for flag in MASK),
        CispoConfig(
            eps_clip=plan.objective.eps_low, eps_clip_high=plan.objective.eps_high
        ),
    )
    assert ours.per_token_loss == pytest.approx(legacy.token_losses, rel=1e-12, abs=1e-15)
    assert ours.selected_tokens == legacy.selected_token_count
    assert ours.clipped_tokens == legacy.clipped_token_count
    assert ours.clipped_fraction == pytest.approx(legacy.clip_fraction)
    assert ours.mean_ratio == pytest.approx(legacy.mean_ratio)


def test_cispo_clips_only_above_when_eps_low_is_one() -> None:
    result = cispo(
        current_logprobs=(0.0, 0.0),
        behavior_logprobs=(-4.0, 4.0),
        advantages=(1.0, 1.0),
        mask=(1, 1),
        eps_low=1.0,
        eps_high=4.0,
    )
    ratios = importance_ratios((0.0, 0.0), (-4.0, 4.0))
    assert ratios[0] > 5.0 and ratios[1] < 0.02
    # High side clamps at 1 + eps_high; low side has no floor above zero.
    assert result.per_token_weight[0] == pytest.approx(5.0)
    assert result.per_token_weight[1] == pytest.approx(ratios[1])
    assert result.clipped_tokens == 1


def test_cispo_minimax_refuses_a_two_sided_epsilon() -> None:
    with pytest.raises(ObjectiveError, match="eps_low >= 1"):
        cispo(
            current_logprobs=(0.0,),
            behavior_logprobs=(0.0,),
            advantages=(1.0,),
            mask=(1,),
            eps_low=0.2,
            eps_high=0.2,
            variant="cispo_minimax",
        )


def test_a_second_objective_is_a_table_entry_and_uses_one_sequence_ratio() -> None:
    plan = PRESETS["gspo"]
    result = evaluate(
        plan.objective,
        current_logprobs=CURRENT,
        behavior_logprobs=BEHAVIOR,
        advantages=(0.5,) * len(CURRENT),
        mask=MASK,
    )
    assert result.ratio_granularity == "sequence"
    deltas = [
        current - behavior
        for current, behavior, flag in zip(CURRENT, BEHAVIOR, MASK, strict=True)
        if flag
    ]
    expected = math.exp(sum(deltas) / len(deltas))
    clamped = min(max(expected, 0.8), 1.2)
    for weight, flag in zip(result.per_token_weight, MASK, strict=True):
        assert weight == pytest.approx(clamped if flag else 0.0)
    assert set(OBJECTIVE_KERNELS) >= {"cispo", "gspo", "reinforce"}


def test_reinforce_carries_no_importance_weight() -> None:
    plan = expand(
        {
            "preset": "cispo",
            "objective": {"kind": "reinforce", "variant": "reinforce"},
        }
    )
    result = evaluate(
        plan.objective,
        current_logprobs=CURRENT,
        behavior_logprobs=BEHAVIOR,
        advantages=(1.0,) * len(CURRENT),
        mask=MASK,
    )
    assert result.mean_ratio == 1.0
    assert result.clipped_tokens == 0
    assert result.per_token_loss[2] == 0.0


def test_objectives_without_a_kernel_raise_instead_of_silently_running() -> None:
    with pytest.raises(ObjectiveError, match="no kernel"):
        kernel_for("jsd_distillation")(
            current_logprobs=(0.0,),
            behavior_logprobs=(0.0,),
            advantages=(1.0,),
            mask=(1,),
        )
    with pytest.raises(ObjectiveError, match="known"):
        kernel_for("not_an_objective")


def test_objective_inputs_must_align_and_select_something() -> None:
    with pytest.raises(ObjectiveError, match="equal length"):
        cispo(
            current_logprobs=(0.0, 0.0),
            behavior_logprobs=(0.0,),
            advantages=(1.0, 1.0),
            mask=(1, 1),
            eps_low=1.0,
            eps_high=4.0,
        )
    with pytest.raises(ObjectiveError, match="denominator is zero"):
        cispo(
            current_logprobs=(0.0, 0.0),
            behavior_logprobs=(0.0, 0.0),
            advantages=(1.0, 1.0),
            mask=(0, 0),
            eps_low=1.0,
            eps_high=4.0,
        )


def test_broadcast_advantage_spreads_one_sample_advantage() -> None:
    assert broadcast_advantage(-0.25, 3) == (-0.25, -0.25, -0.25)


# --- Reducer -----------------------------------------------------------------


def test_reducer_table_covers_the_plan_vocabulary() -> None:
    assert set(REDUCER_KERNELS) == {
        "branch_aware_root_mean",
        "token_mean",
        "sequence_mean",
        "root_rollout_mean",
        "fixed_token_denominator",
    }


def test_branch_aware_root_mean_does_not_triple_a_three_branch_attempt() -> None:
    """One attempt with three branches weighs the same as one with one."""

    single = reduce_loss(
        PRESETS["cispo"].reducer,
        per_item_loss=[6.0],
        per_item_tokens=[3],
        root_ids=["root-a"],
    )
    branched = reduce_loss(
        PRESETS["cispo"].reducer,
        per_item_loss=[2.0, 2.0, 2.0],
        per_item_tokens=[1, 1, 1],
        root_ids=["root-a", "root-a", "root-a"],
    )
    assert single.value == pytest.approx(branched.value)
    assert single.denominator == branched.denominator == 1.0
    two_roots = reduce_loss(
        PRESETS["cispo"].reducer,
        per_item_loss=[2.0, 2.0, 2.0, 6.0],
        per_item_tokens=[1, 1, 1, 3],
        root_ids=["root-a", "root-a", "root-a", "root-b"],
    )
    assert two_roots.denominator == 2.0
    assert two_roots.value == pytest.approx(2.0)


def test_token_mean_uses_one_batch_wide_denominator() -> None:
    result = reduce_loss(
        expand({"preset": "cispo", "reducer": {"kind": "token_mean"}}).reducer,
        per_item_loss=[4.0, 6.0],
        per_item_tokens=[2, 8],
        root_ids=["a", "b"],
    )
    assert result.value == pytest.approx(1.0)
    assert result.denominator == 10.0


def test_sequence_mean_gives_a_short_sequence_the_same_vote() -> None:
    result = reduce_loss(
        PRESETS["gspo"].reducer,
        per_item_loss=[4.0, 4.0],
        per_item_tokens=[1, 100],
        root_ids=["a", "b"],
    )
    assert result.value == pytest.approx((4.0 + 0.04) / 2)
    assert result.denominator == 2.0


def test_root_rollout_mean_averages_branches_as_sequences() -> None:
    result = reduce_loss(
        expand({"preset": "cispo", "reducer": {"kind": "root_rollout_mean"}}).reducer,
        per_item_loss=[2.0, 8.0, 3.0],
        per_item_tokens=[1, 4, 1],
        root_ids=["root-a", "root-a", "root-b"],
    )
    assert result.value == pytest.approx(((2.0 / 1 + 8.0 / 4) / 2 + 3.0) / 2)
    assert result.denominator == 2.0


def test_fixed_token_denominator_needs_its_denominator() -> None:
    reducer = expand(
        {"preset": "cispo", "reducer": {"kind": "fixed_token_denominator"}}
    ).reducer
    with pytest.raises(ReducerError, match="positive token_denominator"):
        reduce_loss(reducer, per_item_loss=[1.0], per_item_tokens=[1], root_ids=["a"])
    result = reduce_loss(
        reducer,
        per_item_loss=[1.0, 1.0],
        per_item_tokens=[1, 1],
        root_ids=["a", "b"],
        token_denominator=8.0,
    )
    assert result.value == pytest.approx(0.25)


def test_streaming_coefficients_land_on_the_same_loss() -> None:
    per_item_loss = [2.0, 4.0, 9.0]
    per_item_tokens = [1, 3, 3]
    root_ids = ["root-a", "root-a", "root-b"]
    batched = reduce_loss(
        PRESETS["cispo"].reducer,
        per_item_loss=per_item_loss,
        per_item_tokens=per_item_tokens,
        root_ids=root_ids,
    )
    scalars = coefficients(
        "branch_aware_root_mean",
        per_item_tokens=per_item_tokens,
        root_ids=root_ids,
        root_weights=[1.0, 1.0, 1.0],
    )
    streamed = sum(
        loss * scalar for loss, scalar in zip(per_item_loss, scalars, strict=True)
    )
    assert streamed == pytest.approx(batched.value)
    assert scalars == branch_aware_root_coefficients(
        per_item_tokens=per_item_tokens, root_ids=root_ids, root_weights=[1.0, 1.0, 1.0]
    )


def test_reducers_without_a_streaming_form_say_so() -> None:
    with pytest.raises(ReducerError, match="no streaming coefficient form"):
        coefficients("token_mean", per_item_tokens=[1], root_ids=["a"], root_weights=[1.0])


def test_misaligned_reducer_inputs_raise() -> None:
    with pytest.raises(ReducerError, match="must align"):
        reduce_loss(
            PRESETS["cispo"].reducer,
            per_item_loss=[1.0, 2.0],
            per_item_tokens=[1],
            root_ids=["a", "b"],
        )
