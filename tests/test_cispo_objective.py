from __future__ import annotations

from synth_optimizers.cispo import CispoConfig, CispoError, group_advantages, is_zero_advantage_group, objective
import pytest


def test_matches_slime_wide_minimax_fixture_and_stop_gradient() -> None:
    ratios = [1.0, 3.0, 9.0, 0.4]
    ppo_kl = [-__import__("math").log(ratio) for ratio in ratios]
    log_probs = [-0.7, -1.2, -0.4, -2.1]
    advantages = [1.0, -0.5, 2.0, -1.0]
    result = objective(ppo_kl, log_probs, advantages, [True] * 4, CispoConfig())
    clamped = [1.0, 3.0, 5.0, 0.4]
    expected_losses = [-ratio * adv * logp for ratio, adv, logp in zip(clamped, advantages, log_probs)]
    assert result.token_losses == pytest.approx(expected_losses)
    expected_grads = [-ratio * adv / 4.0 for ratio, adv in zip(clamped, advantages)]
    assert result.log_prob_gradients == pytest.approx(expected_grads)
    assert result.clip_fraction == pytest.approx(0.25)


def test_group_advantages_use_sample_standard_deviation() -> None:
    actual = group_advantages([1.0, 3.0])
    denom = 2**0.5 + 1e-6
    assert actual[0] == pytest.approx(-1.0 / denom)
    assert actual[1] == pytest.approx(1.0 / denom)


def test_zero_variance_group_is_zero_advantage() -> None:
    advantages = group_advantages([1.0, 1.0, 1.0])
    assert is_zero_advantage_group(advantages)


def test_eps_clip_below_one_is_not_cispo() -> None:
    with pytest.raises(CispoError, match="eps_clip"):
        CispoConfig(eps_clip=0.2).validate()


def test_padding_mask_excludes_tokens_from_the_denominator() -> None:
    result = objective(
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0],
        [1.0, 1.0, 1.0],
        [True, False, True],
        CispoConfig(),
    )
    assert result.selected_token_count == 2
    assert result.token_losses[1] == 0.0
