# Copyright 2026 Synth Laboratories
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from optimizers-beta crates/synth_training/src/algorithms/cispo_slime/mod.rs
# Source commit: d0b8577040cad9a52b45125eee4a3094b40c3185
# Upstream slime commit: 41014d1f29e201137fdffce737bb8bac65bc5219
# The importance ratio is clipped and stop-graded; the gradient flows only
# through log pi. A generic Tinker importance-sampling run is not CISPO.

"""Exact ``cispo.slime.v1`` objective and group-relative advantages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


IMPLEMENTATION = "slime-reference"
IMPLEMENTATION_VERSION = "cispo.slime.v1"
UPSTREAM_COMMIT = "41014d1f29e201137fdffce737bb8bac65bc5219"
ALGORITHM_ID = "cispo"


class CispoError(ValueError):
    """Invalid CISPO configuration, tensors, or group construction."""


@dataclass(frozen=True, slots=True)
class CispoConfig:
    """Canonical CISPO disables the lower bound with ``eps_clip >= 1``."""

    eps_clip: float = 1.0
    eps_clip_high: float = 4.0
    gamma: float = 1.0
    lambda_: float = 1.0
    normalize_group_rewards: bool = True

    def validate(self) -> None:
        if self.eps_clip < 1.0:
            raise CispoError("canonical CISPO requires eps_clip >= 1.0")
        if self.eps_clip_high < 0.0 or not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.lambda_ <= 1.0:
            raise CispoError("invalid clipping or GAE coefficient")


@dataclass(frozen=True, slots=True)
class CispoObjective:
    token_losses: tuple[float, ...]
    log_prob_gradients: tuple[float, ...]
    selected_token_count: int
    loss: float
    clip_fraction: float
    mean_ratio: float
    clipped_token_count: int

    @property
    def implementation_version(self) -> str:
        return IMPLEMENTATION_VERSION


def objective(
    ppo_kl: Sequence[float],
    log_probs: Sequence[float],
    advantages: Sequence[float],
    loss_mask: Sequence[bool],
    config: CispoConfig | None = None,
) -> CispoObjective:
    """Token-level CISPO loss matching the pinned slime reference vectors."""

    cfg = config or CispoConfig()
    cfg.validate()
    size = len(ppo_kl)
    if size == 0 or len(log_probs) != size or len(advantages) != size or len(loss_mask) != size:
        raise CispoError("CISPO tensors and mask must have equal, nonzero length")
    selected = sum(1 for flag in loss_mask if flag)
    if selected == 0:
        raise CispoError("CISPO selected-token denominator is zero")
    denominator = float(selected)
    losses: list[float] = []
    gradients: list[float] = []
    clipped = 0
    ratio_sum = 0.0
    for kl, log_prob, advantage, selected_token in zip(ppo_kl, log_probs, advantages, loss_mask, strict=True):
        ratio = math.exp(-kl)
        truncated = min(max(ratio, 1.0 - cfg.eps_clip), 1.0 + cfg.eps_clip_high)
        weight = 1.0 if selected_token else 0.0
        losses.append(-truncated * advantage * log_prob * weight)
        gradients.append(-truncated * advantage * weight / denominator)
        if selected_token:
            ratio_sum += ratio
            clipped += int(truncated != ratio)
    return CispoObjective(
        token_losses=tuple(losses),
        log_prob_gradients=tuple(gradients),
        selected_token_count=selected,
        loss=sum(losses) / denominator,
        clip_fraction=clipped / denominator,
        mean_ratio=ratio_sum / denominator,
        clipped_token_count=clipped,
    )


def importance_ratios(current_logprobs: Sequence[float], behavior_logprobs: Sequence[float]) -> tuple[float, ...]:
    if len(current_logprobs) != len(behavior_logprobs):
        raise CispoError("current and behavior log-probabilities must align")
    return tuple(
        math.exp(current - behavior)
        for current, behavior in zip(current_logprobs, behavior_logprobs, strict=True)
    )


def ppo_kl_from_ratio(ratio: float) -> float:
    if ratio <= 0.0:
        raise CispoError("importance ratio must be positive")
    return -math.log(ratio)


def normalize_group_rewards(rewards: Sequence[float]) -> tuple[float, ...]:
    """Matches ``torch.std``'s default unbiased estimator in slime's rollout path."""

    if len(rewards) < 2:
        raise CispoError("reward group must contain at least two samples when std normalization is enabled")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
    denominator = math.sqrt(variance) + 1e-6
    return tuple((reward - mean) / denominator for reward in rewards)


def group_advantages(rewards: Sequence[float], *, normalize: bool = True) -> tuple[float, ...]:
    if len(rewards) < 2:
        raise CispoError("CISPO groups require at least two trajectories")
    if not normalize:
        mean = sum(rewards) / len(rewards)
        return tuple(reward - mean for reward in rewards)
    return normalize_group_rewards(rewards)


def is_zero_advantage_group(advantages: Sequence[float], *, atol: float = 1e-8) -> bool:
    return all(abs(value) <= atol for value in advantages)


def clip_importance_ratio(ratio: float, config: CispoConfig | None = None) -> float:
    cfg = config or CispoConfig()
    cfg.validate()
    return min(max(ratio, 1.0 - cfg.eps_clip), 1.0 + cfg.eps_clip_high)
