"""The objective dimension: small pure kernels, never the algorithm.

CISPO clips the importance ratio and stop-grads it into the weight; the
gradient flows only through ``log pi``. That is why it is token level and why a
sequence-level importance ratio is an illegal combination for it, and it is why
a generic importance-sampling run is not CISPO.

Every objective is a row in :data:`OBJECTIVE_KERNELS`. Adding one is a row plus
a preset; nothing in this module, in ``credit.py``, in ``reducer.py``, or in
``assembly.py`` branches on which objective is running. Functions here are pure
over token sequences: no provider call, no tensor library, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .plan import PolicyObjective


class ObjectiveError(ValueError):
    """Malformed objective inputs, or an objective with no table entry."""


@dataclass(frozen=True, slots=True)
class TokenLoss:
    """Per-token loss and the clipping evidence that produced it."""

    per_token_loss: tuple[float, ...]
    per_token_weight: tuple[float, ...]
    selected_tokens: int
    clipped_fraction: float
    mean_ratio: float
    max_ratio: float
    clipped_tokens: int
    ratio_granularity: str

    @property
    def total_loss(self) -> float:
        return math.fsum(self.per_token_loss)

    def receipt(self) -> dict[str, Any]:
        return {
            "selected_tokens": self.selected_tokens,
            "clipped_tokens": self.clipped_tokens,
            "clipped_fraction": self.clipped_fraction,
            "mean_ratio": self.mean_ratio,
            "max_ratio": self.max_ratio,
            "ratio_granularity": self.ratio_granularity,
            "total_loss": self.total_loss,
        }


def broadcast_advantage(advantage: float, length: int) -> tuple[float, ...]:
    """One per-sample advantage spread over that sample's tokens."""

    if length < 0:
        raise ObjectiveError("length must be non-negative")
    return (float(advantage),) * length


def importance_ratios(
    current_logprobs: Sequence[float], behavior_logprobs: Sequence[float]
) -> tuple[float, ...]:
    if len(current_logprobs) != len(behavior_logprobs):
        raise ObjectiveError("current and behavior log-probabilities must align")
    return tuple(
        math.exp(float(current) - float(behavior))
        for current, behavior in zip(current_logprobs, behavior_logprobs, strict=True)
    )


def clip_bounds(eps_low: float, eps_high: float) -> tuple[float, float]:
    if eps_low < 0.0 or eps_high < 0.0:
        raise ObjectiveError("clip bounds must be non-negative")
    return max(0.0, 1.0 - float(eps_low)), 1.0 + float(eps_high)


def _align(
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
) -> int:
    size = len(current_logprobs)
    if size == 0:
        raise ObjectiveError("objective needs at least one token")
    if not (len(behavior_logprobs) == len(advantages) == len(mask) == size):
        raise ObjectiveError("logprobs, advantages, and mask must have equal length")
    selected = sum(1 for flag in mask if flag)
    if selected == 0:
        raise ObjectiveError("objective selected-token denominator is zero")
    return selected


def _sequence_ratio(
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    mask: Sequence[int],
) -> float:
    """One ratio for the whole sequence: exp of the masked mean log-ratio."""

    deltas = [
        float(current) - float(behavior)
        for current, behavior, flag in zip(
            current_logprobs, behavior_logprobs, mask, strict=True
        )
        if flag
    ]
    return math.exp(math.fsum(deltas) / len(deltas))


def _clipped_weight_loss(
    *,
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    eps_low: float,
    eps_high: float,
    ratio_granularity: str,
) -> TokenLoss:
    """The shared kernel: a clipped, stop-gradded weight times ``log pi``.

    ``ratio_granularity`` is the only difference between the token-level and
    sequence-level members of this family, and it is a plan field.
    """

    selected = _align(current_logprobs, behavior_logprobs, advantages, mask)
    lower, upper = clip_bounds(eps_low, eps_high)
    if ratio_granularity == "token":
        ratios = importance_ratios(current_logprobs, behavior_logprobs)
    elif ratio_granularity == "sequence":
        shared = _sequence_ratio(current_logprobs, behavior_logprobs, mask)
        ratios = (shared,) * len(current_logprobs)
    else:
        raise ObjectiveError(f"unknown ratio granularity {ratio_granularity!r}")

    losses: list[float] = []
    weights: list[float] = []
    clipped = 0
    ratio_sum = 0.0
    max_ratio = 0.0
    for ratio, advantage, current, flag in zip(
        ratios, advantages, current_logprobs, mask, strict=True
    ):
        truncated = min(max(ratio, lower), upper)
        active = 1.0 if flag else 0.0
        weights.append(truncated * active)
        losses.append(-truncated * float(advantage) * float(current) * active)
        if flag:
            ratio_sum += ratio
            max_ratio = max(max_ratio, ratio)
            clipped += int(truncated != ratio)
    return TokenLoss(
        per_token_loss=tuple(losses),
        per_token_weight=tuple(weights),
        selected_tokens=selected,
        clipped_fraction=clipped / selected,
        mean_ratio=ratio_sum / selected,
        max_ratio=max_ratio,
        clipped_tokens=clipped,
        ratio_granularity=ratio_granularity,
    )


def cispo(
    *,
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    eps_low: float,
    eps_high: float,
    variant: str = "cispo_minimax",
    **_: Any,
) -> TokenLoss:
    """Token-level CISPO. ``cispo_minimax`` disables the lower bound."""

    if variant == "cispo_minimax" and eps_low < 1.0:
        raise ObjectiveError("cispo_minimax requires eps_low >= 1 (one-sided clip)")
    return _clipped_weight_loss(
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        mask=mask,
        eps_low=eps_low,
        eps_high=eps_high,
        ratio_granularity="token",
    )


def gspo(
    *,
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    eps_low: float,
    eps_high: float,
    **_: Any,
) -> TokenLoss:
    """The same kernel with a sequence-level importance ratio."""

    return _clipped_weight_loss(
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        mask=mask,
        eps_low=eps_low,
        eps_high=eps_high,
        ratio_granularity="sequence",
    )


def reinforce(
    *,
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    **_: Any,
) -> TokenLoss:
    """No importance weight at all: the on-policy score-function estimator."""

    selected = _align(current_logprobs, behavior_logprobs, advantages, mask)
    losses = tuple(
        -float(advantage) * float(current) * (1.0 if flag else 0.0)
        for advantage, current, flag in zip(advantages, current_logprobs, mask, strict=True)
    )
    return TokenLoss(
        per_token_loss=losses,
        per_token_weight=tuple(1.0 if flag else 0.0 for flag in mask),
        selected_tokens=selected,
        clipped_fraction=0.0,
        mean_ratio=1.0,
        max_ratio=1.0,
        clipped_tokens=0,
        ratio_granularity="token",
    )


def _unimplemented(kind: str) -> Callable[..., TokenLoss]:
    def kernel(**_kwargs: Any) -> TokenLoss:
        raise ObjectiveError(
            f"objective {kind!r} expands as a plan dimension but has no kernel in this plane"
        )

    return kernel


OBJECTIVE_KERNELS: Mapping[str, Callable[..., TokenLoss]] = {
    "cispo": cispo,
    "gspo": gspo,
    "reinforce": reinforce,
    "ppo_clipped": _unimplemented("ppo_clipped"),
    "jsd_distillation": _unimplemented("jsd_distillation"),
    "sampled_distillation": _unimplemented("sampled_distillation"),
}


def kernel_for(kind: str) -> Callable[..., TokenLoss]:
    if kind not in OBJECTIVE_KERNELS:
        raise ObjectiveError(f"unknown objective {kind!r}; known: {sorted(OBJECTIVE_KERNELS)}")
    return OBJECTIVE_KERNELS[kind]


def evaluate(
    objective: PolicyObjective,
    *,
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
) -> TokenLoss:
    """Run the plan's objective dimension. The plan supplies every constant."""

    return kernel_for(objective.kind)(
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        mask=mask,
        eps_low=objective.eps_low,
        eps_high=objective.eps_high,
        variant=objective.variant,
        granularity=objective.granularity,
        ratio_granularity=objective.ratio_granularity,
    )
