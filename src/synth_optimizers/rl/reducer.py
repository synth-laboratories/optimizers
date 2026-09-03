"""The reducer dimension: a real interface, not a hidden ``.mean()``.

What a batch divides by is an algorithm decision with a large effect and no
obvious default, so it is a plan field with a table of implementations. Three
subagent branches of one environment attempt must not triple that attempt's
optimization weight, which is what ``branch_aware_root_mean`` exists for.

Pure functions over per-item scalars. Nothing here allocates a tensor, calls a
provider, or knows what an objective is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .plan import LossReducer


class ReducerError(ValueError):
    """Misaligned reducer inputs, or a reducer with no table entry."""


@dataclass(frozen=True, slots=True)
class ReducedLoss:
    """The scalar and the denominator that produced it."""

    kind: str
    value: float
    denominator: float
    contributing_items: int

    def receipt(self) -> dict[str, Any]:
        return {
            "reducer": self.kind,
            "loss": self.value,
            "denominator": self.denominator,
            "contributing_items": self.contributing_items,
        }


def _check(
    per_item_loss: Sequence[float],
    per_item_tokens: Sequence[int],
    root_ids: Sequence[str] | None,
) -> None:
    if len(per_item_loss) != len(per_item_tokens):
        raise ReducerError("per-item loss and token counts must align")
    if root_ids is not None and len(root_ids) != len(per_item_loss):
        raise ReducerError("root ids must align with per-item losses")


def _roots(root_ids: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, root in enumerate(root_ids):
        grouped.setdefault(root, []).append(index)
    return grouped


def branch_aware_root_mean(
    *,
    per_item_loss: Sequence[float],
    per_item_tokens: Sequence[int],
    root_ids: Sequence[str],
    **_: Any,
) -> ReducedLoss:
    """Token-weighted inside a root rollout, then one vote per root rollout."""

    _check(per_item_loss, per_item_tokens, root_ids)
    totals: list[float] = []
    contributing = 0
    for indices in _roots(root_ids).values():
        token_total = sum(per_item_tokens[index] for index in indices)
        if token_total <= 0:
            continue
        loss_total = math.fsum(per_item_loss[index] for index in indices)
        totals.append(loss_total / token_total)
        contributing += len(indices)
    if not totals:
        return ReducedLoss(
            kind="branch_aware_root_mean", value=0.0, denominator=0.0, contributing_items=0
        )
    return ReducedLoss(
        kind="branch_aware_root_mean",
        value=math.fsum(totals) / len(totals),
        denominator=float(len(totals)),
        contributing_items=contributing,
    )


def token_mean(
    *, per_item_loss: Sequence[float], per_item_tokens: Sequence[int], **_: Any
) -> ReducedLoss:
    """One denominator for the whole batch: its trainable token count."""

    _check(per_item_loss, per_item_tokens, None)
    total = sum(per_item_tokens)
    if total <= 0:
        return ReducedLoss(kind="token_mean", value=0.0, denominator=0.0, contributing_items=0)
    return ReducedLoss(
        kind="token_mean",
        value=math.fsum(per_item_loss) / total,
        denominator=float(total),
        contributing_items=sum(1 for tokens in per_item_tokens if tokens > 0),
    )


def sequence_mean(
    *, per_item_loss: Sequence[float], per_item_tokens: Sequence[int], **_: Any
) -> ReducedLoss:
    """Token-weighted inside a sequence, then one vote per sequence."""

    _check(per_item_loss, per_item_tokens, None)
    per_sequence = [
        loss / tokens
        for loss, tokens in zip(per_item_loss, per_item_tokens, strict=True)
        if tokens > 0
    ]
    if not per_sequence:
        return ReducedLoss(
            kind="sequence_mean", value=0.0, denominator=0.0, contributing_items=0
        )
    return ReducedLoss(
        kind="sequence_mean",
        value=math.fsum(per_sequence) / len(per_sequence),
        denominator=float(len(per_sequence)),
        contributing_items=len(per_sequence),
    )


def root_rollout_mean(
    *,
    per_item_loss: Sequence[float],
    per_item_tokens: Sequence[int],
    root_ids: Sequence[str],
    **_: Any,
) -> ReducedLoss:
    """One vote per root rollout, with its branches averaged as sequences.

    Differs from ``branch_aware_root_mean`` in the inner denominator: here a
    long branch and a short branch of one attempt count equally, rather than in
    proportion to their tokens.
    """

    _check(per_item_loss, per_item_tokens, root_ids)
    totals: list[float] = []
    contributing = 0
    for indices in _roots(root_ids).values():
        live = [index for index in indices if per_item_tokens[index] > 0]
        if not live:
            continue
        totals.append(
            math.fsum(per_item_loss[index] / per_item_tokens[index] for index in live)
            / len(live)
        )
        contributing += len(live)
    if not totals:
        return ReducedLoss(
            kind="root_rollout_mean", value=0.0, denominator=0.0, contributing_items=0
        )
    return ReducedLoss(
        kind="root_rollout_mean",
        value=math.fsum(totals) / len(totals),
        denominator=float(len(totals)),
        contributing_items=contributing,
    )


def fixed_token_denominator(
    *,
    per_item_loss: Sequence[float],
    per_item_tokens: Sequence[int],
    token_denominator: float | None = None,
    **_: Any,
) -> ReducedLoss:
    """A denominator that does not move with the batch's realized length."""

    _check(per_item_loss, per_item_tokens, None)
    if token_denominator is None or token_denominator <= 0:
        raise ReducerError("fixed_token_denominator requires a positive token_denominator")
    return ReducedLoss(
        kind="fixed_token_denominator",
        value=math.fsum(per_item_loss) / float(token_denominator),
        denominator=float(token_denominator),
        contributing_items=sum(1 for tokens in per_item_tokens if tokens > 0),
    )


REDUCER_KERNELS: Mapping[str, Callable[..., ReducedLoss]] = {
    "branch_aware_root_mean": branch_aware_root_mean,
    "token_mean": token_mean,
    "sequence_mean": sequence_mean,
    "root_rollout_mean": root_rollout_mean,
    "fixed_token_denominator": fixed_token_denominator,
}


def kernel_for(kind: str) -> Callable[..., ReducedLoss]:
    if kind not in REDUCER_KERNELS:
        raise ReducerError(f"unknown reducer {kind!r}; known: {sorted(REDUCER_KERNELS)}")
    return REDUCER_KERNELS[kind]


def reduce_loss(reducer: LossReducer, **kwargs: Any) -> ReducedLoss:
    """Run the plan's reducer dimension."""

    return kernel_for(reducer.kind)(**kwargs)


def branch_aware_root_coefficients(
    *,
    per_item_tokens: Sequence[int],
    root_ids: Sequence[str],
    root_weights: Sequence[float],
) -> tuple[float, ...]:
    """The same reduction as per-item scalars, for streaming backward passes.

    Holding every item's autograd graph until one backward is how a long-horizon
    agentic batch runs a box out of memory. These coefficients let each item go
    backward on its own and still land on the loss ``branch_aware_root_mean``
    would have produced.
    """

    if not (len(per_item_tokens) == len(root_ids) == len(root_weights)):
        raise ReducerError("coefficient inputs must align")
    grouped = _roots(root_ids)
    live = {
        root: sum(per_item_tokens[index] for index in indices)
        for root, indices in grouped.items()
    }
    live = {root: total for root, total in live.items() if total > 0}
    if not live:
        return (0.0,) * len(root_ids)
    coefficients = [0.0] * len(root_ids)
    for root, indices in grouped.items():
        if root not in live:
            continue
        for index in indices:
            coefficients[index] = float(root_weights[index]) / (live[root] * len(live))
    return tuple(coefficients)


COEFFICIENT_KERNELS: Mapping[str, Callable[..., tuple[float, ...]]] = {
    "branch_aware_root_mean": branch_aware_root_coefficients,
}


def coefficients(kind: str, **kwargs: Any) -> tuple[float, ...]:
    if kind not in COEFFICIENT_KERNELS:
        raise ReducerError(f"reducer {kind!r} has no streaming coefficient form in this plane")
    return COEFFICIENT_KERNELS[kind](**kwargs)
