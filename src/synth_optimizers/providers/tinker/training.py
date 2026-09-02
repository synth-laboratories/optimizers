from __future__ import annotations

from typing import Any

from ..protocols import (
    ForwardRequest,
    ForwardResult,
    ProviderError,
    ProviderSession,
    ProviderUsage,
    TrainingStepRequest,
    TrainingStepResult,
)


def run_training_step(
    client: Any,
    session: ProviderSession,
    request: TrainingStepRequest,
) -> TrainingStepResult:
    trainer = getattr(client, "train_step", None)
    if not callable(trainer):
        raise ProviderError("train_unsupported", "Tinker training is unavailable")
    payload = trainer(session, request)
    usage = payload.get("usage", {})
    return TrainingStepResult(
        request_id=request.request_id,
        step=int(payload.get("step", 0)),
        metrics={str(key): float(value) for key, value in dict(payload.get("metrics", {})).items()},
        usage=_usage(usage),
    )


def forward_logprobs(
    client: Any,
    session: ProviderSession,
    request: ForwardRequest,
) -> ForwardResult:
    forward = getattr(client, "forward", None)
    if not callable(forward):
        raise ProviderError("forward_unsupported", "Tinker logprob forward is unavailable")
    payload = forward(session, request)
    return ForwardResult(
        request_id=request.request_id,
        logprobs=tuple(tuple(float(value) for value in row) for row in payload.get("logprobs", ())),
        usage=_usage(payload.get("usage", {})),
    )


def _usage(payload: Any) -> ProviderUsage:
    data = payload if isinstance(payload, dict) else {}
    cost = data.get("cost_usd")
    return ProviderUsage(
        input_tokens=int(data.get("input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        training_tokens=int(data.get("training_tokens", 0)),
        cost_usd=None if cost is None else float(cost),
        cost_missing=cost is None,
    )
