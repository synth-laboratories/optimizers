from __future__ import annotations

from typing import Any

from ..protocols import ProviderCheckpoint, ProviderError, ProviderSession, SampleRequest, SampleResult, ProviderUsage


def sample_tokens(
    client: Any,
    handle: ProviderSession | ProviderCheckpoint,
    request: SampleRequest,
) -> SampleResult:
    sampler = getattr(client, "sample", None)
    if not callable(sampler):
        raise ProviderError("sample_unsupported", "Tinker sampling is unavailable")
    payload = sampler(handle, request)
    usage = payload.get("usage", {})
    return SampleResult(
        request_id=request.request_id,
        token_ids=tuple(int(token) for token in payload.get("token_ids", ())),
        logprobs=tuple(float(value) for value in payload.get("logprobs", ())),
        text=str(payload.get("text", "")),
        finish_reason=str(payload.get("finish_reason", "stop")),
        usage=ProviderUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            training_tokens=int(usage.get("training_tokens", 0)),
            cost_usd=usage.get("cost_usd"),
            cost_missing=usage.get("cost_usd") is None,
        ),
    )
