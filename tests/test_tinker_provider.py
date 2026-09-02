from __future__ import annotations

import pytest

from synth_optimizers.providers.tinker import FakeTinkerProvider, TinkerAdapter, TinkerCredentials, new_request_id
from synth_optimizers.providers.protocols import (
    CISPO_REQUIRED_CAPABILITIES,
    ProviderError,
    SampleRequest,
    TrainingStepRequest,
    UnsupportedCapability,
)


def test_adapter_is_idempotent_and_does_not_duplicate_paid_work() -> None:
    transport = FakeTinkerProvider()
    adapter = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    request_id = new_request_id("session", "once")
    first = adapter.create_session("gpt-oss-20b", rank=8, seed=1, request_id=request_id)
    second = adapter.create_session("gpt-oss-20b", rank=8, seed=1, request_id=request_id)
    assert first.session_id == second.session_id
    assert first.model_id == "openai/gpt-oss-20b"


def test_retryable_errors_are_classified_and_bounded() -> None:
    transport = FakeTinkerProvider(fail_once="sample")
    adapter = TinkerAdapter(
        TinkerCredentials(api_key="fixture"), transport=transport, max_attempts=2, sleep=lambda _delay: None
    )
    session = adapter.create_session("openai/gpt-oss-20b", rank=4, seed=0, request_id="sess")
    result = adapter.sample(
        session,
        SampleRequest(request_id="sample-1", prompt_token_ids=(1, 2), max_tokens=4),
    )
    assert result.request_id == "sample-1"
    assert transport.calls.count(("sample", "sample-1")) == 2
    replay = adapter.sample(
        session,
        SampleRequest(request_id="sample-1", prompt_token_ids=(1, 2), max_tokens=4),
    )
    assert replay.text == result.text
    assert "sample-1" in transport.paid_requests


def test_generic_importance_sampling_cannot_claim_cispo() -> None:
    transport = FakeTinkerProvider()
    adapter = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    session = adapter.create_session("openai/gpt-oss-20b", rank=4, seed=0, request_id="sess")
    with pytest.raises(ProviderError, match="not cispo.slime.v1"):
        adapter.train_step(
            session,
            TrainingStepRequest(request_id="is-1", loss_name="importance_sampling", data=({},)),
        )


def test_missing_cispo_capability_fails_closed() -> None:
    transport = FakeTinkerProvider(offered_capabilities={"sft.train", "checkpoint.sample"})
    adapter = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    capabilities = adapter.discover_capabilities("openai/gpt-oss-20b")
    with pytest.raises(UnsupportedCapability):
        capabilities.require(CISPO_REQUIRED_CAPABILITIES)
    with pytest.raises(ProviderError, match="unsupported"):
        adapter.require_cispo("openai/gpt-oss-20b")
