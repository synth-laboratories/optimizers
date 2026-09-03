"""Sampler gateway: one renderer, immutable routes, and token evidence or nothing."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from synth_optimizers.contracts.rl_identity import GroupPin
from synth_optimizers.contracts.rl_records import (
    LOGPROB_SENTINEL,
    BehaviorFingerprint,
    EvidenceError,
    RendererProfile,
    SamplingProfile,
    digest,
)
from synth_optimizers.providers.protocols import ProviderUsage, SampleRequest, SampleResult
from synth_optimizers.rl.gateway import (
    COMPACT_MARKER,
    COMPACT_RULE,
    TRANSPORT_MESSAGE_IN,
    TRANSPORT_TOKENS_IN,
    TRUNCATE_RULE,
    WIRE_CHAT_COMPLETIONS,
    WIRE_RESPONSES,
    AttemptFactsError,
    ClosedOriginError,
    GatewayServer,
    PrimeChatRenderer,
    PrimeResponsesRenderer,
    PromptBudget,
    PromptBudgetError,
    RendererMismatchError,
    RouteRebindError,
    SamplerEvidenceError,
    SamplerGatewayService,
    UnknownOriginError,
    WireError,
)
from synth_optimizers.rl.ports import PolicyRevision, SamplerGateway

ROLE_TOKENS = {"system": 190, "developer": 189, "user": 191, "assistant": 200, "tool": 192}
GENERATION_PROMPT = 200
CONTRACT = "sha256:" + "ab" * 32


# --------------------------------------------------------------------- fakes


@dataclass
class StubPrimeRenderer:
    """A prefix-stable stand-in for the Prime ``renderers`` object.

    Codepoint tokenization, so a decoded generation re-renders to exactly the
    ids it came from and a tool loop is a real strict-prefix continuation.
    """

    rendered: list[int] = field(default_factory=list)

    def render_ids(
        self, rows: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False
    ) -> list[int]:
        ids: list[int] = []
        for row in rows:
            role = str(row.get("role", ""))
            if role not in ROLE_TOKENS:
                raise ValueError(f"stub renderer has no role token for {role!r}")
            ids.append(ROLE_TOKENS[role])
            ids.extend(ord(character) for character in str(row.get("content", "")))
        if add_generation_prompt:
            ids.append(GENERATION_PROMPT)
        return ids

    def get_stop_token_ids(self) -> list[int]:
        return [200002, 199999]

    def parse_response(self, token_ids: Sequence[int]) -> Any:
        text = "".join(chr(int(token)) for token in token_ids)
        return type("Parsed", (), {"content": text})()


def tokens_of(text: str) -> tuple[int, ...]:
    return tuple(ord(character) for character in text)


@dataclass
class ScriptedSampler:
    """Deterministic token-in/token-out sampling. No provider, no spend."""

    completions: list[str] = field(default_factory=lambda: ["ok"])
    logprob: float = -0.25
    logprobs_override: tuple[float, ...] | None = None
    finish_reason: str = "stop"
    echo_text: bool = True
    requests: list[SampleRequest] = field(default_factory=list)

    def sample_checkpoint(self, checkpoint: Any, request: SampleRequest) -> SampleResult:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.completions) - 1)
        text = self.completions[index]
        tokens = tokens_of(text)
        logprobs = (
            self.logprobs_override
            if self.logprobs_override is not None
            else (self.logprob,) * len(tokens)
        )
        return SampleResult(
            request_id=request.request_id,
            token_ids=tokens,
            logprobs=logprobs,
            text=text if self.echo_text else "",
            finish_reason=self.finish_reason,
            usage=ProviderUsage(
                input_tokens=len(request.prompt_token_ids), output_tokens=len(tokens)
            ),
        )


def profile(profile_id: str = "renderers.stub.v1") -> RendererProfile:
    return RendererProfile(
        profile_id=profile_id,
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:" + "cd" * 32,
        tokenizer_id="vendor/base-model-a",
        tokenizer_digest="sha256:" + "ef" * 32,
        stop_token_ids=(200002, 199999),
    )


def fingerprint(
    renderer_profile: RendererProfile,
    revision: int,
    *,
    wire_api: str = WIRE_CHAT_COMPLETIONS,
    transport: str = TRANSPORT_MESSAGE_IN,
) -> str:
    return BehaviorFingerprint(
        renderer_profile=renderer_profile,
        model_family="family_a",
        model_id="vendor/base-model-a",
        policy_revision=revision,
        wire_api=wire_api,
        sampling_transport=transport,
        sampling=SamplingProfile(),
    ).value


def make_revision(
    renderer_profile: RendererProfile,
    *,
    revision: int = 3,
    wire_api: str = WIRE_CHAT_COMPLETIONS,
    transport: str = TRANSPORT_MESSAGE_IN,
) -> PolicyRevision:
    return PolicyRevision(
        revision=revision,
        revision_id=f"pg_alpha@{revision}",
        checkpoint_id=f"ckpt_{revision}",
        parameter_group_id="pg_alpha",
        sampler_reference=f"provider://sampler/{revision}",
        behavior_fingerprint=fingerprint(
            renderer_profile, revision, wire_api=wire_api, transport=transport
        ),
        policy_set_revision_id="set_alpha@update_0001",
        metadata={"sampler_digest": "sha256:" + digest({"revision": revision})},
    )


def make_pin(
    revision: PolicyRevision,
    *,
    wire_api: str = WIRE_CHAT_COMPLETIONS,
    transport: str = TRANSPORT_MESSAGE_IN,
) -> GroupPin:
    return GroupPin(
        group_id="group_1",
        run_id="run_a",
        algorithm_plan_hash="sha256:" + "11" * 32,
        behavior_fingerprint=revision.behavior_fingerprint,
        policy_revision=revision.revision,
        wire_api=wire_api,
        sampling_transport=transport,
        policy_kind="declared_by_container",
        model_family="family_a",
        container_image_digest="sha256:" + "22" * 32,
        container_contract_hash=CONTRACT,
        handshake_agreement_digest="sha256:" + "33" * 32,
        task_family="family_one",
        cardinality=4,
        policy_set_revision_id=revision.policy_set_revision_id,
        policy_revision_id=revision.revision_id,
    )


def make_gateway(
    *,
    sampler: ScriptedSampler | None = None,
    budget: PromptBudget | None = None,
    responses_wire: bool = False,
) -> tuple[SamplerGatewayService, ScriptedSampler, RendererProfile]:
    stub = StubPrimeRenderer()
    base = profile()
    renderer = (
        PrimeResponsesRenderer.over(stub, base) if responses_wire else PrimeChatRenderer(stub, base)
    )
    backend = sampler or ScriptedSampler()
    gateway = SamplerGatewayService(
        renderer,
        backend,
        prompt_budget=budget or PromptBudget(max_prompt_tokens=100_000, policy="refuse"),
        credential_salt="test",
    )
    return gateway, backend, renderer.profile


def bind_attempt(
    gateway: SamplerGatewayService,
    renderer_profile: RendererProfile,
    *,
    attempt: str = "attempt_1",
    wire_api: str = WIRE_CHAT_COMPLETIONS,
    transport: str = TRANSPORT_MESSAGE_IN,
    revision: int = 3,
    declare: bool = True,
) -> PolicyRevision:
    policy = make_revision(
        renderer_profile, revision=revision, wire_api=wire_api, transport=transport
    )
    gateway.bind(
        policy,
        pin=make_pin(policy, wire_api=wire_api, transport=transport),
        sample_index=0,
        proxy_request_id=attempt,
    )
    if declare:
        gateway.declare_attempt(attempt, rollout_id="rollout_1", task_id="task_1", seed=7)
    return policy


def chat_body(messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
    return {"messages": messages, "max_tokens": 64, "temperature": 0.8, "seed": 1, **extra}


OPENING = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "first"},
]


# ----------------------------------------------------------------- binding


def test_gateway_satisfies_the_sampler_gateway_port() -> None:
    gateway, _sampler, _profile = make_gateway()
    assert isinstance(gateway, SamplerGateway)


def test_binding_the_same_proxy_id_twice_returns_the_same_origin() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    policy = make_revision(renderer_profile)
    pin = make_pin(policy)
    first = gateway.bind(policy, pin=pin, sample_index=0, proxy_request_id="attempt_1")
    second = gateway.bind(policy, pin=pin, sample_index=0, proxy_request_id="attempt_1")
    assert first == second
    assert first.base_url.endswith("/v1/attempts/attempt_1")
    assert first.policy_revision == policy.revision
    assert first.credential and policy.revision_id not in first.credential


def test_a_route_rebound_to_a_second_revision_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    first = make_revision(renderer_profile, revision=3)
    second = make_revision(renderer_profile, revision=4)
    gateway.bind(first, pin=make_pin(first), sample_index=0, proxy_request_id="attempt_1")
    with pytest.raises(RouteRebindError):
        gateway.bind(second, pin=make_pin(second), sample_index=0, proxy_request_id="attempt_1")
    assert gateway.origin("attempt_1").policy_revision == 3


def test_a_pin_that_disagrees_with_the_revision_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    policy = make_revision(renderer_profile, revision=3)
    other = make_revision(renderer_profile, revision=4)
    with pytest.raises(RouteRebindError):
        gateway.bind(policy, pin=make_pin(other), sample_index=0, proxy_request_id="attempt_1")


def test_binding_a_wire_the_renderer_does_not_serve_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    policy = make_revision(renderer_profile, wire_api=WIRE_RESPONSES)
    with pytest.raises(RendererMismatchError):
        gateway.bind(
            policy,
            pin=make_pin(policy, wire_api=WIRE_RESPONSES),
            sample_index=0,
            proxy_request_id="attempt_1",
        )


def test_calls_against_closed_or_unknown_origins_are_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    gateway.close("attempt_1")
    with pytest.raises(ClosedOriginError):
        gateway.handle("attempt_1", chat_body(OPENING))
    with pytest.raises(UnknownOriginError):
        gateway.handle("attempt_never_bound", chat_body(OPENING))
    with pytest.raises(UnknownOriginError):
        gateway.close("attempt_never_bound")
    # Retiring an origin retires sampling, not the evidence already captured.
    assert len(gateway.episode("attempt_1").segments) == 1


def test_a_credential_from_another_attempt_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile, attempt="attempt_1")
    bind_attempt(gateway, renderer_profile, attempt="attempt_2")
    with pytest.raises(UnknownOriginError):
        gateway.handle(
            "attempt_1",
            chat_body(OPENING),
            credential=gateway.origin("attempt_2").credential,
        )


# ------------------------------------------------------------ token capture


def test_a_proxied_call_records_exact_tokens_logprobs_and_identity() -> None:
    sampler = ScriptedSampler(completions=["ok"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    policy = bind_attempt(gateway, renderer_profile)
    body = gateway.handle("attempt_1", chat_body(OPENING))
    capture = body["synth_capture"]
    (call,) = gateway.calls("attempt_1")
    stub = StubPrimeRenderer()
    expected_prompt = tuple(stub.render_ids(OPENING, add_generation_prompt=True))
    assert call.prompt_token_ids == expected_prompt
    assert call.generation_token_ids == tokens_of("ok")
    assert call.generation_logprobs == (-0.25, -0.25)
    assert call.sampled_mask == (1, 1)
    assert call.token_capture_provenance == "engine_meta"
    assert call.renderer_profile_fingerprint == renderer_profile.fingerprint
    assert call.policy_revision == policy.revision
    assert call.behavior_fingerprint == policy.behavior_fingerprint
    assert call.finish_reason == "stop_token"
    assert call.stop_token_ids == renderer_profile.stop_token_ids
    assert call.wire_request["messages"] == OPENING
    assert call.wire_response["choices"][0]["message"]["content"] == "ok"
    assert capture["prompt_token_ids"] == list(expected_prompt)
    assert capture["generation_logprobs"] == [-0.25, -0.25]
    call.validate_for_training()


def test_the_gateway_decodes_when_the_provider_returns_no_text() -> None:
    sampler = ScriptedSampler(completions=["ok"], echo_text=False)
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    body = gateway.handle("attempt_1", chat_body(OPENING))
    assert body["choices"][0]["message"]["content"] == "ok"


def test_a_tool_loop_stitches_under_the_strict_prefix_rule() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    followup = [
        *OPENING,
        {"role": "assistant", "content": "aa"},
        {"role": "tool", "content": "observation"},
    ]
    gateway.handle("attempt_1", chat_body(followup))
    first, second = gateway.calls("attempt_1")
    assert second.prompt_token_ids[: len(first.full_sequence)] == first.full_sequence
    assert second.branch_id == first.branch_id == "root"
    assert second.compaction is None
    episode = gateway.episode("attempt_1")
    episode.validate()
    assert [segment.branch_id for segment in episode.segments] == ["root", "root"]


def test_an_unexplained_divergence_is_an_evidence_failure_and_is_not_stored() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    rewritten = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "rewritten"},
        {"role": "assistant", "content": "aa"},
    ]
    with pytest.raises(EvidenceError):
        gateway.handle("attempt_1", chat_body(rewritten))
    assert len(gateway.calls("attempt_1")) == 1


def test_a_declared_container_rewrite_forks_a_branch_with_provenance() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    rewritten = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "summary of the first turn"},
    ]
    gateway.handle(
        "attempt_1",
        chat_body(
            rewritten,
            synth_history_rewrite={"rule": "harness.summarize.v1", "removed_message_indices": [1]},
        ),
    )
    first, second = gateway.calls("attempt_1")
    assert second.branch_id == "root.1"
    assert second.parent_branch_id == first.branch_id
    assert second.compaction is not None
    assert second.compaction.rule == "harness.summarize.v1"
    assert second.compaction.removed_message_indices == (1,)
    episode = gateway.episode("attempt_1")
    assert [segment.branch_id for segment in episode.segments] == ["root", "root.1"]


# ---------------------------------------------------------- prompt budgets


def long_turns() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "s" * 20},
        {"role": "user", "content": "u" * 20},
        {"role": "assistant", "content": "a" * 20},
        {"role": "user", "content": "v" * 20},
    ]


def test_prompt_budget_refuse_fails_the_attempt_and_records_it() -> None:
    gateway, _sampler, renderer_profile = make_gateway(
        budget=PromptBudget(
            max_prompt_tokens=40, policy="refuse", reserve_completion_tokens=False
        )
    )
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(PromptBudgetError):
        gateway.handle("attempt_1", chat_body(long_turns()))
    (event,) = gateway.budget_events("attempt_1")
    assert event.policy == "refuse"
    assert event.refused is True
    assert event.prompt_tokens_before > 40
    assert gateway.calls("attempt_1") == ()


def test_prompt_budget_truncate_drops_oldest_turns_and_records_provenance() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(
        sampler=sampler,
        budget=PromptBudget(
            max_prompt_tokens=70,
            policy="truncate",
            keep_head_rows=1,
            keep_tail_rows=1,
            reserve_completion_tokens=False,
        ),
    )
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    gateway.handle("attempt_1", chat_body(long_turns()))
    _first, second = gateway.calls("attempt_1")
    assert len(second.prompt_token_ids) <= 70
    assert second.compaction is not None
    assert second.compaction.rule == TRUNCATE_RULE
    assert second.compaction.removed_message_indices == (1,)
    assert second.branch_id == "root.1"
    event = gateway.budget_events("attempt_1")[-1]
    assert event.policy == "truncate"
    assert event.prompt_tokens_after < event.prompt_tokens_before


def test_prompt_budget_compact_forks_a_branch_and_leaves_a_marker() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(
        sampler=sampler,
        budget=PromptBudget(
            max_prompt_tokens=70,
            policy="compact",
            keep_head_rows=1,
            keep_tail_rows=1,
            reserve_completion_tokens=False,
        ),
    )
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    gateway.handle("attempt_1", chat_body(long_turns()))
    _first, second = gateway.calls("attempt_1")
    assert second.compaction is not None
    assert second.compaction.rule == COMPACT_RULE
    assert second.compaction.removed_message_indices == (1, 2)
    assert second.branch_id == "root.1"
    assert second.parent_branch_id == "root"
    assert COMPACT_MARKER.format(count=2) in "".join(
        chr(token) for token in second.prompt_token_ids if 31 < token < 127
    )


def test_a_budget_that_cannot_be_met_is_refused_rather_than_gutted() -> None:
    gateway, _sampler, renderer_profile = make_gateway(
        budget=PromptBudget(
            max_prompt_tokens=10,
            policy="truncate",
            keep_head_rows=1,
            keep_tail_rows=1,
            reserve_completion_tokens=False,
        )
    )
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(PromptBudgetError):
        gateway.handle("attempt_1", chat_body(long_turns()))


def test_a_rewriting_budget_is_refused_on_a_tokens_in_transport() -> None:
    gateway, _sampler, renderer_profile = make_gateway(
        budget=PromptBudget(max_prompt_tokens=40, policy="compact")
    )
    policy = make_revision(renderer_profile, transport=TRANSPORT_TOKENS_IN)
    with pytest.raises(PromptBudgetError):
        gateway.bind(
            policy,
            pin=make_pin(policy, transport=TRANSPORT_TOKENS_IN),
            sample_index=0,
            proxy_request_id="attempt_1",
        )


# ------------------------------------------------------- sampling evidence


@pytest.mark.parametrize(
    ("override", "completions"),
    [
        ((LOGPROB_SENTINEL, LOGPROB_SENTINEL), ["ok"]),
        ((float("nan"), -0.2), ["ok"]),
        ((float("inf"), -0.2), ["ok"]),
        ((0.0, 0.0), ["ok"]),
        ((-0.2,), ["ok"]),
    ],
    ids=["sentinel", "nan", "inf", "identically_zero", "length_mismatch"],
)
def test_malformed_sampling_logprobs_are_refused_rather_than_stored(
    override: tuple[float, ...], completions: list[str]
) -> None:
    sampler = ScriptedSampler(completions=completions, logprobs_override=override)
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(SamplerEvidenceError):
        gateway.handle("attempt_1", chat_body(OPENING))
    assert gateway.calls("attempt_1") == ()


def test_an_unmappable_finish_reason_is_refused() -> None:
    sampler = ScriptedSampler(finish_reason="content_filter")
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(SamplerEvidenceError):
        gateway.handle("attempt_1", chat_body(OPENING))
    assert gateway.calls("attempt_1") == ()


def test_a_length_capped_span_records_its_own_finish_reason() -> None:
    sampler = ScriptedSampler(finish_reason="length")
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile)
    body = gateway.handle("attempt_1", chat_body(OPENING))
    (call,) = gateway.calls("attempt_1")
    assert call.finish_reason == "length_cap"
    assert body["choices"][0]["finish_reason"] == "length"


# ------------------------------------------------------------- the episode


def test_captured_evidence_validates_for_training_and_as_an_episode() -> None:
    sampler = ScriptedSampler(completions=["aa", "bb"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    policy = bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    gateway.handle(
        "attempt_1",
        chat_body(
            [*OPENING, {"role": "assistant", "content": "aa"}, {"role": "tool", "content": "o"}]
        ),
    )
    gateway.close("attempt_1")
    episode = gateway.episode("attempt_1")
    episode.validate()
    assert episode.rollout_id == "rollout_1"
    assert episode.task_id == "task_1"
    assert episode.seed == 7
    assert episode.policy_revision == policy.revision
    assert episode.behavior_fingerprint == policy.behavior_fingerprint
    assert episode.parameter_groups == ("pg_alpha",)
    assert episode.trace_digest
    for segment, call in zip(episode.segments, gateway.calls("attempt_1"), strict=True):
        call.validate_for_training()
        assert segment.author_kind == "policy"
        assert segment.trainable
        assert segment.trainable_tokens == len(call.generation_token_ids)
        assert segment.behavior_logprobs[len(call.prompt_token_ids) :] == call.generation_logprobs
        assert segment.loss_mask[: len(call.prompt_token_ids)] == (0,) * len(
            call.prompt_token_ids
        )


def test_an_episode_without_declared_attempt_facts_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile, declare=False)
    gateway.handle("attempt_1", chat_body(OPENING))
    with pytest.raises(AttemptFactsError):
        gateway.episode("attempt_1")


def test_an_episode_with_no_calls_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(EvidenceError):
        gateway.episode("attempt_1")


def test_attempt_facts_may_not_change_under_recorded_calls() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile)
    gateway.handle("attempt_1", chat_body(OPENING))
    with pytest.raises(AttemptFactsError):
        gateway.declare_attempt("attempt_1", rollout_id="other", task_id="task_1", seed=7)


# ------------------------------------------------------------------- wires


def test_the_responses_wire_is_served_under_its_own_renderer_identity() -> None:
    sampler = ScriptedSampler(completions=["ok"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler, responses_wire=True)
    chat_profile = profile()
    assert renderer_profile.fingerprint != chat_profile.fingerprint
    bind_attempt(gateway, renderer_profile, wire_api=WIRE_RESPONSES)
    body = gateway.handle(
        "attempt_1",
        {
            "input": [
                {"type": "message", "role": "system", "content": "sys"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                },
            ],
            "max_output_tokens": 64,
        },
    )
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "ok"
    (call,) = gateway.calls("attempt_1")
    assert call.wire_api == WIRE_RESPONSES
    assert "input" in call.wire_request
    call.validate_for_training()


def test_a_chat_payload_against_a_responses_route_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway(responses_wire=True)
    bind_attempt(gateway, renderer_profile, wire_api=WIRE_RESPONSES)
    with pytest.raises(WireError):
        gateway.handle("attempt_1", chat_body(OPENING), wire_api=WIRE_CHAT_COMPLETIONS)
    with pytest.raises(WireError):
        gateway.handle("attempt_1", {"messages": OPENING, "max_output_tokens": 8})


def test_an_unknown_responses_item_type_is_refused_not_dropped() -> None:
    gateway, _sampler, renderer_profile = make_gateway(responses_wire=True)
    bind_attempt(gateway, renderer_profile, wire_api=WIRE_RESPONSES)
    with pytest.raises(WireError):
        gateway.handle(
            "attempt_1",
            {"input": [{"type": "reasoning", "summary": []}], "max_output_tokens": 8},
        )


def test_the_tokens_in_transport_records_the_tokens_it_was_sent() -> None:
    sampler = ScriptedSampler(completions=["ok"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    bind_attempt(gateway, renderer_profile, transport=TRANSPORT_TOKENS_IN)
    prompt = [190, 115, 121, 115, 200]
    body = gateway.handle(
        "attempt_1",
        {
            "prompt_token_ids": prompt,
            "max_tokens": 16,
            "renderer_profile_fingerprint": renderer_profile.fingerprint,
        },
    )
    (call,) = gateway.calls("attempt_1")
    assert call.prompt_token_ids == tuple(prompt)
    assert call.sampling_transport == TRANSPORT_TOKENS_IN
    assert body["synth_capture"]["sampling_transport"] == TRANSPORT_TOKENS_IN


def test_a_tokens_in_call_declaring_a_second_renderer_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile, transport=TRANSPORT_TOKENS_IN)
    with pytest.raises(RendererMismatchError):
        gateway.handle(
            "attempt_1",
            {
                "prompt_token_ids": [190, 200],
                "max_tokens": 8,
                "renderer_profile_fingerprint": "sha256:someone-elses-renderer",
            },
        )


def test_a_tokens_in_call_without_token_ids_is_refused() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile, transport=TRANSPORT_TOKENS_IN)
    with pytest.raises(WireError):
        gateway.handle("attempt_1", chat_body(OPENING))


# ----------------------------------------------------------- http surface


def post(url: str, payload: Mapping[str, Any], *, credential: str | None = None) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode(),
        method="POST",
        headers={
            "content-type": "application/json",
            **({"authorization": f"Bearer {credential}"} if credential else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_the_origin_path_carries_the_attempt_id_over_http() -> None:
    sampler = ScriptedSampler(completions=["ok"])
    gateway, _sampler, renderer_profile = make_gateway(sampler=sampler)
    with GatewayServer(gateway) as server:
        assert gateway.origin_root == server.base_url
        policy = make_revision(renderer_profile)
        origin = gateway.bind(
            policy, pin=make_pin(policy), sample_index=0, proxy_request_id="attempt_http"
        )
        gateway.declare_attempt(
            "attempt_http", rollout_id="rollout_http", task_id="task_1", seed=1
        )
        body = post(
            f"{origin.base_url}/chat/completions",
            chat_body(OPENING),
            credential=origin.credential,
        )
        assert body["choices"][0]["message"]["content"] == "ok"
        assert body["synth_capture"]["proxy_request_id"] == "attempt_http"
        gateway.close("attempt_http")
        with pytest.raises(urllib.error.HTTPError) as closed:
            post(
                f"{origin.base_url}/chat/completions",
                chat_body(OPENING),
                credential=origin.credential,
            )
        assert closed.value.code == 409
        with pytest.raises(urllib.error.HTTPError) as unknown:
            post(f"{server.base_url}/v1/attempts/nobody/chat/completions", chat_body(OPENING))
        assert unknown.value.code == 404
    gateway.episode("attempt_http").validate()


def test_the_origin_root_may_not_move_once_a_route_is_bound() -> None:
    gateway, _sampler, renderer_profile = make_gateway()
    bind_attempt(gateway, renderer_profile)
    with pytest.raises(RouteRebindError):
        gateway.set_origin_root("http://127.0.0.1:1")


def test_binding_carries_the_attempt_facts_the_episode_will_need() -> None:
    """The facts arrive with the binding, not after the evidence exists.

    A pin names a task family; an episode record demands a task id and a seed.
    Passing them at bind time is what stops a run training on evidence that
    names the wrong task.
    """

    from synth_optimizers.rl.ports import AttemptFacts, PortError

    facts = AttemptFacts(rollout_id="rollout-1", task_id="task-7", seed=11)
    assert facts.terminal_status == "completed"
    for bad in ({"rollout_id": " "}, {"task_id": ""}):
        payload = {"rollout_id": "rollout-1", "task_id": "task-7", "seed": 11, **bad}
        with pytest.raises(PortError, match="must name its rollout and its task"):
            AttemptFacts(**payload)
