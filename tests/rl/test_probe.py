"""Stream 2: the probe episode validator, and the line it draws around training."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from synth_optimizers.contracts.rl_records import (
    BehaviorFingerprint,
    CompactionProvenance,
    EvidenceError,
    HorizonEvidence,
    InferenceCall,
    RecordError,
    RendererProfile,
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
)
from synth_optimizers.rl.probe import (
    REQUIRED_PROBE_OPERATIONS,
    ProbeAttempt,
    ProbeError,
    ProbeNotDistinguishable,
    validate_probe,
)

PROFILE = RendererProfile(
    profile_id="renderers.pinned.low.v1",
    package="renderers",
    package_version="0.1.11",
    config_digest="sha256:cfg",
    tokenizer_id="vendor/policy-20b",
    tokenizer_digest="sha256:tok",
    stop_token_ids=(200002, 199999),
)

BEHAVIOR = BehaviorFingerprint(
    renderer_profile=PROFILE,
    model_family="policy_family",
    model_id="vendor/policy-20b",
    policy_revision=0,
    wire_api="chat_completions",
    sampling_transport="message_in_capture_out",
)

TRACE_DIGEST = "sha256:probe-trace"


def call(index: int, **overrides: Any) -> InferenceCall:
    prompts = {1: (11, 12, 13), 2: (11, 12, 13, 21, 22, 31)}
    generations = {1: (21, 22), 2: (41, 42)}
    payload: dict[str, Any] = {
        "call_id": f"call-{index}",
        "proxy_request_id": f"proxy-{index}",
        "rollout_id": "ro-1",
        "group_id": "group-1",
        "sample_index": 0,
        "behavior_fingerprint": BEHAVIOR.value,
        "policy_revision": 0,
        "wire_api": "chat_completions",
        "sampling_transport": "message_in_capture_out",
        "token_capture_provenance": "probe_synthetic",
        "prompt_token_ids": prompts[index],
        "generation_token_ids": generations[index],
        "generation_logprobs": (-0.5, -0.25),
        "sampled_mask": (1, 1),
        "finish_reason": "stop_token",
        "stop_token_ids": PROFILE.stop_token_ids,
        "trainable": False,
    }
    payload.update(overrides)
    return InferenceCall(**payload)


def episode(**overrides: Any) -> TrainableEpisode:
    payload: dict[str, Any] = {
        "rollout_id": "ro-1",
        "task_id": "task-a",
        "seed": 7,
        "policy_revision": 0,
        "behavior_fingerprint": BEHAVIOR.value,
        "segments": (
            TrainableSegment(
                token_ids=(11, 12, 13, 21, 22),
                loss_mask=(0, 0, 0, 1, 1),
                behavior_logprobs=(0.0, 0.0, 0.0, -0.5, -0.25),
                call_ids=("call-1",),
            ),
        ),
        "terminal_status": "completed",
        "trace_digest": TRACE_DIGEST,
    }
    payload.update(overrides)
    return TrainableEpisode(**payload)


def reward(**overrides: Any) -> RewardRecord:
    payload: dict[str, Any] = {
        "reward_id": "rw-1",
        "rollout_id": "ro-1",
        "trace_digest": TRACE_DIGEST,
        "channels": (RewardChannel(channel_id="outcome", team_id="team-1", measure=1.0),),
        "optimized_channel": "outcome",
        "terminal_status": "completed",
        "evaluation_plan_id": "plan-1",
        "horizon": HorizonEvidence(
            horizon_kind="wall_clock",
            horizon_value=5400.0,
            scored_at_offset_seconds=0.0,
            clipped=False,
            quiescence_attested=True,
        ),
    }
    payload.update(overrides)
    return RewardRecord(**payload)


def attempt(**overrides: Any) -> ProbeAttempt:
    payload: dict[str, Any] = {
        "rollout_id": "ro-1",
        "behavior": BEHAVIOR,
        "calls": (call(1), call(2)),
        "episode": episode(),
        "reward": reward(),
        "event_cursors": (1, 2, 7, 12),
        "terminal_results": ("completed",),
        "operations": frozenset(REQUIRED_PROBE_OPERATIONS),
        "resubmit_rollout_id": "ro-1",
        "cancelled_rollout_id": "ro-cancel-1",
        "trace_digest": TRACE_DIGEST,
    }
    payload.update(overrides)
    return ProbeAttempt(**payload)


def test_probe_validates_the_full_evidence_path() -> None:
    report = validate_probe(attempt(), expected_profile=PROFILE, quiescence_accepted=True)
    assert report.trainable is False
    assert report.calls_checked == 2
    assert report.segments_checked == 1
    assert report.operations == tuple(sorted(REQUIRED_PROBE_OPERATIONS))
    assert report.renderer_fingerprint == PROFILE.fingerprint
    assert report.quiescence_attested is True
    assert report.evidence_digest.startswith("sha256:")
    assert report.to_payload()["rollout_id"] == "ro-1"


def test_probe_evidence_indistinguishable_from_real_evidence_is_refused() -> None:
    real = (
        call(1, token_capture_provenance="engine_meta", trainable=True),
        call(2, token_capture_provenance="engine_meta", trainable=True),
    )
    with pytest.raises(ProbeNotDistinguishable) as excinfo:
        validate_probe(
            attempt(calls=real), expected_profile=PROFILE, quiescence_accepted=True
        )
    assert "probe_synthetic" in str(excinfo.value)


def test_probe_marked_trainable_is_refused() -> None:
    with pytest.raises(ProbeNotDistinguishable) as excinfo:
        validate_probe(
            attempt(calls=(call(1, trainable=True), call(2))),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "may never enter a group" in str(excinfo.value)


def test_probe_evidence_is_rejected_for_training_by_its_own_records() -> None:
    for record in attempt().calls:
        with pytest.raises(EvidenceError):
            record.validate_for_training()


def test_a_single_turn_probe_says_it_could_not_check_the_prefix() -> None:
    """A one-turn horizon has one turn to give, and that is not a defect.

    Demanding a second made such containers synthesize a turn they never ran,
    which is a worse answer than recording plainly that the property went
    unchecked. The probe still validates everything else about the path.
    """

    report = validate_probe(
        attempt(calls=(call(1),)), expected_profile=PROFILE, quiescence_accepted=True
    )
    assert report.calls_checked == 1
    assert report.prefix_checked is False
    assert report.to_payload()["prefix_checked"] is False

    two_turns = validate_probe(
        attempt(calls=(call(1), call(2))), expected_profile=PROFILE, quiescence_accepted=True
    )
    assert two_turns.prefix_checked is True


def test_a_probe_that_made_no_call_proves_nothing() -> None:
    with pytest.raises(ProbeError, match="proves nothing"):
        validate_probe(attempt(calls=()), expected_profile=PROFILE, quiescence_accepted=True)


def test_unexplained_prefix_divergence_is_an_evidence_failure() -> None:
    forked = call(2, prompt_token_ids=(11, 99, 13, 21, 22, 31))
    with pytest.raises(EvidenceError) as excinfo:
        validate_probe(
            attempt(calls=(call(1), forked)),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "diverges" in str(excinfo.value)


def test_declared_compaction_forks_a_branch_and_still_validates() -> None:
    forked = call(
        2,
        prompt_token_ids=(11, 12, 90, 91),
        branch_id="branch-1",
        parent_branch_id="root",
        compaction=CompactionProvenance(rule="summarize", divergence_index=2),
    )
    report = validate_probe(
        attempt(calls=(call(1), forked)),
        expected_profile=PROFILE,
        quiescence_accepted=True,
    )
    assert report.calls_checked == 2


def test_logprob_length_disagreement_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(calls=(call(1, generation_logprobs=(-0.5,)), call(2))),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "logprob length" in str(excinfo.value)


def test_absent_mask_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(calls=(call(1, sampled_mask=()), call(2))),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "sampled mask" in str(excinfo.value)


def test_missing_stop_token_ids_are_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(calls=(call(1, stop_token_ids=()), call(2))),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "stop token ids" in str(excinfo.value)


def test_renderer_profile_mismatch_is_refused() -> None:
    other = replace(PROFILE, config_digest="sha256:other")
    with pytest.raises(RecordError) as excinfo:
        validate_probe(attempt(), expected_profile=other, quiescence_accepted=True)
    assert "renderer profile mismatch" in str(excinfo.value)


def test_unstamped_behavior_fingerprint_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(calls=(call(1, behavior_fingerprint="sha256:someone-else"), call(2))),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "behavior" in str(excinfo.value)


def test_non_monotone_event_cursor_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(event_cursors=(1, 5, 4)),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "monotone" in str(excinfo.value)


def test_two_terminal_results_are_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(terminal_results=("completed", "failed")),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "terminal results" in str(excinfo.value)


def test_reward_not_bound_to_the_rollout_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(reward=reward(rollout_id="ro-2")),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "bound to rollout" in str(excinfo.value)


def test_reward_not_bound_to_the_trace_digest_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(reward=reward(trace_digest="sha256:another-trace")),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "not admissible evidence" in str(excinfo.value)


def test_episode_not_sealed_against_the_trace_digest_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(episode=episode(trace_digest="sha256:stale")),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "sealed" in str(excinfo.value)


def test_accepted_quiescence_without_an_attestation_is_refused() -> None:
    clipped = reward(
        horizon=HorizonEvidence(
            horizon_kind="wall_clock",
            horizon_value=5400.0,
            scored_at_offset_seconds=0.0,
            clipped=True,
            quiescence_attested=False,
        )
    )
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(reward=clipped), expected_profile=PROFILE, quiescence_accepted=True
        )
    assert "quiescence attestation" in str(excinfo.value)
    report = validate_probe(
        attempt(reward=clipped), expected_profile=PROFILE, quiescence_accepted=False
    )
    assert report.quiescence_attested is False


@pytest.mark.parametrize("operation", sorted(REQUIRED_PROBE_OPERATIONS))
def test_every_probe_operation_must_be_exercised(operation: str) -> None:
    partial = frozenset(REQUIRED_PROBE_OPERATIONS - {operation})
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(operations=partial),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert operation in str(excinfo.value)


def test_non_idempotent_resubmit_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(resubmit_rollout_id="ro-2"),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "second logical attempt" in str(excinfo.value)


def test_absent_cancellation_is_refused() -> None:
    with pytest.raises(ProbeError) as excinfo:
        validate_probe(
            attempt(cancelled_rollout_id=" "),
            expected_profile=PROFILE,
            quiescence_accepted=True,
        )
    assert "cancellation" in str(excinfo.value)


def test_a_joint_probe_checks_each_instance_stream_separately() -> None:
    """Prefix consistency belongs to a conversation, not to an attempt.

    A joint episode interleaves instances, so checking calls in submission
    order compares one instance's turn against another's and fails every
    correct joint probe.
    """

    from synth_optimizers.rl.probe import _probe_prefix_streams

    def call(instance: str, name: str, prompt: tuple[int, ...]) -> InferenceCall:
        return InferenceCall(
            call_id=name,
            proxy_request_id="prid",
            rollout_id="probe_1",
            group_id="group_1",
            sample_index=0,
            agent_instance_id=instance,
            behavior_fingerprint="fp",
            policy_revision=0,
            wire_api="chat_completions",
            sampling_transport="message_in_capture_out",
            token_capture_provenance="probe_synthetic",
            prompt_token_ids=prompt,
            generation_token_ids=(7, 8),
            generation_logprobs=(-0.5, -0.25),
            sampled_mask=(1, 1),
            finish_reason="stop_token",
            trainable=False,
        )

    # Interleaved, and each instance's own stream is a strict prefix chain.
    interleaved = [
        call("elf_0", "a1", (1, 2)),
        call("barbarian_0", "b1", (5, 6)),
        call("elf_0", "a2", (1, 2, 7, 8)),
        call("barbarian_0", "b2", (5, 6, 7, 8)),
    ]
    streams = _probe_prefix_streams(interleaved)
    assert sorted(streams) == ["barbarian_0", "elf_0"]
    assert [c.call_id for c in streams["elf_0"]] == ["a1", "a2"]
