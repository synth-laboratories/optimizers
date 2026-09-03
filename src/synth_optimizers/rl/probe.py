"""The probe episode: where a container's claims become evidence.

A probe attempt walks the whole path — submit, state, events, renewal, trace,
reward, finalize, terminate, an idempotent resubmit, and one cancellation — at
zero provider cost, because the container returns deterministic canned
generations. This module validates shape, not quality.

Probe evidence is never trainable. A container whose probe evidence is
indistinguishable from real evidence fails conformance here, loudly, rather
than one training step later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.rl_records import (
    TRAINABLE_PROVENANCE,
    BehaviorFingerprint,
    EvidenceError,
    InferenceCall,
    RecordError,
    RendererProfile,
    RewardRecord,
    TrainableEpisode,
    assert_strict_prefix,
    digest,
)

PROBE_PROVENANCE = "probe_synthetic"
PROBE_REPORT_SCHEMA_VERSION = "cispo.probe_report.v1"

# The operations one probe attempt must exercise before real attempts are
# admitted. Each maps to a declared route the executor will depend on.
REQUIRED_PROBE_OPERATIONS: frozenset[str] = frozenset(
    {
        "submit",
        "state",
        "events",
        "renew",
        "trace",
        "reward",
        "finalize",
        "terminate",
        "idempotent_resubmit",
        "cancellation",
    }
)


class ProbeError(RecordError):
    """A probe attempt did not exercise or shape the evidence path correctly."""


class ProbeNotDistinguishable(ProbeError):
    """Probe evidence could be mistaken for real evidence. Conformance failure."""


@dataclass(frozen=True, slots=True)
class ProbeAttempt:
    """Everything one probe attempt produced, as the executor received it."""

    rollout_id: str
    behavior: BehaviorFingerprint
    calls: tuple[InferenceCall, ...]
    episode: TrainableEpisode
    reward: RewardRecord
    event_cursors: tuple[int, ...]
    terminal_results: tuple[str, ...]
    operations: frozenset[str]
    resubmit_rollout_id: str
    cancelled_rollout_id: str
    trace_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rollout_id.strip():
            raise ProbeError("probe attempt has no rollout id")


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """What the probe proved, for the run receipt. Never a training record."""

    rollout_id: str
    calls_checked: int
    segments_checked: int
    operations: tuple[str, ...]
    trainable: bool
    renderer_fingerprint: str
    trace_digest: str
    reward_id: str
    quiescence_attested: bool
    schema_version: str = PROBE_REPORT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "calls_checked": self.calls_checked,
            "segments_checked": self.segments_checked,
            "operations": list(self.operations),
            "trainable": self.trainable,
            "renderer_fingerprint": self.renderer_fingerprint,
            "trace_digest": self.trace_digest,
            "reward_id": self.reward_id,
            "quiescence_attested": self.quiescence_attested,
            "evidence_digest": self.evidence_digest,
        }

    @property
    def evidence_digest(self) -> str:
        return "sha256:" + digest(
            {
                "rollout_id": self.rollout_id,
                "trace_digest": self.trace_digest,
                "reward_id": self.reward_id,
                "renderer_fingerprint": self.renderer_fingerprint,
                "probe": True,
            }
        )


def assert_probe_not_trainable(attempt: ProbeAttempt) -> None:
    """Probe evidence must be refused for training, by its own record fields."""

    for call in attempt.calls:
        if call.token_capture_provenance != PROBE_PROVENANCE:
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} declares provenance "
                f"{call.token_capture_provenance!r}; a probe must declare "
                f"{PROBE_PROVENANCE!r} so it can never be mistaken for real evidence"
            )
        if call.token_capture_provenance in TRAINABLE_PROVENANCE:
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} carries trainable provenance"
            )
        if call.trainable:
            raise ProbeNotDistinguishable(
                f"probe call {call.call_id} is marked trainable; probe episodes may "
                "never enter a group or a batch"
            )
        try:
            call.validate_for_training()
        except EvidenceError:
            continue
        raise ProbeNotDistinguishable(
            f"probe call {call.call_id} passes the training gate: probe evidence is "
            "indistinguishable from real evidence"
        )


def _check_call_shape(call: InferenceCall) -> None:
    for name, value in (
        ("call_id", call.call_id),
        ("proxy_request_id", call.proxy_request_id),
        ("rollout_id", call.rollout_id),
        ("group_id", call.group_id),
        ("finish_reason", call.finish_reason),
    ):
        if not str(value).strip():
            raise ProbeError(f"probe call is missing {name}")
    if not call.prompt_token_ids:
        raise ProbeError(f"probe call {call.call_id} has no prompt tokens")
    if not call.generation_token_ids:
        raise ProbeError(f"probe call {call.call_id} has no generated tokens")
    generated = len(call.generation_token_ids)
    if len(call.generation_logprobs) != generated:
        raise ProbeError(
            f"probe call {call.call_id} logprob length {len(call.generation_logprobs)} "
            f"!= generated token count {generated}"
        )
    if not call.sampled_mask:
        raise ProbeError(f"probe call {call.call_id} carries no sampled mask")
    if len(call.sampled_mask) != generated:
        raise ProbeError(f"probe call {call.call_id} sampled mask length mismatch")
    if not call.stop_token_ids:
        raise ProbeError(
            f"probe call {call.call_id} does not record the renderer's stop token ids"
        )


def _check_operations(attempt: ProbeAttempt) -> tuple[str, ...]:
    missing = tuple(sorted(REQUIRED_PROBE_OPERATIONS - set(attempt.operations)))
    if missing:
        raise ProbeError(f"probe attempt did not exercise {missing}")
    unknown = tuple(sorted(set(attempt.operations) - REQUIRED_PROBE_OPERATIONS))
    if unknown:
        raise ProbeError(f"probe attempt reports unknown operations {unknown}")
    if attempt.resubmit_rollout_id != attempt.rollout_id:
        raise ProbeError(
            "idempotent resubmit produced a second logical attempt: "
            f"{attempt.resubmit_rollout_id!r} != {attempt.rollout_id!r}"
        )
    if not attempt.cancelled_rollout_id.strip():
        raise ProbeError("probe attempt records no cancellation")
    return tuple(sorted(attempt.operations))


def _check_cursors(cursors: Sequence[int]) -> None:
    if not cursors:
        raise ProbeError("probe attempt returned no event cursor")
    for previous, following in zip(cursors, cursors[1:], strict=False):
        if following <= previous:
            raise ProbeError(
                f"probe event cursor is not monotone: {previous} then {following}"
            )


def _probe_prefix_streams(
    calls: Sequence[InferenceCall],
) -> dict[str, list[InferenceCall]]:
    """One conversation per agent instance, in call order."""

    streams: dict[str, list[InferenceCall]] = {}
    for call in calls:
        streams.setdefault(call.agent_instance_id or "", []).append(call)
    return streams


def validate_probe(
    attempt: ProbeAttempt,
    *,
    expected_profile: RendererProfile,
    quiescence_accepted: bool,
) -> ProbeReport:
    """Validate the probe's shape. Raises a typed error, never returns a bool."""

    assert_probe_not_trainable(attempt)
    if len(attempt.calls) < 2:
        raise ProbeError(
            "a probe must exercise at least two turns so prefix consistency is checkable"
        )
    for call in attempt.calls:
        _check_call_shape(call)
        if call.rollout_id != attempt.rollout_id:
            raise ProbeError(
                f"probe call {call.call_id} names rollout {call.rollout_id!r}, "
                f"not {attempt.rollout_id!r}"
            )
        if call.behavior_fingerprint != attempt.behavior.value:
            raise ProbeError(
                f"probe call {call.call_id} is not stamped with the attempt's behavior "
                "fingerprint"
            )
        if call.policy_revision != attempt.behavior.policy_revision:
            raise ProbeError(
                f"probe call {call.call_id} records policy revision "
                f"{call.policy_revision}, not {attempt.behavior.policy_revision}"
            )
        if tuple(call.stop_token_ids) != tuple(expected_profile.stop_token_ids):
            raise ProbeError(
                f"probe call {call.call_id} declares stop token ids "
                f"{tuple(call.stop_token_ids)}, not the renderer's "
                f"{tuple(expected_profile.stop_token_ids)}"
            )
    attempt.behavior.renderer_profile.assert_matches(expected_profile)
    # Prefix consistency is a property of one conversation, not of an attempt.
    # A joint episode interleaves several instances' calls, so checking the
    # attempt's calls in submission order would compare one instance's turn
    # against another's and fail every correct joint probe.
    for stream in _probe_prefix_streams(attempt.calls).values():
        for previous, following in zip(stream, stream[1:], strict=False):
            assert_strict_prefix(previous, following)
    _check_cursors(attempt.event_cursors)
    if len(attempt.terminal_results) != 1:
        raise ProbeError(
            f"probe attempt produced {len(attempt.terminal_results)} terminal results; "
            "exactly one is allowed"
        )
    _check_episode(attempt)
    _check_reward(attempt, quiescence_accepted=quiescence_accepted)
    operations = _check_operations(attempt)
    return ProbeReport(
        rollout_id=attempt.rollout_id,
        calls_checked=len(attempt.calls),
        segments_checked=len(attempt.episode.segments),
        operations=operations,
        trainable=False,
        renderer_fingerprint=expected_profile.fingerprint,
        trace_digest=attempt.trace_digest,
        reward_id=attempt.reward.reward_id,
        quiescence_attested=bool(
            attempt.reward.horizon is not None and attempt.reward.horizon.quiescence_attested
        ),
    )


def _check_episode(attempt: ProbeAttempt) -> None:
    episode = attempt.episode
    if episode.rollout_id != attempt.rollout_id:
        raise ProbeError(
            f"probe episode names rollout {episode.rollout_id!r}, not {attempt.rollout_id!r}"
        )
    if not episode.segments:
        raise ProbeError("probe episode carries no segment to check")
    if not episode.trace_digest or episode.trace_digest != attempt.trace_digest:
        raise ProbeError("probe episode is not sealed against the attempt's trace digest")
    if episode.behavior_fingerprint != attempt.behavior.value:
        raise ProbeError("probe episode is not stamped with the attempt's behavior fingerprint")
    for index, segment in enumerate(episode.segments):
        if len(segment.loss_mask) != len(segment.token_ids):
            raise ProbeError(f"probe segment {index} loss mask length mismatch")
        if len(segment.behavior_logprobs) != len(segment.token_ids):
            raise ProbeError(f"probe segment {index} behavior logprob length mismatch")


def _check_reward(attempt: ProbeAttempt, *, quiescence_accepted: bool) -> None:
    reward = attempt.reward
    if reward.rollout_id != attempt.rollout_id:
        raise ProbeError(
            f"probe reward is bound to rollout {reward.rollout_id!r}, "
            f"not {attempt.rollout_id!r}"
        )
    try:
        reward.validate(episode_trace_digest=attempt.trace_digest)
    except EvidenceError as exc:
        raise ProbeError(f"probe reward is not admissible evidence: {exc}") from exc
    if quiescence_accepted:
        if reward.horizon is None or not reward.horizon.quiescence_attested:
            raise ProbeError(
                "quiescence was an accepted clause but the probe reward carries no "
                "quiescence attestation"
            )
