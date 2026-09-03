"""Frozen evidence records for the container-first RL plane.

These types are the training record. A container may grow new capabilities, but
field names, required keys, and validity rules here are compatibility-sensitive:
a batch is assembled from these objects and nothing else.

Shapes and names deliberately follow the Tito data plane (``InferenceCallV2``,
``RewardRecordV1``, ``BehaviorFingerprint``) so the two planes reconcile by
mapping rather than by rewrite. No type here names a task, a harness, or an
environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

INFERENCE_CALL_SCHEMA_VERSION = "cispo.inference_call.v2"
TRAINABLE_EPISODE_SCHEMA_VERSION = "cispo.trainable_episode.v1"
REWARD_RECORD_SCHEMA_VERSION = "cispo.reward_record.v1"
RENDERER_PROFILE_SCHEMA_VERSION = "cispo.renderer_profile.v1"

WIRE_APIS = frozenset({"chat_completions", "responses"})
SAMPLING_TRANSPORTS = frozenset({"message_in_capture_out", "tokens_in_tokens_out"})
FINISH_REASONS = frozenset({"stop_token", "length_cap", "container_abort"})
ARTIFACT_ROLES = frozenset({"sampler_weights", "training_state"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# Foreign authorship must be declared, never implied by a zero mask.
AUTHOR_KINDS = frozenset(
    {"policy", "foreign_agent", "opponent", "verifier", "judge", "harness"}
)
TRAINABLE_AUTHOR_KINDS = frozenset({"policy"})

# Tokens and logprobs must come from under the public wire. Anything derived by
# detokenizing then retokenizing wire JSON is not a training record.
TOKEN_CAPTURE_PROVENANCE = frozenset({"engine_meta", "probe_synthetic", "wire_derived"})
TRAINABLE_PROVENANCE = frozenset({"engine_meta"})

# vLLM uses this value both for missing sampled-token evidence and as a
# lower-bound clamp, so receiving it can never prove a real logprob came back.
LOGPROB_SENTINEL = -9999.0

# Mask conventions are fixed for the plane, not negotiated per container.
UNTRAINABLE_TOKEN_CLASSES = (
    "template_structure",
    "tool_observation",
    "environment_step",
    "harness_compaction",
    "foreign_agent_message",
    "opponent_message",
    "verifier_text",
    "judge_text",
)


class RecordError(ValueError):
    """A record was incomplete, malformed, or internally inconsistent."""


class EvidenceError(RecordError):
    """Evidence that cannot be trained on. Never degrade this to zero reward."""


def digest(payload: Any, *, length: int = 64) -> str:
    """Canonical sha256 over a JSON-serializable payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:length]


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{name} is required")
    return value.strip()


def _ints(payload: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecordError(f"{name} must be a sequence of integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True, slots=True)
class RendererProfile:
    """Pinned renderer identity. A version string alone is not an identity."""

    profile_id: str
    package: str
    package_version: str
    config_digest: str
    tokenizer_id: str
    tokenizer_digest: str
    stop_token_ids: tuple[int, ...]
    modalities: tuple[str, ...] = ("text",)
    add_generation_prompt: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise RecordError("renderer profile_id is required")
        if not self.stop_token_ids:
            raise RecordError("renderer profile must declare stop token ids")

    @property
    def fingerprint(self) -> str:
        """Digest of everything that changes what a token sequence means."""

        return digest(
            {
                "schema_version": RENDERER_PROFILE_SCHEMA_VERSION,
                "profile_id": self.profile_id,
                "package": self.package,
                "package_version": self.package_version,
                "config_digest": self.config_digest,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_digest": self.tokenizer_digest,
                "stop_token_ids": list(self.stop_token_ids),
                "modalities": list(self.modalities),
                "add_generation_prompt": self.add_generation_prompt,
            },
            length=32,
        )

    def assert_matches(self, other: "RendererProfile") -> None:
        """Binding profile must equal the training session profile, exactly."""

        if self.fingerprint != other.fingerprint:
            raise RecordError(
                "renderer profile mismatch: "
                f"{self.profile_id}@{self.fingerprint} != {other.profile_id}@{other.fingerprint}"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RendererProfile":
        modalities = payload.get("modalities") or ("text",)
        return cls(
            profile_id=_text(payload, "profile_id"),
            package=_text(payload, "package"),
            package_version=_text(payload, "package_version"),
            config_digest=_text(payload, "config_digest"),
            tokenizer_id=_text(payload, "tokenizer_id"),
            tokenizer_digest=_text(payload, "tokenizer_digest"),
            stop_token_ids=_ints(payload, "stop_token_ids"),
            modalities=tuple(str(item) for item in modalities),
            add_generation_prompt=bool(payload.get("add_generation_prompt", True)),
        )


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None

    @property
    def key(self) -> str:
        return digest(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            length=16,
        )


@dataclass(frozen=True, slots=True)
class BehaviorFingerprint:
    """What the tokens were produced by. Groups may not mix these."""

    renderer_profile: RendererProfile
    model_family: str
    model_id: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    sampling: SamplingProfile = field(default_factory=SamplingProfile)

    def __post_init__(self) -> None:
        if self.wire_api not in WIRE_APIS:
            raise RecordError(f"unknown wire_api {self.wire_api!r}")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise RecordError(f"unknown sampling_transport {self.sampling_transport!r}")
        if self.policy_revision < 0:
            raise RecordError("policy_revision must be non-negative")

    @property
    def value(self) -> str:
        return digest(
            {
                "renderer": self.renderer_profile.fingerprint,
                "model_family": self.model_family,
                "model_id": self.model_id,
                "policy_revision": self.policy_revision,
                "wire_api": self.wire_api,
                "sampling_transport": self.sampling_transport,
                "sampling": self.sampling.key,
            },
            length=32,
        )


@dataclass(frozen=True, slots=True)
class CompactionProvenance:
    """Why a turn's prompt is not a strict prefix of the previous sequence."""

    rule: str
    divergence_index: int
    removed_message_indices: tuple[int, ...] = ()
    authored_by_policy: bool = False

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise RecordError("compaction rule is required")
        if self.divergence_index < 0:
            raise RecordError("divergence_index must be non-negative")


@dataclass(frozen=True, slots=True)
class InferenceCall:
    """One immutable record per proxied model call, before any flattening."""

    call_id: str
    proxy_request_id: str
    rollout_id: str
    group_id: str
    sample_index: int
    behavior_fingerprint: str
    policy_revision: int
    wire_api: str
    sampling_transport: str
    token_capture_provenance: str
    prompt_token_ids: tuple[int, ...]
    generation_token_ids: tuple[int, ...]
    generation_logprobs: tuple[float, ...]
    sampled_mask: tuple[int, ...]
    finish_reason: str
    stop_token_ids: tuple[int, ...] = ()
    content_mask: tuple[int, ...] = ()
    # The behavior fingerprint is a digest, so a call alone cannot say which
    # renderer produced it. Stamp the profile fingerprint too: the note requires
    # the trace to identify the renderer that produced the tokens.
    renderer_profile_fingerprint: str = ""
    trainable: bool = True
    branch_id: str = "root"
    parent_branch_id: str | None = None
    compaction: CompactionProvenance | None = None
    agent_instance_id: str | None = None
    team_id: str | None = None
    role_id: str | None = None
    policy_type_id: str | None = None
    parameter_group_id: str | None = None
    policy_set_revision_id: str | None = None
    effect_tick_start: int | None = None
    effect_tick_end: int | None = None
    wire_request: Mapping[str, Any] = field(default_factory=dict)
    wire_response: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = INFERENCE_CALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.wire_api not in WIRE_APIS:
            raise RecordError(f"unknown wire_api {self.wire_api!r}")
        if self.sampling_transport not in SAMPLING_TRANSPORTS:
            raise RecordError(f"unknown sampling_transport {self.sampling_transport!r}")
        if self.token_capture_provenance not in TOKEN_CAPTURE_PROVENANCE:
            raise RecordError(f"unknown provenance {self.token_capture_provenance!r}")
        if self.finish_reason not in FINISH_REASONS:
            raise RecordError(f"unknown finish_reason {self.finish_reason!r}")

    def validate_for_training(self) -> None:
        """Every reason a call may not enter a batch. Raises, never degrades."""

        if not self.trainable:
            raise EvidenceError(f"call {self.call_id} is marked non-trainable")
        if self.token_capture_provenance not in TRAINABLE_PROVENANCE:
            raise EvidenceError(
                f"call {self.call_id} captured via {self.token_capture_provenance}; "
                "training requires engine-level token capture"
            )
        if not self.prompt_token_ids:
            raise EvidenceError(f"call {self.call_id} has no prompt tokens")
        if not self.generation_token_ids:
            raise EvidenceError(f"call {self.call_id} has no generated tokens")
        generated = len(self.generation_token_ids)
        if len(self.generation_logprobs) != generated:
            raise EvidenceError(
                f"call {self.call_id} logprob length {len(self.generation_logprobs)} "
                f"!= generated token count {generated}"
            )
        if self.sampled_mask and len(self.sampled_mask) != generated:
            raise EvidenceError(f"call {self.call_id} sampled mask length mismatch")
        if self.content_mask and len(self.content_mask) != generated:
            raise EvidenceError(f"call {self.call_id} content mask length mismatch")
        for index, value in enumerate(self.generation_logprobs):
            if math.isnan(value) or math.isinf(value):
                raise EvidenceError(f"call {self.call_id} logprob {index} is not finite")
            if value == LOGPROB_SENTINEL:
                raise EvidenceError(
                    f"call {self.call_id} logprob {index} is the provider sentinel "
                    f"{LOGPROB_SENTINEL}; presence of the sentinel cannot prove a real logprob"
                )
        if all(value == 0.0 for value in self.generation_logprobs):
            raise EvidenceError(f"call {self.call_id} logprobs are identically zero")

    @property
    def full_sequence(self) -> tuple[int, ...]:
        return tuple(self.prompt_token_ids) + tuple(self.generation_token_ids)

    @property
    def loss_mask(self) -> tuple[int, ...]:
        prompt = (0,) * len(self.prompt_token_ids)
        if self.sampled_mask:
            return prompt + tuple(int(bool(flag)) for flag in self.sampled_mask)
        return prompt + (1,) * len(self.generation_token_ids)


def assert_strict_prefix(previous: InferenceCall, following: InferenceCall) -> None:
    """Two calls stitch only on a byte-for-byte token prefix.

    Anything else forks a branch and seals the prior segment. Never retokenize
    new text onto old ids; an unexplained divergence is an evidence failure.
    """

    sequence = previous.full_sequence
    prompt = tuple(following.prompt_token_ids)
    if prompt[: len(sequence)] == sequence:
        if following.branch_id != previous.branch_id:
            raise EvidenceError(
                f"call {following.call_id} is a strict prefix continuation but changed branch"
            )
        return
    if following.compaction is None:
        content_divergence = next(
            (i for i, (a, b) in enumerate(zip(sequence, prompt, strict=False)) if a != b),
            None,
        )
        if content_divergence is None:
            raise EvidenceError(
                f"call {following.call_id} truncates {previous.call_id}: its prompt agrees for "
                f"{len(prompt)} tokens but the previous sequence is {len(sequence)} long, "
                "with no branch record and no declared compaction"
            )
        raise EvidenceError(
            f"call {following.call_id} diverges from {previous.call_id} at token "
            f"{content_divergence} with no branch record and no declared compaction"
        )
    if following.parent_branch_id != previous.branch_id:
        raise EvidenceError(
            f"call {following.call_id} declares compaction but does not fork from "
            f"branch {previous.branch_id!r}"
        )
    if following.branch_id == previous.branch_id:
        raise EvidenceError(
            f"call {following.call_id} declares compaction and must open a new branch"
        )


@dataclass(frozen=True, slots=True)
class TrainableSegment:
    """One contiguous trainer sequence with its mask and behavior logprobs."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    branch_id: str = "root"
    parameter_group_id: str | None = None
    agent_instance_id: str | None = None
    call_ids: tuple[str, ...] = ()
    author_kind: str = "policy"
    role_id: str | None = None
    policy_type_id: str | None = None
    team_id: str | None = None
    policy_revision: int | None = None
    policy_set_revision_id: str | None = None
    effect_tick_start: int | None = None
    effect_tick_end: int | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise RecordError("segment has no tokens")
        if len(self.loss_mask) != len(self.token_ids):
            raise RecordError("segment loss mask length mismatch")
        if len(self.behavior_logprobs) != len(self.token_ids):
            raise RecordError("segment behavior logprob length mismatch")
        if self.author_kind not in AUTHOR_KINDS:
            raise RecordError(f"unknown author_kind {self.author_kind!r}")
        if self.author_kind not in TRAINABLE_AUTHOR_KINDS and self.trainable_tokens:
            raise RecordError(
                f"segment authored by {self.author_kind!r} carries trainable tokens; "
                "foreign authorship is never trainable"
            )
        if (
            self.effect_tick_start is not None
            and self.effect_tick_end is not None
            and self.effect_tick_end < self.effect_tick_start
        ):
            raise RecordError("segment effect interval ends before it starts")

    @property
    def trainable_tokens(self) -> int:
        return sum(1 for flag in self.loss_mask if flag)

    @property
    def trainable(self) -> bool:
        return self.author_kind in TRAINABLE_AUTHOR_KINDS and bool(self.trainable_tokens)


@dataclass(frozen=True, slots=True)
class TrainableEpisode:
    """The common training view of one completed attempt."""

    rollout_id: str
    task_id: str
    seed: int
    policy_revision: int
    behavior_fingerprint: str
    segments: tuple[TrainableSegment, ...]
    terminal_status: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    agent_instance_id: str | None = None
    team_id: str | None = None
    policy_set_revision_id: str | None = None
    # Branch fan-out weighting needs to know which segments share one
    # environment attempt; a branch is not an independent episode.
    root_rollout_id: str | None = None
    trace_digest: str = ""
    probe: bool = False
    schema_version: str = TRAINABLE_EPISODE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.probe:
            raise EvidenceError(
                f"episode {self.rollout_id} is probe-derived and may not enter a group or batch"
            )
        if not self.segments:
            raise EvidenceError(f"episode {self.rollout_id} has no trainable segments")
        if not self.trace_digest:
            raise EvidenceError(f"episode {self.rollout_id} has no sealed trace digest")
        if not any(segment.trainable_tokens for segment in self.segments):
            raise EvidenceError(f"episode {self.rollout_id} has no trainable tokens")

    @property
    def parameter_groups(self) -> tuple[str, ...]:
        seen: list[str] = []
        for segment in self.segments:
            group = segment.parameter_group_id
            if group is not None and group not in seen:
                seen.append(group)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class RewardChannel:
    """One team's measure. Absolute and rank are both recorded."""

    channel_id: str
    team_id: str | None
    measure: float
    rank: int | None = None

    def __post_init__(self) -> None:
        if math.isnan(self.measure) or math.isinf(self.measure):
            raise RecordError(f"reward channel {self.channel_id} measure is not finite")


@dataclass(frozen=True, slots=True)
class HorizonEvidence:
    """When the reward was read, and whether the environment was still moving."""

    horizon_kind: str
    horizon_value: float
    scored_at_offset_seconds: float
    clipped: bool
    quiescence_attested: bool
    settlement_window_seconds: float = 0.0
    credited_settlement_seconds: float = 0.0

    def validate(self) -> None:
        if not self.quiescence_attested and not self.clipped:
            raise EvidenceError(
                "reward has neither a quiescence attestation nor a horizon-clipped snapshot"
            )
        if self.credited_settlement_seconds > self.settlement_window_seconds:
            raise EvidenceError(
                f"reward credited {self.credited_settlement_seconds}s of settlement beyond "
                f"its declared {self.settlement_window_seconds}s window"
            )
        if self.credited_settlement_seconds < 0 or self.scored_at_offset_seconds < 0:
            raise EvidenceError("settlement and scored-read offsets must be non-negative")


@dataclass(frozen=True, slots=True)
class RewardRecord:
    """Container-authoritative reward, bound to the rollout and trace digest."""

    reward_id: str
    rollout_id: str
    trace_digest: str
    channels: tuple[RewardChannel, ...]
    optimized_channel: str
    terminal_status: str
    evaluation_plan_id: str
    horizon: HorizonEvidence | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = REWARD_RECORD_SCHEMA_VERSION

    def validate(self, *, episode_trace_digest: str | None = None) -> None:
        if not self.rollout_id.strip():
            raise EvidenceError(f"reward {self.reward_id} names no rollout")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise EvidenceError(
                f"reward {self.reward_id} claims non-terminal status {self.terminal_status!r}"
            )
        if not self.channels:
            raise EvidenceError(f"reward {self.reward_id} carries no channel; absent is not zero")
        if not self.trace_digest:
            raise EvidenceError(f"reward {self.reward_id} is not bound to a trace digest")
        if episode_trace_digest is not None and episode_trace_digest != self.trace_digest:
            raise EvidenceError(
                f"reward {self.reward_id} trace digest does not match its episode"
            )
        if self.optimized_channel not in {channel.channel_id for channel in self.channels}:
            raise EvidenceError(
                f"reward {self.reward_id} optimizes channel {self.optimized_channel!r} "
                "which it does not carry"
            )
        if self.horizon is not None:
            self.horizon.validate()

    def value(self, channel_id: str | None = None) -> float:
        wanted = channel_id or self.optimized_channel
        for channel in self.channels:
            if channel.channel_id == wanted:
                return channel.measure
        raise RecordError(f"reward {self.reward_id} has no channel {wanted!r}")

    def channel_for(self, team_id: str) -> RewardChannel:
        """The one channel belonging to a team. Ambiguity is an error."""

        matches = [channel for channel in self.channels if channel.team_id == team_id]
        if not matches:
            raise RecordError(f"reward {self.reward_id} has no channel for team {team_id!r}")
        if len(matches) > 1:
            raise RecordError(
                f"reward {self.reward_id} carries {len(matches)} channels for team "
                f"{team_id!r}; a team's measure must be unambiguous"
            )
        return matches[0]

    def value_for_team(self, team_id: str) -> float:
        return self.channel_for(team_id).measure
