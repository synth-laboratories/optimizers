"""Shared training-provider interfaces.

SFT, CISPO, FBC, and GoEx consume this adapter. Do not add algorithm-specific
Tinker clients beside it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


CAPABILITY_SFT_TRAIN = "sft.train"
CAPABILITY_CHECKPOINT_SAMPLE = "checkpoint.sample"
CAPABILITY_ROLLOUT_GROUPED = "rollout.grouped"
CAPABILITY_TRAJECTORY_LOGPROBS = "trajectory.logprobs"
CAPABILITY_IMPORTANCE_WEIGHTS = "training.importance_weights"
CAPABILITY_CISPO_SLIME_V1 = "cispo.slime.v1"

SFT_REQUIRED_CAPABILITIES = frozenset({CAPABILITY_SFT_TRAIN, CAPABILITY_CHECKPOINT_SAMPLE})
CISPO_REQUIRED_CAPABILITIES = frozenset(
    {
        CAPABILITY_SFT_TRAIN,
        CAPABILITY_CHECKPOINT_SAMPLE,
        CAPABILITY_ROLLOUT_GROUPED,
        CAPABILITY_TRAJECTORY_LOGPROBS,
        CAPABILITY_IMPORTANCE_WEIGHTS,
        CAPABILITY_CISPO_SLIME_V1,
    }
)


class ProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = request_id


class UnsupportedCapability(ProviderError):
    def __init__(self, missing: Sequence[str]) -> None:
        names = ", ".join(sorted(missing))
        super().__init__("unsupported", f"missing required capabilities: {names}")
        self.missing = tuple(sorted(missing))


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    model_id: str
    capabilities: frozenset[str]
    validated: Mapping[str, bool] = field(default_factory=dict)
    maximums: Mapping[str, int] = field(default_factory=dict)
    spend_free: bool = True

    def supports(self, name: str) -> bool:
        return name in self.capabilities

    def require(self, required: frozenset[str]) -> None:
        missing = required - self.capabilities
        if missing:
            raise UnsupportedCapability(sorted(missing))


@dataclass(frozen=True, slots=True)
class ProviderSession:
    provider: str
    session_id: str
    model_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class SampleRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class SampleResult:
    request_id: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str
    finish_reason: str
    usage: "ProviderUsage"


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    request_id: str
    token_ids: tuple[tuple[int, ...], ...]
    response_masks: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class ForwardResult:
    request_id: str
    logprobs: tuple[tuple[float, ...], ...]
    usage: "ProviderUsage"


@dataclass(frozen=True, slots=True)
class TrainingStepRequest:
    request_id: str
    loss_name: str
    data: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    request_id: str
    step: int
    metrics: Mapping[str, float]
    usage: "ProviderUsage"


@dataclass(frozen=True, slots=True)
class ProviderCheckpoint:
    checkpoint_id: str
    provider_reference: str
    step: int
    digest: str
    kind: str
    resume_token: str | None = None
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    training_tokens: int = 0
    cost_usd: float | None = None
    cost_missing: bool = True
    counters: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageReceipt:
    request_id: str
    provider: str
    usage: ProviderUsage
    algorithm_id: str
    implementation_version: str


class TrainingProvider(Protocol):
    def discover_capabilities(self, model_id: str) -> ProviderCapabilities: ...
    def resolve_model(self, model_id: str) -> str: ...
    def create_session(
        self, model_id: str, *, rank: int, seed: int, request_id: str
    ) -> ProviderSession: ...
    def restore_session(
        self, checkpoint: ProviderCheckpoint, *, request_id: str
    ) -> ProviderSession: ...
    def sample(self, session: ProviderSession, request: SampleRequest) -> SampleResult: ...
    def forward(self, session: ProviderSession, request: ForwardRequest) -> ForwardResult: ...
    def train_step(
        self, session: ProviderSession, request: TrainingStepRequest
    ) -> TrainingStepResult: ...
    def save_checkpoint(
        self, session: ProviderSession, *, step: int, kind: str, request_id: str
    ) -> ProviderCheckpoint: ...
    def sample_checkpoint(
        self, checkpoint: ProviderCheckpoint, request: SampleRequest
    ) -> SampleResult: ...
    def cancel(self, session: ProviderSession) -> None: ...
    def classify_error(self, error: BaseException) -> ProviderError: ...


class CheckpointStore(Protocol):
    def put(self, checkpoint: ProviderCheckpoint, payload: Mapping[str, Any]) -> None: ...
    def get(self, checkpoint_id: str) -> Mapping[str, Any]: ...


class DatasetSource(Protocol):
    def load(self) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...
    def manifest(self) -> Mapping[str, Any]: ...


class RolloutProvider(Protocol):
    def rollout_group(
        self,
        session: ProviderSession,
        prompts: Sequence[Mapping[str, Any]],
        *,
        group_size: int,
        seed: int,
    ) -> Sequence[Mapping[str, Any]]: ...


class TrainingEventSink(Protocol):
    def append(self, kind: str, payload: Mapping[str, Any], *, phase: str) -> Mapping[str, Any]: ...


class UsageReceiptSink(Protocol):
    def record(self, receipt: UsageReceipt) -> None: ...
