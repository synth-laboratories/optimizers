"""Frozen public schemas for standalone SFT and ``cispo.slime.v1``.

These types are the control-plane contract. Execution may grow, but field
names, required keys, and identity rules here are compatibility-sensitive.
Do not mention private executor hosts, service URLs, or historical beta names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


SFT_CONFIG_SCHEMA_VERSION = "sft.request.v1"
CISPO_CONFIG_SCHEMA_VERSION = "cispo.request.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "dataset.manifest.v1"
ROLLOUT_GROUP_SCHEMA_VERSION = "cispo.rollout_group.v1"
CHECKPOINT_SCHEMA_VERSION = "training.checkpoint.v1"
METRIC_EVENT_SCHEMA_VERSION = "training.metric.v1"
TERMINAL_OUTCOME_SCHEMA_VERSION = "training.terminal.v1"
USAGE_RECEIPT_SCHEMA_VERSION = "training.usage_receipt.v1"

SFT_ALGORITHM_ID = "sft"
CISPO_ALGORITHM_ID = "cispo"
CISPO_IMPLEMENTATION = "slime-reference"
CISPO_IMPLEMENTATION_VERSION = "cispo.slime.v1"
SFT_IMPLEMENTATION = "tinker-sft"
SFT_IMPLEMENTATION_VERSION = "sft.tinker.v1"
DEFAULT_SFT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PROVIDER = "tinker"

LIFECYCLE_STATES = (
    "prepared",
    "running",
    "evaluating",
    "materializing",
    "completed",
    "failed",
    "cancelled",
)
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


class SchemaError(ValueError):
    """A public training schema was incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SftRequest:
    schema_version: str
    algorithm_id: str
    implementation: str
    implementation_version: str
    provider: str
    model_id: str
    dataset: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    seed: int
    repeat_index: int = 0
    renderer_version: str = "chat.v1"
    runner_version: str = "synth-optimizers"


@dataclass(frozen=True, slots=True)
class CispoRequest:
    schema_version: str
    algorithm_id: str
    implementation: str
    implementation_version: str
    provider: str
    model_id: str
    dataset: Mapping[str, Any]
    training: Mapping[str, Any]
    reward: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    seed: int
    repeat_index: int = 0
    renderer_version: str = "chat.v1"
    runner_version: str = "synth-optimizers"
    mode: str = "canonical"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    digest: str
    split_digests: Mapping[str, str]
    example_counts: Mapping[str, int]
    label_taxonomy_digest: str
    renderer_version: str


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    schema_version: str
    group_id: str
    iteration: int
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    zero_advantage: bool
    prompt_digest: str


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    schema_version: str
    checkpoint_id: str
    provider: str
    provider_reference: str
    step: int
    digest: str
    kind: str
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class MetricEvent:
    schema_version: str
    name: str
    value: float
    step: int
    split: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    schema_version: str
    state: str
    reason: str | None
    selected_checkpoint_id: str | None
    heldout_metric: float | None
    artifact_digests: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageReceipt:
    schema_version: str
    provider: str
    request_id: str
    input_tokens: int
    output_tokens: int
    training_tokens: int
    cost_usd: float | None
    cost_missing: bool
    algorithm_id: str
    implementation_version: str


def validate_sft_request(value: Mapping[str, Any]) -> SftRequest:
    data = _object(value, "SFT request")
    _expect(data.get("schema_version"), SFT_CONFIG_SCHEMA_VERSION, "schema_version")
    algorithm = _text(data.get("algorithm_id"), "algorithm_id")
    if algorithm != SFT_ALGORITHM_ID:
        raise SchemaError("SFT request algorithm_id must be 'sft'")
    implementation = _text(data.get("implementation", SFT_IMPLEMENTATION), "implementation")
    version = _text(
        data.get("implementation_version", SFT_IMPLEMENTATION_VERSION),
        "implementation_version",
    )
    if version != SFT_IMPLEMENTATION_VERSION:
        raise SchemaError("SFT implementation_version must be sft.tinker.v1")
    if _text(data.get("provider", DEFAULT_PROVIDER), "provider") != DEFAULT_PROVIDER:
        raise SchemaError("standalone SFT requires provider=tinker")
    return SftRequest(
        schema_version=SFT_CONFIG_SCHEMA_VERSION,
        algorithm_id=algorithm,
        implementation=implementation,
        implementation_version=version,
        provider=DEFAULT_PROVIDER,
        model_id=_text(data.get("model_id", DEFAULT_SFT_MODEL), "model_id"),
        dataset=_object(data.get("dataset"), "dataset"),
        training=_object(data.get("training"), "training"),
        evaluation=_object(data.get("evaluation", {}), "evaluation"),
        seed=_int(data.get("seed", 0), "seed", minimum=0),
        repeat_index=_int(data.get("repeat_index", 0), "repeat_index", minimum=0),
        renderer_version=_text(data.get("renderer_version", "chat.v1"), "renderer_version"),
        runner_version=_text(data.get("runner_version", "synth-optimizers"), "runner_version"),
    )


def validate_cispo_request(value: Mapping[str, Any]) -> CispoRequest:
    data = _object(value, "CISPO request")
    _expect(data.get("schema_version"), CISPO_CONFIG_SCHEMA_VERSION, "schema_version")
    algorithm = _text(data.get("algorithm_id"), "algorithm_id")
    if algorithm != CISPO_ALGORITHM_ID:
        raise SchemaError("CISPO request algorithm_id must be 'cispo'")
    implementation = _text(data.get("implementation"), "implementation")
    version = _text(data.get("implementation_version"), "implementation_version")
    if implementation != CISPO_IMPLEMENTATION or version != CISPO_IMPLEMENTATION_VERSION:
        raise SchemaError("CISPO may only claim slime-reference / cispo.slime.v1")
    if _text(data.get("provider", DEFAULT_PROVIDER), "provider") != DEFAULT_PROVIDER:
        raise SchemaError("standalone CISPO requires provider=tinker")
    mode = _text(data.get("mode", "canonical"), "mode")
    if mode not in {"canonical", "learning_signal"}:
        raise SchemaError("CISPO mode must be canonical or learning_signal")
    return CispoRequest(
        schema_version=CISPO_CONFIG_SCHEMA_VERSION,
        algorithm_id=algorithm,
        implementation=implementation,
        implementation_version=version,
        provider=DEFAULT_PROVIDER,
        model_id=_text(data.get("model_id", DEFAULT_SFT_MODEL), "model_id"),
        dataset=_object(data.get("dataset"), "dataset"),
        training=_object(data.get("training"), "training"),
        reward=_object(data.get("reward"), "reward"),
        evaluation=_object(data.get("evaluation", {}), "evaluation"),
        seed=_int(data.get("seed", 0), "seed", minimum=0),
        repeat_index=_int(data.get("repeat_index", 0), "repeat_index", minimum=0),
        renderer_version=_text(data.get("renderer_version", "chat.v1"), "renderer_version"),
        runner_version=_text(data.get("runner_version", "synth-optimizers"), "runner_version"),
        mode=mode,
    )


def validate_dataset_manifest(value: Mapping[str, Any]) -> DatasetManifest:
    data = _object(value, "dataset manifest")
    _expect(data.get("schema_version"), DATASET_MANIFEST_SCHEMA_VERSION, "schema_version")
    splits = _string_map(data.get("split_digests"), "split_digests")
    counts = {
        key: _int(raw, f"example_counts.{key}", minimum=0)
        for key, raw in _object(data.get("example_counts"), "example_counts").items()
    }
    if set(splits) != set(counts):
        raise SchemaError("dataset split identities and counts must cover the same splits")
    if not splits:
        raise SchemaError("dataset manifest requires at least one split")
    return DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        digest=_digest(data.get("digest"), "digest"),
        split_digests=splits,
        example_counts=counts,
        label_taxonomy_digest=_digest(data.get("label_taxonomy_digest"), "label_taxonomy_digest"),
        renderer_version=_text(data.get("renderer_version"), "renderer_version"),
    )


def validate_rollout_group(value: Mapping[str, Any]) -> RolloutGroup:
    data = _object(value, "rollout group")
    _expect(data.get("schema_version"), ROLLOUT_GROUP_SCHEMA_VERSION, "schema_version")
    rewards = _floats(data.get("rewards"), "rewards")
    advantages = _floats(data.get("advantages"), "advantages")
    if len(rewards) != len(advantages) or len(rewards) < 2:
        raise SchemaError("rollout groups need matching rewards and advantages of size >= 2")
    zero = data.get("zero_advantage")
    if not isinstance(zero, bool):
        raise SchemaError("zero_advantage must be a boolean")
    return RolloutGroup(
        schema_version=ROLLOUT_GROUP_SCHEMA_VERSION,
        group_id=_text(data.get("group_id"), "group_id"),
        iteration=_int(data.get("iteration"), "iteration", minimum=0),
        rewards=rewards,
        advantages=advantages,
        zero_advantage=zero,
        prompt_digest=_digest(data.get("prompt_digest"), "prompt_digest"),
    )


def validate_checkpoint(value: Mapping[str, Any]) -> CheckpointRecord:
    data = _object(value, "checkpoint")
    _expect(data.get("schema_version"), CHECKPOINT_SCHEMA_VERSION, "schema_version")
    kind = _text(data.get("kind"), "kind")
    if kind not in {"training", "inference"}:
        raise SchemaError("checkpoint kind must be training or inference")
    eligible = data.get("eligible", True)
    if not isinstance(eligible, bool):
        raise SchemaError("eligible must be a boolean")
    return CheckpointRecord(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id=_text(data.get("checkpoint_id"), "checkpoint_id"),
        provider=_text(data.get("provider", DEFAULT_PROVIDER), "provider"),
        provider_reference=_text(data.get("provider_reference"), "provider_reference"),
        step=_int(data.get("step"), "step", minimum=0),
        digest=_digest(data.get("digest"), "digest"),
        kind=kind,
        eligible=eligible,
    )


def validate_metric_event(value: Mapping[str, Any]) -> MetricEvent:
    data = _object(value, "metric event")
    _expect(data.get("schema_version"), METRIC_EVENT_SCHEMA_VERSION, "schema_version")
    return MetricEvent(
        schema_version=METRIC_EVENT_SCHEMA_VERSION,
        name=_text(data.get("name"), "name"),
        value=_finite(data.get("value"), "value"),
        step=_int(data.get("step"), "step", minimum=0),
        split=_optional_text(data.get("split")),
    )


def validate_terminal_outcome(value: Mapping[str, Any]) -> TerminalOutcome:
    data = _object(value, "terminal outcome")
    _expect(data.get("schema_version"), TERMINAL_OUTCOME_SCHEMA_VERSION, "schema_version")
    state = _text(data.get("state"), "state")
    if state not in TERMINAL_STATES:
        raise SchemaError("terminal state must be completed, failed, or cancelled")
    digests = {
        key: _digest(raw, f"artifact_digests.{key}")
        for key, raw in _object(data.get("artifact_digests", {}), "artifact_digests").items()
    }
    return TerminalOutcome(
        schema_version=TERMINAL_OUTCOME_SCHEMA_VERSION,
        state=state,
        reason=_optional_text(data.get("reason")),
        selected_checkpoint_id=_optional_text(data.get("selected_checkpoint_id")),
        heldout_metric=(
            None if data.get("heldout_metric") is None else _finite(data.get("heldout_metric"), "heldout_metric")
        ),
        artifact_digests=digests,
    )


def validate_usage_receipt(value: Mapping[str, Any]) -> UsageReceipt:
    data = _object(value, "usage receipt")
    _expect(data.get("schema_version"), USAGE_RECEIPT_SCHEMA_VERSION, "schema_version")
    cost = data.get("cost_usd")
    missing = data.get("cost_missing")
    if not isinstance(missing, bool):
        raise SchemaError("cost_missing must be a boolean")
    if missing and cost is not None:
        raise SchemaError("missing cost receipts must not invent a USD amount")
    if not missing and cost is None:
        raise SchemaError("present cost receipts must include cost_usd")
    return UsageReceipt(
        schema_version=USAGE_RECEIPT_SCHEMA_VERSION,
        provider=_text(data.get("provider", DEFAULT_PROVIDER), "provider"),
        request_id=_text(data.get("request_id"), "request_id"),
        input_tokens=_int(data.get("input_tokens", 0), "input_tokens", minimum=0),
        output_tokens=_int(data.get("output_tokens", 0), "output_tokens", minimum=0),
        training_tokens=_int(data.get("training_tokens", 0), "training_tokens", minimum=0),
        cost_usd=None if missing else _finite(cost, "cost_usd"),
        cost_missing=missing,
        algorithm_id=_text(data.get("algorithm_id"), "algorithm_id"),
        implementation_version=_text(data.get("implementation_version"), "implementation_version"),
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be an object")
    return dict(value)


def _text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SchemaError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaError(f"{field} must be a finite number")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise SchemaError(f"{field} must be a finite number")
    return number


def _floats(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise SchemaError(f"{field} must be a list of numbers")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _string_map(value: Any, field: str) -> dict[str, str]:
    data = _object(value, field)
    return {key: _digest(raw, f"{field}.{key}") for key, raw in data.items()}


def _digest(value: Any, field: str) -> str:
    text = _text(value, field)
    if not text.startswith("sha256:"):
        raise SchemaError(f"{field} must be a sha256: digest")
    if len(text) != len("sha256:") + 64 or any(ch not in "0123456789abcdef" for ch in text[7:]):
        raise SchemaError(f"{field} must be a lowercase sha256 digest")
    return text


def _expect(value: Any, expected: str, field: str) -> None:
    if value != expected:
        raise SchemaError(f"{field} must be {expected}")
