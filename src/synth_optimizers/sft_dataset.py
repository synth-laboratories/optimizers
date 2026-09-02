"""SFT dataset fingerprinting, rendering, and split identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts.training_schemas import DATASET_MANIFEST_SCHEMA_VERSION
from .runtime import digest_payload


class DatasetError(ValueError):
    """An SFT or CISPO dataset was empty, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    messages: tuple[Mapping[str, str], ...]
    label: str | None
    text: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SplitDataset:
    train: tuple[Example, ...]
    calibration: tuple[Example, ...]
    heldout: tuple[Example, ...]
    labels: tuple[str, ...]
    renderer_version: str
    manifest: dict[str, Any]


def parse_example(raw: Mapping[str, Any], *, index: int) -> Example:
    if "messages" in raw:
        messages = tuple(_message(item, index=index, position=pos) for pos, item in enumerate(raw["messages"]))
        if len(messages) < 2:
            raise DatasetError(f"example {index} needs at least two chat messages")
        roles = [message["role"] for message in messages]
        if "user" not in roles or "assistant" not in roles:
            raise DatasetError(f"example {index} must include user and assistant turns")
        assistant = next(message["content"] for message in reversed(messages) if message["role"] == "assistant")
        user = next(message["content"] for message in messages if message["role"] == "user")
        return Example(
            example_id=str(raw.get("example_id") or f"ex_{index:06d}"),
            messages=messages,
            label=assistant,
            text=user,
            metadata=dict(raw.get("metadata") or {}),
        )
    text = str(raw.get("text") or "").strip()
    label = str(raw.get("category") or raw.get("label") or "").strip()
    if not text or not label:
        raise DatasetError(f"example {index} is missing text/category")
    return Example(
        example_id=str(raw.get("example_id") or f"ex_{index:06d}"),
        messages=(
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ),
        label=label,
        text=text,
        metadata=dict(raw.get("metadata") or {}),
    )


def fingerprint_examples(examples: Sequence[Example]) -> str:
    payload = "\n".join(
        f"{example.example_id}\t{example.messages[-1]['content']}\t{example.messages[-2]['content']}"
        for example in examples
    )
    return digest_payload(payload)


def materialize_splits(
    examples: Sequence[Mapping[str, Any]],
    *,
    renderer_version: str = "chat.v1",
    train: Sequence[int] | None = None,
    calibration: Sequence[int] | None = None,
    heldout: Sequence[int] | None = None,
) -> SplitDataset:
    parsed = tuple(parse_example(raw, index=index) for index, raw in enumerate(examples))
    if not parsed:
        raise DatasetError("dataset is empty")
    ids = [example.example_id for example in parsed]
    if len(set(ids)) != len(ids):
        raise DatasetError("example identities must be unique")
    labels = tuple(sorted({example.label or "" for example in parsed if example.label}))
    if train is None and calibration is None and heldout is None:
        if len(parsed) < 3:
            raise DatasetError("dataset must provide train, calibration, and held-out examples")
        heldout_idx = [len(parsed) - 1]
        calibration_idx = [len(parsed) - 2]
        train_idx = list(range(0, len(parsed) - 2))
    else:
        train_idx = list(train or [])
        calibration_idx = list(calibration or [])
        heldout_idx = list(heldout or [])
    split_rows = {
        "train": _select(parsed, train_idx, "train"),
        "calibration": _select(parsed, calibration_idx, "calibration"),
        "heldout": _select(parsed, heldout_idx, "heldout"),
    }
    seen: set[str] = set()
    for name, rows in split_rows.items():
        if not rows:
            raise DatasetError(f"{name} split is empty")
        for example in rows:
            if example.example_id in seen:
                raise DatasetError("splits must be disjoint")
            seen.add(example.example_id)
    taxonomy = digest_payload("\n".join(labels))
    split_digests = {name: fingerprint_examples(rows) for name, rows in split_rows.items()}
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "digest": digest_payload("".join(split_digests[name] for name in ("train", "calibration", "heldout"))),
        "split_digests": split_digests,
        "example_counts": {name: len(rows) for name, rows in split_rows.items()},
        "label_taxonomy_digest": taxonomy,
        "renderer_version": renderer_version,
    }
    return SplitDataset(
        train=split_rows["train"],
        calibration=split_rows["calibration"],
        heldout=split_rows["heldout"],
        labels=labels,
        renderer_version=renderer_version,
        manifest=manifest,
    )


def render_chat(example: Example, *, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages = [dict(message) for message in example.messages]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def tokenize_for_sft(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Deterministic stand-in tokenizer used by the fixture provider and tests."""

    encoded: list[int] = []
    weights: list[float] = []
    for message in messages:
        tokens = [1, *((ord(ch) % 97) + 2 for ch in message["content"]), 2]
        encoded.extend(tokens)
        weight = 1.0 if message["role"] == "assistant" else 0.0
        weights.extend([weight] * len(tokens))
    if len(encoded) < 2:
        raise DatasetError("rendered example produced no tokens")
    return {
        "input_ids": encoded[:-1],
        "target_tokens": encoded[1:],
        "weights": weights[1:],
        "n_tokens": sum(1 for weight in weights[1:] if weight > 0),
    }


def _message(raw: Any, *, index: int, position: int) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise DatasetError(f"example {index} message {position} must be an object")
    role = str(raw.get("role") or "").strip()
    content = str(raw.get("content") or "").strip()
    if role not in {"system", "user", "assistant"} or not content:
        raise DatasetError(f"example {index} message {position} is malformed")
    return {"role": role, "content": content}


def _select(examples: Sequence[Example], indexes: Sequence[int], name: str) -> tuple[Example, ...]:
    selected: list[Example] = []
    for index in indexes:
        if index < 0 or index >= len(examples):
            raise DatasetError(f"{name} split index {index} is out of range")
        selected.append(examples[index])
    return tuple(selected)


def load_examples_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(config.get("examples"), list):
        return [dict(item) for item in config["examples"] if isinstance(item, Mapping)]
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping) and isinstance(dataset.get("examples"), list):
        return [dict(item) for item in dataset["examples"] if isinstance(item, Mapping)]
    training_jsonl = config.get("training_jsonl")
    if isinstance(training_jsonl, str) and training_jsonl.strip():
        import json
        from pathlib import Path

        path = Path(training_jsonl)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    raise DatasetError("SFT config requires examples, dataset.examples, or training_jsonl")
