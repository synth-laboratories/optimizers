"""SFT dataset fingerprinting, rendering, and split identity."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
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


BANKING77_NANOCLASSIFY_SPLIT = "banking77.nanoclassify.v1"


def parse_example(raw: Mapping[str, Any], *, index: int) -> Example:
    if "messages" in raw:
        if not isinstance(raw["messages"], (list, tuple)):
            raise DatasetError(f"example {index} messages must be a list")
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
    text = raw.get("text")
    label = raw.get("category", raw.get("label"))
    if not isinstance(text, str) or not isinstance(label, str) or not text or not label:
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
    return digest_payload({
        "schema_version": "training.examples.v2",
        "examples": [
            {"example_id": example.example_id,
             "messages": [dict(message) for message in example.messages],
             "label": example.label, "text": example.text,
             "metadata": dict(example.metadata)}
            for example in examples
        ],
    })


def materialize_splits(
    examples: Sequence[Mapping[str, Any]],
    *,
    renderer_version: str = "chat.v1",
    train: Sequence[int] | None = None,
    calibration: Sequence[int] | None = None,
    heldout: Sequence[int] | None = None,
    require_evaluation: bool = True,
) -> SplitDataset:
    parsed = tuple(parse_example(raw, index=index) for index, raw in enumerate(examples))
    if not parsed:
        raise DatasetError("dataset is empty")
    ids = [example.example_id for example in parsed]
    if len(set(ids)) != len(ids):
        raise DatasetError("example identities must be unique")
    labels = tuple(sorted({example.label or "" for example in parsed if example.label}))
    if not require_evaluation and train is None and calibration is None and heldout is None:
        train = list(range(len(parsed)))
        calibration, heldout = [], []
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
        if not rows and (name == "train" or require_evaluation):
            raise DatasetError(f"{name} split is empty")
        for example in rows:
            if example.example_id in seen:
                raise DatasetError("splits must be disjoint")
            seen.add(example.example_id)
    taxonomy = digest_payload("\n".join(labels))
    split_digests = {name: fingerprint_examples(rows) for name, rows in split_rows.items()}
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "fingerprint_version": "training.examples.v2",
        "digest": digest_payload({"split_digests": split_digests, "renderer_version": renderer_version,
                                  "mask_policy": "assistant_only.next_token.v1"}),
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
    role = raw.get("role")
    if set(raw) - {"role", "content"}:
        raise DatasetError(f"example {index} message {position} has unsupported message metadata")
    content = raw.get("content")
    if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content:
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
    if config.get("training_file_id"):
        raise DatasetError("remote training_file_id is unsupported; provide immutable local data")
    def rows(value):
        if any(not isinstance(item, Mapping) for item in value):
            raise DatasetError("every dataset row must be an object")
        return [dict(item) for item in value]
    if isinstance(config.get("examples"), list):
        return rows(config["examples"])
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping) and isinstance(dataset.get("examples"), list):
        return rows(dataset["examples"])
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


def split_dataset_from_config(config: Mapping[str, Any]) -> SplitDataset:
    """Materialize either explicit indexes or NanoClassify's Banking77 split.

    The NanoClassify contract reserves ten examples per intent from the
    official train CSV, samples a fixed 400-row development set for checkpoint
    selection, and keeps closeout rows disjoint. A private source-index manifest
    marks the closeout set sealed; a seeded sample from the public test CSV is
    useful real evidence but is labeled unsealed in the manifest.
    """

    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    if dataset.get("split_strategy") != BANKING77_NANOCLASSIFY_SPLIT:
        examples = load_examples_from_config(config)
        return materialize_splits(
            examples,
            renderer_version=str(config.get("renderer_version") or "chat.v1"),
            train=dataset.get("train_indexes"),
            calibration=dataset.get("calibration_indexes"),
            heldout=dataset.get("heldout_indexes"),
            require_evaluation=(config.get("checkpoint_evaluation") or {}).get("mode", "builtin") not in {"none", "container"},
        )

    train_csv = _required_path(dataset.get("train_csv"), "dataset.train_csv")
    heldout_csv = _required_path(dataset.get("heldout_csv"), "dataset.heldout_csv")
    split_seed = int(dataset.get("split_seed") or 20260907)
    selection_seed = int(dataset.get("selection_seed") or 20260908)
    heldout_seed = int(dataset.get("heldout_seed") or 20260906)
    dev_per_class = int(dataset.get("dev_per_class") or 10)
    selection_size = int(dataset.get("selection_size") or 400)
    heldout_size = int(dataset.get("heldout_size") or 400)

    source_train = _csv_rows(train_csv, prefix="train")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_train:
        grouped[str(row["category"])].append(row)
    rng = random.Random(split_seed)
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for category in sorted(grouped):
        bucket = grouped[category][:]
        rng.shuffle(bucket)
        if len(bucket) <= dev_per_class:
            raise DatasetError(f"Banking77 intent {category} has no train rows after reservation")
        dev_rows.extend(bucket[:dev_per_class])
        train_rows.extend(bucket[dev_per_class:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    if selection_size > len(dev_rows):
        raise DatasetError("selection_size exceeds the reserved Banking77 development pool")
    selection_rows = random.Random(selection_seed).sample(dev_rows, selection_size)

    source_heldout = _csv_rows(heldout_csv, prefix="heldout")
    manifest_path = dataset.get("heldout_indices_json")
    if isinstance(manifest_path, str) and manifest_path.strip():
        index_payload = json.loads(_required_path(manifest_path, "dataset.heldout_indices_json").read_text())
        indices = index_payload.get("selected_source_indices", index_payload)
        if not isinstance(indices, list) or len(indices) != heldout_size:
            raise DatasetError("sealed heldout index manifest has the wrong sample size")
        if len(set(map(int, indices))) != len(indices):
            raise DatasetError("sealed heldout source indices must be unique")
        heldout_rows = [source_heldout[int(index)] for index in indices]
        heldout_sealed = True
        heldout_method = "explicit_source_indices"
    else:
        if heldout_size > len(source_heldout):
            raise DatasetError("heldout_size exceeds the Banking77 heldout source")
        heldout_rows = random.Random(heldout_seed).sample(source_heldout, heldout_size)
        heldout_sealed = False
        heldout_method = "seeded_public_test_sample"

    examples = [*train_rows, *selection_rows, *heldout_rows]
    train_end = len(train_rows)
    selection_end = train_end + len(selection_rows)
    result = materialize_splits(
        examples,
        renderer_version=str(config.get("renderer_version") or "chat.v1"),
        train=range(train_end),
        calibration=range(train_end, selection_end),
        heldout=range(selection_end, len(examples)),
    )
    result.manifest.update(
        {
            "split_strategy": BANKING77_NANOCLASSIFY_SPLIT,
            "split_seed": split_seed,
            "selection_seed": selection_seed,
            "heldout_seed": heldout_seed,
            "dev_per_class": dev_per_class,
            "selection_role": "development_selection",
            "heldout_role": "post_selection_closeout",
            "heldout_sealed": heldout_sealed,
            "heldout_selection_method": heldout_method,
            "source_counts": {"train": len(source_train), "heldout": len(source_heldout)},
        }
    )
    return result


def _required_path(value: Any, field: str):
    from pathlib import Path

    path = Path(str(value or "").strip())
    if not path.is_file():
        raise DatasetError(f"{field} is not a file: {path}")
    return path


def _csv_rows(path, *, prefix: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise DatasetError(f"{path} must contain Banking77 text,category columns")
    return [
        {
            "example_id": f"banking77_{prefix}_{index:05d}",
            "text": str(row["text"]),
            "category": str(row["category"]),
            "metadata": {"source": str(path), "source_index": index},
        }
        for index, row in enumerate(rows)
    ]
