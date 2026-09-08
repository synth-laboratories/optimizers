from __future__ import annotations

from synth_optimizers.sft_dataset import DatasetError, materialize_splits
import pytest


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(DatasetError, match="empty"):
        materialize_splits([])


def test_malformed_and_inconsistent_examples_are_rejected() -> None:
    with pytest.raises(DatasetError, match="at least two chat messages"):
        materialize_splits([{"messages": [{"role": "user", "content": "hi"}]}])
    with pytest.raises(DatasetError, match="unique"):
        materialize_splits(
            [
                {"example_id": "dup", "text": "a", "category": "x"},
                {"example_id": "dup", "text": "b", "category": "y"},
                {"example_id": "other", "text": "c", "category": "z"},
            ]
        )


def test_identity_includes_all_turns_roles_metadata_and_renderer():
    from copy import deepcopy
    from synth_optimizers.sft_dataset import fingerprint_examples, parse_example

    row = {
        "messages": [
            {"role": "system", "content": "original"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {"mask": "assistant"},
    }
    original = fingerprint_examples([parse_example(row, index=0)])
    for field, value in (("content", "changed"), ("role", "user")):
        changed = deepcopy(row)
        changed["messages"][0][field] = value
        assert fingerprint_examples([parse_example(changed, index=0)]) != original
    changed = deepcopy(row)
    changed["metadata"]["mask"] = "other"
    assert fingerprint_examples([parse_example(changed, index=0)]) != original
    rows = [{"text": str(i), "category": "a"} for i in range(3)]
    assert (
        materialize_splits(rows).manifest["digest"]
        != materialize_splits(rows, renderer_version="chat.v2").manifest["digest"]
    )


def test_exact_whitespace_and_unsupported_chat_forms():
    from synth_optimizers.sft_dataset import parse_example

    row = {
        "messages": [
            {"role": "user", "content": "  question\n"},
            {"role": "assistant", "content": " answer\t"},
        ]
    }
    assert parse_example(row, index=0).messages[0]["content"] == "  question\n"
    for message in (
        {"role": "assistant", "content": ["multimodal"]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "answer", "weight": 0},
    ):
        with pytest.raises(DatasetError):
            parse_example({"messages": [row["messages"][0], message]}, index=0)
