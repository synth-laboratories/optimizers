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
