from __future__ import annotations

import pytest

from synth_optimizers.training_eval import paired_uplift, public_evaluation
from synth_optimizers.sft_dataset import split_dataset_from_config


def _evaluation(correct: list[bool]) -> dict:
    return {
        "accuracy": sum(correct) / len(correct),
        "predictions": [
            {"example_id": f"ex_{index}", "correct": value}
            for index, value in enumerate(correct)
        ],
    }


def test_paired_uplift_requires_practical_and_uncertainty_gates() -> None:
    baseline = _evaluation([False] * 40 + [True] * 60)
    challenger = _evaluation([True] * 30 + [False] * 20 + [True] * 50)

    result = paired_uplift(
        baseline,
        challenger,
        bootstrap_resamples=1_000,
        minimum_paired_examples=100,
        minimum_claim_uplift=0.01,
    )

    assert result["uplift"] == pytest.approx(0.2)
    assert result["improved_examples"] == 30
    assert result["regressed_examples"] == 10
    assert result["ci_low"] > 0
    assert result["mcnemar_exact_p"] < 0.01
    assert result["verdict"] == "material_uplift"
    assert result["claim_ready"] is True


def test_one_extra_correct_of_400_is_not_material_uplift() -> None:
    baseline = _evaluation([False] * 50 + [True] * 350)
    challenger = _evaluation([True] + [False] * 49 + [True] * 350)

    result = paired_uplift(baseline, challenger, bootstrap_resamples=1_000)

    assert result["paired_n"] == 400
    assert result["uplift"] == pytest.approx(0.0025)
    assert result["verdict"] == "inconclusive"
    assert result["claim_ready"] is False


def test_public_evaluation_drops_private_per_example_rows() -> None:
    value = _evaluation([True, False])
    public = public_evaluation(value)

    assert "predictions" not in public
    assert public["accuracy"] == 0.5


def test_nanoclassify_split_is_disjoint_reproducible_and_labels_closeout_truthfully(tmp_path) -> None:
    train = tmp_path / "train.csv"
    heldout = tmp_path / "heldout.csv"
    train.write_text(
        "text,category\n"
        + "".join(f"a-{index},a\n" for index in range(20))
        + "".join(f"b-{index},b\n" for index in range(20)),
        encoding="utf-8",
    )
    heldout.write_text(
        "text,category\n"
        + "".join(f"ha-{index},a\n" for index in range(10))
        + "".join(f"hb-{index},b\n" for index in range(10)),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "split_strategy": "banking77.nanoclassify.v1",
            "train_csv": str(train),
            "heldout_csv": str(heldout),
            "dev_per_class": 5,
            "selection_size": 6,
            "heldout_size": 8,
        }
    }

    first = split_dataset_from_config(config)
    second = split_dataset_from_config(config)

    assert (len(first.train), len(first.calibration), len(first.heldout)) == (30, 6, 8)
    assert first.manifest == second.manifest
    assert first.manifest["heldout_sealed"] is False
    ids = [row.example_id for split in (first.train, first.calibration, first.heldout) for row in split]
    assert len(ids) == len(set(ids))
