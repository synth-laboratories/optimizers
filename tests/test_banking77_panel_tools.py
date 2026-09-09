from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _module(name: str):
    path = ROOT / "docs/e2e" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = _module("freeze_banking77_panel")
validate = _module("validate_banking77_eval")


def corpus(*, choices: int = 2) -> list[dict]:
    return [
        {
            "task_id": f"banking77/heldout/{label * choices + choice}",
            "seed": label * choices + choice,
            "label": f"label-{label:02d}",
        }
        for label in range(77)
        for choice in range(choices)
    ]


def frozen(
    *,
    excluded: set[str] | None = None,
    seed: str = "panel-1",
    examples_per_intent: int = 1,
) -> dict:
    rows = corpus(choices=max(2, examples_per_intent + 1))
    return freeze.freeze_panel(
        rows,
        excluded=excluded or set(),
        panel_seed=seed,
        source={"path": "fixture", "sha256": "sha256:source", "rows": len(rows)},
        exclusion_inventory=[],
        examples_per_intent=examples_per_intent,
    )


def receipt(panel: dict) -> dict:
    seeds = [{"task_id": row["task_id"], "seed": row["seed"]} for row in panel["rows"]]
    arms = {}
    rewards = {}
    for arm, checkpoint, reference in (
        ("baseline", "base", "tinker://base"),
        ("trained", "trained", "tinker://trained"),
    ):
        arm_rewards = [
            float((index + (arm == "trained")) % 3 != 0)
            for index in range(len(panel["rows"]))
        ]
        rewards[arm] = arm_rewards
        attempts = [
            {
                "arm": arm,
                "task_id": row["task_id"],
                "seed": row["seed"],
                "sample_index": index,
                "rollout_id": f"rollout-{arm}-{index}",
                "proxy_request_id": f"proxy-{arm}-{index}",
                "reward": arm_rewards[index],
                "reward_channel": "score::team-0",
                "terminal_status": "completed",
                "checkpoint_ids": [checkpoint],
                "sampler_references": [reference],
                "trace_digest": f"sha256:{arm}-{index}",
            }
            for index, row in enumerate(panel["rows"])
        ]
        arms[arm] = {
            "resolved_id": checkpoint,
            "catalogued_sampler_references": [reference],
            "loaded_sampler_references": [reference],
            "attempt_count": len(panel["rows"]),
            "attempts": attempts,
        }
    return {
        "split": "heldout",
        "seeds": seeds,
        "arms": arms,
        "paired_summary": {
            "rows": [
                {
                    "task_id": row["task_id"],
                    "seed": row["seed"],
                    "baseline_reward": rewards["baseline"][index],
                    "trained_reward": rewards["trained"][index],
                    "delta": rewards["trained"][index] - rewards["baseline"][index],
                }
                for index, row in enumerate(panel["rows"])
            ]
        },
    }


def test_panel_selection_is_deterministic_and_respects_exclusions() -> None:
    first = frozen(seed="declared")
    selected = first["task_ids"][0]
    second = frozen(excluded={selected}, seed="declared")

    assert first == frozen(seed="declared")
    assert len(first["rows"]) == len({row["label"] for row in first["rows"]}) == 77
    assert selected not in second["task_ids"]
    assert first["panel_digest"] != second["panel_digest"]


def test_panel_can_freeze_multiple_balanced_examples_per_intent() -> None:
    panel = frozen(seed="confirmatory", examples_per_intent=5)

    assert panel["examples_per_intent"] == 5
    assert len(panel["rows"]) == len(set(panel["task_ids"])) == 385
    assert set(Counter(row["label"] for row in panel["rows"]).values()) == {5}

    result = validate.validate(
        panel,
        receipt(panel),
        expected_baseline="base",
        expected_trained="trained",
        train_ids=set(),
        prior_ids=set(),
        receipt_sha256="sha256:receipt",
        bootstrap_seed=4,
        bootstrap_replicates=100,
    )
    assert result["pairs"] == 385


def test_panel_refuses_less_than_77_labels() -> None:
    with pytest.raises(ValueError, match="exactly 77"):
        freeze.freeze_panel(
            corpus()[:-2],
            excluded=set(),
            panel_seed="x",
            source={},
            exclusion_inventory=[],
        )


def test_validator_accepts_complete_paired_receipt_and_reports_exact_stats() -> None:
    panel = frozen()
    result = validate.validate(
        panel,
        receipt(panel),
        expected_baseline="base",
        expected_trained="trained",
        train_ids=set(),
        prior_ids=set(),
        receipt_sha256="sha256:receipt",
        bootstrap_seed=4,
        bootstrap_replicates=1000,
    )

    assert result["valid"] is True
    assert result["pairs"] == 77
    assert result["wins"] + result["losses"] + result["ties"] == 77
    assert 0.0 <= result["exact_two_sided_mcnemar_p"] <= 1.0


@pytest.mark.parametrize("defect", ["overlap", "channel", "order", "checkpoint"])
def test_validator_refuses_contamination_and_pairing_defects(defect: str) -> None:
    panel = frozen()
    evaluation = receipt(panel)
    train_ids: set[str] = set()
    if defect == "overlap":
        train_ids.add(panel["task_ids"][0])
    elif defect == "channel":
        evaluation["arms"]["trained"]["attempts"][0]["reward_channel"] = "score"
    elif defect == "order":
        evaluation["arms"]["trained"]["attempts"].reverse()
    else:
        evaluation["arms"]["trained"]["attempts"][0]["checkpoint_ids"] = ["wrong"]

    with pytest.raises(ValueError, match="invalid Banking77 evaluation"):
        validate.validate(
            panel,
            evaluation,
            expected_baseline="base",
            expected_trained="trained",
            train_ids=train_ids,
            prior_ids=set(),
            receipt_sha256="sha256:receipt",
            bootstrap_replicates=10,
        )


def test_exact_mcnemar_is_honest_for_two_wins_no_losses() -> None:
    stats = validate.paired_statistics([1.0, 1.0] + [0.0] * 75, replicates=100)
    assert stats["exact_two_sided_mcnemar_p"] == 0.5
