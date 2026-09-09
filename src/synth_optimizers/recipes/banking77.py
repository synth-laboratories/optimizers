"""Versioned Banking77 SFT and CISPO recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts.training_schemas import (
    CISPO_CONFIG_SCHEMA_VERSION,
    CISPO_IMPLEMENTATION,
    CISPO_IMPLEMENTATION_VERSION,
    DEFAULT_SFT_MODEL,
    SFT_CONFIG_SCHEMA_VERSION,
    SFT_IMPLEMENTATION,
    SFT_IMPLEMENTATION_VERSION,
)
from ..runtime import digest_payload, RUNNER_VERSION


SFT_RECIPE_ID = "banking77.sft.v1"
CISPO_RECIPE_ID = "banking77.cispo.v1"
SCORER_VERSION = "banking77.exact_label.v1"
RENDERER_VERSION = "renderers.gpt-oss.low.v1"
TAXONOMY_VERSION = "banking77.labels.v1"


FIXTURE_ROWS: tuple[dict[str, str], ...] = (
    {"text": "I want to order a new debit card", "category": "order_physical_card"},
    {"text": "Please freeze my lost card", "category": "lost_or_stolen_card"},
    {"text": "What is my checking balance?", "category": "balance_not_updated_after_cheque_or_cash_deposit"},
    {"text": "How do I activate the replacement card?", "category": "activate_my_card"},
    {"text": "The ATM kept my card", "category": "card_swallowed"},
    {"text": "I need a statement for last month", "category": "get_physical_card"},
)


@dataclass(frozen=True, slots=True)
class Banking77Recipe:
    recipe_id: str
    algorithm_id: str
    mode: str
    request: dict[str, Any]
    notes: str


def system_prompt(labels: Sequence[str]) -> str:
    return (
        "Classify the customer banking message. Return exactly one label from this list, "
        "with no explanation or punctuation:\n" + ", ".join(labels)
    )


def fixture_examples() -> list[dict[str, Any]]:
    return [
        {
            "example_id": f"banking77_fixture_{index:02d}",
            "text": row["text"],
            "category": row["category"],
        }
        for index, row in enumerate(FIXTURE_ROWS)
    ]


def sft_recipe(*, seed: int = 20260902, steps: int = 2) -> Banking77Recipe:
    examples = fixture_examples()
    labels = sorted({row["category"] for row in examples})
    request = {
        "schema_version": SFT_CONFIG_SCHEMA_VERSION,
        "algorithm_id": "sft",
        "implementation": SFT_IMPLEMENTATION,
        "implementation_version": SFT_IMPLEMENTATION_VERSION,
        "provider": "tinker",
        "model_id": DEFAULT_SFT_MODEL,
        "base_model": DEFAULT_SFT_MODEL,
        "backend": "tinker",
        "renderer_version": RENDERER_VERSION,
        "runner_version": RUNNER_VERSION,
        "seed": seed,
        "repeat_index": 0,
        "rank": 8,
        "dataset": {
            "recipe_id": SFT_RECIPE_ID,
            "examples": examples,
            "train_indexes": [0, 1, 2, 3],
            "calibration_indexes": [4],
            "heldout_indexes": [5],
            "label_taxonomy": labels,
            "system_prompt": system_prompt(labels),
            "scorer_version": SCORER_VERSION,
        },
        "training": {
            "steps": steps,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "checkpoint_every_steps": 1,
            "eval_every_steps": 1,
        },
        "evaluation": {
            "scorer_version": SCORER_VERSION,
            "heldout_locked": True,
        },
    }
    return Banking77Recipe(
        recipe_id=SFT_RECIPE_ID,
        algorithm_id="sft",
        mode="canonical",
        request=request,
        notes=(
            "Canonical Banking77 SFT. Report base, checkpoint, and held-out accuracy. "
            "A drop such as 0.81 → 0.47 is a regression, not a successful train."
        ),
    )


def cispo_recipe(*, mode: str = "canonical", seed: int = 20260902, updates: int = 1) -> Banking77Recipe:
    if mode not in {"canonical", "learning_signal"}:
        raise ValueError("Banking77 CISPO mode must be canonical or learning_signal")
    examples = fixture_examples()
    labels = sorted({row["category"] for row in examples})
    training = {
        "updates": updates,
        "group_size": 2,
        "prompts_per_update": 1,
        "max_sample_tokens": 8,
        "temperature": 1.0 if mode == "learning_signal" else 0.0,
        "learning_rate": 5e-6,
        "eps_clip": 1.0,
        "eps_clip_high": 4.0,
        "normalize_group_rewards": True,
        "checkpoint_every_updates": 1,
    }
    request = {
        "schema_version": CISPO_CONFIG_SCHEMA_VERSION,
        "algorithm_id": "cispo",
        "implementation": CISPO_IMPLEMENTATION,
        "implementation_version": CISPO_IMPLEMENTATION_VERSION,
        "provider": "tinker",
        "model_id": DEFAULT_SFT_MODEL,
        "base_model": DEFAULT_SFT_MODEL,
        "renderer_version": RENDERER_VERSION,
        "runner_version": RUNNER_VERSION,
        "seed": seed,
        "repeat_index": 0,
        "mode": mode,
        "rank": 8,
        "dataset": {
            "recipe_id": CISPO_RECIPE_ID,
            "examples": examples,
            "train_indexes": [0, 1, 2, 3] if mode == "canonical" else [0, 1],
            "calibration_indexes": [4],
            "heldout_indexes": [5],
            "label_taxonomy": labels,
            "system_prompt": system_prompt(labels),
            "scorer_version": SCORER_VERSION,
            "heldout_locked": True,
        },
        "training": training,
        "reward": {"version": SCORER_VERSION, "task": "banking77"},
        "evaluation": {
            "scorer_version": SCORER_VERSION,
            "heldout_locked": True,
            "mode": mode,
        },
    }
    notes = (
        "Canonical Banking77 CISPO. Held-out split is frozen. Saturation and zero-advantage "
        "groups are primary results, not hidden."
        if mode == "canonical"
        else (
            "Learning-signal demonstration: harder/underfit train subset only. "
            "The canonical held-out split is unchanged."
        )
    )
    return Banking77Recipe(
        recipe_id=f"{CISPO_RECIPE_ID}.{mode}",
        algorithm_id="cispo",
        mode=mode,
        request=request,
        notes=notes,
    )


def evaluation_report(
    *,
    base_accuracy: float,
    checkpoint_accuracy: float,
    heldout_accuracy: float,
    per_intent: Mapping[str, Mapping[str, Any]],
    train_loss: Sequence[float],
    checkpoint_trend: Sequence[float],
) -> dict[str, Any]:
    gap = heldout_accuracy - checkpoint_accuracy
    regression = heldout_accuracy < base_accuracy - 1e-9
    improved = [
        label
        for label, stats in per_intent.items()
        if float(stats.get("accuracy", 0.0)) > float(stats.get("base_accuracy", 0.0))
    ]
    regressed = [
        label
        for label, stats in per_intent.items()
        if float(stats.get("accuracy", 0.0)) < float(stats.get("base_accuracy", 1.0))
    ]
    return {
        "schema_version": "banking77.eval.v1",
        "base_accuracy": base_accuracy,
        "checkpoint_accuracy": checkpoint_accuracy,
        "heldout_accuracy": heldout_accuracy,
        "generalization_gap": gap,
        "per_intent": dict(per_intent),
        "regressed_intents": sorted(regressed),
        "improved_intents": sorted(improved),
        "regression_detected": regression,
        "train_loss": list(train_loss),
        "checkpoint_trend": list(checkpoint_trend),
        "headline": (
            f"held-out {heldout_accuracy:.2f} regressed from base {base_accuracy:.2f}"
            if regression
            else f"held-out {heldout_accuracy:.2f} vs base {base_accuracy:.2f}"
        ),
        "digest": digest_payload(
            {
                "base": base_accuracy,
                "heldout": heldout_accuracy,
                "scorer": SCORER_VERSION,
            }
        ),
    }
