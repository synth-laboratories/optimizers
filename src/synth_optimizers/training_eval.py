"""Shared encoding and checkpoint eval for SFT and CISPO executors."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .providers.protocols import ProviderCheckpoint, SampleRequest, TrainingProvider
from .providers.tinker.client import new_request_id
from .providers.tinker.tokenize import extract_final_label, prompt_messages
from .sft_dataset import Example, render_chat, tokenize_for_sft


def system_prompt_from(config: Mapping[str, Any]) -> str | None:
    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    value = dataset.get("system_prompt") or config.get("system_prompt")
    text = str(value or "").strip()
    return text or None


def eval_max_tokens(config: Mapping[str, Any]) -> int:
    evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), Mapping) else {}
    training = config.get("training") if isinstance(config.get("training"), Mapping) else {}
    return int(evaluation.get("max_tokens") or training.get("max_sample_tokens") or 24)


def encode_example(
    provider: TrainingProvider,
    example: Example,
    *,
    system_prompt: str | None,
    add_generation_prompt: bool = False,
) -> dict[str, Any]:
    tokenize = getattr(provider, "tokenize_chat", None)
    messages = (
        prompt_messages(example, system_prompt)
        if add_generation_prompt
        else render_chat(example, system_prompt=system_prompt)
    )
    if callable(tokenize):
        return tokenize(messages, add_generation_prompt=add_generation_prompt)
    encoded = tokenize_for_sft(messages)
    full = list(encoded["input_ids"]) + [encoded["target_tokens"][-1]]
    return {
        **encoded,
        "prompt_token_ids": tuple(full if add_generation_prompt else encoded["input_ids"]),
    }


def evaluate_checkpoint(
    provider: TrainingProvider,
    checkpoint: Mapping[str, Any],
    examples: Sequence[Example],
    *,
    system_prompt: str | None = None,
    max_tokens: int = 24,
    on_example: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    handle = ProviderCheckpoint(
        checkpoint_id=str(checkpoint["checkpoint_id"]),
        provider_reference=str(checkpoint["provider_reference"]),
        step=int(checkpoint.get("step") or 0),
        digest=str(checkpoint.get("digest") or "sha256:" + "0" * 64),
        kind="inference",
    )
    per_intent: dict[str, list[int]] = {}
    correct = 0
    valid = 0
    predictions: list[dict[str, Any]] = []
    allowed_labels = {extract_final_label(example.label or "") for example in examples}
    for index, example in enumerate(examples):
        tokenized = encode_example(
            provider, example, system_prompt=system_prompt, add_generation_prompt=True
        )
        sampled = provider.sample_checkpoint(
                handle,
                SampleRequest(
                    request_id=new_request_id(handle.checkpoint_id, "eval", str(index)),
                    prompt_token_ids=tuple(tokenized.get("prompt_token_ids") or (1, 2, 3)),
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=index,
                ),
            )
        predicted = extract_final_label(sampled.text)
        label = extract_final_label(example.label or "")
        is_valid = predicted in allowed_labels
        is_correct = predicted == label
        bucket = per_intent.setdefault(label, [0, 0])
        bucket[1] += 1
        if is_valid:
            valid += 1
        if is_correct:
            correct += 1
            bucket[0] += 1
        record = {
            "example_id": example.example_id,
            "index": index,
            "label": label,
            "prediction": predicted,
            "valid_label": is_valid,
            "correct": is_correct,
            "completed": index + 1,
            "total": len(examples),
            "cumulative_accuracy": correct / (index + 1),
        }
        predictions.append(record)
        if on_example is not None:
            on_example(record)
    intent_rows = {
        label: {"correct": wins, "n": total, "accuracy": wins / total}
        for label, (wins, total) in sorted(per_intent.items())
    }
    return {
        "accuracy": correct / max(1, len(examples)),
        "n": len(examples),
        "macro_f1": _macro_f1(predictions, sorted(allowed_labels)),
        "valid_label_rate": valid / max(1, len(examples)),
        "correct": correct,
        "valid_labels": valid,
        "per_intent": intent_rows,
        "predictions": predictions,
    }


def public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Drop per-example rows from an aggregate event or policy bundle."""

    return {key: value for key, value in evaluation.items() if key != "predictions"}


def paired_uplift(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 4_000,
    seed: int = 20260907,
    minimum_claim_uplift: float = 0.01,
    minimum_paired_examples: int = 100,
) -> dict[str, Any]:
    """Compute a reproducible paired accuracy comparison on identical examples.

    The interval is a percentile bootstrap over {-1, 0, +1} per-example
    correctness deltas. McNemar's exact two-sided p-value is included because
    aggregate accuracies alone hide whether the two models changed the same
    examples.
    """

    base_by_id = {
        str(row["example_id"]): bool(row["correct"])
        for row in baseline.get("predictions") or ()
    }
    challenger_by_id = {
        str(row["example_id"]): bool(row["correct"])
        for row in challenger.get("predictions") or ()
    }
    shared = sorted(base_by_id.keys() & challenger_by_id.keys())
    deltas = [float(challenger_by_id[key]) - float(base_by_id[key]) for key in shared]
    improved = sum(delta > 0 for delta in deltas)
    regressed = sum(delta < 0 for delta in deltas)
    unchanged = len(deltas) - improved - regressed
    uplift = statistics.fmean(deltas) if deltas else None
    ci_low, ci_high = _bootstrap_interval(
        deltas,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    exact_p = _mcnemar_exact_p(improved, regressed)
    enough = len(deltas) >= minimum_paired_examples
    material = bool(
        enough
        and uplift is not None
        and uplift >= minimum_claim_uplift
        and ci_low is not None
        and ci_low > 0.0
    )
    if not deltas:
        verdict = "unavailable"
        reason = "baseline and challenger have no shared per-example evidence"
    elif not enough:
        verdict = "inconclusive"
        reason = f"{len(deltas)} paired examples is below the {minimum_paired_examples} claim minimum"
    elif material:
        verdict = "material_uplift"
        reason = "paired uplift clears the practical threshold and its confidence interval excludes zero"
    elif uplift is not None and uplift < 0.0 and ci_high is not None and ci_high < 0.0:
        verdict = "material_regression"
        reason = "paired confidence interval is entirely below zero"
    else:
        verdict = "inconclusive"
        reason = "observed uplift does not clear both the practical and uncertainty gates"
    return {
        "schema_version": "training.paired-uplift.v1",
        "metric": "accuracy",
        "paired_n": len(deltas),
        "baseline_accuracy": baseline.get("accuracy"),
        "challenger_accuracy": challenger.get("accuracy"),
        "uplift": uplift,
        "confidence": confidence,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "improved_examples": improved,
        "regressed_examples": regressed,
        "unchanged_examples": unchanged,
        "discordant_examples": improved + regressed,
        "mcnemar_exact_p": exact_p,
        "minimum_claim_uplift": minimum_claim_uplift,
        "minimum_paired_examples": minimum_paired_examples,
        "verdict": verdict,
        "claim_ready": material,
        "reason": reason,
    }


def _macro_f1(records: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> float:
    values: list[float] = []
    for label in labels:
        tp = sum(row["label"] == label and row["prediction"] == label for row in records)
        fp = sum(row["label"] != label and row["prediction"] == label for row in records)
        fn = sum(row["label"] == label and row["prediction"] != label for row in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(values) if values else 0.0


def _bootstrap_interval(
    values: Sequence[float], *, confidence: float, resamples: int, seed: int
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return means[int(tail * (resamples - 1))], means[int((1.0 - tail) * (resamples - 1))]


def _mcnemar_exact_p(improved: int, regressed: int) -> float | None:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    smaller = min(improved, regressed)
    probability = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)
