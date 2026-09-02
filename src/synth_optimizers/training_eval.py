"""Shared encoding and checkpoint eval for SFT and CISPO executors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    for index, example in enumerate(examples):
        tokenized = encode_example(
            provider, example, system_prompt=system_prompt, add_generation_prompt=True
        )
        predicted = extract_final_label(
            provider.sample_checkpoint(
                handle,
                SampleRequest(
                    request_id=new_request_id(handle.checkpoint_id, "eval", str(index)),
                    prompt_token_ids=tuple(tokenized.get("prompt_token_ids") or (1, 2, 3)),
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=index,
                ),
            ).text
        )
        label = extract_final_label(example.label or "")
        bucket = per_intent.setdefault(label, [0, 0])
        bucket[1] += 1
        if predicted == label:
            correct += 1
            bucket[0] += 1
    return {
        "accuracy": correct / max(1, len(examples)),
        "n": len(examples),
        "per_intent": {
            label: {"correct": wins, "n": total, "accuracy": wins / total}
            for label, (wins, total) in sorted(per_intent.items())
        },
    }
