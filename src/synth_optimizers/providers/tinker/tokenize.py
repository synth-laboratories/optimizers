"""Chat tokenization used by both the fixture transport and live Tinker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...sft_dataset import Example, tokenize_for_sft


def extract_final_label(text: str) -> str:
    """Normalize a model completion to a Banking77-style label."""

    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    for terminator in ("<|return|>", "<|end|>", "<|eot_id|>"):
        text = text.split(terminator, 1)[0]
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def prompt_messages(example: Example, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        dict(message) for message in example.messages if message.get("role") != "assistant"
    )
    return messages


def fallback_tokenize(
    messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
) -> dict[str, Any]:
    encoded = tokenize_for_sft(messages)
    full = list(encoded["input_ids"]) + [encoded["target_tokens"][-1]]
    prompt = tuple(full if add_generation_prompt else encoded["input_ids"])
    return {
        "input_ids": encoded["input_ids"],
        "target_tokens": encoded["target_tokens"],
        "weights": encoded["weights"],
        "n_tokens": encoded["n_tokens"],
        "prompt_token_ids": prompt,
        "stop_token_ids": (),
    }


def tokenize_live(
    renderer: Any | None,
    tokenizer: Any | None,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> dict[str, Any]:
    if renderer is not None:
        from .prime import tokenize_with_renderer

        return tokenize_with_renderer(
            renderer, messages, add_generation_prompt=add_generation_prompt
        )
    if tokenizer is not None:
        return tokenize_with_tokenizer(
            tokenizer, messages, add_generation_prompt=add_generation_prompt
        )
    return fallback_tokenize(messages, add_generation_prompt=add_generation_prompt)


def token_ids_from(value: Any) -> list[int]:
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def tokenize_with_tokenizer(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
) -> dict[str, Any]:
    """Last-resort HF chat template. Live gpt-oss should use Prime ``renderers``."""

    prefix = token_ids_from(
        tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
    )
    if add_generation_prompt or all(message.get("role") != "assistant" for message in messages):
        return {
            "input_ids": prefix[:-1] if len(prefix) > 1 else prefix,
            "target_tokens": prefix[1:] if len(prefix) > 1 else prefix,
            "weights": [0.0] * max(0, len(prefix) - 1),
            "n_tokens": 0,
            "prompt_token_ids": tuple(prefix),
            "stop_token_ids": (),
        }
    full = token_ids_from(
        tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False
        )
    )
    if full[: len(prefix)] != prefix:
        raise ValueError("chat template is not prefix-stable")
    weights = [0.0] * len(prefix) + [1.0] * (len(full) - len(prefix))
    return {
        "input_ids": full[:-1],
        "target_tokens": full[1:],
        "weights": weights[1:],
        "n_tokens": sum(1 for weight in weights[1:] if weight > 0),
        "prompt_token_ids": tuple(prefix),
        "stop_token_ids": (),
    }
