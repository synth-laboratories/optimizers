"""Prime Intellect ``renderers`` — live Tinker chat templates.

Fixture tests keep the stand-in tokenizer. Paid Tinker runs use this package
so ``openai/gpt-oss-20b`` is Harmony-identical with vLLM/Tinker token-in paths,
not a re-rendered ``apply_chat_template`` string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..protocols import ProviderError

GPT_OSS_RENDERER_NAME = "gpt-oss"
DEFAULT_REASONING_EFFORT = "low"
BANKING77_RENDERER_VERSION = "renderers.gpt-oss.low.v1"


def renderer_is_available() -> bool:
    try:
        import renderers  # noqa: F401
    except ImportError:
        return False
    return True


def create_prime_renderer(
    tokenizer: Any,
    *,
    model_id: str = "",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Any:
    try:
        from renderers import GptOssRendererConfig, create_renderer
    except ImportError as exc:
        raise ProviderError(
            "renderers_missing",
            "install the Prime Intellect renderers package for Tinker chat templates",
        ) from exc
    name = str(getattr(tokenizer, "name_or_path", "") or model_id)
    if "gpt-oss" in name.lower() or "gpt-oss" in model_id.lower():
        return create_renderer(
            tokenizer, GptOssRendererConfig(reasoning_effort=reasoning_effort)
        )
    return create_renderer(tokenizer)


def renderer_version(renderer: Any) -> str:
    config = getattr(renderer, "config", None)
    name = str(getattr(config, "name", None) or GPT_OSS_RENDERER_NAME)
    effort = getattr(config, "reasoning_effort", None)
    if effort:
        return f"renderers.{name}.{effort}.v1"
    return f"renderers.{name}.v1"


def tokenize_with_renderer(
    renderer: Any, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
) -> dict[str, Any]:
    rows = [dict(message) for message in messages]
    prompt_only = add_generation_prompt or all(row.get("role") != "assistant" for row in rows)
    if prompt_only:
        prompt = [int(token) for token in renderer.render_ids(rows, add_generation_prompt=True)]
        if len(prompt) < 1:
            raise ProviderError("renderer_empty", "Prime renderer produced no prompt tokens")
        return {
            "input_ids": prompt[:-1] if len(prompt) > 1 else prompt,
            "target_tokens": prompt[1:] if len(prompt) > 1 else prompt,
            "weights": [0.0] * max(0, len(prompt) - 1),
            "n_tokens": 0,
            "prompt_token_ids": tuple(prompt),
            "stop_token_ids": tuple(int(token) for token in renderer.get_stop_token_ids()),
        }
    rendered = renderer.render(rows)
    ids = [int(token) for token in rendered.token_ids]
    indices = list(getattr(rendered, "message_indices", []) or [])
    sampled = list(getattr(rendered, "sampled_mask", []) or [])
    weights = []
    for index, _token in enumerate(ids):
        message_index = indices[index] if index < len(indices) else -1
        if message_index < 0 or message_index >= len(rows):
            weights.append(0.0)
            continue
        if sampled and index < len(sampled) and not sampled[index]:
            weights.append(0.0)
            continue
        weights.append(1.0 if rows[message_index].get("role") == "assistant" else 0.0)
    prompt_messages = [row for row in rows if row.get("role") != "assistant"]
    prompt = [int(token) for token in renderer.render_ids(prompt_messages, add_generation_prompt=True)]
    if len(ids) < 2:
        raise ProviderError("renderer_empty", "Prime renderer produced no training tokens")
    return {
        "input_ids": ids[:-1],
        "target_tokens": ids[1:],
        "weights": weights[1:] if len(weights) == len(ids) else weights[: len(ids) - 1],
        "n_tokens": sum(1 for weight in weights[1 : len(ids)] if weight > 0),
        "prompt_token_ids": tuple(prompt),
        "stop_token_ids": tuple(int(token) for token in renderer.get_stop_token_ids()),
    }


def bridge_with_renderer(
    renderer: Any,
    previous_prompt_token_ids: Sequence[int],
    previous_completion_token_ids: Sequence[int],
    messages: Sequence[Mapping[str, str]],
) -> dict[str, Any] | None:
    """Extend a sampled turn with the next one, carrying its ids through verbatim.

    ``bridge_to_next_turn`` is the renderers package's own answer to multi-turn:
    the next prompt is the previous prompt plus the previous completion plus the
    tokens the new turns add, so nothing sampled is ever tokenized from its text.
    It returns ``None`` when it cannot prove that contract holds, and so does
    this -- the caller then has a real history rewrite on its hands, not a
    rendering choice.
    """

    bridge = getattr(renderer, "bridge_to_next_turn", None)
    if not callable(bridge) or not messages:
        return None
    rendered = bridge(
        [int(token) for token in previous_prompt_token_ids],
        [int(token) for token in previous_completion_token_ids],
        [dict(message) for message in messages],
    )
    if rendered is None:
        return None
    prompt = tuple(int(token) for token in getattr(rendered, "token_ids", ()) or ())
    if not prompt:
        return None
    return {
        "prompt_token_ids": prompt,
        "stop_token_ids": tuple(int(token) for token in renderer.get_stop_token_ids()),
    }


def parse_completion(renderer: Any, token_ids: Sequence[int]) -> str:
    parsed = renderer.parse_response([int(token) for token in token_ids])
    return str(getattr(parsed, "content", "") or "")
