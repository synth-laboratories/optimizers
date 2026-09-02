from __future__ import annotations

from types import SimpleNamespace

import pytest

from synth_optimizers.providers.tinker.prime import (
    BANKING77_RENDERER_VERSION,
    parse_completion,
    tokenize_with_renderer,
)
from synth_optimizers.providers.tinker.tokenize import extract_final_label, tokenize_live
from synth_optimizers.recipes.banking77 import RENDERER_VERSION, sft_recipe


class StubRenderer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(name="gpt-oss", reasoning_effort="low")

    def render_ids(self, messages, *, add_generation_prompt=False):
        ids = [10, 11, 12]
        if add_generation_prompt:
            ids.append(13)
        if any(message.get("role") == "assistant" for message in messages):
            ids.extend([20, 21])
        return ids

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, token_ids):
        return SimpleNamespace(content="order_physical_card")

    def render(self, messages, tools=None, add_generation_prompt=False):
        ids = self.render_ids(messages, add_generation_prompt=add_generation_prompt)
        n = len(ids)
        assistant = 1 if any(message.get("role") == "assistant" for message in messages) else 0
        return SimpleNamespace(
            token_ids=ids,
            message_indices=[-1] * (n - 2) + [assistant] * min(2, n),
            sampled_mask=[False] * (n - 2) + [True] * min(2, n),
            is_content=[False] * n,
        )


def test_banking77_pins_the_prime_gpt_oss_renderer() -> None:
    assert RENDERER_VERSION == BANKING77_RENDERER_VERSION
    assert sft_recipe().request["renderer_version"] == "renderers.gpt-oss.low.v1"


def test_live_tokenize_uses_prime_loss_mask() -> None:
    encoded = tokenize_with_renderer(
        StubRenderer(),
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "order_physical_card"},
        ],
    )
    assert encoded["prompt_token_ids"][-1] == 13
    assert encoded["n_tokens"] > 0
    assert encoded["stop_token_ids"] == (99,)


def test_parse_completion_uses_harmony_final_channel() -> None:
    assert parse_completion(StubRenderer(), [1, 2, 3]) == "order_physical_card"
    assert extract_final_label("<|channel|>final<|message|>Lost-Or-Stolen-Card<|return|>") == (
        "lost_or_stolen_card"
    )


def test_fixture_tokenize_does_not_import_renderers() -> None:
    encoded = tokenize_live(
        None,
        None,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        add_generation_prompt=False,
    )
    assert encoded["n_tokens"] > 0
    assert "prompt_token_ids" in encoded


def test_missing_renderers_package_fails_closed(monkeypatch) -> None:
    import builtins

    from synth_optimizers.providers.protocols import ProviderError
    from synth_optimizers.providers.tinker.prime import create_prime_renderer

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "renderers" or name.startswith("renderers."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match="renderers_missing"):
        create_prime_renderer(SimpleNamespace(name_or_path="openai/gpt-oss-20b"))
