"""Seed search objects: one weak code policy and one SpeedRunner harness per game.

Seeds are deliberately weak. A seed that already sits at the ceiling cannot show
uplift -- the earlier Craftax ReAct arm scored 2.0/2.0 on its seed prompt and the
search had nothing to prove.
"""

from __future__ import annotations

import importlib

from gamebench_levers import GAMES


def code_seed(game: str) -> str:
    return importlib.import_module(f"gamebench_levers.seeds.{game}").SEED_POLICY


def harness_seed(game: str) -> str:
    return importlib.import_module(f"gamebench_levers.seeds.{game}").SEED_HARNESS


def prompt_seed(game: str) -> str:
    return importlib.import_module(f"gamebench_levers.seeds.{game}").SEED_PROMPT


__all__ = ["GAMES", "code_seed", "harness_seed", "prompt_seed"]
