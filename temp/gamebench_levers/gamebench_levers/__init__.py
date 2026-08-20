"""GEPA lever containers over GameBench single-player games.

Two search modes per game, one shape:

  code    policy_script   whole_file.v1 / unified_diff.v1   act(obs) -> action
  harness harness_module  harness_restart.v1                run_episode(env, prompt, ...)

The harness seed is a SpeedRunner-style actor (arXiv:2608.11338): the LLM picks a
public skill, the skill expands to primitive env steps with no further model call.
"""

from __future__ import annotations

ENV_PROTOCOL = "gamebench_env.v1"
GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

GAMES = ("sokoban", "craftax", "rogue", "dungeongrid")
MODES = ("code", "harness")

__all__ = ["ENV_PROTOCOL", "GEPA_OPTIMIZER_CONTRACT_VERSION", "GAMES", "MODES"]
