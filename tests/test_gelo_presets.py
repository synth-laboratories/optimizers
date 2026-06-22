"""Materialization tests for public GELO presets.

Guards the hosted config shape that `synth-optimizers gelo submit --preset ...`
sends to the backend. Run: `python -m pytest tests/test_gelo_presets.py`.
"""
from __future__ import annotations

import pytest

from synth_optimizers.gelo import (
    GeloMaterializeError,
    GeloPreset,
    GeloPresetName,
)

_REQUIRED_SECTIONS = (
    "run",
    "container",
    "taskset",
    "policy",
    "go_ex",
    "seed_candidate",
    "proposers",
)


def test_sokoban_smoke_materializes_gamebench_shape() -> None:
    cfg = GeloPreset.from_name("sokoban_smoke").materialize(
        container_url="http://127.0.0.1:8094",
        run_id="sokoban_smoke_test",
    )

    for section in _REQUIRED_SECTIONS:
        assert isinstance(cfg.get(section), dict), f"missing section {section}"

    assert cfg["run"]["run_id"] == "sokoban_smoke_test"
    assert cfg["container"]["url"] == "http://127.0.0.1:8094"

    taskset = cfg["taskset"]
    assert taskset["profile"] == "sokoban_singleplayer_agent"
    assert taskset["reward_mode"] == "sokoban_sparse_shaped"
    assert taskset["checkpoint_semantics"] == "true_environment_snapshot"
    # GameBench Sokoban seed ranges (proven config), not crafter's 3001/7001.
    assert taskset["train_seeds"] == [101, 102, 103, 104]
    assert taskset["heldout_seeds"] == [201, 202]
    assert taskset["env_config"]["task_path"].endswith("gold_08_double_push.json")
    ladder = [m["milestone_id"] for m in taskset["context"]["milestone_ladder"]]
    assert ladder == ["first_push", "box_on_target", "level_complete"]

    policy = cfg["policy"]
    assert policy["model"] == "openai/gpt-oss-120b"
    assert policy["provider"] == "groq"
    assert policy["api_key_env"] == "GROQ_API_KEY"

    go_ex = cfg["go_ex"]
    assert go_ex["max_rollouts"] > 0
    assert go_ex["proposer_rounds"] > 0
    assert go_ex["max_actions_per_turn"] == 1  # Sokoban is one action/turn

    # Proposers are Synth-managed (same roles as crafter_smoke).
    assert "core_proposer" in cfg["proposers"]


def test_sokoban_smoke_accepts_seed_overrides() -> None:
    cfg = GeloPreset.from_name(
        "sokoban_smoke", train_seed_count=8, heldout_seed_count=4
    ).materialize(container_url="http://127.0.0.1:8094", run_id="r")
    assert cfg["taskset"]["train_seeds"] == [101, 102, 103, 104, 105, 106, 107, 108]
    assert cfg["taskset"]["heldout_seeds"] == [201, 202, 203, 204]


def test_sokoban_smoke_requires_container_target() -> None:
    with pytest.raises(GeloMaterializeError):
        GeloPreset.from_name("sokoban_smoke").materialize(run_id="r")


def test_crafter_gamebench_smoke_materializes_deepseek_shape() -> None:
    cfg = GeloPreset.from_name("crafter_gamebench_smoke").materialize(
        container_url="http://127.0.0.1:8096",
        run_id="crafter_gamebench_test",
    )
    for section in _REQUIRED_SECTIONS:
        assert isinstance(cfg.get(section), dict), f"missing section {section}"
    assert cfg["taskset"]["profile"] == "crafter_singleplayer_agent"
    assert cfg["taskset"]["env_config"]["task_path"].endswith("gc_collect_sapling.json")
    assert cfg["policy"]["model"] == "deepseek-v4-flash"
    assert cfg["policy"]["provider"] == "deepseek"
    assert cfg["proposers"]["core_proposer"]["backend"] == "deepseek_chat"
    assert cfg["proposers"]["core_proposer"]["model"] == "deepseek-v4-flash"
    assert cfg["go_ex"]["submission_mode"] == "async"


def test_crafter_smoke_unchanged() -> None:
    cfg = GeloPreset.from_name("crafter_smoke").materialize(
        container_url="http://x", run_id="r"
    )
    assert cfg["taskset"]["profile"] == "crafter_react"


def test_sokoban_smoke_is_registered_preset() -> None:
    assert GeloPresetName.SOKOBAN_SMOKE.value == "sokoban_smoke"
    assert "sokoban_smoke" in {name.value for name in GeloPresetName}
