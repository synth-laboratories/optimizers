from __future__ import annotations

import difflib
import os

import httpx
import pytest

from craftax_levers.apply import apply_unified_diff, sha256_text
from craftax_levers.orchestrator_app import diff_seed_to_greedy_policy
from craftax_levers.seeds import (
    GREEDY_POLICY,
    SEED_HARNESS,
    SEED_POLICY,
    SPEEDRUNNER_HARNESS,
    WOOD_PROMPT,
    make_speedrunner_harness,
)
from craftax_levers.submit import rollout_request
from craftax_levers.stack import start_stack
from craftax_levers.world import reset, step


def _schemas(side_info: list[dict]) -> set[str]:
    return {str(item.get("schema_id")) for item in side_info}


needs_openai = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")),
    reason="OPENROUTER_API_KEY or OPENAI_API_KEY required for ReAct rollouts",
)


@pytest.fixture(scope="module")
def code_stack():
    stack = start_stack("code")
    yield stack
    stack.stop()


@pytest.fixture(scope="module")
def react_stack():
    stack = start_stack("react")
    yield stack
    stack.stop()


def test_world_greedy_collects_on_seed_0() -> None:
    world, obs = reset(0)
    assert obs["adjacent"]["wood"] is True
    obs, reward, done, info = step(world, "collect")
    assert reward == 1.0
    assert done is False
    assert info["outcome_reward"] == 1.0
    assert "collect_wood" in info["achievements"]
    obs, _, _, _ = step(world, "down")
    assert obs["adjacent"]["wood"] is True
    obs, reward, done, info = step(world, "collect")
    assert reward == 1.0
    assert done is True
    assert info["outcome_reward"] == 2.0


def test_unified_diff_roundtrip() -> None:
    diff = "".join(
        difflib.unified_diff(
            SEED_POLICY.splitlines(keepends=True),
            GREEDY_POLICY.splitlines(keepends=True),
            fromfile="a/policy.py",
            tofile="b/policy.py",
        )
    )
    assert apply_unified_diff(SEED_POLICY, diff) == GREEDY_POLICY


def test_code_program_advertises_policy_script(code_stack) -> None:
    program = httpx.get(f"{code_stack.orch_url}/program", timeout=5.0).json()
    module = program["modules"][0]
    assert module["candidate_field"] == "policy_script"
    assert module["metadata"]["lever_kind"] == "policy_script"
    assert module["metadata"]["protocol_id"] == "whole_file.v1"
    assert module["metadata"]["constraints"]["entrypoint"] == "act"
    assert "code_policy_game_trace.v1" in {row["schema_id"] for row in program["side_info_schemas"]}
    health = httpx.get(f"{code_stack.orch_url}/health", timeout=5.0).json()
    assert health["env"]["version"] == "craftax_env.v1"
    assert health["policy"]["policy_kind"] == "code_policy.v1"


def test_code_seed_noop_zero_reward(code_stack) -> None:
    body = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {}},
        timeout=10.0,
    ).json()
    assert body["status"] == "completed"
    assert body["reward"] == 0.0
    assert body["reward_info"]["outcome_reward"] == 0.0
    assert body["summary"]["outcome_reward"] == 0.0
    assert body["trace"]["event_history"]
    assert "code_policy_game_trace.v1" in _schemas(body["side_info"])
    assert body["actionable_side_info"] == body["side_info"]
    game = next(item for item in body["side_info"] if item["schema_id"] == "code_policy_game_trace.v1")
    assert game["summary"]["compile_ok"] is True
    assert game["summary"]["achievements"] == []


def test_code_whole_file_greedy_reward_and_asi(code_stack) -> None:
    body = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"policy_script": GREEDY_POLICY}},
        timeout=10.0,
    ).json()
    assert body["reward"] == 2.0
    assert body["reward_info"]["metrics"]["achievements"] == ["collect_wood"]
    game = next(item for item in body["side_info"] if item["schema_id"] == "code_policy_game_trace.v1")
    assert game["summary"]["compile_ok"] is True
    assert game["summary"]["achievements"] == ["collect_wood"]
    assert game["summary"]["ticks"] >= 1
    assert any(event.get("type") == "policy_act" for event in body["trace"]["event_history"])


def test_code_unified_diff_and_hash_reject(code_stack) -> None:
    reset = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"policy_script": SEED_POLICY}},
        timeout=10.0,
    ).json()
    assert reset["reward"] == 0.0
    diff = diff_seed_to_greedy_policy()
    ok = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={
            "task_id": "train:0",
            "lever_bundle": {
                "schema_version": "lever_bundle.v1",
                "values": {
                    "policy_script": {
                        "protocol_id": "unified_diff.v1",
                        "path": "policy.py",
                        "diff": diff,
                        "base_hash": sha256_text(SEED_POLICY),
                    }
                },
            },
        },
        timeout=10.0,
    ).json()
    assert ok["reward"] == 2.0
    assert "apply_report.v1" in _schemas(ok["side_info"])

    rejected = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={
            "task_id": "train:0",
            "lever_bundle": {
                "values": {
                    "policy_script": {
                        "path": "policy.py",
                        "content": GREEDY_POLICY,
                        "content_hash": "deadbeef",
                    }
                }
            },
        },
        timeout=10.0,
    ).json()
    assert rejected["success_status"] == "apply_failed"
    assert rejected["reward"] == 0.0
    report = next(item for item in rejected["side_info"] if item.get("reject_reason") == "content_hash_mismatch")
    assert report["patch_ok"] is False


def test_code_register_then_rollout_on_demand(code_stack) -> None:
    registered = httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "greedy_v1", "candidate": {"policy_script": GREEDY_POLICY}},
        timeout=10.0,
    ).json()
    assert registered["apply_ok"] is True
    assert registered["candidate_id"] == "greedy_v1"
    first = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"candidate_id": "greedy_v1", "task_id": "train:0"},
        timeout=10.0,
    ).json()
    second = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"candidate_id": "greedy_v1", "task_id": "train:1"},
        timeout=10.0,
    ).json()
    assert first["candidate_id"] == "greedy_v1"
    assert first["reward"] == 2.0
    assert second["candidate_id"] == "greedy_v1"
    assert first["rollout_id"] != second["rollout_id"]
    fetched = httpx.get(f"{code_stack.orch_url}/rollouts/{first['rollout_id']}", timeout=5.0).json()
    assert fetched["reward"] == 2.0
    seed = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"candidate_id": "seed", "task_id": "train:0"},
        timeout=10.0,
    ).json()
    assert seed["reward"] == 0.0
    meta = httpx.get(f"{code_stack.orch_url}/metadata", timeout=5.0).json()
    assert meta["metadata"]["optimizer_contracts"]["gepa"]["candidates_route"] == "/candidates"
    assert meta["capabilities"]["metadata"]["policy_ready"] is True


def test_code_compile_failure_asi(code_stack) -> None:
    body = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"policy_script": "def nope(obs):\n    return 'noop'\n"}},
        timeout=10.0,
    ).json()
    assert body["success_status"] == "apply_failed"
    assert body["reward"] == 0.0
    report = next(item for item in body["side_info"] if item.get("compile_ok") is False)
    assert "act" in str(report.get("reject_reason") or "").lower() or report["compile_ok"] is False
    restored = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"policy_script": SEED_POLICY}},
        timeout=10.0,
    ).json()
    assert restored["reward"] == 0.0
    assert restored["success_status"] == "succeeded"


def test_react_register_prompt_overlay_without_episode(react_stack) -> None:
    registered = httpx.post(
        f"{react_stack.orch_url}/candidates",
        json={
            "candidate_id": "prompt_v1",
            "candidate": {"react_system_prompt": "Collect adjacent wood."},
        },
        timeout=10.0,
    ).json()
    assert registered["apply_ok"] is True
    assert registered["candidate_id"] == "prompt_v1"
    listed = httpx.get(f"{react_stack.orch_url}/candidates/prompt_v1", timeout=5.0).json()
    assert listed["apply_ok"] is True
    meta = httpx.get(f"{react_stack.orch_url}/metadata", timeout=5.0).json()
    assert meta["metadata"]["optimizer_contracts"]["gepa"]["candidates_route"] == "/candidates"


def test_react_program_advertises_harness(react_stack) -> None:
    program = httpx.get(f"{react_stack.orch_url}/program", timeout=5.0).json()
    fields = {module["candidate_field"] for module in program["modules"]}
    assert fields == {"react_system_prompt", "harness_module"}
    harness_mod = next(m for m in program["modules"] if m["candidate_field"] == "harness_module")
    assert harness_mod["metadata"]["constraints"]["entrypoint"] == "run_episode"
    assert "run_episode" in program["seed_candidate"]["harness_module"]
    kinds = {module["metadata"]["lever_kind"] for module in program["modules"]}
    assert "harness_module" in kinds
    assert "system_prompt" in kinds


@needs_openai
def test_react_program_and_prompt_overlay(react_stack) -> None:
    seed = httpx.post(
        f"{react_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"react_system_prompt": "Wander randomly."}},
        timeout=180.0,
    ).json()
    assert seed["success_status"] == "succeeded"
    assert "harness_v5_trace.v1" in _schemas(seed["side_info"])
    assert "prompt_trace.v1" in _schemas(seed["side_info"])
    harness = next(item for item in seed["side_info"] if item["schema_id"] == "harness_v5_trace.v1")
    assert harness["summary"]["tool_calls"] >= 1
    assert harness["summary"].get("llm_provider") in {"openai", "openrouter"}
    assert any(event.get("type") == "llm_request" for event in seed["trace"]["event_history"])

    improved = httpx.post(
        f"{react_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate": {"react_system_prompt": WOOD_PROMPT}},
        timeout=180.0,
    ).json()
    assert improved["success_status"] == "succeeded"
    assert improved["actionable_side_info"] == improved["side_info"]
    assert any(event.get("type") == "llm_request" for event in improved["trace"]["event_history"])
    summary = next(item["summary"] for item in improved["side_info"] if item["schema_id"] == "harness_v5_trace.v1")
    assert summary.get("llm_provider") in {"openai", "openrouter"}


@needs_openai
def test_react_harness_restart(react_stack) -> None:
    env_before = httpx.get(f"{react_stack.env_url}/health", timeout=5.0).json()
    policy_before = httpx.get(f"{react_stack.policy_url}/health", timeout=5.0).json()
    restarted = httpx.post(
        f"{react_stack.orch_url}/rollout",
        json=rollout_request("train:0", prompt=WOOD_PROMPT, script=SEED_HARNESS),
        timeout=180.0,
    ).json()
    assert restarted["success_status"] == "succeeded"
    report = next(
        item
        for item in restarted["side_info"]
        if "harness_module" in (item.get("lever_ids") or []) and item.get("schema_id") == "apply_report.v1"
    )
    assert report.get("restart_ok") is True
    assert report.get("env_untouched") is True
    assert report.get("policy_pid_after") != policy_before["pid"]
    env_after = httpx.get(f"{react_stack.env_url}/health", timeout=5.0).json()
    assert env_after["pid"] == env_before["pid"]
    harness = next(item for item in restarted["side_info"] if item["schema_id"] == "harness_v5_trace.v1")
    assert harness["summary"]["architecture"] == "react_llm_thought_action"
    assert harness["summary"]["llm_calls"] >= 1
    assert harness["summary"].get("llm_provider") in {"openai", "openrouter"}
    types = {event.get("type") for event in restarted["trace"]["event_history"]}
    assert "llm_request" in types
    assert "tool_call" in types
    llm_events = [event for event in restarted["trace"]["event_history"] if event.get("type") == "llm_request"]
    tool_events = [event for event in restarted["trace"]["event_history"] if event.get("type") == "tool_call"]
    assert llm_events
    assert tool_events
    parsed = None
    text = str(llm_events[0].get("response") or "")
    for action in ("collect", "up", "down", "left", "right", "noop"):
        if f"action: {action}" in text.lower():
            parsed = action
            break
    assert tool_events[0].get("action") == (parsed or "noop")


@needs_openai
def test_react_speedrunner_skill_library_restart(react_stack) -> None:
    env_before = httpx.get(f"{react_stack.env_url}/health", timeout=5.0).json()
    policy_before = httpx.get(f"{react_stack.policy_url}/health", timeout=5.0).json()
    body = httpx.post(
        f"{react_stack.orch_url}/rollout",
        json=rollout_request("train:0", script=SPEEDRUNNER_HARNESS),
        timeout=180.0,
    ).json()
    assert body["success_status"] == "succeeded"
    report = next(
        item
        for item in body["side_info"]
        if "harness_module" in (item.get("lever_ids") or []) and item.get("schema_id") == "apply_report.v1"
    )
    assert report.get("protocol_id") == "harness_restart.v1"
    assert report.get("restart_ok") is True
    assert report.get("env_untouched") is True
    assert report.get("policy_pid_after") != policy_before["pid"]
    env_after = httpx.get(f"{react_stack.env_url}/health", timeout=5.0).json()
    assert env_after["pid"] == env_before["pid"]
    harness = next(item for item in body["side_info"] if item["schema_id"] == "harness_v5_trace.v1")
    assert harness["summary"]["architecture"] == "react_llm_speedrunner"
    assert harness["summary"]["llm_calls"] >= 1
    events = body["trace"]["event_history"]
    assert any(event.get("type") == "llm_request" for event in events)
    skills = [event for event in events if event.get("type") == "skill_invoke"]
    primitives = [event for event in events if event.get("type") == "skill_primitive"]
    if skills:
        assert skills[0].get("skill") == "collect_all_wood"
        assert [event.get("action") for event in primitives] == ["collect", "down", "collect"]
        assert body["reward"] == 2.0
        assert harness["summary"]["llm_calls"] == 1
    inspect = httpx.get(f"{react_stack.policy_url}/inspect", timeout=5.0).json()
    assert inspect["architecture"] == "react_llm_speedrunner"
    names = {row["name"] for row in inspect["functions"]}
    assert {"collect_all_wood", "run_episode", "parse_choice"} <= names


def test_react_live_instantiate_and_inspect(react_stack) -> None:
    seeded = httpx.post(
        f"{react_stack.policy_url}/load",
        json={"source": SEED_HARNESS},
        timeout=5.0,
    ).json()
    assert seeded["ok"] is True
    before = httpx.get(f"{react_stack.policy_url}/inspect", timeout=5.0).json()
    assert before["instantiated"] is True
    assert before["requires_llm"] is True
    assert before["react_llm"] is True
    assert before["architecture"] == "react_llm_thought_action"
    assert before["llm_call_sites"] >= 1
    live_names = {row["name"] for row in before["live_callables"]}
    assert {"run_episode", "parse_action", "format_obs"} <= live_names
    run = next(row for row in before["live_callables"] if row["name"] == "run_episode")
    assert "llm" in run["signature"]
    proxied = httpx.get(f"{react_stack.orch_url}/inspect", timeout=5.0).json()
    assert proxied["architecture"] == "react_llm_thought_action"
    assert proxied["pid"] == before["pid"]


def test_inspect_source_react_llm_without_stack() -> None:
    from craftax_levers.inspect_script import inspect_source

    seed = inspect_source(SEED_HARNESS)
    assert seed["architecture"] == "react_llm_thought_action"
    assert seed["requires_llm"] is True
    assert seed["react_llm"] is True
    assert seed["llm_call_sites"] >= 1
    assert "llm" in (next(row["args"] for row in seed["functions"] if row["name"] == "run_episode"))
    speed = inspect_source(SPEEDRUNNER_HARNESS)
    assert speed["architecture"] == "react_llm_speedrunner"
    assert speed["requires_llm"] is True
    assert speed["llm_call_sites"] == 1
    names = {row["name"] for row in speed["functions"]}
    assert "collect_all_wood" in names
    lava = inspect_source(make_speedrunner_harness(["walk_into_lava"]))
    lava_names = {row["name"] for row in lava["functions"]}
    assert "walk_into_lava" in lava_names
    assert "collect_all_wood" not in lava_names
    mixed = inspect_source(
        make_speedrunner_harness(
            ["stand_still", "walk_into_lava", "collect_adjacent_once", "collect_all_wood"]
        )
    )
    mixed_names = {row["name"] for row in mixed["functions"]}
    assert {"stand_still", "walk_into_lava", "collect_adjacent_once", "collect_all_wood"} <= mixed_names
