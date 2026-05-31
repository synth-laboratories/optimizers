"""
Crafter GEPA dev container with achievement side info.

This is a small public Crafter-style fixture for exercising the GEPA container
contract locally. It preserves the useful shape of the Crafter task: a ReAct-ish
action policy, sparse achievement rewards, and rollout side information that
explains which achievements were unlocked.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from synth_containers import ActionableSideInfo, ObjectiveScore, RolloutResult

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "crafter.achievement_policy"
TASKSET_ID = "crafter_public_achievement_tasks"

POLICY_MODEL = os.environ.get("CRAFTER_POLICY_MODEL", "gemini-3.1-flash-lite")
POLICY_API_KEY_ENV = os.environ.get("CRAFTER_POLICY_API_KEY_ENV", "GEMINI_API_KEY")
POLICY_BASE_URL = os.environ.get(
    "CRAFTER_POLICY_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
MAX_TURNS = int(os.environ.get("CRAFTER_MAX_TURNS", "12"))
MIN_BATCH = int(os.environ.get("CRAFTER_MIN_BATCH", "1"))
MAX_BATCH = int(os.environ.get("CRAFTER_MAX_BATCH", "5"))

ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a Crafter survival-game policy. Each turn, choose exactly one action "
    "from the admissible action list. Return ONLY strict JSON: "
    '{"action": "<action_name>"}. Prioritize achievements: collect resources '
    "with do, place a table before crafting tools, craft pickaxes before mining "
    "harder materials, eat or sleep when survival is low, and avoid repeating "
    "actions that the latest observation says failed."
)

ROWS = [
    {
        "seed": 11,
        "split": "train",
        "example_id": "wood_to_table",
        "scenario": "forest_start",
        "objective": "Unlock collect_wood and place_table quickly.",
        "initial_state": {
            "biome": "forest",
            "inventory": {"wood": 0, "stone": 0, "coal": 0, "iron": 0},
            "nearby": ["tree", "grass", "cow"],
            "health": 9,
            "hunger": 7,
            "thirst": 8,
        },
        "achievement_targets": ["collect_wood", "place_table"],
    },
    {
        "seed": 12,
        "split": "train",
        "example_id": "stone_tools",
        "scenario": "rocky_start",
        "objective": "Build toward a stone pickaxe.",
        "initial_state": {
            "biome": "rocky",
            "inventory": {"wood": 2, "stone": 0, "coal": 0, "iron": 0},
            "nearby": ["stone", "tree", "water"],
            "health": 8,
            "hunger": 6,
            "thirst": 6,
        },
        "achievement_targets": ["collect_stone", "place_table", "make_wood_pickaxe"],
    },
    {
        "seed": 13,
        "split": "train",
        "example_id": "survival_recovery",
        "scenario": "low_survival",
        "objective": "Recover survival stats before crafting.",
        "initial_state": {
            "biome": "plains",
            "inventory": {"wood": 1, "stone": 0, "coal": 0, "iron": 0},
            "nearby": ["cow", "water", "tree"],
            "health": 5,
            "hunger": 2,
            "thirst": 2,
        },
        "achievement_targets": ["eat_cow", "drink_water", "collect_wood"],
    },
    {
        "seed": 101,
        "split": "test",
        "example_id": "coal_progression",
        "scenario": "cave_edge",
        "objective": "Prepare to collect coal safely.",
        "initial_state": {
            "biome": "cave_edge",
            "inventory": {"wood": 2, "stone": 1, "coal": 0, "iron": 0},
            "nearby": ["coal", "stone", "tree"],
            "health": 7,
            "hunger": 5,
            "thirst": 5,
        },
        "achievement_targets": ["place_table", "make_wood_pickaxe", "collect_coal"],
    },
    {
        "seed": 102,
        "split": "test",
        "example_id": "iron_setup",
        "scenario": "mountain_start",
        "objective": "Set up tools and furnace for iron progression.",
        "initial_state": {
            "biome": "mountain",
            "inventory": {"wood": 3, "stone": 2, "coal": 1, "iron": 0},
            "nearby": ["iron", "stone", "tree"],
            "health": 8,
            "hunger": 6,
            "thirst": 4,
        },
        "achievement_targets": ["place_table", "place_furnace", "collect_iron"],
    },
]

_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; container deps in pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    api_key = os.environ.get(POLICY_API_KEY_ENV)
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"{POLICY_API_KEY_ENV} not set in container env; cannot serve live rollouts.",
        )
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if POLICY_BASE_URL:
        client_kwargs["base_url"] = POLICY_BASE_URL
    _openai_client = OpenAI(**client_kwargs)
    return _openai_client


def _render_state(
    state: dict[str, Any], achievements: list[str], row: dict[str, Any], last_event: str
) -> str:
    return (
        f"scenario: {row['scenario']}\n"
        f"objective: {row['objective']}\n"
        f"biome: {state['biome']}\n"
        f"inventory: {json.dumps(state['inventory'], sort_keys=True)}\n"
        f"nearby: {', '.join(state['nearby'])}\n"
        f"health: {state['health']} hunger: {state['hunger']} thirst: {state['thirst']}\n"
        f"unlocked_achievements: {achievements or []}\n"
        f"target_achievements: {row['achievement_targets']}\n"
        f"last_event: {last_event}\n"
        f"admissible_actions: {ACTION_NAMES}"
    )


def _parse_action(raw_text: str) -> str | None:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidate = parsed.get("action")
            if isinstance(candidate, str) and candidate.lower() in ACTION_NAMES:
                return candidate.lower()
        if isinstance(parsed, str) and parsed.lower() in ACTION_NAMES:
            return parsed.lower()
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    if lowered in ACTION_NAMES:
        return lowered
    for action in ACTION_NAMES:
        if action in lowered:
            return action
    return None


def _policy_step(
    client: Any, system_prompt: str, observation_text: str, turn: int
) -> tuple[str | None, dict[str, int], str]:
    user_content = (
        f"Turn {turn + 1}. Current Crafter state:\n\n{observation_text}\n\n"
        'Reply with strict JSON: {"action": "<one admissible action name>"}'
    )
    response = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    text = (response.choices[0].message.content or "").strip()
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    return _parse_action(text), usage, text


def _unlock(achievements: list[str], name: str) -> bool:
    if name in achievements:
        return False
    achievements.append(name)
    return True


def _apply_action(
    state: dict[str, Any], achievements: list[str], action: str
) -> tuple[str, list[str]]:
    inventory = state["inventory"]
    nearby = set(state["nearby"])
    unlocked: list[str] = []

    def add(name: str) -> None:
        if _unlock(achievements, name):
            unlocked.append(name)

    if action == "do":
        if "tree" in nearby:
            inventory["wood"] += 1
            add("collect_wood")
            return "cut tree and collected wood", unlocked
        if "stone" in nearby and inventory.get("wood", 0) > 0:
            inventory["stone"] += 1
            add("collect_stone")
            return "mined stone", unlocked
        if "coal" in nearby and "make_wood_pickaxe" in achievements:
            inventory["coal"] += 1
            add("collect_coal")
            return "mined coal", unlocked
        if "iron" in nearby and "make_stone_pickaxe" in achievements:
            inventory["iron"] += 1
            add("collect_iron")
            return "mined iron", unlocked
        if "cow" in nearby:
            state["hunger"] = min(9, int(state["hunger"]) + 3)
            add("eat_cow")
            return "hunted cow and ate food", unlocked
        if "water" in nearby:
            state["thirst"] = min(9, int(state["thirst"]) + 3)
            add("drink_water")
            return "drank water", unlocked
        return "do had no useful target", unlocked

    if action == "place_table":
        if inventory.get("wood", 0) >= 2:
            inventory["wood"] -= 2
            add("place_table")
            return "placed table", unlocked
        return "not enough wood to place table", unlocked
    if action == "make_wood_pickaxe":
        if "place_table" in achievements and inventory.get("wood", 0) >= 1:
            inventory["wood"] -= 1
            add("make_wood_pickaxe")
            return "crafted wood pickaxe", unlocked
        return "wood pickaxe requires table and wood", unlocked
    if action == "place_furnace":
        if inventory.get("stone", 0) >= 2:
            inventory["stone"] -= 2
            add("place_furnace")
            return "placed furnace", unlocked
        return "not enough stone to place furnace", unlocked
    if action == "make_stone_pickaxe":
        if (
            "place_table" in achievements
            and inventory.get("stone", 0) >= 2
            and inventory.get("wood", 0) >= 1
        ):
            inventory["stone"] -= 2
            inventory["wood"] -= 1
            add("make_stone_pickaxe")
            return "crafted stone pickaxe", unlocked
        return "stone pickaxe requires table, stone, and wood", unlocked
    if action == "sleep":
        state["health"] = min(9, int(state["health"]) + 1)
        return "slept and recovered health", unlocked
    if action.startswith("move_"):
        return (
            f"moved {action.removeprefix('move_')}; nearby resources unchanged in this fixture",
            unlocked,
        )
    if action == "place_plant":
        add("place_plant")
        return "placed plant", unlocked
    return f"{action} did not unlock an achievement", unlocked


def _run_episode(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    client = _get_openai_client()
    state = json.loads(json.dumps(row["initial_state"]))
    achievements: list[str] = []
    events: list[dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_event = "start"
    invalid_actions = 0

    for turn in range(MAX_TURNS):
        observation = _render_state(state, achievements, row, last_event)
        action, usage, raw_text = _policy_step(client, system_prompt, observation, turn)
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
        if action is None:
            invalid_actions += 1
            last_event = "invalid JSON or unknown action"
            events.append(
                {
                    "type": "policy_action",
                    "turn": turn,
                    "raw": raw_text,
                    "action": None,
                    "event": last_event,
                    "new_achievements": [],
                }
            )
            continue
        last_event, new_achievements = _apply_action(state, achievements, action)
        events.append(
            {
                "type": "policy_action",
                "turn": turn,
                "action": action,
                "event": last_event,
                "new_achievements": new_achievements,
                "achievements": list(achievements),
            }
        )
        if set(row["achievement_targets"]).issubset(set(achievements)):
            break

    target_set = set(row["achievement_targets"])
    unlocked_targets = sorted(target_set.intersection(achievements))
    reward = len(unlocked_targets) / max(1, len(target_set))
    return {
        "reward": reward,
        "achievements": sorted(achievements),
        "target_achievements": list(row["achievement_targets"]),
        "unlocked_target_achievements": unlocked_targets,
        "missing_achievements": sorted(target_set.difference(achievements)),
        "final_state": state,
        "events": events,
        "turn_count": len(events),
        "invalid_actions": invalid_actions,
        "usage": total_usage,
    }


app = FastAPI(title="crafter-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "crafter_gepa_dev",
            "name": "Crafter GEPA dev container",
            "description": "Crafter-style achievement policy optimization with achievement side info.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking", "async"],
            "metadata": {
                "trace_schema": "prompt_calls.llm_request.messages.v1",
                "side_info": "actionable_side_info.achievements",
            },
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
                    "rollout_route": "/rollout",
                }
            }
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "Crafter achievement policy",
            "description": "Optimize a system prompt for a Crafter-style action policy.",
            "objective": "Unlock target achievements by choosing valid Crafter actions from text observations.",
            "domain": "survival crafting, sparse achievement rewards, discrete actions",
        },
        "taskset": {
            "taskset_id": TASKSET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "task_id_semantics": "Task IDs select deterministic Crafter-style achievement scenarios.",
        },
        "prompt_program": {
            "mutable_modules": ["react_system_prompt"],
            "candidate_field": "react_system_prompt",
            "output_contract": 'Every policy call must return strict JSON: {"action": "<admissible action>"}.',
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "reward_definition": "fraction of target achievements unlocked",
            "side_info": "Rollouts expose actionable_side_info.achievements for proposer reflection.",
            "rollout_trace_contains": ["policy_action", "new_achievements", "missing_achievements"],
        },
        "proposal_guidance": {
            "premises": [
                "Achievements are sparse and usually require preconditions.",
                "The observation includes inventory, nearby resources, survival stats, targets, and last event.",
                "The policy must return one valid action as JSON every turn.",
            ],
            "high_leverage_heuristics": [
                "Collect wood before table/tool plans.",
                "Place a table before tool crafting.",
                "Use do on nearby trees, stone, coal, iron, cows, or water when preconditions match.",
                "Recover hunger/thirst when survival stats are low.",
                "Use missing achievements and failed last events to change strategy.",
            ],
            "anti_patterns": [
                "Verbose non-JSON responses.",
                "Trying to craft tools before placing a table.",
                "Repeating actions after the observation says they failed.",
            ],
        },
        "metadata": {
            "policy_model": POLICY_MODEL,
            "max_turns": MAX_TURNS,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "crafter_react_policy_gepa",
        "modules": [
            {
                "module_id": "react_system_prompt",
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "react_system_prompt",
                "template_variables": [],
                "metadata": {"task_id": TASK_ID},
            }
        ],
        "target_modules": [
            {
                "module_id": "react_system_prompt",
                "candidate_field": "react_system_prompt",
                "objective": "achievement_unlock_rate",
            }
        ],
        "seed_candidate": {"react_system_prompt": DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["react_system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "taskset_id": TASKSET_ID,
            "side_info": "actionable_side_info.achievements",
        },
    }


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    return {
        "taskset_id": TASKSET_ID,
        "splits": {
            "train": sum(1 for row in ROWS if row["split"] == "train"),
            "test": sum(1 for row in ROWS if row["split"] == "test"),
        },
        "source": "crafter_public_achievement_fixture",
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = [str(task_id) for task_id in payload.get("task_ids") or []]
    return {
        "tasks": [_public_row(_task_for_id(split=split, task_id=task_id)) for task_id in task_ids]
    }


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    incoming_task = payload.get("task") if isinstance(payload.get("task"), dict) else None
    task_id = str((incoming_task or {}).get("task_id") or payload.get("task_id") or "train:11")
    split_hint = str((incoming_task or {}).get("split") or payload.get("split") or "train")
    row = _task_for_id(split=split_hint, task_id=task_id)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("react_system_prompt") or DEFAULT_SYSTEM_PROMPT)

    episode = _run_episode(row, system_prompt)
    reward = float(episode["reward"])
    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    side_info = {
        "achievements": episode["achievements"],
        "target_achievements": episode["target_achievements"],
        "unlocked_target_achievements": episode["unlocked_target_achievements"],
        "missing_achievements": episode["missing_achievements"],
        "achievement_count": len(episode["achievements"]),
        "turn_count": episode["turn_count"],
        "invalid_actions": episode["invalid_actions"],
        "last_events": episode["events"][-5:],
    }
    return RolloutResult(
        rollout_id=rollout_id,
        reward=reward,
        task_id=row["task_id"],
        success_status="succeeded" if reward >= 1.0 else "failed",
        objective="achievement_unlock_rate",
        objective_scores=[
            ObjectiveScore(
                objective="achievement_unlock_rate",
                value=reward,
                source="crafter.achievements",
                metadata={"target_count": len(episode["target_achievements"])},
            ),
            ObjectiveScore(
                objective="turn_count",
                value=float(episode["turn_count"]),
                source="crafter.rollout",
                metadata={"max_turns": MAX_TURNS},
            ),
        ],
        reward_details={
            "example_id": row["example_id"],
            "achievements": episode["achievements"],
            "target_achievements": episode["target_achievements"],
            "missing_achievements": episode["missing_achievements"],
            "policy_model": POLICY_MODEL,
        },
        summary={
            "outcome_reward": reward,
            "example_id": row["example_id"],
            "achievements": episode["achievements"],
            "missing_achievements": episode["missing_achievements"],
            "turn_count": episode["turn_count"],
        },
        usage={**episode["usage"], "cost_usd": 0.0},
        actionable_side_info=ActionableSideInfo(side_info),
        trace={
            "event_history": episode["events"],
            "metadata": {
                "example_id": row["example_id"],
                "call_site_id": "crafter.achievement_policy",
                "achievements": episode["achievements"],
            },
        },
        metadata={
            "candidate": candidate,
            "final_state": episode["final_state"],
        },
    ).to_dict()


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "rng_seed": row["seed"],
        "split": row["split"],
        "example_id": row["example_id"],
        "scenario": row["scenario"],
        "objective": row["objective"],
        "initial_state": row["initial_state"],
        "achievement_targets": row["achievement_targets"],
    }


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    normalized_split = "test" if split in {"heldout", "test", "validation", "val"} else "train"
    rows = [row for row in ROWS if row["split"] == normalized_split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    row = match or rows[int(seed) % len(rows)]
    task = json.loads(json.dumps(row))
    task["task_id"] = f"{task['split']}:{task['seed']}"
    return task


def _task_for_id(*, split: str, task_id: str) -> dict[str, Any]:
    normalized_split = "test" if split in {"heldout", "test", "validation", "val"} else "train"
    prefix, separator, raw_seed = task_id.partition(":")
    if separator and prefix in {"train", "test"} and raw_seed.isdigit():
        return _row_for_seed(split=normalized_split, seed=int(raw_seed))
    rows = [
        _row_for_seed(split=normalized_split, seed=int(row["seed"]))
        for row in ROWS
        if row["split"] == normalized_split
    ]
    match = next((row for row in rows if row["task_id"] == task_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"unknown_task_id:{normalized_split}:{task_id}")
    return match


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
