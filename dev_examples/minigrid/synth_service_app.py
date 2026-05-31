"""
MiniGrid GEPA cookbook container (live gymnasium env, OpenAI-compatible policy).

Speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /taskset
  POST /taskset/tasks
  POST /rollout

Each rollout instantiates a real MiniGrid env, drives it for up to N steps
with an LLM-driven agent using the candidate's `system_prompt`, and returns
the actual env reward.

Required env:
  OPENAI_API_KEY, OPENROUTER_API_KEY, or DEEPSEEK_API_KEY — one must be set.
  MINIGRID_POLICY_MODEL       — default: gpt-4.1-nano
  MINIGRID_POLICY_API_KEY_ENV — optional; explicit policy key env var name.
  MINIGRID_POLICY_BASE_URL    — optional; set to https://openrouter.ai/api/v1
                                for OpenRouter, https://api.deepseek.com for
                                DeepSeek, or omit for OpenAI.
  MINIGRID_MAX_STEPS          — default: 48 (per-episode hard cap)
  MINIGRID_ENV_ID             — default: MiniGrid-DoorKey-5x5-v0
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


TASK_ID = "minigrid.gridworld_policy"
TASKSET_ID = "minigrid_public_episodes"

POLICY_MODEL = os.environ.get("MINIGRID_POLICY_MODEL", "gpt-4.1-nano")
POLICY_API_KEY_ENV = os.environ.get("MINIGRID_POLICY_API_KEY_ENV")
POLICY_BASE_URL = os.environ.get("MINIGRID_POLICY_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MAX_STEPS = int(os.environ.get("MINIGRID_MAX_STEPS", "48"))
ENV_ID = os.environ.get("MINIGRID_ENV_ID", "MiniGrid-DoorKey-5x5-v0")

# Standard MiniGrid 7 actions, ordered by gymnasium Action enum.
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a MiniGrid navigation policy. Each turn you see your position, "
    "facing direction, the visible goal/key/door objects, and admissible actions. "
    'Respond ONLY with strict JSON of the form: {"action": "<one admissible action name>"}. '
    "For navigation-only missions, move toward the green goal using the provided "
    "relative_goal hint: if the goal is east/west, face right/left; if it is "
    "south/north, face down/up; move forward only when facing a direction that "
    "reduces distance and the tile ahead is not blocked. If blocked or facing "
    "away from the target, turn left or right to reduce the angular error. For "
    "door/key missions, pick up needed keys, face locked doors before toggle, "
    "then navigate to the goal. Avoid repeating forward into a wall."
)

# Episode seeds. Train seeds drive GEPA's reflective loop; heldout gates final score.
ROWS = [
    {"seed": 1, "split": "train", "example_id": "ep_train_1"},
    {"seed": 2, "split": "train", "example_id": "ep_train_2"},
    {"seed": 3, "split": "train", "example_id": "ep_train_3"},
    {"seed": 4, "split": "train", "example_id": "ep_train_4"},
    {"seed": 100, "split": "test", "example_id": "ep_heldout_100"},
    {"seed": 101, "split": "test", "example_id": "ep_heldout_101"},
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
    key_env_candidates = []
    if POLICY_API_KEY_ENV:
        key_env_candidates.append(POLICY_API_KEY_ENV)
    if POLICY_BASE_URL and "deepseek" in POLICY_BASE_URL.lower():
        key_env_candidates.append("DEEPSEEK_API_KEY")
    key_env_candidates.extend(["OPENROUTER_API_KEY", "OPENAI_API_KEY"])

    api_key = None
    missing_envs = []
    for key_env in dict.fromkeys(key_env_candidates):
        value = os.environ.get(key_env)
        if value:
            api_key = value
            break
        missing_envs.append(key_env)
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No policy API key found in container env; checked {', '.join(missing_envs)}."
            ),
        )
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if POLICY_BASE_URL:
        client_kwargs["base_url"] = POLICY_BASE_URL
    _openai_client = OpenAI(**client_kwargs)
    return _openai_client


# --- Env helpers --------------------------------------------------------------


def _make_env(seed: int):
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers envs as side effect
    from minigrid.wrappers import FullyObsWrapper

    env = gym.make(ENV_ID, render_mode=None)
    env = FullyObsWrapper(env)
    obs, _ = env.reset(seed=int(seed))
    if hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = MAX_STEPS
    return env, obs


_DIRECTION_NAMES = {0: "right", 1: "down", 2: "left", 3: "up"}
_DIRECTION_VECTORS = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def _object_label(obj: Any) -> str:
    obj_type = str(getattr(obj, "type", "") or "")
    obj_color = str(getattr(obj, "color", "") or "")
    is_locked = bool(getattr(obj, "is_locked", False))
    is_open = bool(getattr(obj, "is_open", False))
    label_bits = [obj_color, obj_type]
    if is_locked:
        label_bits.append("(locked)")
    if is_open:
        label_bits.append("(open)")
    return " ".join(bit for bit in label_bits if bit)


def _direction_toward(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy) and dx != 0:
        return "right" if dx > 0 else "left"
    if dy != 0:
        return "down" if dy > 0 else "up"
    return "here"


def _render_observation_text(env, obs) -> str:
    mission = str(obs.get("mission") or getattr(env.unwrapped, "mission", "") or "")
    agent_pos = list(getattr(env.unwrapped, "agent_pos", []) or [])
    agent_dir = int(getattr(env.unwrapped, "agent_dir", 0) or 0)
    direction_name = _DIRECTION_NAMES.get(agent_dir, "?")
    carrying = getattr(env.unwrapped, "carrying", None)
    carrying_str = (
        f"{getattr(carrying, 'color', '')} {getattr(carrying, 'type', '')}".strip()
        if carrying is not None
        else "nothing"
    )

    grid = getattr(env.unwrapped, "grid", None)
    width = int(getattr(grid, "width", 0) or 0)
    height = int(getattr(grid, "height", 0) or 0)
    visible: list[str] = []
    goal_pos: tuple[int, int] | None = None
    front_label = "unknown"
    if grid is not None:
        dx_ahead, dy_ahead = _DIRECTION_VECTORS.get(agent_dir, (0, 0))
        front_x = int(agent_pos[0]) + dx_ahead if len(agent_pos) >= 2 else 0
        front_y = int(agent_pos[1]) + dy_ahead if len(agent_pos) >= 2 else 0
        if 0 <= front_x < width and 0 <= front_y < height:
            front_obj = grid.get(front_x, front_y)
            front_label = _object_label(front_obj) if front_obj is not None else "empty"
        for x in range(width):
            for y in range(height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = str(getattr(obj, "type", "") or "")
                if obj_type == "goal":
                    goal_pos = (x, y)
                visible.append(f"  ({x},{y}): {_object_label(obj)}")

    visible_block = "\n".join(visible[:25]) if visible else "  (none)"
    if goal_pos is not None and len(agent_pos) >= 2:
        goal_dx = goal_pos[0] - int(agent_pos[0])
        goal_dy = goal_pos[1] - int(agent_pos[1])
        goal_hint = (
            f"goal_position: {list(goal_pos)}  "
            f"relative_goal: dx={goal_dx}, dy={goal_dy}, "
            f"primary_direction={_direction_toward(goal_dx, goal_dy)}"
        )
    else:
        goal_hint = "goal_position: unknown  relative_goal: unknown"
    return (
        f"mission: {mission}\n"
        f"agent_position: {agent_pos}  facing: {direction_name}\n"
        f"{goal_hint}\n"
        f"front_cell: {front_label}\n"
        f"carrying: {carrying_str}\n"
        f"admissible_actions: {ACTION_NAMES}\n"
        f"visible_objects:\n{visible_block}"
    )


def _parse_action(raw_text: str) -> str | None:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            cand = parsed.get("action")
            if isinstance(cand, str) and cand.lower() in ACTION_NAMES:
                return cand.lower()
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


def _llm_step(
    client: Any, system_prompt: str, observation_text: str, step: int
) -> tuple[str | None, dict[str, int]]:
    user_content = (
        f"Step {step + 1}. Current state:\n\n{observation_text}\n\n"
        'Reply with strict JSON: {"action": "<one admissible action name>"}'
    )
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    action = _parse_action(text)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return action, usage


def _run_episode(seed: int, system_prompt: str) -> dict[str, Any]:
    client = _get_openai_client()
    env, obs = _make_env(seed)

    total_reward = 0.0
    all_actions: list[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = False
    step = 0

    try:
        while step < MAX_STEPS:
            obs_text = _render_observation_text(env, obs)
            action_name, usage = _llm_step(client, system_prompt, obs_text, step)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            if action_name is None:
                # Invalid response: count the turn, try again next step.
                step += 1
                continue
            action_idx = ACTION_NAMES.index(action_name)
            obs, reward, terminated, truncated, _info = env.step(action_idx)
            total_reward += float(reward)
            all_actions.append(action_name)
            step += 1
            done = bool(terminated or truncated)
            if done:
                break
    finally:
        env.close()

    return {
        "seed": seed,
        "n_steps": step,
        "total_reward": total_reward,
        "done": done,
        "solved": total_reward > 0.0,  # MiniGrid success → positive reward only on goal
        "actions": all_actions,
        "usage": total_usage,
    }


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="minigrid-gepa-container")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "minigrid_gepa_live",
            "name": "MiniGrid GEPA (live gymnasium env, OpenAI policy)",
            "description": "Public prompt-optimizer cookbook running real MiniGrid episodes with an OpenAI-driven agent.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking", "async"],
            "metadata": {"trace_schema": "prompt_calls.llm_request.messages.v1"},
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
            "name": f"MiniGrid policy ({ENV_ID})",
            "description": (
                "Optimize a system prompt for an OpenAI-controlled MiniGrid agent. "
                "Each rollout is a live gymnasium episode, not a fixture replay."
            ),
            "objective": "Maximize solved episodes and total environment reward before the step cap.",
            "domain": "partially observable gridworld control with text observations and discrete actions",
        },
        "taskset": {
            "taskset_id": TASKSET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(ROWS),
            "task_id_semantics": (
                "Rows are generated from requested episode task IDs. The same seed is deterministic "
                "for a given MiniGrid env id."
            ),
        },
        "prompt_program": {
            "mutable_modules": ["system_prompt"],
            "candidate_field": "system_prompt",
            "output_contract": 'Every policy call must return strict JSON: {"action": "<admissible action>"}.',
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "success_status": "succeeded when the episode reaches the mission goal",
            "rollout_trace_contains": ["episode_complete", "actions_taken", "n_steps", "solved"],
        },
        "proposal_guidance": {
            "premises": [
                "The agent receives mission, position, direction, carried object, visible objects, and admissible actions each turn.",
                "MiniGrid tasks often require short action plans: orient, move, pick up keys, toggle doors, then reach the goal.",
                "Invalid JSON or invalid action names waste the step budget and usually fail the episode.",
            ],
            "constraints": [
                "Do not ask for chain-of-thought or verbose plans in the final response.",
                "Do not introduce actions outside the seven MiniGrid action names.",
                "Keep the system prompt operational and compact enough to run on every step.",
            ],
            "high_leverage_heuristics": [
                "Prioritize mission progress over exploration once the goal object or door is visible.",
                "Use explicit door/key rules: pick up matching keys, face doors before toggle, avoid repeated no-op toggles.",
                "Add recovery behavior for blocked forward moves and loops.",
                "Make JSON compliance non-negotiable.",
            ],
            "anti_patterns": [
                "Generic assistant persona text.",
                "Long reflective reasoning instructions that increase latency without changing actions.",
                "Rules that ignore admissible actions or the current facing direction.",
            ],
        },
        "metadata": {
            "policy_model": POLICY_MODEL,
            "env_id": ENV_ID,
            "max_steps": MAX_STEPS,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "minigrid_system_prompt_gepa",
        "modules": [
            {
                "module_id": "system_prompt",
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "system_prompt",
                "template_variables": [],
                "metadata": {"env_id": ENV_ID},
            }
        ],
        "target_modules": [
            {
                "module_id": "system_prompt",
                "candidate_field": "system_prompt",
                "objective": "task_success_rate",
            }
        ],
        "seed_candidate": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "taskset_id": TASKSET_ID,
            "env_id": ENV_ID,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
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
        "source": "minigrid_public_episode_tasks",
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    task_ids = [str(task_id) for task_id in payload.get("task_ids") or []]
    return {"tasks": [_task_for_id(split=split, task_id=task_id) for task_id in task_ids]}


@app.post("/rollout")
@app.post("/rollouts")
def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    row = payload.get("task") if isinstance(payload.get("task"), dict) else None
    if not row:
        row = _task_for_id(
            split=str(payload.get("split") or "train"),
            task_id=str(payload.get("task_id") or "train:1"),
        )
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    seed = int(row.get("seed") or 0)

    episode = _run_episode(seed=seed, system_prompt=system_prompt)
    reward = float(episode["total_reward"])

    rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")
    return RolloutResult(
        rollout_id=rollout_id,
        reward=reward,
        task_id=str(row.get("task_id") or ""),
        success_status="succeeded" if episode["solved"] else "failed",
        objective="task_success",
        objective_scores=[
            ObjectiveScore(
                objective="task_success",
                value=1.0 if episode["solved"] else 0.0,
                source="minigrid.env",
                metadata={"env_id": ENV_ID},
            ),
            ObjectiveScore(
                objective="episode_reward",
                value=reward,
                source="minigrid.env",
                metadata={"env_id": ENV_ID},
            ),
            ObjectiveScore(
                objective="episode_steps",
                value=float(episode["n_steps"]),
                source="minigrid.env",
                metadata={"max_steps": MAX_STEPS},
            ),
        ],
        reward_details={
            "example_id": row.get("example_id"),
            "env_id": ENV_ID,
            "n_steps": episode["n_steps"],
            "done": episode["done"],
            "solved": episode["solved"],
            "policy_model": POLICY_MODEL,
        },
        summary={
            "outcome_reward": reward,
            "example_id": row.get("example_id"),
            "n_steps": episode["n_steps"],
            "actions_taken": episode["actions"],
        },
        usage={**episode["usage"], "cost_usd": 0.0},
        actionable_side_info=ActionableSideInfo(
            {
                "solved": episode["solved"],
                "n_steps": episode["n_steps"],
                "actions_taken": episode["actions"],
                "env_id": ENV_ID,
            }
        ),
        trace={
            "event_history": [
                {
                    "type": "episode_complete",
                    "seed": seed,
                    "env_id": ENV_ID,
                    "total_reward": reward,
                    "n_steps": episode["n_steps"],
                    "solved": episode["solved"],
                }
            ],
            "metadata": {
                "example_id": row.get("example_id"),
                "call_site_id": "minigrid.gridworld_policy",
            },
        },
        metadata={"candidate": candidate},
    ).to_dict()


def _row_for_seed(*, split: str, seed: int) -> dict[str, Any]:
    normalized_split = "test" if split in {"heldout", "test", "validation", "val"} else "train"
    rows = [row for row in ROWS if row["split"] == normalized_split]
    if not rows:
        rows = list(ROWS)
    match = next((row for row in rows if int(row["seed"]) == int(seed)), None)
    if match:
        row = dict(match)
        row["task_id"] = f"{row['split']}:{row['seed']}"
        return row
    row = {
        "seed": int(seed),
        "split": normalized_split,
        "example_id": f"ep_{normalized_split}_{int(seed)}",
        "task_id": f"{normalized_split}:{int(seed)}",
    }
    return dict(row)


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
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
