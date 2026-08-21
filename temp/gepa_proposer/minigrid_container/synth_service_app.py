"""
MiniGrid GEPA cookbook container (live gymnasium env, OpenAI policy).

Speaks the public synth-optimizers GEPA contract:
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /dataset
  POST /dataset/rows
  POST /rollout

Each rollout instantiates a real MiniGrid env, drives it for up to N steps
with an OpenAI-driven agent using the candidate's `system_prompt`, and
returns the actual env reward.

Required env:
  OPENAI_API_KEY              — required when rollout.policy.credential_mode=byok.
  MINIGRID_MAX_STEPS          — default: 48 (per-episode hard cap)
  MINIGRID_ENV_ID             — default: MiniGrid-DoorKey-5x5-v0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

from tasks import DEFAULT_SYSTEM_PROMPT, HELDOUT_SEEDS, TRAIN_SEEDS, episode_seed, rows_for

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import OpenAI
except Exception as _openai_err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None


TASK_ID = "minigrid.gridworld_policy"
DATASET_ID = "minigrid_public_episodes"

MAX_STEPS = int(os.environ.get("MINIGRID_MAX_STEPS", "48"))
ENV_ID = os.environ.get("MINIGRID_ENV_ID", "MiniGrid-Empty-5x5-v0")

# Standard MiniGrid 7 actions, ordered by gymnasium Action enum.
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


_openai_clients: dict[tuple[str, str, str], Any] = {}
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized = str(raw_key).strip().lower().replace("-", "_")
            if normalized in _RAW_CREDENTIAL_KEYS or normalized.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


def _normalize_policy_enum(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or default


def _strip_openai_endpoint_suffix(url: str) -> str:
    normalized = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _example_row(payload: dict[str, Any]) -> dict[str, Any]:
    """GEPA sends the program id as top-level task_id and the row under task."""
    task = payload.get("task")
    if isinstance(task, dict):
        example = task.get("example") if isinstance(task.get("example"), dict) else task
        if isinstance(example, dict) and (
            example.get("task_id") or example.get("example_id") or example.get("seed") is not None
        ):
            return example
    row = payload.get("dataset_row")
    if isinstance(row, dict) and row:
        return row
    return {}


def _require_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    if not policy.get("provider") or not policy.get("model"):
        policy = {
            **policy,
            "provider": policy.get("provider") or os.environ.get("MINIGRID_POLICY_PROVIDER") or "openai",
            "model": policy.get("model") or os.environ.get("MINIGRID_POLICY_MODEL") or "gpt-4.1-nano",
        }
    raw_key = _find_raw_credential_key(policy.get("config", {}))
    if raw_key is not None:
        raise HTTPException(
            status_code=422,
            detail=f"rollout.policy.config must not carry raw credential field {raw_key!r}.",
        )
    provider = str(policy.get("provider") or "").strip()
    model = str(policy.get("model") or "").strip()
    if not provider or not model:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.provider and rollout.policy.model are required.",
        )
    api_family = _normalize_policy_enum(policy.get("api_family"), "chat_completions")
    if api_family != "chat_completions":
        raise HTTPException(
            status_code=422,
            detail=f"{TASK_ID} supports rollout.policy.api_family='chat_completions'; got {api_family!r}.",
        )
    credential_mode = _normalize_policy_enum(policy.get("credential_mode"), "byok")
    if credential_mode in {"proxy_only", "proxy", "workshop_proxy"}:
        raw_base_url = (
            str(policy.get("inference_url") or "").strip()
            or str(policy.get("base_url") or "").strip()
        )
        if not raw_base_url:
            raise HTTPException(
                status_code=422,
                detail="rollout.policy.inference_url is required when credential_mode is proxied.",
            )
        if credential_mode == "proxy_only":
            credential_mode = "workshop_proxy"
    if credential_mode not in {"byok", "proxy", "workshop_proxy"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.credential_mode: {credential_mode!r}",
        )
    raw_base_url = (
        str(policy.get("inference_url") or "").strip()
        if credential_mode in {"proxy", "workshop_proxy"}
        else str(policy.get("base_url") or "").strip()
    )
    if credential_mode in {"proxy", "workshop_proxy"} and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.inference_url is required when credential_mode is proxied.",
        )
    if provider.lower() == "openrouter" and credential_mode == "byok" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.base_url is required for provider=openrouter.",
        )
    max_tokens = policy.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="rollout.policy.max_tokens must be an integer when set.",
            ) from exc
        if max_tokens <= 0:
            raise HTTPException(
                status_code=422,
                detail="rollout.policy.max_tokens must be positive when set.",
            )
    return {
        "provider": provider,
        "model": model,
        "base_url": _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None,
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
    }


def _policy_api_key(policy: dict[str, Any]) -> str:
    if policy["credential_mode"] in {"proxy", "workshop_proxy"}:
        return os.environ.get("OPENAI_API_KEY", "").strip() or "workshop-proxy"
    env_name = "OPENROUTER_API_KEY" if policy["provider"].lower() == "openrouter" else "OPENAI_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise HTTPException(
        status_code=503,
        detail=f"{env_name} is not set; rollout.policy credential_mode=byok requires a container env credential.",
    )


def _get_openai_client(policy: dict[str, Any]) -> Any:
    if OpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; container deps in pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    base_url = policy.get("base_url")
    key = (policy["provider"].lower(), policy["credential_mode"], str(base_url or ""))
    client = _openai_clients.get(key)
    if client is None:
        client_kwargs = {"api_key": _policy_api_key(policy)}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        _openai_clients[key] = client
    return client


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
    if grid is not None:
        for x in range(width):
            for y in range(height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = str(getattr(obj, "type", "") or "")
                if obj_type in {"wall", "unseen", "floor"}:
                    continue
                obj_color = str(getattr(obj, "color", "") or "")
                is_locked = bool(getattr(obj, "is_locked", False))
                is_open = bool(getattr(obj, "is_open", False))
                label_bits = [obj_color, obj_type]
                if is_locked:
                    label_bits.append("(locked)")
                if is_open:
                    label_bits.append("(open)")
                visible.append(f"  ({x},{y}): {' '.join(b for b in label_bits if b)}")

    visible_block = "\n".join(visible) if visible else "  (none)"
    ascii_map = _ascii_map(env)
    goal_pos = None
    if grid is not None:
        for x in range(width):
            for y in range(height):
                obj = grid.get(x, y)
                if obj is not None and str(getattr(obj, "type", "") or "") == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos is not None:
                break
    goal_hint = "goal not visible"
    if goal_pos and len(agent_pos) == 2:
        dx = int(goal_pos[0]) - int(agent_pos[0])
        dy = int(goal_pos[1]) - int(agent_pos[1])
        needed = []
        if dx > 0:
            needed.append("east/right")
        elif dx < 0:
            needed.append("west/left")
        if dy > 0:
            needed.append("south/down")
        elif dy < 0:
            needed.append("north/up")
        goal_hint = (
            f"goal_at={list(goal_pos)} delta=({dx},{dy}) need_to_move={', '.join(needed) or 'none (on goal)'} "
            f"currently_facing={direction_name}"
        )
    return (
        f"mission: {mission}\n"
        f"agent_position: {agent_pos}  facing: {direction_name}\n"
        f"carrying: {carrying_str}\n"
        f"{goal_hint}\n"
        f"admissible_actions: {ACTION_NAMES}\n"
        f"legend: # wall  . empty  >v<^ agent  K key  D locked door  O open door  G goal\n"
        f"grid:\n{ascii_map}\n"
        f"objects:\n{visible_block}"
    )


def _ascii_map(env) -> str:
    grid = getattr(env.unwrapped, "grid", None)
    if grid is None:
        return ""
    agent_pos = tuple(getattr(env.unwrapped, "agent_pos", ()) or ())
    agent_dir = int(getattr(env.unwrapped, "agent_dir", 0) or 0)
    arrows = {0: ">", 1: "v", 2: "<", 3: "^"}
    lines: list[str] = []
    for y in range(int(getattr(grid, "height", 0) or 0)):
        row: list[str] = []
        for x in range(int(getattr(grid, "width", 0) or 0)):
            if agent_pos == (x, y):
                row.append(arrows.get(agent_dir, "A"))
                continue
            obj = grid.get(x, y)
            if obj is None:
                row.append(".")
                continue
            obj_type = str(getattr(obj, "type", "") or "")
            if obj_type == "wall":
                row.append("#")
            elif obj_type == "key":
                row.append("K")
            elif obj_type == "door":
                if bool(getattr(obj, "is_locked", False)):
                    row.append("D")
                elif bool(getattr(obj, "is_open", False)):
                    row.append("O")
                else:
                    row.append("d")
            elif obj_type == "goal":
                row.append("G")
            else:
                row.append(obj_type[:1].upper() or "?")
        lines.append("".join(row))
    return "\n".join(lines)


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
    client: Any,
    policy: dict[str, Any],
    system_prompt: str,
    observation_text: str,
    step: int,
) -> tuple[str | None, dict[str, int]]:
    user_content = (
        f"Step {step + 1}. Current state:\n\n{observation_text}\n\n"
        'Reply with strict JSON: {"action": "<one admissible action name>"}'
    )
    request_kwargs = {
        "model": policy["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    if policy["max_tokens"] is not None:
        request_kwargs["max_tokens"] = policy["max_tokens"]
    resp = client.chat.completions.create(**request_kwargs)
    text = (resp.choices[0].message.content or "").strip()
    action = _parse_action(text)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return action, usage


def _run_episode(seed: int, system_prompt: str, policy: dict[str, Any]) -> dict[str, Any]:
    client = _get_openai_client(policy)
    env, obs = _make_env(seed)

    total_reward = 0.0
    all_actions: list[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = False
    step = 0
    last_note = "none"

    try:
        while step < MAX_STEPS:
            obs_text = _render_observation_text(env, obs) + f"\nlast_action_result: {last_note}"
            action_name, usage = _llm_step(client, policy, system_prompt, obs_text, step)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            if action_name is None:
                last_note = "invalid JSON; no env step"
                step += 1
                continue
            action_idx = ACTION_NAMES.index(action_name)
            obs, reward, terminated, truncated, _info = env.step(action_idx)
            total_reward += float(reward)
            all_actions.append(action_name)
            step += 1
            last_note = f"{action_name} reward={float(reward):.3f}"
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
    return {
        "status": "ok",
        "env_id": ENV_ID,
        "max_steps": MAX_STEPS,
        "train_rows": len(TRAIN_SEEDS),
        "heldout_rows": len(HELDOUT_SEEDS),
    }


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
            "metadata": {
                "trace_schema": "prompt_calls.llm_request.messages.v1",
                "policy_ready": True,
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
            },
            "benchmark": {
                "name": "MiniGrid DoorKey-5x5",
                "env_id": ENV_ID,
                "scorer": "gymnasium episode reward",
            },
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
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": len(TRAIN_SEEDS) + len(HELDOUT_SEEDS),
            "seed_semantics": (
                "Rows are generated from requested episode seeds. The same seed is deterministic "
                "for a given MiniGrid env id."
            ),
        },
        "prompt_program": {
            "mutable_modules": ["system_prompt"],
            "candidate_field": "system_prompt",
            "output_contract": "Every policy call must return strict JSON: {\"action\": \"<admissible action>\"}.",
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
            "policy_model_source": "rollout.policy.model",
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
                "objective": "outcome_reward",
            }
        ],
        "seed_candidate": {"system_prompt": DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "env_id": ENV_ID,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/taskset")
def taskset() -> dict[str, Any]:
    return {"taskset_id": "minigrid:doorkey", "splits": {"train": len(TRAIN_SEEDS), "heldout": len(HELDOUT_SEEDS)}}


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    raw_ids = payload.get("task_ids") or []
    known = {row["task_id"]: row for row in rows_for("train") + rows_for("heldout")}
    tasks = []
    for raw in raw_ids:
        task_id = str(raw)
        if task_id in known:
            tasks.append(known[task_id])
            continue
        split_name, _, seed_s = task_id.partition(":")
        seed = int(seed_s) if seed_s.isdigit() else 1
        normalized = "heldout" if split_name in {"heldout", "test", "validation"} else (split_name or split)
        if normalized == "test":
            normalized = "heldout"
        tasks.append({"task_id": task_id, "example_id": task_id, "split": normalized, "seed": seed})
    return {"tasks": tasks}


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {"train": len(TRAIN_SEEDS), "heldout": len(HELDOUT_SEEDS)},
        "source": "minigrid_public_episode_seeds",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    normalized = "heldout" if split in {"heldout", "test", "validation", "val"} else "train"
    return {"rows": rows_for(normalized)}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required for MiniGrid rollouts")
    policy = _require_policy(payload)
    example = _example_row(payload)
    split = str(example.get("split") or payload.get("split") or "train")
    if split in {"test", "validation", "val"}:
        split = "heldout"
    seed_raw = example.get("seed")
    if seed_raw is None:
        seed_raw = payload.get("seed") or 1
    seed = int(seed_raw)
    request_task_id = str(
        example.get("task_id")
        or example.get("example_id")
        or payload.get("task_id")
        or f"{split}:{seed}"
    )
    if ":" in request_task_id:
        maybe_split, _, maybe_id = request_task_id.partition(":")
        if maybe_split in {"train", "heldout", "test", "validation"}:
            split = "heldout" if maybe_split in {"heldout", "test", "validation"} else "train"
            if maybe_id.isdigit():
                seed = int(maybe_id)
    seed = episode_seed(request_task_id, split, seed)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    episode = await asyncio.to_thread(_run_episode, seed, system_prompt, policy)
    reward = float(episode["total_reward"])
    rollout_id = str(payload.get("rollout_id") or f"minigrid_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded",
        "task_id": request_task_id,
        "seed": seed,
        "reward": reward,
        "reward_info": {
            "outcome_reward": reward,
            "metrics": {
                "minigrid_reward": reward,
                "n_steps": episode["n_steps"],
                "solved": episode["solved"],
                "env_id": ENV_ID,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": request_task_id,
            "split": split,
            "n_steps": episode["n_steps"],
            "solved": episode["solved"],
            "actions_taken": episode["actions"],
        },
        "usage": {**episode["usage"], "cost_usd": 0.0},
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }


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
