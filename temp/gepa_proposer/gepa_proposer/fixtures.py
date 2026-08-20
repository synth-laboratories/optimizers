from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

BANKING77_DOWNSTREAM = {
    "id": "banking77",
    "image_id": "banking77",
    "url_env": "BANKING77_URL",
    "url_pool_env": "BANKING77_URLS",
    "candidate_field": "stage2_system",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 16,
        "env_var": "OPENAI_API_KEY",
    },
}

HEALTHBENCH_DOWNSTREAM = {
    "id": "healthbench2",
    "image_id": "healthbench2",
    "url_env": "HEALTHBENCH_URL",
    "url_pool_env": "HEALTHBENCH_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "api_family": "chat_completions",
        "base_url": "https://api.groq.com/openai/v1",
        "max_tokens": 1536,
        "env_var": "GROQ_API_KEY",
    },
}

OFFICEQA_DOWNSTREAM = {
    "id": "officeqa",
    "image_id": "officeqa",
    "url_env": "OFFICEQA_URL",
    "url_pool_env": "OFFICEQA_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1",
        "api_family": "chat_completions",
        "max_tokens": 256,
        "env_var": "OPENAI_API_KEY",
    },
}

CRAFTER_DOWNSTREAM = {
    "id": "crafter",
    "image_id": "crafter",
    "url_env": "CRAFTER_URL",
    "url_pool_env": "CRAFTER_URLS",
    "candidate_field": "react_system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 256,
        "env_var": "OPENAI_API_KEY",
    },
}

HEALTHBENCH_OPENAI_DOWNSTREAM = {
    "id": "healthbench2",
    "image_id": "healthbench2",
    "url_env": "HEALTHBENCH_URL",
    "url_pool_env": "HEALTHBENCH_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 1536,
        "env_var": "OPENAI_API_KEY",
    },
}

TAU2_DOWNSTREAM = {
    "id": "tau2",
    "image_id": "tau2_retail",
    "url_env": "TAU2_URL",
    "url_pool_env": "TAU2_URLS",
    "candidate_field": "domain_policy",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 512,
        "env_var": "OPENAI_API_KEY",
    },
}

MINIGRID_DOWNSTREAM = {
    "id": "minigrid",
    "image_id": "minigrid_doorkey",
    "url_env": "MINIGRID_URL",
    "url_pool_env": "MINIGRID_URLS",
    "candidate_field": "system_prompt",
    "policy": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "api_family": "chat_completions",
        "max_tokens": 64,
        "env_var": "OPENAI_API_KEY",
    },
}

TASKS: list[dict[str, Any]] = []


def _infer_downstream(task: dict[str, Any]) -> dict[str, Any]:
    if isinstance(task.get("downstream"), dict) and task["downstream"]:
        return task["downstream"]
    task_id = str(task.get("task_id") or "")
    candidates = (task.get("cursor") or {}).get("candidates") or []
    payload = candidates[0].get("payload") if candidates and isinstance(candidates[0], dict) else {}
    if task_id.startswith("minigrid:"):
        return copy.deepcopy(MINIGRID_DOWNSTREAM)
    if task_id.startswith("officeqa:"):
        return copy.deepcopy(OFFICEQA_DOWNSTREAM)
    if task_id.startswith("tau2:") or (isinstance(payload, dict) and "domain_policy" in payload):
        return copy.deepcopy(TAU2_DOWNSTREAM)
    if task_id.startswith("crafter:") or (isinstance(payload, dict) and "react_system_prompt" in payload):
        return copy.deepcopy(CRAFTER_DOWNSTREAM)
    if task_id.startswith("train:") or (isinstance(payload, dict) and "stage2_system" in payload):
        return copy.deepcopy(BANKING77_DOWNSTREAM)
    if task_id == "healthbench:3":
        return copy.deepcopy(HEALTHBENCH_OPENAI_DOWNSTREAM)
    if isinstance(payload, dict) and "system_prompt" in payload:
        return copy.deepcopy(HEALTHBENCH_DOWNSTREAM)
    return copy.deepcopy(BANKING77_DOWNSTREAM)


def _load() -> None:
    if TASKS:
        return
    paths = sorted(FIXTURES_DIR.glob("*.json"))
    order = {
        "banking77_fresh.json": 0,
        "banking77_first.json": 1,
        "banking77_mature.json": 2,
        "banking77_gen3.json": 3,
        "banking77_async_fresh.json": 4,
        "banking77_async_first.json": 5,
        "healthbench_fresh.json": 6,
        "healthbench_first.json": 7,
        "healthbench_mature.json": 8,
        "healthbench_openai.json": 9,
        "healthbench_accepted.json": 10,
        "crafter_fresh.json": 11,
        "crafter_first.json": 12,
        "crafter_mature.json": 13,
        "crafter_archive_fresh.json": 14,
        "crafter_archive_mature.json": 15,
        "tau2_fresh.json": 16,
        "tau2_first.json": 17,
        "tau2_mature.json": 18,
        "minigrid_fresh.json": 19,
        "minigrid_first.json": 20,
        "minigrid_mature.json": 21,
        "officeqa_fresh.json": 22,
    }
    for path in sorted(paths, key=lambda item: order.get(item.name, 50)):
        payload = json.loads(path.read_text())
        payload["downstream"] = _infer_downstream(payload)
        TASKS.append(payload)


def all_tasks() -> list[dict[str, Any]]:
    _load()
    return TASKS


def by_task_id(task_id: str) -> dict[str, Any]:
    _load()
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def fork_cursor(task: dict[str, Any], run_id: str) -> dict[str, Any]:
    cursor = copy.deepcopy(task["cursor"])
    parent_run = cursor.get("run_id")
    cursor["run_id"] = run_id
    cursor["pending_job_id"] = None
    cursor["pending_effect_id"] = None
    cursor["pending_reservation_ids"] = []
    metadata = dict(cursor.get("metadata") or {})
    metadata["retain"] = True
    metadata["fork"] = {
        "parent_run_id": parent_run,
        "parent_task_id": task["task_id"],
        "parent_label": task["label"],
    }
    cursor["metadata"] = metadata
    cursor["usage"] = _usage_totals(cursor.get("usage"))
    return cursor


def _usage_totals(usage: Any) -> dict[str, int]:
    raw = usage if isinstance(usage, dict) else {}
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "rollout_calls": 0,
        "proposer_calls": 0,
    }
    for key in totals:
        try:
            totals[key] = int(raw.get(key) or 0)
        except (TypeError, ValueError):
            totals[key] = 0
    if not totals["prompt_tokens"]:
        try:
            totals["prompt_tokens"] = int(raw.get("input_tokens") or 0)
        except (TypeError, ValueError):
            pass
    if not totals["completion_tokens"]:
        try:
            totals["completion_tokens"] = int(raw.get("output_tokens") or 0)
        except (TypeError, ValueError):
            pass
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals
