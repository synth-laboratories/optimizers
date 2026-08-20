"""τ²-bench retail inner http_task for GEPA.

GEPA mutates `domain_policy`. Each /rollout runs one retail task through the
native τ²-bench orchestrator (LLM agent + user simulator + evaluator).

Install tau2 (Python >=3.12) before serving:
  uv sync --project tau2_container
Without tau2, /health is ok and /rollout is 503.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from tasks import HELDOUT_IDS, TRAIN_IDS, rows_for, tau2_task_id

TASK_ID = "tau2.retail"
GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"
HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "policy.md"
DATA_DIR = HERE / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(DATA_DIR))
DEFAULT_POLICY = POLICY_PATH.read_text(encoding="utf-8") if POLICY_PATH.is_file() else (
    "You are a retail customer-service agent. Authenticate the user, then follow store policy."
)

_async_rollouts: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()
_tasks_by_id: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tau2_ready() -> bool:
    try:
        import tau2  # noqa: F401
    except Exception:
        return False
    return True


def _require_tau2() -> None:
    if not _tau2_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "τ²-bench is not installed. From tau2_container run "
                "`uv sync` (Python >=3.12) and set OPENAI_API_KEY. "
                "Retail data must exist under TAU2_DATA_DIR/tau2/domains/retail/."
            ),
        )


def _overlay(payload: dict[str, Any]) -> str:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    return str(candidate.get("domain_policy") or DEFAULT_POLICY)


def _example_row(payload: dict[str, Any]) -> dict[str, Any]:
    """GEPA sends the program id as top-level task_id and the row under task."""
    task = payload.get("task")
    if isinstance(task, dict):
        example = task.get("example") if isinstance(task.get("example"), dict) else task
        if isinstance(example, dict) and (
            example.get("task_id") or example.get("example_id") or example.get("seed") is not None
        ):
            return example
    return {}


def _llm_name(payload: dict[str, Any], *, role: str) -> str:
    raw = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    if role == "user":
        override = os.environ.get("TAU2_USER_LLM")
        if override:
            return override
    provider = str(raw.get("provider") or os.environ.get("TAU2_POLICY_PROVIDER") or "openai")
    model = str(raw.get("model") or os.environ.get("TAU2_POLICY_MODEL") or "gpt-4.1-nano")
    if "/" in model:
        return model
    return f"{provider}/{model}"


def _load_task(task_id: str):
    global _tasks_by_id
    _require_tau2()
    from tau2.runner import get_tasks

    if _tasks_by_id is None:
        _tasks_by_id = {str(task.id): task for task in get_tasks("retail")}
    task = _tasks_by_id.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown τ²-bench retail task {task_id}")
    return task


def _run_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tau2()
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_environment, build_user, run_simulation

    example = _example_row(payload)
    split = str(example.get("split") or payload.get("split") or "train")
    seed_raw = example.get("seed")
    if seed_raw is None:
        seed_raw = payload.get("seed") or 0
    seed = int(seed_raw)
    request_task_id = str(
        example.get("task_id")
        or example.get("example_id")
        or payload.get("task_id")
        or f"{split}:{seed}"
    )
    if ":" in request_task_id:
        maybe_split, _, maybe_id = request_task_id.partition(":")
        if maybe_split in {"train", "heldout", "validation"}:
            split = "heldout" if maybe_split in {"heldout", "validation"} else "train"
            if maybe_id.isdigit():
                seed = int(maybe_id)
    retail_id = tau2_task_id(request_task_id, split, seed)
    task = _load_task(retail_id)
    environment = build_environment("retail")
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=_overlay(payload),
        llm=_llm_name(payload, role="agent"),
        llm_args={"temperature": 0.0},
    )
    user = build_user(
        "user_simulator",
        environment,
        task,
        llm=_llm_name(payload, role="user"),
        llm_args={"temperature": 0.0},
    )
    orchestrator = Orchestrator(
        domain="retail",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=int(os.environ.get("TAU2_MAX_STEPS") or 100),
        seed=seed,
    )
    result = run_simulation(orchestrator)
    reward_info = getattr(result, "reward_info", None)
    reward = float(getattr(reward_info, "reward", 0.0) or 0.0) if reward_info is not None else 0.0
    termination = getattr(result, "termination_reason", None)
    if hasattr(termination, "value"):
        termination = termination.value
    messages = getattr(result, "messages", None) or getattr(result, "trajectory", None) or []
    now = _now()
    rollout_id = str(payload.get("rollout_id") or f"tau2_{uuid.uuid4().hex[:12]}")
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
                "tau2_reward": reward,
                "tau2_task_id": retail_id,
                "termination_reason": str(termination or ""),
                "n_messages": len(messages) if hasattr(messages, "__len__") else 0,
            },
        },
        "summary": {
            "outcome_reward": reward,
            "tau2_task_id": retail_id,
            "split": split,
            "termination_reason": str(termination or ""),
        },
        "usage": {},
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }


async def _complete(payload: dict[str, Any]) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required for τ²-bench rollouts")
    try:
        return await asyncio.to_thread(_run_simulation, payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tau2 rollout failed: {exc}") from exc


app = FastAPI(title="tau2-gepa-container")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "tau2_installed": _tau2_ready(),
        "data_dir": str(DATA_DIR),
        "data_mounted": (DATA_DIR / "tau2" / "domains" / "retail" / "tasks.json").is_file()
        and (DATA_DIR / "tau2" / "user_simulator" / "simulation_guidelines.md").is_file(),
        "train_rows": len(TRAIN_IDS),
        "heldout_rows": len(HELDOUT_IDS),
    }


@app.get("/metadata")
@app.get("/info")
def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "tau2_retail_gepa",
            "name": "τ²-bench retail GEPA inner task",
            "description": "Sierra τ²-bench retail customer-service agent with a user simulator.",
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
                "name": "τ²-bench retail",
                "url": "https://github.com/sierra-research/tau2-bench",
                "domain": "retail",
                "scorer": "tau2 native evaluator",
            },
        },
    }


@app.get("/program")
def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "tau2_retail_domain_policy",
        "modules": [
            {
                "module_id": "domain_policy",
                "role": "system",
                "content": DEFAULT_POLICY,
                "mutable": True,
                "candidate_field": "domain_policy",
                "template_variables": [],
            }
        ],
        "target_modules": [
            {"module_id": "domain_policy", "candidate_field": "domain_policy", "objective": "outcome_reward"}
        ],
        "seed_candidate": {"domain_policy": DEFAULT_POLICY},
        "rollout_overlay_schema": {"candidate_fields": ["domain_policy"]},
    }


@app.get("/taskset")
def taskset() -> dict[str, Any]:
    return {"taskset_id": "tau2:retail", "splits": {"train": len(TRAIN_IDS), "heldout": len(HELDOUT_IDS)}}


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
        seed = int(seed_s) if seed_s.isdigit() else 0
        tasks.append({"task_id": task_id, "split": split_name or split, "seed": seed})
    return {"tasks": tasks}


@app.get("/dataset")
def dataset() -> dict[str, Any]:
    return {"splits": {"train": len(TRAIN_IDS), "heldout": len(HELDOUT_IDS)}}


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    return {"rows": rows_for("heldout" if split in {"heldout", "test"} else "train")}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    mode = str(payload.get("submission_mode") or "sync").strip().lower()
    if mode == "async":
        rollout_id = str(payload.get("rollout_id") or f"tau2_{uuid.uuid4().hex[:12]}")
        queued = {
            "rollout_id": rollout_id,
            "status": "running",
            "success_status": "running",
            "task_id": payload.get("task_id"),
            "created_at": _now(),
            "updated_at": _now(),
        }
        async with _lock:
            _async_rollouts[rollout_id] = queued

        async def worker() -> None:
            try:
                finished = await _complete({**payload, "rollout_id": rollout_id})
            except HTTPException as exc:
                finished = {
                    **queued,
                    "status": "failed",
                    "success_status": "failed",
                    "error": exc.detail,
                    "updated_at": _now(),
                }
            async with _lock:
                _async_rollouts[rollout_id] = finished

        asyncio.create_task(worker())
        return queued
    return await _complete(payload)


@app.get("/rollouts/{rollout_id}")
@app.get("/rollouts/{rollout_id}/state")
async def rollout_get(rollout_id: str) -> dict[str, Any]:
    async with _lock:
        record = _async_rollouts.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
