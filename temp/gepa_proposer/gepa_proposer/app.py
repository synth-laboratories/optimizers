from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .episode import build_run_request, parse_episode, program_for_task
from .fixtures import all_tasks, by_task_id, fork_cursor
from .optimizer_client import OptimizerClient
from .scoring import score_episode
from .store import JsonStore

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"
CONTRACT_VERSION = "2026-05-28"
OPTIMIZER_REQUIRED = (
    "GEPA_SERVICE_URL is required. A scored rollout is a finished GEPA episode "
    "against the downstream container, not a dry cursor fork."
)

STORE = JsonStore(
    Path(os.environ.get("GEPA_PROPOSER_STATE_DIR") or (Path(__file__).resolve().parent.parent / ".state"))
)
OPTIMIZER = OptimizerClient()
_INNER_POOLS: dict[str, int] = {}
_POOL_LOCK = threading.Lock()
_LIVE_TASKS: dict[str, asyncio.Task[None]] = {}

app = FastAPI(title="gepa-proposer")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_cursor(cursor: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "candidates": cursor.get("candidates"),
            "train_rows": cursor.get("train_rows"),
            "minibatch_rows": cursor.get("minibatch_rows"),
            "heldout_rows": cursor.get("heldout_rows"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# Arm fields the episode builder understands beyond the three defaults. Anything
# here is passed through verbatim; anything else in `policy` is rejected rather
# than dropped. Silently discarding an unknown key is how `proposer_timeout_seconds`
# went missing on 2026-08-20: the request looked accepted, the arm ran at the 300s
# service default, and the only evidence was a 301s wall clock.
_ARM_PASSTHROUGH_FIELDS = (
    "api_family",
    "auth_mode",
    "api_key_env",
    "allow_unverified_model",
    "codex_home",
    "proposer_timeout_seconds",
    "model_context_window",
    "model_auto_compact_token_limit",
    "label",
)


def _policy_arm(policy: dict[str, Any] | None) -> dict[str, Any]:
    policy = policy or {}
    known = {"provider", "model", "reasoning_effort", *_ARM_PASSTHROUGH_FIELDS}
    unknown = sorted(set(policy) - known)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown policy field(s) {unknown}; expected one of {sorted(known)}"
            ),
        )
    arm: dict[str, Any] = {
        "provider": policy.get("provider") or "openai",
        "model": policy.get("model") or "gpt-5.6-luna",
        "reasoning_effort": policy.get("reasoning_effort") or "low",
    }
    for field in _ARM_PASSTHROUGH_FIELDS:
        if policy.get(field) is not None:
            arm[field] = policy[field]
    return arm


def _episode_candidates(pre: dict[str, Any], post: dict[str, Any]) -> list[dict[str, Any]]:
    pre_ids = {c.get("candidate_id") for c in pre.get("candidates") or []}
    return [c for c in post.get("candidates") or [] if c.get("candidate_id") not in pre_ids]


def _gepa_contract() -> dict[str, str]:
    return {
        "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
        "program_route": "/program",
        "taskset_route": "/taskset",
        "taskset_tasks_route": "/taskset/tasks",
        "rollout_route": "/rollouts",
    }


def _reward_combine(episode: dict[str, Any]) -> dict[str, Any]:
    operator = episode.get("operator") if isinstance(episode.get("operator"), dict) else {}
    reward = operator.get("reward") if isinstance(operator.get("reward"), dict) else {}
    return {
        "exploration_reduce": reward.get("exploration_reduce") or "mean",
        "missing": reward.get("missing") or "zero",
        "include_confidence": bool(reward.get("confidence")),
        "include_time": bool(reward.get("time")),
        "include_cost": bool(reward.get("cost")),
        "include_milestones": bool(reward.get("milestones")),
        "include_rubrics": bool(reward.get("rubrics")),
        "exploration_weight": float(reward.get("exploration_weight", 1.0)),
        "exploitation_weight": float(reward.get("exploitation_weight", 1.0)),
        "eval_uplift_weight": float(reward.get("eval_uplift_weight", 1.0)),
        "confidence_weight": float(reward.get("confidence_weight", 0.0)),
        "time_weight": float(reward.get("time_weight", 0.0)),
        "cost_weight": float(reward.get("cost_weight", 0.0)),
        "milestones_weight": float(reward.get("milestones_weight", 0.0)),
        "rubrics_weight": float(reward.get("rubrics_weight", 0.0)),
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    if payload.get("status") == "completed" and payload.get("reward") is not None:
        payload["success_status"] = payload.get("success_status") or "succeeded"
        payload["reward_info"] = {
            "outcome_reward": float(payload["reward"]),
            "metrics": {
                row["objective"]: row["value"]
                for row in payload.get("objective_scores") or []
                if isinstance(row, dict) and "objective" in row
            },
        }
        payload["summary"] = {"outcome_reward": float(payload["reward"])}
    elif payload.get("status") == "failed":
        payload["success_status"] = payload.get("success_status") or "failed"
    else:
        payload.setdefault("success_status", payload.get("status") or "running")
    return payload


def _score_record(record: dict[str, Any]) -> dict[str, Any]:
    if not record.get("optimizer_run_id"):
        raise HTTPException(status_code=503, detail=OPTIMIZER_REQUIRED)
    if record.get("optimizer_status") not in {"succeeded", "completed"}:
        raise HTTPException(
            status_code=409,
            detail=(
                "refusing to score: GEPA episode did not succeed "
                f"(status={record.get('optimizer_status')!r})"
            ),
        )
    pre = record["pre_fork_cursor"]
    post = record["cursor"]
    scored = score_episode(
        pre_fork=pre.get("candidates") or [],
        episode_candidates=_episode_candidates(pre, post),
        post_candidates=post.get("candidates") or [],
        best_candidate_id=post.get("best_candidate_id"),
        combine=_reward_combine(record.get("episode") or {}),
        context={
            "pre_cursor": pre,
            "post_cursor": post,
            "created_at": record.get("created_at"),
            "completed_at": record.get("completed_at") or _now(),
            "optimizer_finished": record.get("optimizer_finished") or {},
            "episode": record.get("episode") or {},
            "arm": record.get("arm") or {},
            "downstream": record.get("downstream") or {},
        },
    )
    skip_heldout = bool((record.get("episode") or {}).get("skip_heldout"))
    if not skip_heldout and not scored["heldout_evaluated"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "refusing to score: GEPA episode succeeded without heldout "
                "evidence for eval_uplift (train exploration / train "
                "exploitation / eval uplift all required)"
            ),
        )
    record["status"] = "completed"
    record["success_status"] = "succeeded"
    record["reward"] = scored["reward"]
    record["objective_scores"] = [
        {"objective": "train_exploration", "value": scored["train_exploration"]},
        {"objective": "train_exploitation", "value": scored["train_exploitation"]},
        {"objective": "eval_uplift", "value": scored["eval_uplift"]},
        *[
            {"objective": name, "value": value}
            for name, value in (scored.get("optional_terms") or {}).items()
        ],
        *([{
            "objective": "episode_cost_usd",
            "value": scored["episode_cost_usd"],
        }] if scored.get("episode_cost_usd") is not None else []),
    ]
    record["reward_details"] = scored
    record["completed_at"] = _now()
    record["updated_at"] = _now()
    return record


def _env_urls(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def _next_pool_url(pool_env: str, fallback_env: str) -> str:
    urls = _env_urls(pool_env) or _env_urls(fallback_env)
    if not urls:
        return ""
    key = pool_env or fallback_env
    with _POOL_LOCK:
        index = _INNER_POOLS.get(key, 0)
        _INNER_POOLS[key] = index + 1
    return urls[index % len(urls)]


def _inner_url(spec: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    override = str(payload.get("inner_url") or payload.get("container_url") or "").strip().rstrip("/")
    if override and not override.startswith("$"):
        return override
    downstream = spec.get("downstream") if isinstance(spec.get("downstream"), dict) else {}
    url = _next_pool_url(str(downstream.get("url_pool_env") or ""), str(downstream.get("url_env") or ""))
    if not url:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{downstream.get('url_env') or 'DOWNSTREAM_URL'} is required "
                "for a live episode (inner http_task)"
            ),
        )
    return url


async def _wait_existing(record: dict[str, Any]) -> dict[str, Any]:
    run_id = str(record.get("optimizer_run_id") or "")
    if not run_id:
        raise RuntimeError("no optimizer_run_id to wait on")
    timeout = float(
        record["episode"].get("max_wall_seconds") or (1800 * record["episode"]["proposer_rounds"])
    )
    # Inner stop_conditions fire at max_wall_seconds; wait past that for
    # heldout finalize and cursor persist before scoring.
    timeout += float(os.environ.get("GEPA_PROPOSER_WAIT_HEADROOM_SECONDS") or 90)
    waiter = getattr(OPTIMIZER, "await_until_terminal", None)
    if callable(waiter):
        finished = await waiter(run_id, timeout_seconds=timeout)
    else:
        finished = await asyncio.to_thread(
            OPTIMIZER.wait_until_terminal, run_id, timeout_seconds=timeout
        )
    return await _ingest_terminal(record, finished)


async def _ingest_terminal(record: dict[str, Any], finished: dict[str, Any]) -> dict[str, Any]:
    record["optimizer_status"] = str(finished.get("status") or "")
    record["optimizer_finished"] = finished
    getter = getattr(OPTIMIZER, "aget_state", None)
    if callable(getter):
        state = await getter(str(record["optimizer_run_id"])) or {}
    else:
        state = await asyncio.to_thread(OPTIMIZER.get_state, str(record["optimizer_run_id"])) or {}
    cursor = state.get("cursor")
    if isinstance(cursor, dict) and cursor:
        record["cursor"] = cursor
    if record["optimizer_status"] not in {"succeeded", "completed"}:
        record["status"] = "failed"
        record["success_status"] = "failed"
        record["error"] = (
            f"GEPA episode ended {record['optimizer_status']!r}; "
            "refusing a reward from a non-succeeded run"
        )
        record["updated_at"] = _now()
        return record
    return _score_record(record)


async def _execute_live(record: dict[str, Any]) -> dict[str, Any]:
    spec = by_task_id(record["task_id"])
    body = build_run_request(
        spec=spec,
        cursor=record["pre_fork_cursor"],
        arm=record["arm"],
        episode=record["episode"],
        container_url=record["inner_url"],
        run_id=record["run_id"],
        output_dir=record.get("output_dir"),
    )
    creator = getattr(OPTIMIZER, "acreate_run", None)
    if callable(creator):
        created = await creator(body) or {}
    else:
        created = await asyncio.to_thread(OPTIMIZER.create_run, body) or {}
    run_id = str(created.get("run_id") or created.get("id") or "")
    if not run_id:
        raise RuntimeError(f"GEPA create_run returned no run_id: {created}")
    record["optimizer_run_id"] = run_id
    record["optimizer_create"] = created
    STORE.put_rollout(record)
    return await _wait_existing(record)


def _spawn_live_episode(rollout_id: str) -> None:
    def worker() -> None:
        try:
            asyncio.run(_run_live_episode(rollout_id))
        except Exception as exc:  # pragma: no cover - live path
            record = STORE.get_rollout(rollout_id)
            if record is None:
                return
            record["status"] = "failed"
            record["success_status"] = "failed"
            record["error"] = str(exc)
            record["updated_at"] = _now()
            STORE.put_rollout(record)

    thread = threading.Thread(target=worker, name=f"gepa-episode-{rollout_id[:8]}", daemon=True)
    thread.start()


async def _run_live_episode(rollout_id: str) -> None:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        return
    try:
        STORE.put_rollout(await _execute_live(record))
    except Exception as exc:  # pragma: no cover - live path
        record["status"] = "failed"
        record["success_status"] = "failed"
        record["error"] = str(exc)
        record["updated_at"] = _now()
        STORE.put_rollout(record)


def _new_rollout(
    task_id: str,
    policy: dict[str, Any],
    episode: dict[str, Any],
    *,
    inner_url: str | None = None,
    rollout_id: str | None = None,
) -> dict[str, Any]:
    spec = by_task_id(task_id)
    rollout_id = rollout_id or str(uuid.uuid4())
    run_id = f"episode-{rollout_id[:8]}"
    cursor = fork_cursor(spec, run_id)
    now = _now()
    record = {
        "rollout_id": rollout_id,
        "task_id": task_id,
        "run_id": run_id,
        "status": "running",
        "success_status": "running",
        "arm": _policy_arm(policy),
        "episode": episode,
        "inner_url": inner_url,
        "downstream": spec.get("downstream"),
        "pre_fork_cursor": copy.deepcopy(cursor),
        "cursor": cursor,
        "optimizer_run_id": None,
        "created_at": now,
        "updated_at": now,
        "paused": False,
    }
    root = os.environ.get("GEPA_PROPOSER_OUTPUT_DIR")
    if root:
        record["output_dir"] = str(Path(root) / run_id)
    STORE.put_rollout(record)
    return record


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "optimizer_live": OPTIMIZER.live,
        "episode_requires_optimizer": True,
        "target": "gepa_proposer",
    }


@app.get("/metadata")
@app.get("/info")
def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "gepa_proposer",
            "name": "GEPA proposer eval / RL env",
            "description": "Task = (downstream container, GEPA cursor fixture). Action = propose.",
            "runtime_kind": "environment",
        },
        "capabilities": {
            "contract_version": CONTRACT_VERSION,
            "rollout_modes": ["blocking", "async"],
            "profiles": ["harness_managed_benchmark_environment"],
            "protocol_fidelity": {
                "catalog_backed": "native",
                "async_rollout_runnable": "native",
                "verifier_backed": "native",
                "reward_emitting": "native",
                "checkpointable": "native",
                "restorable": "native",
            },
            "statefulness_tier": "episodic",
            "checkpoint_support": True,
            "pause_support": True,
            "resume_support": True,
            "terminate_support": True,
            "reward_support": True,
            "verifier_support": True,
            "route_hints": {
                "metadata_routes": ["/metadata", "/info"],
                "task_info_routes": ["/task_info"],
                "program_routes": ["/program"],
                "taskset_routes": ["/taskset", "/taskset/tasks"],
                "rollout_routes": ["/rollouts"],
                "state_routes": ["/rollouts/{rollout_id}/state"],
                "compatibility_routes": ["/compatibility"],
            },
        },
        "metadata": {
            "optimizer_contracts": {"gepa": _gepa_contract()},
            "gelo_compat": {
                "tier": "B",
                "checkpoint_route": "/rollouts/{id}/checkpoints",
                "resume_route": "/rollouts/{parent}/resume_async",
            },
            "harbor": {
                "target": "harbor_proxy",
                "supported": True,
            },
            "episode": {
                "reward_source": "terminal_gepa_cursor",
                "default_proposer_rounds": 1,
                "skip_heldout": False,
                "parallel_rollouts": True,
                "exploration_reduce": "mean",
            },
            "gepa_is_everything": {
                "harness_opt": "optional",
                "prompt": "default",
                "optimize_anything": "optional",
                "manderqueue": "optional",
                "workspace_fs": "default",
                "scratchpad": "optional",
                "pause": "POST /runs/{id}/pause",
                "restart": "POST /runs/{id}/resume",
                "branch": "POST /runs fork_from",
                "candidate_hypotheses": "optional",
                "pipeline": [
                    "sync_serial",
                    "async_pipelined",
                    "flash_evolve",
                    "combee",
                ],
                "jesterky": "optional",
                "reward": {
                    "terms": [
                        "train_exploration",
                        "train_exploitation",
                        "eval_uplift",
                    ],
                    "exploration_reduce": "mean",
                    "optional": [
                        "missing",
                        "confidence",
                        "time",
                        "cost",
                        "milestones",
                        "rubrics",
                    ],
                },
                "mcp_agent": "optional",
                "style_guides": "proposer.prompt.style_guides",
            },
        },
    }


@app.get("/compatibility")
def compatibility(target: str = "harbor_proxy") -> dict[str, Any]:
    normalized = target.strip().lower().replace("-", "_")
    supported = normalized in {"harbor_proxy", "go_ex", "standard_evals"}
    return {
        "target": normalized,
        "supported": supported,
        "tier": "B" if normalized != "harbor_proxy" else "eval",
        "missing_profiles": [],
        "missing_protocols": [],
        "missing_features": [],
        "issues": [],
        "notes": [
            "Eval is POST /rollout. Harbor consumes async catalog-backed verifier rollouts.",
            "go_ex uses /rollouts/{parent}/resume_async plus checkpoints.",
        ],
        "resume": "/rollouts/{parent}/resume_async",
    }


@app.get("/program")
def program(task_id: str | None = None) -> dict[str, Any]:
    if task_id:
        try:
            return program_for_task(by_task_id(task_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown task_id {task_id}") from exc
    return program_for_task(all_tasks()[0])


@app.get("/task_info")
def task_info() -> dict[str, Any]:
    return {
        "task_id": "gepa_proposer",
        "task_name": "GEPA proposer eval / RL env",
        "splits": {"train": len(all_tasks())},
        "fixtures": [
            {
                "task_id": t["task_id"],
                "label": t["label"],
                "maturity": t["maturity"],
                "downstream": (t.get("downstream") or {}).get("id"),
            }
            for t in all_tasks()
        ],
    }


@app.get("/dataset")
def dataset() -> dict[str, Any]:
    return {
        "dataset_id": "gepa_proposer_fixtures:v0",
        "splits": {"train": len(all_tasks())},
        "visible_splits": ["train"],
        "default_split": "train",
        "row_count": len(all_tasks()),
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train").strip()
    if split != "train":
        raise HTTPException(status_code=422, detail="only split=train is defined in v0")
    rows = []
    for spec in all_tasks():
        rows.append(
            {
                "task_id": spec["task_id"],
                "split": "train",
                "label": spec["label"],
                "downstream": spec.get("downstream"),
            }
        )
    requested = payload.get("task_ids") or payload.get("seeds")
    if isinstance(requested, list) and requested:
        wanted = {str(item) for item in requested}
        rows = [row for row in rows if row["task_id"] in wanted]
    return {"rows": rows}


@app.get("/taskset")
def taskset() -> dict[str, Any]:
    return {
        "taskset_id": "gepa_proposer:v0",
        "splits": {"train": len(all_tasks())},
        "metadata": {"fixture_count": len(all_tasks())},
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train").strip()
    if split != "train":
        raise HTTPException(status_code=422, detail="only split=train is defined in v0")
    raw_ids = payload.get("task_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise HTTPException(status_code=422, detail="task_ids must be a non-empty list")
    tasks = []
    for raw in raw_ids:
        task_id = str(raw).strip()
        try:
            spec = by_task_id(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown task_id {task_id}") from exc
        cursor = spec["cursor"]
        seed = 0
        if ":" in task_id:
            try:
                seed = int(task_id.split(":", 1)[1])
            except ValueError:
                seed = 0
        tasks.append(
            {
                "task_id": task_id,
                "split": "train",
                "seed": seed,
                "label": spec["label"],
                "maturity": spec["maturity"],
                "generation": cursor.get("generation"),
                "candidate_count": len(cursor.get("candidates") or []),
                "cursor_sha256": _hash_cursor(cursor),
                "downstream": spec.get("downstream"),
            }
        )
    return {"tasks": tasks}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> Any:
    payload = await request.json()
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(status_code=422, detail="task_id is required")
    try:
        spec = by_task_id(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown task_id {task_id}") from exc
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    if candidate:
        raise HTTPException(
            status_code=422,
            detail="v0 candidate must be empty; the proposer writes payloads. RL overlay is not wired.",
        )
    try:
        episode = parse_episode(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    mode = str(payload.get("submission_mode") or "sync")
    inner_url = None
    if OPTIMIZER.live or str(payload.get("inner_url") or payload.get("container_url") or ""):
        inner_url = _inner_url(spec, payload)
    rollout_id = str(payload.get("rollout_id") or "").strip() or None
    record = _new_rollout(task_id, policy, episode, inner_url=inner_url, rollout_id=rollout_id)
    if mode == "async":
        if OPTIMIZER.live:
            _spawn_live_episode(record["rollout_id"])
        return _public_record(
            {
                "rollout_id": record["rollout_id"],
                "status": "running",
                "success_status": "running",
                "task_id": task_id,
                "created_at": record["created_at"],
            }
        )
    if not OPTIMIZER.live:
        raise HTTPException(status_code=503, detail=OPTIMIZER_REQUIRED)
    try:
        finished = await _execute_live(record)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    STORE.put_rollout(finished)
    return _public_record(finished)


@app.get("/rollouts/{rollout_id}")
@app.get("/rollouts/{rollout_id}/state")
@app.get("/rollouts/{rollout_id}/summary")
def rollout_get(rollout_id: str) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    return _public_record(record)


@app.post("/rollouts/{rollout_id}/terminate")
def rollout_terminate(rollout_id: str) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    record["status"] = "cancelled"
    record["success_status"] = "failed"
    record["updated_at"] = _now()
    STORE.put_rollout(record)
    task = _LIVE_TASKS.pop(rollout_id, None)
    if task is not None:
        task.cancel()
    if OPTIMIZER.live and record.get("optimizer_run_id"):
        try:
            OPTIMIZER.pause(str(record["optimizer_run_id"]))
        except Exception:
            pass
    return _public_record(record)


@app.post("/rollouts/{rollout_id}/pause")
def rollout_pause(rollout_id: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    record["paused"] = True
    record["status"] = "paused"
    record["success_status"] = "paused"
    record["cursor"]["phase"] = "paused"
    record["pause_timeout_seconds"] = timeout_seconds
    record["updated_at"] = _now()
    STORE.put_rollout(record)
    if OPTIMIZER.live and record.get("optimizer_run_id"):
        OPTIMIZER.pause(str(record["optimizer_run_id"]), timeout_seconds=timeout_seconds)
    return _public_record(record)


@app.post("/rollouts/{rollout_id}/resume")
def rollout_resume(rollout_id: str) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    if not record.get("paused"):
        raise HTTPException(status_code=409, detail="rollout is not paused")
    record["paused"] = False
    record["status"] = "running"
    record["success_status"] = "running"
    record["cursor"]["phase"] = "generation_start"
    record["updated_at"] = _now()
    STORE.put_rollout(record)
    if OPTIMIZER.live and record.get("optimizer_run_id"):
        OPTIMIZER.resume(str(record["optimizer_run_id"]))
    return _public_record(record)


@app.post("/rollouts/{rollout_id}/complete")
async def rollout_complete(rollout_id: str) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    if record.get("status") == "completed":
        return _public_record(record)
    if OPTIMIZER.live and record.get("status") == "running":
        try:
            record = (
                await _wait_existing(record)
                if record.get("optimizer_run_id")
                else await _execute_live(record)
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        STORE.put_rollout(record)
        return _public_record(record)
    finished = _score_record(record)
    STORE.put_rollout(finished)
    return _public_record(finished)


@app.post("/rollouts/{rollout_id}/checkpoints")
async def create_checkpoint(rollout_id: str, request: Request) -> dict[str, Any]:
    record = STORE.get_rollout(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    body = {}
    if request.headers.get("content-length") not in {None, "0"}:
        body = await request.json()
    checkpoint_id = str(body.get("checkpoint_id") or f"ckpt-{uuid.uuid4().hex[:12]}")
    pin = {
        "checkpoint_id": checkpoint_id,
        "parent_rollout_id": rollout_id,
        "task_id": record["task_id"],
        "cursor": copy.deepcopy(record["cursor"]),
        "pre_fork_cursor": copy.deepcopy(record["pre_fork_cursor"]),
        "arm": record["arm"],
        "episode": record.get("episode"),
        "inner_url": record.get("inner_url"),
        "retained": True,
        "created_at": _now(),
    }
    STORE.put_checkpoint(pin)
    if OPTIMIZER.live and record.get("optimizer_run_id"):
        try:
            OPTIMIZER.pin(str(record["optimizer_run_id"]))
        except Exception:
            pass
    return pin


@app.post("/rollouts/{parent_rollout_id}/resume_async")
async def resume_async(parent_rollout_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    checkpoint_id = str(body.get("checkpoint_id") or "").strip()
    pin = STORE.get_checkpoint(checkpoint_id) if checkpoint_id else None
    if pin is None:
        parent = STORE.get_rollout(parent_rollout_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="parent rollout or checkpoint not found")
        cursor = copy.deepcopy(parent["cursor"])
        pre = copy.deepcopy(parent["pre_fork_cursor"])
        task_id = parent["task_id"]
        arm = parent["arm"]
        episode = parent.get("episode") or parse_episode({})
        inner_url = parent.get("inner_url")
    else:
        cursor = copy.deepcopy(pin["cursor"])
        pre = copy.deepcopy(pin["pre_fork_cursor"])
        task_id = pin["task_id"]
        arm = pin["arm"]
        episode = pin.get("episode") or parse_episode({})
        inner_url = pin.get("inner_url")
    rollout_id = str(body.get("target_rollout_id") or body.get("rollout_id") or uuid.uuid4())
    run_id = f"resume-{rollout_id[:8]}"
    cursor["run_id"] = run_id
    now = _now()
    record = {
        "rollout_id": rollout_id,
        "task_id": task_id,
        "run_id": run_id,
        "status": "running",
        "success_status": "running",
        "arm": arm,
        "episode": episode,
        "inner_url": inner_url,
        "pre_fork_cursor": pre,
        "cursor": cursor,
        "parent_rollout_id": parent_rollout_id,
        "resume_checkpoint_id": checkpoint_id or None,
        "optimizer_run_id": None,
        "created_at": now,
        "updated_at": now,
        "paused": False,
    }
    STORE.put_rollout(record)
    if OPTIMIZER.live:
        _spawn_live_episode(rollout_id)
    return {"rollout_id": rollout_id, "status": "running", "task_id": task_id}


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    uvicorn.run("gepa_proposer.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
