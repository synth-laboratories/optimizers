"""GEPA-facing orchestrator. Apply levers, talk to policy service + env, return ASI."""

from __future__ import annotations

import argparse
import difflib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from craftax_levers import ENV_PROTOCOL, GEPA_OPTIMIZER_CONTRACT_VERSION
from craftax_levers.apply import apply_unified_diff, apply_whole_file, sha256_text
from craftax_levers.inspect_script import inspect_summary
from craftax_levers.seeds import GREEDY_POLICY, SEED_HARNESS, SEED_POLICY, SEED_PROMPT

Mode = Literal["code", "react"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _task_id(payload: dict[str, Any]) -> tuple[str, str, int]:
    raw = str(payload.get("task_id") or "")
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    nested = str(task.get("task_id") or "")
    if ":" not in raw and ":" in nested:
        raw = nested
    if not raw:
        raw = nested or "train:0"
    split, _, seed_s = raw.partition(":")
    if split not in {"train", "heldout"}:
        split = str(payload.get("split") or task.get("split") or "train")
    seed = int(seed_s) if seed_s.isdigit() else int(payload.get("seed") or task.get("seed") or 0)
    return f"{split}:{seed}", split, seed


def _program_code() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "craftax_code_policy",
        "modules": [
            {
                "module_id": "policy_script",
                "role": "other",
                "content": SEED_POLICY,
                "mutable": True,
                "candidate_field": "policy_script",
                "template_variables": [],
                "metadata": {
                    "lever_kind": "policy_script",
                    "protocol_id": "whole_file.v1",
                    "constraints": {
                        "runtime": "python_source",
                        "entrypoint": "act",
                        "signature": ENV_PROTOCOL,
                        "load": "import",
                        "path": "policy.py",
                    },
                },
            }
        ],
        "target_modules": [
            {
                "module_id": "policy_script",
                "candidate_field": "policy_script",
                "objective": "outcome_reward",
            }
        ],
        "seed_candidate": {"policy_script": SEED_POLICY},
        "rollout_overlay_schema": {"candidate_fields": ["policy_script"]},
        "side_info_schemas": [
            {
                "schema_id": "code_policy_game_trace.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
            {
                "schema_id": "apply_report.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
        ],
        "metadata": {
            "env_protocol": ENV_PROTOCOL,
            "apply_isolation": "serial_restart",
        },
    }


def _program_react() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "craftax_react_policy",
        "modules": [
            {
                "module_id": "react_system_prompt",
                "role": "system",
                "content": SEED_PROMPT,
                "mutable": True,
                "candidate_field": "react_system_prompt",
                "template_variables": [],
                "metadata": {
                    "lever_kind": "system_prompt",
                    "protocol_id": "prompt_overlay.v1",
                },
            },
            {
                "module_id": "harness_module",
                "role": "other",
                "content": SEED_HARNESS,
                "mutable": True,
                "candidate_field": "harness_module",
                "template_variables": [],
                "metadata": {
                    "lever_kind": "harness_module",
                    "protocol_id": "harness_restart.v1",
                    "constraints": {
                        "paths": ["react_loop.py"],
                        "entrypoint": "run_episode",
                        "restart": "process_restart",
                        "instantiate": "in_process_exec",
                        "inspect_route": "/inspect",
                        "apply_isolation": "serial_restart",
                    },
                },
            },
        ],
        "target_modules": [
            {
                "module_id": "react_system_prompt",
                "candidate_field": "react_system_prompt",
                "objective": "outcome_reward",
            },
            {
                "module_id": "harness_module",
                "candidate_field": "harness_module",
                "objective": "outcome_reward",
            },
        ],
        "seed_candidate": {
            "react_system_prompt": SEED_PROMPT,
            "harness_module": SEED_HARNESS,
        },
        "rollout_overlay_schema": {
            "candidate_fields": ["react_system_prompt", "harness_module"]
        },
        "side_info_schemas": [
            {
                "schema_id": "harness_v5_trace.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
            {
                "schema_id": "apply_report.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
            {
                "schema_id": "prompt_trace.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
            {
                "schema_id": "react_script_inspect.v1",
                "when": "terminal_rollout",
                "purpose": "proposer_actionable",
            },
        ],
        "metadata": {
            "env_protocol": ENV_PROTOCOL,
            "apply_isolation": "serial_restart",
        },
    }


def _bundle_value(payload: dict[str, Any], lever_id: str) -> Any:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    if lever_id in candidate:
        return candidate[lever_id]
    bundle = payload.get("lever_bundle") if isinstance(payload.get("lever_bundle"), dict) else {}
    values = bundle.get("values") if isinstance(bundle, dict) else {}
    if isinstance(values, dict) and lever_id in values:
        return values[lever_id]
    return None


def _as_source(value: Any, current: str) -> tuple[str, dict[str, Any], str]:
    """Return (source, apply_report, protocol_id)."""
    if value is None:
        return current, {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": "identity",
        }, "identity"
    if isinstance(value, str):
        report = {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": "whole_file.v1",
            "content_hash": sha256_text(value),
            "base_hash": sha256_text(current),
        }
        return value, report, "whole_file.v1"
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail=f"unsupported lever payload: {type(value)}")
    protocol = str(value.get("protocol_id") or "")
    if "diff" in value:
        protocol = protocol or "unified_diff.v1"
        expected_base = str(value.get("base_hash") or "").strip()
        if expected_base and expected_base != sha256_text(current):
            return current, {
                "schema_id": "apply_report.v1",
                "patch_ok": False,
                "compile_ok": False,
                "restart_ok": True,
                "reject_reason": "base_hash_mismatch",
                "base_hash": sha256_text(current),
                "protocol_id": protocol,
            }, protocol
        try:
            patched = apply_unified_diff(current, str(value.get("diff") or ""))
        except Exception as exc:  # noqa: BLE001
            return current, {
                "schema_id": "apply_report.v1",
                "patch_ok": False,
                "compile_ok": False,
                "restart_ok": True,
                "reject_reason": f"patch_failed:{exc}",
                "protocol_id": protocol,
            }, protocol
        return patched, {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": protocol,
            "base_hash": sha256_text(current),
            "content_hash": sha256_text(patched),
        }, protocol
    protocol = protocol or "whole_file.v1"
    source, report = apply_whole_file(current, value)
    report["protocol_id"] = protocol
    return source, report, protocol


def _as_prompt(value: Any, current: str) -> str:
    if value is None:
        return current
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "prompt"):
            if value.get(key) is not None:
                return str(value[key])
        return current
    return str(value)


@dataclass
class RegisteredCandidate:
    candidate_id: str
    parent_id: str | None
    policy: str
    harness: str
    prompt: str
    reports: list[dict[str, Any]] = field(default_factory=list)
    apply_failed: bool = False


def _mint_candidate_id(payload: dict[str, Any]) -> str:
    raw = str(payload.get("candidate_id") or "").strip()
    if raw:
        return raw
    return f"cand_{uuid.uuid4().hex[:12]}"


def create_app(
    mode: Mode,
    env_url: str,
    policy_url: str,
    *,
    control_url: str | None = None,
    script_path: str | None = None,
) -> FastAPI:
    app = FastAPI(title=f"craftax-orchestrator-{mode}")
    env_url = env_url.rstrip("/")
    policy_url = policy_url.rstrip("/")
    control_url = (control_url or "").rstrip("/") or None
    script_file = Path(script_path) if script_path else None
    lock = threading.Lock()
    current_policy = SEED_POLICY
    current_harness = SEED_HARNESS
    current_prompt = SEED_PROMPT
    active_id: str | None = "seed"
    registry: dict[str, RegisteredCandidate] = {
        "seed": RegisteredCandidate(
            candidate_id="seed",
            parent_id=None,
            policy=SEED_POLICY,
            harness=SEED_HARNESS,
            prompt=SEED_PROMPT,
        )
    }
    rollouts: dict[str, dict[str, Any]] = {}

    def program() -> dict[str, Any]:
        return _program_code() if mode == "code" else _program_react()

    def _timeout() -> float:
        return 180.0 if mode == "react" else 40.0

    def _configure_code(client: httpx.Client, source: str) -> dict[str, Any]:
        loaded = client.post(f"{policy_url}/load", json={"source": source}).json()
        return loaded

    def _configure_react(
        client: httpx.Client,
        prompt: str,
        harness: str,
        *,
        restart_harness: bool,
    ) -> dict[str, Any]:
        restarted: dict[str, Any] = {"restart_ok": True, "compile_ok": True, "restart_ms": 0.0}
        if restart_harness:
            if script_file is not None:
                script_file.parent.mkdir(parents=True, exist_ok=True)
                script_file.write_text(harness, encoding="utf-8")
            env_before = client.get(f"{env_url}/health").json()
            if control_url:
                restarted = client.post(f"{control_url}/restart_policy", json={}).json()
            else:
                restarted = client.post(
                    f"{policy_url}/restart",
                    json={"source": harness, "protocol_id": "harness_restart.v1"},
                ).json()
            env_after = client.get(f"{env_url}/health").json()
            restarted["env_pid"] = env_after.get("pid")
            restarted["env_untouched"] = env_before.get("pid") == env_after.get("pid")
            restarted["policy_pid_before"] = restarted.get("old_pid")
            restarted["policy_pid_after"] = restarted.get("new_pid") or restarted.get("pid")
        else:
            client.post(f"{policy_url}/reload", json={"prompt_overlay": prompt})
        return restarted

    def _ensure_active(client: httpx.Client, candidate: RegisteredCandidate) -> None:
        nonlocal active_id
        if candidate.apply_failed:
            return
        if active_id == candidate.candidate_id:
            return
        if mode == "code":
            loaded = _configure_code(client, candidate.policy)
            if not loaded.get("compile_ok"):
                raise HTTPException(status_code=500, detail=f"reload registered candidate failed: {loaded}")
        else:
            live = registry.get(active_id) if active_id else None
            restart = live is None or live.harness != candidate.harness
            _configure_react(client, candidate.prompt, candidate.harness, restart_harness=restart)
        active_id = candidate.candidate_id

    def _apply_payload(
        payload: dict[str, Any],
        base: RegisteredCandidate,
    ) -> RegisteredCandidate:
        nonlocal current_policy, current_harness, current_prompt, active_id
        reports: list[dict[str, Any]] = []
        apply_failed = False
        policy = base.policy
        harness = base.harness
        prompt = base.prompt
        candidate_id = _mint_candidate_id(payload)

        with httpx.Client(timeout=_timeout()) as client:
            if mode == "code":
                source, report, _protocol = _as_source(_bundle_value(payload, "policy_script"), base.policy)
                reports.append({**report, "lever_ids": ["policy_script"]})
                if not report.get("patch_ok"):
                    apply_failed = True
                else:
                    loaded = _configure_code(client, source)
                    report["compile_ok"] = bool(loaded.get("compile_ok"))
                    reports[-1] = {**report, "lever_ids": ["policy_script"]}
                    if not loaded.get("compile_ok"):
                        apply_failed = True
                        reports[-1]["reject_reason"] = loaded.get("error")
                    else:
                        policy = source
                        current_policy = source
                        active_id = candidate_id
            else:
                prompt_value = _bundle_value(payload, "react_system_prompt")
                prompt = _as_prompt(prompt_value, base.prompt)
                if prompt_value is not None:
                    reports.append(
                        {
                            "schema_id": "apply_report.v1",
                            "lever_ids": ["react_system_prompt"],
                            "protocol_id": "prompt_overlay.v1",
                            "patch_ok": True,
                            "compile_ok": True,
                            "restart_ok": True,
                        }
                    )
                harness_value = _bundle_value(payload, "harness_module")
                restart_harness = harness_value is not None
                if restart_harness:
                    source, report, protocol = _as_source(harness_value, base.harness)
                    if not report.get("patch_ok"):
                        apply_failed = True
                        reports.append({**report, "lever_ids": ["harness_module"]})
                    else:
                        harness = source
                        restarted = _configure_react(
                            client, prompt, harness, restart_harness=True
                        )
                        report["restart_ok"] = bool(restarted.get("restart_ok"))
                        report["compile_ok"] = bool(restarted.get("compile_ok", True))
                        report["restart_ms"] = restarted.get("restart_ms")
                        report["policy_pid_before"] = restarted.get("policy_pid_before") or restarted.get("old_pid")
                        report["policy_pid_after"] = restarted.get("policy_pid_after") or restarted.get("new_pid") or restarted.get("pid")
                        report["env_pid"] = restarted.get("env_pid")
                        report["env_untouched"] = restarted.get("env_untouched")
                        reports.append(
                            {
                                **report,
                                "lever_ids": ["harness_module"],
                                "protocol_id": "harness_restart.v1",
                            }
                        )
                        if not restarted.get("restart_ok"):
                            apply_failed = True
                        else:
                            current_harness = harness
                            current_prompt = prompt
                            active_id = candidate_id
                else:
                    _configure_react(client, prompt, harness, restart_harness=False)
                    current_prompt = prompt
                    active_id = candidate_id

        registered = RegisteredCandidate(
            candidate_id=candidate_id,
            parent_id=str(payload.get("parent_id") or base.candidate_id or "") or None,
            policy=policy,
            harness=harness,
            prompt=prompt,
            reports=reports,
            apply_failed=apply_failed,
        )
        if not apply_failed:
            registry[candidate_id] = registered
        return registered

    def _run_episode(
        client: httpx.Client,
        candidate: RegisteredCandidate,
        seed: int,
    ) -> dict[str, Any]:
        if candidate.apply_failed:
            if mode == "code":
                return {"reward": 0.0, "ticks": [], "events": [], "achievements": [], "compile_ok": False}
            return {"reward": 0.0, "events": [], "achievements": [], "tool_calls": []}
        _ensure_active(client, candidate)
        if mode == "code":
            return client.post(f"{policy_url}/episode", json={"env_url": env_url, "seed": seed}).json()
        return client.post(
            f"{policy_url}/episode",
            json={"env_url": env_url, "seed": seed, "prompt_overlay": candidate.prompt},
        ).json()

    def _terminal(
        *,
        rollout_id: str,
        task_id: str,
        split: str,
        seed: int,
        candidate: RegisteredCandidate,
        episode: dict[str, Any],
        extra_side_info: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now()
        side_info = list(candidate.reports) + extra_side_info
        apply_failed = candidate.apply_failed
        reward = 0.0 if apply_failed else float(episode.get("reward") or 0.0)
        events = list(episode.get("events") or [])
        status = "failed" if apply_failed else "completed"
        success = "apply_failed" if apply_failed else "succeeded"
        body = {
            "rollout_id": rollout_id,
            "status": status,
            "success_status": success,
            "task_id": task_id,
            "seed": seed,
            "split": split,
            "candidate_id": candidate.candidate_id,
            "reward": reward,
            "reward_info": {
                "outcome_reward": reward,
                "metrics": {
                    "achievements": episode.get("achievements") or [],
                    "apply_failed": apply_failed,
                },
            },
            "summary": {
                "outcome_reward": reward,
                "achievements": episode.get("achievements") or [],
                "mode": mode,
            },
            "usage": {},
            "trace": {"schema_version": "trace.v1", "event_history": events},
            "side_info": side_info,
            "actionable_side_info": side_info,
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
            "metadata": {"env_protocol": ENV_PROTOCOL, "mode": mode},
        }
        rollouts[rollout_id] = body
        return body

    def _episode_side_info(candidate: RegisteredCandidate, episode: dict[str, Any]) -> list[dict[str, Any]]:
        if mode == "code":
            trace_body = {
                "ticks": episode.get("ticks") or [],
                "deaths": [episode["death_cause"]] if episode.get("death_cause") else [],
                "compile_ok": bool(episode.get("compile_ok", True)) and not candidate.apply_failed,
                "achievements": episode.get("achievements") or [],
            }
            return [
                {
                    "schema_id": "code_policy_game_trace.v1",
                    "lever_ids": ["policy_script"],
                    "summary": {
                        "ticks": len(trace_body["ticks"]),
                        "deaths": trace_body["deaths"],
                        "compile_ok": trace_body["compile_ok"],
                        "achievements": trace_body["achievements"],
                    },
                    "body": trace_body,
                }
            ]
        inspected: dict[str, Any] = {}
        try:
            inspected = httpx.get(f"{policy_url}/inspect", timeout=5.0).json()
        except Exception:  # noqa: BLE001
            inspected = {}
        inspect_body = {key: value for key, value in inspected.items() if key != "source"}
        restart_ms = float(episode.get("restart_ms") or 0.0)
        for report in candidate.reports:
            if report.get("restart_ms") is not None:
                restart_ms = float(report.get("restart_ms") or 0.0)
        return [
            {
                "schema_id": "react_script_inspect.v1",
                "lever_ids": ["harness_module"],
                "summary": inspect_summary(inspected) if inspected else {},
                "body": inspect_body,
            },
            {
                "schema_id": "prompt_trace.v1",
                "lever_ids": ["react_system_prompt"],
                "summary": {"prompt": candidate.prompt[:120]},
                "body": {"prompt": candidate.prompt},
            },
            {
                "schema_id": "harness_v5_trace.v1",
                "lever_ids": ["harness_module", "react_system_prompt"],
                "summary": {
                    "tool_calls": len(episode.get("tool_calls") or []),
                    "restart_ms": restart_ms,
                    "architecture": episode.get("architecture"),
                    "policy_pid": episode.get("policy_pid"),
                    "llm_provider": episode.get("llm_provider"),
                    "llm_calls": episode.get("llm_calls"),
                    "achievements": episode.get("achievements") or [],
                    "inspect": inspect_summary(inspected) if inspected else {},
                },
                "body": {
                    "tool_calls": episode.get("tool_calls") or [],
                    "events": episode.get("events") or [],
                },
            },
        ]

    @app.get("/health")
    def health() -> dict[str, Any]:
        env_ok = httpx.get(f"{env_url}/health", timeout=5.0).json()
        policy_ok = httpx.get(f"{policy_url}/health", timeout=5.0).json()
        return {
            "status": "ok",
            "contract_version": GEPA_OPTIMIZER_CONTRACT_VERSION,
            "mode": mode,
            "env": env_ok,
            "policy": policy_ok,
            "registered_candidates": len(registry),
        }

    @app.get("/inspect")
    def inspect_policy() -> dict[str, Any]:
        """Proxy the live ReAct module currently exec'd in the policy process."""
        return httpx.get(f"{policy_url}/inspect", timeout=5.0).json()

    @app.get("/metadata")
    @app.get("/info")
    def metadata() -> dict[str, Any]:
        body = program()
        return {
            "runtime": {
                "runtime_id": f"craftax_{mode}",
                "name": f"Craftax {mode} GEPA container",
                "description": "Split env + policy service for custom lever protocols.",
            },
            "capabilities": {
                "contract_version": "container_contract.v1",
                "rollout_modes": ["blocking", "sync"],
                "metadata": {"policy_ready": True},
            },
            "metadata": {
                "optimizer_contracts": {
                    "gepa": {
                        "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                        "program_route": "/program",
                        "taskset_route": "/taskset",
                        "taskset_tasks_route": "/taskset/tasks",
                        "candidates_route": "/candidates",
                        "rollout_route": "/rollout",
                    }
                },
                "env_protocol": ENV_PROTOCOL,
                "side_info_schemas": body.get("side_info_schemas"),
                "apply_isolation": "serial_restart",
            },
        }

    @app.get("/program")
    def program_route() -> dict[str, Any]:
        return program()

    @app.get("/taskset")
    def taskset() -> dict[str, Any]:
        return {"taskset_id": f"craftax_{mode}", "splits": {"train": 4, "heldout": 2}}

    @app.post("/taskset/tasks")
    async def taskset_tasks(request: Request) -> dict[str, Any]:
        payload = await request.json()
        tasks = []
        for raw in payload.get("task_ids") or []:
            task_id = str(raw)
            split, _, seed_s = task_id.partition(":")
            seed = int(seed_s) if seed_s.isdigit() else 0
            tasks.append({"task_id": task_id, "split": split or "train", "seed": seed})
        return {"tasks": tasks}

    @app.post("/candidates")
    async def register_candidate(request: Request) -> dict[str, Any]:
        payload = await request.json()
        parent_id = str(payload.get("parent_id") or "seed").strip() or "seed"
        with lock:
            base = registry.get(parent_id) or registry["seed"]
            registered = _apply_payload(payload, base)
        apply_ok = not registered.apply_failed
        body = {
            "candidate_id": registered.candidate_id if apply_ok else None,
            "parent_id": registered.parent_id,
            "apply_ok": apply_ok,
            "apply_report": registered.reports[0] if registered.reports else {
                "schema_id": "apply_report.v1",
                "patch_ok": apply_ok,
            },
            "apply_reports": registered.reports,
            "base_hash": sha256_text(registered.policy if mode == "code" else registered.harness),
        }
        if not apply_ok:
            body["side_info"] = registered.reports
        return body

    @app.get("/candidates/{candidate_id}")
    def get_candidate(candidate_id: str) -> dict[str, Any]:
        candidate = registry.get(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail=f"unknown candidate_id {candidate_id}")
        return {
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "apply_ok": not candidate.apply_failed,
            "apply_reports": candidate.reports,
        }

    @app.get("/rollouts/{rollout_id}")
    @app.get("/rollouts/{rollout_id}/state")
    def get_rollout(rollout_id: str) -> dict[str, Any]:
        body = rollouts.get(rollout_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"unknown rollout_id {rollout_id}")
        return body

    @app.post("/rollout")
    @app.post("/rollouts")
    async def rollout(request: Request) -> dict[str, Any]:
        payload = await request.json()
        task_id, split, seed = _task_id(payload)
        rollout_id = str(payload.get("rollout_id") or f"craftax_{uuid.uuid4().hex[:12]}")
        requested_id = str(payload.get("candidate_id") or payload.get("metadata", {}).get("candidate_id") or "").strip()
        has_inline = bool(
            _bundle_value(payload, "policy_script") is not None
            or _bundle_value(payload, "react_system_prompt") is not None
            or _bundle_value(payload, "harness_module") is not None
        )
        with lock:
            if requested_id and not has_inline:
                candidate = registry.get(requested_id)
                if candidate is None:
                    raise HTTPException(status_code=404, detail=f"unknown candidate_id {requested_id}")
            elif has_inline:
                parent_id = str(payload.get("parent_id") or requested_id or "seed").strip() or "seed"
                base = registry.get(parent_id) or RegisteredCandidate(
                    candidate_id="live",
                    parent_id=None,
                    policy=current_policy,
                    harness=current_harness,
                    prompt=current_prompt,
                )
                if requested_id:
                    payload = {**payload, "candidate_id": requested_id}
                candidate = _apply_payload(payload, base)
            else:
                candidate = registry["seed"]
            with httpx.Client(timeout=_timeout()) as client:
                episode = _run_episode(client, candidate, seed)
            extra = _episode_side_info(candidate, episode)
            return _terminal(
                rollout_id=rollout_id,
                task_id=task_id,
                split=split,
                seed=seed,
                candidate=candidate,
                episode=episode,
                extra_side_info=extra,
            )

    return app


def diff_seed_to_greedy_policy() -> str:
    return "".join(
        difflib.unified_diff(
            SEED_POLICY.splitlines(keepends=True),
            GREEDY_POLICY.splitlines(keepends=True),
            fromfile="a/policy.py",
            tofile="b/policy.py",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["code", "react"], default="code")
    parser.add_argument("--env-url", default="http://127.0.0.1:19101")
    parser.add_argument("--policy-url", default="http://127.0.0.1:19102")
    parser.add_argument("--control-url", default="")
    parser.add_argument("--script-path", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    args = parser.parse_args()
    app = create_app(
        args.mode,
        args.env_url,
        args.policy_url,
        control_url=args.control_url or None,
        script_path=args.script_path or None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
