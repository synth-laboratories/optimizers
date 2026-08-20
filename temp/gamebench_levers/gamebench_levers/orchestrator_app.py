"""GEPA plane A for one (game, mode) pair.

Register-then-run: `POST /candidates` applies the lever once (write + load, or
write + restart the policy process), `POST /rollout` names an already-configured
candidate and one task. Apply cost is paid per candidate, not per seed, and a
failed apply never yields a candidate_id that can be rolled out.
"""

from __future__ import annotations

import argparse
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from gamebench_levers import ENV_PROTOCOL, GEPA_OPTIMIZER_CONTRACT_VERSION
from gamebench_levers.apply import apply_unified_diff, apply_whole_file, sha256_text
from gamebench_levers.inspect_script import inspect_summary
from gamebench_levers.seeds import code_seed, harness_seed, prompt_seed

Mode = Literal["code", "harness"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bundle_value(payload: dict[str, Any], lever_id: str) -> Any:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    if lever_id in candidate:
        return candidate[lever_id]
    bundle = payload.get("lever_bundle") if isinstance(payload.get("lever_bundle"), dict) else {}
    values = bundle.get("values") if isinstance(bundle, dict) else None
    if isinstance(values, dict) and lever_id in values:
        return values[lever_id]
    return None


def _as_source(value: Any, current: str) -> tuple[str, dict[str, Any], str]:
    """Resolve a lever payload to source text plus an apply_report.v1."""
    if value is None:
        return current, {
            "schema_id": "apply_report.v1", "patch_ok": True, "compile_ok": True,
            "restart_ok": True, "protocol_id": "identity",
        }, "identity"
    if isinstance(value, str):
        return value, {
            "schema_id": "apply_report.v1", "patch_ok": True, "compile_ok": True, "restart_ok": True,
            "protocol_id": "whole_file.v1", "content_hash": sha256_text(value), "base_hash": sha256_text(current),
        }, "whole_file.v1"
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail=f"unsupported lever payload: {type(value)}")
    protocol = str(value.get("protocol_id") or "")
    if "diff" in value:
        protocol = protocol or "unified_diff.v1"
        expected = str(value.get("base_hash") or "").strip()
        if expected and expected != sha256_text(current):
            return current, {
                "schema_id": "apply_report.v1", "patch_ok": False, "compile_ok": False, "restart_ok": True,
                "reject_reason": "base_hash_mismatch", "base_hash": sha256_text(current), "protocol_id": protocol,
            }, protocol
        try:
            patched = apply_unified_diff(current, str(value.get("diff") or ""))
        except Exception as exc:  # noqa: BLE001
            return current, {
                "schema_id": "apply_report.v1", "patch_ok": False, "compile_ok": False, "restart_ok": True,
                "reject_reason": f"patch_failed:{exc}", "protocol_id": protocol,
            }, protocol
        return patched, {
            "schema_id": "apply_report.v1", "patch_ok": True, "compile_ok": True, "restart_ok": True,
            "protocol_id": protocol, "base_hash": sha256_text(current), "content_hash": sha256_text(patched),
        }, protocol
    protocol = protocol or "whole_file.v1"
    source, report = apply_whole_file(current, value)
    report["protocol_id"] = protocol
    return source, report, protocol


def compile_report(source: str, filename: str) -> dict[str, Any]:
    """Compile a candidate before applying it.

    A syntax error caught here yields the message, the line number and the offending
    line. Discovering it via a failed process restart instead yields only
    "restart_failed", which tells a proposer nothing it can act on -- and the
    proposer's most common mistake is exactly this: a raw newline inside a quoted
    string.
    """
    try:
        compile(source, filename, "exec")
    except SyntaxError as exc:
        lines = source.splitlines()
        lineno = exc.lineno or 0
        return {
            "compile_ok": False,
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc.msg} (line {lineno})",
            "lineno": lineno,
            "offset": exc.offset,
            "source_line": lines[lineno - 1][:200] if 0 < lineno <= len(lines) else None,
            "context": lines[max(0, lineno - 3) : lineno + 2],
        }
    except Exception as exc:  # noqa: BLE001
        return {"compile_ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"compile_ok": True}


def _as_prompt(value: Any, current: str) -> str:
    if value is None:
        return current
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "prompt"):
            if value.get(key) is not None:
                return str(value[key])
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
    worker_url: str | None = None


def create_app(
    game: str,
    mode: Mode,
    env_url: str,
    policy_url: str,
    *,
    control_url: str | None = None,
    script_path: str | None = None,
    train_seeds: tuple[int, ...] = (),
    heldout_seeds: tuple[int, ...] = (),
    max_steps: int | None = None,
    isolation: str = "serial_restart",
) -> FastAPI:
    app = FastAPI(title=f"gamebench-gepa-{game}-{mode}")
    env_url = env_url.rstrip("/")
    policy_url = policy_url.rstrip("/")
    control_url = (control_url or "").rstrip("/") or None
    script_file = Path(script_path) if script_path else None
    lock = threading.Lock()

    seed_policy = code_seed(game)
    seed_harness = harness_seed(game)
    seed_prompt = prompt_seed(game)
    env_context: dict[str, Any] = {}

    def _env_context() -> dict[str, Any]:
        """Env spec + one real observation, cached.

        The proposer only ever sees `/program`. Without the action space, the
        glyph legend and a worked example of the observation, it has to guess the
        schema its policy will be called with -- which is why a Craftax candidate
        that never names a real action scores exactly what the noop seed scores.
        """
        if env_context:
            return env_context
        try:
            spec = httpx.get(f"{env_url}/spec", timeout=30.0).json()
        except Exception:  # noqa: BLE001
            return {}
        sample: dict[str, Any] = {}
        try:
            seed = (spec.get("train_seeds") or [0])[0]
            body: dict[str, Any] = {"seed": seed, "split": "train"}
            if max_steps is not None:
                body["max_steps"] = max_steps
            sample = httpx.post(f"{env_url}/reset", json=body, timeout=60.0).json().get("obs") or {}
        except Exception:  # noqa: BLE001
            sample = {}
        env_context.update({"spec": spec, "sample_observation": sample})
        return env_context

    active_id: str | None = "seed"
    pooled = mode == "harness" and isolation == "per_candidate_worker" and control_url is not None
    registry: dict[str, RegisteredCandidate] = {
        "seed": RegisteredCandidate(
            "seed", None, seed_policy, seed_harness, seed_prompt,
            worker_url=policy_url if pooled else None,
        )
    }
    rollouts: dict[str, dict[str, Any]] = {}
    # Typed side info is addressable in its own right. It still rides on the terminal
    # record (the engine sensor reads `actionable_side_info` from there), but the
    # engine stores that record as an opaque `raw_response` blob, so the only way to
    # read a trace afterwards was to re-parse it. `/asi` is that read path.
    asi_store: dict[str, dict[str, Any]] = {}

    def _seeds(split: str) -> tuple[int, ...]:
        return heldout_seeds if split == "heldout" else train_seeds

    def _task_id(payload: dict[str, Any]) -> tuple[str, str, int]:
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        raw = str(payload.get("task_id") or task.get("task_id") or "")
        if not raw:
            raw = "train:0"
        split, _, index_s = raw.partition(":")
        if split not in {"train", "heldout"}:
            split = str(payload.get("split") or task.get("split") or "train")
            index_s = index_s or "0"
        index = int(index_s) if index_s.isdigit() else 0
        pool = _seeds(split)
        seed = pool[index % len(pool)] if pool else index
        return f"{split}:{index}", split, seed

    def _program() -> dict[str, Any]:
        context = _env_context()
        spec = context.get("spec") or {}
        sample = context.get("sample_observation") or {}
        contract = {
            "objective": spec.get("objective"),
            "action_space": spec.get("action_space"),
            "action_type": spec.get("action_type"),
            "achievements": spec.get("achievements"),
            "max_horizon": spec.get("max_horizon"),
            "observation_notes": spec.get("observation_notes"),
            "sample_observation": sample,
        }
        if mode == "code":
            modules = [{
                "module_id": "policy_script",
                "role": "other",
                "content": seed_policy,
                "mutable": True,
                "candidate_field": "policy_script",
                "template_variables": [],
                "metadata": {
                    "lever_kind": "policy_script",
                    "protocol_id": "unified_diff.v1",
                    "constraints": {
                        "runtime": "python_source",
                        "entrypoint": "act",
                        "signature": ENV_PROTOCOL,
                        "load": "import",
                        "path": "policy.py",
                        "reload": "per_candidate_load",
                        "returns": "one action from action_space per call",
                        **contract,
                    },
                },
            }]
            targets = [{"module_id": "policy_script", "candidate_field": "policy_script", "objective": "outcome_reward"}]
            seed_candidate = {"policy_script": seed_policy}
            schemas = ["code_policy_game_trace.v1", "apply_report.v1", "episode_verdict.v1"]
        else:
            modules = [
                {
                    "module_id": "harness_module",
                    "role": "other",
                    "content": seed_harness,
                    "mutable": True,
                    "candidate_field": "harness_module",
                    "template_variables": [],
                    "metadata": {
                        "lever_kind": "harness_module",
                        "protocol_id": "harness_restart.v1",
                        "constraints": {
                            "paths": ["harness.py"],
                            "entrypoint": "run_episode",
                            "restart": "process_restart",
                            "apply_isolation": "per_candidate_worker" if pooled else "serial_restart",
                            "inspect_route": "/inspect",
                            "returns": "a dict with a numeric `reward` key",
                            "env_client": "env.reset(seed, max_steps) -> obs; env.step(action) -> {obs, reward, terminated, truncated, info}",
                            **contract,
                        },
                    },
                },
                {
                    "module_id": "system_prompt",
                    "role": "system",
                    "content": seed_prompt,
                    "mutable": True,
                    "candidate_field": "system_prompt",
                    "template_variables": [],
                    "metadata": {"lever_kind": "system_prompt", "protocol_id": "prompt_overlay.v1"},
                },
            ]
            targets = [
                {"module_id": "harness_module", "candidate_field": "harness_module", "objective": "outcome_reward"},
                {"module_id": "system_prompt", "candidate_field": "system_prompt", "objective": "outcome_reward"},
            ]
            seed_candidate = {"harness_module": seed_harness, "system_prompt": seed_prompt}
            schemas = [
                "speedrunner_trace.v1", "harness_inspect.v1", "prompt_trace.v1",
                "apply_report.v1", "episode_verdict.v1",
            ]
        return {
            "version": "prompt_program.v1",
            "program_id": f"gamebench_{game}_{mode}",
            "modules": modules,
            "target_modules": targets,
            "seed_candidate": seed_candidate,
            "rollout_overlay_schema": {"candidate_fields": [m["candidate_field"] for m in modules]},
            "side_info_schemas": [
                {"schema_id": s, "when": "terminal_rollout", "purpose": "proposer_actionable"} for s in schemas
            ],
            "metadata": {
                "game": game,
                "mode": mode,
                "env_protocol": ENV_PROTOCOL,
                "apply_isolation": "per_candidate_worker" if pooled else "serial_restart",
                "gepa_candidates_route": "/candidates",
                "asi_route": "/asi",
                "env_spec": spec,
                "sample_observation": sample,
            },
        }

    # -- configure ------------------------------------------------------
    def _load_code(client: httpx.Client, source: str) -> dict[str, Any]:
        return client.post(f"{policy_url}/load", json={"source": source}, timeout=60.0).json()

    def _restart_harness(client: httpx.Client, harness: str, prompt: str) -> dict[str, Any]:
        if script_file is not None:
            script_file.parent.mkdir(parents=True, exist_ok=True)
            script_file.write_text(harness, encoding="utf-8")
        env_before = client.get(f"{env_url}/health").json()
        if control_url:
            result = client.post(f"{control_url}/restart_policy", json={}, timeout=60.0).json()
        else:
            result = client.post(
                f"{policy_url}/restart",
                json={"source": harness, "protocol_id": "harness_restart.v1"},
                timeout=60.0,
            ).json()
        env_after = client.get(f"{env_url}/health").json()
        result["env_pid"] = env_after.get("pid")
        result["env_untouched"] = env_before.get("pid") == env_after.get("pid")
        client.post(f"{policy_url}/reload", json={"prompt_overlay": prompt}, timeout=30.0)
        return result

    def _ensure_worker(client: httpx.Client, candidate: RegisteredCandidate) -> dict[str, Any]:
        """Get (or respawn) this candidate's own policy worker. No restart on switch."""
        result = client.post(
            f"{control_url}/workers",
            json={"candidate_id": candidate.candidate_id, "source": candidate.harness},
            timeout=120.0,
        ).json()
        if result.get("url"):
            candidate.worker_url = str(result["url"])
        return result

    def _ensure_active(client: httpx.Client, candidate: RegisteredCandidate) -> None:
        """Reconfigure only when the live process is serving a different candidate."""
        nonlocal active_id
        if candidate.apply_failed:
            return
        if pooled:
            # Switching candidates is a routing decision; the worker stays warm.
            _ensure_worker(client, candidate)
            client.post(
                f"{candidate.worker_url}/reload",
                json={"prompt_overlay": candidate.prompt},
                timeout=30.0,
            )
            active_id = candidate.candidate_id
            return
        if active_id == candidate.candidate_id:
            return
        if mode == "code":
            loaded = _load_code(client, candidate.policy)
            if not loaded.get("compile_ok"):
                raise HTTPException(status_code=500, detail=f"reload of registered candidate failed: {loaded}")
        else:
            live = registry.get(active_id) if active_id else None
            if live is None or live.harness != candidate.harness:
                _restart_harness(client, candidate.harness, candidate.prompt)
            else:
                client.post(f"{policy_url}/reload", json={"prompt_overlay": candidate.prompt}, timeout=30.0)
        active_id = candidate.candidate_id

    def _register(payload: dict[str, Any]) -> RegisteredCandidate:
        nonlocal active_id
        parent = registry.get(str(payload.get("parent_id") or "seed")) or registry["seed"]
        candidate_id = str(payload.get("candidate_id") or "").strip() or f"cand_{uuid.uuid4().hex[:12]}"
        # Candidates are immutable once registered. Re-applying the same id would let
        # one candidate resolve to two different policies and score a row twice.
        existing = registry.get(candidate_id) or registry.get(f"failed::{candidate_id}")
        if existing is not None:
            return existing
        reports: list[dict[str, Any]] = []
        apply_failed = False
        policy, harness, prompt = parent.policy, parent.harness, parent.prompt

        with httpx.Client(timeout=180.0) as client:
            if mode == "code":
                source, report, _ = _as_source(_bundle_value(payload, "policy_script"), parent.policy)
                report["lever_ids"] = ["policy_script"]
                checked = compile_report(source, "policy.py")
                if not report.get("patch_ok"):
                    apply_failed = True
                elif not checked["compile_ok"]:
                    apply_failed = True
                    report["compile_ok"] = False
                    report["reject_reason"] = checked["error"]
                    report["compile_diagnostics"] = checked
                else:
                    loaded = _load_code(client, source)
                    report["compile_ok"] = bool(loaded.get("compile_ok"))
                    if not loaded.get("compile_ok"):
                        apply_failed = True
                        report["reject_reason"] = loaded.get("error")
                    else:
                        policy = source
                        active_id = candidate_id
                reports.append(report)
            else:
                prompt_value = _bundle_value(payload, "system_prompt")
                prompt = _as_prompt(prompt_value, parent.prompt)
                if prompt_value is not None:
                    reports.append({
                        "schema_id": "apply_report.v1", "lever_ids": ["system_prompt"],
                        "protocol_id": "prompt_overlay.v1", "patch_ok": True, "compile_ok": True, "restart_ok": True,
                    })
                harness_value = _bundle_value(payload, "harness_module")
                if harness_value is not None:
                    source, report, _ = _as_source(harness_value, parent.harness)
                    report["lever_ids"] = ["harness_module"]
                    report["protocol_id"] = "harness_restart.v1"
                    checked = compile_report(source, "harness.py")
                    if not report.get("patch_ok"):
                        apply_failed = True
                    elif not checked["compile_ok"]:
                        # Do not write or restart: nothing to roll back, and the
                        # proposer gets the real error instead of "restart_failed".
                        apply_failed = True
                        report["compile_ok"] = False
                        report["restart_ok"] = False
                        report["reject_reason"] = checked["error"]
                        report["compile_diagnostics"] = checked
                    else:
                        harness = source
                        if pooled:
                            spawned = _ensure_worker(
                                client,
                                RegisteredCandidate(candidate_id, parent.candidate_id, policy, harness, prompt),
                            )
                            restarted = {
                                "restart_ok": bool(spawned.get("ok")),
                                "compile_ok": bool(spawned.get("compile_ok", spawned.get("ok"))),
                                "compile_error": spawned.get("compile_error"),
                                "error": spawned.get("error"),
                                "new_pid": spawned.get("pid"),
                                "worker_url": spawned.get("url"),
                                "env_untouched": spawned.get("env_untouched"),
                                "restart_ms": 0.0,
                                "isolation": "per_candidate_worker",
                            }
                        else:
                            restarted = _restart_harness(client, harness, prompt)
                        report["restart_ok"] = bool(restarted.get("restart_ok"))
                        report["compile_ok"] = bool(restarted.get("compile_ok", True))
                        report["restart_ms"] = restarted.get("restart_ms")
                        report["policy_pid_after"] = restarted.get("new_pid") or restarted.get("pid")
                        report["env_untouched"] = restarted.get("env_untouched")
                        report["isolation"] = restarted.get("isolation", "serial_restart")
                        if restarted.get("worker_url"):
                            report["worker_url"] = restarted["worker_url"]
                        if not report["restart_ok"] or not report["compile_ok"]:
                            apply_failed = True
                            detail = (
                                restarted.get("compile_error")
                                or restarted.get("error")
                                or "restart_failed"
                            )
                            stderr = str(restarted.get("stderr") or "").strip()
                            report["reject_reason"] = f"{detail} :: {stderr[-600:]}" if stderr else detail
                            # Never leave a half-applied tree: put the parent back and
                            # bring the policy process up on it.
                            harness = parent.harness
                            if pooled:
                                # Nothing shared was mutated: the parent's worker is untouched.
                                client.delete(f"{control_url}/workers/{candidate_id}", timeout=30.0)
                                report["rolled_back"] = True
                                report["isolation"] = "per_candidate_worker"
                            else:
                                rollback = _restart_harness(client, parent.harness, parent.prompt)
                                report["rolled_back"] = bool(rollback.get("restart_ok"))
                                active_id = parent.candidate_id if rollback.get("restart_ok") else None
                        else:
                            active_id = candidate_id
                    reports.append(report)
                else:
                    client.post(f"{policy_url}/reload", json={"prompt_overlay": prompt}, timeout=30.0)
                    active_id = candidate_id

        registered = RegisteredCandidate(
            candidate_id=candidate_id,
            parent_id=parent.candidate_id,
            policy=policy,
            harness=harness,
            prompt=prompt,
            reports=reports,
            apply_failed=apply_failed,
            worker_url=next(
                (r.get("worker_url") for r in reports if r.get("worker_url")),
                parent.worker_url if pooled else None,
            ),
        )
        if not apply_failed:
            registry[candidate_id] = registered
        return registered

    # -- routes ---------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        env = httpx.get(f"{env_url}/health", timeout=10.0).json()
        policy = httpx.get(f"{policy_url}/health", timeout=10.0).json()
        return {
            "status": "ok",
            "contract_version": GEPA_OPTIMIZER_CONTRACT_VERSION,
            "game": game, "mode": mode, "env": env, "policy": policy,
        }

    @app.get("/metadata")
    @app.get("/info")
    def metadata() -> dict[str, Any]:
        program = _program()
        return {
            "runtime": {"runtime_id": f"gamebench_{game}_{mode}", "name": f"GameBench {game} ({mode})"},
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
                        "rollout_route": "/rollout",
                        "candidates_route": "/candidates",
                        "asi_route": "/asi",
                    }
                },
                "env_protocol": ENV_PROTOCOL,
                "side_info_schemas": program.get("side_info_schemas"),
                "apply_isolation": "per_candidate_worker" if pooled else "serial_restart",
                "gepa_candidates_route": "/candidates",
                "asi_route": "/asi",
                "asi_schemas_route": "/asi/schemas",
            },
        }

    @app.get("/program")
    def program() -> dict[str, Any]:
        return _program()

    @app.get("/taskset")
    def taskset() -> dict[str, Any]:
        return {
            "taskset_id": f"gamebench_{game}",
            "splits": {"train": len(train_seeds), "heldout": len(heldout_seeds)},
        }

    @app.post("/taskset/tasks")
    async def taskset_tasks(request: Request) -> dict[str, Any]:
        payload = await request.json()
        tasks = []
        for raw in payload.get("task_ids") or []:
            task_id, split, seed = _task_id({"task_id": str(raw)})
            tasks.append({"task_id": task_id, "split": split, "seed": seed})
        return {"tasks": tasks}

    @app.post("/candidates")
    async def register_candidate(request: Request) -> dict[str, Any]:
        payload = await request.json()
        with lock:
            candidate = _register(payload)
        body = {
            "candidate_id": None if candidate.apply_failed else candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "apply_report": candidate.reports,
            "base_hash": sha256_text(candidate.policy if mode == "code" else candidate.harness),
            "status": "apply_failed" if candidate.apply_failed else "registered",
        }
        if candidate.apply_failed:
            body["failed_candidate_id"] = candidate.candidate_id
            registry[f"failed::{candidate.candidate_id}"] = candidate
        return body

    @app.get("/candidates/{candidate_id}")
    def get_candidate(candidate_id: str) -> dict[str, Any]:
        candidate = registry.get(candidate_id) or registry.get(f"failed::{candidate_id}")
        if candidate is None:
            raise HTTPException(status_code=404, detail="unknown candidate")
        return {
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "apply_failed": candidate.apply_failed,
            "apply_report": candidate.reports,
        }

    @app.post("/rollout")
    async def rollout(request: Request) -> dict[str, Any]:
        payload = await request.json()
        task_id, split, seed = _task_id(payload)
        rollout_id = str(payload.get("rollout_id") or f"gb_{game}_{uuid.uuid4().hex[:12]}")
        requested = str(payload.get("candidate_id") or "").strip()

        with lock:
            candidate = registry.get(requested) if requested else None
            if candidate is None:
                failed = registry.get(f"failed::{requested}") if requested else None
                has_bundle = any(
                    _bundle_value(payload, lever) is not None
                    for lever in ("policy_script", "harness_module", "system_prompt")
                )
                if failed is not None:
                    candidate = failed
                elif has_bundle:
                    # Backward compat: an inline bundle registers-if-needed, then runs.
                    candidate = _register(payload)
                elif requested:
                    # An id we have never seen and no bundle to apply: this is the seed
                    # program. Bind the id to the seed permanently rather than resolving
                    # it fresh each time -- otherwise the same id can later pick up a
                    # bundle and score the same row twice with two different policies.
                    seed_row = registry["seed"]
                    candidate = RegisteredCandidate(
                        candidate_id=requested, parent_id=None,
                        policy=seed_row.policy, harness=seed_row.harness, prompt=seed_row.prompt,
                        reports=[{
                            "schema_id": "apply_report.v1", "lever_ids": [],
                            "protocol_id": "identity", "patch_ok": True,
                            "compile_ok": True, "restart_ok": True,
                            "bound_to": "seed",
                        }],
                    )
                    registry[requested] = candidate
                else:
                    candidate = registry["seed"]

            episode: dict[str, Any]
            if candidate.apply_failed:
                episode = {"reward": 0.0, "events": [], "achievements": [], "ticks": [], "tool_calls": []}
            else:
                with httpx.Client(timeout=600.0) as client:
                    _ensure_active(client, candidate)
                    body: dict[str, Any] = {"env_url": env_url, "seed": seed, "split": split}
                    if max_steps is not None:
                        body["max_steps"] = max_steps
                    if mode == "harness":
                        body["prompt_overlay"] = candidate.prompt
                    target = candidate.worker_url if pooled and candidate.worker_url else policy_url
                    episode = client.post(f"{target}/episode", json=body, timeout=600.0).json()

        return _terminal(rollout_id, task_id, split, seed, candidate, episode)

    def _side_info(candidate: RegisteredCandidate, episode: dict[str, Any]) -> list[dict[str, Any]]:
        info = list(candidate.reports)
        # Why the episode produced nothing is the single most actionable fact for a
        # proposer. An empty trace summary reads identically whether the candidate
        # never compiled, never stepped the env, or genuinely scored zero.
        apply_diag = next(
            (
                report
                for report in candidate.reports
                if report.get("compile_ok") is False or report.get("restart_ok") is False
            ),
            None,
        )
        ran = bool(episode.get("ticks") or episode.get("tool_calls"))
        if candidate.apply_failed:
            reason = "apply_failed"
        elif episode.get("infra_errors"):
            reason = "infra_error"
        elif not ran:
            reason = "no_env_steps"
        else:
            reason = None
        verdict = {
            "schema_id": "episode_verdict.v1",
            "lever_ids": ["policy_script"] if mode == "code" else ["harness_module"],
            "summary": {
                "episode_ran": ran,
                "not_scored_because": reason,
                "apply_failed": candidate.apply_failed,
                "compile_ok": None if apply_diag is None else False,
                "reject_reason": (apply_diag or {}).get("reject_reason"),
                "runtime_errors": episode.get("runtime_errors") or [],
                "infra_errors": episode.get("infra_errors") or [],
                "fix_hint": {
                    "apply_failed": "the candidate never compiled or the policy would not start; read compile_diagnostics for the line",
                    "infra_error": "the rollout hit a transport/model error, not a policy mistake; this score is not evidence about the candidate",
                    "no_env_steps": "the candidate loaded but never stepped the env; run_episode must call env.step (directly or via a skill) before returning",
                }.get(reason),
            },
            "body": {"compile_diagnostics": (apply_diag or {}).get("compile_diagnostics")},
        }
        info.append(verdict)
        if mode == "code":
            info.append({
                "schema_id": "code_policy_game_trace.v1",
                "lever_ids": ["policy_script"],
                "summary": {
                    "ticks": len(episode.get("ticks") or []),
                    "final_score": episode.get("reward"),
                    "achievements": episode.get("achievements") or [],
                    "runtime_errors": episode.get("runtime_errors") or [],
                    "infra_errors": episode.get("infra_errors") or [],
                    "stop_reason": episode.get("stop_reason"),
                    "episode_ran": ran,
                    "not_scored_because": reason,
                    "compile_ok": bool(episode.get("compile_ok", True)) and not candidate.apply_failed,
                },
                "body": {"ticks": (episode.get("ticks") or [])[:60], "events": episode.get("events") or []},
            })
        else:
            inspected = episode.get("inspect") or {}
            info.append({
                "schema_id": "speedrunner_trace.v1",
                "lever_ids": ["harness_module", "system_prompt"],
                "summary": {
                    "skills_used": episode.get("skills_used") or [],
                    "skills_available": episode.get("skills_available") or [],
                    "llm_calls": episode.get("llm_calls"),
                    "primitives": len(episode.get("tool_calls") or []),
                    "architecture": episode.get("architecture"),
                    "runtime_errors": episode.get("runtime_errors") or [],
                    "infra_errors": episode.get("infra_errors") or [],
                    "episode_ran": ran,
                    "not_scored_because": reason,
                    "achievements": episode.get("achievements") or [],
                },
                "body": {"tool_calls": (episode.get("tool_calls") or [])[:80], "events": (episode.get("events") or [])[:80]},
            })
            info.append({
                "schema_id": "harness_inspect.v1",
                "lever_ids": ["harness_module"],
                "summary": inspected if isinstance(inspected, dict) else {},
                "body": {},
            })
            info.append({
                "schema_id": "prompt_trace.v1",
                "lever_ids": ["system_prompt"],
                "summary": {"prompt": candidate.prompt[:160]},
                "body": {"prompt": candidate.prompt},
            })
        return info

    def _terminal(
        rollout_id: str, task_id: str, split: str, seed: int,
        candidate: RegisteredCandidate, episode: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        apply_failed = candidate.apply_failed
        reward = 0.0 if apply_failed else float(episode.get("reward") or 0.0)
        side_info = _side_info(candidate, episode)
        asi_envelope = {
            "schema_version": "asi_envelope.v1",
            "rollout_id": rollout_id,
            "candidate_id": candidate.candidate_id,
            "task_id": task_id,
            "split": split,
            "seed": seed,
            "reward": reward,
            "game": game,
            "mode": mode,
            "schema_ids": [entry.get("schema_id") for entry in side_info],
            "side_info": side_info,
            "created_at": now,
        }
        asi_store[rollout_id] = asi_envelope
        record = {
            "rollout_id": rollout_id,
            "status": "failed" if apply_failed else "completed",
            "success_status": "apply_failed" if apply_failed else "succeeded",
            "task_id": task_id, "seed": seed, "split": split,
            "candidate_id": candidate.candidate_id,
            "reward": reward,
            "reward_info": {
                "outcome_reward": reward,
                "metrics": {
                    "achievements": episode.get("achievements") or [],
                    "apply_failed": apply_failed,
                    "runtime_errors": episode.get("runtime_errors") or [],
                },
            },
            "summary": {
                "outcome_reward": reward,
                "achievements": episode.get("achievements") or [],
                "game": game, "mode": mode,
            },
            "usage": {},
            "trace": {"schema_version": "trace.v1", "event_history": list(episode.get("events") or [])},
            "side_info": side_info,
            "actionable_side_info": side_info,
            "asi_ref": f"/asi/{rollout_id}",
            "created_at": now, "updated_at": now, "completed_at": now,
            "metadata": {"env_protocol": ENV_PROTOCOL, "game": game, "mode": mode},
        }
        rollouts[rollout_id] = record
        return record

    @app.get("/rollouts/{rollout_id}")
    def get_rollout(rollout_id: str) -> dict[str, Any]:
        if rollout_id not in rollouts:
            raise HTTPException(status_code=404, detail="unknown rollout")
        return rollouts[rollout_id]

    # -- ASI plane -------------------------------------------------------
    @app.get("/asi/schemas")
    def asi_schemas() -> dict[str, Any]:
        """What this container can emit, and when."""
        return {
            "schema_version": "asi_envelope.v1",
            "game": game,
            "mode": mode,
            "schemas": _program().get("side_info_schemas") or [],
        }

    @app.get("/asi")
    def asi_list(
        candidate_id: str | None = None,
        task_id: str | None = None,
        split: str | None = None,
        schema_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Index of ASI envelopes, newest last. Filters compose."""
        rows = list(asi_store.values())
        if candidate_id:
            rows = [row for row in rows if row["candidate_id"] == candidate_id]
        if task_id:
            rows = [row for row in rows if row["task_id"] == task_id]
        if split:
            rows = [row for row in rows if row["split"] == split]
        if schema_id:
            rows = [row for row in rows if schema_id in (row.get("schema_ids") or [])]
        rows = rows[-max(1, int(limit)) :]
        return {
            "count": len(rows),
            "total": len(asi_store),
            "items": [
                {
                    "rollout_id": row["rollout_id"],
                    "candidate_id": row["candidate_id"],
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "reward": row["reward"],
                    "schema_ids": row["schema_ids"],
                    "asi_ref": f"/asi/{row['rollout_id']}",
                }
                for row in rows
            ],
        }

    @app.get("/asi/{rollout_id}")
    def asi_for_rollout(rollout_id: str, summary_only: bool = False) -> dict[str, Any]:
        envelope = asi_store.get(rollout_id)
        if envelope is None:
            raise HTTPException(status_code=404, detail="unknown rollout")
        if not summary_only:
            return envelope
        return {
            **envelope,
            "side_info": [
                {key: value for key, value in entry.items() if key != "body"}
                for entry in envelope["side_info"]
            ],
        }

    @app.get("/asi/{rollout_id}/{schema_id}")
    def asi_by_schema(rollout_id: str, schema_id: str) -> dict[str, Any]:
        """One typed frame. Large bodies live here rather than in the prompt."""
        envelope = asi_store.get(rollout_id)
        if envelope is None:
            raise HTTPException(status_code=404, detail="unknown rollout")
        matches = [entry for entry in envelope["side_info"] if entry.get("schema_id") == schema_id]
        if not matches:
            raise HTTPException(status_code=404, detail=f"no {schema_id} on this rollout")
        return {
            "rollout_id": rollout_id,
            "candidate_id": envelope["candidate_id"],
            "task_id": envelope["task_id"],
            "schema_id": schema_id,
            "frames": matches,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--mode", required=True, choices=["code", "harness"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--env-url", default=os.environ.get("GAMEBENCH_ENV_URL", "http://127.0.0.1:19401"))
    parser.add_argument("--policy-url", default=os.environ.get("GAMEBENCH_POLICY_URL", "http://127.0.0.1:19402"))
    args = parser.parse_args()
    app = create_app(args.game, args.mode, args.env_url, args.policy_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
