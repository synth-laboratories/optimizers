"""Harness policy service: load an entire SpeedRunner script, drive the env over HTTP.

The search object is the whole file. `harness_restart.v1` applies it by writing the
file and restarting this process, so the loaded loop and the candidate id are the
same object. The env process is never touched.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from gamebench_levers import ENV_PROTOCOL
from gamebench_levers.env_client import EnvClient
from gamebench_levers.inspect_script import inspect_loaded, inspect_summary
from gamebench_levers.llm import Llm
from gamebench_levers.seeds import harness_seed, prompt_seed


def load_harness(source: str, filename: str = "harness.py") -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "harness"}
    exec(compile(source, filename, "exec"), namespace, namespace)  # noqa: S102
    run_episode = namespace.get("run_episode")
    if not callable(run_episode):
        raise ValueError("harness_module must define run_episode(env, prompt, seed=..., max_steps=..., llm=...)")
    return {"fn": run_episode, "namespace": namespace, "kind": "run_episode"}


def create_app(game: str, script_path: str | None = None) -> FastAPI:
    app = FastAPI(title=f"gamebench-policy-harness-{game}")
    path = Path(script_path or os.environ.get("GAMEBENCH_HARNESS_SCRIPT") or "")
    source = path.read_text(encoding="utf-8") if path.is_file() else harness_seed(game)
    try:
        loaded = load_harness(source, str(path) if path else "harness.py")
        load_error = None
    except Exception as exc:  # noqa: BLE001
        loaded = {"fn": None, "namespace": {}, "kind": None}
        load_error = f"{exc}\n{traceback.format_exc()}"
    state: dict[str, Any] = {
        "source": source,
        "harness": loaded,
        "load_error": load_error,
        "prompt": os.environ.get("GAMEBENCH_HARNESS_PROMPT") or prompt_seed(game),
        "game": game,
    }

    def _filename() -> str:
        return str(path) if path else "harness.py"

    def _snapshot(*, include_source: bool = True) -> dict[str, Any]:
        body = inspect_loaded(
            str(state["source"] or ""),
            state["harness"].get("namespace"),
            filename=_filename(),
            load_error=state.get("load_error"),
        )
        skills = (state["harness"].get("namespace") or {}).get("PUBLIC_SKILLS") or {}
        body["public_skills"] = sorted(skills) if isinstance(skills, dict) else []
        if not include_source:
            body = {**body, "source": None}
        return body

    def _instantiate(text: str) -> dict[str, Any]:
        loaded = load_harness(text, _filename())
        state["harness"] = loaded
        state["source"] = text
        state["load_error"] = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return _snapshot()

    @app.get("/health")
    def health() -> dict[str, Any]:
        snapshot = _snapshot(include_source=False)
        ok = state["harness"].get("fn") is not None and state["load_error"] is None
        return {
            "status": "ok" if ok else "degraded",
            "policy_kind": "speedrunner_http.v1",
            "reload": "process_restart",
            "inspect_route": "/inspect",
            "env_protocol": ENV_PROTOCOL,
            "game": game,
            "pid": os.getpid(),
            "compile_ok": ok,
            "compile_error": (str(state.get("load_error") or "").splitlines() or [None])[0],
            "entrypoint": state["harness"].get("kind"),
            "public_skills": snapshot.get("public_skills"),
            "architecture": snapshot.get("architecture"),
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "policy_kind": "speedrunner_http.v1",
            "reload": "process_restart",
            "inspect_route": "/inspect",
            "lever_ids": ["harness_module", "system_prompt"],
            "env_protocol": ENV_PROTOCOL,
            "entrypoint": "run_episode",
            "game": game,
        }

    @app.get("/inspect")
    def inspect_live(include_source: bool = True) -> dict[str, Any]:
        return _snapshot(include_source=include_source)

    @app.post("/reload")
    async def reload_prompt(request: Request) -> dict[str, Any]:
        payload = await request.json()
        overlay = payload.get("prompt_overlay") or payload.get("system_prompt")
        if overlay is not None:
            state["prompt"] = str(overlay)
        return {"ok": True, "pid": os.getpid()}

    @app.post("/restart")
    async def restart_in_process(request: Request) -> dict[str, Any]:
        """Fallback when no supervisor owns this process: exec the new script here."""
        payload = await request.json()
        text = str(payload.get("source") or payload.get("content") or "")
        try:
            snapshot = _instantiate(text) if text else _snapshot()
            return {
                "restart_ok": True,
                "compile_ok": True,
                "error": None,
                "pid": os.getpid(),
                "in_process": True,
                "inspect": inspect_summary(snapshot),
            }
        except Exception as exc:  # noqa: BLE001
            state["load_error"] = f"{exc}\n{traceback.format_exc()}"
            state["harness"] = {"fn": None, "namespace": {}, "kind": None}
            return {"restart_ok": False, "compile_ok": False, "error": str(exc), "pid": os.getpid()}

    @app.post("/episode")
    async def episode(request: Request) -> dict[str, Any]:
        payload = await request.json()
        env_url = str(payload.get("env_url") or "").rstrip("/")
        if not env_url:
            raise HTTPException(status_code=400, detail="env_url is required")
        harness = state["harness"]
        if harness.get("fn") is None:
            raise HTTPException(status_code=503, detail=state["load_error"] or "no harness loaded")
        overlay = payload.get("prompt_overlay")
        prompt = str(overlay) if overlay is not None else str(state["prompt"] or "")
        try:
            llm = Llm()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        max_steps = payload.get("max_steps")
        with httpx.Client(timeout=300.0) as client:
            env = EnvClient(client, env_url, split=str(payload.get("split") or "train"))
            try:
                result = harness["fn"](
                    env,
                    prompt,
                    seed=int(payload.get("seed") or 0),
                    max_steps=int(max_steps) if max_steps is not None else None,
                    llm=llm,
                )
            except Exception as exc:  # noqa: BLE001
                # A broken candidate harness scores 0 with diagnostics; it is not a crash.
                # Transport and model failures are NOT evidence about the candidate, so
                # they are reported on a separate channel from policy bugs.
                infra = isinstance(exc, (httpx.HTTPError, httpx.StreamError))
                label = f"{type(exc).__name__}: {exc}"
                return {
                    "reward": 0.0,
                    "events": [],
                    "tool_calls": [],
                    "achievements": [],
                    "runtime_errors": [] if infra else [label],
                    "infra_errors": [label] if infra else [],
                    "traceback": traceback.format_exc()[-2000:],
                    "architecture": None,
                    "skills_available": sorted(
                        (state["harness"].get("namespace") or {}).get("PUBLIC_SKILLS") or {}
                    ),
                    "llm_calls": len(llm.calls),
                    "policy_pid": os.getpid(),
                }
        if not isinstance(result, dict):
            return {
                "reward": 0.0,
                "runtime_errors": ["run_episode must return a dict"],
                "events": [],
                "tool_calls": [],
                "achievements": [],
                "llm_calls": len(llm.calls),
                "policy_pid": os.getpid(),
            }
        result.setdefault("runtime_errors", [])
        result.setdefault("infra_errors", [])
        result["skills_available"] = sorted(
            (state["harness"].get("namespace") or {}).get("PUBLIC_SKILLS") or {}
        )
        result["policy_pid"] = os.getpid()
        result["llm_provider"] = llm.provider
        result["llm_model"] = llm.model
        result["llm_calls"] = result.get("llm_calls") or len(llm.calls)
        result["inspect"] = inspect_summary(_snapshot(include_source=False))
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19403)
    parser.add_argument("--script-path", default=os.environ.get("GAMEBENCH_HARNESS_SCRIPT") or "")
    args = parser.parse_args()
    uvicorn.run(create_app(args.game, args.script_path or None), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
