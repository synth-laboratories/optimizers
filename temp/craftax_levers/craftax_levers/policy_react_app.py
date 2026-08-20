"""ReAct policy service: load an entire react_loop.py, talk to env over HTTP."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from craftax_levers import ENV_PROTOCOL
from craftax_levers.env_client import EnvClient
from craftax_levers.inspect_script import inspect_loaded, inspect_summary
from craftax_levers.llm import Llm
from craftax_levers.seeds import SEED_HARNESS

RunEpisodeFn = Callable[..., dict[str, Any]]


def load_react_script(source: str, filename: str = "react_loop.py") -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "react_loop"}
    exec(compile(source, filename, "exec"), namespace, namespace)  # noqa: S102
    run_episode = namespace.get("run_episode")
    if callable(run_episode):
        return {"kind": "run_episode", "fn": run_episode, "namespace": namespace, "error": None}
    raise ValueError("react script must define run_episode(env, prompt, seed=0, max_steps=16, llm=...)")


def create_app(script_path: str | None = None) -> FastAPI:
    app = FastAPI(title="craftax-policy-react")
    path = Path(script_path or os.environ.get("CRAFTAX_REACT_SCRIPT") or "")
    source = path.read_text(encoding="utf-8") if path.is_file() else SEED_HARNESS
    try:
        loaded = load_react_script(source, str(path) if path else "react_loop.py")
        load_error = None
    except Exception as exc:  # noqa: BLE001
        loaded = {"kind": None, "fn": None, "namespace": {}}
        load_error = f"{exc}\n{traceback.format_exc()}"
    state: dict[str, Any] = {
        "script_path": str(path) if path else None,
        "source": source,
        "harness": loaded,
        "load_error": load_error,
        "prompt": os.environ.get("CRAFTAX_REACT_PROMPT") or "",
    }

    def _filename() -> str:
        return str(path) if path else "react_loop.py"

    def _snapshot(*, include_source: bool = True) -> dict[str, Any]:
        body = inspect_loaded(
            str(state["source"] or ""),
            state["harness"].get("namespace"),
            filename=_filename(),
            load_error=state.get("load_error"),
        )
        if not include_source:
            body = {**body, "source": None}
        return body

    def _instantiate(text: str) -> dict[str, Any]:
        loaded = load_react_script(text, _filename())
        state["harness"] = loaded
        state["source"] = text
        state["load_error"] = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return _snapshot()

    def _reload_from_disk() -> dict[str, Any]:
        if not path.is_file():
            return _snapshot()
        return _instantiate(path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        snapshot = _snapshot(include_source=False)
        ok = state["harness"].get("fn") is not None and state["load_error"] is None
        return {
            "status": "ok" if ok else "degraded",
            "policy_kind": "react_http.v1",
            "reload": "process_restart",
            "instantiate": "in_process_exec",
            "inspect_route": "/inspect",
            "env_protocol": ENV_PROTOCOL,
            "pid": os.getpid(),
            "compile_ok": ok,
            "entrypoint": state["harness"].get("kind"),
            "script_path": state["script_path"],
            "architecture": snapshot.get("architecture"),
            "react_llm": snapshot.get("react_llm"),
            "llm_call_sites": snapshot.get("llm_call_sites"),
            "instantiated": snapshot.get("instantiated"),
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "policy_kind": "react_http.v1",
            "reload": "process_restart",
            "instantiate": "in_process_exec",
            "inspect_route": "/inspect",
            "lever_ids": ["react_system_prompt", "harness_module"],
            "env_protocol": ENV_PROTOCOL,
            "entrypoint": "run_episode",
        }

    @app.get("/inspect")
    def inspect_live(include_source: bool = True) -> dict[str, Any]:
        return _snapshot(include_source=include_source)

    @app.post("/load")
    async def load_live(request: Request) -> dict[str, Any]:
        """Compile + exec the ReAct script in this process. Same pid; inspect immediately."""
        payload = await request.json()
        text = str(payload.get("source") or payload.get("content") or "")
        try:
            snapshot = _instantiate(text) if text else _reload_from_disk()
            return {
                "ok": True,
                "restart_ok": True,
                "compile_ok": True,
                "in_process": True,
                "pid": os.getpid(),
                "entrypoint": state["harness"].get("kind"),
                "inspect": inspect_summary(snapshot),
            }
        except Exception as exc:  # noqa: BLE001
            state["load_error"] = f"{exc}\n{traceback.format_exc()}"
            state["harness"] = {"kind": None, "fn": None, "namespace": {}}
            return {
                "ok": False,
                "restart_ok": False,
                "compile_ok": False,
                "in_process": True,
                "error": str(exc),
                "pid": os.getpid(),
            }

    @app.post("/reload")
    async def reload_prompt(request: Request) -> dict[str, Any]:
        payload = await request.json()
        overlay = payload.get("prompt_overlay") or payload.get("react_system_prompt")
        if overlay is not None:
            state["prompt"] = str(overlay)
        return {"ok": True, "prompt": state["prompt"], "pid": os.getpid()}

    @app.post("/restart")
    async def restart_in_process(request: Request) -> dict[str, Any]:
        """Fallback when no supervisor: exec the new script in this process."""
        payload = await request.json()
        text = str(payload.get("source") or payload.get("content") or "")
        try:
            snapshot = _instantiate(text) if text else _reload_from_disk()
            return {
                "restart_ok": True,
                "compile_ok": True,
                "error": None,
                "pid": os.getpid(),
                "in_process": True,
                "entrypoint": state["harness"]["kind"],
                "inspect": inspect_summary(snapshot),
            }
        except Exception as exc:  # noqa: BLE001
            state["load_error"] = f"{exc}\n{traceback.format_exc()}"
            state["harness"] = {"kind": None, "fn": None, "namespace": {}}
            return {"restart_ok": False, "compile_ok": False, "error": str(exc), "pid": os.getpid()}

    @app.post("/episode")
    async def episode(request: Request) -> dict[str, Any]:
        payload = await request.json()
        env_url = str(payload.get("env_url") or "").rstrip("/")
        if not env_url:
            raise HTTPException(status_code=400, detail="env_url is required")
        harness = state["harness"]
        if harness.get("fn") is None:
            raise HTTPException(status_code=503, detail=state["load_error"] or "no react script loaded")
        overlay = payload.get("prompt_overlay")
        prompt = str(overlay) if overlay is not None else str(state["prompt"] or "")
        seed = int(payload.get("seed") or 0)
        max_steps = int(payload.get("max_steps") or 16)
        if harness["kind"] != "run_episode":
            raise HTTPException(status_code=500, detail="react policy must expose run_episode(..., llm=...)")
        try:
            llm = Llm()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        with httpx.Client(timeout=60.0) as client:
            env = EnvClient(client, env_url)
            result = harness["fn"](env, prompt, seed=seed, max_steps=max_steps, llm=llm)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="run_episode must return a dict")
        result.setdefault("prompt_used", prompt)
        result["policy_pid"] = os.getpid()
        result["llm_provider"] = llm.provider
        result["llm_model"] = llm.model
        result["llm_calls"] = result.get("llm_calls") or len(llm.calls)
        result["inspect"] = inspect_summary(_snapshot(include_source=False))
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19102)
    parser.add_argument("--script-path", default=os.environ.get("CRAFTAX_REACT_SCRIPT") or "")
    args = parser.parse_args()
    app = create_app(script_path=args.script_path or None)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
