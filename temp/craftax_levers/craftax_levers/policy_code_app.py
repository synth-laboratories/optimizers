"""Code-policy service: load policy_script, call act(obs), drive env over HTTP."""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from craftax_levers import ENV_PROTOCOL
from craftax_levers.seeds import SEED_POLICY

ActFn = Callable[[dict[str, Any]], str]


def _load_act(source: str) -> ActFn:
    namespace: dict[str, Any] = {}
    exec(compile(source, "policy.py", "exec"), namespace, namespace)  # noqa: S102
    act = namespace.get("act")
    if not callable(act):
        raise ValueError("policy_script must define act(obs)")
    return act


def create_app() -> FastAPI:
    app = FastAPI(title="craftax-policy-code")
    state: dict[str, Any] = {
        "source": SEED_POLICY,
        "act": _load_act(SEED_POLICY),
        "load_error": None,
        "restarts": 0,
    }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if state["act"] is not None else "degraded",
            "policy_kind": "code_policy.v1",
            "reload": "in_process",
            "env_protocol": ENV_PROTOCOL,
            "pid": os.getpid(),
            "compile_ok": state["act"] is not None,
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "policy_kind": "code_policy.v1",
            "reload": "in_process",
            "lever_ids": ["policy_script"],
            "env_protocol": ENV_PROTOCOL,
            "entrypoint": "act",
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        payload = await request.json()
        source = str(payload.get("source") or payload.get("content") or "")
        try:
            state["act"] = _load_act(source)
            state["source"] = source
            state["load_error"] = None
            return {"compile_ok": True, "error": None}
        except Exception as exc:  # noqa: BLE001
            state["act"] = None
            state["load_error"] = f"{exc}\n{traceback.format_exc()}"
            return {"compile_ok": False, "error": str(exc)}

    @app.post("/act")
    async def act_route(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if state["act"] is None:
            raise HTTPException(status_code=503, detail=state["load_error"] or "no policy loaded")
        action = str(state["act"](payload.get("obs") or {}))
        return {"action": action}

    @app.post("/episode")
    async def episode(request: Request) -> dict[str, Any]:
        payload = await request.json()
        env_url = str(payload.get("env_url") or "").rstrip("/")
        if not env_url:
            raise HTTPException(status_code=400, detail="env_url is required")
        if state["act"] is None:
            raise HTTPException(status_code=503, detail=state["load_error"] or "no policy loaded")
        seed = int(payload.get("seed") or 0)
        max_steps = int(payload.get("max_steps") or 16)
        events: list[dict[str, Any]] = []
        ticks: list[dict[str, Any]] = []
        with httpx.Client(timeout=10.0) as client:
            reset = client.post(f"{env_url}/reset", json={"seed": seed, "max_steps": max_steps}).json()
            session_id = reset["session_id"]
            obs = reset["obs"]
            events.append({"type": "env_reset", "seed": seed, "session_id": session_id})
            total = 0.0
            terminated = False
            info: dict[str, Any] = {}
            while not obs.get("done"):
                action = str(state["act"](obs))
                events.append({"type": "policy_act", "action": action, "tick": obs["tick"]})
                stepped = client.post(
                    f"{env_url}/step",
                    json={"session_id": session_id, "action": action},
                ).json()
                obs = stepped["obs"]
                total += float(stepped.get("reward") or 0.0)
                terminated = bool(stepped.get("terminated"))
                info = stepped.get("info") or {}
                ticks.append(
                    {
                        "tick": obs["tick"],
                        "action": action,
                        "reward": stepped.get("reward"),
                        "achievements": list(info.get("achievements") or []),
                    }
                )
                events.append({"type": "env_step", "reward": stepped.get("reward"), "terminated": terminated})
                if terminated or obs.get("done"):
                    break
        return {
            "reward": float(info.get("outcome_reward") if info else total),
            "ticks": ticks,
            "events": events,
            "achievements": list(info.get("achievements") or []),
            "death_cause": info.get("death_cause"),
            "compile_ok": True,
            "load_error": None,
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19102)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
