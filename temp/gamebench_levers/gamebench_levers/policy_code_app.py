"""Code-policy service: load a policy_script, call act(obs), drive the env over HTTP."""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from gamebench_levers import ENV_PROTOCOL
from gamebench_levers.seeds import code_seed

ActFn = Callable[[dict[str, Any]], Any]
MAX_TICKS = 4000


def load_act(source: str) -> ActFn:
    namespace: dict[str, Any] = {"__name__": "policy"}
    exec(compile(source, "policy.py", "exec"), namespace, namespace)  # noqa: S102
    act = namespace.get("act")
    if not callable(act):
        raise ValueError("policy_script must define act(obs)")
    return act


def create_app(game: str) -> FastAPI:
    app = FastAPI(title=f"gamebench-policy-code-{game}")
    seed_source = code_seed(game)
    state: dict[str, Any] = {
        "source": seed_source,
        "act": load_act(seed_source),
        "load_error": None,
        "game": game,
    }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if state["act"] is not None else "degraded",
            "policy_kind": "code_policy.v1",
            "reload": "per_candidate_load",
            "env_protocol": ENV_PROTOCOL,
            "game": game,
            "pid": os.getpid(),
            "compile_ok": state["act"] is not None,
            "compile_error": (str(state.get("load_error") or "").splitlines() or [None])[0],
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "policy_kind": "code_policy.v1",
            "reload": "per_candidate_load",
            "lever_ids": ["policy_script"],
            "env_protocol": ENV_PROTOCOL,
            "entrypoint": "act",
            "game": game,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        payload = await request.json()
        source = str(payload.get("source") or payload.get("content") or "")
        try:
            state["act"] = load_act(source)
            state["source"] = source
            state["load_error"] = None
            return {"compile_ok": True, "error": None}
        except Exception as exc:  # noqa: BLE001
            state["act"] = None
            state["load_error"] = f"{exc}\n{traceback.format_exc()}"
            return {"compile_ok": False, "error": str(exc)}

    @app.post("/episode")
    async def episode(request: Request) -> dict[str, Any]:
        payload = await request.json()
        env_url = str(payload.get("env_url") or "").rstrip("/")
        if not env_url:
            raise HTTPException(status_code=400, detail="env_url is required")
        act = state["act"]
        if act is None:
            raise HTTPException(status_code=503, detail=state["load_error"] or "no policy loaded")
        body: dict[str, Any] = {"seed": int(payload.get("seed") or 0), "split": payload.get("split") or "train"}
        if payload.get("max_steps") is not None:
            body["max_steps"] = int(payload["max_steps"])

        events: list[dict[str, Any]] = []
        ticks: list[dict[str, Any]] = []
        runtime_errors: list[str] = []
        infra_errors: list[str] = []
        with httpx.Client(timeout=120.0) as client:
            reset = client.post(f"{env_url}/reset", json=body).json()
            session_id = reset["session_id"]
            obs = reset["obs"]
            events.append({"type": "env_reset", "seed": body["seed"]})
            info: dict[str, Any] = {}
            guard = 0
            while not obs.get("done") and guard < MAX_TICKS:
                guard += 1
                try:
                    action = act(obs)
                except Exception as exc:  # noqa: BLE001
                    # gepa-ai's rule: never raise on one example. Score it and report.
                    runtime_errors.append(f"{type(exc).__name__}: {exc}")
                    events.append({"type": "policy_error", "error": runtime_errors[-1], "tick": obs.get("tick")})
                    break
                stepped = client.post(
                    f"{env_url}/step",
                    json={"session_id": session_id, "action": action},
                ).json()
                obs = stepped["obs"]
                info = stepped.get("info") or {}
                ticks.append(
                    {
                        "tick": obs.get("tick"),
                        "action": action,
                        "reward": stepped.get("reward"),
                        "score": obs.get("score"),
                    }
                )
                if stepped.get("terminated") or stepped.get("truncated") or obs.get("done"):
                    break
        return {
            "reward": float(info.get("outcome_reward") if info else (obs.get("score") or 0.0)),
            "ticks": ticks,
            "events": events,
            "achievements": list(obs.get("achievements") or []),
            "runtime_errors": runtime_errors,
            "infra_errors": infra_errors,
            "compile_ok": True,
            "stop_reason": "policy_error" if runtime_errors else ("done" if obs.get("done") else "guard"),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19402)
    args = parser.parse_args()
    uvicorn.run(create_app(args.game), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
