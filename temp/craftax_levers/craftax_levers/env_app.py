"""craftax_env.v1 — Gymnasium-over-HTTP. No policy, no prompts."""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from craftax_levers import ENV_PROTOCOL
from craftax_levers.world import ACTIONS, reset, step


def create_app() -> FastAPI:
    app = FastAPI(title="craftax-env")
    sessions: dict[str, Any] = {}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "env_id": "craftax_mini",
            "version": ENV_PROTOCOL,
            "pid": os.getpid(),
        }

    @app.get("/spec")
    def spec() -> dict[str, Any]:
        return {
            "protocol_id": ENV_PROTOCOL,
            "action_space": list(ACTIONS),
            "observation_space": {
                "grid": "ascii",
                "inventory": ["wood"],
                "achievements": ["collect_wood"],
            },
            "max_horizon": 16,
        }

    @app.post("/reset")
    async def env_reset(request: Request) -> dict[str, Any]:
        payload = await request.json()
        seed = int(payload.get("seed") or 0)
        max_steps = int(payload.get("max_steps") or 16)
        session_id = str(payload.get("session_id") or uuid.uuid4().hex[:12])
        world, obs = reset(seed, max_steps=max_steps)
        sessions[session_id] = world
        return {"session_id": session_id, "obs": obs, "info": {"seed": seed}}

    @app.post("/step")
    async def env_step(request: Request) -> dict[str, Any]:
        payload = await request.json()
        session_id = str(payload.get("session_id") or "")
        world = sessions.get(session_id)
        if world is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        obs, reward, terminated, info = step(world, str(payload.get("action") or "noop"))
        truncated = bool(world.tick >= world.max_steps and not world.won and not world.dead)
        return {
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19101)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
