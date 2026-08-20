"""gamebench_env.v1 -- Gymnasium-over-HTTP over one GameBench game.

One game per process: every task dir ships its own `gold_python` package, so two
games cannot share an interpreter. The orchestrator starts one of these per stack.
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from gamebench_levers import ENV_PROTOCOL


def create_app(game: str) -> FastAPI:
    from gamebench_levers.adapters import load  # noqa: PLC0415

    adapter = load(game)
    app = FastAPI(title=f"gamebench-env-{game}")
    sessions: dict[str, Any] = {}

    def _make(seed: int, max_steps: int | None, split: str) -> Any:
        kwargs: dict[str, Any] = {}
        if max_steps is not None:
            kwargs["max_steps"] = int(max_steps)
        if game == "dungeongrid":
            kwargs["split"] = split
        return adapter.make_session(int(seed), **kwargs)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "game": game, "env_id": adapter.ENV_ID, "version": ENV_PROTOCOL, "pid": os.getpid()}

    @app.get("/spec")
    def spec() -> dict[str, Any]:
        return {"protocol_id": ENV_PROTOCOL, **adapter.spec()}

    @app.post("/reset")
    async def reset(request: Request) -> dict[str, Any]:
        payload = await request.json()
        session_id = str(payload.get("session_id") or uuid.uuid4().hex[:12])
        session = _make(
            int(payload.get("seed") or 0),
            payload.get("max_steps"),
            str(payload.get("split") or "train"),
        )
        sessions[session_id] = session
        return {"session_id": session_id, "obs": session.observation(), "info": {"seed": payload.get("seed")}}

    @app.post("/step")
    async def step(request: Request) -> dict[str, Any]:
        payload = await request.json()
        session = sessions.get(str(payload.get("session_id") or ""))
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        return session.step(payload.get("action"))

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19401)
    args = parser.parse_args()
    uvicorn.run(create_app(args.game), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
