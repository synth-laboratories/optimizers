"""Serve the container, printing every CISPO request body as it arrives."""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "tests"))

import uvicorn  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from synth_containers.http_adapter import create_reference_app  # noqa: E402
from test_cispo_target import installed  # noqa: E402


class Dump(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        body = await request.body()
        print(f">>> {request.method} {request.url.path} {body[:2000].decode(errors='replace')}", flush=True)
        response = await call_next(request)
        return response


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
    runtime, _target = installed(target_count=2)
    app = create_reference_app(runtime)
    app.add_middleware(Dump)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
