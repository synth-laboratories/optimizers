"""Serve DungeonGrid gold's CISPO surface over real HTTP, with a real sampler.

Wired the way the image's own contract test wires it -- one real
``dungeongrid_gold`` Rust process on loopback, as PID 1 runs it in the image --
except that the transport posts to whatever origin the executor bound instead of
synthesizing an answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
IMAGE = Path("/Users/joshuapurtell/GitHub/evals/containers/images/dungeongrid-gold")
ENGINE = Path(
    "/Users/joshuapurtell/GitHub/gamebench/tasks/dungeongrid-multiplayer/gold_rust"
    "/target/release/dungeongrid_gold"
)
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(IMAGE))

import uvicorn  # noqa: E402

from dungeongrid_gold import cispo  # noqa: E402
from dungeongrid_gold.targets import DUNGEONGRID_REACT  # noqa: E402
from synth_containers.platform.app import create_compat_app  # noqa: E402

RENDER_VOCAB_BASE = 100_000
RENDER_VOCAB_SIZE = 50_000
E2E_CANARY_DIGEST = os.environ.get(
    "SYNTH_CISPO_RENDERER_CANARY_DIGEST", "96db06cead43f00b514724ef74c58fdf"
)


def render_tokens(text: str) -> tuple[int, ...]:
    words = text.split() or [""]
    return tuple(
        RENDER_VOCAB_BASE
        + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % RENDER_VOCAB_SIZE
        for word in words
    )


class HttpSampler:
    """Posts to the bound origin. The container never renders a token itself.

    A ``probe://`` endpoint is the one exception, and it is not an exception to
    that rule: a probe has no origin to post to, by definition, so it is
    answered here by a deterministic stand-in that reaches no network and reads
    the legal action list off the prompt.
    """

    def __init__(self, *, timeout: float = 180.0) -> None:
        self.timeout = timeout
        self.calls = 0
        self.probe_calls = 0
        self._lock = threading.Lock()

    def reachable(self, origin: Any) -> bool:
        return True

    def _probe_answer(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = body.get("messages") or ()
        prompt = "\n".join(str(row.get("content") or "") for row in messages)
        marker = "valid_actions="
        at = prompt.rfind(marker)
        if at < 0:
            raise RuntimeError("the rendered prompt names no legal action list")
        legal = json.loads(prompt[at + len(marker) :].strip().splitlines()[0])
        if not legal:
            raise RuntimeError("the rendered prompt names an empty legal action list")
        answer = str(legal[0])
        tokens = list(render_tokens(answer))
        logprobs = [
            round(-0.05 - ((int(token) + index) % 97) / 500.0, 6)
            for index, token in enumerate(tokens)
        ]
        with self._lock:
            self.probe_calls += 1
            index = self.probe_calls
        return {
            "id": f"probe-{index}",
            "object": "chat.completion",
            "model": str(body.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": answer},
                }
            ],
            "prompt_token_ids": list(render_tokens(prompt)),
            "token_ids": {"completion": tokens},
            "logprobs": {"completion": logprobs},
            "usage": {"completion_tokens": len(tokens)},
        }

    def post(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if url.startswith("probe://"):
            if "Authorization" not in headers:
                raise RuntimeError(f"probe call to {url} carries no Authorization header")
            return self._probe_answer(body)
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
        request.add_header("content-type", "application/json")
        for name, value in headers.items():
            request.add_header(name, value)
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"sampler {url} returned {exc.code}: {detail}") from exc


def engine_server() -> str:
    binary = Path(os.environ.get("SYNTH_DUNGEONGRID_GOLD_BIN") or ENGINE)
    if not binary.is_file():
        raise SystemExit(f"the gold engine binary is not built at {binary}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    subprocess.Popen(  # noqa: S603
        [str(binary), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=0.5) as reply:
                if json.loads(reply.read().decode()).get("ok"):
                    return url
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.05)
    raise SystemExit("the gold engine never became healthy")


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8245
    engine_url = engine_server()
    os.environ["SYNTH_CISPO_RENDERER_CANARY_DIGEST"] = E2E_CANARY_DIGEST
    # The compatibility target's health check reads the same environment
    # contract used by the image entrypoint.  The CISPO target receives the
    # URL directly below, but without this the outer /health route reports a
    # false negative and the optimizer correctly refuses to start.
    os.environ["SYNTH_DUNGEONGRID_URL"] = engine_url
    target = cispo.DungeonGridCispoTarget(
        engine=cispo.HttpDungeonGridEngine(base_url=engine_url),
        transport=HttpSampler(),
    )
    cispo.set_installed_target(target)
    app = create_compat_app(DUNGEONGRID_REACT)
    cispo.mount_cispo_routes(app)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
