"""Serve Craftax GameBench's CISPO surface over real HTTP, with a real sampler.

Wired the way the image's own contract test wires it -- the same rust-gold
stub on loopback, so the attempt drives the image's real environment client --
except that the transport posts to whatever origin the executor bound instead
of synthesizing an answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Mapping

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
IMAGE = Path("/Users/joshuapurtell/GitHub/evals/containers/images/craftax-gamebench-rust")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(IMAGE))
sys.path.insert(0, str(IMAGE / "tests"))

import uvicorn  # noqa: E402

from craftax_gold import cispo  # noqa: E402
from craftax_gold.stack import extend_app  # noqa: E402
from craftax_gold.targets import CRAFTAX_REACT  # noqa: E402
from synth_containers.platform.app import create_compat_app  # noqa: E402
from test_craftax_cispo_contract import GoldStub, _gold_handler  # noqa: E402

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
    answered here by a deterministic stand-in that reaches no network. It reads
    the legal action list off the prompt, because a sampler that guesses an
    action is not sampling the environment's own vocabulary.
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
        answer = json.dumps([str(legal[0])])
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


def gold_server() -> str:
    server = HTTPServer(("127.0.0.1", 0), _gold_handler(GoldStub()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8243
    gold_url = gold_server()
    app = create_compat_app(CRAFTAX_REACT)
    extend_app(
        app,
        declaration=cispo.craftax_cispo_declaration(
            profile=cispo.renderer_profile(
                tokenizer_id="openai/gpt-oss-20b",
                tokenizer_digest="sha256:gpt-oss-20b-tokenizer-unpinned",
                stop_token_ids=(200002, 199999),
                canary_digest=E2E_CANARY_DIGEST,
            ),
            image_digest="sha256:craftax-socket-run",
            # Three policy calls, not the default thirty-two. The rust-gold stub
            # unlocks its three achievements by tick eleven, so any horizon long
            # enough to reach tick eleven scores 3.0 for every plan and a group
            # has nothing to compare. At three calls the plan the policy writes
            # is what decides how far the episode gets.
            policy_calls=3,
        ),
        transport=HttpSampler(),
        world_factory=lambda: cispo.gold_world(
            base_url=gold_url, steps=cispo.max_env_steps()
        ),
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
