"""Serve Banking77's CISPO surface over real HTTP, with a real sampler.

Wired the way the image's own tests wire it, except that the transport posts
to whatever origin the executor bound instead of synthesizing an answer.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
IMAGE = Path("/Users/joshuapurtell/GitHub/evals/containers/images/banking77")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(IMAGE))

import uvicorn  # noqa: E402

from banking77_classify.cispo import banking77_cispo_declaration, renderer_profile  # noqa: E402
from banking77_classify.stack import extend_app  # noqa: E402
from banking77_classify.targets import BANKING77_CLASSIFY  # noqa: E402
from synth_containers.platform.app import create_compat_app  # noqa: E402


#: The renderer the unpaid plane declares, restated so the probe's synthetic
#: capture is rendered by the same rule the run's one renderer uses.
RENDER_VOCAB_BASE = 100_000
RENDER_VOCAB_SIZE = 50_000


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
    answered here by the deterministic stand-in this image's own tests use. It
    reaches no network and buys nothing, which is the whole point of a probe.
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
        answer = "card_arrival"
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


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8230
    app = create_compat_app(BANKING77_CLASSIFY)
    extend_app(
        app,
        declaration=banking77_cispo_declaration(
            profile=renderer_profile(
                tokenizer_id="openai/gpt-oss-20b",
                tokenizer_digest="sha256:gpt-oss-20b-tokenizer-unpinned",
                stop_token_ids=(200002, 199999),
            ),
            image_digest="sha256:banking77-socket-run",
        ),
        transport=HttpSampler(),
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
