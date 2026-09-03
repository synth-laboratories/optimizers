"""Serve harbor-tblite's CISPO surface over real HTTP, with a real sampler.

Wired the way the image's own contract test wires it -- the same stub substrate,
so neither sibling container is started -- except that the transport posts to
whatever origin the executor bound, and the verifier reports on its own instead
of waiting for a test to call ``report``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
IMAGE = Path("/Users/joshuapurtell/GitHub/evals/containers/images/harbor-tblite")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(IMAGE))
sys.path.insert(0, str(IMAGE / "tests"))

import uvicorn  # noqa: E402

from harbor_tblite.cispo import renderer_profile, tblite_cispo_declaration  # noqa: E402
from harbor_tblite.stack import extend_app  # noqa: E402
from harbor_tblite.targets import HARBOR_TBLITE  # noqa: E402
from synth_containers.platform.app import create_compat_app  # noqa: E402
from test_harbor_tblite_cispo_contract import TRIALS, StubSubstrate  # noqa: E402

RENDER_VOCAB_BASE = 100_000
RENDER_VOCAB_SIZE = 50_000


def render_tokens(text: str) -> tuple[int, ...]:
    words = text.split() or [""]
    return tuple(
        RENDER_VOCAB_BASE
        + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % RENDER_VOCAB_SIZE
        for word in words
    )


class ReportingSubstrate(StubSubstrate):
    """The test's substrate, with a verifier that actually answers.

    The image's own stub leaves ``poll_verifier`` answering ``None`` until a
    test calls ``report``; a run has nobody to call it. The verifier here reads
    the workspace the episode left behind and scores it, which is what the
    sibling verifier container does -- it is stubbed because this run starts no
    container, not because the measure is invented.
    """

    def submit_verifier(self, *, trial: Any, workspace: Any, rollout_id: str) -> str:
        handle = super().submit_verifier(trial=trial, workspace=workspace, rollout_id=rollout_id)
        marker = int(
            hashlib.sha256(workspace.content_digest().encode("utf-8")).hexdigest()[:8], 16
        )
        self.report(rollout_id, reward=round((marker % 1000) / 1000.0, 3), exit_code=0)
        return handle


class HttpSampler:
    """Posts to the bound origin. The container never renders a token itself.

    A ``probe://`` endpoint is the one exception, and it is not an exception to
    that rule: a probe has no origin to post to, by definition, so it is
    answered here by a deterministic stand-in that reaches no network.
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
        turn = sum(1 for row in messages if str(row.get("role")) == "assistant")
        answer = (
            "```bash\necho MINI_SWE_DONE\n```"
            if turn >= 1
            else "```bash\nls -a\n```"
        )
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8244
    root = Path(tempfile.mkdtemp(prefix="tblite-cispo-socket-"))
    app = create_compat_app(HARBOR_TBLITE)
    extend_app(
        app,
        trials=TRIALS,
        declaration=tblite_cispo_declaration(
            trials=TRIALS,
            profile=renderer_profile(
                tokenizer_id="openai/gpt-oss-20b",
                tokenizer_digest="sha256:gpt-oss-20b-tokenizer-unpinned",
                stop_token_ids=(200002, 199999),
            ),
            image_digest="sha256:harbor-tblite-socket-run",
        ),
        substrate=ReportingSubstrate(root),
        transport=HttpSampler(),
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
