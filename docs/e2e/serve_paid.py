"""Serve the reference container with a sampler that really calls the gateway.

The deterministic sampler synthesizes tokens in-process, which is right for a
conformance run and wrong for a paid one: nothing would ever reach the policy.
This transport posts to the session-scoped origin the executor bound, so the
tokens and logprobs in the evidence come from the model being trained.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "tests"))

import uvicorn  # noqa: E402

from synth_containers.cispo_target import (  # noqa: E402
    CispoReferenceTarget,
    DeterministicSampler,
)
from synth_containers.http_adapter import create_reference_app  # noqa: E402
from synth_containers.reference_runtime import ReferenceManagedRuntime  # noqa: E402


class HttpSampler:
    """Posts to whatever origin the binding carried. Opens a real socket."""

    def __init__(self, *, timeout: float = 120.0) -> None:
        self.timeout = timeout
        self.calls = 0
        self.probe_calls = 0
        # A probe reaches no provider by definition, and the container routes
        # it through this same transport with a `probe://` URL. A real sampler
        # therefore needs its own unpaid branch, or every deployment has to
        # invent one.
        self._probe = DeterministicSampler()

    def reachable(self, origin: Any) -> bool:
        # The origin is a loopback gateway in this run; reachability is proven
        # by the sampling call itself rather than by a pre-flight guess.
        return True

    def post(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if url.startswith("probe://"):
            self.probe_calls += 1
            return self._probe.post(url, headers=headers, body=body)
        payload = json.dumps(body).encode()
        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("content-type", "application/json")
        for name, value in headers.items():
            request.add_header(name, value)
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                reply_body = json.loads(reply.read().decode())
            capture = reply_body.get("synth_capture") or {}
            print(
                "TURN turns={turns} prompt={prompt} gen={gen} branch={branch} "
                "parent={parent} compaction={compaction}".format(
                    turns=len(body.get("messages") or ()),
                    prompt=len(capture.get("prompt_token_ids") or ()),
                    gen=len(capture.get("generation_token_ids") or ()),
                    branch=capture.get("branch_id"),
                    parent=capture.get("parent_branch_id"),
                    compaction=capture.get("compaction"),
                ),
                flush=True,
            )
            return reply_body
        except urllib.error.HTTPError as exc:  # surface the gateway's own words
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"sampler {url} returned {exc.code}: {detail}") from exc


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8210
    runtime = ReferenceManagedRuntime.counter_default(target=2)
    sampler = HttpSampler()
    CispoReferenceTarget.install(runtime, transport=sampler, reachability=sampler)
    uvicorn.run(create_reference_app(runtime), host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
