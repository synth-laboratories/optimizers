"""Serve the CISPO reference container over real HTTP.

Runs in the synth-containers worktree's environment. The optimizer drives this
as a separate process over a socket, which is the only way to find out what
in-process tests cannot tell us.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

WORKTREE = Path("/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "tests"))

import uvicorn  # noqa: E402

from synth_containers.cispo_target import CispoTargetError, DeterministicSampler  # noqa: E402
from synth_containers.http_adapter import create_reference_app  # noqa: E402
from test_cispo_target import installed  # noqa: E402


class RosterSampler(DeterministicSampler):
    """Two samples of the same task have to be allowed to differ.

    The reference sampler always picks the first legal action, so every episode
    in a group scores identically, CISPO skips the group for zero advantage --
    correctly -- and no training step is ever reached. A real policy's samples
    differ; this one differs per attempt, deterministically, by hashing the
    per-attempt origin the container was told to sample through. It still
    reaches no network and still spends nothing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def post(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._local.url = url
        return super().post(url, headers=headers, body=body)

    def _choose(self, prompt: str) -> str:
        marker = "valid_actions="
        start = prompt.rfind(marker)
        if start < 0:
            raise CispoTargetError("the rendered prompt names no legal action list")
        raw = prompt[start + len(marker) :].strip().splitlines()[0]
        actions = json.loads(raw)
        if not isinstance(actions, Sequence) or not actions:
            raise CispoTargetError("the rendered prompt names an empty legal action list")
        url = str(getattr(self._local, "url", ""))
        index = int(hashlib.sha256(url.encode("utf-8")).hexdigest()[:8], 16) % len(actions)
        return str(actions[index])


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
    runtime, _target = installed(target_count=2, transport=RosterSampler())
    app = create_reference_app(runtime)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
