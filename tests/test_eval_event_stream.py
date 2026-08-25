from __future__ import annotations

import subprocess
import io
import json
import threading

from synth_optimizers.eval.executor import OciTrialExecutor, _poll_events


def test_published_event_url_is_loopback_and_uses_runtime_port(monkeypatch) -> None:
    executor = object.__new__(OciTrialExecutor)
    executor.binary = "/usr/bin/docker"
    monkeypatch.setattr(executor, "_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="127.0.0.1:49177\n", stderr=""
        ),
    )

    assert (
        executor._published_event_url("synth-eval-trial-a")
        == "http://127.0.0.1:49177"
    )


def test_declared_cursor_stream_is_the_primary_event_source(monkeypatch) -> None:
    page = {
        "events": [
            {
                "schema_version": "synth.trace-stream-event.v1",
                "sequence": 1,
                "event": "environment.step",
                "actions": ["up"],
            }
        ],
        "cursor": {"next": 1, "high_water": 1, "has_more": False, "closed": True},
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(json.dumps(page).encode()),
    )
    observed = []
    _poll_events("http://127.0.0.1:49177/rollouts/trial/events", observed.append, threading.Event())

    assert observed == page["events"]
