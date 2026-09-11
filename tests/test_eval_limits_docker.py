"""Opt-in real Docker actuator checks; no provider calls or deployment stack."""

import hashlib
import os
import subprocess
import time
from uuid import uuid4

import pytest

from synth_optimizers.eval.executor import OciTrialExecutor, TrialRunRequest
from synth_optimizers.eval.models import TrialLimits

pytestmark = pytest.mark.skipif(
    os.environ.get("SYNTH_EVAL_DOCKER_FIXTURE") != "1", reason="explicit local Docker fixture only"
)


@pytest.mark.parametrize("mode", ["output", "timeout"])
def test_real_container_limit_stop_and_removal(tmp_path, mode):
    image = subprocess.check_output(
        ["docker", "image", "inspect", "synth-eval-limit-fixture:20260910", "--format", "{{.Id}}"],
        text=True,
        timeout=10,
    ).strip()
    for name in ("input", "policy", "output"):
        (tmp_path / name).mkdir()
    (tmp_path / "input" / "mode").write_text(mode)
    trial_id = "limit-qualification-" + uuid4().hex[:12]
    request = TrialRunRequest(
        trial_id,
        image,
        tmp_path / "input",
        tmp_path / "policy",
        tmp_path / "output",
        TrialLimits(1, 90 if mode == "output" else 45, 1, 64, 1024 * 64),
        "none",
    )
    driver = OciTrialExecutor()
    events = []
    outcome = driver.run(
        request, on_event=events.append, should_cancel=lambda: False, heartbeat=lambda: None
    )
    assert outcome.output_limit_exceeded == (mode == "output")
    assert outcome.timed_out == (mode == "timeout")
    assert outcome.exit_code != 0
    assert any(event.get("event") == "eval.limit.stop" for event in events)
    name = f"synth-eval-{trial_id[:40]}-{hashlib.sha256(trial_id.encode()).hexdigest()[:12]}"
    # --rm removal can lag the attached CLI's exit; inspect only this owned name.
    for _ in range(20):
        remaining = subprocess.check_output(
            ["docker", "container", "ls", "-aq", "--filter", f"name=^/{name}$"],
            text=True,
            timeout=10,
        ).strip()
        if not remaining:
            break
        time.sleep(0.1)
    assert remaining == "", f"owned fixture resource remains: {name}"
