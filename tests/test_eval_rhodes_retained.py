"""A retained native binding must never fall back to OCI execution."""

import json

import pytest

from synth_optimizers.eval import EvalHome, EvalRunner, WorkerManifest
from synth_optimizers.eval import runner as runner_module
from synth_optimizers.eval.staging import CandidateSource, stage_candidate_set


@pytest.mark.parametrize("recipe", ["eval.rhodes.retained-native.v1", "eval.rhodes.retained-native.v2"])
def test_missing_retained_transport_refuses_before_oci(tmp_path, monkeypatch, recipe):
    home = EvalHome.open(tmp_path / "home")
    home.write_pin(recipe, "sha256:" + "a" * 64)
    source = tmp_path / "binding.json"
    source.write_text(json.dumps({"original": "retained-only"}))
    candidates = stage_candidate_set(home, [CandidateSource(label="original", path=source,
        entrypoint="binding.json", kind="rhodes.native-binding.v1", is_baseline=True)])
    manifest = WorkerManifest(run_id="retained-test", recipe_id=recipe, home=home.root,
        candidate_set_path=candidates.root / "candidate_set.json", session_ref=None)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("OCI executor must not be constructed for retained data")
    monkeypatch.setattr(runner_module, "OciTrialExecutor", forbidden)
    runner = EvalRunner(manifest)
    assert runner.execute() == 1
    journal = (home.root / "runs" / manifest.run_id / "events.jsonl").read_text()
    assert "OCI fallback is forbidden" in journal
    assert not calls
