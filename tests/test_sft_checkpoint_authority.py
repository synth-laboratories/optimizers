import pytest
import json
import time
import urllib.request

from synth_optimizers.eval.checkpoint_authority import CheckpointEvaluationAuthority
from synth_optimizers.eval.executor import TrialExecution
from synth_optimizers.providers.tinker import FakeTinkerProvider, TinkerAdapter, TinkerCredentials
from synth_optimizers.runtime import JobStore
from synth_optimizers.sft_executor import TinkerSftExecutor


PROFILE = dict(profile_id="fixture.text.v1", package="fixture", package_version="1",
               config_digest="sha256:"+"a"*64, tokenizer_id="fixture",
               tokenizer_digest="sha256:"+"b"*64, stop_token_ids=(2,))


class CheckpointProvider(FakeTinkerProvider):
    def sample(self, handle, request):
        self._record("sample", request.request_id)
        return {"token_ids": [65 + handle.step], "logprobs": [-.1], "text": chr(97+handle.step),
                "finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}


class HttpTarget:
    """A deterministic eval.target.v1 double that actually calls the checkpoint HTTP route."""
    def run(self, request, *, on_event, should_cancel, heartbeat):
        trial = json.loads((request.input_dir / "trial.json").read_text())
        route = trial["models"][0]
        body = {"model": route["id"], "policy_snapshot_id": trial["policy_snapshot_id"],
                "messages": [{"role": "user", "content": "respond"}], "max_tokens": 2}
        req = urllib.request.Request(route["route"], data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer "+request.secrets[route["secret"]], "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
        assert result["synth"]["policy_snapshot_id"] == trial["policy_snapshot_id"]
        value = ord(result["choices"][0]["message"]["content"])-97
        (request.output_dir / "trace.json").write_text(json.dumps({"request": body, "response": result}))
        (request.output_dir / "result.json").write_text(json.dumps({
            "schema_version": "eval.container-result.v1", "trial_id": trial["trial_id"],
            "status": "evaluated", "benchmark_status": "passed", "metrics": {"accuracy": value},
            "gates": [{"id": gate, "passed": True} for gate in trial["required_gates"] if gate != "exact_checkpoint"],
            "artifacts": [{"role": "trace", "path": "trace.json"}]}))
        return TrialExecution(0, False, False, time.time(), time.time(), "")


@pytest.mark.parametrize("direction,final_value", [("maximize", 2), ("minimize", 1)])
def test_container_checkpoints_have_real_child_jobs_live_results_and_heldout(tmp_path, direction, final_value):
    home = tmp_path / "eval"
    authority = CheckpointEvaluationAuthority(home, executor=HttpTarget())
    digest = "sha256:"+"c"*64
    (home / "pins.toml").write_text('[pins."eval.tinker.checkpoint.gsm8k.v1"]\nimage_digest = "'+digest+'"\n')
    evaluator = {"id": "environment", "recipe_id": "eval.tinker.checkpoint.gsm8k.v1",
                 "image_digest": digest, "selection_seeds": [101,102], "final_seeds": [201,202],
                 "metric_ref": "accuracy", "reward_version": "gsm8k.exact.v1", "units": "fraction"}
    config = {"training": {"steps": 2}, "checkpoint_steps": [1,2],
              "checkpoint_evaluation": {"mode": "container", "evaluators": [evaluator],
                  "selection": {"evaluator_id": "environment", "direction": direction}},
              "evaluation_renderer_profile": PROFILE,
              "examples": [{"text": "hello", "category": "arbitrary completion"}]}
    store = JobStore(tmp_path / "jobs.sqlite")
    provider = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=CheckpointProvider())
    executor = TinkerSftExecutor(store, provider, eval_authority=authority)
    result = executor.submit(config, job_id="container")
    assert result["status"] == "completed", result
    events = store.events("container", limit=5000)
    evaluations = [e["payload"] for e in events if e["kind"] == "sft.child_eval.completed"]
    assert len(evaluations) == 4
    assert [e["value"] for e in evaluations] == [0,1,2,final_value]
    assert len({e["eval_job_id"] for e in evaluations}) == 4
    assert all(e["rollouts"] and e["evidence_refs"] for e in evaluations)
    assert [e["role"] for e in evaluations] == ["baseline", "selection", "selection", "final"]
    first_child = next(e["sequence"] for e in events if e["kind"] == "sft.child_eval.completed")
    last_train = max(e["sequence"] for e in events if e["kind"] == "sft.step.metrics")
    assert first_child < last_train
    final_seeds = {r["seed"] for r in evaluations[-1]["rollouts"]}
    assert final_seeds == {201,202}
    from synth_optimizers.read_models import sft_collections
    assert len(sft_collections(store, "container", collection="child_evaluations").items) == 4
    assert len(sft_collections(store, "container", collection="rollouts").items) == 8
    assert sft_collections(store, "container", collection="evidence_refs").items
    store.close()


def test_missing_pinned_image_rejected_before_any_provider_session(tmp_path):
    class MissingImage(HttpTarget):
        def resolve_reference(self, image, digest):
            raise RuntimeError("pinned image is missing")
    home = tmp_path / "eval"
    authority = CheckpointEvaluationAuthority(home, executor=MissingImage())
    digest = "sha256:" + "c" * 64
    (home / "pins.toml").write_text('[pins."eval.tinker.checkpoint.gsm8k.v1"]\nimage_digest = "'+digest+'"\n')
    evaluator = {"id": "environment", "recipe_id": "eval.tinker.checkpoint.gsm8k.v1",
        "image_digest": digest, "selection_seeds": [101], "final_seeds": [201],
        "metric_ref": "accuracy", "reward_version": "gsm8k.exact.v1", "units": "fraction"}
    transport = CheckpointProvider()
    provider = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = TinkerSftExecutor(store, provider, eval_authority=authority)
    with pytest.raises(RuntimeError, match="pinned image is missing"):
        executor.submit({"training": {"steps": 2}, "checkpoint_steps": [1,2],
            "checkpoint_evaluation": {"mode": "container", "evaluators": [evaluator],
                "selection": {"evaluator_id": "environment", "direction": "maximize"}},
            "evaluation_renderer_profile": PROFILE, "examples": [{"text": "hello", "category": "world"}]}, job_id="missing")
    assert not transport.calls
    store.close()
