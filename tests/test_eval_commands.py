from types import SimpleNamespace

from synth_optimizers.eval.commands import _runtime_recipe_readiness
from synth_optimizers.eval.executor import ContainerRuntimeError


def recipe(*, digest: str | None = "sha256:" + "ab" * 32):
    return SimpleNamespace(
        id="eval.craftax.code-policy.smoke.v1",
        image="ghcr.io/synth-laboratories/workshop-craftax-eval-target",
        image_digest=digest,
        unavailable_reason=None if digest else "target image is not published and pinned yet",
    )


class Executor:
    def __init__(self, error: str | None = None) -> None:
        self.error = error

    def resolve_reference(self, image: str, digest: str) -> str:
        if self.error:
            raise ContainerRuntimeError(self.error)
        return f"{image}@{digest}"


def test_doctor_blocks_digest_pinned_but_missing_image_before_run_creation():
    result = _runtime_recipe_readiness([recipe()], Executor("image is not present locally"))
    assert result == [
        {
            "id": "eval.craftax.code-policy.smoke.v1",
            "available": False,
            "reason": "image is not present locally",
            "image": "ghcr.io/synth-laboratories/workshop-craftax-eval-target",
            "imageDigest": "sha256:" + "ab" * 32,
            "resolvedReference": None,
        }
    ]


def test_doctor_advertises_only_the_exact_resolved_digest():
    result = _runtime_recipe_readiness([recipe()], Executor())
    assert result[0]["available"] is True
    assert result[0]["resolvedReference"].endswith("@sha256:" + "ab" * 32)


def event_args(home, **overrides):
    fields = {
        "eval_command": "events",
        "home": str(home),
        "run_id": "run-1",
        "after_sequence": 0,
        "limit": 100,
        "follow": False,
        "timeout_seconds": 0.01,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_events_replay_does_not_need_runtime_config_or_results_index(tmp_path, capsys):
    import json

    from synth_optimizers.eval.commands import dispatch
    from synth_optimizers.eval.runner import EventEmitter

    path = tmp_path / "runs" / "run-1" / "events.jsonl"
    emitter = EventEmitter(path, run_id="run-1")
    emitter.emit("eval.run.started")
    emitter.emit("eval.trial.completed", score=None)
    (tmp_path / "runtime.toml").write_text("invalid [[ TOML")
    assert dispatch(event_args(tmp_path, after_sequence=1)) == 0
    page = json.loads(capsys.readouterr().out)
    assert page["run_id"] == "run-1"
    assert page["next_sequence"] == 2
    assert page["events"][0]["score"] is None
    assert not (tmp_path / "secrets.toml").exists()
    assert not (tmp_path / "pins.toml").exists()


def test_events_follow_keeps_worker_schema_and_cursor(tmp_path, capsys):
    import json

    from synth_optimizers.eval.commands import dispatch
    from synth_optimizers.eval.runner import EventEmitter

    path = tmp_path / "runs" / "run-1" / "events.jsonl"
    EventEmitter(path, run_id="run-1").emit("eval.run.started")
    assert dispatch(event_args(tmp_path, follow=True)) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["seq"] == 1
    assert event["event"] == "eval.run.started"
    assert "schema_version" in event


def test_events_missing_run_does_not_create_home(tmp_path, capsys):
    from synth_optimizers.eval.commands import dispatch

    home = tmp_path / "absent"
    assert dispatch(event_args(home)) == 1
    assert not home.exists()
    assert "error:" in capsys.readouterr().err


def test_events_refuses_path_traversal_and_foreign_journal(tmp_path, capsys):
    from synth_optimizers.eval.commands import dispatch
    from synth_optimizers.eval.runner import EventEmitter

    assert dispatch(event_args(tmp_path, run_id="../outside")) == 1
    path = tmp_path / "runs" / "run-1" / "events.jsonl"
    EventEmitter(path, run_id="foreign").emit("eval.run.started")
    assert dispatch(event_args(tmp_path)) == 1
    assert "identity mismatch" in capsys.readouterr().err


def test_events_refuses_symlink_outside_runs(tmp_path, capsys):
    from synth_optimizers.eval.commands import dispatch

    outside = tmp_path / "outside"
    outside.mkdir()
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    (runs / "run-1").symlink_to(outside, target_is_directory=True)
    assert dispatch(event_args(tmp_path / "home")) == 1
    assert "escapes" in capsys.readouterr().err
