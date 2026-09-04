from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from synth_optimizers.contracts.rl_identity import TaskSpec
from synth_optimizers.rl.config import load
from synth_optimizers.rl.ports import PolicyRevision, SamplerOrigin


SCRIPT = Path(__file__).parents[1] / "docs/e2e/screen_banking77.py"
SPEC = importlib.util.spec_from_file_location("screen_banking77", SCRIPT)
assert SPEC and SPEC.loader
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


class FakeReward:
    optimized_channel = "score::team-0"
    terminal_status = "completed"

    def __init__(self, value: float) -> None:
        self._value = value

    def validate(self) -> None:
        pass

    def value(self, channel: str) -> float:
        assert channel == self.optimized_channel
        return self._value


class FakeBinder:
    def __init__(self, revision: PolicyRevision) -> None:
        self.revision = revision

    def resolve(self, selector: str):
        assert selector == "checkpoint-1"
        return {"pg-0": self.revision}

    def train(self, **_kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("screening must not train")

    def publish(self, **_kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("screening must not save/publish")


class FakeGateway:
    def __init__(self) -> None:
        self.closed: list[str] = []
        self.declared: list[str] = []

    def bind(self, revision, *, proxy_request_id, **_kwargs):
        return SamplerOrigin(
            base_url="http://sampler.invalid",
            credential="secret",
            proxy_request_id=proxy_request_id,
            policy_revision=revision.revision,
            behavior_fingerprint=revision.behavior_fingerprint,
            wire_api="responses",
            sampling_transport="message_in_capture_out",
        )

    def declare_attempt(self, proxy_request_id: str, **_kwargs) -> None:
        self.declared.append(proxy_request_id)

    def close(self, proxy_request_id: str) -> None:
        self.closed.append(proxy_request_id)


class FakeSession:
    handshake_id = "handshake-1"
    agreement_digest = "agreement-1"
    capability = SimpleNamespace(
        container_image_digest="sha256:image",
        topology=SimpleNamespace(topology_id="banking77.classify.solo.v1"),
    )
    startup = SimpleNamespace(contract=SimpleNamespace(contract_hash="sha256:contract"))

    def __init__(self, tasks: tuple[TaskSpec, ...]) -> None:
        self._tasks = tasks
        self.active: set[str] = set()
        self.max_active = 0
        self.seeds: dict[str, int] = {}
        self.terminated: list[str] = []

    def tasks(self, **_kwargs):
        return self._tasks

    def submit(self, task, _origin, *, sample_index, **_kwargs):
        rollout = f"rollout-{task.task_id.rsplit('/', 1)[-1]}-{sample_index}"
        self.active.add(rollout)
        self.max_active = max(self.max_active, len(self.active))
        self.seeds[rollout] = task.seed
        return rollout

    def poll(self, _rollout_id: str):
        return {"state": "completed", "terminal": True}

    def finalize(self, rollout_id: str) -> None:
        self.active.remove(rollout_id)

    def evidence(self, rollout_id: str):
        sample = int(rollout_id.rsplit("-", 1)[-1])
        task = rollout_id.split("-")[-2]
        # task 10 is mixed (4/8); task 20 is solved (8/8).
        value = float(sample < 4) if task == "10" else 1.0
        return SimpleNamespace(trace_digest=f"trace-{rollout_id}"), FakeReward(value)

    def terminate(self, _rollout_id: str, **_kwargs) -> None:
        self.terminated.append(_rollout_id)
        self.active.discard(_rollout_id)


def _config():
    config = load(Path(__file__).parents[1] / "docs/e2e/configs/run_b77_hard20_paid_12.toml")
    return replace(
        config,
        taskset=replace(
            config.taskset,
            train_ids=("banking77/train/10", "banking77/train/20"),
        ),
    )


def test_screen_runs_single_arm_with_bound_and_writes_durable_receipts(tmp_path: Path) -> None:
    config = _config()
    tasks = tuple(
        TaskSpec(
            task_id=task_id,
            split="train",
            seed=100,
            group_id="source",
            task_family="banking77",
            content_digest=f"digest-{task_id}",
        )
        for task_id in config.taskset.train_ids
    )
    revision = PolicyRevision(
        revision=20,
        revision_id="pg-0@20",
        checkpoint_id="checkpoint-1",
        parameter_group_id="pg-0",
        sampler_reference="tinker://sampler",
        behavior_fingerprint="fingerprint-1",
    )
    session = FakeSession(tasks)
    gateway = FakeGateway()
    plane = SimpleNamespace(session=session, gateway=gateway, binder=FakeBinder(revision))

    manifest = screen.run_screen(
        config,
        plane,
        selector="checkpoint-1",
        output=tmp_path,
        samples=8,
        concurrency=3,
        poll_interval=0,
    )

    assert manifest["attempt_count"] == 16
    assert manifest["selected_train_ids"] == ["banking77/train/10"]
    assert session.max_active == 3
    assert len(gateway.declared) == len(gateway.closed) == 16
    # Samples repeat the exact task instance. The sample index/idempotency key,
    # not a fabricated dataset seed, distinguishes stochastic rollouts.
    assert list(session.seeds.values()) == [100] * 16
    attempts = json.loads((tmp_path / "attempts.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert len(attempts) == 16
    assert summary["tasks"][0]["successes"] == 4
    assert summary["tasks"][1]["successes"] == 8
    assert persisted == manifest


def test_screen_receipt_keeps_declared_task_seed_constant(tmp_path: Path) -> None:
    config = replace(
        _config(), taskset=replace(_config().taskset, train_ids=("banking77/train/10",))
    )
    task = TaskSpec(
        task_id="banking77/train/10",
        split="train",
        seed=407,
        group_id="source",
        task_family="banking77",
        content_digest="digest-task",
    )
    revision = PolicyRevision(
        revision=20,
        revision_id="pg-0@20",
        checkpoint_id="checkpoint-1",
        parameter_group_id="pg-0",
        sampler_reference="tinker://sampler",
        behavior_fingerprint="fingerprint-1",
    )
    plane = SimpleNamespace(
        session=FakeSession((task,)), gateway=FakeGateway(), binder=FakeBinder(revision)
    )

    screen.run_screen(
        config,
        plane,
        selector="checkpoint-1",
        output=tmp_path,
        samples=8,
        concurrency=4,
        poll_interval=0,
    )

    attempts = json.loads((tmp_path / "attempts.json").read_text())
    assert [row["sample_index"] for row in attempts] == list(range(8))
    assert {row["base_seed"] for row in attempts} == {407}
    assert {row["seed"] for row in attempts} == {407}


def test_selected_task_ids_excludes_zero_and_all_correct() -> None:
    rows = [
        {"task_id": "zero", "successes": 0, "samples": 8},
        {"task_id": "mixed", "successes": 7, "samples": 8},
        {"task_id": "all", "successes": 8, "samples": 8},
    ]
    assert screen.selected_task_ids(rows) == ["mixed"]


def test_failed_attempt_aborts_and_cleans_up_all_other_active_rollouts(tmp_path: Path) -> None:
    config = replace(
        _config(),
        taskset=replace(_config().taskset, train_ids=("banking77/train/10",)),
    )
    task = TaskSpec(
        task_id=config.taskset.train_ids[0],
        split="train",
        seed=100,
        group_id="source",
        task_family="banking77",
        content_digest="digest-task",
    )
    revision = PolicyRevision(
        revision=20,
        revision_id="pg-0@20",
        checkpoint_id="checkpoint-1",
        parameter_group_id="pg-0",
        sampler_reference="tinker://sampler",
        behavior_fingerprint="fingerprint-1",
    )
    session = FakeSession((task,))
    session.poll = lambda rollout_id: (
        {"state": "failed", "terminal": True}
        if rollout_id.endswith("-0")
        else {"state": "running", "terminal": False}
    )
    gateway = FakeGateway()
    plane = SimpleNamespace(session=session, gateway=gateway, binder=FakeBinder(revision))

    with pytest.raises(RuntimeError, match="terminal state 'failed'"):
        screen.run_screen(
            config,
            plane,
            selector="checkpoint-1",
            output=tmp_path,
            samples=3,
            concurrency=3,
            poll_interval=0,
        )

    assert sorted(session.terminated) == ["rollout-10-0", "rollout-10-1", "rollout-10-2"]
    assert session.active == set()
    assert len(gateway.closed) == 3
    assert not (tmp_path / "manifest.json").exists()
