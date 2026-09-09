"""Real process exits exercise the installed journal, not an in-memory cache."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from synth_optimizers.runtime import JobStore, JobStoreError
from synth_optimizers.runtime.operations import DurableProvider, UncertainOperation


def prepare(path):
    store = JobStore(path)
    store.persist_prepared(algorithm_id="sft", implementation_version="sft.tinker.v1",
        provider="tinker", model_id="model", idempotency_key="run", config={}, job_id="run")
    return store


@pytest.mark.parametrize("kind", ["session", "train", "save", "sample", "forward"])
@pytest.mark.parametrize("boundary", ["intent", "dispatched", "confirmed"])
def test_process_exit_retains_exact_operation_outcome(tmp_path, kind, boundary):
    path = tmp_path / "jobs.sqlite"
    prepare(path).close()
    code = '''
import os, sys
from pathlib import Path
from synth_optimizers.runtime import JobStore
from synth_optimizers.runtime.operations import DurableProvider
path, kind, boundary = sys.argv[1:]
store = JobStore(path)
store.claim("run", "original")
provider = DurableProvider(object(), store, "run", "original")
def call():
    if boundary == "intent": os._exit(71)
    with open(path + ".dispatch", "a") as log:
        log.write(kind + "\\n"); log.flush(); os.fsync(log.fileno())
    if boundary == "dispatched": os._exit(72)
    return {"provider_result_id": "retained-result"}
provider._call(kind, "request", {"input": "frozen"}, call)
os._exit(73)
'''
    child = subprocess.run([sys.executable, "-c", code, str(path), kind, boundary],
        cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, timeout=15)
    assert child.returncode == {"intent": 71, "dispatched": 72, "confirmed": 73}[boundary]
    store = JobStore(path)
    store.claim("run", "replacement", stale_after_seconds=-1)
    provider = DurableProvider(object(), store, "run", "replacement")
    calls = []
    def replay():
        calls.append("unexpected replay")
    if boundary == "confirmed":
        assert provider._call(kind, "request", {"input": "frozen"}, replay) == {
            "provider_result_id": "retained-result"}
    else:
        with pytest.raises(UncertainOperation, match="reconciliation"):
            provider._call(kind, "request", {"input": "frozen"}, replay)
    assert calls == []
    dispatched = Path(str(path) + ".dispatch")
    assert (dispatched.read_text().splitlines() if dispatched.exists() else []) == (
        [] if boundary == "intent" else [kind])
    store.close()


@pytest.mark.parametrize("phase", ["running", "evaluating", "materializing"])
def test_independent_processes_accept_only_one_owner(tmp_path, phase):
    path = tmp_path / "jobs.sqlite"
    store = prepare(path)
    store.transition("run", phase)
    code = '''
import sys
from synth_optimizers.runtime import JobStore, JobStoreError
store = JobStore(sys.argv[1])
try:
    store.claim("run", sys.argv[2])
    with store.owned(sys.argv[2]):
        store.append_event("run", "accepted", {"owner": sys.argv[2]}, phase="running")
except JobStoreError:
    sys.exit(2)
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(path), owner], cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}) for owner in ("one", "two")]
    assert sorted(child.wait(timeout=15) for child in children) == [0, 2]
    assert len([event for event in store.events("run") if event["kind"] == "accepted"]) == 1
    store.close()


def test_long_evaluation_renews_lease_across_processes(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = prepare(path)
    code = '''
import sys, time
from pathlib import Path
from synth_optimizers.runtime import JobStore
from synth_optimizers.runtime.worker import execute_owned
store = JobStore(sys.argv[1])
def evaluate(job, owner):
    store.transition("run", "evaluating")
    Path(sys.argv[1] + ".ready").touch()
    time.sleep(34)
    store.append_event("run", "evaluation.accepted", {}, phase="evaluating")
execute_owned(store, "run", evaluate)
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    try:
        deadline = time.monotonic() + 10
        while not Path(str(path) + ".ready").exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        original = store.require("run").heartbeat_at
        time.sleep(31)  # Longer than the production 30-second stale lease.
        assert store.require("run").heartbeat_at != original
        with pytest.raises(JobStoreError, match="active owner"):
            store.claim("run", "contender")
        assert child.wait(timeout=10) == 0
        assert len([e for e in store.events("run") if e["kind"] == "evaluation.accepted"]) == 1
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        store.close()
