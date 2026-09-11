import json

import pytest

from synth_optimizers.eval.semaphore import SemaphoreTimeout, TrialSemaphore


def test_unknown_resource_stop_retains_capacity_after_release_and_restart(tmp_path):
    store = TrialSemaphore(tmp_path, capacity=1, ttl_seconds=1)
    lease = store.acquire(run_id="r1", trial_id="t1")
    store.quarantine(lease, {"container_id": "c1", "runtime": "docker"})
    record = json.loads(lease.path.read_text())
    record.update({"pid": -1, "expires_at": 0})
    lease.path.write_text(json.dumps(record))
    store.release(lease)
    assert store.release_run("r1") == 0
    restarted = TrialSemaphore(tmp_path, capacity=1, ttl_seconds=1)
    assert restarted.snapshot()["available"] == 0
    assert restarted.snapshot()["leases"][0]["resource"]["container_id"] == "c1"
    with pytest.raises(SemaphoreTimeout):
        restarted.acquire(run_id="r2", trial_id="t2", timeout_seconds=0)


@pytest.mark.parametrize("release_run", [False, True])
def test_bound_launch_survives_owner_loss_and_run_release(tmp_path, release_run):
    store = TrialSemaphore(tmp_path, capacity=1, ttl_seconds=1)
    lease = store.acquire(run_id="r1", trial_id="t1")
    store.bind_resource(lease, {"container_id": "c1", "runtime": "docker"})
    record = json.loads(lease.path.read_text())
    record.update({"pid": -1, "expires_at": 0})
    lease.path.write_text(json.dumps(record))
    if release_run:
        assert store.release_run("r1") == 0
    snapshot = store.snapshot()
    assert snapshot["available"] == 0
    assert snapshot["leases"][0]["quarantined"] is True


def test_missing_heartbeat_record_fails_instead_of_allowing_execution(tmp_path):
    store = TrialSemaphore(tmp_path, capacity=1, ttl_seconds=1)
    lease = store.acquire(run_id="r1", trial_id="t1")
    lease.path.unlink()
    with pytest.raises(FileNotFoundError):
        store.heartbeat(lease)
