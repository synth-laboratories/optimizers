from __future__ import annotations

import threading
import time

from synth_optimizers.contracts.training_schemas import TERMINAL_STATES
from synth_optimizers.runtime import JobStore, iter_live_events, start_job_worker
from synth_optimizers.runtime.stream import format_sse, wants_live_stream


def test_append_event_wakes_a_live_tail(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.persist_prepared(
        algorithm_id="sft",
        implementation_version="sft.tinker.v1",
        provider="tinker",
        model_id="openai/gpt-oss-20b",
        idempotency_key="k1",
        config={"seed": 1},
        job_id="run_live",
    )
    seen: list[str] = []

    def _tail() -> None:
        for item in iter_live_events(store, job.job_id, idle_timeout=0.2):
            if item is None:
                continue
            seen.append(str(item["event_type"]))

    thread = threading.Thread(target=_tail)
    thread.start()
    time.sleep(0.05)
    store.append_event(job.job_id, "sft.training.started", {"ok": True}, phase="running")
    store.append_event(job.job_id, "sft.completed", {"ok": True}, phase="completed")
    store.transition(job.job_id, "completed")
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert seen == ["sft.training.started", "sft.completed", "training.lifecycle"]
    assert seen == [event["event_type"] for event in store.events(job.job_id, after_sequence=0)]
    store.close()


def test_terminal_tail_drains_all_pages(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.persist_prepared(
        algorithm_id="sft", implementation_version="sft.tinker.v1",
        provider="tinker", model_id="test", idempotency_key="paged", config={},
        job_id="run_paged",
    )
    for index in range(1001):
        store.append_event(job.job_id, "sft.step.metrics", {"index": index}, phase="running")
    store.transition(job.job_id, "completed")
    events = list(iter_live_events(store, job.job_id))
    assert len(events) == 1002
    assert [event["sequence"] for event in events] == list(range(1, 1003))
    assert events[-1]["event_type"] == "training.lifecycle"
    store.close()


def test_disconnected_tail_does_not_stop_the_journal(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.persist_prepared(
        algorithm_id="cispo",
        implementation_version="cispo.slime.v1",
        provider="tinker",
        model_id="openai/gpt-oss-20b",
        idempotency_key="k2",
        config={"seed": 1},
        job_id="run_detach",
    )
    stop = threading.Event()

    def _tail() -> None:
        for item in iter_live_events(store, job.job_id, idle_timeout=0.1):
            if stop.is_set():
                return
            if item is None:
                continue

    thread = threading.Thread(target=_tail, daemon=True)
    thread.start()
    store.append_event(job.job_id, "cispo.canary.started", {}, phase="running")
    stop.set()
    thread.join(timeout=1)
    store.append_event(job.job_id, "cispo.update.completed", {"update": 1}, phase="running")
    kinds = [event["event_type"] for event in store.events(job.job_id, after_sequence=0)]
    assert kinds == ["cispo.canary.started", "cispo.update.completed"]
    store.close()


def test_background_worker_runs_once(tmp_path) -> None:
    hits: list[int] = []

    def _run() -> None:
        hits.append(1)
        time.sleep(0.05)

    first = start_job_worker("job-a", _run)
    second = start_job_worker("job-a", _run)
    assert first is not None
    assert second is None
    first.join(timeout=2)
    assert hits == [1]


def test_sse_framing_and_stream_query() -> None:
    frame = format_sse({"sequence": 3, "event_type": "sft.step.metrics", "payload": {"step": 1}})
    assert frame.startswith("id: 3\n")
    assert "event: optimizer\n" in frame
    assert frame.endswith("\n\n")
    assert wants_live_stream("/v1/runs/x/optimizer-events/stream", {}) is True
    assert wants_live_stream("/v1/runs/x/optimizer-events", {"stream": ["1"]}) is True
    assert wants_live_stream("/v1/runs/x/optimizer-events", {}) is False
    assert "completed" in TERMINAL_STATES
