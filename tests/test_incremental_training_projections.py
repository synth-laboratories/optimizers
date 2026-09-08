import json
import sqlite3

import pytest

from synth_optimizers.runtime import JobStore
from synth_optimizers.read_models import reduce_summary, sft_collections


def prepared(path):
    store = JobStore(path)
    store.persist_prepared(algorithm_id="sft", implementation_version="sft.tinker.v1", provider="tinker",
                           model_id="model", idempotency_key="key", config={}, job_id="run")
    return store


def test_pages_and_historical_summaries_use_incremental_index_after_restart(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = prepared(path)
    for step in range(130):
        store.append_event("run", "sft.step.metrics", {"step": step, "loss": 1 / (step + 1)}, phase="running")
    first = sft_collections(store, "run", collection="training_metrics")
    bound = first.projected_at_sequence
    store.close()
    store = JobStore(path)
    # A normal read must not reconstruct the journal through the event API.
    store.events = lambda *args, **kwargs: pytest.fail("read replayed raw journal")
    store.put_receipt("run", "sample", {"request_id": "sample", "cost_usd": .1, "input_tokens": 7})
    second = sft_collections(store, "run", collection="training_metrics", after_key=first.next_key)
    assert [row["step"] for row in second.items] == list(range(100, 130))
    assert second.projected_at_sequence == bound
    assert reduce_summary(store, "run", at_sequence=bound)["usage"]["cost_usd"] is None
    assert reduce_summary(store, "run")["usage"]["input_tokens"] == 7
    store.put_receipt("run", "sample", {"request_id": "sample", "cost_usd": .2, "input_tokens": 9})
    assert reduce_summary(store, "run")["usage"]["input_tokens"] == 9
    assert reduce_summary(store, "run")["usage"]["cost_usd"] == pytest.approx(.2)
    assert len(sft_collections(store, "run", collection="receipts").items) == 1
    store.close()


def test_large_details_are_offloaded_to_exact_immutable_source(tmp_path):
    store = prepared(tmp_path / "jobs.sqlite")
    payload = {"step": 1, "detail": "🐈" * 30_000}
    event = store.append_event("run", "sft.step.metrics", payload, phase="running")
    page = sft_collections(store, "run", collection="training_metrics", byte_limit=2048)
    row = page.items[0]
    assert row["details_offloaded"] and page.bytes <= 2048
    assert row["source_ref"]["sequence"] == event["sequence"]
    assert store.events("run", after_sequence=event["sequence"] - 1, limit=1)[0]["payload"] == payload
    assert len(json.dumps(reduce_summary(store, "run"))) < 2048
    store.close()


def test_projection_and_event_commit_or_rollback_together(tmp_path):
    store = prepared(tmp_path / "jobs.sqlite")
    before = reduce_summary(store, "run")
    with pytest.raises(RuntimeError, match="crash"):
        with store._write("run"):
            store._insert_event("run", "sft.step.metrics", {"step": 1}, "running")
            raise RuntimeError("crash before transaction commit")
    assert reduce_summary(store, "run") == before
    assert not sft_collections(store, "run", collection="training_metrics").items
    store.close()


def test_old_database_backfills_once_without_changing_journal(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = prepared(path)
    store.append_event("run", "sft.step.metrics", {"step": 1}, phase="running")
    original = store.events("run")
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE training_projection_v1")
        db.execute("DROP TABLE training_summary_v1")
    store = JobStore(path)
    assert sft_collections(store, "run", collection="training_metrics").items[0]["step"] == 1
    assert store.events("run") == original
    store.append_event("run", "sft.step.metrics", {"step": 2}, phase="running")
    assert reduce_summary(store, "run")["progress"]["completed_units"] == 2
    assert len(sft_collections(store, "run", collection="training_metrics").items) == 2
    store.close()


def test_receipt_identity_is_normalized_and_conflicts_rejected(tmp_path):
    store = prepared(tmp_path / "jobs.sqlite")
    store.put_receipt("run", "request", {"cost_usd": .1})
    assert sft_collections(store, "run", collection="receipts").items[0]["request_id"] == "request"
    with pytest.raises(ValueError, match="durable key"):
        store.put_receipt("run", "request", {"request_id": "different"})
    assert reduce_summary(store, "run")["usage"]["cost_usd"] == pytest.approx(.1)
    store.close()
