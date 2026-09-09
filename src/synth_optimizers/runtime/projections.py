"""Incremental, versioned training read indexes derived from the durable journal.

Writes share the event transaction. Old databases backfill once in bounded batches;
normal page and historical-summary reads do not replay the journal.
"""
import json

TABLE = "training_projection_v1"
SNAPSHOTS = "training_summary_v1"
MAX_ROW_BYTES = 16_384


def setup(db):
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            job_id TEXT NOT NULL, collection TEXT NOT NULL, item_key TEXT NOT NULL,
            sequence INTEGER NOT NULL, ordinal INTEGER NOT NULL, row_json TEXT NOT NULL,
            PRIMARY KEY(job_id, collection, sequence, ordinal));
        CREATE INDEX IF NOT EXISTS training_projection_v1_key
            ON {TABLE}(job_id, collection, item_key, sequence);
        CREATE TABLE IF NOT EXISTS {SNAPSHOTS} (
            job_id TEXT NOT NULL, sequence INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
            PRIMARY KEY(job_id, sequence));
    """)


def compact(row, event):
    encoded = json.dumps(row, ensure_ascii=True)
    if len(encoded.encode()) <= MAX_ROW_BYTES:
        return row
    identity = {key: row[key] for key in ("item_id", "checkpoint_id", "evaluation_id", "eval_job_id",
                "step", "update", "group_id", "event_id", "request_id", "name", "digest", "phase",
                "input_tokens", "output_tokens", "training_tokens", "cost_usd", "cost_missing") if key in row}
    return {**identity, "details_offloaded": True, "source_bytes": len(encoded.encode()),
            "source_ref": {"job_id": event["job_id"], "event_id": event["event_id"],
                           "sequence": event["sequence"], "collection": "events"}}


def rows_for(event):
    from ..read_models import _collection_item
    kind, payload = event["kind"], event["payload"]
    algorithm, _, fact = kind.partition(".")
    mapping = {
        "step.metrics": [("training_metrics", "step"), ("metric_points", "step")],
        "update.completed": [("iterations", "update"), ("metric_points", "update")],
        "checkpoint.created": [("checkpoints", "checkpoint_id"), ("candidates", "checkpoint_id")],
        "checkpoint_eval.completed": [("checkpoint_evaluations", "checkpoint_id"), ("evaluations", "checkpoint_id")],
        "heldout_eval.completed": [("per_intent", "event_id")],
        "dataset.validated": [("dataset_errors", "event_id")],
        "rollout_group.completed": [(name, "group_id") for name in ("rollout_groups", "rollouts", "reward_distributions")],
        "group_advantage.computed": [("advantage_distributions", "group_id")],
        "importance_ratio.measured": [("importance_ratios", "update")],
        "zero_advantage.detected": [("zero_advantage_groups", "group_id")],
    }
    if algorithm in {"sft", "cispo"}:
        for collection, key in mapping.get(fact, []):
            row = _collection_item(event, key)
            if algorithm == "sft" and collection in {"checkpoint_evaluations", "evaluations"}:
                row["item_id"] = event["event_id"]
                key = "item_id"
            yield collection, str(row[key]), compact(row, event)
    if kind == "training.receipt":
        yield "receipts", payload["request_id"], compact(dict(payload), event)
    if kind == "training.artifact":
        row = {key: payload[key] for key in ("name", "digest")}
        yield "artifacts", row["name"], row
    if kind == "sft.child_eval.completed":
        result = {key: value for key, value in payload.items() if key not in {"rollouts", "evidence_refs"}}
        result["item_id"] = payload["eval_job_id"]
        yield "child_evaluations", result["item_id"], compact(result, event)
        evaluation = {**result, "evaluation_id": payload["eval_job_id"], "checkpointId": payload["checkpoint_id"],
                      "phase": payload["role"], "score": payload.get("value"),
                      "evaluator": payload["evaluator_id"], "metric": payload.get("metric_ref")}
        for collection in ("evaluations", "checkpoint_evaluations"):
            yield collection, result["item_id"], compact(evaluation, event)
        provenance = {key: payload[key] for key in ("eval_job_id", "checkpoint_id", "evaluator_id", "role")}
        for collection in ("rollouts", "evidence_refs"):
            for index, reference in enumerate(payload.get(collection, [])):
                key = f"{payload['eval_job_id']}:{collection}:{index}"
                yield collection, key, compact({**provenance, "reference": reference, "item_id": key}, event)


def initial_summary():
    return {"state": "prepared", "error": None, "has_lifecycle": False, "steps": 0,
            "metric": None, "checkpoint": None, "receipt_count": 0, "missing_cost_count": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "training_tokens": 0, "cost_usd": 0.0}}


def apply_summary(db, summary, event):
    kind, payload = event["kind"], event["payload"]
    if kind == "training.lifecycle":
        summary.update(state=payload["state"], error=payload.get("error"), has_lifecycle=True)
    elif not summary["has_lifecycle"]:
        summary["state"] = event["phase"]
    if kind in {"sft.step.metrics", "cispo.update.completed"}:
        summary["steps"] += 1
        summary["metric"] = compact(payload, event)
    if kind in {"sft.checkpoint.promoted", "sft.checkpoint.selected", "cispo.checkpoint.promoted"}:
        summary["checkpoint"] = compact(payload, event)
    if kind == "training.receipt":
        previous = db.execute(f"SELECT row_json FROM {TABLE} WHERE job_id=? AND collection='receipts' AND item_key=? ORDER BY sequence DESC LIMIT 1",
                              (event["job_id"], payload["request_id"])).fetchone()
        old = json.loads(previous[0]) if previous else None
        if old is None:
            summary["receipt_count"] += 1
        for receipt, sign in ((old, -1), (payload, 1)):
            if receipt is None:
                continue
            summary["missing_cost_count"] += sign * int(receipt.get("cost_missing", receipt.get("cost_usd") is None))
            for field in ("input_tokens", "output_tokens", "training_tokens"):
                summary["usage"][field] += sign * int(receipt.get(field) or 0)
            summary["usage"]["cost_usd"] += sign * float(receipt.get("cost_usd") or 0)


def materialize(db, job_id):
    latest = db.execute(f"SELECT sequence,snapshot_json FROM {SNAPSHOTS} WHERE job_id=? ORDER BY sequence DESC LIMIT 1", (job_id,)).fetchone()
    sequence = latest[0] if latest else 0
    summary = json.loads(latest[1]) if latest else initial_summary()
    while True:
        events = db.execute("SELECT * FROM training_events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 64", (job_id, sequence)).fetchall()
        if not events:
            return
        for raw in events:
            if raw["sequence"] != sequence + 1:
                raise ValueError("projection journal contains a gap")
            event = dict(raw)
            event["payload"] = json.loads(event.pop("payload_json"))
            apply_summary(db, summary, event)
            for ordinal, (collection, key, row) in enumerate(rows_for(event)):
                db.execute(f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?)", (job_id, collection, key, event["sequence"], ordinal, json.dumps(row)))
            sequence = event["sequence"]
            db.execute(f"INSERT INTO {SNAPSHOTS} VALUES (?,?,?)", (job_id, sequence, json.dumps(summary)))


def ensure(store, job_id):
    with store._lock:
        transaction = not store._db.in_transaction
        if transaction:
            store._db.execute("BEGIN IMMEDIATE")
        try:
            materialize(store._db, job_id)
            if transaction:
                store._db.commit()
        except BaseException:
            if transaction:
                store._db.rollback()
            raise


def summary_at(store, job_id, sequence):
    ensure(store, job_id)
    with store._lock:
        row = store._db.execute(f"SELECT snapshot_json FROM {SNAPSHOTS} WHERE job_id=? AND sequence<=? ORDER BY sequence DESC LIMIT 1", (job_id, sequence)).fetchone()
    return json.loads(row[0]) if row else initial_summary()


def collection_rows(store, job_id, collection, bound, key):
    ensure(store, job_id)
    db = store._db
    with store._lock:
        deduplicate = collection in {"artifacts", "receipts"}
        if not deduplicate:
            duplicate = db.execute(f"SELECT 1 FROM {TABLE} WHERE job_id=? AND collection=? AND sequence<=? GROUP BY item_key HAVING COUNT(*)>1 LIMIT 1", (job_id, collection, bound)).fetchone()
            if duplicate:
                raise ValueError("duplicate projection ordering key")
        after_sequence, after_ordinal = 0, -1
        if key is not None:
            cursor = db.execute(f"SELECT sequence,ordinal FROM {TABLE} WHERE job_id=? AND collection=? AND item_key=? AND sequence<=? ORDER BY sequence DESC LIMIT 1", (job_id, collection, key, bound)).fetchone()
            if cursor is None:
                raise ValueError("unknown or stale projection cursor")
            after_sequence, after_ordinal = cursor
        if deduplicate:
            rows = db.execute(f"""SELECT p.row_json,p.item_key FROM {TABLE} p
                WHERE job_id=? AND collection=? AND sequence<=? AND (? IS NULL OR item_key>?)
                AND sequence=(SELECT MAX(q.sequence) FROM {TABLE} q WHERE q.job_id=p.job_id AND q.collection=p.collection AND q.item_key=p.item_key AND q.sequence<=?)
                ORDER BY item_key LIMIT 101""", (job_id, collection, bound, key, key, bound)).fetchall()
        else:
            rows = db.execute(f"SELECT row_json,item_key FROM {TABLE} WHERE job_id=? AND collection=? AND sequence<=? AND (sequence>? OR (sequence=? AND ordinal>?)) ORDER BY sequence,ordinal LIMIT 101",
                              (job_id, collection, bound, after_sequence, after_sequence, after_ordinal)).fetchall()
    return [(json.loads(row[0]), row[1]) for row in rows]
