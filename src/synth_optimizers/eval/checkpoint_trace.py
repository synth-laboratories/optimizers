"""Retain checkpoint rollout sources in portable, explicitly partial Trace V5.

This promotes observed container output; it does not claim raw provider capture
or rerun/change the frozen grader. Original source digests remain authoritative.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile


def materialize_checkpoint_trace(output, *, job_id, trial_id, checkpoint, evaluator, reward):
    from synth_containers.tracing.adapters.optimizer_event_history import import_optimizer_event_history
    from synth_containers.tracing.adapters.native import write_imported_document
    from synth_containers.tracing.canonical import bytes_digest, canonical_bytes
    from synth_containers.tracing.capture.redaction import redact_payload, assert_no_secrets
    from synth_containers.tracing.models.identity import TraceIdentityV5
    from synth_containers.tracing.store.bundle import LocalTraceBundle
    from synth_containers.tracing.native_evaluation import attach_native_evaluation
    from synth_containers.tracing.inspection import inspect_trace_input

    output = Path(output)
    raw = (output / "trace.jsonl").read_bytes()
    binding = (output / "checkpoint_binding.json").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("checkpoint trace has no retained observations")
    for row in rows:
        if row.get("served_policy_snapshot_id") != checkpoint["checkpoint_id"]:
            raise ValueError("retained trace checkpoint identity mismatch")
    source = {"trace_jsonl": rows, "checkpoint_binding": json.loads(binding),
              "source_digests": {"trace.jsonl": bytes_digest(raw), "checkpoint_binding.json": bytes_digest(binding)},
              "checkpoint": checkpoint, "eval_job_id": job_id, "trial_id": trial_id,
              "evaluator": evaluator, "reward": reward}
    safe, redaction = redact_payload(source)
    assert_no_secrets(safe, where="checkpoint trace promotion")
    digest = bytes_digest(canonical_bytes(safe))
    archive = output / "checkpoint.trace-v5.zip"
    receipt_path = output / "checkpoint.trace-v5.receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_digest"] != digest or bytes_digest(archive.read_bytes()) != receipt["digest"]:
            raise ValueError("retained checkpoint trace source or archive changed")
        return receipt
    history = {"rollout_id": trial_id, "event_history": [{"event_type": "lm_call", "event_id": f"{trial_id}:{i}",
        "llm_request": {"messages": row["messages"], "model": checkpoint.get("model_id"),
                        "max_tokens": row.get("max_tokens"), "temperature": row.get("temperature")},
        "llm_response": {"message": {"role": "assistant", "content": row["completion"]}, "usage": row.get("usage")},
        "metadata": {"checkpoint_id": checkpoint["checkpoint_id"], "source_digest": digest}}
        for i, row in enumerate(safe["trace_jsonl"])]}
    document = import_optimizer_event_history(history)
    document = replace(document, identity=TraceIdentityV5(run_id=job_id, rollout_id=trial_id,
        trial_id=trial_id, episode_id=trial_id, task_id=str(rows[0].get("scenario", "checkpoint-evaluation")),
        seed=rows[0].get("seed")), provenance=replace(document.provenance,
        container_image_digest=evaluator["image_digest"],
        extra={**document.provenance.extra, "checkpoint": checkpoint, "eval_job_id": job_id,
               "source_digest": digest, "coverage": "partial imported container observations"}), content_digest="").sealed()
    with tempfile.TemporaryDirectory(prefix="checkpoint-trace-", dir=output) as temp:
        bundle = LocalTraceBundle(Path(temp) / "bundle", bundle_id=f"checkpoint-{trial_id}")
        stored = bundle.blobs.put(canonical_bytes(safe))
        imported = write_imported_document(document, source_digest=digest, source_format="checkpoint.container-output.v1",
            bundle=bundle, stored_source_digest=stored, source_redaction=redaction)
        attached = attach_native_evaluation(bundle.root, payload={"schema_version": evaluator["reward_version"],
            "authority": evaluator["id"], "trace_id": imported["trace_id"], "task_id": rows[0].get("scenario"),
            "status": "completed", "reward": {"name": evaluator["metric_ref"], "value": reward,
            "version": evaluator["reward_version"], "units": evaluator["units"]}}, source_name="checkpoint-reward.json")
        if not attached["validation_valid"]:
            raise ValueError("checkpoint trace reward failed validation")
        archive.write_bytes(bundle.archive_bytes())
    inspection = inspect_trace_input(archive)
    if not inspection.validation.valid or not inspection.self_contained:
        raise ValueError("checkpoint trace bundle failed validation")
    receipt = {"role": "trace_v5_partial", "path": str(archive), "digest": bytes_digest(archive.read_bytes()),
               "source_digest": digest, "trace_id": imported["trace_id"], "trace_digest": imported["trace_digest"],
               "eval_job_id": job_id, "trial_id": trial_id, "checkpoint_id": checkpoint["checkpoint_id"],
               "bytes": archive.stat().st_size, "capture_status": "partial"}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt
