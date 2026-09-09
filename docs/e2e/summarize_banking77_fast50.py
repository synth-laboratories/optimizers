"""Verify and summarize the completed frozen fast50 experiment receipts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime

from freeze_banking77_panel import _atomic_json
from run_banking77_fast50 import ROOT, checkpoints


def read(relative):
    return json.loads((ROOT / relative).read_text())


def usage_sum(usages):
    usages = list(usages)
    return {key: sum(u.get(key, 0) for u in usages)
            for key in ("calls", "prompt_tokens", "completion_tokens")}


def estimate(usage, training_tokens=0):
    fixed = (usage["completion_tokens"] * .45 + training_tokens * .396) / 1e6
    return {"all_prefill_cached_usd": fixed + usage["prompt_tokens"] * .036 / 1e6,
            "no_prefill_cached_usd": fixed + usage["prompt_tokens"] * .18 / 1e6}


def main():
    assert read("status.json")["state"] == "completed"
    final = read("final_results.json")
    validation = read("validation_results.json")["results"]
    best = max(validation, key=lambda r: (r["trained_mean"], -r["revision"]))
    assert best == read("selection.json")
    assert final["selected_revision"] == best["revision"]
    rows = checkpoints()
    revisions = {int(r["policy_revision_id"].split("@")[-1]) for r in rows}
    assert set(range(25, 75)).issubset(revisions) and max(revisions) == 74
    provider = read("training_resume25/provider_usage.json")
    assert provider["totals"]["train_calls"] == 49
    assert all(c["metrics"]["loss_weight_nonzero"] == 32 and
               c["metrics"]["loss:sum"] != 0 for c in provider["train_calls"])
    assert read("training_resume25/manifest.json")["stop_reason"] == "target_train_updates_reached"
    groups = [json.loads(line) for line in (ROOT / "training_resume25/groups.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (ROOT / "training_resume25/traces.jsonl").read_text().splitlines()]
    screens = [read(f"screen_{i}/manifest.json") for i in range(4)]
    assert sum(s["attempt_count"] for s in screens) == 12320
    tasks = [t for i in range(4) for t in read(f"screen_{i}/summary.json")["tasks"]]
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 1540
    assert all(t["samples"] == 8 and t["selected"] == (1 <= t["successes"] <= 7) for t in tasks)
    assert {t["task_id"] for t in tasks if t["selected"]} == set(read("curriculum.json")["selected_train_ids"])
    validation_panel, final_panel = read("validation_panel.json"), read("final_panel.json")
    assert not set(validation_panel["task_ids"]) & set(final_panel["task_ids"])
    assert not set(final_panel["task_ids"]) & set(final_panel["excluded_task_ids"])
    elapsed = max(s["finished_at_unix"] for s in screens) - min(s["started_at_unix"] for s in screens)
    names = [f"b77_fast50_19_val_{r}" for r in (34, 44, 54, 64, 74)]
    names += ["b77_fast50_19_final_original", "b77_fast50_19_final_incremental"]
    receipts = [read(f"{n}/{n}.evaluation.json") for n in names]
    for name, receipt in zip(names, receipts):
        result = read(f"{name}/result.json")
        assert result["valid"] and result["pairs"] in (154, 770)
        digest = "sha256:" + hashlib.sha256((ROOT / name / f"{name}.evaluation.json").read_bytes()).hexdigest()
        assert result["receipt_sha256"] == digest
        assert receipt["attempt_count"] == result["pairs"] * 2
    usages = {
        "completed_screening": usage_sum(s["usage_totals"] for s in screens),
        "interrupted_screening_observed": usage_sum(read(f"screen_{i}_interrupted/progress.json")["usage_totals"] for i in range(4)),
        "resumed_training_sampling": usage_sum(t["usage"] for t in traces),
        "validation_and_final": usage_sum(r["usage_totals"] for r in receipts),
    }
    total = usage_sum(usages.values())
    first = next(r for r in rows if r["checkpoint_id"] == "ckpt_0278ebdd569252e2f583b9a0")
    assert first["training_evidence"]["examples"] == 32 and len(first["train_call_ids"]) == 1
    training_tokens = provider["totals"]["training_tokens"] + first["training_evidence"]["tokens"]
    evaluation_elapsed = (max(datetime.fromisoformat(r["finished_at"]) for r in receipts) -
                          min(datetime.fromisoformat(r["started_at"]) for r in receipts)).total_seconds()
    summary = {
        "schema_version": "banking77.fast50.summary.v1", "root": str(ROOT),
        "additional_updates": 50, "final_revision": 74,
        "training_examples": 1600, "training_tokens": training_tokens,
        "training_evidence_note": "First update preserved in revision-25 checkpoint; its final metrics receipt was lost on dispatch failure. Remaining 49 calls have nonzero loss and 32 nonzero weights each.",
        "screening": {"attempts": 12320, "seconds": elapsed, "samples_per_minute": 12320 * 60 / elapsed,
                      "selected_tasks": len(read("curriculum.json")["selected_train_ids"]),
                      "intent_count": len(read("curriculum.json")["intent_counts"])},
        "resumed_training": {"sampling_tps": {k: v for k, v in read("training_resume25/sampling_tps.json").items() if k != "by_call"},
                             "group_dispositions": dict(Counter(g["disposition"] for g in groups)),
                             "host_sleep_wall_seconds": 2378},
        "evaluation": {"elapsed_seconds": evaluation_elapsed,
                       "attempts": sum(r["attempt_count"] for r in receipts),
                       "samples_per_minute": sum(r["attempt_count"] for r in receipts) * 60 / evaluation_elapsed},
        "validation": validation, "final": final,
        "selected_checkpoint": next(r for r in rows if r["checkpoint_id"] == best["trained_checkpoint_id"]),
        "counted_usage": usages, "counted_usage_total": total,
        "counted_token_cost_estimate": estimate(total, training_tokens),
        "cost_note": "Not an invoice or complete spend: excludes first interrupted training sampling, unrecorded in-flight/failed calls, and checkpoint storage. Cache hits and provider dollars unavailable; partial screening progress is included only as observed usage. Historical checkpoint training_evidence.provider_cost=0 is a missing-cost placeholder, not evidence of free compute; provider_usage marks missing dollars explicitly.",
        "rates_usd_per_million": {"prefill": .18, "cached_prefill": .036, "sample": .45, "train": .396},
        "rate_source": "https://tinker-docs.thinkingmachines.ai/tinker/models.json",
        "artifact_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ("experiment.json", "candidates.json", "validation_panel.json", "final_panel.json", "curriculum.json", "recovery.json", "training_resume25/provider_usage.json", "training_resume25/manifest.json")},
    }
    _atomic_json(ROOT / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("additional_updates", "screening", "evaluation", "counted_token_cost_estimate", "final")}, indent=2))


if __name__ == "__main__":
    main()
