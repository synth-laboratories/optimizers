"""PRBench task/rubric loading and scoring.

Real data. Source: https://huggingface.co/datasets/ScaleAI/PRBench (CC-BY-4.0),
paper arXiv:2511.11562 "PRBench: Large-Scale Expert Rubrics for Evaluating
High-Stakes Professional Reasoning". The parquet files under ``data/`` are the
upstream release artifacts, unmodified.

Nothing in this module invents rubric content. Every criterion title, weight and
category is read straight out of the released parquet.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("PRBENCH_DATA_DIR") or (Path(__file__).parent / "data"))

# Which released subsets to serve. Upstream ships finance, finance_hard, legal,
# legal_hard; this container vendors the two Hard subsets by default.
SUBSETS = [
    s.strip()
    for s in (os.environ.get("PRBENCH_SUBSETS") or "finance_hard,legal_hard").split(",")
    if s.strip()
]

N_TRAIN = int(os.environ.get("PRBENCH_TRAIN_ROWS", "24"))
N_HELDOUT = int(os.environ.get("PRBENCH_HELDOUT_ROWS", "16"))
# Long multi-turn conversations blow up prompt cost; default to <=2-turn tasks.
MAX_TURNS = int(os.environ.get("PRBENCH_MAX_TURNS", "2"))
MAX_CONTEXT_CHARS = int(os.environ.get("PRBENCH_MAX_CONTEXT_CHARS", "24000"))
MAX_REFERENCE_CHARS = int(os.environ.get("PRBENCH_MAX_REFERENCE_CHARS", "8000"))

# Weight fields, keyed by the authoritative `weight_class` annotation. Some rows
# carry stale weights in sibling fields (annotation history); weight_class is the
# only field that resolves to the criterion's real weight, and it is non-null and
# resolvable for all 9,543 criteria across the two Hard subsets.
_WEIGHT_FIELD = {
    "critically important": "critically_important_weight",
    "important": "important_weight",
    "slightly important": "slightly_important_weight",
    "slightly detrimental": "slightly_detrimental_weight",
    "detrimental": "detrimental_weight",
    "critically detrimental": "critically_detrimental_weight",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional assistant answering questions from practitioners in "
    "finance and law. Answer the user's question directly and completely."
)


def _criterion_weight(annotations: dict[str, Any]) -> int:
    weight_class = str(annotations.get("weight_class") or "").strip().lower()
    field = _WEIGHT_FIELD.get(weight_class)
    if field is None:
        return 0
    raw = annotations.get(field)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n[...truncated {len(text) - limit} chars...]"


def _build_record(subset: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    turns = int(raw.get("turns") or 0)
    if turns < 1:
        return None
    final_prompt = str(raw.get(f"prompt_{turns - 1}") or "").strip()
    if not final_prompt:
        return None

    criteria: list[dict[str, Any]] = []
    for item in raw.get("rubric") or []:
        annotations = item.get("annotations") or {}
        weight = _criterion_weight(annotations)
        if weight == 0:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        criteria.append(
            {
                "criterion_id": str(item.get("id") or ""),
                "title": title,
                "description": str(annotations.get("criteria_description") or "").strip(),
                "category": str(annotations.get("criteria_category") or "").strip(),
                "weight_class": str(annotations.get("weight_class") or "").strip(),
                "weight": weight,
            }
        )
    if not criteria or not any(c["weight"] > 0 for c in criteria):
        return None

    context: list[dict[str, str]] = []
    used = 0
    for i in range(turns - 1):
        for role, key in (("user", f"prompt_{i}"), ("assistant", f"response_{i}")):
            text = str(raw.get(key) or "").strip()
            if not text:
                continue
            remaining = MAX_CONTEXT_CHARS - used
            if remaining <= 0:
                break
            clipped = _clip(text, remaining)
            used += len(clipped)
            context.append({"role": role, "content": clipped})

    references = [
        str(t or "").strip()
        for t in (raw.get(f"reference_texts_{turns - 1}") or [])
        if str(t or "").strip()
    ]
    reference_block = _clip("\n\n".join(references), MAX_REFERENCE_CHARS) if references else ""

    return {
        "prbench_task": str(raw.get("task") or ""),
        "subset": subset,
        "field": str(raw.get("field") or ""),
        "topic": str(raw.get("topic") or ""),
        "turns": turns,
        "prompt": final_prompt,
        "context": context,
        "reference_text": reference_block,
        "criteria": criteria,
        "max_weight": sum(c["weight"] for c in criteria if c["weight"] > 0),
        "canary": str(raw.get("canary") or ""),
    }


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Load the released parquet subsets and cut a deterministic train/heldout split."""
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    loaded: list[str] = []
    missing: list[str] = []
    for subset in SUBSETS:
        path = DATA_DIR / f"{subset}.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        loaded.append(subset)
        for raw in pq.read_table(path).to_pylist():
            record = _build_record(subset, raw)
            if record is not None and record["turns"] <= MAX_TURNS:
                records.append(record)

    # Deterministic, subset-stratified ordering: hash the upstream task id so the
    # split is stable across processes and independent of parquet row order.
    def _key(rec: dict[str, Any]) -> str:
        return hashlib.sha256(f"{rec['subset']}/{rec['prbench_task']}".encode()).hexdigest()

    by_subset: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_subset.setdefault(rec["subset"], []).append(rec)
    for bucket in by_subset.values():
        bucket.sort(key=_key)

    ordered: list[dict[str, Any]] = []
    idx = 0
    while any(idx < len(b) for b in by_subset.values()):
        for subset in loaded:
            bucket = by_subset.get(subset) or []
            if idx < len(bucket):
                ordered.append(bucket[idx])
        idx += 1

    train = ordered[:N_TRAIN]
    heldout = ordered[N_TRAIN : N_TRAIN + N_HELDOUT]

    registry: dict[str, dict[str, Any]] = {}
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    for split, bucket in (("train", train), ("heldout", heldout)):
        for i, rec in enumerate(bucket):
            row_id = f"{split}:{i}"
            full = {**rec, "task_id": row_id, "example_id": row_id, "split": split, "index": i}
            splits[split].append(full)
            registry[row_id] = full
            registry[rec["prbench_task"]] = full

    return {
        "registry": registry,
        "splits": splits,
        "loaded_subsets": loaded,
        "missing_files": missing,
        "pool_size": len(records),
        "total_criteria": sum(len(r["criteria"]) for r in train + heldout),
    }


def load_error() -> str | None:
    data = _load()
    if not data["loaded_subsets"]:
        return f"no PRBench parquet found; expected {data['missing_files']}"
    if not data["splits"]["train"]:
        return "PRBench parquet loaded but train split is empty"
    return None


def dataset_stats() -> dict[str, Any]:
    data = _load()
    return {
        "loaded_subsets": data["loaded_subsets"],
        "missing_files": data["missing_files"],
        "candidate_pool_size": data["pool_size"],
        "train_rows": len(data["splits"]["train"]),
        "heldout_rows": len(data["splits"]["heldout"]),
        "criteria_in_served_split": data["total_criteria"],
        "max_turns": MAX_TURNS,
    }


def public_row(rec: dict[str, Any]) -> dict[str, Any]:
    """Row shape handed to GEPA.

    Deliberately omits rubric criterion text: the rubric is the answer key, and
    leaking it into the optimizer's view would let a candidate prompt be tuned to
    recite criteria rather than answer well. Only the count and category mix are
    exposed. Grading resolves the full rubric server-side by ``example_id``.
    """
    categories = sorted({c["category"] for c in rec["criteria"] if c["category"]})
    return {
        "task_id": rec["task_id"],
        "example_id": rec["example_id"],
        "split": rec["split"],
        "prbench_task": rec["prbench_task"],
        "subset": rec["subset"],
        "field": rec["field"],
        "topic": rec["topic"],
        "turns": rec["turns"],
        "prompt": rec["prompt"],
        "context": rec["context"],
        "reference_text": rec["reference_text"],
        "n_criteria": len(rec["criteria"]),
        "max_weight": rec["max_weight"],
        "rubric_categories": categories,
    }


def rows_for(split: str) -> list[dict[str, Any]]:
    normalized = "heldout" if split in {"heldout", "test", "validation", "val"} else "train"
    return [public_row(r) for r in _load()["splits"][normalized]]


def lookup(*candidates: Any) -> dict[str, Any] | None:
    registry = _load()["registry"]
    for candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate).strip()
        if key and key in registry:
            return registry[key]
    return None


def score_from_judgements(rec: dict[str, Any], met: dict[str, bool]) -> dict[str, Any]:
    """PRBench Appendix D.1 score.

        s = (sum of weights of satisfied criteria, signed) / (sum of positive weights)

    Clipped into [0, 1]: negative weights penalize but the paper clips the score
    at zero, and the denominator caps it at one.
    """
    numerator = 0
    denominator = 0
    satisfied = 0
    penalties = 0
    per_criterion: list[dict[str, Any]] = []
    for c in rec["criteria"]:
        weight = c["weight"]
        if weight > 0:
            denominator += weight
        hit = bool(met.get(c["criterion_id"], False))
        if hit:
            numerator += weight
            if weight > 0:
                satisfied += 1
            else:
                penalties += 1
        per_criterion.append(
            {
                "criterion_id": c["criterion_id"],
                "category": c["category"],
                "weight": weight,
                "met": hit,
            }
        )
    raw = (numerator / denominator) if denominator else 0.0
    reward = max(0.0, min(1.0, raw))
    positive_total = sum(1 for c in rec["criteria"] if c["weight"] > 0)
    return {
        "reward": reward,
        "raw_score": raw,
        "weighted_numerator": numerator,
        "weighted_denominator": denominator,
        "criteria_total": len(rec["criteria"]),
        "positive_criteria": positive_total,
        "positive_criteria_met": satisfied,
        "unweighted_positive_fraction": (satisfied / positive_total) if positive_total else 0.0,
        "detrimental_criteria_triggered": penalties,
        "per_criterion": per_criterion,
    }
