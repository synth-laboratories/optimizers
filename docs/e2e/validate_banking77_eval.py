"""Validate a Banking77 paired receipt and emit paired statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_CHANNEL = "score::team-0"
VALID_TERMINAL = frozenset({"completed", "scored"})


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _task_ids(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "task_id" and isinstance(value, str):
                found.add(value)
            elif key in {"task_ids", "evaluation_ids", "train_ids", "selected_train_ids"} and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                found.update(str(item) for item in value)
            found.update(_task_ids(value))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            found.update(_task_ids(value))
    return found


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_statistics(deltas: Sequence[float], *, bootstrap_seed: int = 0, replicates: int = 20_000) -> dict[str, Any]:
    if not deltas or replicates < 1:
        raise ValueError("paired statistics require deltas and positive replicates")
    wins = sum(value > 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(0, min(wins, losses) + 1)) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    rng = random.Random(bootstrap_seed)
    boot = [statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(replicates)]
    return {
        "pairs": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "paired_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - discordant,
        "discordant_pairs": discordant,
        "exact_two_sided_mcnemar_p": mcnemar_p,
        "paired_bootstrap_95_percentile_interval": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": replicates,
    }


def validate(
    panel: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_baseline: str,
    expected_trained: str,
    train_ids: set[str],
    prior_ids: set[str],
    receipt_sha256: str,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    errors: list[str] = []
    rows = list(panel.get("rows") or ())
    panel_ids = [str(row.get("task_id")) for row in rows]
    labels = [str(row.get("label")) for row in rows]
    seeds = [(str(row.get("task_id")), int(row.get("seed"))) for row in rows]
    if len(rows) != 77 or len(set(panel_ids)) != 77 or len(set(labels)) != 77:
        errors.append("panel must contain exactly 77 unique task ids and 77 unique labels")
    panel_core = [
        {"label": row.get("label"), "task_id": row.get("task_id"), "seed": row.get("seed")}
        for row in rows
    ]
    if panel.get("panel_digest") != _digest_json(panel_core):
        errors.append("panel digest does not match its rows")
    excluded_declared = sorted(str(item) for item in panel.get("excluded_task_ids", ()))
    if panel.get("exclusion_set_digest") != _digest_json(excluded_declared):
        errors.append("exclusion-set digest does not match declared exclusions")
    if set(panel_ids) & set(excluded_declared):
        errors.append("panel contains an id in its own declared exclusion set")
    if any(not task_id.startswith("banking77/heldout/") for task_id in panel_ids):
        errors.append("every panel task must belong to banking77/heldout")
    overlap_train = sorted(set(panel_ids) & train_ids)
    overlap_prior = sorted(set(panel_ids) & prior_ids)
    if overlap_train:
        errors.append(f"panel overlaps training ids: {overlap_train}")
    if overlap_prior:
        errors.append(f"panel overlaps prior panels: {overlap_prior}")
    if receipt.get("split") != "heldout":
        errors.append("receipt split is not heldout")
    receipt_seeds = [(str(row.get("task_id")), int(row.get("seed"))) for row in receipt.get("seeds", ())]
    if receipt_seeds != seeds:
        errors.append("receipt seed order does not exactly match frozen panel order")

    arms = receipt.get("arms") or {}
    attempts_by_arm: dict[str, list[Mapping[str, Any]]] = {}
    identities: dict[str, dict[str, set[str]]] = {}
    expected = {"baseline": expected_baseline, "trained": expected_trained}
    for arm in ("baseline", "trained"):
        arm_payload = arms.get(arm) or {}
        attempts = list(arm_payload.get("attempts") or ())
        attempts_by_arm[arm] = attempts
        if arm_payload.get("resolved_id") != expected[arm]:
            errors.append(f"{arm} resolved_id is not {expected[arm]}")
        catalogued = list(arm_payload.get("catalogued_sampler_references") or ())
        loaded = list(arm_payload.get("loaded_sampler_references") or ())
        if not catalogued or catalogued != loaded:
            errors.append(f"{arm} catalogued and loaded sampler references differ or are empty")
        if len(attempts) != 77 or int(arm_payload.get("attempt_count", -1)) != 77:
            errors.append(f"{arm} must contain exactly 77 attempts")
        order = [(str(row.get("task_id")), int(row.get("seed"))) for row in attempts]
        if order != seeds:
            errors.append(f"{arm} attempt order does not match frozen panel")
        for index, row in enumerate(attempts):
            if row.get("arm") != arm or int(row.get("sample_index", -1)) != index:
                errors.append(f"{arm} attempt {index} has wrong arm or sample_index")
            if row.get("terminal_status") not in VALID_TERMINAL:
                errors.append(f"{arm} attempt {index} is not successfully terminal")
            if row.get("reward_channel") != EXPECTED_CHANNEL:
                errors.append(f"{arm} attempt {index} uses wrong reward channel")
            if list(row.get("checkpoint_ids") or ()) != [expected[arm]]:
                errors.append(f"{arm} attempt {index} binds the wrong checkpoint")
            if list(row.get("sampler_references") or ()) != loaded:
                errors.append(f"{arm} attempt {index} binds the wrong sampler reference")
            if not str(row.get("trace_digest") or "").startswith("sha256:"):
                errors.append(f"{arm} attempt {index} has no trace digest")
            reward = row.get("reward")
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                errors.append(f"{arm} attempt {index} has a non-finite reward")
        identities[arm] = {
            "rollout": {str(row.get("rollout_id") or "") for row in attempts},
            "proxy": {str(row.get("proxy_request_id") or "") for row in attempts},
            "ref": set(loaded),
        }
        for kind in ("rollout", "proxy"):
            if "" in identities[arm][kind] or len(identities[arm][kind]) != len(attempts):
                errors.append(f"{arm} {kind} identities are missing or duplicated")
    for kind in ("rollout", "proxy", "ref"):
        overlap = identities.get("baseline", {}).get(kind, set()) & identities.get("trained", {}).get(kind, set())
        if overlap:
            errors.append(f"baseline/trained {kind} identities overlap: {sorted(overlap)}")

    summary_rows = list((receipt.get("paired_summary") or {}).get("rows") or ())
    if len(summary_rows) != 77:
        errors.append("paired summary must contain exactly 77 rows")
    elif len(attempts_by_arm.get("baseline", ())) == len(attempts_by_arm.get("trained", ())) == 77:
        for index, (summary, baseline, trained) in enumerate(
            zip(
                summary_rows,
                attempts_by_arm["baseline"],
                attempts_by_arm["trained"],
                strict=True,
            )
        ):
            expected_summary = (
                baseline.get("task_id"),
                int(baseline.get("seed")),
                float(baseline.get("reward")),
                float(trained.get("reward")),
            )
            observed_summary = (
                summary.get("task_id"),
                int(summary.get("seed")),
                float(summary.get("baseline_reward")),
                float(summary.get("trained_reward")),
            )
            if observed_summary != expected_summary or float(summary.get("delta")) != expected_summary[3] - expected_summary[2]:
                errors.append(f"paired summary row {index} disagrees with arm attempts")
    if errors:
        raise ValueError("invalid Banking77 evaluation:\n- " + "\n- ".join(errors))

    baseline_rewards = [float(row["reward"]) for row in attempts_by_arm["baseline"]]
    trained_rewards = [float(row["reward"]) for row in attempts_by_arm["trained"]]
    deltas = [trained - baseline for baseline, trained in zip(baseline_rewards, trained_rewards, strict=True)]
    stats = paired_statistics(deltas, bootstrap_seed=bootstrap_seed, replicates=bootstrap_replicates)
    return {
        "valid": True,
        "estimand": panel.get("estimand", "macro intent accuracy"),
        "panel_digest": panel.get("panel_digest"),
        "receipt_sha256": receipt_sha256,
        "baseline_checkpoint_id": expected_baseline,
        "trained_checkpoint_id": expected_trained,
        "baseline_mean": statistics.fmean(baseline_rewards),
        "trained_mean": statistics.fmean(trained_rewards),
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--train-config", action="append", default=[])
    parser.add_argument("--prior-panel", action="append", default=[])
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    panel_path, receipt_path = Path(args.panel), Path(args.receipt)
    panel, receipt = json.loads(panel_path.read_text()), json.loads(receipt_path.read_text())
    train_ids: set[str] = set()
    for name in args.train_config:
        config = tomllib.loads(Path(name).read_text(encoding="utf-8"))
        train_ids.update(str(item) for item in config.get("taskset", {}).get("train_ids", ()))
    prior_ids: set[str] = set()
    for name in args.prior_panel:
        prior_ids.update(_task_ids(json.loads(Path(name).read_text(encoding="utf-8"))))
    result = validate(
        panel,
        receipt,
        expected_baseline=args.baseline,
        expected_trained=args.trained,
        train_ids=train_ids,
        prior_ids=prior_ids,
        receipt_sha256=_sha(receipt_path),
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
