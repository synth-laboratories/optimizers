"""The paired reducer: a pure function of plan plus outcome rows.

Nothing here reads a run directory, calls an executor, or knows what an arm was
testing.  It takes what was planned and what was sealed, and it reports the
comparison the design declared — including, and especially, the reasons the
comparison does not support a headline.

Three rules do most of the work:

* a failed trial is missing evidence, never a zero;
* missingness is only harmless when it is *symmetric*, so it is measured per arm
  and the difference is what gates a claim;
* the order trials actually ran in is evidence, and nominal counterbalancing is
  not, so fairness is computed from recorded dispatch and start times.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import (
    EXPERIMENT_REPORT_SCHEMA,
    TrialOutcome,
    canonical_json,
)
from .outcomes import OutcomeSet, reduce_replicates
from .plan import ExperimentPlan

#: Above this many paired blocks, enumerating every sign flip stops being cheap
#: and the permutation test switches to a seeded Monte Carlo approximation.
EXACT_PERMUTATION_LIMIT = 14


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seeded_rng(plan: ExperimentPlan, metric_id: str) -> random.Random:
    """One RNG per (plan, metric), so a report is reproducible byte for byte."""

    material = canonical_json([plan.plan_digest, metric_id])
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


@dataclass(frozen=True, slots=True)
class ArmAggregate:
    arm_id: str
    label: str
    planned_trials: int
    completed_trials: int
    failed_trials: int
    missing_trials: int
    blocks_expected: int
    blocks_completed: int
    missing_blocks: tuple[str, ...]
    mean: float | None
    stdev: float | None
    cost_usd: float | None
    wall_seconds: float | None
    failure_classes: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "planned_trials": self.planned_trials,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "missing_trials": self.missing_trials,
            "blocks_expected": self.blocks_expected,
            "blocks_completed": self.blocks_completed,
            "missing_blocks": list(self.missing_blocks),
            "mean": self.mean,
            "stdev": self.stdev,
            "cost_usd": self.cost_usd,
            "wall_seconds": self.wall_seconds,
            "failure_classes": self.failure_classes,
        }


@dataclass(frozen=True, slots=True)
class PairedComparison:
    baseline_arm_id: str
    treatment_arm_id: str
    metric_id: str
    direction: str
    blocks_paired: int
    deltas: tuple[tuple[str, float], ...]
    mean_delta: float | None
    signed_mean_delta: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    p_value: float | None
    p_method: str
    wins: int
    losses: int
    ties: int
    unpaired_blocks: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline_arm_id": self.baseline_arm_id,
            "treatment_arm_id": self.treatment_arm_id,
            "metric_id": self.metric_id,
            "direction": self.direction,
            "blocks_paired": self.blocks_paired,
            "deltas": [{"block_id": block, "delta": delta} for block, delta in self.deltas],
            "mean_delta": self.mean_delta,
            "signed_mean_delta": self.signed_mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "p_method": self.p_method,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "unpaired_blocks": list(self.unpaired_blocks),
        }


@dataclass(frozen=True, slots=True)
class FairnessFacts:
    """What actually happened, as opposed to what the design intended."""

    counterbalanced: bool
    mean_dispatch_index: dict[str, float]
    mean_start_rank: dict[str, float]
    position_bias: float | None
    median_queue_seconds: dict[str, float]
    image_references: dict[str, list[str]]
    distinct_cache_namespaces: int
    trials_observed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "counterbalanced": self.counterbalanced,
            "mean_dispatch_index": self.mean_dispatch_index,
            "mean_start_rank": self.mean_start_rank,
            "position_bias": self.position_bias,
            "median_queue_seconds": self.median_queue_seconds,
            "image_references": self.image_references,
            "distinct_cache_namespaces": self.distinct_cache_namespaces,
            "trials_observed": self.trials_observed,
        }


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    allowed: bool
    blockers: tuple[str, ...]
    notes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "headline_claim_allowed": self.allowed,
            "blockers": list(self.blockers),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    experiment_id: str
    plan_digest: str
    executor: str
    primary_metric: str
    direction: str
    mode: str
    arms: tuple[ArmAggregate, ...]
    comparisons: tuple[PairedComparison, ...]
    secondary: dict[str, tuple[PairedComparison, ...]]
    fairness: FairnessFacts
    claim: ClaimVerdict
    totals: dict[str, Any]
    conflicts: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_REPORT_SCHEMA,
            "experiment_id": self.experiment_id,
            "plan_digest": self.plan_digest,
            "executor": self.executor,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "mode": self.mode,
            "arms": [arm.to_json() for arm in self.arms],
            "comparisons": [item.to_json() for item in self.comparisons],
            "secondary": {
                metric: [item.to_json() for item in items]
                for metric, items in self.secondary.items()
            },
            "fairness": self.fairness.to_json(),
            "totals": self.totals,
            "conflicts": [dict(item) for item in self.conflicts],
            **self.claim.to_json(),
        }


# --------------------------------------------------------------------- reduce


def reduce_experiment(
    plan: ExperimentPlan, outcomes: OutcomeSet, *, mode: str = "experiment"
) -> ExperimentReport:
    """Turn a plan and its sealed rows into the report the design declared.

    `mode` is `experiment` or `aa`.  An A/A run is an identity and isolation
    smoke test: it can prove the harness does not manufacture a difference, and
    it cannot establish a noise ceiling from three blocks, so it never yields a
    headline regardless of what it measures.
    """

    design = plan.design
    metric_id = design["primary_metric"]
    direction = plan.primary_metric_direction
    rows = list(outcomes.rows)
    by_trial = {row.trial_id: row for row in rows}

    arms = tuple(
        _aggregate_arm(plan, arm_id=arm.arm_id, label=arm.label, rows=rows, metric_id=metric_id)
        for arm in plan.arms
    )
    baseline = plan.arms[0]
    comparisons = tuple(
        _compare(plan, rows, baseline.arm_id, arm.arm_id, metric_id, direction)
        for arm in plan.arms[1:]
    )
    directions = plan.secondary_metric_directions
    secondary = {
        secondary_metric: tuple(
            _compare(
                plan,
                rows,
                baseline.arm_id,
                arm.arm_id,
                secondary_metric,
                directions.get(secondary_metric, "maximize"),
            )
            for arm in plan.arms[1:]
        )
        for secondary_metric in design.get("secondary_metrics", [])
    }
    fairness = _fairness(plan, rows)
    totals = _totals(plan, rows, by_trial, retried=outcomes.retried_trial_ids)
    claim = _claim(
        plan,
        mode=mode,
        arms=arms,
        comparisons=comparisons,
        fairness=fairness,
        totals=totals,
        conflicts=outcomes.conflicts,
        retried=outcomes.retried_trial_ids,
    )
    return ExperimentReport(
        experiment_id=plan.experiment_id,
        plan_digest=plan.plan_digest,
        executor=plan.executor,
        primary_metric=metric_id,
        direction=direction,
        mode=mode,
        arms=arms,
        comparisons=comparisons,
        secondary=secondary,
        fairness=fairness,
        claim=claim,
        totals=totals,
        conflicts=tuple(item.to_json() for item in outcomes.conflicts),
    )


def _aggregate_arm(
    plan: ExperimentPlan,
    *,
    arm_id: str,
    label: str,
    rows: Sequence[TrialOutcome],
    metric_id: str,
) -> ArmAggregate:
    planned = [trial for trial in plan.trials if trial.arm_id == arm_id]
    mine = [row for row in rows if row.arm_id == arm_id]
    completed = [row for row in mine if row.counted]
    values = [row.metrics[metric_id] for row in completed if metric_id in row.metrics]
    per_block = reduce_replicates(mine, metric_id=metric_id)
    blocks_done = {block for (arm, block) in per_block if arm == arm_id}
    expected_blocks = set(plan.block_ids)
    failure_classes: dict[str, int] = {}
    for row in mine:
        if row.failure_class:
            failure_classes[row.failure_class] = failure_classes.get(row.failure_class, 0) + 1
    costs = [
        float(row.usage["cost_usd"])
        for row in mine
        if isinstance(row.usage.get("cost_usd"), (int, float))
        and not isinstance(row.usage.get("cost_usd"), bool)
    ]
    walls = []
    for row in mine:
        start, finish = _parse_time(row.started_at), _parse_time(row.finished_at)
        if start and finish:
            walls.append((finish - start).total_seconds())
    return ArmAggregate(
        arm_id=arm_id,
        label=label,
        planned_trials=len(planned),
        completed_trials=len(completed),
        failed_trials=len(mine) - len(completed),
        missing_trials=len(planned) - len(mine),
        blocks_expected=len(expected_blocks),
        blocks_completed=len(blocks_done),
        missing_blocks=tuple(sorted(expected_blocks - blocks_done)),
        mean=(sum(values) / len(values)) if values else None,
        stdev=_stdev(values),
        cost_usd=sum(costs) if costs else None,
        wall_seconds=sum(walls) if walls else None,
        failure_classes=failure_classes,
    )


def _stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _compare(
    plan: ExperimentPlan,
    rows: Sequence[TrialOutcome],
    baseline_arm_id: str,
    treatment_arm_id: str,
    metric_id: str,
    direction: str,
) -> PairedComparison:
    per_block = reduce_replicates(rows, metric_id=metric_id)
    baseline = {block: value for (arm, block), value in per_block.items() if arm == baseline_arm_id}
    treatment = {
        block: value for (arm, block), value in per_block.items() if arm == treatment_arm_id
    }
    shared = [block for block in plan.block_ids if block in baseline and block in treatment]
    unpaired = tuple(
        block for block in plan.block_ids if block not in baseline or block not in treatment
    )
    deltas = tuple((block, treatment[block] - baseline[block]) for block in shared)
    confidence = float(plan.design["confidence"])
    if not deltas:
        return PairedComparison(
            baseline_arm_id=baseline_arm_id,
            treatment_arm_id=treatment_arm_id,
            metric_id=metric_id,
            direction=direction,
            blocks_paired=0,
            deltas=(),
            mean_delta=None,
            signed_mean_delta=None,
            ci_low=None,
            ci_high=None,
            confidence=confidence,
            p_value=None,
            p_method="none",
            wins=0,
            losses=0,
            ties=0,
            unpaired_blocks=unpaired,
        )
    values = [delta for _, delta in deltas]
    sign = 1.0 if direction == "maximize" else -1.0
    mean_delta = sum(values) / len(values)
    rng = _seeded_rng(plan, metric_id)
    ci_low, ci_high = _paired_bootstrap_ci(
        values, confidence=confidence, resamples=int(plan.design["bootstrap_resamples"]), rng=rng
    )
    p_value, method = _permutation_p(
        values, rng=rng, resamples=int(plan.design["bootstrap_resamples"])
    )
    return PairedComparison(
        baseline_arm_id=baseline_arm_id,
        treatment_arm_id=treatment_arm_id,
        metric_id=metric_id,
        direction=direction,
        blocks_paired=len(values),
        deltas=deltas,
        mean_delta=mean_delta,
        signed_mean_delta=sign * mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=p_value,
        p_method=method,
        wins=sum(1 for value in values if sign * value > 0),
        losses=sum(1 for value in values if sign * value < 0),
        ties=sum(1 for value in values if value == 0),
        unpaired_blocks=unpaired,
    )


def _paired_bootstrap_ci(
    values: Sequence[float], *, confidence: float, resamples: int, rng: random.Random
) -> tuple[float | None, float | None]:
    """Percentile CI over the paired differences.

    Resampling the *differences* rather than the two arms separately is what
    keeps the pairing: a block contributes as one observation, so the block-level
    variance that common seeds were meant to remove stays removed.
    """

    if len(values) < 2:
        return None, None
    count = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(resamples - 1, math.floor(tail * resamples)))
    high_index = max(0, min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1))
    return means[low_index], means[high_index]


def _permutation_p(
    values: Sequence[float], *, rng: random.Random, resamples: int
) -> tuple[float | None, str]:
    """Two-sided paired permutation test on the sign of each difference.

    Under the null the treatment label is exchangeable within a block, so every
    difference could equally have carried the opposite sign.  For small samples
    every assignment is enumerated, which is exact; beyond that the same seeded
    RNG samples them.
    """

    count = len(values)
    if count == 0:
        return None, "none"
    observed = abs(sum(values) / count)
    if count <= EXACT_PERMUTATION_LIMIT:
        extreme = 0
        total = 0
        for signs in itertools.product((1.0, -1.0), repeat=count):
            total += 1
            mean = sum(sign * value for sign, value in zip(signs, values, strict=True)) / count
            if abs(mean) >= observed - 1e-12:
                extreme += 1
        return extreme / total, "exact-permutation"
    extreme = 1  # the observed assignment is always one of the draws
    for _ in range(resamples):
        mean = sum(value if rng.random() < 0.5 else -value for value in values) / count
        if abs(mean) >= observed - 1e-12:
            extreme += 1
    return extreme / (resamples + 1), "monte-carlo-permutation"


def _fairness(plan: ExperimentPlan, rows: Sequence[TrialOutcome]) -> FairnessFacts:
    dispatch_index = {trial.trial_id: trial.dispatch_index for trial in plan.trials}
    arm_of = {trial.trial_id: trial.arm_id for trial in plan.trials}
    observed = [row for row in rows if row.trial_id in dispatch_index]
    started = sorted(
        (row for row in observed if _parse_time(row.started_at)),
        key=lambda row: _parse_time(row.started_at),  # type: ignore[arg-type]
    )
    start_rank = {row.trial_id: rank for rank, row in enumerate(started)}

    def mean_by_arm(source: dict[str, float | int]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for trial_id, value in source.items():
            arm = arm_of.get(trial_id)
            if arm is not None:
                buckets.setdefault(arm, []).append(float(value))
        return {arm: sum(values) / len(values) for arm, values in sorted(buckets.items())}

    mean_dispatch = mean_by_arm({row.trial_id: dispatch_index[row.trial_id] for row in observed})
    mean_start = mean_by_arm(dict(start_rank))
    bias = None
    if len(mean_start) >= 2 and started:
        spread = max(mean_start.values()) - min(mean_start.values())
        bias = spread / max(1, len(started) - 1)

    queue: dict[str, list[float]] = {}
    for row in observed:
        dispatched, start = _parse_time(row.dispatched_at), _parse_time(row.started_at)
        arm = arm_of.get(row.trial_id)
        if dispatched and start and arm:
            queue.setdefault(arm, []).append((start - dispatched).total_seconds())
    median_queue = {arm: _median(values) for arm, values in sorted(queue.items())}

    images: dict[str, list[str]] = {}
    namespaces: set[str] = set()
    for row in observed:
        arm = arm_of.get(row.trial_id)
        reference = row.infra.get("image_reference")
        if arm and isinstance(reference, str) and reference not in images.setdefault(arm, []):
            images[arm].append(reference)
        namespace = row.infra.get("cache_namespace")
        if isinstance(namespace, str):
            namespaces.add(namespace)
    return FairnessFacts(
        counterbalanced=bool(plan.design["counterbalance"]),
        mean_dispatch_index=mean_dispatch,
        mean_start_rank=mean_start,
        position_bias=bias,
        median_queue_seconds=median_queue,
        image_references={arm: sorted(items) for arm, items in sorted(images.items())},
        distinct_cache_namespaces=len(namespaces),
        trials_observed=len(observed),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _totals(
    plan: ExperimentPlan,
    rows: Sequence[TrialOutcome],
    by_trial: dict[str, TrialOutcome],
    *,
    retried: Sequence[str] = (),
) -> dict[str, Any]:
    planned = len(plan.trials)
    sealed = sum(1 for trial in plan.trials if trial.trial_id in by_trial)
    completed = sum(1 for row in rows if row.counted)
    costs = [
        float(row.usage["cost_usd"])
        for row in rows
        if isinstance(row.usage.get("cost_usd"), (int, float))
        and not isinstance(row.usage.get("cost_usd"), bool)
    ]
    starts = [t for t in (_parse_time(row.started_at) for row in rows) if t]
    finishes = [t for t in (_parse_time(row.finished_at) for row in rows) if t]
    return {
        "planned_trials": planned,
        "sealed_trials": sealed,
        "completed_trials": completed,
        "unsealed_trials": planned - sealed,
        "completion_rate": (completed / planned) if planned else 0.0,
        "cost_usd": sum(costs) if costs else None,
        "elapsed_seconds": (
            (max(finishes) - min(starts)).total_seconds() if starts and finishes else None
        ),
        "retried_trials": len(retried),
        "unplanned_rows": sorted(
            {row.trial_id for row in rows} - {trial.trial_id for trial in plan.trials}
        ),
    }


def _claim(
    plan: ExperimentPlan,
    *,
    mode: str,
    arms: Sequence[ArmAggregate],
    comparisons: Sequence[PairedComparison],
    fairness: FairnessFacts,
    totals: dict[str, Any],
    conflicts: Sequence[Any],
    retried: Sequence[str] = (),
) -> ClaimVerdict:
    design = plan.design
    blockers: list[str] = []
    notes: list[str] = []

    if mode == "aa":
        blockers.append(
            "this is an A/A run: it tests identity and isolation, and three blocks of "
            "the same arm cannot establish a noise ceiling"
        )
    if retried:
        # Not a blocker: a rig failure is not evidence about an arm. It is
        # never silent either -- a rig that needed retries is a fact about the
        # comparison, and the superseded rows stay in the log.
        notes.append(
            f"{len(retried)} trial(s) were re-dispatched after a rig failure: {list(retried)[:5]}"
        )
    if conflicts:
        blockers.append(f"{len(conflicts)} trial id(s) were sealed twice with different content")
    if totals["unplanned_rows"]:
        blockers.append(
            f"{len(totals['unplanned_rows'])} outcome row(s) are not in the plan: "
            f"{totals['unplanned_rows'][:3]}"
        )

    missing_by_arm = {arm.arm_id: len(arm.missing_blocks) for arm in arms}
    total_missing = sum(missing_by_arm.values())
    policy = design["missing_policy"]
    if total_missing:
        if policy == "fail":
            blockers.append(
                f"{total_missing} arm-block(s) produced no evidence and design.missing_policy "
                f"is 'fail': {missing_by_arm}"
            )
        else:
            differential = max(missing_by_arm.values()) - min(missing_by_arm.values())
            if differential > int(design["max_differential_missing_blocks"]):
                blockers.append(
                    f"missingness differs by {differential} block(s) between arms "
                    f"(limit {design['max_differential_missing_blocks']}); an arm cannot be "
                    "allowed to win by failing on the hard blocks"
                )
            else:
                notes.append(
                    f"pairwise-complete analysis over incomplete data ({missing_by_arm}); "
                    "no values were imputed"
                )
    if totals["completion_rate"] < float(design["min_completion_rate"]):
        blockers.append(
            f"completion rate {totals['completion_rate']:.2f} is below the declared "
            f"minimum {design['min_completion_rate']:.2f}"
        )

    for comparison in comparisons:
        if comparison.blocks_paired < int(design["min_blocks_for_claim"]):
            blockers.append(
                f"{comparison.treatment_arm_id} has {comparison.blocks_paired} paired block(s), "
                f"below the declared minimum {design['min_blocks_for_claim']}"
            )
        elif comparison.ci_low is None or comparison.ci_high is None:
            blockers.append(
                f"{comparison.treatment_arm_id}: too few paired blocks to interval-estimate "
                "the difference"
            )
        elif comparison.ci_low <= 0.0 <= comparison.ci_high:
            # A null result is a real finding and the report states it plainly.
            # What it cannot support is a directional headline, so the gate is
            # closed rather than annotated.
            blockers.append(
                f"{comparison.treatment_arm_id}: the {comparison.confidence:.0%} interval "
                f"[{comparison.ci_low:.6g}, {comparison.ci_high:.6g}] contains zero, so the "
                f"direction of the effect is not established (observed "
                f"{comparison.mean_delta:+.6g}, p={comparison.p_value:.3g})"
            )
        else:
            notes.append(
                f"{comparison.treatment_arm_id}: {comparison.confidence:.0%} interval "
                f"[{comparison.ci_low:.6g}, {comparison.ci_high:.6g}] excludes zero "
                f"(observed {comparison.mean_delta:+.6g}, p={comparison.p_value:.3g})"
            )

    if fairness.position_bias is not None and fairness.position_bias > float(
        design["max_position_bias"]
    ):
        blockers.append(
            f"observed start-order bias {fairness.position_bias:.2f} exceeds the declared "
            f"limit {design['max_position_bias']:.2f}; one arm systematically ran earlier"
        )
    drifted = {
        arm: references
        for arm, references in fairness.image_references.items()
        if len(references) > 1
    }
    if drifted:
        blockers.append(f"the target image changed inside an arm: {drifted}")
    distinct_images = {
        reference for references in fairness.image_references.values() for reference in references
    }
    if len(distinct_images) > 1:
        blockers.append(
            f"arms did not run against the same target image: {sorted(distinct_images)}"
        )
    if (
        plan.isolation["cache_namespace"] == "per_trial"
        and fairness.trials_observed
        and fairness.distinct_cache_namespaces < fairness.trials_observed
    ):
        blockers.append(
            f"isolation declares per-trial caches but only "
            f"{fairness.distinct_cache_namespaces} distinct namespace(s) were used across "
            f"{fairness.trials_observed} trials"
        )

    if not comparisons:
        blockers.append("no treatment arm to compare against the baseline")

    return ClaimVerdict(allowed=not blockers, blockers=tuple(blockers), notes=tuple(notes))


__all__ = [
    "EXACT_PERMUTATION_LIMIT",
    "ArmAggregate",
    "ClaimVerdict",
    "ExperimentReport",
    "FairnessFacts",
    "PairedComparison",
    "reduce_experiment",
]
