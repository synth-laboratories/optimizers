"""Scoring, elimination, and selection.

The container evaluates a trial; it never picks a winner. Everything here works
from normalized trial records, keeps every raw metric vector, and treats a
failed container as failed evidence rather than as a zero score — the single
most common way an evaluation lies about a policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import (
    CandidateScorecard,
    MetricSpec,
    MetricSummary,
    PolicyCandidate,
    SelectionDecision,
    SelectionSpec,
    TargetManifest,
    TrialRecord,
)


def summarize_candidate(
    candidate: PolicyCandidate,
    records: Sequence[TrialRecord],
    *,
    target: TargetManifest,
    stage: str,
    is_baseline: bool,
    baseline_records: Sequence[TrialRecord] = (),
    primary_metric: str | None = None,
    eliminated_at: str | None = None,
    elimination_reason: str | None = None,
) -> CandidateScorecard:
    mine = [record for record in records if record.key.candidate_id == candidate.id]
    valid = [record for record in mine if record.valid]
    summaries = []
    for metric in target.metrics:
        values = [record.metrics[metric.id] for record in valid if metric.id in record.metrics]
        summaries.append(
            MetricSummary(
                metric_id=metric.id,
                mean=(sum(values) / len(values)) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                count=len(values),
            )
        )
    gate_failures: dict[str, int] = {}
    for record in mine:
        for gate_id, passed in record.gates.items():
            if not passed:
                gate_failures[gate_id] = gate_failures.get(gate_id, 0) + 1
        for gate_id in record.missing_gates:
            gate_failures[gate_id] = gate_failures.get(gate_id, 0) + 1
    lift, paired = (None, 0)
    if primary_metric and not is_baseline and baseline_records:
        lift, paired = paired_lift(mine, baseline_records, metric_id=primary_metric, target=target)
    costs = [
        float(record.usage["cost_usd"])
        for record in mine
        if isinstance(record.usage.get("cost_usd"), (int, float))
    ]
    return CandidateScorecard(
        candidate_id=candidate.id,
        label=candidate.label,
        stage=stage,
        is_baseline=is_baseline,
        trials_total=len(mine),
        trials_valid=len(valid),
        trials_failed=sum(1 for record in mine if record.status != "evaluated"),
        metrics=tuple(summaries),
        gate_failures=gate_failures,
        paired_lift=lift,
        paired_trials=paired,
        eliminated_at=eliminated_at,
        elimination_reason=elimination_reason,
        cost_usd=sum(costs) if costs else None,
    )


def paired_lift(
    candidate_records: Sequence[TrialRecord],
    baseline_records: Sequence[TrialRecord],
    *,
    metric_id: str,
    target: TargetManifest,
) -> tuple[float | None, int]:
    """Mean signed difference over the seeds both arms actually completed.

    Common random numbers make this a paired comparison: only `(scenario, seed)`
    pairs where both the candidate and the baseline produced valid evidence
    contribute, so an arm cannot win by failing on the hard seeds.
    """

    spec = target.metric(metric_id)
    mine = {
        (record.key.scenario, record.key.seed): record.metrics[metric_id]
        for record in candidate_records
        if record.valid and metric_id in record.metrics
    }
    theirs = {
        (record.key.scenario, record.key.seed): record.metrics[metric_id]
        for record in baseline_records
        if record.valid and metric_id in record.metrics
    }
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return None, 0
    diffs = [spec.signed(mine[key]) - spec.signed(theirs[key]) for key in shared]
    return sum(diffs) / len(diffs), len(diffs)


def apply_elimination(
    selection: SelectionSpec,
    scorecards: Sequence[CandidateScorecard],
    *,
    target: TargetManifest,
    baseline_id: str | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return (survivor ids, {eliminated id: recorded reason}).

    Only the recipe's declared rule may remove a candidate, and the reason it
    gave is recorded alongside the candidate's retained screening evidence.
    Ranking is on the signed metric, so a `minimize` primary is not silently
    ranked backwards.
    """

    spec = target.metric(selection.primary_metric)
    contenders = [card for card in scorecards if card.candidate_id != baseline_id]
    eliminated: dict[str, str] = {}
    if selection.elimination.kind == "none":
        survivors = [card.candidate_id for card in contenders]
    elif selection.elimination.kind == "keep_top_k":
        keep = max(1, int(selection.elimination.value or 1))
        ranked = sorted(contenders, key=lambda card: _signed_mean(card, spec), reverse=True)
        survivors = [card.candidate_id for card in ranked[:keep]]
        for card in ranked[keep:]:
            eliminated[card.candidate_id] = (
                f"screening rank below keep_top_k={keep} on {selection.primary_metric}"
            )
    else:  # min_primary_mean
        floor = float(selection.elimination.value or 0.0)
        survivors = []
        for card in contenders:
            mean = _primary_mean(card, selection.primary_metric)
            if mean is not None and spec.signed(mean) >= spec.signed(floor):
                survivors.append(card.candidate_id)
            else:
                eliminated[card.candidate_id] = (
                    f"screening {selection.primary_metric} mean "
                    f"{'missing' if mean is None else round(mean, 6)} did not clear {floor}"
                )
    if baseline_id:
        survivors = [baseline_id, *survivors]
    return tuple(survivors), eliminated


def decide(
    selection: SelectionSpec,
    *,
    scorecards: Sequence[CandidateScorecard],
    records: Iterable[TrialRecord],
    baseline_id: str | None,
    cancelled: bool = False,
) -> SelectionDecision:
    records = list(records)
    incomplete = [record for record in records if not record.valid]
    if cancelled:
        return SelectionDecision(
            status="invalid_evidence",
            winner_id=None,
            baseline_id=baseline_id,
            primary_metric=selection.primary_metric,
            lift=None,
            min_lift=selection.min_lift,
            reason="run was cancelled before the trial matrix completed",
        )
    if incomplete:
        return SelectionDecision(
            status="invalid_evidence",
            winner_id=None,
            baseline_id=baseline_id,
            primary_metric=selection.primary_metric,
            lift=None,
            min_lift=selection.min_lift,
            reason=(
                f"{len(incomplete)} of {len(records)} trials did not produce valid evidence; "
                "failed trials are retained as failures, not scored as zero"
            ),
        )
    if selection.decision_mode == "report_only":
        return SelectionDecision(
            status="inconclusive",
            winner_id=None,
            baseline_id=baseline_id,
            primary_metric=selection.primary_metric,
            lift=None,
            min_lift=selection.min_lift,
            reason="recipe is report-only; it measures candidates but never promotes one",
        )
    if not baseline_id:
        return SelectionDecision(
            status="inconclusive",
            winner_id=None,
            baseline_id=None,
            primary_metric=selection.primary_metric,
            lift=None,
            min_lift=selection.min_lift,
            reason="no baseline was designated, so no paired lift could be computed",
        )
    eligible = [
        card
        for card in scorecards
        if not card.is_baseline
        and card.eliminated_at is None
        and card.paired_lift is not None
        and card.paired_trials >= selection.min_valid_trials
    ]
    if not eligible:
        return SelectionDecision(
            status="inconclusive",
            winner_id=None,
            baseline_id=baseline_id,
            primary_metric=selection.primary_metric,
            lift=None,
            min_lift=selection.min_lift,
            reason=(
                f"no surviving candidate reached {selection.min_valid_trials} paired "
                f"confirmation trials against the baseline"
            ),
        )
    best = max(eligible, key=lambda card: card.paired_lift or 0.0)
    lift = best.paired_lift or 0.0
    if lift >= selection.min_lift:
        return SelectionDecision(
            status="promoted",
            winner_id=best.candidate_id,
            baseline_id=baseline_id,
            primary_metric=selection.primary_metric,
            lift=lift,
            min_lift=selection.min_lift,
            reason=(
                f"{best.label} beat the baseline by {lift:.6g} on "
                f"{selection.primary_metric} over {best.paired_trials} paired trials"
            ),
        )
    return SelectionDecision(
        status="no_champion",
        winner_id=None,
        baseline_id=baseline_id,
        primary_metric=selection.primary_metric,
        lift=lift,
        min_lift=selection.min_lift,
        reason=(
            f"best paired lift {lift:.6g} on {selection.primary_metric} did not reach the "
            f"required {selection.min_lift:.6g}"
        ),
    )


def _primary_mean(card: CandidateScorecard, metric_id: str) -> float | None:
    for summary in card.metrics:
        if summary.metric_id == metric_id:
            return summary.mean
    return None


def _signed_mean(card: CandidateScorecard, spec: MetricSpec) -> float:
    mean = _primary_mean(card, spec.id)
    return float("-inf") if mean is None else spec.signed(mean)
