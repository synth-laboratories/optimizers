"""Dispatch: walk the frozen plan and seal one outcome row per trial.

This is deliberately the least clever file in the package. It does not schedule,
retry, adapt, or reorder — the plan already decided the order, and the executor
already owns concurrency through its own semaphore. Dispatching serially in the
planned order is what makes the counterbalancing real rather than aspirational,
and it is why the recorded dispatch/start/finish times mean something.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .adapters.base import TrialContext
from .analysis import ExperimentReport, reduce_experiment
from .models import ExperimentContractError, TrialOutcome
from .outcomes import OutcomeLog
from .plan import ExperimentPlan, assert_only_treatment_differs, compile_plan
from .spec import ExperimentSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .adapters.base import ExecutorAdapter

CANCEL_SENTINEL = "CANCEL"

PLAN_FILE = "plan.json"
OUTCOMES_FILE = "trial_outcomes.jsonl"
DISPATCH_FILE = "dispatch.jsonl"
REPORT_FILE = "report.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class RunSummary:
    dispatched: int
    sealed: int
    skipped: int
    stopped_reason: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "dispatched": self.dispatched,
            "sealed": self.sealed,
            "skipped": self.skipped,
            "stopped_reason": self.stopped_reason,
        }


class ExperimentRunner:
    def __init__(
        self,
        spec: ExperimentSpec,
        adapter: ExecutorAdapter,
        root: Path | str,
        *,
        mode: str = "experiment",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.mode = mode
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.outcomes = OutcomeLog(self.root / OUTCOMES_FILE)
        self._on_event = on_event

    # ------------------------------------------------------------------ plan

    @property
    def plan_path(self) -> Path:
        return self.root / PLAN_FILE

    def prepare(self) -> ExperimentPlan:
        """Compile the plan, or adopt a matching one that is already frozen.

        Recompiling on every command is what makes resume safe: if the spec, the
        recipe, the target image, or the staged candidate set has moved since the
        first dispatch, the digests disagree and this refuses rather than mixing
        two measurements into one comparison.
        """

        plan = compile_plan(self.spec, self.adapter, mode=self.mode)
        assert_only_treatment_differs(plan)
        if self.plan_path.is_file():
            frozen = ExperimentPlan.from_mapping(
                json.loads(self.plan_path.read_text(encoding="utf-8"))
            )
            if frozen.plan_digest != plan.plan_digest:
                raise ExperimentContractError(
                    f"{self.plan_path} was frozen at {frozen.plan_digest} but the spec and "
                    f"executor now compile to {plan.plan_digest}; this is a different "
                    "experiment, so start a new one rather than resuming into it"
                )
            return frozen
        self.plan_path.write_text(
            json.dumps(plan.to_json(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return plan

    # --------------------------------------------------------------- dispatch

    def run(self, *, limit: int | None = None) -> RunSummary:
        plan = self.prepare()
        sealed = self.outcomes.sealed_trial_ids()
        pending = [
            trial
            for trial in sorted(plan.trials, key=lambda item: item.dispatch_index)
            if trial.trial_id not in sealed
        ]
        skipped = len(plan.trials) - len(pending)
        self._emit(
            "experiment.run.planned",
            experiment_id=plan.experiment_id,
            plan_digest=plan.plan_digest,
            mode=self.mode,
            trials=len(plan.trials),
            pending=len(pending),
            resumed=skipped,
        )
        spent = self._spent()
        started = time.monotonic()
        dispatched = 0
        stopped: str | None = None
        for trial in pending:
            if limit is not None and dispatched >= limit:
                stopped = f"stopped after the requested {limit} trial(s)"
                break
            if (self.root / CANCEL_SENTINEL).exists():
                stopped = "cancelled by sentinel"
                break
            budget_stop = self._budget_exceeded(spent, started, dispatched)
            if budget_stop:
                stopped = budget_stop
                break
            arm = plan.arm(trial.arm_id)
            correlation = plan.correlation_for(trial)
            dispatched_at = _now()
            self._append(
                DISPATCH_FILE,
                {
                    "trial_id": trial.trial_id,
                    "arm_id": trial.arm_id,
                    "block_id": trial.block_id,
                    "replicate": trial.replicate,
                    "dispatch_index": trial.dispatch_index,
                    "dispatched_at": dispatched_at,
                    "correlation_digest": correlation.digest,
                    "aliases": correlation.aliases(),
                },
            )
            self._emit(
                "experiment.trial.dispatched",
                trial_id=trial.trial_id,
                arm_id=trial.arm_id,
                block_id=trial.block_id,
                dispatch_index=trial.dispatch_index,
            )
            context = TrialContext(
                spec=self.spec,
                plan=plan,
                trial=trial,
                treatment=dict(arm.treatment),
                fixed=dict(plan.fixed),
                correlation=correlation,
                workspace=self.root / "trials" / trial.trial_id,
                dispatched_at=dispatched_at,
            )
            context.workspace.mkdir(parents=True, exist_ok=True)
            outcome = self.adapter.run_trial(context)
            _assert_matches(outcome, trial.trial_id, correlation.digest)
            self.outcomes.append(outcome)
            dispatched += 1
            if isinstance(outcome.usage.get("cost_usd"), (int, float)):
                spent += float(outcome.usage["cost_usd"])
            self._emit(
                "experiment.trial.terminal",
                trial_id=trial.trial_id,
                status=outcome.status,
                failure_class=outcome.failure_class,
                metrics=outcome.metrics,
            )
        summary = RunSummary(
            dispatched=dispatched,
            sealed=len(self.outcomes.sealed_trial_ids()),
            skipped=skipped,
            stopped_reason=stopped,
        )
        self._emit("experiment.run.terminal", **summary.to_json())
        return summary

    def _spent(self) -> float:
        total = 0.0
        for row in self.outcomes.load():
            cost = row.usage.get("cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total += float(cost)
        return total

    def _budget_exceeded(self, spent: float, started: float, dispatched: int) -> str | None:
        budget = self.spec.budget
        if budget.max_cost_usd is not None and spent >= budget.max_cost_usd:
            return f"cost ceiling reached: ${spent:.2f} of ${budget.max_cost_usd:.2f}"
        if budget.max_wall_minutes is not None:
            elapsed = (time.monotonic() - started) / 60.0
            if elapsed >= budget.max_wall_minutes:
                return f"wall ceiling reached: {elapsed:.1f} of {budget.max_wall_minutes} minutes"
        return None

    # ----------------------------------------------------------------- report

    def report(self, *, write: bool = True) -> ExperimentReport:
        plan = self.prepare()
        report = reduce_experiment(plan, self.outcomes.load(), mode=self.mode)
        if write:
            (self.root / REPORT_FILE).write_text(
                json.dumps(report.to_json(), indent=2, sort_keys=True), encoding="utf-8"
            )
        return report

    # ------------------------------------------------------------------ plumbing

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        with open(self.root / name, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def _emit(self, event: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event({"event": event, "occurred_at": _now(), **fields})


def _assert_matches(outcome: TrialOutcome, trial_id: str, correlation_digest: str) -> None:
    """An adapter that seals someone else's trial would silently corrupt the log."""

    if outcome.trial_id != trial_id:
        raise ExperimentContractError(
            f"adapter sealed {outcome.trial_id!r} while running {trial_id!r}"
        )
    if outcome.correlation_digest != correlation_digest:
        raise ExperimentContractError(
            f"trial {trial_id} sealed a correlation digest the plan did not mint"
        )


__all__ = ["CANCEL_SENTINEL", "ExperimentRunner", "RunSummary"]
