"""Dispatch: walk the frozen plan and seal one outcome row per trial.

This is deliberately the least clever file in the package. It does not schedule,
retry, adapt, or reorder — the plan already decided the order, and the executor
already owns concurrency through its own semaphore. Dispatching serially in the
planned order is what makes the counterbalancing real rather than aspirational,
and it is why the recorded dispatch/start/finish times mean something.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .adapters.base import TrialContext
from .analysis import ExperimentReport, reduce_experiment
from .models import ExperimentContractError, TrialOutcome, digest_of
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


@dataclass
class _DispatchState:
    """Shared across dispatch threads; every field is guarded by `lock`."""

    spent: float
    started: float
    stopped: str | None = None
    dispatched: int = 0
    halted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


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

    def run(self, *, limit: int | None = None, retry_rig_failures: bool = False) -> RunSummary:
        """Dispatch pending trials in the planned order.

        `retry_rig_failures` re-dispatches trials whose sealed failure was the
        rig's, not the arm's. A crashed container says nothing about a treatment,
        so re-running it is not cherry-picking -- but the superseded row stays in
        the log and the report counts every retry, because a rig that needed
        three attempts is itself a fact about the comparison.
        """

        plan = self.prepare()
        held = self.outcomes.load().by_trial()
        retryable = {
            trial_id
            for trial_id, outcome in held.items()
            if retry_rig_failures and outcome.retryable
        }
        pending = [
            trial
            for trial in sorted(plan.trials, key=lambda item: item.dispatch_index)
            if trial.trial_id not in held or trial.trial_id in retryable
        ]
        skipped = len(plan.trials) - len(pending)
        if retryable:
            self._emit(
                "experiment.retry.planned",
                trials=sorted(retryable),
                reason="sealed rig failure",
            )
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
        stopped: str | None = None
        if limit is not None:
            pending = pending[:limit]
            stopped = f"stopped after the requested {limit} trial(s)"

        parallel = max(1, int(plan.execution.get("max_parallel_trials", 1)))
        state = _DispatchState(spent=spent, started=started, stopped=stopped)

        if parallel == 1:
            for trial in pending:
                if not self._dispatch(plan, trial, held, state):
                    break
        else:
            # Bounded fan-out, still submitted in planned order. Both arms of a
            # block meet the same machine at the same moment, which removes
            # temporal drift rather than adding it -- and the reducer keeps
            # measuring the order that actually happened either way.
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [
                    pool.submit(self._dispatch, plan, trial, held, state) for trial in pending
                ]
                for future in futures:
                    future.result()

        summary = RunSummary(
            dispatched=state.dispatched,
            sealed=len(self.outcomes.sealed_trial_ids()),
            skipped=skipped,
            stopped_reason=state.stopped,
        )
        self._emit("experiment.run.terminal", **summary.to_json())
        return summary

    def _dispatch(
        self,
        plan: ExperimentPlan,
        trial: Any,
        held: dict[str, TrialOutcome],
        state: _DispatchState,
    ) -> bool:
        """Run one trial. Returns False when the run should stop dispatching."""

        with state.lock:
            if state.halted:
                return False
            if (self.root / CANCEL_SENTINEL).exists():
                state.stopped = "cancelled by sentinel"
                state.halted = True
                return False
            budget_stop = self._budget_exceeded(state.spent, state.started, state.dispatched)
            if budget_stop:
                state.stopped = budget_stop
                state.halted = True
                return False
        if True:
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
            previous = held.get(trial.trial_id)
            attempt = previous.attempt + 1 if previous is not None else 0
            context = TrialContext(
                spec=self.spec,
                plan=plan,
                trial=trial,
                treatment=dict(arm.treatment),
                fixed=dict(plan.fixed),
                correlation=correlation,
                workspace=self.root / "trials" / trial.trial_id,
                dispatched_at=dispatched_at,
                attempt=attempt,
            )
            context.workspace.mkdir(parents=True, exist_ok=True)
            outcome = self.adapter.run_trial(context)
            _assert_matches(outcome, trial.trial_id, correlation.digest)
            if previous is not None:
                outcome = replace(
                    outcome,
                    attempt=attempt,
                    supersedes=digest_of(previous.to_json()),
                )
            self.outcomes.append(outcome)
            with state.lock:
                state.dispatched += 1
                if isinstance(outcome.usage.get("cost_usd"), (int, float)):
                    state.spent += float(outcome.usage["cost_usd"])
            self._emit(
                "experiment.trial.terminal",
                trial_id=trial.trial_id,
                status=outcome.status,
                failure_class=outcome.failure_class,
                metrics=outcome.metrics,
            )
        return True

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
