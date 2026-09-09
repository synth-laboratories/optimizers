"""Adapter for the local `eval` runtime.

`eval` already runs a fair `candidate x seed x scenario` matrix with common
random numbers, so this adapter does not reimplement any of that. It drives one
`eval` run per cell of the experiment matrix, which is what buys the two things
the experiment layer needs and a single batched run cannot give: an exact
dispatch order to counterbalance, and one sealed, separately resumable receipt
per trial.

Every override it sends is a narrowing of the trusted recipe. The recipe stays
the only place an image, a seed schedule, a model route, or a resource ceiling
can come from; the experiment only chooses among what is already there.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...eval.home import EvalHome
from ...eval.models import (
    WORKER_MANIFEST_SCHEMA,
    CandidateSet,
    EvalContractError,
    PolicyCandidate,
    TrialRecord,
)
from ...eval.recipes import EvalRecipe
from ...eval.runner import run_worker
from ..models import (
    AblatableFactor,
    ExperimentContractError,
    FactorCatalog,
    SubjectRef,
    TrialOutcome,
    digest_of,
)
from .base import TrialContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..spec import ExperimentSpec

CANDIDATE_FACTOR = "policy.candidate"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class EvalRuntimeAdapter:
    """Drives `python -m synth_optimizers.eval worker` once per trial."""

    executor_id = "eval.runtime"

    def __init__(
        self,
        *,
        home: Path | str,
        candidate_set: Path | str,
        executor: Any | None = None,
        stream: Any = None,
    ) -> None:
        self.home = EvalHome.open(home)
        self.candidate_set = self._load_candidate_set(Path(candidate_set))
        self._executor = executor
        self._stream = stream
        self._recipes: dict[str, EvalRecipe] = {}

    @classmethod
    def from_spec(cls, spec: ExperimentSpec, **overrides: Any) -> EvalRuntimeAdapter:
        options = dict(spec.executor_options)
        for key in ("home", "candidate_set"):
            if key not in options:
                raise ExperimentContractError(
                    f"executor_options.{key} is required for {cls.executor_id}"
                )
        options.update(overrides)
        unknown = sorted(set(options) - {"home", "candidate_set", "executor", "stream"})
        if unknown:
            raise ExperimentContractError(f"executor_options does not accept {unknown}")
        return cls(**options)

    def _load_candidate_set(self, path: Path) -> CandidateSet:
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "candidate_set.json"
        if not path.is_file():
            # A bare id is the common case, because that is what `eval stage` prints.
            candidate = self.home.candidates_dir / path.name / "candidate_set.json"
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            raise ExperimentContractError(f"no staged candidate set at {path}")
        return CandidateSet.load(path)

    def recipe(self, spec: ExperimentSpec) -> EvalRecipe:
        if spec.base not in self._recipes:
            try:
                self._recipes[spec.base] = self.home.recipe(spec.base)
            except EvalContractError as error:
                raise ExperimentContractError(str(error)) from error
        return self._recipes[spec.base]

    # ------------------------------------------------------------- the catalog

    def factor_catalog(self, spec: ExperimentSpec) -> FactorCatalog:
        recipe = self.recipe(spec)
        factors = [
            AblatableFactor(
                path=CANDIDATE_FACTOR,
                kind="enum",
                description=(
                    "Which staged policy artifact runs, named by its staging label. "
                    "The label is what a spec is written against; the content digest is "
                    "what the comparison is keyed on."
                ),
                values=tuple(candidate.label for candidate in self.candidate_set.candidates),
            )
        ]
        for model in recipe.models:
            if len(model.efforts) < 2:
                # One declared effort is not a knob; offering it as one would
                # invite a spec that looks like an ablation and cannot vary.
                continue
            factors.append(
                AblatableFactor(
                    path=f"model.{model.id}.effort",
                    kind="enum",
                    description=f"Reasoning effort for the {model.id} route.",
                    values=tuple(model.efforts),
                )
            )
        return FactorCatalog(executor=self.executor_id, base_ref=recipe.id, factors=tuple(factors))

    # ------------------------------------------------------------- projections

    def fixed_projection(self, spec: ExperimentSpec) -> dict[str, Any]:
        recipe = self.recipe(spec)
        return {
            "recipe.id": recipe.id,
            "recipe.digest": digest_of(recipe.to_json()),
            "recipe.image": recipe.image,
            "recipe.image_digest": recipe.image_digest,
            "target.manifest_digest": digest_of(recipe.target.to_json()),
            "target.scenarios": list(recipe.scenarios),
            "target.network": recipe.target.network,
            "limits": recipe.limits.to_json(),
            "selection": recipe.selection.to_json(),
            "budget": recipe.budget.to_json() if recipe.budget else None,
            "candidate_set.id": self.candidate_set.id,
            "candidate_set.digest": self.candidate_set.digest(),
            "model.routes": [
                {"id": model.id, "route": model.route, "secret": model.secret}
                for model in recipe.models
            ],
        }

    def provenance(self, spec: ExperimentSpec) -> dict[str, Any]:
        recipe = self.recipe(spec)
        return {
            "executor": self.executor_id,
            "recipe_id": recipe.id,
            "recipe_digest": digest_of(recipe.to_json()),
            "image": recipe.image,
            "image_digest": recipe.image_digest,
            "target_manifest_digest": digest_of(recipe.target.to_json()),
            "candidate_set_digest": self.candidate_set.digest(),
        }

    def environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "eval_home": str(self.home.root),
            "container_runtime": self.home.config.container_runtime,
            "global_max_concurrent_trials": self.home.config.max_concurrent_trials,
        }

    def validate_blocks(self, spec: ExperimentSpec) -> None:
        recipe = self.recipe(spec)
        if spec.blocks.kind != "seed":
            raise ExperimentContractError(
                f"{self.executor_id} blocks on seeds; blocks.kind {spec.blocks.kind!r} "
                "has no meaning for it"
            )
        declared = set(recipe.screening_seeds) | set(recipe.confirmation_seeds)
        requested = {value for value in spec.blocks.values if isinstance(value, int)}
        if len(requested) != len(spec.blocks.values):
            raise ExperimentContractError("blocks.values must be integer seeds for eval.runtime")
        extra = sorted(requested - declared)
        if extra:
            raise ExperimentContractError(
                f"recipe {recipe.id} does not declare seeds {extra}; a seed schedule comes "
                "from the trusted catalog, so widen the recipe rather than the experiment"
            )
        if not recipe.available:
            raise ExperimentContractError(
                f"recipe {recipe.id} is unavailable: {recipe.unavailable_reason}"
            )

    def subject_for(self, spec: ExperimentSpec, treatment: dict[str, Any]) -> SubjectRef:
        candidate = self._candidate(spec, treatment)
        return SubjectRef(
            subject_kind="policy-candidate",
            subject_id=candidate.id,
            subject_content_digest=candidate.artifact_digest,
            parent_subject_id=candidate.metadata.get("parent_candidate_id"),
        )

    def metric_direction(self, spec: ExperimentSpec, metric_id: str) -> str:
        recipe = self.recipe(spec)
        try:
            return recipe.target.metric(metric_id).direction
        except EvalContractError as error:
            raise ExperimentContractError(
                f"{metric_id!r} is not a metric target {recipe.image} reports: {error}"
            ) from error

    def trial_derived(
        self,
        spec: ExperimentSpec,
        *,
        arm_id: str,
        block_id: str,
        replicate: int,
        trial_id: str,
    ) -> dict[str, Any]:
        run_id = f"opt_eval_{trial_id}"
        return {
            "run_id": run_id,
            "cache_namespace": run_id,
            "seed": int(block_id.split(":", 1)[1]),
            "evidence_dir": str(self.home.run_dir(run_id)),
        }

    # -------------------------------------------------------------- execution

    def run_trial(self, context: TrialContext) -> TrialOutcome:
        spec, trial = context.spec, context.trial
        recipe = self.recipe(spec)
        candidate = self._candidate(spec, context.treatment)
        seed = int(trial.trial_derived["seed"])
        run_id = _attempt_run_id(trial.trial_derived["run_id"], context.attempt)
        efforts = {
            path.split(".", 2)[1]: value
            for path, value in {**context.fixed, **context.treatment}.items()
            if path.startswith("model.") and path.endswith(".effort")
        }
        manifest_path = context.workspace / "worker_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": WORKER_MANIFEST_SCHEMA,
                    "run_id": run_id,
                    "recipe_id": recipe.id,
                    "home": str(self.home.root),
                    "candidate_set_path": str(self._candidate_set_path()),
                    "session_ref": context.plan.experiment_id,
                    "correlation": context.correlation.to_json(),
                    "plan_override": {
                        "candidate_ids": [candidate.id],
                        "seeds": [seed],
                        "model_efforts": efforts,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        started_at = _now()
        try:
            exit_code = run_worker(manifest_path, executor=self._executor, stream=self._stream)
            admission_error: str | None = None
        except EvalContractError as error:
            exit_code, admission_error = 1, str(error)
        finished_at = _now()
        return self._seal(
            context,
            run_id=run_id,
            exit_code=exit_code,
            admission_error=admission_error,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _candidate_set_path(self) -> Path:
        root = self.candidate_set.root
        if root is None:  # pragma: no cover - CandidateSet.load always sets it
            raise ExperimentContractError("candidate set was loaded without a staging root")
        return root / "candidate_set.json"

    # ------------------------------------------------------------------ seal

    def _seal(
        self,
        context: TrialContext,
        *,
        run_id: str,
        exit_code: int,
        admission_error: str | None,
        started_at: str,
        finished_at: str,
    ) -> TrialOutcome:
        run_dir = self.home.run_dir(run_id)
        manifest_path = run_dir / "result_manifest.json"
        records = self._records(run_dir)
        base = {
            "experiment_id": context.plan.experiment_id,
            "trial_id": context.trial.trial_id,
            "arm_id": context.trial.arm_id,
            "block_id": context.trial.block_id,
            "replicate": context.trial.replicate,
            "executor": self.executor_id,
            "executor_run_id": run_id,
            "correlation_digest": context.correlation.digest,
            "dispatched_at": context.dispatched_at,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        receipt = None
        image_reference = None
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            receipt = digest_of(payload)
            image_reference = payload.get("image_reference")
        infra = {
            "cache_namespace": run_id,
            "attempt": context.attempt,
            "container_runtime": self.home.config.container_runtime,
            "image_reference": image_reference,
            "evidence_dir": str(run_dir),
            "worker_exit_code": exit_code,
        }
        evidence = {
            "result_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "events": str(run_dir / "events.jsonl"),
            "trials": [str(Path(record.evidence_dir) / "job_result.json") for record in records],
            "trace_refs": [
                {
                    "trial_id": record.trial_id,
                    "role": artifact.get("role"),
                    "relative_path": artifact.get("relative_path"),
                    "digest": artifact.get("digest"),
                    "bytes": artifact.get("bytes"),
                    "path": artifact.get("path"),
                }
                for record in records
                for artifact in record.artifacts
                if artifact.get("role") == "trace"
            ],
        }
        if admission_error is not None:
            # Refused before a container started: no evidence exists, and the
            # gap is the honest record.
            return TrialOutcome(
                **base,
                status="failed",
                metrics={},
                usage={},
                failure_class="rig",
                failure_detail=admission_error,
                receipt_digest=receipt,
                evidence=evidence,
                infra=infra,
            )
        if not records:
            return TrialOutcome(
                **base,
                status="failed",
                metrics={},
                usage={},
                failure_class="infra" if exit_code != 0 else "rig",
                failure_detail=(
                    f"the eval worker exited {exit_code} without sealing a trial record"
                    + (f": {_terminal_error(run_dir)}" if _terminal_error(run_dir) else "")
                ),
                receipt_digest=receipt,
                evidence=evidence,
                infra=infra,
            )
        status, failure_class, detail = _classify(records)
        metrics: dict[str, float] = {}
        if status == "completed":
            # Averaged across the recipe's scenarios for this one seed; with a
            # single-scenario recipe this is just the trial's own number.
            keys = set(records[0].metrics)
            for record in records[1:]:
                keys &= set(record.metrics)
            metrics = {
                key: sum(record.metrics[key] for record in records) / len(records) for key in keys
            }
        costs = [
            float(record.usage["cost_usd"])
            for record in records
            if isinstance(record.usage.get("cost_usd"), (int, float))
            and not isinstance(record.usage.get("cost_usd"), bool)
        ]
        usage: dict[str, Any] = {
            "scenarios": len(records),
            "budget_exhausted_trials": sum(
                1 for record in records if record.usage.get("budget_exhausted")
            ),
        }
        if costs:
            usage["cost_usd"] = sum(costs)
        return TrialOutcome(
            **base,
            status=status,
            metrics=metrics,
            usage=usage,
            failure_class=failure_class,
            failure_detail=detail,
            receipt_digest=receipt,
            evidence=evidence,
            infra=infra,
        )

    def _records(self, run_dir: Path) -> list[TrialRecord]:
        records = []
        for path in sorted((run_dir / "trials").glob("*/job_result.json")):
            try:
                records.append(
                    TrialRecord.from_mapping(json.loads(path.read_text(encoding="utf-8")))
                )
            except (EvalContractError, json.JSONDecodeError, OSError):
                continue
        return records

    # ------------------------------------------------------------- candidates

    def _selector(self, spec: ExperimentSpec, treatment: dict[str, Any]) -> str | None:
        if CANDIDATE_FACTOR in treatment:
            return str(treatment[CANDIDATE_FACTOR])
        if CANDIDATE_FACTOR in spec.fixed:
            return str(spec.fixed[CANDIDATE_FACTOR])
        if len(self.candidate_set.candidates) == 1:
            return self.candidate_set.candidates[0].label
        return None

    def _candidate(self, spec: ExperimentSpec, treatment: dict[str, Any]) -> PolicyCandidate:
        selector = self._selector(spec, treatment)
        if selector is None:
            raise ExperimentContractError(
                f"the staged set holds {len(self.candidate_set.candidates)} candidates, so the "
                f"spec must either vary {CANDIDATE_FACTOR!r} or pin it under [fixed]; "
                "otherwise a trial has no single subject"
            )
        for candidate in self.candidate_set.candidates:
            if selector in (candidate.label, candidate.id):
                return candidate
        labels = ", ".join(candidate.label for candidate in self.candidate_set.candidates)
        raise ExperimentContractError(
            f"no staged candidate named {selector!r} in set {self.candidate_set.id}; "
            f"staged labels are: {labels}"
        )


def _terminal_error(run_dir: Path) -> str | None:
    """Lift the worker's own reason out of its event log.

    Without this a rig problem reads as `exited 1`, and the actual cause — an
    unpinned image, a missing secret, a runtime that is not installed — stays
    buried in a file nobody opens until the experiment is already over.
    """

    events = run_dir / "events.jsonl"
    if not events.is_file():
        return None
    reason = None
    try:
        for line in events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("event") == "eval.run.terminal" and payload.get("error"):
                reason = str(payload["error"])
    except (json.JSONDecodeError, OSError):
        return None
    return reason


def _attempt_run_id(base: Any, attempt: int) -> str:
    """A retry runs in its own directory.

    The eval runner treats an existing `job_result.json` as terminal and
    replays it, so re-dispatching under the same run id would return the very
    failure the retry exists to clear.
    """

    return str(base) if attempt == 0 else f"{base}_r{attempt}"


def _classify(records: list[TrialRecord]) -> tuple[str, str | None, str | None]:
    """One trial is complete only when every scenario in it produced valid evidence.

    Averaging over whichever scenarios happened to survive would let an arm win
    by failing on the hard ones, which is the same error as scoring a failure as
    zero with an extra step.
    """

    if all(record.valid for record in records):
        return "completed", None, None
    invalid = [record for record in records if not record.valid]
    if any(record.status == "cancelled" for record in invalid):
        return "cancelled", "cancelled", "the run was cancelled before this trial sealed"
    if any(record.status == "timeout" for record in invalid):
        return "timeout", "timeout", "a scenario exceeded the recipe's per-trial timeout"
    reasons = []
    for record in invalid:
        if record.missing_gates:
            reasons.append(f"{record.trial_id}: missing gates {list(record.missing_gates)}")
        elif record.missing_artifacts:
            reasons.append(f"{record.trial_id}: missing artifacts {list(record.missing_artifacts)}")
        elif record.status != "evaluated":
            reasons.append(f"{record.trial_id}: {record.status} ({record.error or 'no detail'})")
        else:
            failed = sorted(gate for gate, passed in record.gates.items() if not passed)
            reasons.append(f"{record.trial_id}: gates failed {failed}")
    return "failed", "rig", "; ".join(reasons[:4])


__all__ = ["CANDIDATE_FACTOR", "EvalRuntimeAdapter"]
