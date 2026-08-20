"""Deterministic expansion of a spec into a frozen plan.

Everything that decides what gets measured happens here, before a container
starts: the arms, the blocks, the dispatch order, and the three config
projections.  A plan is a pure function of the spec plus the executor's
provenance, so `plan` twice gives the same digest and `resume` recomputes the
same trial identities from the spec alone.

The dispatch order is materialised rather than described.  Counterbalancing that
exists only as a policy string is not evidence of fairness under a semaphore, so
the order is written down and the times each trial was actually dispatched,
started, and finished are recorded against it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .models import (
    EXPERIMENT_PLAN_SCHEMA,
    ArmPlan,
    CorrelationEnvelope,
    ExperimentContractError,
    FactorCatalog,
    SubjectRef,
    TrialPlan,
    digest_of,
    mint_trial_id,
)
from .spec import ExperimentSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .adapters.base import ExecutorAdapter


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    experiment_id: str
    spec_digest: str
    executor: str
    base_ref: str
    design: dict[str, Any]
    blocks: dict[str, Any]
    budget: dict[str, Any]
    isolation: dict[str, Any]
    execution: dict[str, Any]
    factor_catalog: dict[str, Any]
    fixed: dict[str, Any]
    provenance: dict[str, Any]
    arms: tuple[ArmPlan, ...]
    trials: tuple[TrialPlan, ...]
    primary_metric_direction: str
    secondary_metric_directions: dict[str, str]
    created_at: str
    environment: dict[str, Any]

    def core(self) -> dict[str, Any]:
        """The part of the plan that defines the measurement.

        `created_at` and `environment` are excluded so replanning the same spec
        against the same executor reproduces the same digest.  Provenance is
        included: if the target image moves, this is a different measurement and
        a resume must refuse it.
        """

        return {
            "schema_version": EXPERIMENT_PLAN_SCHEMA,
            "experiment_id": self.experiment_id,
            "spec_digest": self.spec_digest,
            "executor": self.executor,
            "base_ref": self.base_ref,
            "design": self.design,
            "blocks": self.blocks,
            "budget": self.budget,
            "isolation": self.isolation,
            "execution": self.execution,
            "factor_catalog": self.factor_catalog,
            "fixed": self.fixed,
            "provenance": self.provenance,
            "primary_metric_direction": self.primary_metric_direction,
            "secondary_metric_directions": self.secondary_metric_directions,
            "arms": [arm.to_json() for arm in self.arms],
            "trials": [trial.to_json() for trial in self.trials],
        }

    @property
    def plan_digest(self) -> str:
        return digest_of(self.core())

    def to_json(self) -> dict[str, Any]:
        return {
            **self.core(),
            "plan_digest": self.plan_digest,
            "created_at": self.created_at,
            "environment": self.environment,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> ExperimentPlan:
        if not isinstance(value, dict):
            raise ExperimentContractError("experiment plan must be an object")
        schema = value.get("schema_version")
        if schema != EXPERIMENT_PLAN_SCHEMA:
            raise ExperimentContractError(f"unsupported experiment plan schema {schema!r}")
        plan = cls(
            experiment_id=value["experiment_id"],
            spec_digest=value["spec_digest"],
            executor=value["executor"],
            base_ref=value["base_ref"],
            design=value["design"],
            blocks=value["blocks"],
            budget=value["budget"],
            isolation=value["isolation"],
            execution=value.get("execution", {"max_parallel_trials": 1}),
            factor_catalog=value["factor_catalog"],
            fixed=value["fixed"],
            provenance=value["provenance"],
            arms=tuple(ArmPlan.from_mapping(item) for item in value["arms"]),
            trials=tuple(TrialPlan.from_mapping(item) for item in value["trials"]),
            primary_metric_direction=value["primary_metric_direction"],
            secondary_metric_directions=value.get("secondary_metric_directions", {}),
            created_at=value.get("created_at", ""),
            environment=value.get("environment", {}),
        )
        recorded = value.get("plan_digest")
        if recorded is not None and recorded != plan.plan_digest:
            raise ExperimentContractError(
                "plan.json has been edited after it was written: its recorded digest "
                f"{recorded} does not match its contents"
            )
        return plan

    # ---------------------------------------------------------------- lookups

    def arm(self, arm_id: str) -> ArmPlan:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ExperimentContractError(f"unknown arm {arm_id!r}")

    def trial(self, trial_id: str) -> TrialPlan:
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        raise ExperimentContractError(f"unknown trial {trial_id!r}")

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(self.blocks["ids"])

    def correlation_for(self, trial: TrialPlan) -> CorrelationEnvelope:
        arm = self.arm(trial.arm_id)
        candidate_id = trial.trial_derived.get("candidate_id")
        if candidate_id is None and arm.subject.subject_kind.endswith("candidate"):
            # The alias exists so a trace can be walked back to a candidate. When
            # the subject *is* the candidate, that is the answer.
            candidate_id = arm.subject.subject_id
        return CorrelationEnvelope(
            experiment_id=self.experiment_id,
            arm_id=trial.arm_id,
            block_id=trial.block_id,
            replicate=trial.replicate,
            trial_id=trial.trial_id,
            plan_digest=self.plan_digest,
            subject=arm.subject,
            candidate_id=candidate_id,
        )


# ---------------------------------------------------------------- compilation


#: How many blocks an A/A preflight uses.  It is a smoke test for identity and
#: isolation, so it is deliberately too small to be mistaken for a noise
#: estimate; the reducer refuses a headline from it regardless.
AA_BLOCKS = 3


def compile_plan(
    spec: ExperimentSpec, adapter: ExecutorAdapter, *, mode: str = "experiment"
) -> ExperimentPlan:
    if mode not in ("experiment", "aa"):
        raise ExperimentContractError(f"unknown plan mode {mode!r}")
    if spec.executor != adapter.executor_id:
        raise ExperimentContractError(
            f"spec targets executor {spec.executor!r} but adapter is {adapter.executor_id!r}"
        )
    catalog = adapter.factor_catalog(spec)
    adapter.validate_blocks(spec)

    fixed = dict(adapter.fixed_projection(spec))
    for path, value in sorted(spec.fixed.items()):
        fixed[path] = catalog.normalize(path, value)

    if mode == "aa":
        spec = _aa_spec(spec)
        arms = _aa_arms(spec, catalog, adapter)
    else:
        arms = _arms(spec, catalog, adapter)
        _refuse_shared_subject_across_arms(arms)

    block_ids = spec.blocks.block_ids()
    trials = _dispatch_order(spec, arms, block_ids, adapter)

    if spec.budget.max_trials is not None and len(trials) > spec.budget.max_trials:
        raise ExperimentContractError(
            f"plan expands to {len(trials)} trials, over budget.max_trials {spec.budget.max_trials}"
        )

    return ExperimentPlan(
        experiment_id=spec.experiment_id,
        spec_digest=spec.digest,
        executor=spec.executor,
        base_ref=spec.base,
        design=spec.design.to_json(),
        blocks={**spec.blocks.to_json(), "ids": list(block_ids)},
        budget=spec.budget.to_json(),
        isolation=spec.isolation.to_json(),
        execution=spec.execution.to_json(),
        factor_catalog=catalog.to_json(),
        fixed=fixed,
        provenance=dict(adapter.provenance(spec)),
        arms=arms,
        trials=trials,
        primary_metric_direction=adapter.metric_direction(spec, spec.design.primary_metric),
        secondary_metric_directions={
            metric: adapter.metric_direction(spec, metric)
            for metric in spec.design.secondary_metrics
        },
        created_at=_now(),
        environment=dict(adapter.environment(spec)),
    )


def _aa_spec(spec: ExperimentSpec) -> ExperimentSpec:
    """The same spec, pinned to its first level and cut down to a few blocks."""

    from dataclasses import replace

    from .spec import BlockSpec

    return replace(
        spec,
        experiment_id=f"{spec.experiment_id}.aa",
        blocks=BlockSpec(
            kind=spec.blocks.kind,
            values=spec.blocks.values[:AA_BLOCKS],
            replicates=1,
        ),
    )


def _aa_arms(
    spec: ExperimentSpec, catalog: FactorCatalog, adapter: ExecutorAdapter
) -> tuple[ArmPlan, ...]:
    """Two arms that are identical on purpose.

    An A/A run answers one question: does this harness manufacture a difference
    between two runs of the same thing?  A non-zero paired delta here is an
    isolation bug — a shared cache, a warm container, a leaked seed — and finding
    it costs two arms rather than a retracted result.
    """

    treatment = {
        path: catalog.normalize(path, spec.factors[path][0]) for path in sorted(spec.factors)
    }
    subject = adapter.subject_for(spec, treatment)
    return tuple(
        ArmPlan(
            arm_id=f"arm_aa_{side}",
            label=f"A/A control {side.upper()} ({_label(catalog, treatment)})",
            treatment=treatment,
            subject=subject,
        )
        for side in ("a", "b")
    )


def _arms(
    spec: ExperimentSpec, catalog: FactorCatalog, adapter: ExecutorAdapter
) -> tuple[ArmPlan, ...]:
    paths = sorted(spec.factors)
    levels = [[catalog.normalize(path, value) for value in spec.factors[path]] for path in paths]
    arms: list[ArmPlan] = []
    for combination in itertools.product(*levels):
        treatment = dict(zip(paths, combination, strict=True))
        subject = adapter.subject_for(spec, treatment)
        if not isinstance(subject, SubjectRef):
            raise ExperimentContractError(
                f"{adapter.executor_id} returned a subject that is not a SubjectRef"
            )
        arms.append(
            ArmPlan(
                arm_id="arm_" + digest_of(treatment)[7:19],
                label=_label(catalog, treatment),
                treatment=treatment,
                subject=subject,
            )
        )
    ids = [arm.arm_id for arm in arms]
    if len(set(ids)) != len(ids):  # pragma: no cover - digest collision
        raise ExperimentContractError("arm ids collided; two arms have the same treatment")
    return tuple(arms)


def _label(catalog: FactorCatalog, treatment: dict[str, Any]) -> str:
    parts = []
    for path in sorted(treatment):
        factor = catalog.factor(path)
        parts.append(f"{path}={factor.display(treatment[path])}")
    return " ".join(parts)


def _refuse_shared_subject_across_arms(arms: Sequence[ArmPlan]) -> None:
    """Two arms may share a subject; they may not share a subject *and* nothing else.

    A distinct treatment that resolves to an identical subject digest and an
    identical treatment payload would be the same arm twice, which the reducer
    would then read as a paired comparison of a thing against itself.
    """

    seen: dict[tuple[str, str], str] = {}
    for arm in arms:
        key = (arm.subject.subject_content_digest, arm.treatment_digest)
        if key in seen:
            raise ExperimentContractError(
                f"arms {seen[key]} and {arm.arm_id} are identical in both treatment and "
                "subject; that is one arm, not two"
            )
        seen[key] = arm.arm_id


def _dispatch_order(
    spec: ExperimentSpec,
    arms: Sequence[ArmPlan],
    block_ids: Sequence[str],
    adapter: ExecutorAdapter,
) -> tuple[TrialPlan, ...]:
    """Materialise the order trials are handed to the executor.

    Blocks stay contiguous so the arms of a pair meet the same machine within a
    few minutes of each other, and the arm order rotates by block so no arm is
    systematically first.  With `counterbalance = false` the declared order is
    used unchanged, which is honest but leaves position confounded with arm.
    """

    trials: list[TrialPlan] = []
    index = 0
    for block_index, block_id in enumerate(block_ids):
        for replicate in range(spec.blocks.replicates):
            if spec.design.counterbalance:
                shift = (block_index + replicate) % len(arms)
                ordered = [*arms[shift:], *arms[:shift]]
            else:
                ordered = list(arms)
            for arm in ordered:
                trial_id = mint_trial_id(
                    experiment_id=spec.experiment_id,
                    arm_id=arm.arm_id,
                    block_id=block_id,
                    replicate=replicate,
                )
                trials.append(
                    TrialPlan(
                        trial_id=trial_id,
                        arm_id=arm.arm_id,
                        block_id=block_id,
                        replicate=replicate,
                        dispatch_index=index,
                        trial_derived=dict(
                            adapter.trial_derived(
                                spec,
                                arm_id=arm.arm_id,
                                block_id=block_id,
                                replicate=replicate,
                                trial_id=trial_id,
                            )
                        ),
                    )
                )
                index += 1
    ids = [trial.trial_id for trial in trials]
    if len(set(ids)) != len(ids):  # pragma: no cover - would be a minting bug
        raise ExperimentContractError("trial ids collided during expansion")
    return tuple(trials)


# ------------------------------------------------------------- drift guarding


def assert_only_treatment_differs(plan: ExperimentPlan) -> None:
    """The guard that makes the whole thing an ablation rather than two runs.

    Compares every arm's resolved configuration against every other's and
    refuses any difference that is not a declared factor.  The three projections
    exist precisely so this comparison has somewhere to put an expected
    difference.
    """

    declared = set(plan.arms[0].treatment) if plan.arms else set()
    for arm in plan.arms:
        if set(arm.treatment) != declared:
            raise ExperimentContractError(
                f"arm {arm.arm_id} varies {sorted(arm.treatment)}, but the experiment "
                f"declared {sorted(declared)}"
            )
    overlap = declared & set(plan.fixed)
    if overlap:
        raise ExperimentContractError(
            f"these paths are both fixed and treated as factors: {sorted(overlap)}"
        )
    per_trial_keys = {key for trial in plan.trials for key in trial.trial_derived}
    collision = per_trial_keys & declared
    if collision:
        raise ExperimentContractError(
            f"these factors are also derived per trial, so an arm difference could not be "
            f"attributed: {sorted(collision)}"
        )


def diff_projections(plan: ExperimentPlan, left: str, right: str) -> dict[str, Any]:
    """Human-readable three-way diff between two arms, for a report or the UI."""

    a, b = plan.arm(left), plan.arm(right)
    treatment = {
        path: {"left": a.treatment.get(path), "right": b.treatment.get(path)}
        for path in sorted(set(a.treatment) | set(b.treatment))
        if a.treatment.get(path) != b.treatment.get(path)
    }
    return {
        "fixed": plan.fixed,
        "treatment": treatment,
        "trial_derived_keys": sorted({key for trial in plan.trials for key in trial.trial_derived}),
        "subject": {"left": a.subject.to_json(), "right": b.subject.to_json()},
    }


__all__ = [
    "AA_BLOCKS",
    "ExperimentPlan",
    "assert_only_treatment_differs",
    "compile_plan",
    "diff_projections",
]
