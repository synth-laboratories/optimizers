"""The contract an executor implements to become ablatable.

An adapter is a translator, never a scheduler.  It answers four questions about
an existing executor — what may vary, what is held fixed, what the subject is,
and how a metric is oriented — and then runs one trial through that executor's
own queue, semaphore, and evidence path.

The experiment layer never learns how any executor works, which is what keeps
`eval`, the matrix runner, and GEPA on one honest comparison story without any
of them growing a second result store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..models import CorrelationEnvelope, FactorCatalog, SubjectRef, TrialOutcome

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..plan import ExperimentPlan, TrialPlan
    from ..spec import ExperimentSpec


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Everything an adapter is given to run exactly one cell of the matrix."""

    spec: ExperimentSpec
    plan: ExperimentPlan
    trial: TrialPlan
    treatment: dict[str, Any]
    fixed: dict[str, Any]
    correlation: CorrelationEnvelope
    workspace: Path
    dispatched_at: str


class ExecutorAdapter(Protocol):
    """Implemented once per executor.  Additive to that executor, never invasive."""

    executor_id: str

    def factor_catalog(self, spec: ExperimentSpec) -> FactorCatalog:
        """What this executor is prepared to defend as a fair treatment."""

    def provenance(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Digests that must not change mid-experiment: image, config, runtime.

        Anything returned here is folded into the plan digest, so a resume after
        one of them moves is refused instead of quietly mixing two measurements.
        """

    def environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Recorded, but outside the plan digest: host, checkout, tool versions."""

    def fixed_projection(self, spec: ExperimentSpec) -> dict[str, Any]:
        """The resolved inputs that must be identical in every arm."""

    def validate_blocks(self, spec: ExperimentSpec) -> None:
        """Refuse a block universe this executor cannot honour exactly."""

    def subject_for(self, spec: ExperimentSpec, treatment: dict[str, Any]) -> SubjectRef:
        """The minimal durable reference to what this arm is testing."""

    def metric_direction(self, spec: ExperimentSpec, metric_id: str) -> str:
        """`maximize` or `minimize`, from the executor's own metric contract."""

    def trial_derived(
        self,
        spec: ExperimentSpec,
        *,
        arm_id: str,
        block_id: str,
        replicate: int,
        trial_id: str,
    ) -> dict[str, Any]:
        """Per-trial identifiers, paths, and cache namespace.

        These differ between trials by construction, which is exactly why they
        are held in their own projection: a diff that lands here is expected, and
        a diff that lands anywhere else is a bug.
        """

    def run_trial(self, context: TrialContext) -> TrialOutcome:
        """Execute one trial through the executor and seal its outcome row."""


__all__ = ["ExecutorAdapter", "TrialContext"]
