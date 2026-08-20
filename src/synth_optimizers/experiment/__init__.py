"""`experiment`: a thin, declarative layer for controlled ablations.

It does three things and refuses the fourth.  It **assigns** — expanding a spec
into a frozen matrix of arms, blocks, and replicates with a materialised
dispatch order.  It **adapts** — handing each cell to an executor that already
exists, with a correlation envelope that survives into that executor's own
evidence.  It **reduces** — pairing sealed outcome rows by block and reporting
the comparison the design declared, including every reason the comparison does
not support a headline.

What it will not do is schedule, optimize, or keep a second copy of results.
`eval`, the matrix runner, and GEPA already own execution, and the moment this
layer starts owning it too, the two stories about what ran diverge.
"""

from __future__ import annotations

from .adapters import (
    REGISTRY,
    EvalRuntimeAdapter,
    ExecutorAdapter,
    GepaCliAdapter,
    TrialContext,
    register_adapter,
)
from .analysis import (
    ArmAggregate,
    ClaimVerdict,
    ExperimentReport,
    FairnessFacts,
    PairedComparison,
    reduce_experiment,
)
from .models import (
    CORRELATION_SCHEMA,
    EXPERIMENT_PLAN_SCHEMA,
    EXPERIMENT_REPORT_SCHEMA,
    EXPERIMENT_SPEC_SCHEMA,
    FACTOR_CATALOG_SCHEMA,
    TRIAL_OUTCOME_SCHEMA,
    AblatableFactor,
    ArmPlan,
    CorrelationEnvelope,
    ExperimentContractError,
    FactorCatalog,
    SubjectRef,
    TrialOutcome,
    TrialPlan,
    mint_trial_id,
)
from .outcomes import OutcomeLog, OutcomeSet, reduce_replicates
from .plan import (
    ExperimentPlan,
    assert_only_treatment_differs,
    compile_plan,
    diff_projections,
)
from .runner import ExperimentRunner, RunSummary
from .spec import ExperimentSpec, load_spec, parse_spec

__all__ = [
    "CORRELATION_SCHEMA",
    "EXPERIMENT_PLAN_SCHEMA",
    "EXPERIMENT_REPORT_SCHEMA",
    "EXPERIMENT_SPEC_SCHEMA",
    "FACTOR_CATALOG_SCHEMA",
    "REGISTRY",
    "TRIAL_OUTCOME_SCHEMA",
    "AblatableFactor",
    "ArmAggregate",
    "ArmPlan",
    "ClaimVerdict",
    "CorrelationEnvelope",
    "EvalRuntimeAdapter",
    "ExecutorAdapter",
    "ExperimentContractError",
    "ExperimentPlan",
    "ExperimentReport",
    "ExperimentRunner",
    "ExperimentSpec",
    "FactorCatalog",
    "FairnessFacts",
    "GepaCliAdapter",
    "OutcomeLog",
    "OutcomeSet",
    "PairedComparison",
    "RunSummary",
    "SubjectRef",
    "TrialContext",
    "TrialOutcome",
    "TrialPlan",
    "assert_only_treatment_differs",
    "compile_plan",
    "diff_projections",
    "load_spec",
    "mint_trial_id",
    "parse_spec",
    "reduce_experiment",
    "reduce_replicates",
    "register_adapter",
]
