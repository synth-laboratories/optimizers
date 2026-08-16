"""`eval`: a local Optimizers algorithm that scores policy candidates.

`eval` takes an immutable set of policy candidates and an allowlisted recipe,
expands them into a fair `candidate × seed × scenario` matrix, runs each trial
in the recipe's pinned container, and returns a promotable winner only when the
declared gates pass.

It is deliberately local. `eval` is not a hosted algorithm and must never be
added to `synth_optimizers.hosted.OptimizerAlgorithmSlug` or to
`future_algorithms.py`: that enum describes hosted API compatibility, which
this is not.

There are no Craftax-, GameBench-, or Harbor-specific execution paths here.
Those are containers implementing one standard contract, `eval.target.v1`.
"""

from __future__ import annotations

from .executor import (
    ContainerRuntimeError,
    OciTrialExecutor,
    TrialExecution,
    TrialExecutor,
    TrialRunRequest,
)
from .home import EvalHome, RuntimeConfig
from .models import (
    BENCHMARK_STATUSES,
    CANDIDATE_SET_SCHEMA,
    CONTAINER_RESULT_SCHEMA,
    CONTAINER_STATUSES,
    EVAL_ALGORITHM_ID,
    EVAL_ALGORITHM_VERSION,
    POLICY_CANDIDATE_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    SEED_LEDGER_SCHEMA,
    SELECTION_STATUSES,
    TARGET_MANIFEST_SCHEMA,
    TRIAL_MANIFEST_SCHEMA,
    TRIAL_RECORD_SCHEMA,
    TRIAL_STATUSES,
    WORKER_EVENT_SCHEMA,
    WORKER_MANIFEST_SCHEMA,
    CandidateScorecard,
    CandidateSet,
    ContainerResult,
    EvalContractError,
    MetricSpec,
    PolicyCandidate,
    SeedLedger,
    SelectionDecision,
    SelectionSpec,
    TargetManifest,
    TrialKey,
    TrialLimits,
    TrialRecord,
    digest_of,
    digest_of_tree,
)
from .recipes import EvalRecipe, catalog, get_recipe
from .runner import EvalRunner, WorkerManifest, request_cancel, run_worker
from .scoring import apply_elimination, decide, paired_lift, summarize_candidate
from .semaphore import Lease, SemaphoreTimeout, TrialSemaphore

__all__ = [
    "BENCHMARK_STATUSES",
    "CANDIDATE_SET_SCHEMA",
    "CONTAINER_RESULT_SCHEMA",
    "CONTAINER_STATUSES",
    "EVAL_ALGORITHM_ID",
    "EVAL_ALGORITHM_VERSION",
    "POLICY_CANDIDATE_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "SEED_LEDGER_SCHEMA",
    "SELECTION_STATUSES",
    "TARGET_MANIFEST_SCHEMA",
    "TRIAL_MANIFEST_SCHEMA",
    "TRIAL_RECORD_SCHEMA",
    "TRIAL_STATUSES",
    "WORKER_EVENT_SCHEMA",
    "WORKER_MANIFEST_SCHEMA",
    "CandidateScorecard",
    "CandidateSet",
    "ContainerResult",
    "ContainerRuntimeError",
    "EvalContractError",
    "EvalHome",
    "EvalRecipe",
    "EvalRunner",
    "Lease",
    "MetricSpec",
    "OciTrialExecutor",
    "PolicyCandidate",
    "RuntimeConfig",
    "SeedLedger",
    "SelectionDecision",
    "SelectionSpec",
    "SemaphoreTimeout",
    "TargetManifest",
    "TrialExecution",
    "TrialExecutor",
    "TrialKey",
    "TrialLimits",
    "TrialRecord",
    "TrialRunRequest",
    "TrialSemaphore",
    "WorkerManifest",
    "apply_elimination",
    "catalog",
    "decide",
    "digest_of",
    "digest_of_tree",
    "get_recipe",
    "paired_lift",
    "request_cancel",
    "run_worker",
    "summarize_candidate",
]
