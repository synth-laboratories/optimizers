"""Executor adapters.

Two ship here because their executors do: the local `eval` container runtime and
GEPA driven through its own config file. `evals.matrix` and `gepa.service` live
in other repositories and register themselves through `register_adapter` rather
than being imported across a repository boundary — the experiment layer must not
grow a dependency on every executor it can drive.
"""

from __future__ import annotations

from typing import Any

from ..models import ExperimentContractError
from .base import ExecutorAdapter, TrialContext
from .eval_runtime import EvalRuntimeAdapter
from .gepa_cli import GepaCliAdapter

#: Adapters resolvable from a spec's `executor` field.
REGISTRY: dict[str, Any] = {
    EvalRuntimeAdapter.executor_id: EvalRuntimeAdapter,
    GepaCliAdapter.executor_id: GepaCliAdapter,
}


def register_adapter(factory: Any, *, replace: bool = False) -> None:
    """Make an out-of-repo adapter resolvable by its `executor_id`.

    The factory must expose `executor_id` and a `from_spec(spec, **overrides)`
    classmethod; everything else is the `ExecutorAdapter` protocol and is checked
    by use, not here.

    Re-registering a name is refused unless `replace=True`. Two adapters
    answering to one `executor` would make which executor a sealed plan actually
    ran a function of import order, which is not a question a plan digest can
    answer.
    """

    executor_id = getattr(factory, "executor_id", None)
    if not isinstance(executor_id, str) or not executor_id.strip():
        raise ExperimentContractError("an adapter must declare a non-empty executor_id")
    if not callable(getattr(factory, "from_spec", None)):
        raise ExperimentContractError(
            f"adapter {executor_id!r} must expose from_spec(spec, **overrides)"
        )
    if executor_id in REGISTRY and not replace:
        raise ExperimentContractError(
            f"executor {executor_id!r} is already registered by "
            f"{REGISTRY[executor_id]!r}; pass replace=True to take it over"
        )
    REGISTRY[executor_id] = factory


__all__ = [
    "REGISTRY",
    "EvalRuntimeAdapter",
    "ExecutorAdapter",
    "GepaCliAdapter",
    "TrialContext",
    "register_adapter",
]
