"""`synth.experiment.v1`: the declarative spec an ablation is compiled from.

A spec says what varies and what must not.  It never says how to run anything:
the image, the seeds a target is allowed to see, the metrics, and the resource
ceilings all come from the executor's own trusted configuration, and the spec
may only select among what that configuration already declared.

That asymmetry is the point.  It is what makes an ablation cheap to write and
still impossible to accidentally turn into an unfair comparison.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import (
    EXPERIMENT_SPEC_SCHEMA,
    MISSING_POLICIES,
    ExperimentContractError,
    _identifier,
    _object,
    _text,
    digest_of,
)

PAIRING_MODES = ("block", "none")
BLOCK_KINDS = ("seed", "task", "unit")
CACHE_ISOLATION = ("per_trial", "per_arm", "shared")
CONTAINER_ISOLATION = ("fresh_per_trial", "reused")


@dataclass(frozen=True, slots=True)
class DesignSpec:
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    pairing: str
    counterbalance: bool
    missing_policy: str
    min_blocks_for_claim: int
    min_completion_rate: float
    max_differential_missing_blocks: int
    max_position_bias: float
    confidence: float
    bootstrap_resamples: int

    def to_json(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "pairing": self.pairing,
            "counterbalance": self.counterbalance,
            "missing_policy": self.missing_policy,
            "min_blocks_for_claim": self.min_blocks_for_claim,
            "min_completion_rate": self.min_completion_rate,
            "max_differential_missing_blocks": self.max_differential_missing_blocks,
            "max_position_bias": self.max_position_bias,
            "confidence": self.confidence,
            "bootstrap_resamples": self.bootstrap_resamples,
        }


@dataclass(frozen=True, slots=True)
class BlockSpec:
    kind: str
    values: tuple[Any, ...]
    replicates: int

    def block_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.kind}:{value}" for value in self.values)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": list(self.values), "replicates": self.replicates}


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    max_trials: int | None
    max_cost_usd: float | None
    max_wall_minutes: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "max_trials": self.max_trials,
            "max_cost_usd": self.max_cost_usd,
            "max_wall_minutes": self.max_wall_minutes,
        }


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """How many trials may be in flight at once.

    Serial dispatch is the default because it makes the planned order the
    executed order. Raising this is safe for a *paired* design -- both arms of a
    block meet the same machine at the same moment, which removes temporal drift
    rather than adding it -- but it does put arms in contention for CPU and
    provider quota, so the reducer keeps measuring the order that actually
    happened rather than trusting this number.
    """

    max_parallel_trials: int

    def to_json(self) -> dict[str, Any]:
        return {"max_parallel_trials": self.max_parallel_trials}


@dataclass(frozen=True, slots=True)
class IsolationSpec:
    cache_namespace: str
    container: str

    def to_json(self) -> dict[str, Any]:
        return {"cache_namespace": self.cache_namespace, "container": self.container}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    executor: str
    base: str
    design: DesignSpec
    blocks: BlockSpec
    factors: dict[str, tuple[Any, ...]]
    fixed: dict[str, Any]
    budget: BudgetSpec
    isolation: IsolationSpec
    execution: ExecutionSpec = field(default_factory=lambda: ExecutionSpec(1))
    executor_options: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def arm_count(self) -> int:
        count = 1
        for values in self.factors.values():
            count *= len(values)
        return count

    @property
    def trial_count(self) -> int:
        return self.arm_count * len(self.blocks.values) * self.blocks.replicates

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": EXPERIMENT_SPEC_SCHEMA,
            "experiment_id": self.experiment_id,
            "executor": self.executor,
            "base": self.base,
            "design": self.design.to_json(),
            "blocks": self.blocks.to_json(),
            "factors": {path: list(values) for path, values in self.factors.items()},
            "fixed": self.fixed,
            "budget": self.budget.to_json(),
            "isolation": self.isolation.to_json(),
            "execution": self.execution.to_json(),
            "executor_options": self.executor_options,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_json())


def load_spec(path: Path | str) -> ExperimentSpec:
    path = Path(path)
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ExperimentContractError(f"{path}: {error}") from error
    return parse_spec(payload, source=str(path))


def parse_spec(payload: Any, *, source: str | None = None) -> ExperimentSpec:
    data = _object(payload, context="experiment spec")
    schema = data.get("schema")
    if schema != EXPERIMENT_SPEC_SCHEMA:
        raise ExperimentContractError(
            f"experiment spec schema must be {EXPERIMENT_SPEC_SCHEMA}, got {schema!r}"
        )
    known = {
        "schema",
        "experiment_id",
        "executor",
        "base",
        "design",
        "blocks",
        "factors",
        "fixed",
        "budget",
        "isolation",
        "execution",
        "executor_options",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        # A silently ignored table is how an intended control becomes a
        # decoration, so an unrecognised key is a spec error rather than a
        # comment.
        raise ExperimentContractError(f"unknown top-level keys in experiment spec: {unknown}")

    design = _design(data.get("design"))
    blocks = _blocks(data.get("blocks"))
    factors = _factors(data.get("factors"))
    if not factors:
        raise ExperimentContractError(
            "an experiment must declare at least one factor; an experiment with no "
            "treatment is a benchmark run, not an ablation"
        )
    spec = ExperimentSpec(
        experiment_id=_identifier(data.get("experiment_id"), field_name="experiment_id"),
        executor=_identifier(data.get("executor"), field_name="executor"),
        base=_text(data.get("base"), field_name="base"),
        design=design,
        blocks=blocks,
        factors=factors,
        fixed=_object(data.get("fixed", {}), context="fixed"),
        budget=_budget(data.get("budget", {})),
        isolation=_isolation(data.get("isolation", {})),
        execution=_execution(data.get("execution", {})),
        executor_options=_object(data.get("executor_options", {}), context="executor_options"),
        source=source,
    )
    overlap = sorted(set(spec.fixed) & set(spec.factors))
    if overlap:
        raise ExperimentContractError(f"these paths are declared both fixed and varying: {overlap}")
    if design.min_blocks_for_claim > len(blocks.values):
        raise ExperimentContractError(
            f"design.min_blocks_for_claim ({design.min_blocks_for_claim}) exceeds the "
            f"{len(blocks.values)} blocks the spec declares, so no claim could ever pass"
        )
    if spec.budget.max_trials is not None and spec.trial_count > spec.budget.max_trials:
        raise ExperimentContractError(
            f"the matrix expands to {spec.trial_count} trials, over the declared "
            f"budget.max_trials of {spec.budget.max_trials}"
        )
    return spec


def _design(value: Any) -> DesignSpec:
    data = _object(value, context="design")
    pairing = data.get("pairing", "block")
    if pairing not in PAIRING_MODES:
        raise ExperimentContractError(f"design.pairing must be one of {PAIRING_MODES}")
    missing_policy = data.get("missing_policy", "fail")
    if missing_policy not in MISSING_POLICIES:
        raise ExperimentContractError(
            f"design.missing_policy must be one of {MISSING_POLICIES}; there is no "
            "imputation option in v0.7"
        )
    confidence = float(data.get("confidence", 0.95))
    if not 0.5 < confidence < 1.0:
        raise ExperimentContractError("design.confidence must be strictly between 0.5 and 1.0")
    completion = float(data.get("min_completion_rate", 0.9))
    if not 0.0 < completion <= 1.0:
        raise ExperimentContractError("design.min_completion_rate must be in (0, 1]")
    resamples = int(data.get("bootstrap_resamples", 10000))
    if resamples < 1000:
        raise ExperimentContractError("design.bootstrap_resamples must be at least 1000")
    position_bias = float(data.get("max_position_bias", 0.25))
    if not 0.0 <= position_bias <= 1.0:
        raise ExperimentContractError("design.max_position_bias must be in [0, 1]")
    secondary = data.get("secondary_metrics", []) or []
    if not isinstance(secondary, Sequence) or isinstance(secondary, (str, bytes)):
        raise ExperimentContractError("design.secondary_metrics must be a list")
    return DesignSpec(
        primary_metric=_identifier(data.get("primary_metric"), field_name="design.primary_metric"),
        secondary_metrics=tuple(
            _identifier(item, field_name="design.secondary_metrics[]") for item in secondary
        ),
        pairing=pairing,
        counterbalance=bool(data.get("counterbalance", True)),
        missing_policy=missing_policy,
        min_blocks_for_claim=int(data.get("min_blocks_for_claim", 0)),
        min_completion_rate=completion,
        max_differential_missing_blocks=int(data.get("max_differential_missing_blocks", 0)),
        max_position_bias=position_bias,
        confidence=confidence,
        bootstrap_resamples=resamples,
    )


def _blocks(value: Any) -> BlockSpec:
    data = _object(value, context="blocks")
    kind = data.get("kind", "seed")
    if kind not in BLOCK_KINDS:
        raise ExperimentContractError(f"blocks.kind must be one of {BLOCK_KINDS}")
    raw = data.get("values")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ExperimentContractError("blocks.values must be a non-empty list")
    values: list[Any] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ExperimentContractError("blocks.values entries must be integers or strings")
        values.append(item)
    if len(set(map(repr, values))) != len(values):
        raise ExperimentContractError("blocks.values must not repeat; a block is an identity")
    replicates = int(data.get("replicates", 1))
    if replicates < 1:
        raise ExperimentContractError("blocks.replicates must be at least 1")
    return BlockSpec(kind=kind, values=tuple(values), replicates=replicates)


def _factors(value: Any) -> dict[str, tuple[Any, ...]]:
    data = _object(value or {}, context="factors")
    factors: dict[str, tuple[Any, ...]] = {}
    for path, values in sorted(data.items()):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ExperimentContractError(
                f"factors.{path!r} must be a list of the values to compare"
            )
        if len(values) < 2:
            raise ExperimentContractError(
                f"factors.{path!r} needs at least two levels; one level is a fixed value "
                "and belongs in [fixed]"
            )
        if len(set(map(repr, values))) != len(values):
            raise ExperimentContractError(f"factors.{path!r} repeats a level")
        factors[path] = tuple(values)
    return factors


def _budget(value: Any) -> BudgetSpec:
    data = _object(value, context="budget")

    def optional_number(field_name: str) -> float | None:
        raw = data.get(field_name)
        if raw is None:
            return None
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
            raise ExperimentContractError(f"budget.{field_name} must be a positive number")
        return float(raw)

    max_trials = data.get("max_trials")
    if max_trials is not None and (
        not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 1
    ):
        raise ExperimentContractError("budget.max_trials must be a positive integer")
    return BudgetSpec(
        max_trials=max_trials,
        max_cost_usd=optional_number("max_cost_usd"),
        max_wall_minutes=optional_number("max_wall_minutes"),
    )


def _execution(value: Any) -> ExecutionSpec:
    data = _object(value, context="execution")
    parallel = data.get("max_parallel_trials", 1)
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
        raise ExperimentContractError("execution.max_parallel_trials must be a positive integer")
    return ExecutionSpec(max_parallel_trials=parallel)


def _isolation(value: Any) -> IsolationSpec:
    data = _object(value, context="isolation")
    cache = data.get("cache_namespace", "per_trial")
    if cache not in CACHE_ISOLATION:
        raise ExperimentContractError(f"isolation.cache_namespace must be one of {CACHE_ISOLATION}")
    container = data.get("container", "fresh_per_trial")
    if container not in CONTAINER_ISOLATION:
        raise ExperimentContractError(f"isolation.container must be one of {CONTAINER_ISOLATION}")
    return IsolationSpec(cache_namespace=cache, container=container)


__all__ = [
    "BLOCK_KINDS",
    "CACHE_ISOLATION",
    "CONTAINER_ISOLATION",
    "PAIRING_MODES",
    "BlockSpec",
    "BudgetSpec",
    "DesignSpec",
    "ExecutionSpec",
    "ExperimentSpec",
    "IsolationSpec",
    "load_spec",
    "parse_spec",
]
