"""Adapter for GEPA driven through its own config file.

One rendered TOML per trial, run through the same `GepaRun` path a human would
use. The experiment layer adds no scheduler and no second result store: GEPA
keeps its run directory, its manifest, its run registry, and its event feed, and
this reads the terminal record it already writes.

The factor catalog here is **curated, not reflected**. GEPA's TOML sections are
`extra="ignore"` by construction — the `[dataset]`-versus-`[taskset]` trap in the
public cookbooks is exactly what that costs — so a schema walk would happily
offer knobs the engine silently drops, and would offer knobs that are in the
schema but destroy the comparison when varied. Publishing an explicit list makes
adding a treatment a deliberate act. Every rendered arm is then validated
through GEPA's own typed document model before a single run starts, so a value
the engine would reject is a plan-time error rather than a wasted generation.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import (
    AblatableFactor,
    ExperimentContractError,
    FactorCatalog,
    SubjectRef,
    TrialOutcome,
    canonical_json,
    digest_of,
)
from .base import TrialContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..spec import ExperimentSpec

#: What GEPA is prepared to defend as a fair treatment.
#:
#: Deliberately short. `gepa.generations`, `gepa.minibatch_size`, and the budget
#: caps are all in the schema and all change how much work an arm does, which
#: makes a wall-clock or cost comparison between arms void; they are held fixed
#: instead. What is left varies *how* the proposer thinks, not *how much* the
#: run runs.
ABLATABLE_FACTORS: tuple[AblatableFactor, ...] = (
    AblatableFactor(
        path="proposer.model",
        kind="string",
        description="Which proposer model authors candidates.",
    ),
    AblatableFactor(
        path="proposer.reasoning_effort",
        kind="enum",
        description="Proposer reasoning effort.",
        values=("none", "low", "medium", "high"),
    ),
    AblatableFactor(
        path="proposer.service_tier",
        kind="enum",
        description="Codex app-server service tier.",
        values=("default", "fast"),
    ),
    AblatableFactor(
        path="proposer.backend",
        kind="string",
        description="Proposer backend implementation.",
    ),
    AblatableFactor(
        path="gepa.pipeline.mode",
        kind="enum",
        description="Scheduler mode: how propose and rollout lanes interleave.",
        values=("sync_serial", "async_pipelined", "flash_evolve"),
    ),
    AblatableFactor(
        path="jesterky_workflow.enabled",
        kind="bool",
        description="Whether annotations are produced as evidence for this run.",
    ),
    AblatableFactor(
        path="policy.model",
        kind="string",
        description="Which policy model executes rollouts.",
    ),
)

#: Sections the experiment layer owns per trial. A spec that names one of these
#: as a factor would make an arm difference indistinguishable from bookkeeping.
RESERVED_PATHS = ("run.run_id", "run.output_dir", "run.seed", "run.correlation", "cache.namespace")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _get(document: dict[str, Any], path: str) -> Any:
    node: Any = document
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = document
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


class GepaCliAdapter:
    """Runs one GEPA config per cell of the matrix."""

    executor_id = "gepa.cli"

    def __init__(
        self,
        *,
        base: Path | str,
        output_dir: Path | str | None = None,
        runner: Callable[[Path], Any] | None = None,
    ) -> None:
        self.base_path = Path(base).expanduser().resolve()
        if not self.base_path.is_file():
            raise ExperimentContractError(f"no GEPA config at {self.base_path}")
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.base_path.parent / "runs"
        )
        self._runner = runner
        self._base_document: dict[str, Any] | None = None

    @classmethod
    def from_spec(cls, spec: ExperimentSpec, **overrides: Any) -> GepaCliAdapter:
        options: dict[str, Any] = {"base": spec.base, **spec.executor_options, **overrides}
        unknown = sorted(set(options) - {"base", "output_dir", "runner"})
        if unknown:
            raise ExperimentContractError(f"executor_options does not accept {unknown}")
        return cls(**options)

    def base_document(self) -> dict[str, Any]:
        if self._base_document is None:
            try:
                self._base_document = tomllib.loads(self.base_path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as error:
                raise ExperimentContractError(f"{self.base_path}: {error}") from error
        return self._base_document

    # ------------------------------------------------------------- the catalog

    def factor_catalog(self, spec: ExperimentSpec) -> FactorCatalog:
        return FactorCatalog(
            executor=self.executor_id,
            base_ref=str(self.base_path),
            factors=ABLATABLE_FACTORS,
        )

    # ------------------------------------------------------------- projections

    def fixed_projection(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Everything the base config pins, minus what this trial derives.

        The whole base document goes in rather than a hand-picked subset: the
        guard's job is to catch an inter-arm difference nobody declared, and it
        cannot catch what it was never shown.
        """

        document = json.loads(json.dumps(self.base_document()))
        for path in RESERVED_PATHS:
            parts = path.split(".")
            node = document
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node.get(part), dict) else {}
            if isinstance(node, dict):
                node.pop(parts[-1], None)
        for path in spec.factors:
            parts = path.split(".")
            node = document
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node.get(part), dict) else {}
            if isinstance(node, dict):
                node.pop(parts[-1], None)
        return {"base_config": document, "base_config_path": str(self.base_path)}

    def provenance(self, spec: ExperimentSpec) -> dict[str, Any]:
        document = self.base_document()
        return {
            "executor": self.executor_id,
            "base_config_path": str(self.base_path),
            "base_config_digest": digest_of(document),
            "container_url": _get(document, "container.url"),
            "taskset_digest": digest_of(document.get("taskset", {})),
            "task_pools_digest": digest_of(_get(document, "gepa.task_pools") or {}),
        }

    def environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        from ... import __version__ as optimizers_version

        return {"optimizers_version": optimizers_version, "output_dir": str(self.output_dir)}

    def validate_blocks(self, spec: ExperimentSpec) -> None:
        if spec.blocks.kind != "seed":
            raise ExperimentContractError(
                f"{self.executor_id} blocks on `[run] seed`; blocks.kind "
                f"{spec.blocks.kind!r} has no meaning for it"
            )
        for value in spec.blocks.values:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExperimentContractError("blocks.values must be integer seeds for gepa.cli")
        reserved = sorted(set(spec.factors) & set(RESERVED_PATHS))
        if reserved:
            raise ExperimentContractError(
                f"these paths are derived per trial and cannot be treatments: {reserved}"
            )
        # Prove every arm renders into a config GEPA itself accepts, before any
        # run starts. A value the engine would reject is a plan-time error, not
        # a wasted generation.
        catalog = self.factor_catalog(spec)
        for path, values in spec.factors.items():
            for value in values:
                self._validate_document(
                    self._render(spec, {path: catalog.normalize(path, value)}, trial=None)
                )

    def subject_for(self, spec: ExperimentSpec, treatment: dict[str, Any]) -> SubjectRef:
        """The resolved proposer policy is what a GEPA outer experiment tests.

        GEPA's own prompt and code candidates keep their existing candidate ids
        and parent lineage inside the run; they are a different, inner subject
        and this does not attempt to stand in for them.
        """

        document = self._render(spec, treatment, trial=None)
        proposer = document.get("proposer", {})
        model = proposer.get("model") or "unset"
        effort = proposer.get("reasoning_effort") or "unset"
        return SubjectRef(
            subject_kind="proposer-policy",
            subject_id=f"{model}@{effort}",
            subject_content_digest=digest_of(proposer),
        )

    def metric_direction(self, spec: ExperimentSpec, metric_id: str) -> str:
        """GEPA reports rewards and costs; a cost or a duration improves downward."""

        lowered = metric_id.lower()
        if any(token in lowered for token in ("cost", "usd", "wall", "seconds", "latency")):
            return "minimize"
        return "maximize"

    def trial_derived(
        self,
        spec: ExperimentSpec,
        *,
        arm_id: str,
        block_id: str,
        replicate: int,
        trial_id: str,
    ) -> dict[str, Any]:
        run_id = f"gepa_{trial_id}"
        return {
            "run_id": run_id,
            "cache_namespace": run_id,
            "seed": int(block_id.split(":", 1)[1]),
            "output_dir": str(self.output_dir / trial_id),
            "config_path": str(self.output_dir / trial_id / "config.toml"),
        }

    # -------------------------------------------------------------- rendering

    def _render(
        self,
        spec: ExperimentSpec,
        treatment: dict[str, Any],
        *,
        trial: dict[str, Any] | None,
        correlation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document = json.loads(json.dumps(self.base_document()))
        for path, value in sorted({**spec.fixed, **treatment}.items()):
            _set(document, path, value)
        if trial is not None:
            _set(document, "run.run_id", trial["run_id"])
            _set(document, "run.output_dir", trial["output_dir"])
            _set(document, "run.seed", trial["seed"])
            _set(document, "cache.namespace", trial["cache_namespace"])
            if correlation is not None:
                _set(document, "run.correlation", correlation)
        return document

    def _validate_document(self, document: dict[str, Any]) -> None:
        from ...gepa import GepaTomlDocument

        try:
            GepaTomlDocument.model_validate(document)
        except Exception as error:
            raise ExperimentContractError(
                f"the rendered arm is not a valid GEPA config: {error}"
            ) from error

    # -------------------------------------------------------------- execution

    def run_trial(self, context: TrialContext) -> TrialOutcome:
        from ...gepa import _toml_dumps

        trial = dict(context.trial.trial_derived)
        correlation = context.correlation.to_json()
        document = self._render(
            context.spec, context.treatment, trial=trial, correlation=correlation
        )
        self._validate_document(document)
        config_path = Path(trial["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_toml_dumps(document), encoding="utf-8")

        started_at = _now()
        failure: str | None = None
        try:
            self._run(config_path)
        except Exception as error:  # noqa: BLE001 - any engine failure is one trial's failure
            failure = f"{type(error).__name__}: {error}"
        finished_at = _now()
        return self._seal(
            context,
            config_path=config_path,
            trial=trial,
            failure=failure,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _run(self, config_path: Path) -> Any:
        if self._runner is not None:
            return self._runner(config_path)
        from ...gepa import GepaRun

        return GepaRun.from_toml(config_path).execute()

    def _seal(
        self,
        context: TrialContext,
        *,
        config_path: Path,
        trial: dict[str, Any],
        failure: str | None,
        started_at: str,
        finished_at: str,
    ) -> TrialOutcome:
        run_dir = Path(trial["output_dir"]) / trial["run_id"]
        manifest_path = run_dir / "result_manifest.json"
        base = {
            "experiment_id": context.plan.experiment_id,
            "trial_id": context.trial.trial_id,
            "arm_id": context.trial.arm_id,
            "block_id": context.trial.block_id,
            "replicate": context.trial.replicate,
            "executor": self.executor_id,
            "executor_run_id": trial["run_id"],
            "correlation_digest": context.correlation.digest,
            "dispatched_at": context.dispatched_at,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        infra = {
            "cache_namespace": trial["cache_namespace"],
            "config_path": str(config_path),
            "config_digest": digest_of(
                tomllib.loads(config_path.read_text(encoding="utf-8"))
                if config_path.is_file()
                else {}
            ),
            "evidence_dir": str(run_dir),
        }
        evidence = {
            "result_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "events": str(run_dir / "events.jsonl"),
            "run_registry": str(Path(trial["output_dir"]) / "run_registry.jsonl"),
        }
        if not manifest_path.is_file():
            return TrialOutcome(
                **base,
                status="failed",
                metrics={},
                usage={},
                failure_class="infra" if failure else "rig",
                failure_detail=failure or "the run sealed no result manifest",
                receipt_digest=None,
                evidence=evidence,
                infra=infra,
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._assert_manifest_is_ours(manifest, context)
        receipt = digest_of(manifest)
        status = str(manifest.get("status") or ("failed" if failure else "completed"))
        metrics = _metrics_from_manifest(manifest)
        cost = manifest.get("cost_usd")
        usage = {"gepa_usage": manifest.get("usage")}
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            usage["cost_usd"] = float(cost)
        if status in ("completed", "succeeded") and not failure:
            return TrialOutcome(
                **base,
                status="completed",
                metrics=metrics,
                usage=usage,
                failure_class=None,
                failure_detail=None,
                receipt_digest=receipt,
                evidence=evidence,
                infra=infra,
            )
        return TrialOutcome(
            **base,
            status="failed",
            metrics={},
            usage=usage,
            failure_class="rig",
            failure_detail=failure or json.dumps(manifest.get("failure"))[:400],
            receipt_digest=receipt,
            evidence=evidence,
            infra=infra,
        )

    def _assert_manifest_is_ours(self, manifest: dict[str, Any], context: TrialContext) -> None:
        """The envelope must come back byte-identical, or the join is a guess.

        The digest is recomputed here rather than transmitted, so the two sides
        never have to agree on JSON canonicalisation across a language boundary —
        only on the content.
        """

        echoed = manifest.get("correlation")
        if echoed is None:
            raise ExperimentContractError(
                f"trial {context.trial.trial_id}: the run manifest carries no correlation "
                "envelope, so its evidence cannot be attributed to an arm"
            )
        if digest_of(echoed) != context.correlation.digest:
            raise ExperimentContractError(
                f"trial {context.trial.trial_id}: the run manifest echoed a different "
                f"correlation envelope than the plan minted\n"
                f"  sent: {canonical_json(context.correlation.to_json())}\n"
                f"  back: {canonical_json(echoed)}"
            )


def _metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    """Lift GEPA's terminal scalars into a flat metric map.

    Only what GEPA itself sealed. `train_exploitation` and `eval_uplift` are
    metrics of the separate GEPA-as-container proposer experiment and are not
    invented here for a run that never reported them.
    """

    metrics: dict[str, float] = {}
    best = manifest.get("best_candidate")
    if isinstance(best, dict):
        for key in ("heldout_reward", "train_reward", "minibatch_reward"):
            value = best.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
    cost = manifest.get("cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        metrics["cost_usd"] = float(cost)
    usage = manifest.get("usage")
    if isinstance(usage, dict):
        for key in ("rollouts", "total_tokens", "wall_time_seconds"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
    return metrics


__all__ = ["ABLATABLE_FACTORS", "RESERVED_PATHS", "GepaCliAdapter"]
