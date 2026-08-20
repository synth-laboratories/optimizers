"""V5 annotation projection for sealed local ``eval`` trial evidence.

The evaluator keeps the target container as the authority for benchmark facts.
This module only archives already-sealed trial records as Trace V5 and attaches
descriptive Jesterky output in append-only V5 evidence bundles.  In particular,
an annotation never becomes a metric, gate, or selection input.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from synth_containers.tracing import (
    AnnotationInspectionV1,
    AnnotationStatus,
    AnnotationV1,
    GroundingStatus,
    ProducerRefV1,
    ReceiptV1,
    TraceAnnotatorDefinitionV1,
    TraceCompletenessV5,
    TraceDocumentV5,
    TraceEvidenceBundleV5,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
    TraceRefV5,
    TraceStatus,
    selector_for,
    utc_now,
)
from synth_containers.tracing.models.completeness import TraceLifecycleV5
from synth_containers.tracing.models.document import TraceCaptureSummaryV5

from .models import EvalContractError, TrialRecord, canonical_json, digest_of, write_json


ANNOTATION_POLICY_SCHEMA = "eval.annotation-policy.v1"
ANNOTATION_RECEIPT_SCHEMA = "eval.annotation-receipt.v1"
ANNOTATION_INDEX_SCHEMA = "eval.annotation-index.v1"
ANNOTATION_CADENCE = "sealed_matrix_before_selection"


@dataclass(frozen=True, slots=True)
class AnnotationPolicy:
    """Immutable, app-owned annotation policy carried by a worker manifest."""

    enabled: bool = False
    command: str = "jesterky"
    spec: str = "examples/gepa_trace_annotate.json"
    actor: str = "codex"
    model: str | None = None
    provider: str = "chatgpt"
    max_targets: int = 4
    cadence: str = ANNOTATION_CADENCE
    deduplicate_by_trace_digest: bool = True
    max_spend_usd: float = 1.0
    timeout_seconds: int = 600
    fail_closed: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "AnnotationPolicy":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise EvalContractError("annotation_policy must be a JSON object")
        schema = value.get("schema_version", ANNOTATION_POLICY_SCHEMA)
        if schema != ANNOTATION_POLICY_SCHEMA:
            raise EvalContractError(f"unsupported annotation policy schema {schema!r}")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise EvalContractError("annotation_policy.enabled must be boolean")
        policy = cls(
            enabled=enabled,
            command=str(value.get("command", "jesterky")).strip(),
            spec=str(value.get("spec", "examples/gepa_trace_annotate.json")).strip(),
            actor=str(value.get("actor", "codex")).strip(),
            model=(str(value["model"]).strip() if value.get("model") is not None else None),
            provider=str(value.get("provider", "chatgpt")).strip(),
            max_targets=value.get("max_targets", 4),
            cadence=str(value.get("cadence", ANNOTATION_CADENCE)).strip(),
            deduplicate_by_trace_digest=value.get("deduplicate_by_trace_digest", True),
            max_spend_usd=value.get("max_spend_usd", 1.0),
            timeout_seconds=value.get("timeout_seconds", 600),
            fail_closed=value.get("fail_closed", True),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.command:
            raise EvalContractError("annotation_policy.command must be non-empty when enabled")
        if not self.spec:
            raise EvalContractError("annotation_policy.spec must be non-empty when enabled")
        if self.actor not in {"fake", "codex"}:
            raise EvalContractError("annotation_policy.actor must be fake or codex when enabled")
        if not self.provider:
            raise EvalContractError("annotation_policy.provider must be non-empty when enabled")
        if not isinstance(self.max_targets, int) or isinstance(self.max_targets, bool):
            raise EvalContractError("annotation_policy.max_targets must be an integer")
        if not 1 <= self.max_targets <= 64:
            raise EvalContractError("annotation_policy.max_targets must be in [1, 64]")
        if self.cadence != ANNOTATION_CADENCE:
            raise EvalContractError(f"annotation_policy.cadence must be {ANNOTATION_CADENCE!r}")
        if (
            not isinstance(self.deduplicate_by_trace_digest, bool)
            or not self.deduplicate_by_trace_digest
        ):
            raise EvalContractError("annotation_policy must deduplicate by trace digest")
        if not isinstance(self.max_spend_usd, (int, float)) or isinstance(self.max_spend_usd, bool):
            raise EvalContractError("annotation_policy.max_spend_usd must be numeric")
        if float(self.max_spend_usd) <= 0:
            raise EvalContractError("annotation_policy.max_spend_usd must be positive")
        if not isinstance(self.timeout_seconds, int) or isinstance(self.timeout_seconds, bool):
            raise EvalContractError("annotation_policy.timeout_seconds must be an integer")
        if self.timeout_seconds <= 0:
            raise EvalContractError("annotation_policy.timeout_seconds must be positive")
        if not isinstance(self.fail_closed, bool):
            raise EvalContractError("annotation_policy.fail_closed must be boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_POLICY_SCHEMA,
            "enabled": self.enabled,
            "command": self.command,
            "spec": self.spec,
            "actor": self.actor,
            "model": self.model,
            "provider": self.provider,
            "max_targets": self.max_targets,
            "cadence": self.cadence,
            "deduplicate_by_trace_digest": self.deduplicate_by_trace_digest,
            "max_spend_usd": float(self.max_spend_usd),
            "timeout_seconds": self.timeout_seconds,
            "fail_closed": self.fail_closed,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_json())


@dataclass(frozen=True, slots=True)
class AnnotationReceipt:
    enabled: bool
    status: str
    reason: str | None
    trace_digests: tuple[str, ...] = ()
    evidence_bundle_paths: tuple[str, ...] = ()
    evidence_bundle_digests: tuple[str, ...] = ()
    annotated: int = 0
    blockers: int = 0
    themes: int = 0
    manifest_path: str | None = None
    elapsed_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_RECEIPT_SCHEMA,
            "enabled": self.enabled,
            "status": self.status,
            "reason": self.reason,
            "trace_digests": list(self.trace_digests),
            "evidence_bundle_paths": list(self.evidence_bundle_paths),
            "evidence_bundle_digests": list(self.evidence_bundle_digests),
            "annotated": self.annotated,
            "blockers": self.blockers,
            "themes": self.themes,
            "manifest_path": self.manifest_path,
            "elapsed_ms": self.elapsed_ms,
        }


class EvalAnnotationProjector:
    """Materialize V5 trace/evidence files from terminal evaluator records."""

    def __init__(self, *, policy: AnnotationPolicy, run_id: str, run_dir: Path) -> None:
        self.policy = policy
        self.run_id = run_id
        self.run_dir = run_dir
        self.trace_dir = run_dir / "trace_v5"
        self.bundle_dir = run_dir / "trace_evidence_v5"

    def project(self, records: Sequence[TrialRecord]) -> AnnotationReceipt:
        if not self.policy.enabled:
            return AnnotationReceipt(enabled=False, status="disabled", reason=None)
        started = time.monotonic()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        sealed = [self._seal_trial_trace(record) for record in records]
        selected = self._new_traces(sealed)
        if not selected:
            receipt = AnnotationReceipt(
                enabled=True,
                status="skipped",
                reason="no new sealed Trace V5 evidence matched the annotation policy",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            write_json(self.run_dir / "annotation_receipt.json", receipt.to_json())
            return receipt

        manifest_path = self.run_dir / "jesterky_eval_annotate.manifest.json"
        projection_dir = self.run_dir / "jesterky_traces"
        projection_dir.mkdir(parents=True, exist_ok=True)
        for trace in selected:
            projection = self._v4_projection(trace)
            write_json(projection_dir / f"{trace.trace_id}.json", projection)

        manifest: dict[str, Any] | None = None
        error: str | None = None
        try:
            self._run_jesterky(projection_dir, manifest_path)
            manifest = _read_object(manifest_path, "jesterky annotate manifest")
        except (OSError, subprocess.SubprocessError, EvalContractError) as exc:
            error = str(exc)
            if self.policy.fail_closed:
                raise EvalContractError(f"annotation policy failed closed: {error}") from exc

        registry = _theme_registry(manifest or {})
        scans = _scans(registry)
        bundles = [
            self._bundle_for(trace, scans.get(trace.trace_id), manifest_path, error)
            for trace in selected
        ]
        paths: list[str] = []
        digests: list[str] = []
        for bundle in bundles:
            path = self.bundle_dir / f"{bundle.bundle_id}.v5.json"
            write_json(path, bundle.to_dict())
            paths.append(str(path))
            digests.append(bundle.content_digest)
        self._write_index(sealed, selected, paths, digests)
        receipt = AnnotationReceipt(
            enabled=True,
            status="completed" if error is None else "failed_open",
            reason=error,
            trace_digests=tuple(item.content_digest for item in selected),
            evidence_bundle_paths=tuple(paths),
            evidence_bundle_digests=tuple(digests),
            annotated=sum(1 for item in selected if scans.get(item.trace_id) is not None),
            blockers=sum(bool(row.get("blocker")) for row in scans.values()),
            themes=len(registry.get("themes") or []),
            manifest_path=str(manifest_path) if manifest_path.is_file() else None,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        write_json(self.run_dir / "annotation_receipt.json", receipt.to_json())
        return receipt

    def _seal_trial_trace(self, record: TrialRecord) -> TraceDocumentV5:
        identity_key = {
            "run_id": self.run_id,
            "trial_id": record.trial_id,
            "record": record.to_json(),
        }
        # The IDs are deterministic in the same domain as the sealed record, so
        # re-running a worker cannot create a second trace for identical evidence.
        from synth_containers.tracing.models.identity import mint_capture_id, mint_trace_id

        trace_id = mint_trace_id(kind="evaluation_attempt", key=identity_key)
        path = self.trace_dir / f"{trace_id}.v5.json"
        if path.is_file():
            from synth_containers.tracing import rehydrate_trace

            existing = rehydrate_trace(_read_object(path, "sealed Trace V5"))
            return existing
        artifacts = [
            {
                "role": item.get("role"),
                "relative_path": item.get("relative_path"),
                "digest": item.get("digest"),
                "bytes": item.get("bytes"),
                "declared": item.get("declared"),
            }
            for item in record.artifacts
        ]
        lifecycle = "completed" if record.status == "evaluated" else "failed"
        document = TraceDocumentV5(
            trace_id=trace_id,
            trace_kind=TraceKind.EVALUATION_ATTEMPT,
            identity=TraceIdentityV5(
                run_id=self.run_id,
                trial_id=record.trial_id,
                task_id=record.key.scenario,
                seed=record.key.seed,
            ),
            lifecycle=TraceLifecycleV5(
                status=TraceStatus(lifecycle),
                started_at=record.started_at,
                ended_at=record.finished_at,
            ),
            capture=TraceCaptureSummaryV5(
                capture_id=mint_capture_id(trace_id=trace_id, key=record.trial_id),
                binding_id="eval.trial-record.v1",
                binding_digest=digest_of(record.to_json()),
                capture_profile="synth_optimizers.eval.trial_import.v1",
                interception="producer_event_import",
                mode="import",
                raw_record_count=len(artifacts),
            ),
            provenance=TraceProvenanceV5(
                producer="synth_optimizers.eval",
                producer_version="v5-annotation-projection.v1",
                source_format="eval.trial-result.v1",
                harness="eval.target.v1",
                captured_at=utc_now(),
                transformation_chain=("eval terminal trial record", "Trace V5 aggregate import"),
                extra={"trial_record_digest": digest_of(record.to_json())},
            ),
            completeness=TraceCompletenessV5(
                capture_status="partial",
                terminal_event_observed=True,
                model_calls="aggregate_only",
                raw_provider="unavailable",
                agent_events="aggregate_only",
                environment_events="aggregate_only",
                tool_events="aggregate_only",
                usage="aggregate_only",
                artifact_finalization="complete",
                reasons=(
                    "Imported from the eval terminal record; provider-native streaming is not claimed.",
                ),
            ),
            extensions={
                "synth_optimizers_eval_trial_import": {
                    "schema_version": "synth_optimizers.eval.trace_v5_import.v1",
                    "trial_record": record.to_json(),
                    "retained_artifacts": artifacts,
                }
            },
        ).sealed()
        write_json(path, document.to_dict())
        return document

    def _new_traces(self, traces: Sequence[TraceDocumentV5]) -> list[TraceDocumentV5]:
        seen: set[str] = set()
        index_path = self.run_dir / "annotation_index.json"
        if index_path.is_file():
            index = _read_object(index_path, "annotation index")
            seen = {
                str(item)
                for item in index.get(
                    "seen_trace_digests", index.get("annotated_trace_digests", [])
                )
            }
        return [item for item in traces if item.content_digest not in seen][
            : self.policy.max_targets
        ]

    def _v4_projection(self, trace: TraceDocumentV5) -> dict[str, Any]:
        return {
            "schema_version": "synth_rollout_trace_v4",
            "trace_schema_version": 4,
            "rollout_id": f"v5-{trace.trace_id}",
            "trace_correlation_id": trace.trace_id,
            "status": "completed",
            "spans": [
                {
                    "span_id": f"v5-{trace.trace_id}-projection",
                    "call_index": 1,
                    "run_id": trace.trace_id,
                    "request": {
                        "messages": [
                            {"role": "system", "content": "Trace V5 projection for eval annotation"}
                        ]
                    },
                    "response": {
                        "message": {"role": "assistant", "content": canonical_json(trace.to_dict())}
                    },
                }
            ],
            "events": [],
            "summary": {
                "source_trace_ref": {
                    "trace_id": trace.trace_id,
                    "content_digest": trace.content_digest,
                }
            },
            "metadata": {
                "source": "synth.trace.v5.jesterky_projection",
                "source_trace_ref": {
                    "trace_id": trace.trace_id,
                    "content_digest": trace.content_digest,
                },
                "projection_loss": "Jesterky currently reads a V4 transport file; sealed Trace V5 remains the annotation authority.",
            },
        }

    def _run_jesterky(self, projection_dir: Path, manifest_path: Path) -> None:
        spec = Path(self.policy.spec)
        if not spec.is_file():
            raise EvalContractError(f"annotation_policy.spec not found: {self.policy.spec}")
        args = json.dumps({"trace_dir": str(projection_dir), "artifact_dir": str(self.run_dir)})
        command = [
            self.policy.command,
            "run",
            str(spec),
            "--actor",
            self.policy.actor,
            "--args",
            args,
            "--out",
            str(manifest_path),
            "--cd",
            str(self.run_dir),
            "--run-id",
            f"eval-{self.run_id}-jesterky",
        ]
        if self.policy.model:
            command.extend(["--model", self.policy.model])
        completed = subprocess.run(
            command,
            cwd=self.run_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.policy.timeout_seconds,
            check=False,
        )
        if completed.returncode:
            raise EvalContractError(
                f"jesterky annotation failed (exit {completed.returncode}): {completed.stderr[-1000:]}"
            )
        if not manifest_path.is_file():
            raise EvalContractError("jesterky annotation completed without a manifest")

    def _bundle_for(
        self,
        trace: TraceDocumentV5,
        scan: dict[str, Any] | None,
        manifest_path: Path,
        error: str | None,
    ) -> TraceEvidenceBundleV5:
        config_digest = self.policy.digest
        definition = TraceAnnotatorDefinitionV1(
            annotator_id=f"jesterky-eval-{config_digest.split(':', 1)[-1][:16]}",
            name="Jesterky eval trace annotator",
            purpose="Produce descriptive evaluator context from a sealed Trace V5 projection.",
            taxonomy=("theme", "failure_mode", "reusable_rule", "blocker"),
            reasoning_policy="annotation policy; no benchmark or selection authority",
            grounding_requirement="selector",
            program_ref=self.policy.spec,
            model=self.policy.model,
            metadata={
                "provider": self.policy.provider,
                "policy_digest": config_digest,
                "cadence": self.policy.cadence,
                "max_targets": self.policy.max_targets,
                "deduplicate_by_trace_digest": True,
                "max_spend_usd": float(self.policy.max_spend_usd),
                "not_authoritative_for": ["evaluation_scores", "gates", "selection_decisions"],
            },
        ).sealed()
        target = selector_for(trace, kind="trace")
        theme_tags = tuple(str(item) for item in (scan or {}).get("theme_tags", []) if str(item))
        blocked = bool((scan or {}).get("blocker"))
        applied = scan is not None and bool(theme_tags) and not blocked and error is None
        projection_digest = digest_of(self._v4_projection(trace))
        annotation = AnnotationV1(
            annotation_id=f"ann_{digest_of([trace.content_digest, config_digest, str(manifest_path)])[-16:]}",
            annotator_id=definition.annotator_id,
            annotator_version=definition.version,
            annotator_digest=definition.content_digest,
            target=target,
            annotation_type="jesterky.trace_theme",
            labels=("theme",) if applied else (),
            author_kind="agentic",
            producer=ProducerRefV1(
                kind="agentic",
                name="jesterky",
                version="eval-v5-adapter.v1",
                model=self.policy.model,
                config_digest=config_digest,
            ),
            created_at=utc_now(),
            grounding=GroundingStatus.SUMMARY_ONLY
            if applied
            else GroundingStatus.SOURCE_UNAVAILABLE,
            payload=(
                {
                    "theme_tags": list(theme_tags),
                    "severity": (scan or {}).get("severity"),
                    "blocker": blocked,
                    "failure_modes": (scan or {}).get("failure_modes", []),
                    "reusable_rules": (scan or {}).get("reusable_rules", []),
                }
                if applied
                else {}
            ),
            rationale="Descriptive Jesterky output over a lossy V4 projection; never evaluator authority.",
            evidence=(target,) if applied else (),
            inspected_projection="jesterky.v4-projection",
            status=AnnotationStatus.APPLIED if applied else AnnotationStatus.ABSTAINED,
            review_state="unreviewed",
            abstention_reason=None
            if applied
            else (error or "Jesterky returned no usable descriptive labels"),
            inspection=AnnotationInspectionV1(
                source="projection",
                trace_body_read=False,
                projection_id="jesterky.v4-projection",
                projection_digest=projection_digest,
                projection_manifest_digest=_file_digest(manifest_path)
                if manifest_path.is_file()
                else None,
            ),
            annotator_execution_trace_id=(
                f"jesterky_manifest_{_file_digest(manifest_path).split(':', 1)[-1][:16]}"
                if manifest_path.is_file()
                else f"jesterky_policy_{config_digest.split(':', 1)[-1][:16]}"
            ),
            annotator_execution_trace_digest=(
                _file_digest(manifest_path) if manifest_path.is_file() else config_digest
            ),
        ).sealed()
        receipt = ReceiptV1(
            receipt_id=f"rcpt_{digest_of([trace.content_digest, config_digest, utc_now()])[-16:]}",
            operation="jesterky.annotate.trace_v5",
            status="completed" if error is None else "failed_open",
            started_at=utc_now(),
            ended_at=utc_now(),
            target_ids=(trace.trace_id,),
            producer=ProducerRefV1(
                kind="agentic",
                name="jesterky",
                model=self.policy.model,
                config_digest=config_digest,
            ),
            input_digests=(trace.content_digest, config_digest),
            output_digests=(annotation.content_digest,),
            completeness="complete" if error is None else "partial",
            errors=(error,) if error else (),
            detail={
                "max_spend_usd": float(self.policy.max_spend_usd),
                "spend_metering": "delegated_to_jesterky_actor",
            },
        ).sealed()
        return TraceEvidenceBundleV5(
            bundle_id=f"evb_{digest_of([trace.content_digest, config_digest, annotation.content_digest])[-16:]}",
            trace_ref=TraceRefV5(trace_id=trace.trace_id, content_digest=trace.content_digest),
            created_at=utc_now(),
            annotator_definitions=(definition,),
            annotations=(annotation,),
            receipts=(receipt,),
            metadata={"producer": "synth_optimizers.eval", "policy_digest": config_digest},
        ).sealed()

    def _write_index(
        self,
        sealed: Sequence[TraceDocumentV5],
        traces: Sequence[TraceDocumentV5],
        paths: Sequence[str],
        digests: Sequence[str],
    ) -> None:
        index_path = self.run_dir / "annotation_index.json"
        prior = _read_object(index_path, "annotation index") if index_path.is_file() else {}
        annotated = {str(item) for item in prior.get("annotated_trace_digests", [])}
        annotated.update(item.content_digest for item in traces)
        seen = {
            str(item)
            for item in prior.get("seen_trace_digests", prior.get("annotated_trace_digests", []))
        }
        # The policy makes one bounded selection from the sealed matrix.  On a
        # resume the remaining rows are not "new" evidence and cannot silently
        # become a second annotation batch after selection already completed.
        seen.update(item.content_digest for item in sealed)
        write_json(
            index_path,
            {
                "schema_version": ANNOTATION_INDEX_SCHEMA,
                "annotated_trace_digests": sorted(annotated),
                "seen_trace_digests": sorted(seen),
                "bundles": [
                    {
                        "trace_id": trace.trace_id,
                        "trace_digest": trace.content_digest,
                        "bundle_path": path,
                        "bundle_digest": digest,
                    }
                    for trace, path, digest in zip(traces, paths, digests, strict=True)
                ],
            },
        )


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalContractError(f"invalid {context} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalContractError(f"{context} at {path} must be a JSON object")
    return value


def _file_digest(path: Path) -> str:
    return "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def _theme_registry(manifest: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = [manifest.get("theme_registry")]
    trace = manifest.get("trace")
    if isinstance(trace, dict):
        outputs = trace.get("outputs")
        if isinstance(outputs, dict):
            candidates.extend(
                (
                    outputs.get("theme_registry"),
                    (outputs.get("summary") or {}).get("theme_registry"),
                )
            )
    for item in candidates:
        if isinstance(item, dict):
            return item
    return {"themes": [], "traces": []}


def _scans(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = registry.get("traces", registry.get("theme_matrix", []))
    if not isinstance(rows, list):
        return {}
    return {
        str(row["trace_id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("trace_id"), str)
    }


__all__ = [
    "ANNOTATION_CADENCE",
    "ANNOTATION_POLICY_SCHEMA",
    "AnnotationPolicy",
    "AnnotationReceipt",
    "EvalAnnotationProjector",
]
