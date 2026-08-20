"""Versioned wire records for controlled experiments.

The Platform owns these schemas and the identifiers in them.  A compiler, a CLI,
or Workshop may author a spec, but none of them may invent an id, a digest, or a
correlation envelope: those are minted here so that a trial found in a container
trace, in an optimizer run registry, and in an outcome row is provably the same
trial.

Nothing here executes anything.  It is deliberately stdlib-only so the record
layer stays usable from the app, from a worker, and from a report reducer that
has no optimizer runtime available.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EXPERIMENT_SPEC_SCHEMA = "synth.experiment.v1"
EXPERIMENT_PLAN_SCHEMA = "synth.experiment-plan.v1"
CORRELATION_SCHEMA = "synth.correlation.v1"
TRIAL_OUTCOME_SCHEMA = "synth.trial-outcome.v1"
FACTOR_CATALOG_SCHEMA = "synth.ablatable-factors.v1"
EXPERIMENT_REPORT_SCHEMA = "synth.experiment-report.v1"

#: Terminal states for one trial.  `skipped` is planned-but-never-dispatched and
#: is missingness, not a zero.
TRIAL_OUTCOME_STATUSES = ("completed", "failed", "timeout", "cancelled", "skipped")

#: How a trial failed, coarsely enough to be comparable across executors.  The
#: distinction that matters for a claim is `policy` (the thing under test) versus
#: everything else (the rig), because only the latter is differential missingness
#: an arm could be silently winning on.
FAILURE_CLASSES = (
    "policy",
    "rig",
    "infra",
    "budget",
    "timeout",
    "cancelled",
    "unknown",
)

#: The only failures a retry may re-dispatch.
#:
#: A crashed container or a dropped connection says nothing about the arm, so
#: re-running it is not cherry-picking.  A `policy` failure is the thing under
#: test, and a `budget` or `timeout` failure may itself be the arm difference --
#: retrying either would be selecting for the result you wanted.
RETRYABLE_FAILURE_CLASSES = frozenset({"rig", "infra"})

#: v0.7 missingness policies.  There is no imputation option, on purpose.
MISSING_POLICIES = ("fail", "pairwise_complete")

#: The only alias keys a downstream index is asked to carry.  Everything else in
#: the envelope lives in the sealed evidence record, not in an index.
ALIAS_KEYS = ("experiment_id", "trial_id", "candidate_id")

#: `@` is allowed because a subject is routinely a versioned reference —
#: `gpt-5.6-luna@low`, `image@sha256:…`. `/` is not: several of these ids become
#: directory names downstream.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExperimentContractError(ValueError):
    """A record refused rather than defaulted."""


# --------------------------------------------------------------------- helpers


def canonical_json(value: Any) -> str:
    """One byte string per value, so a digest means the same thing everywhere."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentContractError(f"{context} must be an object")
    return dict(value)


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if not _ID_PATTERN.match(text):
        raise ExperimentContractError(f"{field_name} is not a valid identifier: {text!r}")
    return text


def _digest(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if not _DIGEST_PATTERN.match(text):
        raise ExperimentContractError(f"{field_name} must be a sha256:<hex> digest")
    return text


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExperimentContractError(f"{field_name} must be a non-negative integer")
    return value


# --------------------------------------------------------------------- subject


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """The minimal durable reference to the thing an arm is testing.

    A full candidate revision registry can follow; correlation cannot wait for
    it.  `subject_content_digest` is what makes two arms comparable — a label is
    presentation, a digest is identity.
    """

    subject_kind: str
    subject_id: str
    subject_content_digest: str
    parent_subject_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> SubjectRef:
        data = _object(value, context="subject")
        parent = data.get("parent_subject_id")
        return cls(
            subject_kind=_identifier(data.get("subject_kind"), field_name="subject.subject_kind"),
            subject_id=_identifier(data.get("subject_id"), field_name="subject.subject_id"),
            subject_content_digest=_digest(
                data.get("subject_content_digest"),
                field_name="subject.subject_content_digest",
            ),
            parent_subject_id=(
                _identifier(parent, field_name="subject.parent_subject_id")
                if parent is not None
                else None
            ),
        )

    def to_json(self) -> dict[str, Any]:
        payload = {
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "subject_content_digest": self.subject_content_digest,
        }
        if self.parent_subject_id is not None:
            payload["parent_subject_id"] = self.parent_subject_id
        return payload


# ----------------------------------------------------------------- correlation


@dataclass(frozen=True, slots=True)
class CorrelationEnvelope:
    """What every executor, rollout, and trace must carry back unchanged.

    It deliberately does not contain a run id.  A service mints its own run id
    and reports it; a caller that supplied one would be asserting authority over
    a namespace it does not own, and resume would then have two candidate
    truths.
    """

    experiment_id: str
    arm_id: str
    block_id: str
    replicate: int
    trial_id: str
    plan_digest: str
    subject: SubjectRef
    candidate_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> CorrelationEnvelope:
        data = _object(value, context="correlation")
        schema = data.get("schema_version", CORRELATION_SCHEMA)
        if schema != CORRELATION_SCHEMA:
            raise ExperimentContractError(f"unsupported correlation schema {schema!r}")
        candidate_id = data.get("candidate_id")
        return cls(
            experiment_id=_identifier(data.get("experiment_id"), field_name="experiment_id"),
            arm_id=_identifier(data.get("arm_id"), field_name="arm_id"),
            block_id=_identifier(data.get("block_id"), field_name="block_id"),
            replicate=_non_negative_int(data.get("replicate"), field_name="replicate"),
            trial_id=_identifier(data.get("trial_id"), field_name="trial_id"),
            plan_digest=_digest(data.get("plan_digest"), field_name="plan_digest"),
            subject=SubjectRef.from_mapping(data.get("subject")),
            candidate_id=(
                _identifier(candidate_id, field_name="candidate_id")
                if candidate_id is not None
                else None
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """Absent keys, never null ones.

        The envelope has to survive being written into an executor's own config
        format and read back byte-identical, and TOML has no null. An optional
        field that is present-and-empty here would silently vanish on the round
        trip and take the digest with it.
        """

        payload: dict[str, Any] = {
            "schema_version": CORRELATION_SCHEMA,
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "block_id": self.block_id,
            "replicate": self.replicate,
            "trial_id": self.trial_id,
            "plan_digest": self.plan_digest,
            "subject": self.subject.to_json(),
        }
        if self.candidate_id is not None:
            payload["candidate_id"] = self.candidate_id
        return payload

    @property
    def digest(self) -> str:
        return digest_of(self.to_json())

    def aliases(self) -> dict[str, str]:
        """The bounded subset an index is allowed to carry.

        Indexing every factor would make the trace index a second, divergent copy
        of the plan.  Three keys are enough to get from a trace back to the
        sealed envelope, which is the authority.
        """

        alias = {"experiment_id": self.experiment_id, "trial_id": self.trial_id}
        if self.candidate_id:
            alias["candidate_id"] = self.candidate_id
        return alias


# --------------------------------------------------------------------- factors


@dataclass(frozen=True, slots=True)
class AblatableFactor:
    """One knob an executor is prepared to defend as a fair treatment.

    Being present in a config schema is not the same as being ablatable: a
    timeout is in the schema and varying it across arms silently changes the
    measurement.  Only what an executor publishes here may become a treatment.
    """

    path: str
    kind: str
    description: str
    values: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    redact: bool = False
    ui_exposed: bool = True

    def normalize(self, value: Any) -> Any:
        """Coerce to the canonical form, or refuse.  Never widen the allowlist."""

        if self.kind == "enum":
            if value not in self.values:
                allowed = ", ".join(repr(item) for item in self.values)
                raise ExperimentContractError(
                    f"factor {self.path} does not accept {value!r}; allowed: {allowed}"
                )
            return value
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ExperimentContractError(f"factor {self.path} must be a boolean")
            return value
        if self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExperimentContractError(f"factor {self.path} must be an integer")
            self._bounded(float(value))
            return value
        if self.kind == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ExperimentContractError(f"factor {self.path} must be a number")
            self._bounded(float(value))
            return float(value)
        if self.kind == "string":
            return _text(value, field_name=f"factor {self.path}")
        raise ExperimentContractError(f"factor {self.path} has unknown kind {self.kind!r}")

    def _bounded(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            raise ExperimentContractError(f"factor {self.path} is below its minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ExperimentContractError(f"factor {self.path} is above its maximum {self.maximum}")
        return True

    def display(self, value: Any) -> Any:
        return "<redacted>" if self.redact else value

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
            "values": list(self.values),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "redact": self.redact,
            "ui_exposed": self.ui_exposed,
        }


@dataclass(frozen=True, slots=True)
class FactorCatalog:
    """What one executor, for one base config, is willing to vary."""

    executor: str
    base_ref: str
    factors: tuple[AblatableFactor, ...]

    def factor(self, path: str) -> AblatableFactor:
        for entry in self.factors:
            if entry.path == path:
                return entry
        known = ", ".join(entry.path for entry in self.factors) or "(none)"
        raise ExperimentContractError(
            f"{self.executor} does not publish {path!r} as an ablatable factor; "
            f"declared factors are: {known}"
        )

    def normalize(self, path: str, value: Any) -> Any:
        return self.factor(path).normalize(value)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": FACTOR_CATALOG_SCHEMA,
            "executor": self.executor,
            "base_ref": self.base_ref,
            "factors": [entry.to_json() for entry in self.factors],
        }


# ------------------------------------------------------------------ plan parts


@dataclass(frozen=True, slots=True)
class ArmPlan:
    arm_id: str
    label: str
    treatment: dict[str, Any]
    subject: SubjectRef

    @property
    def treatment_digest(self) -> str:
        return digest_of(self.treatment)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "treatment": self.treatment,
            "treatment_digest": self.treatment_digest,
            "subject": self.subject.to_json(),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> ArmPlan:
        data = _object(value, context="arm")
        return cls(
            arm_id=_identifier(data.get("arm_id"), field_name="arm_id"),
            label=_text(data.get("label"), field_name="arm.label"),
            treatment=_object(data.get("treatment", {}), context="arm.treatment"),
            subject=SubjectRef.from_mapping(data.get("subject")),
        )


@dataclass(frozen=True, slots=True)
class TrialPlan:
    """One cell of the matrix, with its place in the dispatch order fixed."""

    trial_id: str
    arm_id: str
    block_id: str
    replicate: int
    dispatch_index: int
    trial_derived: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "arm_id": self.arm_id,
            "block_id": self.block_id,
            "replicate": self.replicate,
            "dispatch_index": self.dispatch_index,
            "trial_derived": self.trial_derived,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> TrialPlan:
        data = _object(value, context="trial")
        return cls(
            trial_id=_identifier(data.get("trial_id"), field_name="trial_id"),
            arm_id=_identifier(data.get("arm_id"), field_name="arm_id"),
            block_id=_identifier(data.get("block_id"), field_name="block_id"),
            replicate=_non_negative_int(data.get("replicate"), field_name="replicate"),
            dispatch_index=_non_negative_int(
                data.get("dispatch_index"), field_name="dispatch_index"
            ),
            trial_derived=_object(data.get("trial_derived", {}), context="trial.trial_derived"),
        )


# ------------------------------------------------------------------- outcomes


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """The append-only terminal row for one trial.

    Written once per trial and never rewritten.  A rerun that produces a second
    row for the same `trial_id` is a contradiction the reducer surfaces rather
    than resolves.
    """

    experiment_id: str
    trial_id: str
    arm_id: str
    block_id: str
    replicate: int
    status: str
    metrics: dict[str, float]
    usage: dict[str, Any]
    failure_class: str | None
    failure_detail: str | None
    executor: str
    executor_run_id: str | None
    receipt_digest: str | None
    correlation_digest: str
    evidence: dict[str, Any]
    infra: dict[str, Any]
    dispatched_at: str | None
    started_at: str | None
    finished_at: str | None
    #: 0 for the first dispatch.  A retry appends a new row rather than editing
    #: the one it replaces, so what was retried stays legible forever.
    attempt: int = 0
    #: Digest of the row this one supersedes, when it is a retry.
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TRIAL_OUTCOME_STATUSES:
            raise ExperimentContractError(
                f"trial outcome status must be one of {TRIAL_OUTCOME_STATUSES}"
            )
        if self.status != "completed" and self.failure_class is None:
            raise ExperimentContractError(
                "a non-completed trial must classify its failure; "
                "an unclassified gap cannot be tested for differential missingness"
            )
        if self.failure_class is not None and self.failure_class not in FAILURE_CLASSES:
            raise ExperimentContractError(f"failure_class must be one of {FAILURE_CLASSES}")

    @property
    def counted(self) -> bool:
        """Whether this row may contribute a number to an arm aggregate."""

        return self.status == "completed"

    @property
    def retryable(self) -> bool:
        """A rig failure says nothing about the arm, so re-running it is honest."""

        return not self.counted and self.failure_class in RETRYABLE_FAILURE_CLASSES

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TRIAL_OUTCOME_SCHEMA,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "arm_id": self.arm_id,
            "block_id": self.block_id,
            "replicate": self.replicate,
            "status": self.status,
            "metrics": self.metrics,
            "usage": self.usage,
            "failure_class": self.failure_class,
            "failure_detail": self.failure_detail,
            "executor": self.executor,
            "executor_run_id": self.executor_run_id,
            "receipt_digest": self.receipt_digest,
            "correlation_digest": self.correlation_digest,
            "evidence": self.evidence,
            "infra": self.infra,
            "dispatched_at": self.dispatched_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempt": self.attempt,
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> TrialOutcome:
        data = _object(value, context="trial outcome")
        schema = data.get("schema_version")
        if schema != TRIAL_OUTCOME_SCHEMA:
            raise ExperimentContractError(f"unsupported trial outcome schema {schema!r}")
        metrics = _object(data.get("metrics", {}), context="outcome.metrics")
        for key, metric in metrics.items():
            if not isinstance(metric, (int, float)) or isinstance(metric, bool):
                raise ExperimentContractError(f"outcome.metrics.{key} must be numeric")
        return cls(
            experiment_id=_identifier(data.get("experiment_id"), field_name="experiment_id"),
            trial_id=_identifier(data.get("trial_id"), field_name="trial_id"),
            arm_id=_identifier(data.get("arm_id"), field_name="arm_id"),
            block_id=_identifier(data.get("block_id"), field_name="block_id"),
            replicate=_non_negative_int(data.get("replicate"), field_name="replicate"),
            status=_text(data.get("status"), field_name="status"),
            metrics={key: float(item) for key, item in metrics.items()},
            usage=_object(data.get("usage", {}), context="outcome.usage"),
            failure_class=data.get("failure_class"),
            failure_detail=data.get("failure_detail"),
            executor=_identifier(data.get("executor"), field_name="executor"),
            executor_run_id=data.get("executor_run_id"),
            receipt_digest=data.get("receipt_digest"),
            correlation_digest=_digest(
                data.get("correlation_digest"), field_name="correlation_digest"
            ),
            evidence=_object(data.get("evidence", {}), context="outcome.evidence"),
            infra=_object(data.get("infra", {}), context="outcome.infra"),
            dispatched_at=data.get("dispatched_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            attempt=_non_negative_int(data.get("attempt", 0), field_name="attempt"),
            supersedes=data.get("supersedes"),
        )


def mint_trial_id(*, experiment_id: str, arm_id: str, block_id: str, replicate: int) -> str:
    """`trial_id = hash(experiment_id, arm_id, block_id, replicate)`.

    Deterministic so that resume recomputes the same identity from the spec
    alone, and so a trace found later can be matched without a lookup table.
    """

    payload = canonical_json(
        {
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "block_id": block_id,
            "replicate": replicate,
        }
    )
    return "t" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ALIAS_KEYS",
    "CORRELATION_SCHEMA",
    "EXPERIMENT_PLAN_SCHEMA",
    "EXPERIMENT_REPORT_SCHEMA",
    "EXPERIMENT_SPEC_SCHEMA",
    "FACTOR_CATALOG_SCHEMA",
    "FAILURE_CLASSES",
    "MISSING_POLICIES",
    "RETRYABLE_FAILURE_CLASSES",
    "TRIAL_OUTCOME_SCHEMA",
    "TRIAL_OUTCOME_STATUSES",
    "AblatableFactor",
    "ArmPlan",
    "CorrelationEnvelope",
    "ExperimentContractError",
    "FactorCatalog",
    "SubjectRef",
    "TrialOutcome",
    "TrialPlan",
    "canonical_json",
    "digest_of",
    "mint_trial_id",
]
