"""Portable v0.8 candidate/task identities and Workshop fact producer.

The JSON schemas and golden corpus under ``contracts/synth-spine-v1`` are the
wire authority.  This module deliberately has no persistence: Workshop is the
single writer and consumes the facts emitted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_CANDIDATE = "synth.candidate.v1"
SCHEMA_TASK = "synth.task-contract.v1"
SCHEMA_FACT = "synth.workshop-experiment-fact.v1"
SCHEMA_RECEIPT = "synth.evidence-receipt.v1"
SCHEMA_EDGE = "synth.lineage-edge.v1"
DOMAIN = b"synth.canonical-json.v1\0"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractError(ValueError):
    """Validation failure with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}{': ' + detail if detail else ''}")
        self.code = code


def _validate_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if not (-(2**53) + 1 <= value <= 2**53 - 1):
            raise ContractError("canonical_integer_range")
        return
    if isinstance(value, float):
        code = "canonical_non_finite" if not math.isfinite(value) else "canonical_float_forbidden"
        raise ContractError(code)
    if isinstance(value, list):
        for item in value:
            _validate_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("canonical_key_type")
        for item in value.values():
            _validate_value(item)
        return
    raise ContractError("canonical_type_forbidden", type(value).__name__)


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    keys = sorted(value, key=lambda key: key.encode("utf-8"))
    return "{" + ",".join(f"{_canonical(key)}:{_canonical(value[key])}" for key in keys) + "}"


def canonical_json(value: Any, *, omit_self_digest: bool = False) -> bytes:
    """Return the one v1 byte representation.

    Unicode is preserved exactly (no NFC/NFD rewriting); object keys sort by
    UTF-8 bytes; only interoperable integers are accepted.  When requested,
    only a top-level ``digest`` member is omitted.
    """

    if omit_self_digest:
        if not isinstance(value, Mapping):
            raise ContractError("canonical_self_digest_root")
        value = {key: item for key, item in value.items() if key != "digest"}
    _validate_value(value)
    return _canonical(value).encode("utf-8")


def canonical_digest(value: Any, *, omit_self_digest: bool = False) -> str:
    raw = DOMAIN + canonical_json(value, omit_self_digest=omit_self_digest)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ContractError("digest_malformed", field_name)
    return value


def verify_self_digest(document: Mapping[str, Any]) -> str:
    offered = require_digest(document.get("digest"), "digest")
    computed = canonical_digest(document, omit_self_digest=True)
    if offered != computed:
        raise ContractError("digest_mismatch")
    return offered


def _exact_keys(
    document: Mapping[str, Any], required: set[str], optional: set[str] = set()
) -> None:
    missing = required - document.keys()
    if missing:
        raise ContractError("field_missing", sorted(missing)[0])
    unknown = document.keys() - required - optional
    if unknown:
        raise ContractError("field_unknown", sorted(unknown)[0])


def validate_candidate(document: Mapping[str, Any]) -> str:
    required = {
        "schema_version",
        "candidate_id",
        "kind",
        "content",
        "task_contract_digest",
        "digest",
    }
    _exact_keys(document, required, {"producing_run_id", "metadata"})
    if document["schema_version"] != SCHEMA_CANDIDATE:
        raise ContractError("schema_unsupported")
    content = document["content"]
    if not isinstance(content, Mapping):
        raise ContractError("field_type", "content")
    _exact_keys(content, {"digest", "media_type", "size_bytes"})
    require_digest(content["digest"], "content.digest")
    require_digest(document["task_contract_digest"], "task_contract_digest")
    if not isinstance(content["size_bytes"], int) or content["size_bytes"] < 0:
        raise ContractError("field_type", "content.size_bytes")
    return verify_self_digest(document)


def validate_task_contract(document: Mapping[str, Any]) -> str:
    required = {
        "schema_version",
        "task_id",
        "family",
        "revision",
        "runtime",
        "datasets",
        "evaluator",
        "seed_policy",
        "capability_requirements",
        "digest",
    }
    _exact_keys(document, required, {"metadata"})
    if document["schema_version"] != SCHEMA_TASK:
        raise ContractError("schema_unsupported")
    runtime = document["runtime"]
    evaluator = document["evaluator"]
    seed_policy = document["seed_policy"]
    if not all(isinstance(item, Mapping) for item in (runtime, evaluator, seed_policy)):
        raise ContractError("field_type")
    _exact_keys(runtime, {"image_digest", "entrypoint"})
    _exact_keys(evaluator, {"evaluator_id", "digest"})
    _exact_keys(seed_policy, {"kind", "seeds"})
    require_digest(runtime["image_digest"], "runtime.image_digest")
    require_digest(evaluator["digest"], "evaluator.digest")
    if seed_policy["kind"] != "explicit" or not isinstance(seed_policy["seeds"], list):
        raise ContractError("seed_policy_invalid")
    if len(set(seed_policy["seeds"])) != len(seed_policy["seeds"]):
        raise ContractError("seed_duplicate")
    for dataset in document["datasets"]:
        _exact_keys(dataset, {"dataset_id", "digest"})
        require_digest(dataset["digest"], "datasets.digest")
    return verify_self_digest(document)


NODE_KINDS = {
    "experiment",
    "run",
    "candidate",
    "task",
    "artifact",
    "evaluation",
    "trajectory",
    "deployment",
}
EDGE_ENDPOINTS = {
    "forked_from": ({"candidate", "run"}, {"candidate", "run"}),
    "rerun_of": ({"run"}, {"run"}),
    "warm_started_from": ({"run", "candidate"}, {"candidate", "artifact"}),
    "produced": ({"run"}, {"candidate", "artifact", "trajectory"}),
    "evaluated": ({"evaluation"}, {"candidate"}),
    "compared_with": ({"candidate", "evaluation", "run"}, {"candidate", "evaluation", "run"}),
    "promoted_to": ({"candidate"}, {"deployment"}),
    "reproduced_on": ({"evaluation", "candidate"}, {"evaluation", "candidate"}),
    "rolled_back_to": ({"deployment", "candidate"}, {"candidate", "deployment"}),
}


def validate_lineage_edge(edge: Mapping[str, Any]) -> str:
    _exact_keys(
        edge,
        {
            "schema_version",
            "edge_id",
            "edge_type",
            "source",
            "target",
            "evidence_refs",
            "recorded_at",
            "digest",
        },
    )
    if edge["schema_version"] != SCHEMA_EDGE or edge["edge_type"] not in EDGE_ENDPOINTS:
        raise ContractError("lineage_edge_type")
    sources, targets = EDGE_ENDPOINTS[edge["edge_type"]]
    for field_name, allowed in (("source", sources), ("target", targets)):
        endpoint = edge[field_name]
        _exact_keys(endpoint, {"kind", "id", "digest"})
        if endpoint["kind"] not in NODE_KINDS or endpoint["kind"] not in allowed:
            raise ContractError("lineage_endpoint_kind", field_name)
        require_digest(endpoint["digest"], f"{field_name}.digest")
    for ref in edge["evidence_refs"]:
        _exact_keys(ref, {"kind", "uri", "digest"})
        require_digest(ref["digest"], "evidence_refs.digest")
    return verify_self_digest(edge)


RECEIPT_KINDS = {"rerun", "fork", "evaluation", "artifact", "trajectory"}


def validate_receipt(receipt: Mapping[str, Any]) -> str:
    required = {
        "schema_version",
        "receipt_id",
        "receipt_kind",
        "idempotency_key",
        "candidate_digest",
        "task_contract_digest",
        "run_id",
        "content",
        "recorded_at",
        "digest",
    }
    _exact_keys(receipt, required, {"evaluator_id", "seed"})
    if receipt["schema_version"] != SCHEMA_RECEIPT or receipt["receipt_kind"] not in RECEIPT_KINDS:
        raise ContractError("receipt_kind")
    require_digest(receipt["candidate_digest"], "candidate_digest")
    require_digest(receipt["task_contract_digest"], "task_contract_digest")
    _exact_keys(receipt["content"], {"kind", "uri", "digest"})
    require_digest(receipt["content"]["digest"], "content.digest")
    if receipt["receipt_kind"] == "evaluation":
        if "evaluator_id" not in receipt or "seed" not in receipt:
            raise ContractError("evaluation_authority_missing")
    elif "evaluator_id" in receipt or "seed" in receipt:
        raise ContractError("evaluation_authority_unexpected")
    return verify_self_digest(receipt)


@dataclass(slots=True)
class WorkshopFactProducer:
    """Stateless-order adapter with bounded idempotency memory for local IPC."""

    producer_id: str
    _seen: dict[str, str] = field(default_factory=dict)

    def emit(
        self,
        *,
        sequence: int,
        fact_kind: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        recorded_at: str,
    ) -> dict[str, Any] | None:
        if sequence < 0:
            raise ContractError("fact_sequence")
        payload_digest = canonical_digest({"fact_kind": fact_kind, "payload": payload})
        previous = self._seen.get(idempotency_key)
        if previous is not None:
            if previous != payload_digest:
                raise ContractError("idempotency_conflict")
            return None
        if fact_kind not in {
            "rerun",
            "fork",
            "evaluation",
            "artifact",
            "trajectory",
            "lineage_edge",
        }:
            raise ContractError("fact_kind")
        fact = {
            "schema_version": SCHEMA_FACT,
            "producer_id": self.producer_id,
            "sequence": sequence,
            "fact_kind": fact_kind,
            "idempotency_key": idempotency_key,
            "payload": dict(payload),
            "recorded_at": recorded_at,
        }
        fact["digest"] = canonical_digest(fact)
        self._seen[idempotency_key] = payload_digest
        return fact
