from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_optimizers.contracts import (
    ContractError,
    WorkshopFactProducer,
    canonical_digest,
    canonical_json,
    validate_candidate,
    validate_lineage_edge,
    validate_receipt,
    validate_task_contract,
)

FIXTURES = Path(__file__).parents[1] / "contracts" / "synth-spine-v1" / "fixtures"
D = "sha256:" + "a" * 64


def test_python_matches_shared_golden_corpus() -> None:
    corpus = json.loads((FIXTURES / "valid" / "canonical-values.json").read_text())
    for case in corpus["cases"]:
        omit = case.get("omit_self_digest", False)
        assert canonical_json(case["value"], omit_self_digest=omit).decode() == case["canonical"]
        assert canonical_digest(case["value"], omit_self_digest=omit) == case["digest"]


def test_valid_candidate_and_task_use_one_join_identity() -> None:
    task = json.loads((FIXTURES / "valid" / "task.json").read_text())
    candidate = json.loads((FIXTURES / "valid" / "candidate.json").read_text())
    assert validate_task_contract(task) == task["digest"]
    assert validate_candidate(candidate) == candidate["digest"]
    assert candidate["task_contract_digest"] == task["digest"]


def test_all_versioned_receipt_fixtures_validate() -> None:
    corpus = json.loads((FIXTURES / "valid" / "evidence-receipts.json").read_text())
    assert {receipt["receipt_kind"] for receipt in corpus["receipts"]} == {
        "rerun",
        "fork",
        "evaluation",
        "artifact",
        "trajectory",
    }
    for receipt in corpus["receipts"]:
        assert validate_receipt(receipt) == receipt["digest"]
    edge = json.loads((FIXTURES / "valid" / "lineage-edge.json").read_text())
    assert validate_lineage_edge(edge) == edge["digest"]


@pytest.mark.parametrize(
    "name", ["unknown-field.json", "malformed-digest.json", "digest-mismatch.json"]
)
def test_invalid_fixture_fails_with_stable_code(name: str) -> None:
    fixture = json.loads((FIXTURES / "invalid" / name).read_text())
    validator = validate_candidate if fixture["contract"] == "candidate" else validate_task_contract
    with pytest.raises(ContractError) as raised:
        validator(fixture["document"])
    assert raised.value.code == fixture["expected_error"]


def _signed(payload: dict) -> dict:
    payload["digest"] = canonical_digest(payload)
    return payload


def test_typed_edges_validate_kinds_and_immutable_evidence() -> None:
    edge = _signed(
        {
            "schema_version": "synth.lineage-edge.v1",
            "edge_id": "edge-1",
            "edge_type": "produced",
            "source": {"kind": "run", "id": "run-1", "digest": D},
            "target": {"kind": "candidate", "id": "candidate-1", "digest": D},
            "evidence_refs": [{"kind": "receipt", "uri": "cas://sha256/a", "digest": D}],
            "recorded_at": "2026-08-24T14:00:00Z",
        }
    )
    assert validate_lineage_edge(edge) == edge["digest"]
    edge["source"]["kind"] = "candidate"
    edge["digest"] = canonical_digest(edge, omit_self_digest=True)
    with pytest.raises(ContractError, match="lineage_endpoint_kind"):
        validate_lineage_edge(edge)


@pytest.mark.parametrize(
    ("edge_type", "source_kind", "target_kind"),
    [
        ("forked_from", "candidate", "candidate"),
        ("rerun_of", "run", "run"),
        ("warm_started_from", "run", "artifact"),
        ("produced", "run", "trajectory"),
        ("evaluated", "evaluation", "candidate"),
        ("compared_with", "candidate", "candidate"),
        ("promoted_to", "candidate", "deployment"),
        ("reproduced_on", "evaluation", "evaluation"),
        ("rolled_back_to", "deployment", "candidate"),
    ],
)
def test_every_v1_lineage_edge_has_a_frozen_endpoint_rule(
    edge_type: str, source_kind: str, target_kind: str
) -> None:
    edge = _signed(
        {
            "schema_version": "synth.lineage-edge.v1",
            "edge_id": f"edge-{edge_type}",
            "edge_type": edge_type,
            "source": {"kind": source_kind, "id": "source", "digest": D},
            "target": {"kind": target_kind, "id": "target", "digest": D},
            "evidence_refs": [{"kind": "receipt", "uri": "cas://receipt", "digest": D}],
            "recorded_at": "2026-08-24T14:00:00Z",
        }
    )
    assert validate_lineage_edge(edge) == edge["digest"]


def test_evaluation_receipt_requires_evaluator_and_seed_without_fabrication() -> None:
    receipt = _signed(
        {
            "schema_version": "synth.evidence-receipt.v1",
            "receipt_id": "receipt-1",
            "receipt_kind": "evaluation",
            "idempotency_key": "eval:run-1:7",
            "candidate_digest": D,
            "task_contract_digest": D,
            "run_id": "run-1",
            "evaluator_id": "eval-v1",
            "seed": 7,
            "content": {"kind": "evaluation", "uri": "cas://sha256/a", "digest": D},
            "recorded_at": "2026-08-24T14:00:00Z",
        }
    )
    assert validate_receipt(receipt) == receipt["digest"]
    del receipt["seed"]
    receipt["digest"] = canonical_digest(receipt, omit_self_digest=True)
    with pytest.raises(ContractError, match="evaluation_authority_missing"):
        validate_receipt(receipt)


def test_workshop_facts_accept_out_of_order_and_dedupe_without_storage() -> None:
    producer = WorkshopFactProducer("optimizer:run-1")
    emitted = {
        kind: producer.emit(
            sequence=sequence,
            fact_kind=kind,
            payload={"receipt_digest": D},
            idempotency_key=f"{kind}-1",
            recorded_at="2026-08-24T14:00:00Z",
        )
        for kind, sequence in zip(
            ("rerun", "fork", "evaluation", "artifact", "trajectory"),
            (8, 2, 9, 1, 6),
            strict=True,
        )
    }
    assert all(emitted.values())
    late = emitted["evaluation"]
    early = emitted["fork"]
    assert late and early and late["sequence"] == 9 and early["sequence"] == 2
    assert (
        producer.emit(
            sequence=9,
            fact_kind="trajectory",
            payload={"receipt_digest": D},
            idempotency_key="trajectory-1",
            recorded_at="later",
        )
        is None
    )
    with pytest.raises(ContractError, match="idempotency_conflict"):
        producer.emit(
            sequence=10,
            fact_kind="trajectory",
            payload={"receipt_digest": "changed"},
            idempotency_key="trajectory-1",
            recorded_at="later",
        )


def test_canonicalizer_rejects_floats_and_non_interoperable_integers() -> None:
    with pytest.raises(ContractError, match="canonical_float_forbidden"):
        canonical_digest({"score": 0.5})
    with pytest.raises(ContractError, match="canonical_integer_range"):
        canonical_digest({"count": 2**53})
