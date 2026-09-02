from __future__ import annotations

import pytest

from synth_optimizers.contracts.training_schemas import (
    SchemaError,
    validate_cispo_request,
    validate_sft_request,
    validate_usage_receipt,
)


def test_sft_request_round_trip() -> None:
    request = validate_sft_request(
        {
            "schema_version": "sft.request.v1",
            "algorithm_id": "sft",
            "implementation": "tinker-sft",
            "implementation_version": "sft.tinker.v1",
            "provider": "tinker",
            "model_id": "openai/gpt-oss-20b",
            "dataset": {"examples": []},
            "training": {"steps": 1},
            "evaluation": {},
            "seed": 1,
        }
    )
    assert request.model_id == "openai/gpt-oss-20b"


def test_cispo_cannot_claim_an_alternative_algorithm() -> None:
    with pytest.raises(SchemaError, match="cispo.slime.v1"):
        validate_cispo_request(
            {
                "schema_version": "cispo.request.v1",
                "algorithm_id": "cispo",
                "implementation": "tinker-is",
                "implementation_version": "importance_sampling.v1",
                "provider": "tinker",
                "model_id": "openai/gpt-oss-20b",
                "dataset": {},
                "training": {},
                "reward": {},
                "seed": 0,
            }
        )


def test_missing_cost_cannot_invent_a_usd_amount() -> None:
    with pytest.raises(SchemaError, match="missing cost"):
        validate_usage_receipt(
            {
                "schema_version": "training.usage_receipt.v1",
                "provider": "tinker",
                "request_id": "req_1",
                "input_tokens": 1,
                "output_tokens": 1,
                "training_tokens": 1,
                "cost_usd": 0.01,
                "cost_missing": True,
                "algorithm_id": "sft",
                "implementation_version": "sft.tinker.v1",
            }
        )
