"""Persisted proof that ``cispo.slime.v1`` ran a real Tinker update."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import resolve_tinker_model
from ...runtime import digest_payload, utcnow


RECEIPT_SCHEMA = "tinker.capability_validation.v1"
CISPO_CAPABILITY = "cispo.slime.v1"
ENV_RECEIPT_PATH = "TINKER_CISPO_VALIDATION_RECEIPT"


def default_receipt_path() -> Path | None:
    raw = os.environ.get(ENV_RECEIPT_PATH, "").strip()
    if raw:
        return Path(raw)
    return None


def is_cispo_validated(path: Path | None, model_id: str) -> bool:
    receipt = load_receipt(path)
    if receipt is None:
        return False
    if receipt.get("capability") != CISPO_CAPABILITY:
        return False
    if receipt.get("validated") is not True:
        return False
    if not receipt.get("paid_update"):
        return False
    stored = str(receipt.get("model_id") or "")
    return resolve_tinker_model(stored) == resolve_tinker_model(model_id) if stored else True


def load_receipt(path: Path | None) -> dict[str, Any] | None:
    target = path or default_receipt_path()
    if target is None or not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_SCHEMA:
        return None
    return payload


def write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "capability": CISPO_CAPABILITY,
        "validated_at": utcnow(),
        **payload,
    }
    body["digest"] = digest_payload(
        {
            "capability": body["capability"],
            "model_id": body.get("model_id"),
            "sft_job_id": body.get("sft_job_id"),
            "cispo_job_id": body.get("cispo_job_id"),
            "paid_update": body.get("paid_update"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body
