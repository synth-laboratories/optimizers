"""Frozen Containers CISPO producer bytes against pinned and current decoders.

SYN-4648. ``fixtures/containers_cispo`` holds compact sorted JSON emitted by
``synth_containers.cispo_contract.build_cispo_capability_document`` and
``synth_containers.cispo_handshake.HandshakeVerdict.to_payload`` at the
manifest's ``producer_revision``; the Containers repo carries byte-identical
copies at ``tests/fixtures/optimizer_cispo``.

``pinned_consumer/synth_optimizers_pinned`` is the decoder closure copied
verbatim from ``manifest.source_revision`` (hashes in ``source_files``), so a
change to the current decoder cannot silently re-pin what an installed consumer
at that revision accepts.

The consumer is closed on schema version, on the capability hash, and on
clause-id / verdict enums. It is OPEN on unknown object keys: an extra key that
is covered by a recomputed capability hash is tolerated. The tests below pin
that tolerance explicitly instead of claiming deny-unknown behavior.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from synth_optimizers.contracts import rl_records as current_records
from synth_optimizers.rl import capabilities as current_capabilities
from synth_optimizers.rl import handshake as current_handshake

FIXTURES = Path(__file__).parent / "fixtures" / "containers_cispo"
PINNED_ROOT = FIXTURES / "pinned_consumer"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
PIN = f"{MANIFEST['consumer']}@{MANIFEST['source_revision']}"
PRODUCER = f"{MANIFEST['producer']}@{MANIFEST['producer_revision']}"
CANARY_TOKENS = (101, 102, 103)

DECODES = (
    "capabilities.rich",
    "capabilities.minimal",
    "capabilities.canaried",
)
VERDICTS = ("handshake_verdict.accepted", "handshake_verdict.unsupported_optional")


def _pinned(module: str) -> ModuleType:
    if str(PINNED_ROOT) not in sys.path:
        sys.path.insert(0, str(PINNED_ROOT))
    return importlib.import_module(f"synth_optimizers_pinned.{module}")


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _rehash(module: ModuleType, document: dict[str, Any]) -> dict[str, Any]:
    document = {key: value for key, value in document.items() if key != "capability_hash"}
    document["capability_hash"] = module.canonical_capability_hash(document)
    return document


FAMILIES = {
    "pinned": lambda: (_pinned("rl.capabilities"), _pinned("rl.handshake")),
    "current": lambda: (current_capabilities, current_handshake),
}


def test_fixture_bytes_and_pinned_sources_match_the_manifest() -> None:
    for name, expected in MANIFEST["fixtures"].items():
        data = (FIXTURES / f"{name}.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, f"{name} drifted from {PRODUCER}"
        compact = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":")) + "\n"
        assert data == compact.encode(), f"{name} is not compact sorted JSON"
    prefix = "optimizers/src/synth_optimizers/"
    for path, expected in MANIFEST["source_files"].items():
        if not path.startswith(prefix):
            continue
        frozen = PINNED_ROOT / "synth_optimizers_pinned" / path[len(prefix) :]
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == expected, (
            f"{frozen} is not the verbatim decoder at {PIN}"
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("name", DECODES)
def test_every_capability_fixture_decodes(family: str, name: str) -> None:
    capabilities, _ = FAMILIES[family]()
    document = capabilities.CapabilityDocument.from_payload(_fixture(name))
    assert document.contract_version == "synth_optimizers.cispo.v1", f"{family} {PIN}"
    assert document.content_hash == _fixture(name)["capability_hash"]


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("name", VERDICTS)
def test_every_verdict_fixture_decodes(family: str, name: str) -> None:
    _, handshake = FAMILIES[family]()
    payload = _fixture(name)
    verdict = handshake.HandshakeVerdict.from_payload(payload)
    assert verdict.handshake_id == payload["handshake_id"]
    assert {item.clause_id: item.verdict for item in verdict.clauses} == {
        item["clause_id"]: item["verdict"] for item in payload["clauses"]
    }


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_an_unhashed_injected_field_is_refused(family: str) -> None:
    capabilities, _ = FAMILIES[family]()
    payload = {**_fixture("capabilities.rich"), "injected_unknown_field": True}
    with pytest.raises(capabilities.CapabilityHashError):
        capabilities.CapabilityDocument.from_payload(payload)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_hash_covered_unknown_field_is_tolerated_not_refused(family: str) -> None:
    """Recorded openness: the decoder ignores keys it does not read."""

    capabilities, _ = FAMILIES[family]()
    payload = _rehash(capabilities, {**_fixture("capabilities.rich"), "injected": True})
    capabilities.CapabilityDocument.from_payload(payload)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_an_unknown_clause_id_is_refused(family: str) -> None:
    capabilities, handshake = FAMILIES[family]()
    payload = _fixture("handshake_verdict.accepted")
    extra = {"clause_id": "evidence.injected_clause", "verdict": "accepted", "reason": "x"}
    with pytest.raises(capabilities.CapabilityError, match="unknown clause"):
        handshake.HandshakeVerdict.from_payload({**payload, "clauses": [*payload["clauses"], extra]})


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unsupported_capability_version_is_refused(family: str) -> None:
    capabilities, _ = FAMILIES[family]()
    with pytest.raises(capabilities.CapabilityError, match="is unsupported"):
        capabilities.CapabilityDocument.from_payload(
            _fixture("refuse.capabilities.unsupported_version")
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unsupported_handshake_version_is_refused(family: str) -> None:
    _, handshake = FAMILIES[family]()
    with pytest.raises(handshake.HandshakeError, match="is unsupported"):
        handshake.HandshakeVerdict.from_payload(
            _fixture("refuse.handshake_verdict.unsupported_version")
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_an_unknown_clause_verdict_is_refused(family: str) -> None:
    capabilities, handshake = FAMILIES[family]()
    with pytest.raises(capabilities.CapabilityError, match="unknown verdict"):
        handshake.HandshakeVerdict.from_payload(
            _fixture("refuse.handshake_verdict.unknown_verdict")
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_producer_canary_is_the_consumer_canary(family: str) -> None:
    records = _pinned("contracts.rl_records") if family == "pinned" else current_records
    capabilities, _ = FAMILIES[family]()
    document = capabilities.CapabilityDocument.from_payload(_fixture("capabilities.canaried"))
    assert document.renderer_profile.canary_digest == records.canary_digest(CANARY_TOKENS)
    document.renderer_profile.assert_renders_like(CANARY_TOKENS)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_renderer_canary_mismatch_is_refused(family: str) -> None:
    records = _pinned("contracts.rl_records") if family == "pinned" else current_records
    capabilities, _ = FAMILIES[family]()
    document = capabilities.CapabilityDocument.from_payload(_fixture("capabilities.canaried"))
    with pytest.raises(records.RecordError, match="renderer disagreement"):
        document.renderer_profile.assert_renders_like((101, 102, 104))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_an_undeclared_canary_cannot_prove_agreement(family: str) -> None:
    records = _pinned("contracts.rl_records") if family == "pinned" else current_records
    capabilities, _ = FAMILIES[family]()
    document = capabilities.CapabilityDocument.from_payload(_fixture("capabilities.rich"))
    with pytest.raises(records.RecordError, match="declares no canary digest"):
        document.renderer_profile.assert_renders_like(CANARY_TOKENS)
