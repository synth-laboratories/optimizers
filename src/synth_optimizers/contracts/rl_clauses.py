"""Canonical handshake clause identifiers.

The clause list is generic. No clause names a task, a harness, or an
environment, and a container may answer every clause without knowing which
optimizer asked. Verdicts are per clause: a bare boolean tells you a run will
fail without telling you what to change.
"""

from __future__ import annotations

HANDSHAKE_SCHEMA_VERSION = "cispo.handshake.v1"

VERDICTS = ("accepted", "degraded", "rejected", "unsupported")

CLAUSE_GROUPS: dict[str, tuple[str, ...]] = {
    "contract": ("contract.version", "contract.routes"),
    "discovery": ("discovery.taskset", "discovery.task_digests", "discovery.topology"),
    "policy": (
        "policy.binding_transport",
        "policy.renderer_profile_match",
        "policy.revision_immutability",
        "policy.no_embedded_credentials",
        "policy.session_scoped_origin",
    ),
    "lifecycle": (
        "lifecycle.idempotency",
        "lifecycle.lease_renewal",
        "lifecycle.cancellation",
        "lifecycle.concurrency",
        "lifecycle.exactly_one_terminal",
        "lifecycle.pause_resume",
    ),
    "evidence": (
        "evidence.trace_v5",
        "evidence.behavior_logprobs",
        "evidence.strict_prefix",
        "evidence.masking",
        "evidence.wire_objects",
        "evidence.artifact_reference",
        "evidence.tito",
    ),
    "reward": (
        "reward.authority",
        "reward.binding_digest",
        "reward.horizon_quiescence",
        "reward.settlement_window",
        "reward.channels",
    ),
    "recovery": ("recovery.restart", "recovery.stale_discard"),
    "topology": (
        "topology.roster",
        "topology.channels",
        "topology.minimum_roster",
        "topology.opponent_pinning",
    ),
}

ALL_CLAUSES: tuple[str, ...] = tuple(
    clause for clauses in CLAUSE_GROUPS.values() for clause in clauses
)

# Optional clauses may come back unsupported; the run records its fallback.
# Everything else is mandatory and a rejection stops the run before spend.
OPTIONAL_CLAUSES: frozenset[str] = frozenset(
    {
        "evidence.tito",
        "evidence.artifact_reference",
        "reward.settlement_window",
        "lifecycle.pause_resume",
        "topology.channels",
        "topology.minimum_roster",
        "topology.opponent_pinning",
    }
)

MANDATORY_CLAUSES: tuple[str, ...] = tuple(
    clause for clause in ALL_CLAUSES if clause not in OPTIONAL_CLAUSES
)

# Clauses that only apply to a multi-instance topology.
TOPOLOGY_ONLY_CLAUSES: frozenset[str] = frozenset(CLAUSE_GROUPS["topology"])


def clause_group(clause_id: str) -> str:
    for group, clauses in CLAUSE_GROUPS.items():
        if clause_id in clauses:
            return group
    raise KeyError(f"unknown clause {clause_id!r}")
