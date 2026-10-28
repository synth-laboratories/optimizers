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
        "lifecycle.clock_skew",
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

# A clause that only applies under some declared condition. It is mandatory when
# its condition holds and not applicable otherwise, which is different from
# being optional: an optional clause may be declined, a conditional one cannot
# be declined when it applies. Clock skew matters only where the horizon is
# read off a wall clock, because the horizon is the instant reward is read.
CONDITIONAL_CLAUSES: dict[str, str] = {
    "lifecycle.clock_skew": "horizon_kind == 'wall_clock'",
}

# Some mandatory clauses admit a declared substitute. Rejecting one of these
# stops the run, and accepting it is not the only way to satisfy it: a container
# that cannot quiesce may instead clip its state to the horizon, which answers
# the same question by other means. This is what a bare accepted/rejected
# verdict cannot express.
CLAUSE_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "reward.horizon_quiescence": ("horizon_clipped_snapshot",),
    "evidence.artifact_reference": ("inline_evidence_only",),
    "lifecycle.pause_resume": ("cancel_and_replace",),
}
FALLBACK_SATISFIABLE_CLAUSES: frozenset[str] = frozenset(CLAUSE_SUBSTITUTES)

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

# Mandatory clauses that always apply, whatever the run's shape. The requirement
# document must name every one of these; conditional clauses are named only when
# their condition holds.
UNCONDITIONAL_MANDATORY_CLAUSES: tuple[str, ...] = tuple(
    clause for clause in MANDATORY_CLAUSES if clause not in CONDITIONAL_CLAUSES
)


def applies(clause_id: str, *, horizon_kind: str | None = None) -> bool:
    """Whether a conditional clause applies to a run of this shape."""

    if clause_id not in CONDITIONAL_CLAUSES:
        return True
    if clause_id == "lifecycle.clock_skew":
        return horizon_kind == "wall_clock"
    raise KeyError(f"conditional clause {clause_id!r} has no applicability rule")


def substitutes_for(clause_id: str) -> tuple[str, ...]:
    """Declared substitutes that satisfy a mandatory clause by other means."""

    return CLAUSE_SUBSTITUTES.get(clause_id, ())

# Clauses that only apply to a multi-instance topology.
TOPOLOGY_ONLY_CLAUSES: frozenset[str] = frozenset(CLAUSE_GROUPS["topology"])


def clause_group(clause_id: str) -> str:
    for group, clauses in CLAUSE_GROUPS.items():
        if clause_id in clauses:
            return group
    raise KeyError(f"unknown clause {clause_id!r}")
