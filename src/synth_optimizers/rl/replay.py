"""Replay mode: the same plan against stored evidence, off-policy by construction.

Offline parity is what makes algorithm iteration affordable. Every per-call
record, reward receipt, and sealed trace is durable, so the plan's credit,
correction, and packing dimensions must be runnable with no live container and
no provider sampling. This module imports the assembly path and nothing that
can talk to a container or a provider; it is the same code path with the
rollout production removed.

Two rules are enforced here rather than documented:

* A replay-derived update is off-policy. It may not be published as an
  on-policy result, and every attestation carries its source runs and the
  staleness it accepted.
* Replaying an online run's stored evidence must reproduce that run's
  advantages and batch composition bit-for-bit under the same plan hash.
  :func:`compare` returns that as a structured diff, so it can gate changes to
  credit, reducer, and masking code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts.rl_records import EvidenceError
from .assembly import EvidenceBundle, TrainingBatch, assemble
from .plan import AlgorithmPlan

PRESENTATIONS = frozenset({"on_policy", "off_policy"})


class ReplayError(EvidenceError):
    """Replay was given evidence it cannot run, or a claim it cannot honor."""


class ReplayPublishError(ReplayError):
    """A replay-derived revision was presented as an on-policy result."""


@dataclass(frozen=True, slots=True)
class ReplaySource:
    """Stored evidence selected by run, group, or selector. No container."""

    run_ids: tuple[str, ...]
    bundles: tuple[EvidenceBundle, ...]
    selector: str = ""

    def __post_init__(self) -> None:
        if not self.run_ids:
            raise ReplayError("replay consumes stored evidence and must name its source runs")
        if not self.bundles:
            raise ReplayError("replay source carries no stored episodes")

    @property
    def accepted_staleness(self) -> int:
        return max(bundle.staleness_steps for bundle in self.bundles)


@dataclass(frozen=True, slots=True)
class PublishAttestation:
    """What the catalog records for a replay-derived update."""

    plan_hash: str
    off_policy: bool
    presented_as: str
    source_run_ids: tuple[str, ...]
    accepted_staleness: int
    advantage_digest: str
    composition_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "off_policy": self.off_policy,
            "presented_as": self.presented_as,
            "source_run_ids": list(self.source_run_ids),
            "accepted_staleness": self.accepted_staleness,
            "advantage_digest": self.advantage_digest,
            "composition_digest": self.composition_digest,
        }


def replay(
    plan: AlgorithmPlan,
    source: ReplaySource,
    *,
    round_index: int = 0,
) -> TrainingBatch:
    """Run the plan over stored evidence. Marked off-policy, always."""

    return assemble(
        plan,
        source.bundles,
        round_index=round_index,
        off_policy=True,
        source_run_ids=source.run_ids,
        accepted_staleness=source.accepted_staleness,
    )


def guard_publication(batch: TrainingBatch, *, presented_as: str) -> PublishAttestation:
    """Refuse to publish an off-policy batch as an on-policy revision."""

    if presented_as not in PRESENTATIONS:
        raise ReplayError(
            f"unknown presentation {presented_as!r}; known: {sorted(PRESENTATIONS)}"
        )
    if batch.off_policy and presented_as == "on_policy":
        raise ReplayPublishError(
            "this batch was assembled from stored evidence and is off-policy; it may not "
            f"publish a revision presented as on-policy (source runs: "
            f"{list(batch.source_run_ids)}, accepted staleness: {batch.accepted_staleness})"
        )
    return PublishAttestation(
        plan_hash=batch.plan_hash,
        off_policy=batch.off_policy,
        presented_as=presented_as,
        source_run_ids=batch.source_run_ids,
        accepted_staleness=batch.accepted_staleness,
        advantage_digest=batch.advantage_digest,
        composition_digest=batch.composition_digest,
    )


# --- Structured comparison ---------------------------------------------------


def _diff(path: str, left: Any, right: Any, out: list[str]) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key}: absent online, present in replay")
            elif key not in right:
                out.append(f"{path}.{key}: present online, absent in replay")
            else:
                _diff(f"{path}.{key}", left[key], right[key], out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} online vs {len(right)} in replay")
        for index in range(min(len(left), len(right))):
            _diff(f"{path}[{index}]", left[index], right[index], out)
        return
    if left != right:
        out.append(f"{path}: {left!r} online vs {right!r} in replay")


@dataclass(frozen=True, slots=True)
class ReplayDiff:
    """Structured, not a bool: the point is to say what moved."""

    plan_hash_online: str
    plan_hash_replayed: str
    advantage_digest_online: str
    advantage_digest_replayed: str
    composition_digest_online: str
    composition_digest_replayed: str
    advantage_differences: tuple[str, ...]
    composition_differences: tuple[str, ...]

    @property
    def plan_hash_matches(self) -> bool:
        return self.plan_hash_online == self.plan_hash_replayed

    @property
    def advantages_match(self) -> bool:
        return (
            self.advantage_digest_online == self.advantage_digest_replayed
            and not self.advantage_differences
        )

    @property
    def composition_matches(self) -> bool:
        return (
            self.composition_digest_online == self.composition_digest_replayed
            and not self.composition_differences
        )

    @property
    def identical(self) -> bool:
        return self.plan_hash_matches and self.advantages_match and self.composition_matches

    @property
    def differences(self) -> tuple[str, ...]:
        head: tuple[str, ...] = ()
        if not self.plan_hash_matches:
            head = (
                f"plan_hash: {self.plan_hash_online!r} online vs "
                f"{self.plan_hash_replayed!r} in replay",
            )
        return head + self.advantage_differences + self.composition_differences

    def to_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "plan_hash_matches": self.plan_hash_matches,
            "advantages_match": self.advantages_match,
            "composition_matches": self.composition_matches,
            "plan_hash": [self.plan_hash_online, self.plan_hash_replayed],
            "advantage_digest": [
                self.advantage_digest_online,
                self.advantage_digest_replayed,
            ],
            "composition_digest": [
                self.composition_digest_online,
                self.composition_digest_replayed,
            ],
            "differences": list(self.differences),
        }


def compare(online: TrainingBatch, replayed: TrainingBatch) -> ReplayDiff:
    """The cheapest regression test the system has, as a callable."""

    advantage_differences: list[str] = []
    _diff(
        "advantages",
        online.advantage_payload(),
        replayed.advantage_payload(),
        advantage_differences,
    )
    composition_differences: list[str] = []
    _diff(
        "composition",
        online.composition_payload(),
        replayed.composition_payload(),
        composition_differences,
    )
    return ReplayDiff(
        plan_hash_online=online.plan_hash,
        plan_hash_replayed=replayed.plan_hash,
        advantage_digest_online=online.advantage_digest,
        advantage_digest_replayed=replayed.advantage_digest,
        composition_digest_online=online.composition_digest,
        composition_digest_replayed=replayed.composition_digest,
        advantage_differences=tuple(advantage_differences),
        composition_differences=tuple(composition_differences),
    )


def assert_reproduces(online: TrainingBatch, replayed: TrainingBatch) -> ReplayDiff:
    """Raise unless the replay reproduced the online batch bit-for-bit."""

    diff = compare(online, replayed)
    if not diff.identical:
        raise ReplayError(
            "replay did not reproduce the online batch: " + "; ".join(diff.differences[:8])
        )
    return diff


def replayed_from(
    plan: AlgorithmPlan,
    online: TrainingBatch,
    bundles: Sequence[EvidenceBundle],
    run_ids: Sequence[str],
) -> ReplayDiff:
    """Convenience gate: replay these bundles and diff against an online run."""

    source = ReplaySource(run_ids=tuple(run_ids), bundles=tuple(bundles))
    return compare(online, replay(plan, source, round_index=online.round_index))
