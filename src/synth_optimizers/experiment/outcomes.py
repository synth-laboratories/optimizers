"""The append-only outcome log.

One file, one row per terminal trial, never rewritten.  Resume reads it to learn
what is already sealed; the reducer reads it and nothing else.  Keeping it
append-only is what makes "this trial failed" survive a rerun instead of being
quietly replaced by a later success — which is the same failure mode as scoring
a failed trial as zero, just slower.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import ExperimentContractError, TrialOutcome, digest_of


@dataclass(frozen=True, slots=True)
class OutcomeConflict:
    """The same trial sealed twice with different content."""

    trial_id: str
    first_digest: str
    second_digest: str

    def to_json(self) -> dict[str, str]:
        return {
            "trial_id": self.trial_id,
            "first_digest": self.first_digest,
            "second_digest": self.second_digest,
        }


@dataclass(frozen=True, slots=True)
class OutcomeSet:
    #: The winning row per trial: the highest attempt.
    rows: tuple[TrialOutcome, ...]
    conflicts: tuple[OutcomeConflict, ...]
    #: Rows a later attempt superseded, kept so a retry is never invisible.
    superseded: tuple[TrialOutcome, ...] = ()

    def by_trial(self) -> dict[str, TrialOutcome]:
        return {row.trial_id: row for row in self.rows}

    @property
    def retried_trial_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.trial_id for row in self.superseded}))

    def __iter__(self) -> Iterator[TrialOutcome]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


class OutcomeLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, outcome: TrialOutcome) -> None:
        payload = outcome.to_json()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # O_APPEND on a single line under the pipe buffer keeps concurrent
        # adapters from interleaving mid-record; the runner also serialises
        # writes, and this is the belt to that pair of braces.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> OutcomeSet:
        if not self.path.is_file():
            return OutcomeSet(rows=(), conflicts=())
        winner: dict[str, TrialOutcome] = {}
        digests: dict[tuple[str, int], str] = {}
        conflicts: list[OutcomeConflict] = []
        superseded: list[TrialOutcome] = []
        for number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ExperimentContractError(
                    f"{self.path}:{number} is not valid JSON: {error}"
                ) from error
            outcome = TrialOutcome.from_mapping(payload)
            digest = digest_of(payload)
            key = (outcome.trial_id, outcome.attempt)
            if key in digests:
                # Two rows for the same attempt are a contradiction, not a
                # retry: something sealed the same trial twice with different
                # content and only one of them can be true.
                if digests[key] != digest:
                    conflicts.append(
                        OutcomeConflict(
                            trial_id=outcome.trial_id,
                            first_digest=digests[key],
                            second_digest=digest,
                        )
                    )
                continue
            digests[key] = digest
            held = winner.get(outcome.trial_id)
            if held is None:
                winner[outcome.trial_id] = outcome
            elif outcome.attempt > held.attempt:
                superseded.append(held)
                winner[outcome.trial_id] = outcome
            else:
                superseded.append(outcome)
        return OutcomeSet(
            rows=tuple(winner.values()),
            conflicts=tuple(conflicts),
            superseded=tuple(superseded),
        )

    def sealed_trial_ids(self) -> set[str]:
        return set(self.load().by_trial())

    def last_attempt(self, trial_id: str) -> TrialOutcome | None:
        return self.load().by_trial().get(trial_id)


def reduce_replicates(
    rows: Sequence[TrialOutcome], *, metric_id: str
) -> dict[tuple[str, str], float]:
    """Collapse replicates to one number per (arm, block) before pairing.

    Only completed rows contribute.  A block where an arm has some completed and
    some failed replicates is reported at the mean of what completed, and the
    failures stay visible in the missingness accounting rather than being
    averaged away.
    """

    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if not row.counted or metric_id not in row.metrics:
            continue
        buckets.setdefault((row.arm_id, row.block_id), []).append(row.metrics[metric_id])
    return {key: sum(values) / len(values) for key, values in buckets.items()}


__all__ = ["OutcomeConflict", "OutcomeLog", "OutcomeSet", "reduce_replicates"]
