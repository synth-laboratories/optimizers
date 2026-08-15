"""Compatibility identifiers for hosted algorithms not yet publicly supported.

These identifiers let the SDK parse a hosted catalog and preserve run history without
implying that the package provides a supported local executor, cookbook, or release
commitment.  Promote an entry only when its public contract and end-to-end evidence
are ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FutureHostedAlgorithmSlug(StrEnum):
    MAPO = "mapo"
    OHCO = "ohco"
    ONLINE_REFLEXION = "online-reflexion"
    MARL_PROMPTOPT = "marl-promptopt"


@dataclass(frozen=True, slots=True)
class FutureHostedAlgorithm:
    slug: FutureHostedAlgorithmSlug
    summary: str


FUTURE_HOSTED_ALGORITHMS: tuple[FutureHostedAlgorithm, ...] = (
    FutureHostedAlgorithm(FutureHostedAlgorithmSlug.MAPO, "Private hosted compatibility lane."),
    FutureHostedAlgorithm(FutureHostedAlgorithmSlug.OHCO, "Private hosted compatibility lane."),
    FutureHostedAlgorithm(
        FutureHostedAlgorithmSlug.ONLINE_REFLEXION,
        "Gated hosted compatibility lane pending release evidence.",
    ),
    FutureHostedAlgorithm(
        FutureHostedAlgorithmSlug.MARL_PROMPTOPT,
        "Research executable; not a public hosted product lane.",
    ),
)
