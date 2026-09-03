"""An in-process fake CISPO container plus the client that drives it.

The fake serves the full declared route surface over stdlib ``http.server`` on
``127.0.0.1`` at an ephemeral port. It is a *contract* fake, not a simulator:
it has no environment, no model, and no task logic. Everything a conformance
test needs to vary is a flag on :class:`ContainerConfig`, and everything the
fake emits is derived deterministically from ``ContainerConfig.seed`` plus the
requested task row, so a replay test can reproduce an attempt bit-for-bit.

Two rules the fake never breaks, because they are the reason it exists:

* it round-trips the executor's opaque correlation metadata untouched; and
* it emits exactly one terminal result per accepted attempt.

Everything else -- including well-formedness of the evidence -- is a flag, so
that the deliberately non-conformant scenarios in :mod:`fakes.scenarios` are
the *same code path* as the conformant ones with one bit flipped.

No real time passes anywhere: leases, horizons, and handshake expiry all read
an injected :class:`Clock`.

Layout: :mod:`.config` declares a container, :mod:`.evidence` synthesizes what
it returns, :mod:`.codecs` moves records across the wire, :mod:`.server` serves
the routes, and :mod:`.client` drives them.
"""

from __future__ import annotations

from .client import AttemptResult, ContainerClient
from .codecs import (
    group_pin_from_fields,
    inference_call_from_payload,
    reward_record_from_payload,
    rollout_receipt_from_payload,
    segment_from_payload,
    topology_from_payload,
    trainable_episode_from_payload,
)
from .config import (
    ALIAS_REFS,
    CONTRACT_VERSION,
    CORRELATION_FIELDS,
    DECLARED_ROUTES,
    PROMPT_BUDGET_POLICIES,
    REWARD_KINDS,
    Clock,
    ContainerConfig,
    ContainerError,
    EvidenceDefects,
)
from .server import RunningContainer, serve

__all__ = [
    "ALIAS_REFS",
    "CONTRACT_VERSION",
    "CORRELATION_FIELDS",
    "DECLARED_ROUTES",
    "PROMPT_BUDGET_POLICIES",
    "REWARD_KINDS",
    "AttemptResult",
    "Clock",
    "ContainerClient",
    "ContainerConfig",
    "ContainerError",
    "EvidenceDefects",
    "RunningContainer",
    "group_pin_from_fields",
    "inference_call_from_payload",
    "reward_record_from_payload",
    "rollout_receipt_from_payload",
    "segment_from_payload",
    "serve",
    "topology_from_payload",
    "trainable_episode_from_payload",
]
