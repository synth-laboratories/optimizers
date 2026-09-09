"""Conformance assertions the shared records do not yet cover.

Everything here raises one of the typed errors from the shared contract
modules -- ``EvidenceError`` or ``TopologyError`` -- so a caller can tell
*which* rule a container broke rather than only that something broke. When
stream 2 (preflight) and stream 4 (assembly) land, these move into engine
modules; the fakes and their tests then import from there instead.

No function here names a task, a harness, or an environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from synth_optimizers.contracts.rl_identity import (
    CommunicationChannel,
    Topology,
    TopologyError,
)
from synth_optimizers.contracts.rl_records import EvidenceError, InferenceCall


def assert_declared_channels_present(
    declared: Sequence[CommunicationChannel] | Topology,
    observed: Iterable[Mapping[str, object]],
) -> None:
    """A declared channel that carried no message for a whole episode fails.

    A silently dropped channel is an evidence failure, not a truncated
    observation history to train on.
    """

    channels = (
        declared.communication_channels if isinstance(declared, Topology) else tuple(declared)
    )
    counts = {
        str(row["channel_id"]): int(row.get("message_count") or 0) for row in observed
    }
    for channel in channels:
        if channel.channel_id not in counts:
            raise EvidenceError(
                f"declared channel {channel.channel_id!r} is absent from the trace"
            )
        if counts[channel.channel_id] == 0:
            raise EvidenceError(
                f"declared {channel.scope} channel {channel.channel_id!r} returned no "
                "messages for the whole episode"
            )


def assert_effects_within_horizon(
    calls: Iterable[InferenceCall], *, horizon_value: float, quiesced: bool, clipped: bool
) -> None:
    """A deferred-program span may not author effects past the horizon.

    Under ``deferred_program`` actuation the policy emits a loop whose effects
    outlive the sampling call. A span whose authored interval extends past the
    horizon and is neither quiesced nor clipped must be refused, not masked
    into the batch.
    """

    if quiesced or clipped:
        return
    for call in calls:
        end = call.effect_tick_end
        if end is None:
            continue
        if end > horizon_value:
            raise EvidenceError(
                f"call {call.call_id} authored effects to tick {end} past horizon "
                f"{horizon_value} with no quiescence attestation and no horizon clipping"
            )


def assert_no_flattened_wire(
    calls: Iterable[InferenceCall], *, declared_wire_api: str
) -> None:
    """The persisted wire object must be the wire the container declared.

    Flattening responses output items into chat messages and training on the
    result is prohibited, as is presenting a chat trajectory as a responses
    distribution: they are two datasets, not one.
    """

    for call in calls:
        for name, payload in (("request", call.wire_request), ("response", call.wire_response)):
            persisted = payload.get("wire")
            if persisted is None:
                raise EvidenceError(
                    f"call {call.call_id} persisted no wire {name} object"
                )
            if persisted != declared_wire_api:
                raise EvidenceError(
                    f"call {call.call_id} declares wire {declared_wire_api!r} but persisted a "
                    f"{persisted!r} {name} object; a flattened wire is a different dataset"
                )


def assert_probe_evidence_marked(calls: Iterable[InferenceCall]) -> None:
    """Probe evidence must be distinguishable from real evidence.

    A container whose probe attempt returns evidence indistinguishable from a
    paid attempt fails conformance: nothing downstream can keep it out of a
    group.
    """

    for call in calls:
        if call.token_capture_provenance != "probe_synthetic" or call.trainable:
            raise EvidenceError(
                f"probe call {call.call_id} is indistinguishable from real evidence "
                f"(trainable={call.trainable}, "
                f"provenance={call.token_capture_provenance!r})"
            )


def assert_instance_trajectories(
    topology: Topology, trace: Mapping[str, object], *, disposition: str
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    """Apply the declared partial-roster disposition to a sealed trace.

    Returns ``(missing_instance_ids, recorded_absences)``. Under ``refuse`` any
    missing instance raises ``TopologyError``; under ``drop_instance`` the
    absence, death time, and last live tick must be recorded, and a trace that
    omits an instance without recording it is refused too. Silently training on
    twenty-three of twenty-four instances is prohibited.
    """

    rows = tuple(trace.get("instances") or ())  # type: ignore[arg-type]
    live = [str(row["agent_instance_id"]) for row in rows if row.get("present")]
    missing = topology.check_roster(live, disposition=disposition)
    absences = tuple(row for row in rows if not row.get("present"))
    for row in absences:
        if row.get("absent_at_tick") is None or row.get("last_live_tick") is None:
            raise TopologyError(
                f"instance {row['agent_instance_id']!r} is absent with no recorded "
                "death time or last live tick"
            )
    if len(absences) != len(missing):
        raise TopologyError(
            f"topology {topology.topology_id} reports {len(absences)} absences but "
            f"{len(missing)} instances are missing from the roster"
        )
    return missing, absences
