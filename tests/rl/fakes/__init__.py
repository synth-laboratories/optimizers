"""Reusable conformance fakes for the container-first RL plane.

A fake is a real HTTP server on loopback that serves the declared CISPO route
surface from a declarative configuration. Every conformance-relevant behavior
is a flag on :class:`~fakes.container.ContainerConfig`; nothing in here
selects behavior by task name, harness name, or environment name.

Typical use from another work stream::

    from fakes import scenarios, serve

    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        client.preflight()
        attempt = client.run_attempt(task_id=client.task_ids()[0])
        for call in attempt.trainable_calls:
            call.validate_for_training()

Both ``serve(...)`` as a context manager and ``container.shutdown()`` are
supported. Everything is deterministic under ``ContainerConfig.seed``: two
containers with the same configuration serve byte-identical evidence.
"""

from __future__ import annotations

from .checks import (
    assert_effects_within_horizon,
    assert_instance_trajectories,
    assert_no_flattened_wire,
    assert_probe_evidence_marked,
    assert_declared_channels_present,
)
from .container import (
    CONTRACT_VERSION,
    DECLARED_ROUTES,
    AttemptResult,
    Clock,
    ContainerClient,
    ContainerConfig,
    ContainerError,
    EvidenceDefects,
    RunningContainer,
    group_pin_from_fields,
    inference_call_from_payload,
    reward_record_from_payload,
    serve,
    topology_from_payload,
    trainable_episode_from_payload,
)

__all__ = [
    "CONTRACT_VERSION",
    "DECLARED_ROUTES",
    "AttemptResult",
    "Clock",
    "ContainerClient",
    "ContainerConfig",
    "ContainerError",
    "EvidenceDefects",
    "RunningContainer",
    "assert_declared_channels_present",
    "assert_effects_within_horizon",
    "assert_instance_trajectories",
    "assert_no_flattened_wire",
    "assert_probe_evidence_marked",
    "group_pin_from_fields",
    "inference_call_from_payload",
    "reward_record_from_payload",
    "serve",
    "topology_from_payload",
    "trainable_episode_from_payload",
]
