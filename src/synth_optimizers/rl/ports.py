"""The seams between the plane's islands.

Admission, queues, training, and artifacts were each built against the records
rather than against each other. These protocols are how the executor reaches
them without any island importing another, and how a second provider or a
second sampler transport arrives without touching the loop.

Nothing here names a task, a harness, an environment, or a provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..contracts.rl_identity import GroupPin, RolloutReceipt, TaskSpec
from ..contracts.rl_records import RendererProfile, RewardRecord, TrainableEpisode

PORTS_SCHEMA_VERSION = "cispo.ports.v1"


class PortError(RuntimeError):
    """A seam refused. Never degrade one of these into a zero reward."""


@dataclass(frozen=True, slots=True)
class SamplerOrigin:
    """Where a bound policy is reachable, for one attempt only.

    The per-attempt identity lives in the path so stitching is a URL parse and
    a leaked credential cannot cross rollouts. The credential names the group,
    sample, wire, and pinned revision; the harness never sees those fields.
    """

    base_url: str
    credential: str
    policy_revision: int
    behavior_fingerprint: str
    proxy_request_id: str
    wire_api: str
    sampling_transport: str
    expires_at: str = ""


@dataclass(frozen=True, slots=True)
class AttemptFacts:
    """What an attempt is, beyond which policy it samples.

    A group pin says which policy and which task family; an episode record
    demands the task id and seed. Without these on the binding call the gateway
    can only fabricate them from the pin, which is how a run ends up training
    on evidence that names the wrong task.
    """

    rollout_id: str
    task_id: str
    seed: int
    terminal_status: str = "completed"

    def __post_init__(self) -> None:
        if not self.rollout_id.strip() or not self.task_id.strip():
            raise PortError("an attempt must name its rollout and its task")


@dataclass(frozen=True, slots=True)
class PolicyRevision:
    """One materialized, immutable, sampleable policy."""

    revision: int
    revision_id: str
    checkpoint_id: str
    parameter_group_id: str
    sampler_reference: str
    behavior_fingerprint: str
    training_state_reference: str | None = None
    policy_set_revision_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """What one provider training step actually cost and produced."""

    request_ids: tuple[str, ...]
    examples: int
    tokens: int
    provider_cost: float
    metrics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SamplerGateway(Protocol):
    """Owns the renderer and the token capture, so containers need neither."""

    @property
    def renderer_profile(self) -> RendererProfile: ...

    def bind(
        self,
        revision: PolicyRevision,
        *,
        pin: GroupPin,
        sample_index: int,
        proxy_request_id: str,
        attempt: AttemptFacts,
    ) -> SamplerOrigin:
        """Open a session-scoped origin pinned to one immutable revision.

        Binding the same proxy_request_id twice must return the same origin;
        binding a route to a second revision must raise. The attempt facts are
        required here rather than inferred later, so the evidence this origin
        captures can name the task and seed it actually ran.
        """

    def close(self, proxy_request_id: str) -> None:
        """Retire an origin. Calls against it afterwards must fail."""

    def episode(self, proxy_request_id: str) -> TrainableEpisode:
        """The captured evidence for one attempt, already validated."""


@runtime_checkable
class PolicyBinder(Protocol):
    """Bridges a training provider to the catalog, in that order.

    A revision exists when it is catalogued, not when a provider returns a
    path. Publication and retirement are the catalog's business; this port is
    how the executor asks for them without importing either side.
    """

    def baseline(self, *, run_id: str, parameter_group_id: str) -> PolicyRevision: ...

    def train(
        self,
        *,
        parameter_group_id: str,
        batch: Sequence[Mapping[str, Any]],
        update_id: str,
        plan_hash: str,
    ) -> TrainOutcome: ...

    def publish(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_groups: Sequence[str],
        outcome: Mapping[str, TrainOutcome],
    ) -> Mapping[str, PolicyRevision]:
        """Materialize and publish one revision per group, atomically.

        Either every component publishes or none does; a one-sided failure
        leaves the prior set live and catalogues the orphan.
        """

    def resolve(self, selector: str) -> Mapping[str, PolicyRevision]:
        """An immutable id or an alias that records what it resolved to."""


@runtime_checkable
class ContainerSession(Protocol):
    """The admitted container, after handshake and probe, for one run."""

    @property
    def handshake_id(self) -> str: ...

    @property
    def agreement_digest(self) -> str: ...

    def tasks(self, *, split: str, task_ids: Sequence[str]) -> tuple[TaskSpec, ...]: ...

    def submit(
        self,
        task: TaskSpec,
        origin: SamplerOrigin,
        *,
        pin: GroupPin,
        sample_index: int,
        idempotency_key: str,
    ) -> str:
        """Accept one attempt and return its rollout id. Idempotent by key."""

    def poll(self, rollout_id: str) -> Mapping[str, Any]: ...

    def renew(self, rollout_id: str) -> Mapping[str, Any]: ...

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        """Quiesce at the horizon, or return a horizon-clipped snapshot."""

    def terminate(self, rollout_id: str, *, reason: str) -> RolloutReceipt: ...

    def evidence(self, rollout_id: str) -> tuple[TrainableEpisode, RewardRecord]:
        """The sealed episode and its reward, both already validated."""
