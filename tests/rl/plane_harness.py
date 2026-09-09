"""In-process wiring that lets the executor drive the conformance fakes.

Three pieces, none of which belongs in the engine:

* :class:`CanonicalClient` is a ``ContainerClient`` over a running fake: every
  method is a straight pass-through to the declared route, and it records the
  call order so a test can assert the startup sequence. The fake serves the
  canonical ``cispo.capabilities.v1`` document and the canonical agreement
  digest itself; nothing here rewrites what the container said.
* :class:`RecordingGateway` is a ``SamplerGateway``. It owns the renderer
  profile and mints one origin per attempt; it never samples, because the fake
  produces the generations.
* :class:`CatalogBinder` is a ``PolicyBinder`` over the real checkpoint catalog
  and the real atomic publisher, so a published round is exercised end to end
  with no provider and no spend.

Nothing here names a task, a harness or an environment: a scenario is chosen by
capability configuration, exactly as the conformance suite does it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fakes.container import ContainerConfig, RunningContainer, serve
from synth_optimizers.contracts.rl_identity import GroupPin
from synth_optimizers.contracts.rl_records import (
    BehaviorFingerprint,
    RendererProfile,
    SamplingProfile,
    TrainableEpisode,
)
from synth_optimizers.rl.catalog import (
    CheckpointArtifacts,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    SamplerWeightsRef,
    TrainingEvidence,
    TrainingStateRef,
)
from synth_optimizers.rl.contract import ContainerClient, ContainerContract, ContainerStatusError
from synth_optimizers.rl.policy_sets import (
    ComponentSaveAttempt,
    PolicySetComponent,
    PolicySetPublisher,
    PolicySetRevision,
)
from synth_optimizers.rl.ports import (
    AttemptFacts,
    PolicyRevision,
    PortError,
    SamplerOrigin,
    TrainOutcome,
)
from synth_optimizers.rl.session import RunClock

#: The epoch the fakes stamp their RFC3339 times from.
FAKE_EPOCH = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)


def _digest(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #


class PlaneClock:
    """One injected clock for both sides. Nothing in a test ever sleeps."""

    def __init__(self, container: RunningContainer) -> None:
        self._container = container
        self.run = RunClock(epoch=FAKE_EPOCH)

    def advance(self, seconds: float) -> None:
        self._container.clock.advance(seconds)
        self.run.advance(seconds)


# --------------------------------------------------------------------------- #
# The canonical contract client over a fake
# --------------------------------------------------------------------------- #


class CanonicalClient(ContainerClient):
    """The declared routes of a running fake, behind the shared interface."""

    def __init__(
        self,
        container: RunningContainer,
        *,
        reward_for: Callable[[str, int], float] | None = None,
    ) -> None:
        self._container = container
        self._config = container.config
        self._client = container.client()
        self._contract = ContainerContract.from_metadata(self._client.call("GET", "/metadata"))
        # One constant measure would tie every attempt in a group; the
        # container carries the per-attempt reward source itself.
        container.set_reward_source(reward_for or (lambda _task_id, index: 0.25 * (index + 1)))
        self.handshake_id = ""
        self.agreement_digest = ""
        self.calls: list[str] = []

    # -- plumbing ----------------------------------------------------------

    @property
    def contract(self) -> ContainerContract:
        return self._contract

    @property
    def config(self) -> ContainerConfig:
        return self._config

    def _record(self, name: str) -> None:
        self.calls.append(name)

    def fetch_reference(self, reference: str) -> Mapping[str, Any]:
        return self._client.call("GET", reference)

    # -- declared routes ---------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        self._record("health")
        return self._client.health()

    def metadata(self) -> Mapping[str, Any]:
        self._record("metadata")
        return self._client.call("GET", "/metadata")

    def capabilities(self) -> Mapping[str, Any]:
        self._record("capabilities")
        return self._client.capabilities()

    def handshake(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("handshake")
        payload = dict(self._client.handshake(dict(request)))
        if payload.get("accepted"):
            self.handshake_id = str(payload["handshake_id"])
            self.agreement_digest = str(payload["agreement_digest"])
        return payload

    def taskset(self) -> Mapping[str, Any]:
        self._record("taskset")
        return self._client.taskset()

    def taskset_tasks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("taskset_tasks")
        return self._client.taskset_tasks(
            request.get("ids") or (), split=str(request.get("split") or "train")
        )

    def topology(self, topology_id: str) -> Mapping[str, Any]:
        self._record("topology")
        return self._client.topology(topology_id)

    def bind_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("bind_policy")
        return self._client.bind_policy(**dict(request))

    def bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("bind_policy_set")
        return self._client.bind_policy_set(**dict(request))

    def submit_rollout(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("submit_rollout")
        return self._client.submit(**dict(request))

    def rollout_state(self, rollout_id: str) -> Mapping[str, Any]:
        self._record("rollout_state")
        return self._client.state(rollout_id)

    def rollout_events(self, rollout_id: str, *, cursor: str | None = None) -> Mapping[str, Any]:
        self._record("rollout_events")
        return self._client.events(rollout_id, int(cursor or 0))

    def renew_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("renew_rollout")
        return self._client.renew(rollout_id)

    def finalize_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("finalize_rollout")
        return self._client.finalize(rollout_id)

    def terminate_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record("terminate_rollout")
        return self._client.terminate(rollout_id, str(request.get("reason") or "cancelled"))

    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        self._record("trace")
        return self._client.trace(rollout_id)

    def artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        self._record("artifacts")
        return self._client.artifacts(rollout_id)

    def reward(
        self, rollout_id: str, *, request: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        self._record("reward")
        status, payload = self._client.reward(rollout_id)
        if status == 202:
            return {"rollout_id": rollout_id, "state": "pending"}
        if status >= 400:
            raise ContainerStatusError("/reward", status, json.dumps(payload))
        return payload


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #


def behavior_fingerprint(config: ContainerConfig, revision: int) -> str:
    """The fingerprint the container will stamp on this revision's calls."""

    return BehaviorFingerprint(
        renderer_profile=config.renderer_profile,
        model_family=config.model_family,
        model_id=config.model_id,
        policy_revision=revision,
        wire_api=config.wire_api,
        sampling_transport=config.sampling_transport,
        sampling=SamplingProfile(temperature=1.0, top_p=1.0, seed=config.seed),
    ).value


@dataclass
class RecordingGateway:
    """A sampler gateway that mints origins and remembers every binding."""

    profile: RendererProfile
    origins: dict[str, SamplerOrigin] = field(default_factory=dict)
    bindings: list[tuple[str, int, str]] = field(default_factory=list)
    declared: list[tuple[str, str, str, int]] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    @property
    def renderer_profile(self) -> RendererProfile:
        return self.profile

    def bind(
        self,
        revision: PolicyRevision,
        *,
        pin: GroupPin,
        sample_index: int,
        proxy_request_id: str,
        attempt: AttemptFacts,
    ) -> SamplerOrigin:
        existing = self.origins.get(proxy_request_id)
        if existing is not None:
            if existing.policy_revision != revision.revision:
                raise PortError(
                    f"origin {proxy_request_id} is already pinned to revision "
                    f"{existing.policy_revision}"
                )
            return existing
        origin = SamplerOrigin(
            base_url="https://sampler.invalid",
            credential=(
                f"{pin.group_id}/{sample_index}/{revision.parameter_group_id}"
                f"/{revision.revision_id}"
            ),
            policy_revision=revision.revision,
            behavior_fingerprint=revision.behavior_fingerprint,
            proxy_request_id=proxy_request_id,
            wire_api=pin.wire_api,
            sampling_transport=pin.sampling_transport,
        )
        self.origins[proxy_request_id] = origin
        self.bindings.append((proxy_request_id, revision.revision, attempt.task_id))
        return origin

    def declare_attempt(
        self,
        proxy_request_id: str,
        *,
        rollout_id: str,
        task_id: str,
        seed: int,
        terminal_status: str = "completed",
    ) -> None:
        if proxy_request_id not in self.origins:
            raise PortError(f"origin {proxy_request_id} is not bound")
        self.declared.append((proxy_request_id, rollout_id, task_id, seed))

    def close(self, proxy_request_id: str) -> None:
        self.origins.pop(proxy_request_id, None)
        self.closed.append(proxy_request_id)

    def episode(self, proxy_request_id: str) -> TrainableEpisode:
        raise PortError(
            "this gateway captures nothing: the container seals the evidence and the "
            "session validates it"
        )


# --------------------------------------------------------------------------- #
# Binder over the real catalog and the real atomic publisher
# --------------------------------------------------------------------------- #


class CatalogBinder:
    """A policy binder with no provider: real catalog, synthetic weights."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        renderer_profile: RendererProfile,
        contract_hash: str,
        base_model: str,
        policy_types: Mapping[str, str],
        fingerprints: Callable[[int], str],
        policy_set_id: str = "policy-set-run",
        fail_groups: Sequence[str] = (),
    ) -> None:
        self.catalog = CheckpointCatalog(catalog_path)
        self.publisher = PolicySetPublisher(self.catalog)
        self.compatibility = CheckpointCompatibility.from_renderer_profile(
            renderer_profile, container_contract_hash=contract_hash
        )
        self.base_model = base_model
        self.policy_types = dict(policy_types)
        self.fingerprints = fingerprints
        self.policy_set_id = policy_set_id
        self.fail_groups = tuple(fail_groups)
        self.revision = 0
        self.train_calls: list[Mapping[str, Any]] = []
        self.published: list[str] = []
        self.baselines: dict[str, PolicyRevision] = {}
        self._parent: dict[str, str] = {}
        self._parent_set: str | None = None

    # -- helpers -----------------------------------------------------------

    def _policy_type(self, parameter_group_id: str) -> str:
        return self.policy_types.get(parameter_group_id, f"type::{parameter_group_id}")

    def _record(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_group_id: str,
        revision: int,
        train_call_ids: Sequence[str],
        evidence: TrainingEvidence,
        status: str,
    ) -> CheckpointRecord:
        checkpoint_id = f"ckpt::{parameter_group_id}::{revision}"
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            update_id=update_id,
            train_call_ids=tuple(train_call_ids),
            parameter_group_id=parameter_group_id,
            policy_type_ids=(self._policy_type(parameter_group_id),),
            policy_revision_id=f"rev::{parameter_group_id}::{revision}",
            base_model=self.base_model,
            artifacts=CheckpointArtifacts(
                sampler_weights=SamplerWeightsRef(
                    ref=f"weights://{checkpoint_id}", digest=_digest(checkpoint_id, "sampler")
                ),
                training_state=TrainingStateRef(
                    ref=f"state://{checkpoint_id}", digest=_digest(checkpoint_id, "state")
                ),
            ),
            training_evidence=evidence,
            compatibility=self.compatibility,
            created_at=self.catalog.now(),
            parent_checkpoint_id=self._parent.get(parameter_group_id),
            publication_status=status,
        )

    def _revision_of(self, record: CheckpointRecord, revision: int) -> PolicyRevision:
        return PolicyRevision(
            revision=revision,
            revision_id=record.policy_revision_id,
            checkpoint_id=record.checkpoint_id,
            parameter_group_id=record.parameter_group_id,
            sampler_reference=record.sampler_weights.ref,
            behavior_fingerprint=self.fingerprints(revision),
            training_state_reference=record.training_state.ref,
            policy_set_revision_id=self._parent_set,
            metadata={
                "sampler_digest": record.sampler_weights.digest,
                "training_state_digest": record.training_state.digest,
            },
        )

    # -- the port ----------------------------------------------------------

    def baseline(self, *, run_id: str, parameter_group_id: str) -> PolicyRevision:
        record = self._record(
            run_id=run_id,
            update_id=f"{run_id}::baseline",
            parameter_group_id=parameter_group_id,
            revision=0,
            train_call_ids=("import::baseline",),
            evidence=TrainingEvidence(),
            status="published",
        )
        self.catalog.register_baseline(record, alias=f"baseline::{parameter_group_id}")
        self._parent[parameter_group_id] = record.checkpoint_id
        revision = self._revision_of(record, 0)
        self.baselines[parameter_group_id] = revision
        return revision

    def train(
        self,
        *,
        parameter_group_id: str,
        batch: Sequence[Mapping[str, Any]],
        update_id: str,
        plan_hash: str,
    ) -> TrainOutcome:
        if not batch:
            raise PortError("a train call needs at least one packed span")
        tokens = sum(int(item["trainable_tokens"]) for item in batch)
        request_id = f"train::{update_id}::{parameter_group_id}"
        self.train_calls.append(
            {
                "parameter_group_id": parameter_group_id,
                "update_id": update_id,
                "plan_hash": plan_hash,
                "examples": len(batch),
                "tokens": tokens,
                "group_ids": sorted({str(item["group_id"]) for item in batch}),
                "advantages": [float(item["advantage"]) for item in batch],
            }
        )
        return TrainOutcome(
            request_ids=(request_id,),
            examples=len(batch),
            tokens=tokens,
            provider_cost=0.0,
            metrics={"plan_hash": plan_hash},
        )

    def publish(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_groups: Sequence[str],
        outcome: Mapping[str, TrainOutcome],
    ) -> Mapping[str, PolicyRevision]:
        revision = self.revision + 1
        attempts: list[ComponentSaveAttempt] = []
        components: list[PolicySetComponent] = []
        records: dict[str, CheckpointRecord] = {}
        for parameter_group_id in sorted(parameter_groups):
            result = outcome[parameter_group_id]
            record = self._record(
                run_id=run_id,
                update_id=update_id,
                parameter_group_id=parameter_group_id,
                revision=revision,
                train_call_ids=result.request_ids,
                evidence=TrainingEvidence(
                    examples=result.examples,
                    tokens=result.tokens,
                    provider_cost=result.provider_cost,
                ),
                status="staged",
            )
            records[parameter_group_id] = record
            if parameter_group_id in self.fail_groups:
                attempts.append(
                    ComponentSaveAttempt(
                        parameter_group_id=parameter_group_id,
                        error="save refused by the provider",
                        provider_request_ids=result.request_ids,
                    )
                )
            else:
                attempts.append(
                    ComponentSaveAttempt(
                        parameter_group_id=parameter_group_id,
                        record=record,
                        provider_request_ids=result.request_ids,
                    )
                )
            components.append(
                PolicySetComponent(
                    policy_type_id=self._policy_type(parameter_group_id),
                    parameter_group_id=parameter_group_id,
                    checkpoint_id=record.checkpoint_id,
                    policy_revision_id=record.policy_revision_id,
                )
            )
        policy_set_revision_id = f"{self.policy_set_id}::{revision}"
        published = self.publisher.publish_round(
            PolicySetRevision(
                policy_set_revision_id=policy_set_revision_id,
                policy_set_id=self.policy_set_id,
                run_id=run_id,
                update_id=update_id,
                components=tuple(components),
                created_at=self.catalog.now(),
                parent_policy_set_revision_id=self._parent_set,
            ),
            attempts,
        )
        self.revision = revision
        self._parent_set = published.policy_set_revision_id
        self.published.append(policy_set_revision_id)
        revisions: dict[str, PolicyRevision] = {}
        for parameter_group_id, record in records.items():
            self._parent[parameter_group_id] = record.checkpoint_id
            revisions[parameter_group_id] = self._revision_of(record, revision)
        return revisions

    def resolve(self, selector: str) -> Mapping[str, PolicyRevision]:
        record = self.catalog.get_checkpoint(selector)
        return {
            record.parameter_group_id: self._revision_of(
                record, int(record.policy_revision_id.rsplit("::", 1)[-1])
            )
        }

    def catalog_rows(self) -> list[Mapping[str, Any]]:
        return [view.to_payload() for view in self.catalog.list_checkpoints()]

    def lineage_rows(self) -> list[Mapping[str, Any]]:
        return [edge.to_payload() for edge in self.catalog.lineage_edges()]

    def close(self) -> None:
        self.catalog.close()


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class Plane:
    """One wired plane: a fake container, a client, a gateway and a binder."""

    container: RunningContainer
    client: CanonicalClient
    gateway: RecordingGateway
    binder: CatalogBinder
    clock: PlaneClock

    def shutdown(self) -> None:
        self.binder.close()
        self.container.shutdown()

    def __enter__(self) -> "Plane":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()


def build_plane(
    config: ContainerConfig,
    tmp_path: Path,
    *,
    reward_for: Callable[[str, int], float] | None = None,
    fail_groups: Sequence[str] = (),
) -> Plane:
    """Serve a fake and wire every port the executor needs against it."""

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    container = serve(config)
    client = CanonicalClient(container, reward_for=reward_for)
    profile = config.renderer_profile
    binder = CatalogBinder(
        tmp_path / "checkpoints.sqlite3",
        renderer_profile=profile,
        contract_hash=client.contract.contract_hash,
        base_model=config.model_id,
        policy_types={
            group: policy_type
            for policy_type, group in config.topology.parameter_groups.items()
        },
        fingerprints=lambda revision: behavior_fingerprint(config, revision),
        fail_groups=fail_groups,
    )
    return Plane(
        container=container,
        client=client,
        gateway=RecordingGateway(profile=profile),
        binder=binder,
        clock=PlaneClock(container),
    )


CONFIG_TEMPLATE = """
schema_version = "cispo.container.v1"
run_id = "{run_id}"

[container]
url = "{url}"

[taskset]
train_split = "train"
evaluation_split = "eval"
train_ids = [{train_ids}]
evaluation_ids = []

[model]
provider = "fake"
id = "{model_id}"
family = "{model_family}"
policy_kind = "{policy_kind}"

[plan]
preset = "{preset}"
group_size = {group_size}
groups_per_step = {groups_per_step}
target_train_updates = {target_train_updates}
maximum_sampled_groups = {maximum_sampled_groups}
correction = {{max_weight_staleness = {maximum_policy_lag}}}
schedule = {{weight_mode = "{weight_mode}"}}

[pipeline]
max_execution_slots = {slots}
rollout_queue_capacity = {rollout_capacity}
score_queue_capacity = 8
scored_result_queue_capacity = 8
train_ready_capacity = {train_ready_capacity}
maximum_policy_lag = {maximum_policy_lag}
max_open_groups = {max_open_groups}
stale_disposition = "{stale_disposition}"

[topology]
expected_topology_id = "{topology_id}"
trainable_teams = [{trainable_teams}]
partial_roster = "{partial_roster}"

[opponents]
match_set_revision = "match-set-0001"

[reward]
optimized_channel = "{optimized_channel}"

[evaluation]
paired = false

[lifecycle]
resume_requires_rehandshake = true

[offline]
mode = "off"

[artifacts]
catalog = "checkpoints.sqlite3"
directory = "runs"
"""


def config_text(
    config: ContainerConfig,
    url: str,
    *,
    run_id: str = "run_test",
    preset: str = "cispo",
    group_size: int = 2,
    groups_per_step: int = 1,
    target_train_updates: int = 1,
    maximum_sampled_groups: int = 4,
    slots: int = 2,
    rollout_capacity: int = 8,
    train_ready_capacity: int = 1,
    maximum_policy_lag: int = 0,
    max_open_groups: int = 1,
    stale_disposition: str = "discard",
    task_ids: Sequence[str] | None = None,
    optimized_channel: str | None = None,
) -> str:
    """A ``cispo.container.v1`` document aimed at one fake."""

    rows = tuple(task_ids or config.task_ids[:1])
    teams = tuple(team.team_id for team in config.topology.teams if team.trainable)
    return CONFIG_TEMPLATE.format(
        run_id=run_id,
        url=url,
        train_ids=", ".join(f'"{item}"' for item in rows),
        model_id=config.model_id,
        model_family=config.model_family,
        policy_kind=config.policy_kind,
        preset=preset,
        group_size=group_size,
        groups_per_step=groups_per_step,
        target_train_updates=target_train_updates,
        maximum_sampled_groups=maximum_sampled_groups,
        slots=slots,
        rollout_capacity=rollout_capacity,
        train_ready_capacity=train_ready_capacity,
        maximum_policy_lag=maximum_policy_lag,
        weight_mode='async_lag' if maximum_policy_lag else 'sync_pin',
        max_open_groups=max_open_groups,
        stale_disposition=stale_disposition,
        topology_id=config.topology.topology_id,
        trainable_teams=", ".join(f'"{item}"' for item in teams),
        partial_roster=config.partial_roster_disposition,
        optimized_channel=optimized_channel or config.reward_channel_ids[0],
    )
