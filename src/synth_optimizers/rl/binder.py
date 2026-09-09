"""The bridge from a training provider to the checkpoint catalog, in that order.

A revision exists when it is catalogued, not when a provider returns a path. So
every method here materializes through the provider and then registers, and a
failure between the two is recorded rather than swallowed: the artifact was
paid for, and losing it would lose the evidence as well as the spend.

Publication is atomic and happens once per published round, never once per
packed training group. A one-sided failure leaves the previously active policy
set live and catalogues whatever did materialize as an orphan. Resolution goes
through the shared resolver and records the requested selector beside the
immutable id it produced; nothing here has a ``latest`` to fall back to.

Nothing in this module names a task, a harness, an environment, or an algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..contracts.rl_records import (
    BehaviorFingerprint,
    RendererProfile,
    SamplingProfile,
    digest,
)
from ..providers.protocols import (
    ProviderCheckpoint,
    ProviderSession,
    TrainingProvider,
    TrainingStepRequest,
)
from .catalog import (
    CheckpointArtifacts,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    SamplerWeightsRef,
    TrainingEvidence,
    TrainingStateRef,
    checkpoint_id_for,
    utc_now,
)
from .policy_sets import (
    ComponentSaveAttempt,
    PolicySetComponent,
    PolicySetPublisher,
    PolicySetRevision,
)
from .ports import PolicyRevision, PortError, TrainOutcome
from .resolver import (
    CompatibilityRequirement,
    EvaluationResolver,
    ResolutionScope,
    selectors_are_immutable,
)

BINDER_SCHEMA_VERSION = "cispo.policy_binder.v1"

BASELINE_UPDATE_ID = "update_0000"
BASELINE_ALIAS = "baseline"
SAMPLER_KIND = "sampler_weights"
TRAINING_STATE_KIND = "training_state"


class BinderError(PortError):
    """The binder refused. A revision that is not catalogued does not exist."""


class BaselineRequiredError(BinderError):
    """Training or publication was asked for before a baseline was catalogued."""


class ProviderArtifactError(BinderError):
    """The provider returned something that cannot be catalogued as an artifact."""


class RevisionNumberError(BinderError):
    """A catalogued revision id carries no integer revision to bind against."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BinderError(f"{name} is required")
    return value.strip()


def revision_number_of(policy_revision_id: str) -> int:
    """``group@7`` -> ``7``. The queue counts revisions; the catalog names them."""

    _, separator, tail = policy_revision_id.rpartition("@")
    if not separator or not tail.isdigit():
        raise RevisionNumberError(
            f"policy revision id {policy_revision_id!r} carries no integer revision; "
            "a group pin needs both identities"
        )
    return int(tail)


class CatalogPolicyBinder:
    """A :class:`~synth_optimizers.rl.ports.PolicyBinder` over a provider and the catalog.

    One binder per run. Each parameter group gets its own provider session, so
    two groups are two sets of weights rather than one set with two names.
    """

    def __init__(
        self,
        provider: TrainingProvider,
        publisher: PolicySetPublisher,
        resolver: EvaluationResolver,
        *,
        base_model: str,
        model_family: str,
        renderer_profile: RendererProfile,
        container_contract_hash: str,
        policy_set_id: str,
        wire_api: str,
        sampling_transport: str,
        loss_name: str,
        policy_types: Mapping[str, Sequence[str]] | None = None,
        sampling: SamplingProfile | None = None,
        rank: int = 8,
        learning_rate: float = 2e-5,
        eps_low: float = 1.0,
        eps_high: float = 4.0,
        seed: int = 0,
        save_training_state: bool = False,
        resume_from_checkpoint: str | None = None,
        health_check: Callable[[Any], bool] | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if publisher.catalog is not resolver.catalog:
            raise BinderError(
                "the publisher and the resolver must share one catalog, or a published "
                "revision would resolve against a different set of records"
            )
        self._provider = provider
        self._publisher = publisher
        self._resolver = resolver
        self._base_model = _text(base_model, "base_model")
        self._model_family = _text(model_family, "model_family")
        self._profile = renderer_profile
        self._contract_hash = _text(container_contract_hash, "container_contract_hash")
        self._policy_set_id = _text(policy_set_id, "policy_set_id")
        self._wire_api = _text(wire_api, "wire_api")
        self._sampling_transport = _text(sampling_transport, "sampling_transport")
        self._loss_name = _text(loss_name, "loss_name")
        self._policy_types = {
            group: tuple(str(item) for item in types)
            for group, types in dict(policy_types or {}).items()
        }
        self._sampling = sampling or SamplingProfile()
        self._rank = int(rank)
        self._learning_rate = float(learning_rate)
        if self._learning_rate <= 0:
            raise BinderError("learning_rate must be positive")
        self._eps_low = float(eps_low)
        self._eps_high = float(eps_high)
        self._seed = int(seed)
        self._save_training_state = bool(save_training_state)
        self._resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint is not None:
            selectors_are_immutable((resume_from_checkpoint,))
        self._health_check = health_check
        self._clock = clock
        self._run_id: str | None = None
        self._sessions: dict[str, ProviderSession] = {}
        self._revisions: dict[str, PolicyRevision] = {}
        self._train_calls: dict[tuple[str, str], tuple[str, ...]] = {}
        self._packed_groups: dict[tuple[str, str], tuple[str, ...]] = {}
        self._receipts: list[Mapping[str, Any]] = []
        self._resume_identity: dict[str, str] = {}

    # ------------------------------------------------------------- identity

    @property
    def catalog(self) -> CheckpointCatalog:
        return self._publisher.catalog

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def revision_for(self, parameter_group_id: str) -> PolicyRevision:
        try:
            return self._revisions[parameter_group_id]
        except KeyError as error:
            raise BaselineRequiredError(
                f"parameter group {parameter_group_id!r} has no catalogued revision"
            ) from error

    def resolution_receipts(self) -> tuple[Mapping[str, Any], ...]:
        """Every selector this binder resolved, beside what it resolved to."""

        return tuple(self._receipts)

    def resume_artifact_identity(self) -> Mapping[str, str]:
        """Exact independently verified training-state identity used to resume."""

        payload = self._resume_identity or getattr(self._provider, "_resume_artifact_identity", {})
        if not isinstance(payload, Mapping):
            return {}
        reference = payload.get("ref")
        artifact_digest = payload.get("digest")
        if not isinstance(reference, str) or not isinstance(artifact_digest, str):
            return {}
        return {"ref": reference, "digest": artifact_digest}

    # ------------------------------------------------------------- baseline

    def baseline(self, *, run_id: str, parameter_group_id: str, save_training_state: bool = False) -> PolicyRevision:
        """Materialize and catalogue the imported baseline before any attempt."""

        run = _text(run_id, "run_id")
        group = _text(parameter_group_id, "parameter_group_id")
        if self._run_id is None:
            self._run_id = run
        elif self._run_id != run:
            raise BinderError(
                f"binder is bound to run {self._run_id!r} and was asked for {run!r}"
            )
        existing = self._revisions.get(group)
        if existing is not None:
            if existing.metadata.get("update_id") != BASELINE_UPDATE_ID:
                raise BinderError(
                    f"parameter group {group} is already at revision {existing.revision}; "
                    "a baseline cannot be re-imported under a trained run"
                )
            return existing
        parent_checkpoint_id: str | None = None
        baseline_revision = 0
        if self._resume_from_checkpoint is not None:
            resolution = self._resolver.resolve_training_state(
                self._resume_from_checkpoint,
                scope=ResolutionScope(parameter_group_id=group),
                compatibility=CompatibilityRequirement.from_renderer_profile(
                    self._profile, container_contract_hash=self._contract_hash
                ),
            )
            policy = resolution.policy_for_group(group)
            if policy.base_model != self._base_model:
                raise BinderError(
                    f"resume checkpoint base model {policy.base_model!r} does not match "
                    f"configured model {self._base_model!r}"
                )
            artifact = policy.artifact
            restored = ProviderCheckpoint(
                checkpoint_id=policy.checkpoint_id,
                provider_reference=artifact.ref,
                step=revision_number_of(policy.policy_revision_id),
                digest=artifact.digest,
                kind=TRAINING_STATE_KIND,
                resume_token=artifact.ref,
                model_id=policy.base_model,
            )
            request_id = "restore-" + digest(
                {"run_id": run, "parameter_group_id": group, "checkpoint_id": policy.checkpoint_id},
                length=32,
            )
            session = self._provider.restore_session(restored, request_id=request_id)
            self._resume_identity = {'ref': artifact.ref, 'digest': artifact.digest}
            self._sessions[group] = session
            parent_checkpoint_id = policy.checkpoint_id
            baseline_revision = revision_number_of(policy.policy_revision_id)
            self._receipts.append(resolution.to_receipt())
        else:
            session = self._session_for(group)
        checkpoint = self._save(
            session,
            group,
            step=baseline_revision,
            kind=SAMPLER_KIND,
            update_id=BASELINE_UPDATE_ID,
        )
        policy_revision_id = f"{group}@{baseline_revision}"
        training_state = self._save(session, group, step=baseline_revision,
            kind=TRAINING_STATE_KIND, update_id=BASELINE_UPDATE_ID) if save_training_state else None
        record = self._record(
            run_id=run,
            update_id=BASELINE_UPDATE_ID,
            parameter_group_id=group,
            policy_revision_id=policy_revision_id,
            sampler=checkpoint,
            training_state=training_state,
            parent_checkpoint_id=parent_checkpoint_id,
            train_call_ids=(),
            evidence=TrainingEvidence(),
        )
        self.catalog.register_baseline(
            record,
            alias=f"{BASELINE_ALIAS}.{group}",
            resumed=parent_checkpoint_id is not None,
        )
        if self.catalog.alias(f"{BASELINE_ALIAS}:{run}") is None:
            self.catalog.put_alias(f"{BASELINE_ALIAS}:{run}", "checkpoint", record.checkpoint_id)
        revision = self._revision(
            record=record,
            revision=baseline_revision,
            sampler=checkpoint,
            training_state=training_state,
            policy_set_revision_id=None,
        )
        self._revisions[group] = revision
        return revision

    # ------------------------------------------------------------- training

    def train(
        self,
        *,
        parameter_group_id: str,
        batch: Sequence[Mapping[str, Any]],
        update_id: str,
        plan_hash: str,
    ) -> TrainOutcome:
        """One provider training step for one parameter group."""

        group = _text(parameter_group_id, "parameter_group_id")
        update = _text(update_id, "update_id")
        plan = _text(plan_hash, "plan_hash")
        run = self._require_run()
        self.catalog.assert_baseline_registered(run)
        if group not in self._revisions:
            raise BaselineRequiredError(
                f"parameter group {group!r} has no catalogued baseline; a revision that is "
                "not catalogued does not exist"
            )
        rows = tuple(dict(row) for row in batch)
        if not rows:
            raise BinderError(f"update {update} for {group} carries no training examples")
        prior = self._train_calls.get((update, group), ())
        request_id = "train-" + digest(
            {
                "run_id": run,
                "update_id": update,
                "parameter_group_id": group,
                "plan_hash": plan,
                "attempt": len(prior),
            },
            length=32,
        )
        result = self._provider.train_step(
            self._sessions[group],
            TrainingStepRequest(
                request_id=request_id,
                loss_name=self._loss_name,
                data=rows,
                metadata={
                    "run_id": run,
                    "update_id": update,
                    "parameter_group_id": group,
                    "plan_hash": plan,
                    "learning_rate": self._learning_rate,
                    "eps_clip": self._eps_low,
                    "eps_clip_high": self._eps_high,
                },
            ),
        )
        packed = _packed_group_ids(rows)
        self._train_calls[(update, group)] = prior + (request_id,)
        self._packed_groups[(update, group)] = (
            self._packed_groups.get((update, group), ()) + packed
        )
        usage = result.usage
        loss_weights = [float(row.get("loss_weight", 0.0)) for row in rows]
        return TrainOutcome(
            request_ids=(request_id,),
            examples=len(rows),
            tokens=int(usage.training_tokens),
            provider_cost=float(usage.cost_usd or 0.0),
            metrics={
                **dict(result.metrics),
                "step": result.step,
                "plan_hash": plan,
                "packed_group_ids": list(packed),
                "cost_missing": usage.cost_missing,
                "loss_weight_nonzero": sum(weight != 0.0 for weight in loss_weights),
                "loss_weight_l1": sum(abs(weight) for weight in loss_weights),
                "loss_weight_token_mass": sum(
                    abs(weight) * sum(bool(flag) for flag in row.get("loss_mask", ()))
                    for weight, row in zip(loss_weights, rows, strict=True)
                ),
                "loss_weight_l2_squared": sum(weight * weight for weight in loss_weights),
                "loss_weight_min": min(loss_weights),
                "loss_weight_max": max(loss_weights),
            },
        )

    # ---------------------------------------------------------- publication

    def publish(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_groups: Sequence[str],
        outcome: Mapping[str, TrainOutcome],
    ) -> Mapping[str, PolicyRevision]:
        """One sampler artifact per group per published round, published atomically."""

        run = _text(run_id, "run_id")
        update = _text(update_id, "update_id")
        if run != self._require_run():
            raise BinderError(f"binder is bound to run {self._run_id!r} and was asked for {run!r}")
        groups = tuple(
            dict.fromkeys(_text(group, "parameter_group_id") for group in parameter_groups)
        )
        if not groups:
            raise BinderError(f"update {update} publishes no parameter group")
        for group in groups:
            if group not in self._revisions:
                raise BaselineRequiredError(
                    f"parameter group {group!r} has no catalogued baseline to succeed"
                )
            if group not in outcome:
                raise BinderError(
                    f"update {update} publishes {group!r} without its training outcome; "
                    "a checkpoint carries the evidence of what produced it"
                )
        policy_set_revision_id = f"{self._policy_set_id}@{update}"
        plan = self._plan(run, update, groups, outcome)
        attempts: list[ComponentSaveAttempt] = []
        for group, component in plan.items():
            packed = self._packed_groups.get((update, group), ()) or tuple(
                str(value) for value in outcome[group].metrics.get("packed_group_ids") or ()
            )
            request_ids = self._train_calls.get((update, group), ()) or outcome[group].request_ids
            try:
                record = self._materialize(
                    run=run,
                    update=update,
                    group=group,
                    component=component,
                    outcome=outcome[group],
                    packed=packed,
                    request_ids=request_ids,
                )
            except Exception as error:  # noqa: BLE001 - a failed save is evidence
                attempts.append(
                    ComponentSaveAttempt(
                        parameter_group_id=group,
                        error=f"{type(error).__name__}: {error}",
                        packed_group_ids=packed,
                        provider_request_ids=request_ids,
                    )
                )
                continue
            attempts.append(
                ComponentSaveAttempt(
                    parameter_group_id=group,
                    record=record,
                    packed_group_ids=packed,
                    provider_request_ids=request_ids,
                )
            )
        revision = PolicySetRevision(
            policy_set_revision_id=policy_set_revision_id,
            policy_set_id=self._policy_set_id,
            run_id=run,
            update_id=update,
            components=tuple(
                PolicySetComponent(
                    policy_type_id=component["policy_type_id"],
                    parameter_group_id=group,
                    checkpoint_id=component["checkpoint_id"],
                    policy_revision_id=component["policy_revision_id"],
                )
                for group, component in plan.items()
            ),
            created_at=self._clock(),
            parent_policy_set_revision_id=self.catalog.active_revision_id(self._policy_set_id),
        )
        self._publisher.publish_round(revision, attempts)
        if self._health_check is not None:
            self._publisher.mark_loaded(policy_set_revision_id, detail=update)
            self._publisher.mark_ready(policy_set_revision_id, health_check=self._health_check)
        published: dict[str, PolicyRevision] = {}
        for attempt in attempts:
            record = attempt.record
            if record is None:  # pragma: no cover - publish_round already refused
                raise BinderError("publish_round returned on a failed component")
            group = attempt.parameter_group_id
            revision_for_group = self._revision(
                record=record,
                revision=revision_number_of(record.policy_revision_id),
                sampler=plan[group]["sampler"],
                training_state=plan[group]["training_state"],
                policy_set_revision_id=policy_set_revision_id,
            )
            self._revisions[group] = revision_for_group
            published[group] = revision_for_group
        return published

    # ----------------------------------------------------------- resolution

    def resolve(self, selector: str) -> Mapping[str, PolicyRevision]:
        """An immutable id or an alias, recorded beside what it resolved to."""

        wanted = _text(selector, "selector")
        selectors_are_immutable((wanted,))
        resolution = self._resolver.resolve_sampler(
            wanted,
            compatibility=CompatibilityRequirement.from_renderer_profile(
                self._profile, container_contract_hash=self._contract_hash
            ),
            scope=ResolutionScope(run_id=self._run_id),
        )
        receipt = resolution.to_receipt()
        evaluation_id = "resolve-" + digest(receipt, length=32)
        self._resolver.record_evaluation(evaluation_id, resolution)
        self._receipts.append({**receipt, "evaluation_id": evaluation_id})
        resolved: dict[str, PolicyRevision] = {}
        for policy in resolution.policies:
            number = revision_number_of(policy.policy_revision_id)
            resolved[policy.parameter_group_id] = PolicyRevision(
                revision=number,
                revision_id=policy.policy_revision_id,
                checkpoint_id=policy.checkpoint_id,
                parameter_group_id=policy.parameter_group_id,
                sampler_reference=policy.artifact.ref,
                behavior_fingerprint=self._fingerprint(number),
                training_state_reference=None,
                policy_set_revision_id=resolution.policy_set_revision_id,
                metadata={
                    "run_id": self._run_id,
                    "requested_selector": resolution.requested_selector,
                    "resolved_kind": resolution.resolved_kind,
                    "resolved_id": resolution.resolved_id,
                    "evaluation_id": evaluation_id,
                    "sampler_digest": policy.artifact.digest,
                    "publication_status": policy.publication_status,
                    "base_model": policy.base_model,
                    "policy_type_ids": list(policy.policy_type_ids),
                },
            )
        return resolved

    # -------------------------------------------------------------- private

    def _require_run(self) -> str:
        if self._run_id is None:
            raise BaselineRequiredError(
                "no baseline has been catalogued for this binder; register the imported "
                "baseline before training, publishing, or admitting an attempt"
            )
        return self._run_id

    def _session_for(self, parameter_group_id: str) -> ProviderSession:
        session = self._sessions.get(parameter_group_id)
        if session is not None:
            return session
        request_id = "session-" + digest(
            {
                "run_id": self._run_id,
                "parameter_group_id": parameter_group_id,
                "base_model": self._base_model,
                "rank": self._rank,
                "seed": self._seed,
            },
            length=32,
        )
        session = self._provider.create_session(
            self._base_model, rank=self._rank, seed=self._seed, request_id=request_id
        )
        self._sessions[parameter_group_id] = session
        return session

    def _save(
        self,
        session: ProviderSession,
        parameter_group_id: str,
        *,
        step: int,
        kind: str,
        update_id: str,
    ) -> ProviderCheckpoint:
        request_id = "save-" + digest(
            {
                "run_id": self._run_id,
                "update_id": update_id,
                "parameter_group_id": parameter_group_id,
                "kind": kind,
                "step": step,
            },
            length=32,
        )
        checkpoint = self._provider.save_checkpoint(
            session, step=step, kind=kind, request_id=request_id
        )
        if not str(checkpoint.provider_reference or "").strip():
            raise ProviderArtifactError(
                f"provider returned a {kind} checkpoint with no reference for "
                f"{parameter_group_id}"
            )
        return checkpoint

    def _policy_type_ids(self, parameter_group_id: str) -> tuple[str, ...]:
        declared = self._policy_types.get(parameter_group_id)
        return declared if declared else (parameter_group_id,)

    def _fingerprint(self, revision: int) -> str:
        return BehaviorFingerprint(
            renderer_profile=self._profile,
            model_family=self._model_family,
            model_id=self._base_model,
            policy_revision=revision,
            wire_api=self._wire_api,
            sampling_transport=self._sampling_transport,
            sampling=self._sampling,
        ).value

    def _record(
        self,
        *,
        run_id: str,
        update_id: str,
        parameter_group_id: str,
        policy_revision_id: str,
        sampler: ProviderCheckpoint,
        training_state: ProviderCheckpoint | None,
        parent_checkpoint_id: str | None,
        train_call_ids: Sequence[str],
        evidence: TrainingEvidence,
    ) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=checkpoint_id_for(
                run_id=run_id,
                update_id=update_id,
                parameter_group_id=parameter_group_id,
                policy_revision_id=policy_revision_id,
            ),
            run_id=run_id,
            update_id=update_id,
            train_call_ids=tuple(train_call_ids),
            parameter_group_id=parameter_group_id,
            policy_type_ids=self._policy_type_ids(parameter_group_id),
            policy_revision_id=policy_revision_id,
            base_model=self._base_model,
            artifacts=CheckpointArtifacts(
                sampler_weights=SamplerWeightsRef(
                    ref=sampler.provider_reference, digest=sampler.digest
                ),
                training_state=None
                if training_state is None
                else TrainingStateRef(
                    ref=training_state.provider_reference, digest=training_state.digest
                ),
            ),
            training_evidence=evidence,
            compatibility=CheckpointCompatibility.from_renderer_profile(
                self._profile, container_contract_hash=self._contract_hash
            ),
            created_at=self._clock(),
            parent_checkpoint_id=parent_checkpoint_id,
            publication_status="staged",
        )

    def _plan(
        self,
        run: str,
        update: str,
        groups: Sequence[str],
        outcome: Mapping[str, TrainOutcome],
    ) -> dict[str, dict[str, Any]]:
        plan: dict[str, dict[str, Any]] = {}
        for group in groups:
            number = self._revisions[group].revision + 1
            policy_revision_id = f"{group}@{number}"
            plan[group] = {
                "revision": number,
                "policy_revision_id": policy_revision_id,
                "policy_type_id": self._policy_type_ids(group)[0],
                "checkpoint_id": checkpoint_id_for(
                    run_id=run,
                    update_id=update,
                    parameter_group_id=group,
                    policy_revision_id=policy_revision_id,
                ),
                "sampler": None,
                "training_state": None,
            }
        return plan

    def _materialize(
        self,
        *,
        run: str,
        update: str,
        group: str,
        component: dict[str, Any],
        outcome: TrainOutcome,
        packed: Sequence[str],
        request_ids: Sequence[str],
    ) -> CheckpointRecord:
        session = self._sessions[group]
        sampler = self._save(
            session, group, step=component["revision"], kind=SAMPLER_KIND, update_id=update
        )
        state: ProviderCheckpoint | None = None
        if self._save_training_state:
            state = self._save(
                session,
                group,
                step=component["revision"],
                kind=TRAINING_STATE_KIND,
                update_id=update,
            )
        component["sampler"] = sampler
        component["training_state"] = state
        return self._record(
            run_id=run,
            update_id=update,
            parameter_group_id=group,
            policy_revision_id=component["policy_revision_id"],
            sampler=sampler,
            training_state=state,
            parent_checkpoint_id=self._revisions[group].checkpoint_id,
            train_call_ids=tuple(request_ids),
            evidence=TrainingEvidence(
                groups=tuple(packed),
                examples=outcome.examples,
                tokens=outcome.tokens,
                provider_cost=outcome.provider_cost,
            ),
        )

    def _revision(
        self,
        *,
        record: CheckpointRecord,
        revision: int,
        sampler: ProviderCheckpoint | None,
        training_state: ProviderCheckpoint | None,
        policy_set_revision_id: str | None,
    ) -> PolicyRevision:
        sampler_ref = record.sampler_weights
        state_ref = record.artifacts.training_state
        return PolicyRevision(
            revision=revision,
            revision_id=record.policy_revision_id,
            checkpoint_id=record.checkpoint_id,
            parameter_group_id=record.parameter_group_id,
            sampler_reference=sampler_ref.ref,
            behavior_fingerprint=self._fingerprint(revision),
            training_state_reference=None if state_ref is None else state_ref.ref,
            policy_set_revision_id=policy_set_revision_id,
            metadata={
                "run_id": record.run_id,
                "update_id": record.update_id,
                "sampler_digest": sampler_ref.digest,
                "training_state_digest": None if state_ref is None else state_ref.digest,
                "base_model": record.base_model,
                "policy_type_ids": list(record.policy_type_ids),
                "provider_checkpoint_id": None if sampler is None else sampler.checkpoint_id,
                "provider_training_state_id": None
                if training_state is None
                else training_state.checkpoint_id,
                "parent_checkpoint_id": record.parent_checkpoint_id,
                "schema_version": BINDER_SCHEMA_VERSION,
            },
        )


def _packed_group_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The rollout groups packed into one training call, in first-seen order."""

    seen: list[str] = []
    for row in rows:
        value = row.get("group_id")
        if isinstance(value, str) and value.strip() and value not in seen:
            seen.append(value)
    return tuple(seen)


__all__ = [
    "BASELINE_ALIAS",
    "BASELINE_UPDATE_ID",
    "BINDER_SCHEMA_VERSION",
    "BaselineRequiredError",
    "BinderError",
    "CatalogPolicyBinder",
    "ProviderArtifactError",
    "RevisionNumberError",
    "SAMPLER_KIND",
    "TRAINING_STATE_KIND",
    "revision_number_of",
]
