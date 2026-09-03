"""The one resolver both rollout and evaluation entrypoints go through.

An immutable checkpoint id resolves one policy. An immutable policy-set
revision resolves every component as an atomic team. A match-set revision
resolves the whole match, trainee side and opponent side together. Nothing
resolves to "whatever is newest": a selector that cannot be turned into an
immutable id is a refusal, and a component whose artifact is missing, whose
digest disagrees, or whose renderer/tokenizer does not match the evaluation is
an evidence failure. Silently falling back to the latest thing on hand would
produce a number that looks like a result and is not one.

Human aliases (``baseline``, ``latest-published``, ``best:<metric>``) are
optional mutable pointers. They are allowed, but the receipt always records
both the selector that was requested and the immutable id it resolved to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from synth_optimizers.contracts.rl_records import EvidenceError, RendererProfile
from synth_optimizers.rl.catalog import (
    MUTABLE_SELECTOR_TOKENS,
    SAMPLER_ROLE,
    TRAINING_STATE_ROLE,
    ArtifactRoleError,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    CheckpointView,
    EvaluationBinding,
    UnknownRecordError,
    assert_sampler_ref,
    assert_training_state_ref,
    utc_now,
)
from synth_optimizers.rl.policy_sets import (
    MatchSetRevision,
    OpponentBinding,
    PolicySetRevision,
)

RESOLUTION_RECEIPT_SCHEMA_VERSION = "cispo.resolution_receipt.v1"

LATEST_PUBLISHED_ALIAS = "latest-published"
BEST_ALIAS_PREFIX = "best:"
BASELINE_ALIAS = "baseline"

RESOLVED_KINDS = frozenset({"checkpoint", "policy_set", "match_set"})
METRIC_DIRECTIONS = frozenset({"max", "min"})


class ResolutionError(EvidenceError):
    """A selector could not be resolved to verified immutable artifacts."""


class UnknownSelectorError(ResolutionError):
    """The selector names nothing the catalog holds. Never guess a substitute."""


class MutableSelectorError(ResolutionError):
    """The selector could change meaning under the run. Refused by name."""


class AmbiguousSelectorError(ResolutionError):
    """The selector matches more than one immutable id. Narrow the scope."""


class ArtifactMissingError(ResolutionError):
    """A declared artifact reference does not exist where it claims to."""


class DigestMismatchError(ResolutionError):
    """The artifact at the reference is not the artifact that was catalogued."""


class RoleMismatchError(ResolutionError):
    """A sampler artifact was requested as training state, or the reverse."""


class CompatibilityMismatchError(ResolutionError):
    """Renderer, tokenizer, or container contract disagrees with the request."""


class UnpublishedComponentError(ResolutionError):
    """A staged or orphaned component is not a thing you can evaluate."""


class RevisionNotReadyError(ResolutionError):
    """The revision has not been loaded and health-checked."""


class RetiredRevisionError(ResolutionError):
    """The revision was retired; its artifacts are no longer guaranteed loaded."""


class ArtifactProbe(Protocol):
    """Existence and digest oracle for provider artifact references.

    Injected: the resolver never reaches a provider itself, and the probe is
    what makes "verify before an evaluation starts" a real check rather than a
    restatement of the catalog.
    """

    def exists(self, ref: str) -> bool: ...

    def digest_of(self, ref: str) -> str: ...


@dataclass(frozen=True, slots=True)
class CompatibilityRequirement:
    """What the caller needs to still be true. Unset fields are not checked."""

    renderer_profile: str | None = None
    tokenizer: str | None = None
    container_contract_hash: str | None = None

    @classmethod
    def from_renderer_profile(
        cls, profile: RendererProfile, *, container_contract_hash: str | None = None
    ) -> "CompatibilityRequirement":
        return cls(
            renderer_profile=profile.profile_id,
            tokenizer=profile.tokenizer_id,
            container_contract_hash=container_contract_hash,
        )

    def assert_satisfied_by(self, compatibility: CheckpointCompatibility, *, context: str) -> None:
        pairs = (
            ("renderer_profile", self.renderer_profile, compatibility.renderer_profile),
            ("tokenizer", self.tokenizer, compatibility.tokenizer),
            (
                "container_contract_hash",
                self.container_contract_hash,
                compatibility.container_contract_hash,
            ),
        )
        for name, required, actual in pairs:
            if required is not None and required != actual:
                raise CompatibilityMismatchError(
                    f"{context}: {name} is {actual!r}, the evaluation requires {required!r}"
                )


@dataclass(frozen=True, slots=True)
class ResolutionScope:
    """Narrows a computed alias to one run, group, or policy type."""

    run_id: str | None = None
    parameter_group_id: str | None = None
    policy_type_id: str | None = None
    policy_set_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """The exact provider reference that will be loaded, and its verified digest."""

    checkpoint_id: str
    role: str
    ref: str
    digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "role": self.role,
            "ref": self.ref,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    """One immutable policy, verified and ready to bind."""

    checkpoint_id: str
    policy_revision_id: str
    parameter_group_id: str
    policy_type_ids: tuple[str, ...]
    base_model: str
    publication_status: str
    artifact: ResolvedArtifact
    compatibility: CheckpointCompatibility

    def to_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "policy_revision_id": self.policy_revision_id,
            "parameter_group_id": self.parameter_group_id,
            "policy_type_ids": list(self.policy_type_ids),
            "base_model": self.base_model,
            "publication_status": self.publication_status,
            "artifact": self.artifact.to_payload(),
            "compatibility": self.compatibility.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class ResolvedOpponent:
    """A non-trainable participant, pinned to something that cannot move."""

    opponent_id: str
    binding_kind: str
    identity: str
    role_id: str | None = None
    artifact: ResolvedArtifact | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "binding_kind": self.binding_kind,
            "identity": self.identity,
            "role_id": self.role_id,
            "artifact": None if self.artifact is None else self.artifact.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class Resolution:
    """Requested selector plus the immutable resolution it produced."""

    requested_selector: str
    resolved_kind: str
    resolved_id: str
    role: str
    policies: tuple[ResolvedPolicy, ...]
    opponents: tuple[ResolvedOpponent, ...] = ()
    alias: str | None = None
    policy_set_revision_id: str | None = None
    match_set_revision_id: str | None = None
    resolved_at: str = ""

    def __post_init__(self) -> None:
        if self.resolved_kind not in RESOLVED_KINDS:
            raise ResolutionError(f"unknown resolved kind {self.resolved_kind!r}")
        if not self.policies:
            raise ResolutionError("a resolution must bind at least one policy")

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(policy.checkpoint_id for policy in self.policies)

    @property
    def loaded_refs(self) -> tuple[str, ...]:
        refs = [policy.artifact.ref for policy in self.policies]
        refs.extend(
            opponent.artifact.ref for opponent in self.opponents if opponent.artifact is not None
        )
        return tuple(refs)

    def policy_for_group(self, parameter_group_id: str) -> ResolvedPolicy:
        for policy in self.policies:
            if policy.parameter_group_id == parameter_group_id:
                return policy
        raise UnknownSelectorError(
            f"resolution {self.resolved_id} binds no policy for parameter group "
            f"{parameter_group_id!r}"
        )

    def to_receipt(self) -> dict[str, Any]:
        """What the evaluation receipt persists. Selector and immutable id both."""

        return {
            "schema_version": RESOLUTION_RECEIPT_SCHEMA_VERSION,
            "requested_selector": self.requested_selector,
            "alias": self.alias,
            "resolved_kind": self.resolved_kind,
            "resolved_id": self.resolved_id,
            "role": self.role,
            "policy_set_revision_id": self.policy_set_revision_id,
            "match_set_revision_id": self.match_set_revision_id,
            "resolved_checkpoint_ids": list(self.checkpoint_ids),
            "policies": [policy.to_payload() for policy in self.policies],
            "opponents": [opponent.to_payload() for opponent in self.opponents],
            "loaded_refs": list(self.loaded_refs),
            "resolved_at": self.resolved_at,
        }


@dataclass(frozen=True, slots=True)
class _Target:
    kind: str
    identifier: str
    alias: str | None = None


class EvaluationResolver:
    """Shared resolver for rollout binding and evaluation admission."""

    def __init__(
        self,
        catalog: CheckpointCatalog,
        *,
        probe: ArtifactProbe,
        clock: Callable[[], str] = utc_now,
        metric_directions: Mapping[str, str] | None = None,
        allow_staged: bool = False,
        require_ready: bool = True,
    ) -> None:
        self._catalog = catalog
        self._probe = probe
        self._clock = clock
        directions = dict(metric_directions or {})
        for metric, direction in directions.items():
            if direction not in METRIC_DIRECTIONS:
                raise ResolutionError(
                    f"metric {metric!r} direction {direction!r} must be one of "
                    f"{sorted(METRIC_DIRECTIONS)}"
                )
        self._metric_directions = directions
        self._allow_staged = allow_staged
        self._require_ready = require_ready

    @property
    def catalog(self) -> CheckpointCatalog:
        return self._catalog

    # ------------------------------------------------------------- resolving

    def resolve(
        self,
        selector: str,
        *,
        role: str = SAMPLER_ROLE,
        compatibility: CompatibilityRequirement | None = None,
        scope: ResolutionScope | None = None,
    ) -> Resolution:
        """Resolve any selector to verified immutable artifacts, or refuse."""

        target = self._target_for(selector, scope=scope)
        if target.kind == "checkpoint":
            return self._resolve_checkpoint_target(selector, target, role, compatibility)
        if target.kind == "policy_set":
            return self._resolve_policy_set_target(selector, target, role, compatibility)
        return self._resolve_match_set_target(selector, target, role, compatibility)

    def resolve_checkpoint(self, selector: str, **kwargs: Any) -> Resolution:
        resolution = self.resolve(selector, **kwargs)
        return self._expect(resolution, "checkpoint")

    def resolve_policy_set(self, selector: str, **kwargs: Any) -> Resolution:
        resolution = self.resolve(selector, **kwargs)
        return self._expect(resolution, "policy_set")

    def resolve_match_set(self, selector: str, **kwargs: Any) -> Resolution:
        resolution = self.resolve(selector, **kwargs)
        return self._expect(resolution, "match_set")

    def resolve_sampler(self, selector: str, **kwargs: Any) -> Resolution:
        """Rollout and evaluation path: immutable sampler artifacts only."""

        kwargs.pop("role", None)
        return self.resolve(selector, role=SAMPLER_ROLE, **kwargs)

    def resolve_training_state(self, selector: str, **kwargs: Any) -> Resolution:
        """Resume path: resumable artifacts only. A sampler ref is a refusal."""

        kwargs.pop("role", None)
        return self.resolve(selector, role=TRAINING_STATE_ROLE, **kwargs)

    def record_evaluation(
        self,
        evaluation_id: str,
        resolution: Resolution,
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> EvaluationBinding:
        """Append the evaluation relation. The checkpoint record is untouched."""

        return self._catalog.record_evaluation(
            EvaluationBinding(
                evaluation_id=evaluation_id,
                target_kind=resolution.resolved_kind,
                target_id=resolution.resolved_id,
                requested_selector=resolution.requested_selector,
                resolved_checkpoint_ids=resolution.checkpoint_ids,
                loaded_refs=resolution.loaded_refs,
                metrics=dict(metrics or {}),
                policy_set_revision_id=resolution.policy_set_revision_id,
                match_set_revision_id=resolution.match_set_revision_id,
            )
        )

    # ------------------------------------------------------- list / describe

    def list_checkpoints(
        self,
        *,
        run_id: str | None = None,
        update_id: str | None = None,
        parameter_group_id: str | None = None,
        policy_type_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        publication_status: str | None = None,
        base_model: str | None = None,
        train_call_id: str | None = None,
        policy_set_revision_id: str | None = None,
        evaluation_metric: str | None = None,
        limit: int | None = None,
    ) -> tuple[CheckpointView, ...]:
        """Every declared index: run, policy type, group, update, parent, status, metric."""

        return self._catalog.list_checkpoints(
            run_id=run_id,
            update_id=update_id,
            parameter_group_id=parameter_group_id,
            policy_type_id=policy_type_id,
            parent_checkpoint_id=parent_checkpoint_id,
            publication_status=publication_status,
            base_model=base_model,
            train_call_id=train_call_id,
            policy_set_revision_id=policy_set_revision_id,
            evaluation_metric=evaluation_metric,
            limit=limit,
        )

    def list_by_metric(
        self, metric: str, *, target_kind: str | None = None
    ) -> tuple[tuple[str, str, float], ...]:
        """``(target_kind, target_id, value)`` for one evaluation metric, best first."""

        return self._catalog.metric_rows(
            metric, target_kind=target_kind, direction=self._direction(metric)
        )

    def describe(self, identifier: str) -> dict[str, Any]:
        """Describe a checkpoint, policy-set revision, or match-set revision."""

        if self._catalog.has_checkpoint(identifier):
            view = self._catalog.describe_checkpoint(identifier)
            payload = view.to_payload()
            payload["record_kind"] = "checkpoint"
            payload["parent_chain"] = list(self._catalog.ancestry(identifier))
            payload["lineage_edges"] = [
                edge.to_payload()
                for edge in self._catalog.lineage_edges(child_checkpoint_id=identifier)
            ]
            payload["save_attempts"] = [
                attempt.to_payload()
                for attempt in self._catalog.save_attempts(
                    run_id=view.record.run_id,
                    update_id=view.record.update_id,
                    parameter_group_id=view.record.parameter_group_id,
                )
            ]
            return payload
        kind = self._catalog.revision_kind(identifier)
        if kind is None:
            raise UnknownSelectorError(f"{identifier!r} is absent from the catalog")
        row = self._catalog.get_revision(identifier)
        payload = dict(row.payload)
        payload["record_kind"] = kind
        payload["transitions"] = [
            {
                "transition": transition.transition,
                "attempt_id": transition.attempt_id,
                "detail": transition.detail,
                "recorded_at": transition.recorded_at,
            }
            for transition in self._catalog.revision_transitions(identifier)
        ]
        payload["active_attempts"] = list(self._catalog.active_attempts(identifier))
        payload["is_active"] = (
            self._catalog.active_revision_id(row.family_id, revision_kind=kind) == identifier
        )
        payload["evaluations"] = [
            binding.to_payload()
            for binding in self._catalog.evaluations(target_id=identifier, target_kind=kind)
        ]
        return payload

    # --------------------------------------------------------------- private

    def _direction(self, metric: str) -> str:
        return self._metric_directions.get(metric, "max")

    def _expect(self, resolution: Resolution, kind: str) -> Resolution:
        if resolution.resolved_kind != kind:
            raise UnknownSelectorError(
                f"selector {resolution.requested_selector!r} resolves a "
                f"{resolution.resolved_kind}, but a {kind} was required"
            )
        return resolution

    def _target_for(self, selector: str, *, scope: ResolutionScope | None) -> _Target:
        if not isinstance(selector, str) or not selector.strip():
            raise UnknownSelectorError("a selector is required")
        text = selector.strip()
        if text.lower() in MUTABLE_SELECTOR_TOKENS:
            raise MutableSelectorError(
                f"selector {text!r} is not an identity; it would change under the run. "
                "Pass an immutable checkpoint, policy-set, or match-set id"
            )
        if self._catalog.has_checkpoint(text):
            return _Target(kind="checkpoint", identifier=text)
        kind = self._catalog.revision_kind(text)
        if kind is not None:
            return _Target(kind=kind, identifier=text)
        return self._resolve_alias(text, scope=scope)

    def _resolve_alias(self, selector: str, *, scope: ResolutionScope | None) -> _Target:
        if selector == LATEST_PUBLISHED_ALIAS:
            return _Target(
                kind="checkpoint",
                identifier=self._latest_published(scope),
                alias=selector,
            )
        if selector.startswith(BEST_ALIAS_PREFIX):
            metric = selector[len(BEST_ALIAS_PREFIX) :].strip()
            if not metric:
                raise UnknownSelectorError("best:<metric> requires a metric name")
            return _Target(
                kind="checkpoint", identifier=self._best_by_metric(metric, scope), alias=selector
            )
        pointer = None
        if scope is not None and scope.run_id:
            pointer = self._catalog.alias(f"{selector}:{scope.run_id}")
        if pointer is None:
            pointer = self._catalog.alias(selector)
        if pointer is None:
            raise UnknownSelectorError(
                f"selector {selector!r} is neither an immutable id nor a registered alias; "
                "the resolver has no 'latest' to fall back to"
            )
        if pointer.target_kind == "checkpoint" and not self._catalog.has_checkpoint(
            pointer.target_id
        ):
            raise UnknownSelectorError(
                f"alias {selector!r} points at absent checkpoint {pointer.target_id}"
            )
        return _Target(kind=pointer.target_kind, identifier=pointer.target_id, alias=selector)

    def _latest_published(self, scope: ResolutionScope | None) -> str:
        narrow = scope or ResolutionScope()
        views = self._catalog.list_checkpoints(
            run_id=narrow.run_id,
            parameter_group_id=narrow.parameter_group_id,
            policy_type_id=narrow.policy_type_id,
            publication_status="published",
        )
        if not views:
            raise UnknownSelectorError(
                f"{LATEST_PUBLISHED_ALIAS} matched no published checkpoint in this scope"
            )
        groups = {view.record.parameter_group_id for view in views}
        if len(groups) > 1:
            raise AmbiguousSelectorError(
                f"{LATEST_PUBLISHED_ALIAS} matches parameter groups {sorted(groups)}; "
                "narrow the scope or name a policy-set revision"
            )
        return views[-1].checkpoint_id

    def _best_by_metric(self, metric: str, scope: ResolutionScope | None) -> str:
        rows = self._catalog.metric_rows(
            metric, target_kind="checkpoint", direction=self._direction(metric)
        )
        narrow = scope or ResolutionScope()
        for _kind, target_id, _value in rows:
            if not self._catalog.has_checkpoint(target_id):
                continue
            record = self._catalog.get_checkpoint(target_id)
            if narrow.run_id and record.run_id != narrow.run_id:
                continue
            if narrow.parameter_group_id and record.parameter_group_id != narrow.parameter_group_id:
                continue
            if narrow.policy_type_id and narrow.policy_type_id not in record.policy_type_ids:
                continue
            return target_id
        raise UnknownSelectorError(
            f"{BEST_ALIAS_PREFIX}{metric} matched no evaluated checkpoint in this scope"
        )

    def _resolve_checkpoint_target(
        self,
        selector: str,
        target: _Target,
        role: str,
        compatibility: CompatibilityRequirement | None,
    ) -> Resolution:
        policy = self._verify_policy(target.identifier, role, compatibility)
        return Resolution(
            requested_selector=selector,
            resolved_kind="checkpoint",
            resolved_id=target.identifier,
            role=role,
            policies=(policy,),
            alias=target.alias,
            resolved_at=self._clock(),
        )

    def _resolve_policy_set_target(
        self,
        selector: str,
        target: _Target,
        role: str,
        compatibility: CompatibilityRequirement | None,
    ) -> Resolution:
        revision = self._policy_set(target.identifier)
        self._assert_bindable(target.identifier)
        policies = tuple(
            self._verify_policy(
                component.checkpoint_id,
                role,
                compatibility,
                expected_policy_type=component.policy_type_id,
                expected_group=component.parameter_group_id,
            )
            for component in revision.components
        )
        return Resolution(
            requested_selector=selector,
            resolved_kind="policy_set",
            resolved_id=target.identifier,
            role=role,
            policies=policies,
            alias=target.alias,
            policy_set_revision_id=target.identifier,
            resolved_at=self._clock(),
        )

    def _resolve_match_set_target(
        self,
        selector: str,
        target: _Target,
        role: str,
        compatibility: CompatibilityRequirement | None,
    ) -> Resolution:
        match = self._match_set(target.identifier)
        self._assert_bindable(target.identifier)
        trainee = self._policy_set(match.policy_set_revision_id)
        policies = tuple(
            self._verify_policy(
                component.checkpoint_id,
                role,
                compatibility,
                expected_policy_type=component.policy_type_id,
                expected_group=component.parameter_group_id,
            )
            for component in trainee.components
        )
        opponents = tuple(
            self._verify_opponent(opponent, compatibility) for opponent in match.opponents
        )
        return Resolution(
            requested_selector=selector,
            resolved_kind="match_set",
            resolved_id=target.identifier,
            role=role,
            policies=policies,
            opponents=opponents,
            alias=target.alias,
            policy_set_revision_id=match.policy_set_revision_id,
            match_set_revision_id=target.identifier,
            resolved_at=self._clock(),
        )

    def _policy_set(self, revision_id: str) -> PolicySetRevision:
        try:
            row = self._catalog.get_revision(revision_id)
        except UnknownRecordError as error:
            raise UnknownSelectorError(str(error)) from error
        if row.revision_kind != "policy_set":
            raise UnknownSelectorError(
                f"revision {revision_id} is a {row.revision_kind}, not a policy set"
            )
        return PolicySetRevision.from_payload(row.payload)

    def _match_set(self, revision_id: str) -> MatchSetRevision:
        row = self._catalog.get_revision(revision_id)
        if row.revision_kind != "match_set":
            raise UnknownSelectorError(
                f"revision {revision_id} is a {row.revision_kind}, not a match set"
            )
        return MatchSetRevision.from_payload(row.payload)

    def _assert_bindable(self, revision_id: str) -> None:
        if self._catalog.has_transition(revision_id, "retire"):
            raise RetiredRevisionError(
                f"revision {revision_id} is retired; its artifacts are not guaranteed resident"
            )
        if self._require_ready and not self._catalog.has_transition(revision_id, "ready"):
            raise RevisionNotReadyError(
                f"revision {revision_id} was never marked ready: its sampler artifacts have not "
                "been materialized and health-checked"
            )

    def _verify_policy(
        self,
        checkpoint_id: str,
        role: str,
        compatibility: CompatibilityRequirement | None,
        *,
        expected_policy_type: str | None = None,
        expected_group: str | None = None,
    ) -> ResolvedPolicy:
        try:
            record = self._catalog.get_checkpoint(checkpoint_id)
        except UnknownRecordError as error:
            raise UnknownSelectorError(str(error)) from error
        status = self._catalog.publication_status(checkpoint_id)
        if status == "orphaned" or (status == "staged" and not self._allow_staged):
            raise UnpublishedComponentError(
                f"checkpoint {checkpoint_id} is {status}; it is not an evaluable policy"
            )
        if expected_policy_type is not None and expected_policy_type not in record.policy_type_ids:
            raise RoleMismatchError(
                f"checkpoint {checkpoint_id} does not serve policy type {expected_policy_type!r}"
            )
        if expected_group is not None and record.parameter_group_id != expected_group:
            raise RoleMismatchError(
                f"checkpoint {checkpoint_id} belongs to parameter group "
                f"{record.parameter_group_id!r}, bound as {expected_group!r}"
            )
        artifact = self._verify_artifact(record, role)
        if compatibility is not None:
            compatibility.assert_satisfied_by(
                record.compatibility, context=f"checkpoint {checkpoint_id}"
            )
        return ResolvedPolicy(
            checkpoint_id=record.checkpoint_id,
            policy_revision_id=record.policy_revision_id,
            parameter_group_id=record.parameter_group_id,
            policy_type_ids=record.policy_type_ids,
            base_model=record.base_model,
            publication_status=status,
            artifact=artifact,
            compatibility=record.compatibility,
        )

    def _verify_opponent(
        self, opponent: OpponentBinding, compatibility: CompatibilityRequirement | None
    ) -> ResolvedOpponent:
        if not opponent.is_pinned_checkpoint:
            return ResolvedOpponent(
                opponent_id=opponent.opponent_id,
                binding_kind=opponent.binding_kind,
                identity=opponent.identity,
                role_id=opponent.role_id,
            )
        if not self._catalog.has_checkpoint(opponent.identity):
            raise UnknownSelectorError(
                f"opponent {opponent.opponent_id} pins checkpoint {opponent.identity}, "
                "which is absent from the catalog"
            )
        record = self._catalog.get_checkpoint(opponent.identity)
        # An opponent is sampled, never trained: only the sampler role applies.
        artifact = self._verify_artifact(record, SAMPLER_ROLE)
        if compatibility is not None:
            compatibility.assert_satisfied_by(
                record.compatibility, context=f"opponent {opponent.opponent_id}"
            )
        return ResolvedOpponent(
            opponent_id=opponent.opponent_id,
            binding_kind=opponent.binding_kind,
            identity=opponent.identity,
            role_id=opponent.role_id,
            artifact=artifact,
        )

    def _verify_artifact(self, record: CheckpointRecord, role: str) -> ResolvedArtifact:
        try:
            reference = record.artifacts.ref_for_role(role)
        except ArtifactRoleError as error:
            raise RoleMismatchError(f"checkpoint {record.checkpoint_id}: {error}") from error
        if role == SAMPLER_ROLE:
            assert_sampler_ref(reference)
        else:
            assert_training_state_ref(reference)
        if not self._probe.exists(reference.ref):
            raise ArtifactMissingError(
                f"checkpoint {record.checkpoint_id} {role} artifact {reference.ref} does not exist"
            )
        observed = self._probe.digest_of(reference.ref)
        if observed != reference.digest:
            raise DigestMismatchError(
                f"checkpoint {record.checkpoint_id} {role} artifact {reference.ref} digests "
                f"{observed!r}, catalogued as {reference.digest!r}"
            )
        return ResolvedArtifact(
            checkpoint_id=record.checkpoint_id,
            role=role,
            ref=reference.ref,
            digest=reference.digest,
        )


@dataclass(frozen=True, slots=True)
class MappingArtifactProbe:
    """Probe over a ``ref -> digest`` mapping. For offline and replay paths."""

    digests: Mapping[str, str]

    def exists(self, ref: str) -> bool:
        return ref in self.digests

    def digest_of(self, ref: str) -> str:
        try:
            return self.digests[ref]
        except KeyError as error:
            raise ArtifactMissingError(f"artifact {ref} does not exist") from error


def selectors_are_immutable(selectors: Sequence[str]) -> None:
    """Refuse a batch of selectors that contains a moving pointer."""

    for selector in selectors:
        if not isinstance(selector, str) or selector.strip().lower() in MUTABLE_SELECTOR_TOKENS:
            raise MutableSelectorError(f"selector {selector!r} is not an immutable identity")


__all__ = [
    "AmbiguousSelectorError",
    "ArtifactMissingError",
    "ArtifactProbe",
    "BASELINE_ALIAS",
    "BEST_ALIAS_PREFIX",
    "CompatibilityMismatchError",
    "CompatibilityRequirement",
    "DigestMismatchError",
    "EvaluationResolver",
    "LATEST_PUBLISHED_ALIAS",
    "MappingArtifactProbe",
    "MutableSelectorError",
    "RESOLUTION_RECEIPT_SCHEMA_VERSION",
    "ResolutionError",
    "ResolutionScope",
    "ResolvedArtifact",
    "ResolvedOpponent",
    "ResolvedPolicy",
    "Resolution",
    "RetiredRevisionError",
    "RevisionNotReadyError",
    "RoleMismatchError",
    "UnknownSelectorError",
    "UnpublishedComponentError",
    "selectors_are_immutable",
]
