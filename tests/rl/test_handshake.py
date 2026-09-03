"""Stream 2: capability preflight and the readiness agreement, before spend."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from synth_optimizers.contracts.rl_clauses import (
    HANDSHAKE_SCHEMA_VERSION,
    MANDATORY_CLAUSES,
    OPTIONAL_CLAUSES,
)
from synth_optimizers.contracts.rl_records import RendererProfile
from synth_optimizers.rl.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    CapabilityDocument,
    CapabilityDriftError,
    CapabilityHashError,
    ClauseResult,
    ExecutorRequirements,
    PreflightRejected,
    canonical_capability_hash,
    check_requirements,
    preflight_capabilities,
)
from synth_optimizers.rl.contract import CISPO_CONTRACT_VERSION, ContainerContract
from synth_optimizers.rl.handshake import (
    AgreementMismatch,
    ClauseRejected,
    HandshakeExpired,
    HandshakeLedger,
    HandshakeRequest,
    HandshakeRevoked,
    HandshakeVerdict,
    Obligations,
    OptimizerIdentity,
    PlanNotLowerable,
    PolicyRequest,
    RenegotiationRequired,
    RunPlan,
    ClockStamp,
    TaskResolution,
    TasksetRequest,
    TopologyExpectation,
    UnknownHandshake,
    build_request,
    compute_agreement_digest,
    evaluate_handshake,
    format_rfc3339,
)
NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)
TASK_IDS = ("task-a", "task-b")

PROFILE = RendererProfile(
    profile_id="renderers.pinned.low.v1",
    package="renderers",
    package_version="0.1.11",
    config_digest="sha256:cfg",
    tokenizer_id="vendor/policy-20b",
    tokenizer_digest="sha256:tok",
    stop_token_ids=(200002, 199999),
)


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def capability_payload(**overrides: Any) -> dict[str, Any]:
    """A compliant capability document, hashed the way the container hashes it."""

    document: dict[str, Any] = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "container_id": "container-1",
        "container_image_digest": "sha256:image",
        "contract_version": CISPO_CONTRACT_VERSION,
        "renderer_profile": {
            "profile_id": PROFILE.profile_id,
            "package": PROFILE.package,
            "package_version": PROFILE.package_version,
            "config_digest": PROFILE.config_digest,
            "tokenizer_id": PROFILE.tokenizer_id,
            "tokenizer_digest": PROFILE.tokenizer_digest,
            "stop_token_ids": list(PROFILE.stop_token_ids),
            "modalities": ["text"],
            "add_generation_prompt": True,
        },
        "discovery": {
            "taskset_id": "taskset-1",
            "taskset_version": "3",
            "splits": ["train", "eval"],
            "task_content_digests": True,
            "deterministic_lookup": True,
            "duplicate_free": True,
        },
        "policy": {
            "binding_transport": "message_in_capture_out",
            "wire_api": "chat_completions",
            "session_scoped_sampler_origin": True,
            "embeds_credentials": False,
            "revision_immutable_after_admission": True,
            "records_policy_revision": True,
        },
        "lifecycle": {
            "max_concurrency": 30,
            "lease_ttl_seconds": 900.0,
            "supports_idempotency": True,
            "supports_cancellation": True,
            "supports_lease_renewal": True,
            "exactly_one_terminal_result": True,
            "supports_pause_resume": False,
            "straggler_grace_seconds": 120.0,
        },
        "evidence": {
            "trace_v5": True,
            "behavior_logprobs": True,
            "strict_prefix": True,
            "masking": True,
            "wire_objects": True,
            "artifact_reference": True,
            "tokens_in_tokens_out": False,
        },
        "reward": {
            "authority": "container",
            "binds_trace_digest": True,
            "quiescence": True,
            "horizon_clipping": True,
            "channels": ["outcome"],
            "reward_relation": "cooperative",
            "evaluation_plan_id": "plan-1",
            "settlement_window_seconds": 0.0,
            "deferred_scoring": False,
        },
        "recovery": {"restart": True, "stale_discard": True},
        "topology": {
            "topology_id": "topology-1",
            "turn_model": "sequential",
            "actuation_model": "direct_action",
            "reward_relation": "cooperative",
            "agent_instances": [
                {
                    "agent_instance_id": "instance-1",
                    "role_id": "role-1",
                    "policy_type_id": "policy-1",
                    "team_id": "team-1",
                    "trainable": True,
                }
            ],
            "teams": [
                {"team_id": "team-1", "trainable": True, "minimum_viable_roster": 1}
            ],
            "communication_channels": [
                {"channel_id": "channel-1", "scope": "intra_team", "trainable_for_author": True}
            ],
            "horizon": {
                "horizon_kind": "wall_clock",
                "value_seconds": 5400.0,
                "time_dilation": 1.0,
            },
            "parameter_groups": {"policy-1": "group-1"},
        },
        "clock": {"skew_tolerance_seconds": 2.0},
    }
    document = _merge(document, overrides)
    document.pop("capability_hash", None)
    document["capability_hash"] = canonical_capability_hash(document)
    return document


def requirements(**overrides: Any) -> ExecutorRequirements:
    values: dict[str, Any] = {
        "renderer_profile": PROFILE,
        "min_concurrency": 8,
        "horizon_seconds": 5400.0,
        "optimized_channel": "outcome",
        "expected_taskset_id": "taskset-1",
        "expected_topology_ref": "topology-1",
    }
    values.update(overrides)
    return ExecutorRequirements(**values)


def run_plan(**overrides: Any) -> RunPlan:
    values: dict[str, Any] = {
        "group_size": 8,
        "groups_per_step": 2,
        "max_execution_slots": 16,
        "maximum_policy_lag": 1,
        "target_train_updates": 10,
        "expected_horizon_seconds": 5400.0,
    }
    values.update(overrides)
    return RunPlan(**values)


ROUTES: dict[str, str] = {
    "health_route": "/health",
    "capabilities_route": "/training/capabilities",
    "handshake_route": "/training/handshake",
    "taskset_route": "/taskset",
    "taskset_tasks_route": "/taskset/tasks",
    "topology_route": "/topologies/{topology_id}",
    "policy_bind_route": "/policy-configs",
    "policy_set_bind_route": "/policy-sets",
    "rollout_route": "/rollout",
    "rollout_state_route": "/rollouts/{rollout_id}",
    "rollout_events_route": "/rollouts/{rollout_id}/events",
    "rollout_renew_route": "/rollouts/{rollout_id}/renew",
    "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
    "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
    "trace_route": "/rollouts/{rollout_id}/trace",
    "artifacts_route": "/rollouts/{rollout_id}/artifacts",
    "reward_route": "/reward",
}

CONTRACT = ContainerContract.from_metadata(
    {"metadata": {"optimizer_contracts": {"cispo": {"version": CISPO_CONTRACT_VERSION, **ROUTES}}}}
)


class FakeContainer:
    """A local fake. It answers clauses; it never opens a socket."""

    def __init__(
        self,
        *,
        capability: Mapping[str, Any] | None = None,
        concurrency_ceiling: int = 30,
        overrides: Mapping[str, tuple[str, str]] | None = None,
        drop_clauses: Sequence[str] = (),
        skew_seconds: float = 0.4,
        expires_in_seconds: float = 600.0,
        digest_override: str | None = None,
        quiescence: bool = True,
    ) -> None:
        self._capability = dict(capability or capability_payload())
        self._ceiling = concurrency_ceiling
        self._overrides = dict(overrides or {})
        self._drop = frozenset(drop_clauses)
        self._skew = skew_seconds
        self._expires_in = expires_in_seconds
        self._digest_override = digest_override
        self._quiescence = quiescence
        self.capability_calls = 0
        self.handshake_calls = 0
        self.bind_calls = 0
        self.requests: list[HandshakeRequest] = []

    def set_capability(self, payload: Mapping[str, Any]) -> None:
        self._capability = dict(payload)

    def capabilities(self) -> Mapping[str, Any]:
        self.capability_calls += 1
        return dict(self._capability)

    def bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.bind_calls += 1
        return {"policy_set_revision_id": "set-1"}

    def _obligations(self) -> Obligations:
        lifecycle = self._capability["lifecycle"]
        return Obligations(
            max_concurrency=self._ceiling,
            lease_ttl_seconds=float(lifecycle["lease_ttl_seconds"]),
            deferred_scoring=False,
            quiescence=self._quiescence,
            settlement_window_seconds=0.0,
            horizon=None,
        )

    def _clauses(self, request: HandshakeRequest) -> list[ClauseResult]:
        demand = request.run_plan.group_size * request.run_plan.groups_per_step
        results: list[ClauseResult] = []
        for clause in MANDATORY_CLAUSES:
            if clause in self._drop:
                continue
            verdict, reason = self._overrides.get(clause, ("accepted", ""))
            if clause == "lifecycle.concurrency" and demand > self._ceiling:
                verdict, reason = (
                    "degraded",
                    f"{self._ceiling} leases available, {demand} requested in flight",
                )
            results.append(
                ClauseResult(
                    clause_id=clause, verdict=verdict, reason=reason, source="container"
                )
            )
        for clause in sorted(OPTIONAL_CLAUSES):
            verdict, reason = self._overrides.get(clause, ("unsupported", "not offered"))
            results.append(
                ClauseResult(
                    clause_id=clause, verdict=verdict, reason=reason, source="container"
                )
            )
        return results

    def handshake(self, request: HandshakeRequest, *, now: datetime = NOW) -> dict[str, Any]:
        self.handshake_calls += 1
        self.requests.append(request)
        clauses = self._clauses(request)
        obligations = self._obligations()
        resolution = tuple(
            TaskResolution(
                task_id=task_id,
                content_digest=f"sha256:{task_id}",
                topology_ref=self._capability["topology"]["topology_id"],
            )
            for task_id in request.taskset.task_ids
        )
        handshake_id = f"hs-{self.handshake_calls}"
        accepted = all(item.verdict == "accepted" for item in clauses if item.mandatory)
        agreement_digest = self._digest_override or compute_agreement_digest(
            request,
            handshake_id=handshake_id,
            capability_hash=str(self._capability["capability_hash"]),
            renderer_fingerprint=CapabilityDocument.from_payload(
                self._capability
            ).renderer_profile.fingerprint,
            taskset_resolution=resolution,
            obligations=obligations,
            clauses=clauses,
        )
        return {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "handshake_id": handshake_id,
            "accepted": accepted,
            "clauses": [item.to_payload() for item in clauses],
            "obligations": obligations.to_payload(),
            "taskset_resolution": [
                {
                    "task_id": item.task_id,
                    "content_digest": item.content_digest,
                    "topology_ref": item.topology_ref,
                }
                for item in resolution
            ],
            "capability_hash": self._capability["capability_hash"],
            "agreement_digest": agreement_digest,
            "expires_at": format_rfc3339(now + timedelta(seconds=self._expires_in)),
            "clock": {
                "container_time": format_rfc3339(now),
                "measured_skew_seconds": self._skew,
            },
        }


def _evaluate(
    container: FakeContainer,
    *,
    needs: ExecutorRequirements | None = None,
    plan: RunPlan | None = None,
    now: datetime = NOW,
    request: HandshakeRequest | None = None,
):
    needs = needs or requirements()
    document, results = preflight_capabilities(
        container.capabilities(), needs, contract=CONTRACT
    )
    if request is None:
        request = build_request(
            run_id="run-1",
            optimizer=OptimizerIdentity(name="synth_optimizers.cispo", version="0.2.20"),
            policy=PolicyRequest(
                provider="provider-1",
                model_id="vendor/policy-20b",
                transport="message_in_capture_out",
            ),
            requirements=needs,
            topology=TopologyExpectation(
                expected_topology_id="topology-1",
                trainable_teams=("team-1",),
                partial_roster="refuse",
            ),
            run_plan=plan or run_plan(),
            task_ids=TASK_IDS,
            taskset_id="taskset-1",
            now=now,
        )
    verdict = HandshakeVerdict.from_payload(container.handshake(request, now=now))
    decision = evaluate_handshake(
        request,
        verdict,
        capability=document,
        contract=CONTRACT,
        executor_clauses=results,
        now=now,
    )
    return document, request, verdict, decision


def ledger() -> HandshakeLedger:
    return HandshakeLedger(clock=lambda: NOW)


def test_clean_document_and_verdict_are_admissible() -> None:
    container = FakeContainer(concurrency_ceiling=30)
    document, request, verdict, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    assert decision.outcome == "admissible"
    agreement = ledger().admit(decision)
    assert agreement.agreement_digest == verdict.agreement_digest
    assert agreement.capability_hash == document.content_hash
    assert agreement.contract_hash == CONTRACT.contract_hash
    assert agreement.task_digest("task-a") == "sha256:task-a"
    assert {item.clause_id for item in agreement.fallbacks} >= {
        "evidence.tito",
        "lifecycle.pause_resume",
        "reward.settlement_window",
    }
    assert agreement.to_receipt()["handshake_id"] == "hs-1"


def test_capability_clause_check_accepts_every_mandatory_clause() -> None:
    document = CapabilityDocument.from_payload(capability_payload())
    results = check_requirements(document, requirements(), contract=CONTRACT)
    unsupported = {item.clause_id for item in results if item.verdict != "accepted"}
    assert unsupported == {
        "evidence.tito",
        "lifecycle.pause_resume",
        "reward.settlement_window",
    }
    assert all(item.clause_id in OPTIONAL_CLAUSES for item in results if item.blocks_run) is True


def test_capability_hash_is_fail_closed_on_change() -> None:
    payload = capability_payload()
    tampered = copy.deepcopy(payload)
    tampered["lifecycle"]["max_concurrency"] = 64
    with pytest.raises(CapabilityHashError):
        CapabilityDocument.from_payload(tampered)
    rehashed = capability_payload(lifecycle={"max_concurrency": 64})
    first = CapabilityDocument.from_payload(payload)
    second = CapabilityDocument.from_payload(rehashed)
    assert first.content_hash != second.content_hash
    with pytest.raises(CapabilityDriftError):
        second.assert_unchanged(first.content_hash)


def test_renderer_profile_mismatch_is_refused_before_any_session() -> None:
    container = FakeContainer()
    other = RendererProfile(
        profile_id=PROFILE.profile_id,
        package=PROFILE.package,
        package_version="0.1.12",
        config_digest="sha256:different",
        tokenizer_id=PROFILE.tokenizer_id,
        tokenizer_digest=PROFILE.tokenizer_digest,
        stop_token_ids=PROFILE.stop_token_ids,
    )
    with pytest.raises(PreflightRejected) as excinfo:
        preflight_capabilities(
            container.capabilities(),
            requirements(renderer_profile=other),
            contract=CONTRACT,
        )
    assert "policy.renderer_profile_match" in excinfo.value.clause_ids
    assert container.handshake_calls == 0
    assert container.bind_calls == 0


def test_rejected_mandatory_clause_stops_before_spend() -> None:
    container = FakeContainer(
        overrides={
            "reward.horizon_quiescence": (
                "rejected",
                "cannot kill agent-authored background processes",
            )
        },
    )
    with pytest.raises(ClauseRejected) as excinfo:
        _evaluate(container, plan=run_plan(groups_per_step=1))
    assert "reward.horizon_quiescence" in excinfo.value.clause_ids
    assert container.bind_calls == 0


def test_unanswered_mandatory_clause_is_rejected() -> None:
    container = FakeContainer(drop_clauses=("lifecycle.idempotency",))
    with pytest.raises(ClauseRejected) as excinfo:
        _evaluate(container, plan=run_plan(groups_per_step=1))
    assert "lifecycle.idempotency" in excinfo.value.clause_ids


def test_degraded_concurrency_produces_a_lowered_plan_and_a_second_handshake() -> None:
    container = FakeContainer(concurrency_ceiling=4)
    book = ledger()
    _, _, _, first = _evaluate(container, plan=run_plan())
    assert first.outcome == "renegotiate"
    assert first.degraded_clauses == ("lifecycle.concurrency",)
    assert first.next_request is not None
    lowered = first.next_request.run_plan
    assert (lowered.group_size, lowered.groups_per_step, lowered.max_execution_slots) == (4, 1, 4)
    assert first.next_request.attempt == 2
    with pytest.raises(RenegotiationRequired):
        book.admit(first)
    _, _, verdict, second = _evaluate(container, request=first.next_request)
    assert second.outcome == "admissible"
    agreement = book.admit(second)
    assert agreement.obligations.max_concurrency == 4
    assert agreement.request.run_plan == lowered
    assert container.handshake_calls == 2
    assert verdict.clause("lifecycle.concurrency") is not None


def test_a_plan_already_at_the_bound_cannot_be_lowered_again() -> None:
    container = FakeContainer(
        concurrency_ceiling=1,
        overrides={"lifecycle.concurrency": ("degraded", "the pool shrank under us")},
    )
    with pytest.raises(PlanNotLowerable):
        _evaluate(
            container,
            plan=run_plan(group_size=1, groups_per_step=1, max_execution_slots=1),
        )


def test_clock_skew_beyond_tolerance_is_a_rejected_clause() -> None:
    container = FakeContainer(skew_seconds=9.5)
    with pytest.raises(ClauseRejected) as excinfo:
        _evaluate(container, plan=run_plan(groups_per_step=1))
    assert "reward.horizon_quiescence" in excinfo.value.clause_ids
    assert "clock skew" in str(excinfo.value)


def test_step_horizon_does_not_reject_on_skew() -> None:
    payload = capability_payload(
        topology={"horizon": {"horizon_kind": "steps", "value_seconds": 5400.0}}
    )
    container = FakeContainer(capability=payload, skew_seconds=9.5)
    _, _, _, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    assert decision.outcome == "admissible"


def test_verdict_built_on_another_capability_document_is_refused() -> None:
    container = FakeContainer()
    needs = requirements()
    document, results = preflight_capabilities(
        container.capabilities(), needs, contract=CONTRACT
    )
    # The container re-publishes a changed document, then answers the handshake
    # against the new one. The preflight the executor holds is now stale.
    container.set_capability(capability_payload(container_id="container-2"))
    request = build_request(
        run_id="run-1",
        optimizer=OptimizerIdentity(name="o", version="1"),
        policy=PolicyRequest(provider="p", model_id="m", transport="message_in_capture_out"),
        requirements=needs,
        topology=TopologyExpectation(
            expected_topology_id="topology-1", trainable_teams=("team-1",)
        ),
        run_plan=run_plan(groups_per_step=1),
        task_ids=TASK_IDS,
        taskset_id="taskset-1",
        now=NOW,
    )
    verdict = HandshakeVerdict.from_payload(container.handshake(request, now=NOW))
    with pytest.raises(CapabilityDriftError):
        evaluate_handshake(
            request,
            verdict,
            capability=document,
            contract=CONTRACT,
            executor_clauses=results,
            now=NOW,
        )


def test_agreement_digest_disagreement_is_refused() -> None:
    container = FakeContainer(digest_override="sha256:not-what-we-computed")
    with pytest.raises(AgreementMismatch):
        _evaluate(container, plan=run_plan(groups_per_step=1))


def test_expired_verdict_is_refused_at_evaluation() -> None:
    container = FakeContainer(expires_in_seconds=-1.0)
    with pytest.raises(HandshakeExpired):
        _evaluate(container, plan=run_plan(groups_per_step=1))


def test_expired_agreement_is_refused_at_admission() -> None:
    container = FakeContainer(expires_in_seconds=60.0)
    _, _, _, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    book = ledger()
    agreement = book.admit(decision)
    book.assert_admissible(agreement.handshake_id, agreement.agreement_digest)
    with pytest.raises(HandshakeExpired):
        book.assert_admissible(
            agreement.handshake_id,
            agreement.agreement_digest,
            now=NOW + timedelta(seconds=61),
        )


def test_revoked_agreement_is_refused_at_admission() -> None:
    container = FakeContainer()
    _, _, _, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    book = ledger()
    agreement = book.admit(decision)
    book.revoke(agreement.handshake_id, "container degraded")
    with pytest.raises(HandshakeRevoked):
        book.assert_admissible(agreement.handshake_id, agreement.agreement_digest)


def test_mismatched_agreement_digest_is_refused_at_admission() -> None:
    container = FakeContainer()
    _, _, _, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    book = ledger()
    agreement = book.admit(decision)
    with pytest.raises(AgreementMismatch):
        book.assert_admissible(agreement.handshake_id, "sha256:some-other-run")
    with pytest.raises(UnknownHandshake):
        book.assert_admissible("hs-unknown", agreement.agreement_digest)


def test_renewal_extends_expiry_without_changing_the_agreement() -> None:
    container = FakeContainer(expires_in_seconds=60.0)
    document, request, _, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    book = ledger()
    agreement = book.admit(decision)
    renewal = HandshakeVerdict.from_payload(
        container.handshake(request, now=NOW + timedelta(seconds=30))
    )
    renewed = book.renew(
        agreement.handshake_id,
        capability=document,
        verdict=HandshakeVerdict(
            handshake_id=agreement.handshake_id,
            accepted=renewal.accepted,
            clauses=renewal.clauses,
            obligations=renewal.obligations,
            taskset_resolution=renewal.taskset_resolution,
            capability_hash=renewal.capability_hash,
            agreement_digest=agreement.agreement_digest,
            expires_at=renewal.expires_at,
            container_time=renewal.container_time,
            measured_skew_seconds=renewal.measured_skew_seconds,
        ),
        now=NOW + timedelta(seconds=30),
    )
    assert renewed.expires_at > agreement.expires_at
    assert renewed.agreement_digest == agreement.agreement_digest
    book.assert_admissible(
        agreement.handshake_id, agreement.agreement_digest, now=NOW + timedelta(seconds=61)
    )


def test_renewal_fails_closed_when_the_capability_document_changed() -> None:
    container = FakeContainer()
    _, request, verdict, decision = _evaluate(container, plan=run_plan(groups_per_step=1))
    book = ledger()
    agreement = book.admit(decision)
    changed = CapabilityDocument.from_payload(
        capability_payload(lifecycle={"max_concurrency": 12})
    )
    with pytest.raises(CapabilityDriftError):
        book.renew(agreement.handshake_id, capability=changed, verdict=verdict)
    with pytest.raises(HandshakeRevoked):
        book.assert_admissible(agreement.handshake_id, agreement.agreement_digest)


def test_requirement_document_carries_the_declared_shape() -> None:
    needs = requirements()
    request = build_request(
        run_id="run-1",
        optimizer=OptimizerIdentity(name="synth_optimizers.cispo", version="0.2.20"),
        policy=PolicyRequest(
            provider="provider-1",
            model_id="vendor/policy-20b",
            transport="message_in_capture_out",
        ),
        requirements=needs,
        topology=TopologyExpectation(
            expected_topology_id="topology-1", trainable_teams=("team-1",)
        ),
        run_plan=run_plan(),
        task_ids=TASK_IDS,
        taskset_id="taskset-1",
        now=NOW,
    )
    payload = request.to_payload()
    assert set(payload) == {
        "schema_version",
        "run_id",
        "attempt",
        "optimizer",
        "policy",
        "renderer_profile",
        "requirements",
        "topology",
        "run_plan",
        "taskset",
        "clock",
    }
    assert payload["schema_version"] == HANDSHAKE_SCHEMA_VERSION
    assert set(MANDATORY_CLAUSES).issubset(set(payload["requirements"]))
    assert request.request_digest.startswith("sha256:")


def test_requirement_document_must_name_every_mandatory_clause() -> None:
    with pytest.raises(Exception) as excinfo:
        HandshakeRequest(
            run_id="run-1",
            optimizer=OptimizerIdentity(name="o", version="1"),
            policy=PolicyRequest(
                provider="p", model_id="m", transport="message_in_capture_out"
            ),
            renderer_profile=PROFILE,
            requirements=("contract.version",),
            topology=TopologyExpectation(
                expected_topology_id="topology-1", trainable_teams=("team-1",)
            ),
            run_plan=run_plan(),
            taskset=TasksetRequest(
                taskset_id="taskset-1", split="train", task_ids=TASK_IDS
            ),
            clock=ClockStamp(executor_time=format_rfc3339(NOW)),
        )
    assert "omits mandatory clauses" in str(excinfo.value)


def test_taskset_resolution_must_cover_every_requested_task() -> None:
    container = FakeContainer()
    document, _ = preflight_capabilities(
        container.capabilities(), requirements(), contract=CONTRACT
    )
    del document
    request_ids = ("task-a", "task-b", "task-c")

    class ShortResolution(FakeContainer):
        def handshake(
            self, request: HandshakeRequest, *, now: datetime = NOW
        ) -> dict[str, Any]:
            payload = super().handshake(request, now=now)
            payload["taskset_resolution"] = payload["taskset_resolution"][:1]
            return payload

    short = ShortResolution()
    plan = run_plan(groups_per_step=1)
    needs = requirements()
    request = build_request(
        run_id="run-1",
        optimizer=OptimizerIdentity(name="o", version="1"),
        policy=PolicyRequest(provider="p", model_id="m", transport="message_in_capture_out"),
        requirements=needs,
        topology=TopologyExpectation(
            expected_topology_id="topology-1", trainable_teams=("team-1",)
        ),
        run_plan=plan,
        task_ids=request_ids,
        taskset_id="taskset-1",
        now=NOW,
    )
    with pytest.raises(ClauseRejected) as excinfo:
        _evaluate(short, request=request)
    assert "discovery.task_digests" in excinfo.value.clause_ids
