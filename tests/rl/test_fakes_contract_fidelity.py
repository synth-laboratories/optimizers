"""The fakes as a faithful reference for the real client contract.

The conformance suite drives the fakes with the fakes' own driver. These tests
do the opposite: they hand what a fake actually serves to the shipped client
code -- ``CapabilityDocument.from_payload``, ``ContainerContract.from_metadata``,
``evaluate_handshake``, ``HandshakeLedger.renew``, ``ROUTE_METHODS`` -- and
require it to be accepted there. A fake that diverges from the contract is a
fake that hides bugs, so every divergence has to fail here rather than pass
quietly through an adapter.
"""

from __future__ import annotations

import re

import pytest
from fakes import scenarios
from fakes.container import (
    DECLARED_ROUTES,
    ContainerClient,
    ContainerConfig,
    RunningContainer,
    serve,
)
from synth_optimizers.contracts.rl_records import RendererProfile
from synth_optimizers.rl.capabilities import (
    CapabilityDocument,
    ExecutorRequirements,
    assert_preflight_passed,
    canonical_capability_hash,
    check_requirements,
)
from synth_optimizers.rl.contract import (
    MANDATORY_ROUTES,
    ROUTE_METHODS,
    ROUTE_PARAMETERS,
    ContainerContract,
)
from synth_optimizers.rl.handshake import (
    HandshakeLedger,
    HandshakeRequest,
    HandshakeVerdict,
    OptimizerIdentity,
    PolicyRequest,
    RunPlan,
    TopologyExpectation,
    build_request,
    evaluate_handshake,
)

CONFORMANT_NAMES = sorted(scenarios.CONFORMANT)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _capability(client: ContainerClient) -> CapabilityDocument:
    return CapabilityDocument.from_payload(client.capabilities())


def _requirements(document: CapabilityDocument) -> ExecutorRequirements:
    return ExecutorRequirements(
        renderer_profile=document.renderer_profile,
        min_concurrency=1,
        horizon_seconds=document.horizon.value,
        optimized_channel=document.reward.channels[0],
        sampling_transport=document.policy.binding_transport,
        wire_api=document.policy.wire_api,
        split="train",
        expected_taskset_id=document.discovery.taskset_id,
        expected_topology_ref=document.topology_ref,
        require_quiescence=document.reward.quiescence,
    )


def _request(
    client: ContainerClient,
    document: CapabilityDocument,
    *,
    run_id: str = "run_fidelity",
) -> HandshakeRequest:
    """A real ``HandshakeRequest``, built by the shipped builder."""

    return build_request(
        run_id=run_id,
        optimizer=OptimizerIdentity(name="synth_optimizers.cispo", version="0.0.0-test"),
        policy=PolicyRequest(
            provider="fake",
            model_id=document.renderer_profile.tokenizer_id,
            transport=document.policy.binding_transport,
        ),
        requirements=_requirements(document),
        topology=TopologyExpectation(
            expected_topology_id=document.topology.topology_id,
            trainable_teams=tuple(
                team.team_id for team in document.topology.teams if team.trainable
            ),
            partial_roster="refuse",
        ),
        run_plan=RunPlan(
            group_size=2,
            groups_per_step=1,
            max_execution_slots=2,
            maximum_policy_lag=1,
            target_train_updates=1,
            expected_horizon_seconds=document.horizon.value,
        ),
        task_ids=client.task_ids()[:1],
        taskset_id=document.discovery.taskset_id,
    )


def _route_pattern(template: str) -> re.Pattern[str]:
    return re.compile("^" + re.sub(r"\{[a-z_]+\}", "[^/]+", template) + "$")


def _drive_every_route(config: ContainerConfig) -> tuple[tuple[str, str], ...]:
    """Every declared route this container has, called once each."""

    with serve(config) as container:
        client = container.client()
        assert client.negotiate()[-1]["accepted"]
        task_id = client.task_ids()[0]
        client.topology(config.topology.topology_id)
        client.run_attempt(task_id=task_id)
        binding = client.bind()
        cancelled = client.submit(
            task_id=task_id,
            idempotency_key="fidelity-cancel",
            policy_config_id=binding["config_id"],
        )
        client.terminate(str(cancelled["rollout_id"]), reason="fidelity")
        return container.requested_paths


# --------------------------------------------------------------------------- #
# The capability document
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_capability_document_round_trips_through_the_client_parser(name: str) -> None:
    """What a fake serves is what ``CapabilityDocument`` parses, hash included."""

    with serve(scenarios.CONFORMANT[name]()) as container:
        client = container.client()
        payload = client.capabilities()
        # The body is the document: no envelope to unwrap on the way in.
        assert payload["schema_version"] == "cispo.capabilities.v1"
        assert "capabilities" not in payload

        document = CapabilityDocument.from_payload(payload)
        assert document.content_hash == canonical_capability_hash(payload)
        assert document.content_hash == payload["capability_hash"]

        config = container.config
        assert document.container_id == config.container_id
        assert document.container_image_digest == config.image_digest
        assert document.discovery.taskset_id == config.taskset_id
        assert document.policy.binding_transport == config.sampling_transport
        assert document.recovery.restart is True
        assert document.reward.channels == config.reward_channel_ids
        assert document.clock_skew_tolerance_seconds == config.skew_tolerance_seconds
        # A declared horizon is always there, with the conversion a unit
        # horizon needs; the executor never guesses one.
        assert document.horizon.horizon_kind == config.horizon.horizon_kind
        assert document.horizon.value == pytest.approx(config.horizon.value)
        if document.horizon.horizon_kind != "wall_clock":
            assert document.horizon.declared_seconds_per_unit() > 0


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_capability_document_passes_the_executor_clause_check(name: str) -> None:
    """The advertisement alone admits a run, before any session or spend."""

    with serve(scenarios.CONFORMANT[name]()) as container:
        client = container.client()
        document = _capability(client)
        contract = ContainerContract.from_metadata(client.call("GET", "/metadata"))
        assert_preflight_passed(
            check_requirements(document, _requirements(document), contract=contract)
        )


def test_a_pinned_opponent_is_named_where_the_canonical_parser_reads_it() -> None:
    with serve(scenarios.competitive_realtime()) as container:
        document = _capability(container.client())
        opponents = document.topology.opponent_instances
        assert opponents
        for instance in opponents:
            assert instance.pinned_identity == scenarios.PINNED_OPPONENT


# --------------------------------------------------------------------------- #
# The contract advertisement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_metadata_satisfies_the_client_contract_and_resolves_every_route(name: str) -> None:
    with serve(scenarios.CONFORMANT[name]()) as container:
        client = container.client()
        contract = ContainerContract.from_metadata(client.call("GET", "/metadata"))
        assert contract.route_table.declared == dict(DECLARED_ROUTES)
        for route in MANDATORY_ROUTES:
            resolved = contract.resolve(
                route,
                rollout_id="ro_fidelity",
                topology_id=container.config.topology.topology_id,
            )
            assert resolved.startswith("/")
            assert "{" not in resolved
            assert contract.route_table.method(route) == ROUTE_METHODS[route]


def test_every_declared_route_is_exercised_with_the_method_the_contract_declares() -> None:
    """The fake's own driver speaks the methods ``ROUTE_METHODS`` declares."""

    observed = _drive_every_route(scenarios.one_call_classification()) + _drive_every_route(
        scenarios.competitive_realtime()
    )
    touched: dict[str, set[str]] = {}
    for method, path in observed:
        for name, template in DECLARED_ROUTES.items():
            if _route_pattern(template).match(path):
                touched.setdefault(name, set()).add(method)
    assert set(touched) == set(MANDATORY_ROUTES), sorted(set(MANDATORY_ROUTES) - set(touched))
    for name, methods in sorted(touched.items()):
        assert methods == {ROUTE_METHODS[name]}, (name, sorted(methods))
    # The row lookup is the one the note calls out by name.
    assert touched["taskset_tasks_route"] == {"POST"}
    assert ROUTE_PARAMETERS["taskset_tasks_route"] == ()


# --------------------------------------------------------------------------- #
# The handshake
# --------------------------------------------------------------------------- #


def _admissible(container: RunningContainer):
    client = container.client()
    document = _capability(client)
    contract = ContainerContract.from_metadata(client.call("GET", "/metadata"))
    request = _request(client, document)
    verdict = HandshakeVerdict.from_payload(client.handshake(request.to_payload()))
    decision = evaluate_handshake(
        request,
        verdict,
        capability=document,
        contract=contract,
        now=verdict.container_time,
    )
    return client, document, request, verdict, decision


def test_handshake_verdict_is_admissible_to_the_real_evaluator() -> None:
    """The container's own agreement digest is the one the executor recomputes."""

    with serve(scenarios.multi_turn_environment_reward()) as container:
        _client, document, _request_doc, verdict, decision = _admissible(container)
        assert decision.outcome == "admissible"
        agreement = decision.agreement
        assert agreement is not None
        assert agreement.agreement_digest == verdict.agreement_digest
        assert agreement.capability_hash == document.content_hash
        assert verdict.schema_version == "cispo.handshake.v1"


@pytest.mark.parametrize(
    "name", ["one_call_classification", "multi_turn_environment_reward", "competitive_realtime"]
)
def test_the_agreement_digest_is_the_shared_one_not_a_local_one(name: str) -> None:
    with serve(scenarios.CONFORMANT[name]()) as container:
        _client, _document, _request_doc, verdict, decision = _admissible(container)
        assert decision.admissible
        assert verdict.agreement_digest.startswith("sha256:")


def test_a_renewal_extends_the_agreement_the_ledger_already_holds() -> None:
    """``HandshakeLedger.renew`` accepts the fake's renewal unchanged."""

    with serve(scenarios.multi_turn_environment_reward()) as container:
        client, document, request, verdict, decision = _admissible(container)
        ledger = HandshakeLedger()
        agreement = ledger.admit(decision)

        container.clock.advance(60.0)
        renewal = HandshakeVerdict.from_payload(
            client.handshake({**request.to_payload(), "renew_of": agreement.handshake_id})
        )
        # A renewal extends an agreement; it never replaces one.
        assert renewal.handshake_id == agreement.handshake_id
        assert renewal.agreement_digest == agreement.agreement_digest
        assert renewal.capability_hash == document.content_hash

        renewed = ledger.renew(
            agreement.handshake_id,
            capability=document,
            verdict=renewal,
            now=verdict.container_time,
        )
        assert renewed.expires_at > agreement.expires_at
        assert renewed.agreement_digest == agreement.agreement_digest
        assert (
            ledger.assert_admissible(
                agreement.handshake_id,
                agreement.agreement_digest,
                now=verdict.container_time,
            ).expires_at
            == renewed.expires_at
        )


def test_the_renderer_profile_the_container_declares_is_the_one_it_agreed_on() -> None:
    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        document = _capability(client)
        declared = RendererProfile.from_payload(client.capabilities()["renderer_profile"])
        assert document.renderer_profile.fingerprint == declared.fingerprint


# --------------------------------------------------------------------------- #
# The per-attempt reward source
# --------------------------------------------------------------------------- #


def test_a_container_can_declare_a_measure_per_attempt() -> None:
    """One constant would tie every attempt, and a tied group carries no order."""

    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        assert client.negotiate()[-1]["accepted"]
        task_id = client.task_ids()[0]
        container.set_reward_source(lambda _task_id, index: 0.25 * (index + 1))
        measures = []
        for index in range(2):
            attempt = client.run_attempt(
                task_id=task_id,
                idempotency_key=f"varied-{index}",
                correlation={"sample_index": index},
            )
            measures.append(attempt.reward.value())
        assert measures == pytest.approx([0.25, 0.5])


def test_a_declared_mapping_answers_per_sample_and_falls_back_to_the_constant() -> None:
    config = scenarios.one_call_classification()
    assert config.reward_for("any", 0) == pytest.approx(config.reward_value)
    varied = ContainerConfig(
        **{
            **{
                field: getattr(config, field)
                for field in config.__dataclass_fields__
                if field != "reward_value_by_sample"
            },
            "reward_value_by_sample": {1: 0.75},
        }
    )
    assert varied.reward_for("row", 1) == pytest.approx(0.75)
    assert varied.reward_for("row", 0) == pytest.approx(config.reward_value)
