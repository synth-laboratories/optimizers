"""Stream 6: the reusable container conformance suite.

Two halves. The conformant fakes must serve every declared route and produce
evidence that passes the shared validators. The non-conformant fakes must each
fail with the *specific* typed error the design note requires -- not merely
with some error -- because "a run that looks healthy and optimizes nothing" is
the failure mode this suite exists to catch.

No test selects a fake by task name: a fake is chosen by capability
configuration only, and :func:`test_fakes_never_name_a_task_harness_or_env`
holds that line in the fakes themselves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from fakes import checks, scenarios
from fakes.container import (
    DECLARED_ROUTES,
    Clock,
    ContainerClient,
    ContainerConfig,
    ContainerError,
    RunningContainer,
    group_pin_from_fields,
    rollout_receipt_from_payload,
    serve,
    topology_from_payload,
)
from synth_optimizers.contracts.rl_clauses import ALL_CLAUSES, MANDATORY_CLAUSES, VERDICTS
from synth_optimizers.contracts.rl_identity import (
    MixedGroupError,
    TopologyError,
    assert_uniform_group,
)
from synth_optimizers.contracts.rl_records import (
    EvidenceError,
    InferenceCall,
    RecordError,
    assert_strict_prefix,
)

FAKES_DIR = Path(__file__).parent / "fakes"

# The house rules forbid these literals in engine code; the fakes are held to
# the same bar so that no downstream stream can dispatch on one.
BANNED_LITERALS = (
    "banking77",
    "healthbench",
    "craftax",
    "tblite",
    "dungeongrid",
    "runite",
    "harbor",
    "mini_swe",
    "opencode",
    "react",
    "elf",
    "barbarian",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _route_pattern(template: str) -> re.Pattern[str]:
    return re.compile("^" + re.sub(r"\{[a-z_]+\}", "[^/]+", template) + "$")


def _routes_touched(container: RunningContainer) -> set[str]:
    touched: set[str] = set()
    for _method, path in container.requested_paths:
        for key, template in DECLARED_ROUTES.items():
            if _route_pattern(template).match(path):
                touched.add(key)
    return touched


def _drive(config: ContainerConfig) -> tuple[RunningContainer, ContainerClient]:
    container = serve(config)
    client = container.client()
    exchanges = client.negotiate()
    assert exchanges[-1]["accepted"], exchanges[-1]
    return container, client


def _first_task(client: ContainerClient) -> str:
    return client.task_ids()[0]


def _declared_topology(client: ContainerClient, config: ContainerConfig):
    return topology_from_payload(client.topology(config.topology.topology_id))


def _by_instance(calls: Iterable[InferenceCall]) -> dict[str | None, list[InferenceCall]]:
    grouped: dict[str | None, list[InferenceCall]] = {}
    for call in calls:
        grouped.setdefault(call.agent_instance_id, []).append(call)
    return grouped


def _pin(state: Mapping[str, object]):
    return group_pin_from_fields(
        state["group_pin_fields"],  # type: ignore[arg-type]
        group_id="group_conformance",
        run_id="run_fake",
        algorithm_plan_hash="plan#cispo",
        cardinality=2,
    )


CONFORMANT_NAMES = sorted(scenarios.CONFORMANT)


# --------------------------------------------------------------------------- #
# Declared surface and preflight
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_metadata_declares_the_full_route_table(name: str) -> None:
    with serve(scenarios.CONFORMANT[name]()) as container:
        client = container.client()
        assert set(client.routes) == set(DECLARED_ROUTES)
        assert client.contract_version == "synth_optimizers.cispo.v1"


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_conformant_containers_serve_every_declared_route(name: str) -> None:
    config = scenarios.CONFORMANT[name]()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        client.topology(config.topology.topology_id)
        client.events(attempt.rollout_id)
        client.terminate(attempt.rollout_id)
        client.bind_policy(kind="trainable")
        client.bind_policy_set(
            bindings=[
                {
                    "agent_instance_id": instance.agent_instance_id,
                    "policy_ref": instance.pinned_identity or "checkpoint::rev0",
                }
                for instance in config.topology.agent_instances
            ]
        )
        assert _routes_touched(container) == set(DECLARED_ROUTES)
    finally:
        container.shutdown()


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_handshake_answers_every_clause_with_a_verdict(name: str) -> None:
    with serve(scenarios.CONFORMANT[name]()) as container:
        client = container.client()
        payload = client.preflight()
        clauses = {row["clause_id"]: row for row in payload["clauses"]}
        assert set(clauses) == set(ALL_CLAUSES)
        for clause_id, row in clauses.items():
            assert row["verdict"] in VERDICTS, clause_id
            if row["verdict"] != "accepted":
                assert row["reason"], clause_id
        assert payload["agreement_digest"]
        assert payload["capability_hash"]
        assert payload["expires_at"]
        assert payload["taskset_resolution"]
        assert "measured_skew_seconds" in payload["clock"]


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_conformant_evidence_validates_for_training(name: str) -> None:
    config = scenarios.CONFORMANT[name]()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        assert attempt.trainable_calls
        for call in attempt.trainable_calls:
            call.validate_for_training()
        assert attempt.episodes
        for episode in attempt.episodes:
            episode.validate()
            assert episode.trace_digest == attempt.trace_digest
        reward = attempt.reward
        reward.validate(episode_trace_digest=attempt.trace_digest)
        checks.assert_no_flattened_wire(attempt.calls, declared_wire_api=config.wire_api)
        checks.assert_declared_channels_present(
            _declared_topology(client, config), attempt.trace["declared_channels"]
        )
    finally:
        container.shutdown()


def test_degraded_concurrency_is_re_handshaked_not_assumed() -> None:
    with serve(scenarios.degraded_concurrency()) as container:
        client = container.client()
        exchanges = client.negotiate()
        assert len(exchanges) == 2
        assert exchanges[0]["accepted"] is False
        assert exchanges[0]["degraded_clauses"] == ["lifecycle.concurrency"]
        assert exchanges[1]["accepted"] is True
        assert exchanges[1]["degraded_clauses"] == []
        assert exchanges[0]["agreement_digest"] != exchanges[1]["agreement_digest"]


def test_degraded_quiescence_is_accepted_only_when_the_fallback_is_named() -> None:
    with serve(scenarios.clipped_no_quiescence()) as container:
        client = container.client()
        first = client.preflight()
        assert first["accepted"] is False
        assert first["unaccepted_degraded_clauses"] == ["reward.horizon_quiescence"]
        second = client.negotiate()[-1]
        assert second["accepted"] is True
        assert second["obligations"]["quiescence"] is False


def test_rejected_mandatory_clause_stops_before_any_attempt() -> None:
    with serve(scenarios.rejected_mandatory_clause()) as container:
        client = container.client()
        exchanges = client.negotiate()
        assert len(exchanges) == 1
        payload = exchanges[0]
        assert payload["accepted"] is False
        assert payload["rejected_mandatory_clauses"] == ["evidence.behavior_logprobs"]
        assert set(payload["rejected_mandatory_clauses"]) <= set(MANDATORY_CLAUSES)
        assert client.handshake_id == ""
        with pytest.raises(ContainerError) as caught:
            client.submit(
                task_id="row_0001", idempotency_key="never", policy_config_id="never"
            )
        assert caught.value.payload["error"] == "handshake_absent"
        assert container.attempt_count == 0


def test_clock_skew_beyond_tolerance_is_a_rejected_clause() -> None:
    config = scenarios.skewed_clock()
    with serve(config) as container:
        client = container.client()
        payload = client.negotiate()[-1]
        assert payload["accepted"] is False
        assert config.skew_clause_id in payload["rejected_mandatory_clauses"]
        clause = next(
            row for row in payload["clauses"] if row["clause_id"] == config.skew_clause_id
        )
        assert clause["verdict"] == "rejected"
        assert "skew" in clause["reason"]
        assert payload["clock"]["measured_skew_seconds"] == config.clock_skew_seconds


def test_renderer_profile_mismatch_is_rejected_at_preflight() -> None:
    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        payload = client.preflight(renderer_profile={"config_digest": "sha256:a-second-build"})
        assert payload["accepted"] is False
        assert "policy.renderer_profile_match" in payload["rejected_mandatory_clauses"]
        assert not any(path == "/rollout" for _m, path in container.requested_paths)


def test_rollout_is_refused_for_every_bad_handshake_state() -> None:
    clock = Clock()
    with serve(scenarios.one_call_classification(), clock=clock) as container:
        client = container.client()
        client.negotiate()
        binding = client.bind()
        good = dict(
            task_id="row_0001", policy_config_id=binding["config_id"], correlation={}
        )
        cases = {
            "handshake_absent": dict(good, idempotency_key="a", handshake_id=""),
            "handshake_unknown": dict(good, idempotency_key="b", handshake_id="hs_nope"),
            "agreement_digest_mismatch": dict(
                good, idempotency_key="c", agreement_digest="sha256:not-this-agreement"
            ),
        }
        for expected, body in cases.items():
            with pytest.raises(ContainerError) as caught:
                client.submit(**body)
            assert caught.value.payload["error"] == expected

        clock.advance(601.0)
        with pytest.raises(ContainerError) as caught:
            client.submit(**dict(good, idempotency_key="d"))
        assert caught.value.payload["error"] == "handshake_expired"

        client.negotiate()
        container.revoke_handshake(client.handshake_id)
        with pytest.raises(ContainerError) as caught:
            client.submit(**dict(good, idempotency_key="e"))
        assert caught.value.payload["error"] == "handshake_revoked"
        assert container.attempt_count == 0


def test_capability_change_invalidates_the_handshake_on_renewal() -> None:
    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        accepted = client.negotiate()[-1]
        before = accepted["capability_hash"]
        after = container.bump_capability_epoch()
        assert after != before
        with pytest.raises(ContainerError) as caught:
            client.handshake({"renew_of": accepted["handshake_id"]})
        assert caught.value.payload["error"] == "capability_document_changed"
        assert caught.value.payload["prior_capability_hash"] == before


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_idempotency_key_yields_one_logical_attempt() -> None:
    with serve(scenarios.one_call_classification()) as container:
        client = container.client()
        client.negotiate()
        binding = client.bind()
        body = dict(
            task_id="row_0001",
            idempotency_key="lost-http-response",
            policy_config_id=binding["config_id"],
            correlation={"run_id": "r", "group_id": "g", "sample_index": 2, "seed": 5},
        )
        first = client.submit(**body)
        second = client.submit(**body)
        assert first["rollout_id"] == second["rollout_id"]
        assert first["idempotent_replay"] is False
        assert second["idempotent_replay"] is True
        assert second["correlation"] == first["correlation"]


def test_exactly_one_terminal_result_per_accepted_attempt() -> None:
    with serve(scenarios.multi_turn_environment_reward()) as container:
        client = container.client()
        client.negotiate()
        attempt = client.run_attempt(task_id="row_0001")
        again = client.finalize(attempt.rollout_id)
        cancelled = client.terminate(attempt.rollout_id)
        assert again["already_terminal"] is True
        assert cancelled["already_terminal"] is True
        assert container.terminal_count(attempt.rollout_id) == 1
        kinds = [event["kind"] for event in client.events(attempt.rollout_id)["events"]]
        assert kinds.count("episode") == 1
        assert "failure" not in kinds and "cancellation" not in kinds


def test_expired_lease_fails_the_attempt_and_renewal_extends_it() -> None:
    clock = Clock()
    config = scenarios.one_call_classification()
    with serve(config, clock=clock) as container:
        client = container.client()
        client.negotiate()
        binding = client.bind()
        submitted = client.submit(
            task_id="row_0001", idempotency_key="lease", policy_config_id=binding["config_id"]
        )
        rollout_id = submitted["rollout_id"]
        clock.advance(config.lease_ttl_seconds - 1)
        renewed = client.renew(rollout_id)
        assert renewed["lease_expires_at_offset"] > submitted["lease_expires_at_offset"]
        clock.advance(config.lease_ttl_seconds + 1)
        state = client.state(rollout_id)
        assert state["state"] == "failed"
        assert state["failure_code"] == "lease_expired"
        assert state["lease_expired"] is True
        assert container.terminal_count(rollout_id) == 1


def test_correlation_metadata_round_trips_untouched() -> None:
    correlation = {
        "run_id": "run-9",
        "group_id": "group-4",
        "sample_index": 6,
        "seed": 20260902,
        "policy_revision": 17,
        "agent_instance_id": "home_1",
        "team_id": "team_home",
        "policy_set_revision": "policy-set-20",
        "match_set_revision_id": "match-set-0007",
    }
    with serve(scenarios.competitive_realtime()) as container:
        client = container.client()
        client.negotiate()
        attempt = client.run_attempt(task_id="row_0002", correlation=correlation)
        assert attempt.submit["correlation"] == correlation
        assert attempt.states[-1]["correlation"] == correlation
        assert attempt.trace["correlation"] == correlation
        assert attempt.events[0]["correlation"] == correlation


def test_event_cursor_is_monotone_and_resumable() -> None:
    with serve(scenarios.multi_turn_environment_reward()) as container:
        client = container.client()
        client.negotiate()
        attempt = client.run_attempt(task_id="row_0001")
        page = client.events(attempt.rollout_id, cursor=0)
        cursors = [event["cursor"] for event in page["events"]]
        assert cursors == sorted(cursors) == list(range(1, len(cursors) + 1))
        resumed = client.events(attempt.rollout_id, cursor=page["next_cursor"])
        assert resumed["events"] == []
        midpoint = cursors[len(cursors) // 2]
        tail = client.events(attempt.rollout_id, cursor=midpoint)
        assert [event["cursor"] for event in tail["events"]] == [
            cursor for cursor in cursors if cursor > midpoint
        ]


def test_advertised_concurrency_is_enforced() -> None:
    config = scenarios.degraded_concurrency()
    container, client = _drive(config)
    try:
        binding = client.bind()
        client.submit(
            task_id="row_0001", idempotency_key="slot-1", policy_config_id=binding["config_id"]
        )
        with pytest.raises(ContainerError) as caught:
            client.submit(
                task_id="row_0002",
                idempotency_key="slot-2",
                policy_config_id=binding["config_id"],
            )
        assert caught.value.status == 429
        assert caught.value.payload["error"] == "concurrency_exhausted"
    finally:
        container.shutdown()


# --------------------------------------------------------------------------- #
# Evidence: the conformant cases from the note's list
# --------------------------------------------------------------------------- #


def test_one_call_container_produces_one_trainable_call_and_one_reward() -> None:
    container, client = _drive(scenarios.one_call_classification())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        assert len(attempt.calls) == 1
        call = attempt.calls[0]
        call.validate_for_training()
        assert call.token_capture_provenance == "engine_meta"
        assert len(call.generation_logprobs) == len(call.generation_token_ids)
        assert call.loss_mask.count(1) == len(call.generation_token_ids)
        assert attempt.reward.value() == pytest.approx(0.75)
    finally:
        container.shutdown()


def test_multi_turn_turns_stitch_under_the_strict_prefix_rule() -> None:
    container, client = _drive(scenarios.multi_turn_environment_reward())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        calls = attempt.calls
        assert len(calls) == 3
        for previous, following in zip(calls, calls[1:], strict=False):
            assert_strict_prefix(previous, following)
            assert following.prompt_token_ids[: len(previous.full_sequence)] == (
                previous.full_sequence
            )
            assert following.branch_id == previous.branch_id == "root"
        episode = attempt.episodes[0]
        episode.validate()
        assert len(episode.segments) == 1
        segment = episode.segments[0]
        assert segment.token_ids == calls[-1].full_sequence
        assert segment.trainable_tokens == sum(
            len(call.generation_token_ids) for call in calls
        )
    finally:
        container.shutdown()


def test_declared_compaction_forks_a_branch_and_masks_the_retained_prefix() -> None:
    container, client = _drive(scenarios.multi_turn_declared_compaction())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        first, second, third = attempt.calls
        assert_strict_prefix(first, second)
        assert second.compaction is not None
        assert second.compaction.authored_by_policy is True
        assert second.parent_branch_id == first.branch_id == "root"
        assert second.branch_id != first.branch_id
        assert_strict_prefix(second, third)
        episode = attempt.episodes[0]
        episode.validate()
        assert {segment.branch_id for segment in episode.segments} == {
            "root",
            second.branch_id,
        }
        sealed = next(s for s in episode.segments if s.branch_id == "root")
        assert sealed.trainable_tokens == len(first.generation_token_ids)
        forked = next(s for s in episode.segments if s.branch_id == second.branch_id)
        retained_prefix = forked.loss_mask[: len(second.prompt_token_ids)]
        assert set(retained_prefix) == {0}
    finally:
        container.shutdown()


def test_joint_episode_maps_four_instances_onto_two_parameter_groups() -> None:
    config = scenarios.joint_episode_two_groups()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        assert len(topology.agent_instances) == 4
        assert len(attempt.episodes) == 4
        groups = {
            segment.parameter_group_id
            for episode in attempt.episodes
            for segment in episode.segments
        }
        assert groups == {"pg_alpha", "pg_beta"}
        for instance in topology.agent_instances:
            episode = attempt.episode_for(instance.agent_instance_id)
            episode.validate()
            expected = topology.parameter_group_for(instance.agent_instance_id)
            assert episode.segments[0].parameter_group_id == expected
            for call in attempt.calls_for(instance.agent_instance_id):
                call.validate_for_training()
                assert call.role_id == instance.role_id
                assert call.team_id == instance.team_id
        assert topology.trainable_parameter_groups() == ("pg_alpha", "pg_beta")
        checks.assert_instance_trajectories(
            topology, attempt.trace, disposition=config.partial_roster_disposition
        )
    finally:
        container.shutdown()


def test_group_is_uniform_across_samples_from_one_container() -> None:
    container, client = _drive(scenarios.joint_episode_two_groups())
    try:
        pins = []
        for index, task_id in enumerate(client.task_ids()[:2]):
            attempt = client.run_attempt(
                task_id=task_id,
                idempotency_key=f"sample-{index}",
                correlation={"sample_index": index},
            )
            pins.append(_pin(attempt.states[-1]))
        head = assert_uniform_group(pins)
        assert head.pin_digest == pins[1].pin_digest
        assert head.policy_span_count == 1
    finally:
        container.shutdown()


def test_deferred_verifier_reward_pends_until_finalize() -> None:
    config = scenarios.deferred_verifier()
    container, client = _drive(config)
    try:
        binding = client.bind()
        submitted = client.submit(
            task_id="row_0001", idempotency_key="deferred", policy_config_id=binding["config_id"]
        )
        rollout_id = submitted["rollout_id"]
        assert client.state(rollout_id)["state"] == "awaiting_score"
        status, pending = client.reward(rollout_id)
        assert status == 202
        assert pending["state"] == "pending"
        finalized = client.finalize(rollout_id)
        assert finalized["state"] == "completed"
        status, payload = client.reward(rollout_id)
        assert status == 200
        assert payload["metadata"]["deferred_scoring"] is True
        assert payload["horizon"]["settlement_window_seconds"] == pytest.approx(150.0)
    finally:
        container.shutdown()


def test_rubric_judge_spans_are_recorded_but_untrainable() -> None:
    container, client = _drive(scenarios.rubric_scored_judge())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        judge = [call for call in attempt.calls if call.role_id == "judge"]
        assert len(judge) == 1
        span = judge[0]
        assert span.trainable is False
        assert span.wire_request["author"] == "judge"
        assert set(span.sampled_mask) == {0}
        with pytest.raises(EvidenceError, match="non-trainable"):
            span.validate_for_training()
        assert all(
            span.call_id not in segment.call_ids
            for episode in attempt.episodes
            for segment in episode.segments
        )
        judged = attempt.context_segments_by("judge")
        assert len(judged) == 1
        assert judged[0].call_ids == (span.call_id,)
        assert judged[0].trainable_tokens == 0
        assert judged[0].trainable is False
        assert attempt.reward.metadata["judge_model_id"]
        assert attempt.reward.metadata["judge_spans_trainable"] is False
    finally:
        container.shutdown()


def test_competitive_container_pins_opponents_and_ranks_both_teams() -> None:
    config = scenarios.competitive_realtime()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        assert topology.turn_model == "concurrent_realtime"
        assert topology.horizon is not None

        opponents = {i.agent_instance_id for i in topology.opponent_instances}
        assert opponents == {"away_1", "away_2"}
        for instance in topology.opponent_instances:
            assert instance.pinned_identity == scenarios.PINNED_OPPONENT
            calls = attempt.calls_for(instance.agent_instance_id)
            assert calls, "an opponent's identity must still be recorded"
            for call in calls:
                assert call.trainable is False
                assert call.parameter_group_id is None
                with pytest.raises(EvidenceError):
                    call.validate_for_training()

        opponent_segments = attempt.context_segments_by("opponent")
        assert {segment.agent_instance_id for segment in opponent_segments} == opponents
        for segment in opponent_segments:
            assert segment.trainable_tokens == 0
            assert segment.trainable is False
            assert segment.team_id == "team_away"

        trained = {episode.agent_instance_id for episode in attempt.episodes}
        assert trained == {"home_1", "home_2"}
        for episode in attempt.episodes:
            episode.validate()
            assert episode.team_id == "team_home"

        # Per-instance streams are individually monotone; no environment effect
        # is attributed to two instances.
        for instance_id, calls in _by_instance(attempt.calls).items():
            ticks = [call.created_at for call in calls]
            assert ticks == sorted(ticks), instance_id
        assert len({call.call_id for call in attempt.calls}) == len(attempt.calls)

        reward = attempt.reward
        reward.validate(episode_trace_digest=attempt.trace_digest)
        assert {channel.team_id for channel in reward.channels} == {"team_home", "team_away"}
        assert sorted(channel.rank for channel in reward.channels) == [1, 2]
        assert reward.optimized_channel == "score::team_home"
        checks.assert_declared_channels_present(topology, attempt.trace["declared_channels"])
    finally:
        container.shutdown()


def test_deferred_program_quiesced_scores_within_its_horizon() -> None:
    config = scenarios.deferred_program_quiesced()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        assert topology.actuation_model == "deferred_program"
        for call in attempt.trainable_calls:
            call.validate_for_training()
            assert call.effect_tick_start is not None
            assert call.effect_tick_end is not None
        snapshot = attempt.finalize["snapshot"]
        assert snapshot["quiescence_attested"] is True
        assert snapshot["agent_authored_programs_killed"] is True
        reward = attempt.reward
        reward.validate(episode_trace_digest=attempt.trace_digest)
        assert reward.horizon is not None
        checks.assert_effects_within_horizon(
            attempt.calls,
            horizon_value=reward.horizon.horizon_value,
            quiesced=reward.horizon.quiescence_attested,
            clipped=reward.horizon.clipped,
        )
    finally:
        container.shutdown()


def test_clipped_container_serves_a_snapshot_instead_of_quiescence() -> None:
    container, client = _drive(scenarios.clipped_no_quiescence())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        snapshot = attempt.finalize["snapshot"]
        assert snapshot["quiescence_attested"] is False
        assert snapshot["clipped"] is True
        reward = attempt.reward
        reward.validate(episode_trace_digest=attempt.trace_digest)
        assert reward.horizon is not None
        assert reward.horizon.clipped is True
    finally:
        container.shutdown()


def test_zero_reward_is_scored_not_absent() -> None:
    container, client = _drive(scenarios.zero_reward_classification())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        reward = attempt.reward
        reward.validate(episode_trace_digest=attempt.trace_digest)
        assert reward.value() == 0.0
        assert reward.channels
    finally:
        container.shutdown()


def test_trace_by_reference_resolves_and_matches_its_digest() -> None:
    config = scenarios.artifact_by_reference()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        raw = client.call(
            "GET", client.route("trace_route", rollout_id=attempt.rollout_id)
        )
        assert raw["inline"] is False
        assert raw["trace_ref"].endswith("/trace/body")
        assert raw["trace_digest"] == attempt.trace_digest
        inventory = attempt.artifacts["artifacts"]
        by_role = {row["role"]: row for row in inventory}
        assert by_role["recording"]["by_reference"] is True
        assert by_role["recording"]["digest"]
        assert by_role["trace"]["fetch_handle"] == raw["trace_ref"]
    finally:
        container.shutdown()


def test_tito_and_message_in_reach_identical_prompt_token_ids() -> None:
    prompts: dict[str, tuple[int, ...]] = {}
    for config in (scenarios.one_call_classification(), scenarios.tito_classification()):
        container, client = _drive(config)
        try:
            binding = client.bind_policy(
                kind="trainable", transport=config.sampling_transport
            )
            attempt = client.run_attempt(task_id="row_0002", binding=binding)
            call = attempt.calls[0]
            assert call.sampling_transport == config.sampling_transport
            prompts[config.sampling_transport] = call.prompt_token_ids
        finally:
            container.shutdown()
    assert set(prompts) == {"message_in_capture_out", "tokens_in_tokens_out"}
    assert prompts["message_in_capture_out"] == prompts["tokens_in_tokens_out"]


def test_tito_is_refused_when_it_was_not_declared() -> None:
    container, client = _drive(scenarios.one_call_classification())
    try:
        with pytest.raises(ContainerError) as caught:
            client.bind_policy(kind="trainable", transport="tokens_in_tokens_out")
        assert caught.value.payload["error"] == "transport_unsupported"
    finally:
        container.shutdown()


@pytest.mark.parametrize(
    ("factory", "expected_rule"),
    [
        (scenarios.prompt_budget_truncate, "prompt_budget_truncate_head"),
        (scenarios.prompt_budget_compact, "prompt_budget_compact_middle"),
    ],
)
def test_prompt_budget_policies_record_their_provenance(factory, expected_rule: str) -> None:
    config = factory()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        call = attempt.calls[0]
        assert len(call.prompt_token_ids) == config.max_prompt_tokens
        assert call.compaction is not None
        assert call.compaction.rule == expected_rule
        assert call.compaction.authored_by_policy is False
        call.validate_for_training()
    finally:
        container.shutdown()


def test_prompt_budget_refuse_fails_the_attempt_rather_than_dropping_tokens() -> None:
    container, client = _drive(scenarios.prompt_budget_refuse())
    try:
        binding = client.bind()
        submitted = client.submit(
            task_id="row_0001", idempotency_key="overlong", policy_config_id=binding["config_id"]
        )
        state = client.state(submitted["rollout_id"])
        assert state["state"] == "failed"
        assert state["failure_code"] == "prompt_budget_refused"
        assert container.terminal_count(submitted["rollout_id"]) == 1
        with pytest.raises(ContainerError) as caught:
            client.trace(submitted["rollout_id"])
        assert caught.value.payload["error"] == "trace_not_sealed"
    finally:
        container.shutdown()


def test_finish_reason_distinguishes_a_length_cap_from_a_stop_token() -> None:
    stopped = scenarios.one_call_classification()
    truncated = replace(stopped, container_id="fake-lengthcap", finish_reason="length_cap")
    reasons = {}
    for config in (stopped, truncated):
        container, client = _drive(config)
        try:
            attempt = client.run_attempt(task_id=_first_task(client))
            call = attempt.calls[0]
            reasons[config.finish_reason] = call.finish_reason
            assert call.stop_token_ids == config.renderer_profile.stop_token_ids
        finally:
            container.shutdown()
    assert reasons == {"stop_token": "stop_token", "length_cap": "length_cap"}


def test_probe_walks_the_whole_path_and_stays_out_of_training() -> None:
    config = scenarios.one_call_classification()
    container, client = _drive(config)
    try:
        binding = client.bind(probe=True)
        assert binding["probe"] is True
        attempt = client.run_attempt(
            task_id=_first_task(client), idempotency_key="probe-1", binding=binding
        )
        assert attempt.trace["probe"] is True
        checks.assert_probe_evidence_marked(attempt.calls)
        for call in attempt.calls:
            assert call.token_capture_provenance == "probe_synthetic"
            with pytest.raises(EvidenceError, match="non-trainable"):
                call.validate_for_training()
        # Probe evidence is produced and then refused by its own marking; an
        # empty episode list would hide that the path was walked at all.
        assert attempt.episodes
        for episode in attempt.episodes:
            assert episode.probe is True
            with pytest.raises(EvidenceError, match="probe-derived"):
                episode.validate()

        # Idempotent resubmit of the same key is the same logical attempt.
        replay = client.submit(
            task_id=_first_task(client),
            idempotency_key="probe-1",
            policy_config_id=binding["config_id"],
        )
        assert replay["rollout_id"] == attempt.rollout_id
        assert replay["idempotent_replay"] is True

        # ... and one cancellation.
        cancelled = client.submit(
            task_id=_first_task(client),
            idempotency_key="probe-2",
            policy_config_id=binding["config_id"],
        )
        terminated = client.terminate(cancelled["rollout_id"], reason="probe_cancellation")
        assert terminated["state"] == "cancelled"
        assert container.terminal_count(cancelled["rollout_id"]) == 1

        status, _payload = client.reward(attempt.rollout_id)
        assert status == 200
        assert _routes_touched(container) >= {
            "health_route",
            "capabilities_route",
            "handshake_route",
            "taskset_route",
            "taskset_tasks_route",
            "policy_bind_route",
            "rollout_route",
            "rollout_state_route",
            "rollout_events_route",
            "rollout_renew_route",
            "rollout_finalize_route",
            "rollout_terminate_route",
            "trace_route",
            "artifacts_route",
            "reward_route",
        }
    finally:
        container.shutdown()


def test_probe_binding_is_refused_when_it_was_not_advertised() -> None:
    config = replace(scenarios.one_call_classification(), probe_binding_supported=False)
    container, client = _drive(config)
    try:
        with pytest.raises(ContainerError) as caught:
            client.bind_policy(kind="probe")
        assert caught.value.payload["error"] == "probe_unsupported"
        assert client.capabilities()["policy"]["probe_binding"] is False
    finally:
        container.shutdown()


def test_policy_set_binding_is_atomic_over_the_whole_roster() -> None:
    config = scenarios.competitive_realtime()
    container, client = _drive(config)
    try:
        with pytest.raises(ContainerError) as caught:
            client.bind_policy_set(
                bindings=[{"agent_instance_id": "home_1", "policy_ref": "ckpt::rev0"}]
            )
        assert caught.value.payload["error"] == "partial_roster_binding"
        assert set(caught.value.payload["missing"]) == {"home_2", "away_1", "away_2"}
        full = client.bind()
        assert len(full["bindings"]) == 4
        assert full["atomic"] is True
        trainable = {
            row["agent_instance_id"]: row["parameter_group_id"] for row in full["bindings"]
        }
        assert trainable["home_1"] == "pg_alpha"
        assert trainable["away_1"] is None
    finally:
        container.shutdown()


def test_policy_binding_never_carries_an_embedded_credential() -> None:
    container, client = _drive(scenarios.one_call_classification())
    try:
        with pytest.raises(ContainerError) as caught:
            client.bind_policy(kind="trainable", api_key="sk-should-never-be-inline")
        assert caught.value.payload["error"] == "embedded_credential"
        binding = client.bind_policy(kind="trainable")
        assert binding["sampler_ready"] is True
        assert binding["sampler_origin"].endswith(binding["config_id"])
        assert "api_key" not in binding
    finally:
        container.shutdown()


def test_taskset_lookup_is_deterministic_and_duplicate_free() -> None:
    container, client = _drive(scenarios.one_call_classification())
    try:
        taskset = client.taskset()
        assert taskset["taskset_id"] and taskset["version"]
        assert set(taskset["splits"]) == {"train", "eval"}
        requested = ("row_0002", "row_0001", "row_0002")
        rows = client.taskset_tasks(requested)["rows"]
        assert [row["task_id"] for row in rows] == ["row_0002", "row_0001"]
        assert all(row["topology_ref"] == "topo-solo-1" for row in rows)
        assert all(row["content_digest"] for row in rows)
        assert rows == client.taskset_tasks(requested)["rows"]
        with pytest.raises(ContainerError) as caught:
            client.taskset_tasks(("row_absent",))
        assert caught.value.payload["error"] == "unknown_task"
    finally:
        container.shutdown()


@pytest.mark.parametrize("name", CONFORMANT_NAMES)
def test_replay_of_the_same_configuration_is_bit_for_bit(name: str) -> None:
    def snapshot() -> tuple[object, ...]:
        config = scenarios.CONFORMANT[name]()
        container, client = _drive(config)
        try:
            attempt = client.run_attempt(task_id="row_0003", idempotency_key="replay")
            return (
                attempt.trace_digest,
                tuple(call.prompt_token_ids for call in attempt.calls),
                tuple(call.generation_token_ids for call in attempt.calls),
                tuple(call.generation_logprobs for call in attempt.calls),
                tuple(
                    tuple(segment.loss_mask) for e in attempt.episodes for segment in e.segments
                ),
                attempt.reward_payload,
            )
        finally:
            container.shutdown()

    assert snapshot() == snapshot()


def test_every_call_stamps_the_renderer_profile_that_produced_it() -> None:
    config = scenarios.multi_turn_environment_reward()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        expected = config.renderer_profile.fingerprint
        assert attempt.trace["renderer_profile"]["profile_id"] == (
            config.renderer_profile.profile_id
        )
        for call in attempt.calls:
            assert call.renderer_profile_fingerprint == expected
    finally:
        container.shutdown()


def test_a_foreign_authored_segment_may_never_carry_trainable_tokens() -> None:
    """The fake respects the rule, and the shared record enforces it."""

    container, client = _drive(scenarios.rubric_scored_judge())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        judged = attempt.context_segments_by("judge")[0]
        with pytest.raises(RecordError, match="foreign authorship is never trainable"):
            replace(judged, loss_mask=(1,) * len(judged.token_ids))
    finally:
        container.shutdown()


def test_the_terminal_transition_seals_a_receipt_bound_to_its_agreement() -> None:
    container, client = _drive(scenarios.one_call_classification())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        receipt = attempt.receipt
        assert receipt.rollout_id == attempt.rollout_id
        assert receipt.terminal_status == "completed"
        assert receipt.trace_digest == attempt.trace_digest
        assert receipt.evidence_digest
        assert receipt.agreement_digest == client.agreement_digest
        assert receipt.handshake_id == client.handshake_id
        assert receipt.reward_id == attempt.reward.reward_id
        assert receipt.probe is False
        assert receipt.replacement_index == 0
        assert receipt.replacement_reason is None
    finally:
        container.shutdown()


def test_a_straggler_replacement_is_recorded_as_such() -> None:
    """Replacing a straggler may not silently change group membership."""

    container, client = _drive(scenarios.one_call_classification())
    try:
        binding = client.bind()
        first = client.submit(
            task_id="row_0001", idempotency_key="straggler", policy_config_id=binding["config_id"]
        )
        cancelled = client.terminate(first["rollout_id"], reason="exceeded_horizon_plus_grace")
        assert cancelled["state"] == "cancelled"
        second = client.submit(
            task_id="row_0001",
            idempotency_key="straggler-replacement",
            policy_config_id=binding["config_id"],
            replaces={
                "attempt_id": first["rollout_id"],
                "index": 1,
                "reason": "exceeded_horizon_plus_grace",
            },
        )
        client.state(second["rollout_id"])
        receipt = rollout_receipt_from_payload(client.finalize(second["rollout_id"])["receipt"])
        assert receipt.replaced_attempt_id == first["rollout_id"]
        assert receipt.replacement_index == 1
        assert receipt.replacement_reason == "exceeded_horizon_plus_grace"
    finally:
        container.shutdown()


def test_a_lease_is_advertised_not_derived_from_the_horizon() -> None:
    config = scenarios.competitive_realtime()
    container, client = _drive(config)
    try:
        capabilities = client.capabilities()
        lease = capabilities["lifecycle"]["lease"]
        horizon = capabilities["topology"]["horizon"]
        assert lease["ttl_seconds"] == config.lease_ttl_seconds
        assert lease["renewable"] is True
        assert lease["heartbeat_route"] == DECLARED_ROUTES["rollout_renew_route"]
        # The horizon is advertised separately; the two are different clocks.
        assert horizon["horizon_kind"] == "wall_clock"
        assert horizon["value"] == pytest.approx(5400.0)
        assert horizon["time_dilation"] == pytest.approx(4.0)
        assert lease["ttl_seconds"] != horizon["value"]
    finally:
        container.shutdown()


def test_a_unit_horizon_declares_its_conversion() -> None:
    config = scenarios.deferred_program_quiesced()
    container, client = _drive(config)
    try:
        horizon = client.capabilities()["topology"]["horizon"]
        assert horizon["horizon_kind"] == "env_ticks"
        assert horizon["seconds_per_unit"] == pytest.approx(0.5)
        assert horizon["value"] * horizon["seconds_per_unit"] == pytest.approx(50.0)
    finally:
        container.shutdown()


def test_an_unrenewable_lease_under_the_horizon_is_a_rejected_clause() -> None:
    with serve(scenarios.lease_too_short_for_horizon()) as container:
        client = container.client()
        payload = client.negotiate()[-1]
        assert payload["accepted"] is False
        assert payload["rejected_mandatory_clauses"] == ["lifecycle.lease_renewal"]
        clause = next(
            row for row in payload["clauses"] if row["clause_id"] == "lifecycle.lease_renewal"
        )
        assert clause["verdict"] == "rejected"
        assert "shorter than the declared horizon" in clause["reason"]
        assert payload["obligations"]["lease_renewable"] is False


def test_the_seed_is_load_bearing_for_the_sampled_evidence() -> None:
    """Same row and renderer profile, different seed: same prompt, new sample."""

    base = scenarios.one_call_classification()
    other = replace(base, container_id="fake-oneshot-b", seed=base.seed + 1)
    prompts, generations = [], []
    for config in (base, other):
        container, client = _drive(config)
        try:
            attempt = client.run_attempt(task_id="row_0001", idempotency_key="seeded")
            call = attempt.calls[0]
            prompts.append(call.prompt_token_ids)
            generations.append((call.generation_token_ids, call.generation_logprobs))
        finally:
            container.shutdown()
    assert prompts[0] == prompts[1]
    assert generations[0] != generations[1]


# --------------------------------------------------------------------------- #
# Non-conformant: the right error, not any error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("factory", "fragment"),
    [
        (scenarios.missing_logprobs, "logprob length 0 != generated token count"),
        (scenarios.sentinel_logprobs, "is the provider sentinel"),
        (scenarios.zero_logprobs, "logprobs are identically zero"),
        (scenarios.short_logprobs, "logprob length 4 != generated token count"),
    ],
)
def test_bad_logprob_vectors_are_each_an_evidence_error(factory, fragment: str) -> None:
    container, client = _drive(factory())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        call = attempt.calls[0]
        with pytest.raises(EvidenceError) as caught:
            call.validate_for_training()
        assert type(caught.value) is EvidenceError
        assert fragment in str(caught.value)
    finally:
        container.shutdown()


def test_absent_reward_is_an_evidence_error_not_a_zero() -> None:
    container, client = _drive(scenarios.absent_reward())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        for call in attempt.trainable_calls:
            call.validate_for_training()
        with pytest.raises(EvidenceError) as caught:
            attempt.reward.validate(episode_trace_digest=attempt.trace_digest)
        assert type(caught.value) is EvidenceError
        assert "absent is not zero" in str(caught.value)
        assert attempt.reward_payload is not None
        assert attempt.reward_payload["channels"] == []
    finally:
        container.shutdown()


def test_dropped_cross_team_channel_is_an_evidence_error() -> None:
    config = scenarios.dropped_cross_team_channel()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        for call in attempt.trainable_calls:
            call.validate_for_training()
        with pytest.raises(EvidenceError) as caught:
            checks.assert_declared_channels_present(
                topology, attempt.trace["declared_channels"]
            )
        assert type(caught.value) is EvidenceError
        assert "cross_team channel 'public' returned no messages" in str(caught.value)
    finally:
        container.shutdown()


def test_rerendering_second_turn_is_an_unexplained_prefix_divergence() -> None:
    container, client = _drive(scenarios.rerendering_multi_turn())
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        first, second = attempt.calls[0], attempt.calls[1]
        assert second.compaction is None
        assert second.branch_id == first.branch_id
        with pytest.raises(EvidenceError) as caught:
            assert_strict_prefix(first, second)
        assert type(caught.value) is EvidenceError
        assert "no branch record and no declared compaction" in str(caught.value)
    finally:
        container.shutdown()


def test_flattened_wire_is_an_evidence_error() -> None:
    config = scenarios.flattened_wire()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        assert config.wire_api == "responses"
        assert attempt.calls[0].wire_api == "responses"
        with pytest.raises(EvidenceError) as caught:
            checks.assert_no_flattened_wire(attempt.calls, declared_wire_api="responses")
        assert type(caught.value) is EvidenceError
        assert "a flattened wire is a different dataset" in str(caught.value)
    finally:
        container.shutdown()


def test_opponent_resolved_as_latest_is_a_topology_error() -> None:
    config = scenarios.opponent_alias_resolution()
    container, client = _drive(config)
    try:
        with pytest.raises(TopologyError) as caught:
            _declared_topology(client, config)
        assert type(caught.value) is TopologyError
        assert "resolves alias 'latest'" in str(caught.value)
        with pytest.raises(ContainerError) as refused:
            client.bind()
        assert refused.value.payload["error"] == "alias_opponent_binding"
    finally:
        container.shutdown()


def test_missing_instance_trajectory_is_refused_under_refuse() -> None:
    config = scenarios.missing_instance_trajectory()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        assert len(attempt.episodes) == 3
        with pytest.raises(TopologyError) as caught:
            checks.assert_instance_trajectories(topology, attempt.trace, disposition="refuse")
        assert type(caught.value) is TopologyError
        assert "missing instances: ('inst_b2',)" in str(caught.value)
    finally:
        container.shutdown()


def test_missing_instance_trajectory_is_recorded_under_drop_instance() -> None:
    config = scenarios.missing_instance_dropped()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        topology = _declared_topology(client, config)
        missing, absences = checks.assert_instance_trajectories(
            topology, attempt.trace, disposition="drop_instance"
        )
        assert missing == ("inst_b2",)
        assert [row["agent_instance_id"] for row in absences] == ["inst_b2"]
        assert absences[0]["last_live_tick"] is not None
        assert absences[0]["absent_at_tick"] is not None
        for episode in attempt.episodes:
            episode.validate()
        attempt.reward.validate(episode_trace_digest=attempt.trace_digest)
        with pytest.raises(EvidenceError):
            attempt.episode_for("inst_b2")
    finally:
        container.shutdown()


def test_probe_evidence_indistinguishable_from_real_is_an_evidence_error() -> None:
    container, client = _drive(scenarios.probe_indistinguishable())
    try:
        binding = client.bind(probe=True)
        attempt = client.run_attempt(task_id=_first_task(client), binding=binding)
        # It passes training validation, which is exactly the problem.
        attempt.calls[0].validate_for_training()
        with pytest.raises(EvidenceError) as caught:
            checks.assert_probe_evidence_marked(attempt.calls)
        assert type(caught.value) is EvidenceError
        assert "indistinguishable from real evidence" in str(caught.value)
    finally:
        container.shutdown()


def test_unquiesced_deferred_program_fails_rather_than_inflating_reward() -> None:
    config = scenarios.deferred_program_unquiesced()
    container, client = _drive(config)
    try:
        attempt = client.run_attempt(task_id=_first_task(client))
        snapshot = attempt.finalize["snapshot"]
        assert snapshot["quiescence_attested"] is False
        assert snapshot["clipped"] is False
        with pytest.raises(EvidenceError) as caught:
            attempt.reward.validate(episode_trace_digest=attempt.trace_digest)
        assert type(caught.value) is EvidenceError
        assert "neither a quiescence attestation nor a horizon-clipped snapshot" in str(
            caught.value
        )
        with pytest.raises(EvidenceError) as effects:
            checks.assert_effects_within_horizon(
                attempt.calls,
                horizon_value=config.horizon.value,
                quiesced=False,
                clipped=False,
            )
        assert type(effects.value) is EvidenceError
        assert "past horizon" in str(effects.value)
    finally:
        container.shutdown()


def test_match_set_drift_inside_one_group_is_a_mixed_group_error() -> None:
    requested = {"match_set_revision_id": "match-set-0007"}
    pins = []
    for index, factory in enumerate(
        (scenarios.competitive_realtime, scenarios.competitive_match_set_drift)
    ):
        container, client = _drive(factory())
        try:
            attempt = client.run_attempt(
                task_id="row_0001",
                idempotency_key="member",
                correlation={"sample_index": index, **requested},
            )
            pins.append(_pin(attempt.states[-1]))
        finally:
            container.shutdown()
    assert pins[0].match_set_revision_id == "match-set-0007"
    assert pins[1].match_set_revision_id == "match-set-0099"
    with pytest.raises(MixedGroupError) as caught:
        assert_uniform_group(pins)
    assert type(caught.value) is MixedGroupError
    assert "mixes match_set_revision_id" in str(caught.value)


def test_every_non_conformant_scenario_is_registered_with_its_error_type() -> None:
    expected = {
        "missing_logprobs": EvidenceError,
        "sentinel_logprobs": EvidenceError,
        "zero_logprobs": EvidenceError,
        "short_logprobs": EvidenceError,
        "absent_reward": EvidenceError,
        "dropped_cross_team_channel": EvidenceError,
        "rerendering_multi_turn": EvidenceError,
        "flattened_wire": EvidenceError,
        "opponent_alias_resolution": TopologyError,
        "missing_instance_trajectory": TopologyError,
        "probe_indistinguishable": EvidenceError,
        "deferred_program_unquiesced": EvidenceError,
        "competitive_match_set_drift": MixedGroupError,
    }
    for name, error in expected.items():
        assert scenarios.NON_CONFORMANT[name][1] is error, name
    assert set(expected) <= set(scenarios.NON_CONFORMANT)


# --------------------------------------------------------------------------- #
# House rules
# --------------------------------------------------------------------------- #


def test_fakes_never_name_a_task_harness_or_env() -> None:
    for path in sorted(FAKES_DIR.glob("*.py")):
        source = path.read_text().lower()
        for literal in BANNED_LITERALS:
            assert not re.search(rf"\b{re.escape(literal)}\b", source), f"{path.name}:{literal}"


def test_behavior_is_selected_by_capability_configuration_not_task_row() -> None:
    """Every task row exercises the same declared behavior, only new tokens."""

    container, client = _drive(scenarios.multi_turn_environment_reward())
    try:
        shapes = set()
        prompts = set()
        for index, task_id in enumerate(client.task_ids()):
            attempt = client.run_attempt(task_id=task_id, idempotency_key=f"row-{index}")
            shapes.add(
                (
                    len(attempt.calls),
                    len(attempt.episodes),
                    tuple(channel.channel_id for channel in attempt.reward.channels),
                    attempt.reward.value(),
                )
            )
            prompts.add(attempt.calls[0].prompt_token_ids)
        assert len(shapes) == 1
        assert len(prompts) == len(client.task_ids())
    finally:
        container.shutdown()
