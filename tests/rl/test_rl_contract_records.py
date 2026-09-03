"""Foundation records: the rules a batch is assembled under."""

from __future__ import annotations

import pytest

from synth_optimizers.contracts.rl_clauses import (
    ALL_CLAUSES,
    MANDATORY_CLAUSES,
    OPTIONAL_CLAUSES,
)
from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    GroupPin,
    Horizon,
    MixedGroupError,
    Team,
    Topology,
    TopologyError,
    assert_uniform_group,
)
from synth_optimizers.contracts.rl_records import (
    LOGPROB_SENTINEL,
    BehaviorFingerprint,
    CompactionProvenance,
    EvidenceError,
    InferenceCall,
    RecordError,
    RendererProfile,
    assert_strict_prefix,
)


def profile(**overrides: object) -> RendererProfile:
    payload = {
        "profile_id": "renderers.gpt-oss.low.v1",
        "package": "renderers",
        "package_version": "0.1.11",
        "config_digest": "sha256:cfg",
        "tokenizer_id": "openai/gpt-oss-20b",
        "tokenizer_digest": "sha256:tok",
        "stop_token_ids": [200002, 199999],
    }
    payload.update(overrides)
    return RendererProfile.from_payload(payload)


def call(**overrides: object) -> InferenceCall:
    payload: dict[str, object] = {
        "call_id": "call_1",
        "proxy_request_id": "prid_1",
        "rollout_id": "rollout_1",
        "group_id": "group_1",
        "sample_index": 0,
        "behavior_fingerprint": "fp",
        "policy_revision": 3,
        "wire_api": "chat_completions",
        "sampling_transport": "message_in_capture_out",
        "token_capture_provenance": "engine_meta",
        "prompt_token_ids": (1, 2, 3),
        "generation_token_ids": (4, 5),
        "generation_logprobs": (-0.5, -1.25),
        "sampled_mask": (1, 1),
        "finish_reason": "stop_token",
    }
    payload.update(overrides)
    return InferenceCall(**payload)  # type: ignore[arg-type]


def test_renderer_profile_fingerprint_detects_any_pinned_change() -> None:
    base = profile()
    assert base.fingerprint == profile().fingerprint
    for field, value in (
        ("package_version", "0.1.12"),
        ("config_digest", "sha256:other"),
        ("tokenizer_digest", "sha256:other"),
        ("stop_token_ids", [1]),
        ("add_generation_prompt", False),
    ):
        assert profile(**{field: value}).fingerprint != base.fingerprint
    with pytest.raises(RecordError):
        base.assert_matches(profile(package_version="0.1.12"))


def test_behavior_fingerprint_separates_wires_and_transports() -> None:
    def fingerprint(**overrides: object) -> str:
        payload: dict[str, object] = {
            "renderer_profile": profile(),
            "model_family": "gpt_oss",
            "model_id": "openai/gpt-oss-20b",
            "policy_revision": 4,
            "wire_api": "chat_completions",
            "sampling_transport": "message_in_capture_out",
        }
        payload.update(overrides)
        return BehaviorFingerprint(**payload).value  # type: ignore[arg-type]

    assert fingerprint() != fingerprint(wire_api="responses")
    assert fingerprint() != fingerprint(sampling_transport="tokens_in_tokens_out")
    assert fingerprint() != fingerprint(policy_revision=5)
    with pytest.raises(RecordError):
        fingerprint(wire_api="grpc")


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"generation_logprobs": (-0.5,)}, "logprob length"),
        ({"generation_logprobs": (-0.5, float("nan"))}, "not finite"),
        ({"generation_logprobs": (-0.5, float("inf"))}, "not finite"),
        ({"generation_logprobs": (-0.5, LOGPROB_SENTINEL)}, "sentinel"),
        ({"generation_logprobs": (0.0, 0.0)}, "identically zero"),
        ({"sampled_mask": (1,)}, "sampled mask"),
        ({"token_capture_provenance": "wire_derived"}, "engine-level token capture"),
        ({"token_capture_provenance": "probe_synthetic"}, "engine-level token capture"),
        ({"trainable": False}, "non-trainable"),
        ({"generation_token_ids": ()}, "no generated tokens"),
    ],
)
def test_invalid_evidence_never_reaches_a_batch(
    overrides: dict[str, object], fragment: str
) -> None:
    if "generation_token_ids" in overrides:
        overrides.setdefault("generation_logprobs", ())
        overrides.setdefault("sampled_mask", ())
    with pytest.raises(EvidenceError, match=fragment):
        call(**overrides).validate_for_training()


def test_valid_call_has_prompt_masked_and_generation_trainable() -> None:
    record = call()
    record.validate_for_training()
    assert record.loss_mask == (0, 0, 0, 1, 1)
    assert record.full_sequence == (1, 2, 3, 4, 5)


def test_tool_loop_stitches_on_a_byte_for_byte_prefix() -> None:
    first = call()
    second = call(call_id="call_2", prompt_token_ids=(1, 2, 3, 4, 5, 6))
    assert_strict_prefix(first, second)


def test_unexplained_divergence_is_an_evidence_failure() -> None:
    first = call()
    rerendered = call(call_id="call_2", prompt_token_ids=(1, 2, 9, 4, 5))
    with pytest.raises(EvidenceError, match="diverges from"):
        assert_strict_prefix(first, rerendered)


def test_declared_compaction_forks_a_branch_instead_of_diverging() -> None:
    first = call()
    forked = call(
        call_id="call_2",
        prompt_token_ids=(1, 2, 9),
        branch_id="branch_1",
        parent_branch_id="root",
        compaction=CompactionProvenance(rule="drop_middle", divergence_index=2),
    )
    assert_strict_prefix(first, forked)

    same_branch = call(
        call_id="call_3",
        prompt_token_ids=(1, 2, 9),
        parent_branch_id="root",
        compaction=CompactionProvenance(rule="drop_middle", divergence_index=2),
    )
    with pytest.raises(EvidenceError, match="must open a new branch"):
        assert_strict_prefix(first, same_branch)

    orphan = call(
        call_id="call_4",
        prompt_token_ids=(1, 2, 9),
        branch_id="branch_2",
        compaction=CompactionProvenance(rule="drop_middle", divergence_index=2),
    )
    with pytest.raises(EvidenceError, match="does not fork from"):
        assert_strict_prefix(first, orphan)


def pin(**overrides: object) -> GroupPin:
    payload: dict[str, object] = {
        "group_id": "group_1",
        "run_id": "run_1",
        "algorithm_plan_hash": "plan",
        "behavior_fingerprint": "fp",
        "policy_revision": 3,
        "wire_api": "chat_completions",
        "sampling_transport": "message_in_capture_out",
        "policy_kind": "declared",
        "model_family": "gpt_oss",
        "container_image_digest": "sha256:img",
        "container_contract_hash": "sha256:contract",
        "handshake_agreement_digest": "sha256:agreement",
        "task_family": "family",
        "cardinality": 4,
    }
    payload.update(overrides)
    return GroupPin(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algorithm_plan_hash", "other-plan"),
        ("behavior_fingerprint", "other-fp"),
        ("policy_revision", 4),
        ("wire_api", "responses"),
        ("sampling_transport", "tokens_in_tokens_out"),
        ("policy_kind", "other"),
        ("model_family", "laguna"),
        ("container_image_digest", "sha256:other"),
        ("container_contract_hash", "sha256:other"),
        ("handshake_agreement_digest", "sha256:other"),
        ("task_family", "other"),
        ("policy_set_revision_id", "party-set-21"),
        ("match_set_revision_id", "match-set-8"),
        ("topology_id", "other"),
    ],
)
def test_every_pinned_field_rejects_a_mixed_group(field: str, value: object) -> None:
    with pytest.raises(MixedGroupError, match=field):
        assert_uniform_group([pin(), pin(**{field: value})])


def test_uniform_group_is_accepted_and_a_group_may_not_straddle_revisions() -> None:
    assert assert_uniform_group([pin(), pin()]).group_id == "group_1"
    with pytest.raises(RecordError, match="policy_span_count"):
        pin(policy_span_count=2)


def runite_topology(**overrides: object) -> Topology:
    instances = [
        AgentInstance(f"terra_{s}", "miner", "miner", "terra", True) for s in "abcde"
    ]
    instances.append(AgentInstance("terra_f", "scout", "scout", "terra", True))
    instances.extend(
        AgentInstance(f"gemini37_{s}", "miner", "opponent", "gemini37", False, "ckpt_x")
        for s in "abcdef"
    )
    payload: dict[str, object] = {
        "topology_id": "runite-race-4x6",
        "turn_model": "concurrent_realtime",
        "actuation_model": "deferred_program",
        "reward_relation": "competitive_rank",
        "agent_instances": tuple(instances),
        "teams": (Team("terra", True, 3), Team("gemini37", False)),
        "horizon": Horizon("wall_clock", 5400.0, 4.0),
        "parameter_groups": {"miner": "miner_policy", "scout": "scout_policy"},
    }
    payload.update(overrides)
    return Topology(**payload)  # type: ignore[arg-type]


def test_topology_binds_instances_to_parameter_groups_without_role_branches() -> None:
    topology = runite_topology()
    assert topology.is_multi_policy
    assert topology.parameter_group_for("terra_a") == "miner_policy"
    assert topology.parameter_group_for("terra_f") == "scout_policy"
    assert topology.trainable_parameter_groups() == ("miner_policy", "scout_policy")
    assert len(topology.opponent_instances) == 6
    with pytest.raises(TopologyError, match="not trainable"):
        topology.parameter_group_for("gemini37_a")


def test_opponent_instance_must_pin_an_immutable_identity() -> None:
    with pytest.raises(TopologyError, match="pin an immutable identity"):
        AgentInstance("rival", "miner", "opponent", "rival_team", False)


def test_realtime_topology_requires_a_declared_horizon() -> None:
    with pytest.raises(TopologyError, match="must declare a horizon"):
        runite_topology(horizon=None)


def test_partial_roster_follows_the_declared_disposition() -> None:
    topology = runite_topology()
    live = [i.agent_instance_id for i in topology.agent_instances]
    assert topology.check_roster(live, disposition="drop_instance") == ()

    lost_one = [i for i in live if i != "terra_d"]
    with pytest.raises(TopologyError, match="missing instances"):
        topology.check_roster(lost_one, disposition="refuse")
    assert topology.check_roster(lost_one, disposition="drop_instance") == ("terra_d",)

    collapsed = [i for i in live if not i.startswith("terra_") or i == "terra_a"]
    with pytest.raises(TopologyError, match="minimum viable roster"):
        topology.check_roster(collapsed, disposition="drop_instance")


def test_clause_registry_is_complete_and_partitioned() -> None:
    assert len(ALL_CLAUSES) == len(set(ALL_CLAUSES))
    assert OPTIONAL_CLAUSES <= set(ALL_CLAUSES)
    assert set(MANDATORY_CLAUSES).isdisjoint(OPTIONAL_CLAUSES)
    assert set(MANDATORY_CLAUSES) | OPTIONAL_CLAUSES == set(ALL_CLAUSES)


def test_probe_derived_episode_may_not_enter_a_group_or_batch() -> None:
    from synth_optimizers.contracts.rl_records import TrainableEpisode, TrainableSegment

    segment = TrainableSegment(
        token_ids=(1, 2, 3),
        loss_mask=(0, 1, 1),
        behavior_logprobs=(0.0, -0.5, -0.25),
    )
    kwargs = {
        "rollout_id": "rollout_1",
        "task_id": "task_1",
        "seed": 7,
        "policy_revision": 3,
        "behavior_fingerprint": "fp",
        "segments": (segment,),
        "terminal_status": "completed",
        "trace_digest": "sha256:trace",
    }
    TrainableEpisode(**kwargs).validate()
    with pytest.raises(EvidenceError, match="probe-derived"):
        TrainableEpisode(**kwargs, probe=True).validate()


def test_horizon_lease_covers_step_and_tick_budgets() -> None:
    wall = Horizon("wall_clock", 5400.0, 4.0, grace_seconds=120.0)
    assert wall.lease_seconds == 5520.0
    ticks = Horizon("env_ticks", 1000.0, seconds_per_unit=0.6, grace_seconds=30.0)
    assert ticks.lease_seconds == 630.0
    with pytest.raises(TopologyError, match="seconds_per_unit"):
        Horizon("env_ticks", 1000.0, seconds_per_unit=0.0)


def test_foreign_authored_span_may_not_carry_trainable_tokens() -> None:
    from synth_optimizers.contracts.rl_records import TrainableSegment

    kwargs = {
        "token_ids": (1, 2, 3),
        "loss_mask": (0, 1, 1),
        "behavior_logprobs": (0.0, -0.5, -0.25),
    }
    assert TrainableSegment(**kwargs).trainable
    for author in ("foreign_agent", "opponent", "verifier", "judge", "harness"):
        with pytest.raises(RecordError, match="never trainable"):
            TrainableSegment(**kwargs, author_kind=author)
        masked = TrainableSegment(
            token_ids=(1, 2, 3),
            loss_mask=(0, 0, 0),
            behavior_logprobs=(0.0, 0.0, 0.0),
            author_kind=author,
        )
        assert not masked.trainable
    with pytest.raises(RecordError, match="unknown author_kind"):
        TrainableSegment(**kwargs, author_kind="mystery")


def test_effect_interval_must_not_end_before_it_starts() -> None:
    from synth_optimizers.contracts.rl_records import TrainableSegment

    with pytest.raises(RecordError, match="ends before it starts"):
        TrainableSegment(
            token_ids=(1,),
            loss_mask=(1,),
            behavior_logprobs=(-0.5,),
            effect_tick_start=90,
            effect_tick_end=12,
        )


def test_team_channel_lookup_refuses_absence_and_ambiguity() -> None:
    from synth_optimizers.contracts.rl_records import RewardChannel, RewardRecord

    def record(*channels: RewardChannel) -> RewardRecord:
        return RewardRecord(
            reward_id="reward_1",
            rollout_id="rollout_1",
            trace_digest="sha256:trace",
            channels=channels,
            optimized_channel="team_rank",
            terminal_status="completed",
            evaluation_plan_id="plan_1",
        )

    ranked = record(
        RewardChannel("team_rank", "terra", 14.0, rank=1),
        RewardChannel("team_rank_rival", "gemini37", 13.0, rank=2),
    )
    assert ranked.value_for_team("terra") == 14.0
    assert ranked.channel_for("gemini37").rank == 2
    with pytest.raises(RecordError, match="no channel for team"):
        ranked.value_for_team("grok46")
    doubled = record(
        RewardChannel("team_rank", "terra", 14.0),
        RewardChannel("team_margin", "terra", 1.0),
    )
    with pytest.raises(RecordError, match="must be unambiguous"):
        doubled.value_for_team("terra")


def test_quiescent_container_is_not_penalised_for_reading_the_reward() -> None:
    from synth_optimizers.contracts.rl_records import HorizonEvidence

    # The failure the runite run actually had: still moving, read late, no clip.
    with pytest.raises(EvidenceError, match="neither a quiescence attestation"):
        HorizonEvidence("wall_clock", 5400.0, 1560.0, clipped=False,
                        quiescence_attested=False).validate()
    # Quiesced then read at leisure is harmless: nothing was moving.
    HorizonEvidence("wall_clock", 5400.0, 90.0, clipped=False,
                    quiescence_attested=True).validate()
    # Clipped to the horizon is equally fine without an attestation.
    HorizonEvidence("wall_clock", 5400.0, 1560.0, clipped=True,
                    quiescence_attested=False).validate()
    # What the window bounds is credited settlement, not read latency.
    with pytest.raises(EvidenceError, match="beyond"):
        HorizonEvidence("wall_clock", 5400.0, 200.0, clipped=False, quiescence_attested=True,
                        settlement_window_seconds=150.0,
                        credited_settlement_seconds=200.0).validate()


def test_reward_must_claim_a_terminal_status_and_name_its_rollout() -> None:
    from synth_optimizers.contracts.rl_records import RewardChannel, RewardRecord

    def record(**overrides: object) -> RewardRecord:
        payload: dict[str, object] = {
            "reward_id": "reward_1",
            "rollout_id": "rollout_1",
            "trace_digest": "sha256:trace",
            "channels": (RewardChannel("reward", None, 1.0),),
            "optimized_channel": "reward",
            "terminal_status": "completed",
            "evaluation_plan_id": "plan_1",
        }
        payload.update(overrides)
        return RewardRecord(**payload)  # type: ignore[arg-type]

    record().validate()
    with pytest.raises(EvidenceError, match="non-terminal status"):
        record(terminal_status="running").validate()
    with pytest.raises(EvidenceError, match="names no rollout"):
        record(rollout_id="  ").validate()


def test_refuse_team_actually_refuses_the_team_it_is_named_for() -> None:
    topology = runite_topology()
    live = [
        i.agent_instance_id
        for i in topology.agent_instances
        if not i.agent_instance_id.startswith("terra_") or i.agent_instance_id == "terra_a"
    ]
    outcome = topology.roster_disposition(live, disposition="refuse_team")
    assert outcome.refused_teams == ("terra",)
    assert outcome.degraded
    with pytest.raises(TopologyError, match="minimum viable roster"):
        topology.roster_disposition(live, disposition="drop_instance")


def test_a_truncated_prompt_reads_differently_from_a_content_divergence() -> None:
    first = call()
    truncated = call(call_id="call_2", prompt_token_ids=(1, 2))
    with pytest.raises(EvidenceError, match="truncates"):
        assert_strict_prefix(first, truncated)
    diverged = call(call_id="call_3", prompt_token_ids=(1, 9, 3, 4, 5))
    with pytest.raises(EvidenceError, match="diverges from"):
        assert_strict_prefix(first, diverged)
