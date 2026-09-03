"""Batch assembly: validated evidence only, per-group advantages, packing."""

from __future__ import annotations

from typing import Any

import pytest

from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    GroupPin,
    MixedGroupError,
    Team,
    Topology,
)
from synth_optimizers.contracts.rl_records import (
    LOGPROB_SENTINEL,
    EvidenceError,
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
)
from synth_optimizers.rl.assembly import (
    DROP_FOREIGN_AUTHOR,
    DROP_NO_TRAINABLE_TOKENS,
    DROP_UNATTRIBUTED_AUTHOR,
    SOLO_PARAMETER_GROUP,
    AssemblyError,
    EvidenceBundle,
    assemble,
)
from synth_optimizers.rl.credit import CreditSample, estimate
from synth_optimizers.rl.plan import PRESETS, expand

CISPO = PRESETS["cispo"]
FINGERPRINT = "behavior-fingerprint-1"


def _pin(group_id: str, *, plan_hash: str | None = None, **overrides: Any) -> GroupPin:
    base: dict[str, Any] = {
        "group_id": group_id,
        "run_id": "run-1",
        "algorithm_plan_hash": plan_hash or CISPO.plan_hash,
        "behavior_fingerprint": FINGERPRINT,
        "policy_revision": 3,
        "wire_api": "chat_completions",
        "sampling_transport": "message_in_capture_out",
        "policy_kind": "adapter",
        "model_family": "family-a",
        "container_image_digest": "sha256:image",
        "container_contract_hash": "contract-1",
        "handshake_agreement_digest": "agreement-1",
        "task_family": "family-x",
        "cardinality": 2,
    }
    base.update(overrides)
    return GroupPin(**base)


def _segment(
    *,
    base: int = 100,
    tokens: int = 4,
    trainable: int = 3,
    branch_id: str = "root",
    parameter_group_id: str | None = None,
    agent_instance_id: str | None = None,
    logprobs: tuple[float, ...] | None = None,
) -> TrainableSegment:
    mask = tuple(0 if index < tokens - trainable else 1 for index in range(tokens))
    return TrainableSegment(
        token_ids=tuple(range(base, base + tokens)),
        loss_mask=mask,
        behavior_logprobs=logprobs or tuple(-0.1 * (index + 1) for index in range(tokens)),
        branch_id=branch_id,
        parameter_group_id=parameter_group_id,
        agent_instance_id=agent_instance_id,
        call_ids=(f"call-{base}",),
    )


def _episode(
    rollout_id: str,
    segments: tuple[TrainableSegment, ...],
    *,
    team_id: str | None = None,
    agent_instance_id: str | None = None,
    seed: int = 0,
) -> TrainableEpisode:
    return TrainableEpisode(
        rollout_id=rollout_id,
        task_id="task-1",
        seed=seed,
        policy_revision=3,
        behavior_fingerprint=FINGERPRINT,
        segments=segments,
        terminal_status="completed",
        team_id=team_id,
        agent_instance_id=agent_instance_id,
        trace_digest=f"trace-{rollout_id}",
    )


def _reward(
    rollout_id: str,
    measure: float,
    *,
    channel_id: str = "outcome",
    team_id: str | None = None,
    extra: tuple[RewardChannel, ...] = (),
) -> RewardRecord:
    return RewardRecord(
        reward_id=f"reward-{rollout_id}",
        rollout_id=rollout_id,
        trace_digest=f"trace-{rollout_id}",
        channels=(RewardChannel(channel_id=channel_id, team_id=team_id, measure=measure),)
        + extra,
        optimized_channel=channel_id,
        terminal_status="completed",
        evaluation_plan_id="evaluation-plan-1",
    )


def _solo_group(
    group_id: str,
    rewards: list[float],
    *,
    plan_hash: str | None = None,
    tokens: int = 4,
) -> list[EvidenceBundle]:
    pin = _pin(group_id, plan_hash=plan_hash, cardinality=len(rewards))
    bundles = []
    for index, reward in enumerate(rewards):
        rollout_id = f"{group_id}-r{index}"
        bundles.append(
            EvidenceBundle(
                group_id=group_id,
                sample_index=index,
                pin=pin,
                episode=_episode(
                    rollout_id, (_segment(base=100 + 10 * index, tokens=tokens),), seed=index
                ),
                reward=_reward(rollout_id, reward),
                source_run_id="run-1",
            )
        )
    return bundles


HOME_TOPOLOGY = Topology(
    topology_id="topology-1",
    turn_model="sequential",
    actuation_model="direct_action",
    reward_relation="competitive_rank",
    agent_instances=(
        AgentInstance(
            agent_instance_id="instance_a",
            role_id="role_one",
            policy_type_id="policy_alpha",
            team_id="home",
            trainable=True,
        ),
        AgentInstance(
            agent_instance_id="instance_b",
            role_id="role_two",
            policy_type_id="policy_beta",
            team_id="home",
            trainable=True,
        ),
        AgentInstance(
            agent_instance_id="instance_x",
            role_id="role_one",
            policy_type_id="policy_frozen",
            team_id="away",
            trainable=False,
            pinned_identity="checkpoint-0007",
        ),
    ),
    teams=(Team(team_id="home", trainable=True), Team(team_id="away", trainable=False)),
    parameter_groups={"policy_alpha": "pg_alpha", "policy_beta": "pg_beta"},
)

SHARED_GROUP_TOPOLOGY = Topology(
    topology_id="topology-2",
    turn_model="sequential",
    actuation_model="direct_action",
    reward_relation="cooperative",
    agent_instances=(
        AgentInstance(
            agent_instance_id="instance_quiet",
            role_id="role_one",
            policy_type_id="policy_alpha",
            team_id="home",
            trainable=True,
        ),
        AgentInstance(
            agent_instance_id="instance_chatty",
            role_id="role_two",
            policy_type_id="policy_alpha",
            team_id="home",
            trainable=True,
        ),
    ),
    teams=(Team(team_id="home", trainable=True),),
    parameter_groups={"policy_alpha": "pg_alpha"},
)


def _joint_group(
    group_id: str,
    rewards: list[float],
    *,
    topology: Topology = HOME_TOPOLOGY,
    plan: Any = CISPO,
) -> list[EvidenceBundle]:
    pin = _pin(
        group_id,
        plan_hash=plan.plan_hash,
        cardinality=len(rewards),
        topology_id=topology.topology_id,
        match_set_revision_id="match-set-0007",
    )
    bundles = []
    for index, reward in enumerate(rewards):
        rollout_id = f"{group_id}-r{index}"
        segments = (
            _segment(
                base=200 + 20 * index, tokens=4, trainable=3, agent_instance_id="instance_a"
            ),
            _segment(
                base=300 + 20 * index, tokens=5, trainable=4, agent_instance_id="instance_b"
            ),
            _segment(
                base=400 + 20 * index, tokens=4, trainable=3, agent_instance_id="instance_x"
            ),
        )
        bundles.append(
            EvidenceBundle(
                group_id=group_id,
                sample_index=index,
                pin=pin,
                episode=_episode(rollout_id, segments, team_id="home", seed=index),
                reward=_reward(
                    rollout_id,
                    reward,
                    channel_id="team_rank",
                    team_id="home",
                    extra=(
                        RewardChannel(channel_id="away_rank", team_id="away", measure=1.0 - reward),
                    ),
                ),
                topology=topology,
                source_run_id="run-1",
            )
        )
    return bundles


# --- Solo path ---------------------------------------------------------------


def test_a_solo_batch_carries_its_group_advantages_and_provenance() -> None:
    bundles = _solo_group("g1", [1.0, 0.0, 0.0, 0.0])
    batch = assemble(CISPO, bundles)

    assert batch.plan_hash == CISPO.plan_hash
    assert batch.off_policy is False
    assert [group.parameter_group_id for group in batch.parameter_groups] == [
        SOLO_PARAMETER_GROUP
    ]
    step = batch.steps[0]
    assert step.group_ids == ("g1",)
    assert len(step.items) == 4

    provenance = batch.provenance_for("g1")
    assert provenance.credit_kind == "length_weighted_leave_one_out"
    assert provenance.resolved_channel == "outcome"
    assert provenance.same_policy_reduction == "token_weighted_mean"
    assert provenance.rewards == (1.0, 0.0, 0.0, 0.0)
    assert provenance.lengths == (3, 3, 3, 3)
    assert provenance.skipped is False
    assert provenance.pin_digest == bundles[0].pin.pin_digest

    expected = estimate(
        CISPO.credit,
        [
            CreditSample(
                sample_key=f"g1-r{index}",
                reward=reward,
                length=3,
                reward_channel_id="outcome",
            )
            for index, reward in enumerate([1.0, 0.0, 0.0, 0.0])
        ],
    )
    advantages = {item.rollout_id: item.advantage for item in batch.items}
    assert provenance.advantages == expected.advantages
    assert advantages["g1-r0"] > 0
    assert all(value < 0 for key, value in advantages.items() if key != "g1-r0")
    assert provenance.advantages == tuple(
        advantages[rollout_id] for rollout_id in provenance.rollout_ids
    )


def test_every_record_is_validated_before_it_is_used() -> None:
    bundles = _solo_group("g1", [1.0, 0.0])
    broken = TrainableEpisode(
        rollout_id="g1-r0",
        task_id="task-1",
        seed=0,
        policy_revision=3,
        behavior_fingerprint=FINGERPRINT,
        segments=(_segment(),),
        terminal_status="completed",
        trace_digest="",
    )
    with pytest.raises(EvidenceError, match="sealed trace digest"):
        assemble(CISPO, [_replace_episode(bundles[0], broken), bundles[1]])

    mismatched = _reward("other-rollout", 1.0)
    with pytest.raises(EvidenceError, match="trace digest does not match"):
        assemble(CISPO, [_replace_reward(bundles[0], mismatched), bundles[1]])


def _replace_episode(bundle: EvidenceBundle, episode: TrainableEpisode) -> EvidenceBundle:
    return EvidenceBundle(
        group_id=bundle.group_id,
        sample_index=bundle.sample_index,
        pin=bundle.pin,
        episode=episode,
        reward=bundle.reward,
        topology=bundle.topology,
        staleness_steps=bundle.staleness_steps,
    )


def _replace_reward(bundle: EvidenceBundle, reward: RewardRecord) -> EvidenceBundle:
    return EvidenceBundle(
        group_id=bundle.group_id,
        sample_index=bundle.sample_index,
        pin=bundle.pin,
        episode=bundle.episode,
        reward=reward,
        topology=bundle.topology,
        staleness_steps=bundle.staleness_steps,
    )


def test_a_group_that_mixes_a_pinned_field_is_rejected() -> None:
    bundles = _solo_group("g1", [1.0, 0.0])
    tainted = EvidenceBundle(
        group_id="g1",
        sample_index=1,
        pin=_pin("g1", cardinality=2, model_family="family-b"),
        episode=bundles[1].episode,
        reward=bundles[1].reward,
    )
    with pytest.raises(MixedGroupError, match="model_family"):
        assemble(CISPO, [bundles[0], tainted])


def test_evidence_from_another_plan_hash_never_enters_this_batch() -> None:
    bundles = _solo_group("g1", [1.0, 0.0], plan_hash="a-different-plan-hash")
    with pytest.raises(AssemblyError, match="plan hash"):
        assemble(CISPO, bundles)


def test_sentinel_and_malformed_behavior_logprobs_are_refused() -> None:
    bundles = _solo_group("g1", [1.0, 0.0])
    sentinel = _segment(logprobs=(-0.1, -0.2, -0.3, LOGPROB_SENTINEL))
    with pytest.raises(EvidenceError, match="sentinel"):
        assemble(
            CISPO,
            [_replace_episode(bundles[0], _episode("g1-r0", (sentinel,))), bundles[1]],
        )
    zeros = _segment(logprobs=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(EvidenceError, match="identically zero"):
        assemble(CISPO, [_replace_episode(bundles[0], _episode("g1-r0", (zeros,))), bundles[1]])


def test_staleness_beyond_the_plan_bound_is_refused_not_trained() -> None:
    bundles = _solo_group("g1", [1.0, 0.0])
    stale = EvidenceBundle(
        group_id="g1",
        sample_index=1,
        pin=bundles[1].pin,
        episode=bundles[1].episode,
        reward=bundles[1].reward,
        staleness_steps=2,
    )
    with pytest.raises(AssemblyError, match="revisions stale"):
        assemble(CISPO, [bundles[0], stale])


def test_a_declared_staleness_drop_discards_rather_than_refuses() -> None:
    plan = expand(
        {
            "preset": "cispo",
            "correction": {"kind": "staleness_drop", "enabled": True, "max_weight_staleness": 1},
            "schedule": {"weight_mode": "async_lag"},
        }
    )
    bundles = _solo_group("g1", [1.0, 0.0, 0.0], plan_hash=plan.plan_hash)
    stale = EvidenceBundle(
        group_id="g1",
        sample_index=2,
        pin=bundles[2].pin,
        episode=bundles[2].episode,
        reward=bundles[2].reward,
        staleness_steps=5,
    )
    batch = assemble(plan, [bundles[0], bundles[1], stale])
    assert [drop.reason for drop in batch.dropped_bundles] == ["staleness_bound"]
    assert {item.rollout_id for item in batch.items} == {"g1-r0", "g1-r1"}


def test_branches_of_one_attempt_do_not_multiply_its_weight() -> None:
    pin = _pin("g1", cardinality=2)
    forked = _episode(
        "g1-r0",
        (
            _segment(base=100, branch_id="root"),
            _segment(base=140, branch_id="branch-1"),
            _segment(base=180, branch_id="branch-2"),
        ),
    )
    plain = _episode("g1-r1", (_segment(base=300),))
    batch = assemble(
        CISPO,
        [
            EvidenceBundle(
                group_id="g1",
                sample_index=0,
                pin=pin,
                episode=forked,
                reward=_reward("g1-r0", 1.0),
            ),
            EvidenceBundle(
                group_id="g1",
                sample_index=1,
                pin=pin,
                episode=plain,
                reward=_reward("g1-r1", 0.0),
            ),
        ],
    )
    weights = {
        (item.rollout_id, item.branch_id): item.root_rollout_weight for item in batch.items
    }
    assert weights[("g1-r0", "root")] == pytest.approx(1 / 3)
    assert weights[("g1-r0", "branch-1")] == pytest.approx(1 / 3)
    assert weights[("g1-r1", "root")] == pytest.approx(1.0)


def test_spans_with_no_trainable_token_are_dropped_with_a_reason() -> None:
    pin = _pin("g1", cardinality=2)
    episode = _episode(
        "g1-r0",
        (_segment(base=100), _segment(base=200, tokens=3, trainable=0)),
    )
    batch = assemble(
        CISPO,
        [
            EvidenceBundle(
                group_id="g1",
                sample_index=0,
                pin=pin,
                episode=episode,
                reward=_reward("g1-r0", 1.0),
            ),
            EvidenceBundle(
                group_id="g1",
                sample_index=1,
                pin=pin,
                episode=_episode("g1-r1", (_segment(base=300),)),
                reward=_reward("g1-r1", 0.0),
            ),
        ],
    )
    assert [drop.reason for drop in batch.dropped_spans] == [DROP_NO_TRAINABLE_TOKENS]
    assert len(batch.items) == 2


# --- Joint episodes ----------------------------------------------------------


def test_a_team_advantage_fans_out_and_each_batch_holds_only_its_own_spans() -> None:
    batch = assemble(CISPO, _joint_group("gj", [1.0, 0.0]))

    assert [group.parameter_group_id for group in batch.parameter_groups] == [
        "pg_alpha",
        "pg_beta",
    ]
    provenance = batch.provenance_for("gj")
    assert provenance.topology_id == "topology-1"
    assert provenance.resolved_channel == "team_rank"
    assert provenance.fanout_parameter_groups == ("pg_alpha", "pg_beta")

    alpha = batch.batch_for("pg_alpha")
    beta = batch.batch_for("pg_beta")
    assert {item.agent_instance_id for item in alpha.items} == {"instance_a"}
    assert {item.agent_instance_id for item in beta.items} == {"instance_b"}
    alpha_tokens = {token for item in alpha.items for token in item.token_ids}
    beta_tokens = {token for item in beta.items for token in item.token_ids}
    assert not alpha_tokens & beta_tokens

    # One group-relative team advantage per episode, identical in both batches.
    by_rollout_alpha = {item.rollout_id: item.advantage for item in alpha.items}
    by_rollout_beta = {item.rollout_id: item.advantage for item in beta.items}
    assert by_rollout_alpha == by_rollout_beta
    assert by_rollout_alpha["gj-r0"] > 0 > by_rollout_alpha["gj-r1"]


def test_foreign_authored_spans_are_dropped_with_their_author_named() -> None:
    batch = assemble(CISPO, _joint_group("gj", [1.0, 0.0]))
    foreign = [drop for drop in batch.dropped_spans if drop.reason == DROP_FOREIGN_AUTHOR]
    assert {drop.agent_instance_id for drop in foreign} == {"instance_x"}
    assert len(foreign) == 2
    assert all(item.agent_instance_id != "instance_x" for item in batch.items)


def test_a_joint_span_with_no_author_is_never_implied_into_a_batch() -> None:
    bundles = _joint_group("gj", [1.0, 0.0])
    unattributed = _episode(
        "gj-r0",
        (
            _segment(base=200, agent_instance_id="instance_a"),
            _segment(base=250),
        ),
        team_id="home",
    )
    batch = assemble(CISPO, [_replace_episode(bundles[0], unattributed), bundles[1]])
    assert DROP_UNATTRIBUTED_AUTHOR in {drop.reason for drop in batch.dropped_spans}


def test_a_span_contradicting_the_declared_parameter_group_fails_the_batch() -> None:
    bundles = _joint_group("gj", [1.0, 0.0])
    lying = _episode(
        "gj-r0",
        (
            _segment(
                base=200, agent_instance_id="instance_a", parameter_group_id="pg_beta"
            ),
        ),
        team_id="home",
    )
    with pytest.raises(AssemblyError, match="topology declares"):
        assemble(CISPO, [_replace_episode(bundles[0], lying), bundles[1]])


def test_the_same_policy_reduction_is_applied_and_receipted_in_the_batch() -> None:
    """Two instances share one parameter group; one emits 25x the tokens."""

    def group(plan: Any) -> list[EvidenceBundle]:
        pin = _pin(
            "gs",
            plan_hash=plan.plan_hash,
            cardinality=2,
            topology_id=SHARED_GROUP_TOPOLOGY.topology_id,
        )
        bundles = []
        for index, reward in enumerate([1.0, 0.0]):
            rollout_id = f"gs-r{index}"
            segments = (
                _segment(
                    base=1000 + 200 * index,
                    tokens=5,
                    trainable=4,
                    agent_instance_id="instance_quiet",
                ),
                _segment(
                    base=1100 + 200 * index,
                    tokens=101,
                    trainable=100,
                    agent_instance_id="instance_chatty",
                ),
            )
            bundles.append(
                EvidenceBundle(
                    group_id="gs",
                    sample_index=index,
                    pin=pin,
                    episode=_episode(rollout_id, segments, team_id="home", seed=index),
                    reward=_reward(rollout_id, reward, team_id="home"),
                    topology=SHARED_GROUP_TOPOLOGY,
                )
            )
        return bundles

    naive_plan = expand({"preset": "cispo", "credit": {"same_policy_reduction": "none"}})
    naive = assemble(naive_plan, group(naive_plan)).batch_for("pg_alpha")
    naive_shares = naive.same_policy.applied_shares
    assert naive_shares["instance_chatty"] / naive_shares["instance_quiet"] == pytest.approx(
        25.0
    )

    reduced = assemble(CISPO, group(CISPO)).batch_for("pg_alpha")
    shares = reduced.same_policy.applied_shares
    assert shares == pytest.approx({"instance_chatty": 0.5, "instance_quiet": 0.5})
    assert reduced.same_policy.naive_shares == pytest.approx(naive_shares)
    per_instance = {
        item.agent_instance_id: item.same_policy_weight for item in reduced.items
    }
    assert per_instance["instance_quiet"] == pytest.approx(0.25)
    assert per_instance["instance_chatty"] == pytest.approx(0.25)
    receipt = reduced.same_policy.receipt()
    assert receipt["same_policy_reduction"] == "token_weighted_mean"
    assert receipt["naive_shares"] != receipt["applied_shares"]


# --- Zero advantage, packing, and a second preset ---------------------------


def test_a_zero_advantage_group_is_skipped_and_still_receipted() -> None:
    bundles = _solo_group("g_live", [1.0, 0.0]) + _solo_group("g_tied", [0.5, 0.5])
    batch = assemble(CISPO, bundles)
    tied = batch.provenance_for("g_tied")
    assert tied.zero_variance is True
    assert tied.skipped is True
    live = batch.provenance_for("g_live")
    assert (live.zero_variance, live.skipped) == (False, False)
    assert {item.group_id for item in batch.items} == {"g_live"}
    assert batch.steps[0].group_ids == ("g_live",)


def test_every_group_skipped_leaves_nothing_to_train_on() -> None:
    with pytest.raises(AssemblyError, match="nothing to train on"):
        assemble(CISPO, _solo_group("g_tied", [0.5, 0.5]))


def test_packing_puts_several_groups_in_one_provider_step() -> None:
    bundles: list[EvidenceBundle] = []
    for index in range(7):
        bundles.extend(_solo_group(f"g{index}", [1.0, 0.0]))
    batch = assemble(CISPO, bundles)
    steps = batch.batch_for(SOLO_PARAMETER_GROUP).steps
    assert [len(step.group_ids) for step in steps] == [3, 3, 1]
    assert steps[0].group_ids == ("g0", "g1", "g2")
    assert [step.step_index for step in steps] == [0, 1, 2]
    # Per-group advantages survive packing.
    for step in steps:
        for group_id in step.group_ids:
            assert batch.provenance_for(group_id).advantages[0] > 0


def test_the_step_ceiling_is_a_plan_field_and_is_enforced() -> None:
    plan = expand(
        {"preset": "cispo", "schedule": {"groups_per_step": 1, "max_steps_per_round": 2}}
    )
    bundles: list[EvidenceBundle] = []
    for index in range(3):
        bundles.extend(_solo_group(f"g{index}", [1.0, 0.0], plan_hash=plan.plan_hash))
    with pytest.raises(AssemblyError, match="allows 2"):
        assemble(plan, bundles)


def test_a_second_preset_reaches_a_batch_by_configuration_alone() -> None:
    """No assembly change: only the plan and the pins differ."""

    gspo = PRESETS["gspo"]
    bundles = _solo_group("g1", [1.0, 0.5, 0.0, 0.0], plan_hash=gspo.plan_hash)
    batch = assemble(gspo, bundles)
    provenance = batch.provenance_for("g1")
    assert provenance.credit_kind == "group_mean"
    assert provenance.plan_hash == gspo.plan_hash != CISPO.plan_hash
    assert len(batch.items) == 4
    assert provenance.advantages == pytest.approx((0.625, 0.125, -0.375, -0.375))


def test_a_joint_second_preset_also_reaches_a_batch_by_configuration_alone() -> None:
    plan = expand({"preset": "gspo", "credit": {"same_policy_reduction": "episode_uniform"}})
    batch = assemble(plan, _joint_group("gj", [1.0, 0.0], plan=plan))
    assert {group.parameter_group_id for group in batch.parameter_groups} == {
        "pg_alpha",
        "pg_beta",
    }
    assert batch.provenance_for("gj").same_policy_reduction == "episode_uniform"


def test_an_empty_batch_and_a_channel_the_receipt_lacks_both_raise() -> None:
    with pytest.raises(AssemblyError, match="at least one scored attempt"):
        assemble(CISPO, [])
    bundles = _solo_group("g1", [1.0, 0.0])
    ambiguous = RewardRecord(
        reward_id="reward-g1-r0",
        rollout_id="g1-r0",
        trace_digest="trace-g1-r0",
        channels=(
            RewardChannel(channel_id="team_rank", team_id="home", measure=1.0),
            RewardChannel(channel_id="team_margin", team_id="home", measure=2.0),
        ),
        optimized_channel="team_rank",
        terminal_status="completed",
        evaluation_plan_id="evaluation-plan-1",
    )
    joint = _episode("g1-r0", (_segment(base=100),), team_id="other")
    with pytest.raises(AssemblyError, match="must be unambiguous"):
        assemble(
            CISPO,
            [
                _replace_reward(_replace_episode(bundles[0], joint), ambiguous),
                bundles[1],
            ],
        )


def test_a_single_team_run_with_one_untargeted_measure_assembles() -> None:
    """The common case: one team stamped everywhere, one measure reported.

    A team is a comparison key only where the reward separates teams. Refusing
    an untargeted channel because a trajectory carries a team id would make
    every cooperative single-team run unassemblable.
    """

    from synth_optimizers.contracts.rl_records import RewardChannel, RewardRecord
    from synth_optimizers.rl.assembly import _resolve_channel

    untargeted = RewardRecord(
        reward_id="reward-1",
        rollout_id="rollout-1",
        trace_digest="sha256:trace",
        channels=(RewardChannel("reward", None, 1.0),),
        optimized_channel="reward",
        terminal_status="completed",
        evaluation_plan_id="plan-1",
    )
    assert _resolve_channel(untargeted, None).measure == 1.0
    assert _resolve_channel(untargeted, "team-1").measure == 1.0
    assert _resolve_channel(untargeted, "any-other-team").measure == 1.0


def test_a_competitive_reward_still_resolves_per_team() -> None:
    from synth_optimizers.contracts.rl_records import RewardChannel, RewardRecord
    from synth_optimizers.rl.assembly import AssemblyError, _resolve_channel

    ranked = RewardRecord(
        reward_id="reward-2",
        rollout_id="rollout-2",
        trace_digest="sha256:trace",
        channels=(
            RewardChannel("team_rank", "terra", 14.0, rank=1),
            RewardChannel("rival_rank", "gemini37", 13.0, rank=2),
        ),
        optimized_channel="team_rank",
        terminal_status="completed",
        evaluation_plan_id="plan-1",
    )
    assert _resolve_channel(ranked, "terra").measure == 14.0
    assert _resolve_channel(ranked, "gemini37").measure == 13.0
    with pytest.raises(AssemblyError, match="channels for team"):
        _resolve_channel(ranked, "grok46")
