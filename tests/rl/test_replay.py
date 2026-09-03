"""Replay mode: off-policy by construction, bit-for-bit against a stored run."""

from __future__ import annotations

from typing import Any

import pytest

from synth_optimizers.contracts.rl_identity import GroupPin
from synth_optimizers.contracts.rl_records import (
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
)
from synth_optimizers.rl.assembly import EvidenceBundle, assemble
from synth_optimizers.rl.plan import PRESETS, expand
from synth_optimizers.rl.replay import (
    ReplayError,
    ReplayPublishError,
    ReplaySource,
    assert_reproduces,
    compare,
    guard_publication,
    replay,
    replayed_from,
)

CISPO = PRESETS["cispo"]
FINGERPRINT = "behavior-fingerprint-1"


def _pin(group_id: str, *, plan_hash: str, cardinality: int) -> GroupPin:
    return GroupPin(
        group_id=group_id,
        run_id="run-1",
        algorithm_plan_hash=plan_hash,
        behavior_fingerprint=FINGERPRINT,
        policy_revision=3,
        wire_api="chat_completions",
        sampling_transport="message_in_capture_out",
        policy_kind="adapter",
        model_family="family-a",
        container_image_digest="sha256:image",
        container_contract_hash="contract-1",
        handshake_agreement_digest="agreement-1",
        task_family="family-x",
        cardinality=cardinality,
    )


def _segment(base: int, *, tokens: int = 5) -> TrainableSegment:
    return TrainableSegment(
        token_ids=tuple(range(base, base + tokens)),
        loss_mask=(0,) + (1,) * (tokens - 1),
        behavior_logprobs=tuple(-0.1 * (index + 1) for index in range(tokens)),
        call_ids=(f"call-{base}",),
    )


def _bundle(
    group_id: str, index: int, reward: float, *, plan: Any, cardinality: int, staleness: int = 0
) -> EvidenceBundle:
    rollout_id = f"{group_id}-r{index}"
    return EvidenceBundle(
        group_id=group_id,
        sample_index=index,
        pin=_pin(group_id, plan_hash=plan.plan_hash, cardinality=cardinality),
        episode=TrainableEpisode(
            rollout_id=rollout_id,
            task_id="task-1",
            seed=index,
            policy_revision=3,
            behavior_fingerprint=FINGERPRINT,
            segments=(_segment(100 + 20 * index), _segment(500 + 20 * index, tokens=3)),
            terminal_status="completed",
            trace_digest=f"trace-{rollout_id}",
        ),
        reward=RewardRecord(
            reward_id=f"reward-{rollout_id}",
            rollout_id=rollout_id,
            trace_digest=f"trace-{rollout_id}",
            channels=(RewardChannel(channel_id="outcome", team_id=None, measure=reward),),
            optimized_channel="outcome",
            terminal_status="completed",
            evaluation_plan_id="evaluation-plan-1",
        ),
        staleness_steps=staleness,
        source_run_id="run-1",
    )


REWARDS = {
    "g0": [1.0, 0.0, 0.25, 0.0],
    "g1": [0.5, 0.75, 0.0, 1.0],
    "g2": [0.0, 0.0, 1.0, 0.0],
    "g3": [0.2, 0.4, 0.6, 0.8],
}


def _stored(plan: Any = CISPO) -> list[EvidenceBundle]:
    bundles: list[EvidenceBundle] = []
    for group_id, rewards in REWARDS.items():
        for index, reward in enumerate(rewards):
            bundles.append(
                _bundle(group_id, index, reward, plan=plan, cardinality=len(rewards))
            )
    return bundles


def test_replay_reproduces_an_online_run_bit_for_bit() -> None:
    bundles = _stored()
    online = assemble(CISPO, bundles, round_index=4)
    replayed = replay(
        CISPO, ReplaySource(run_ids=("run-1",), bundles=tuple(bundles)), round_index=4
    )

    diff = compare(online, replayed)
    assert diff.identical is True
    assert diff.differences == ()
    assert diff.advantage_digest_online == diff.advantage_digest_replayed
    assert diff.composition_digest_online == diff.composition_digest_replayed
    assert assert_reproduces(online, replayed) is not None
    assert diff.to_dict()["identical"] is True


def test_replay_is_order_independent_so_the_gate_is_not_a_coincidence() -> None:
    bundles = _stored()
    online = assemble(CISPO, bundles)
    shuffled = list(reversed(bundles))
    replayed = replay(CISPO, ReplaySource(run_ids=("run-1",), bundles=tuple(shuffled)))
    assert compare(online, replayed).identical is True


def test_replay_carries_its_source_runs_and_the_staleness_it_accepted() -> None:
    plan = expand(
        {
            "preset": "cispo",
            "correction": {"kind": "staleness_drop", "enabled": True, "max_weight_staleness": 6},
            "schedule": {"weight_mode": "async_lag"},
        }
    )
    bundles = [
        _bundle("g0", 0, 1.0, plan=plan, cardinality=2, staleness=0),
        _bundle("g0", 1, 0.0, plan=plan, cardinality=2, staleness=5),
    ]
    batch = replay(plan, ReplaySource(run_ids=("run-a", "run-b"), bundles=tuple(bundles)))
    assert batch.off_policy is True
    assert batch.source_run_ids == ("run-a", "run-b")
    assert batch.accepted_staleness == 5


def test_a_replay_derived_batch_refuses_to_publish_as_on_policy() -> None:
    bundles = _stored()
    batch = replay(CISPO, ReplaySource(run_ids=("run-1",), bundles=tuple(bundles)))
    with pytest.raises(ReplayPublishError, match="off-policy"):
        guard_publication(batch, presented_as="on_policy")

    attestation = guard_publication(batch, presented_as="off_policy")
    assert attestation.off_policy is True
    assert attestation.presented_as == "off_policy"
    assert attestation.source_run_ids == ("run-1",)
    assert attestation.plan_hash == CISPO.plan_hash
    assert attestation.to_dict()["advantage_digest"] == batch.advantage_digest


def test_an_online_batch_may_still_publish_on_policy() -> None:
    online = assemble(CISPO, _stored())
    attestation = guard_publication(online, presented_as="on_policy")
    assert attestation.off_policy is False
    assert attestation.presented_as == "on_policy"


def test_an_unknown_presentation_is_rejected() -> None:
    online = assemble(CISPO, _stored())
    with pytest.raises(ReplayError, match="unknown presentation"):
        guard_publication(online, presented_as="probably_on_policy")


def test_a_changed_batch_composition_shows_up_as_a_structured_diff() -> None:
    bundles = _stored()
    online = assemble(CISPO, bundles)
    partial = [bundle for bundle in bundles if bundle.group_id != "g3"]
    diff = replayed_from(CISPO, online, partial, ("run-1",))

    assert diff.identical is False
    assert diff.plan_hash_matches is True
    assert diff.advantages_match is False
    assert diff.composition_matches is False
    assert any("length 4 online vs 3" in line for line in diff.advantage_differences)
    assert any("length 2 online vs 1" in line for line in diff.composition_differences)
    with pytest.raises(ReplayError, match="did not reproduce"):
        assert_reproduces(online, replay(CISPO, ReplaySource(("run-1",), tuple(partial))))


def test_a_changed_stored_reward_shows_up_at_its_exact_path() -> None:
    bundles = _stored()
    online = assemble(CISPO, bundles)
    tampered = [
        bundle
        if bundle.group_id != "g3" or bundle.sample_index != 0
        else _bundle("g3", 0, 0.9, plan=CISPO, cardinality=4)
        for bundle in bundles
    ]
    diff = replayed_from(CISPO, online, tampered, ("run-1",))

    assert diff.identical is False
    assert diff.plan_hash_matches is True
    assert any(
        line.startswith("advantages[3].rewards[0]") for line in diff.advantage_differences
    )
    assert any(
        line.startswith("advantages[3].advantages[") for line in diff.advantage_differences
    )
    assert any("advantage" in line for line in diff.composition_differences)


def test_a_changed_credit_dimension_shows_up_as_a_plan_hash_and_advantage_diff() -> None:
    climb = PRESETS["cispo_climb"]
    online = assemble(CISPO, _stored())
    replayed = replay(climb, ReplaySource(run_ids=("run-1",), bundles=tuple(_stored(climb))))

    diff = compare(online, replayed)
    assert diff.identical is False
    assert diff.plan_hash_matches is False
    assert diff.advantages_match is False
    assert any("plan_hash" in line for line in diff.differences)
    assert any("credit_kind" in line for line in diff.advantage_differences)
    assert diff.to_dict()["plan_hash"] == [CISPO.plan_hash, climb.plan_hash]


def test_replay_must_name_its_stored_evidence() -> None:
    bundles = _stored()
    with pytest.raises(ReplayError, match="must name its source runs"):
        ReplaySource(run_ids=(), bundles=tuple(bundles))
    with pytest.raises(ReplayError, match="carries no stored episodes"):
        ReplaySource(run_ids=("run-1",), bundles=())


def test_replay_reproduces_packing_as_well_as_advantages() -> None:
    bundles = _stored()
    online = assemble(CISPO, bundles)
    replayed = replay(CISPO, ReplaySource(run_ids=("run-1",), bundles=tuple(bundles)))
    assert [len(step.group_ids) for step in online.steps] == [3, 1]
    assert [step.group_ids for step in replayed.steps] == [
        step.group_ids for step in online.steps
    ]
    assert compare(online, replayed).composition_matches is True
