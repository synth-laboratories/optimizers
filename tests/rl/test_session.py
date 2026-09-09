"""The admitted container session, driven against the conformance fakes.

Every test here runs the ordered startup for real: health, metadata,
capabilities and their hash, taskset rows, handshake, renderer equality, probe.
Nothing sleeps, nothing reaches a network beyond loopback, and nothing spends.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fakes import scenarios
from plane_harness import build_plane, config_text
from synth_optimizers.contracts.rl_records import SamplingProfile
from synth_optimizers.rl import config as config_module
from synth_optimizers.rl.capabilities import PreflightRejected
from synth_optimizers.rl.handshake import ClauseRejected
from synth_optimizers.rl.probe import ProbeError
from synth_optimizers.rl import session as session_module
from synth_optimizers.rl.session import LiveRunClock, RunClock, SessionError, start_session


def test_live_clock_advances_while_explicit_run_clock_remains_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter((102.5, 104.0))
    monkeypatch.setattr(session_module.time, "monotonic", lambda: next(readings))
    live = LiveRunClock(
        _monotonic_origin=100.0,
        _utc_origin=datetime(2026, 9, 4, tzinfo=UTC),
    )

    assert live.now() == pytest.approx(2.5)
    assert live.utc() == datetime(2026, 9, 4, 0, 0, 4, tzinfo=UTC)

    deterministic = RunClock()
    assert deterministic.now() == 0.0
    deterministic.advance(3.0)
    assert deterministic.now() == 3.0

def _sampling(config) -> SamplingProfile:
    return SamplingProfile(temperature=1.0, top_p=1.0, seed=config.seed)


def test_probe_accepts_unchanged_event_snapshots(tmp_path, monkeypatch):
    original = session_module.ContractContainerSession.poll
    polls = 0
    def delayed(self, rollout_id):
        nonlocal polls
        result = original(self, rollout_id)
        polls += 1
        if polls <= 2:
            return {**result, 'state':'running', 'terminal':False}
        return result
    monkeypatch.setattr(session_module.ContractContainerSession, 'poll', delayed)
    with build_plane(scenarios.multi_turn_environment_reward(), tmp_path) as plane:
        session = open_session(plane)
        assert session.startup.probe is not None
        assert polls >= 3


def test_probe_waits_for_deferred_verifier_receipt(tmp_path, monkeypatch):
    original = session_module.ContractContainerSession.reward_payload
    calls = 0
    def deferred(self, rollout_id):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return {'scoring_state':'awaiting_score', 'reward':None, 'deferred_scoring':True}
        return original(self, rollout_id)
    monkeypatch.setattr(session_module.ContractContainerSession, 'reward_payload', deferred)
    with build_plane(scenarios.multi_turn_environment_reward(), tmp_path) as plane:
        assert open_session(plane).startup.probe is not None
    assert calls >= 4


def open_session(plane, *, run_config=None, renderer_profile=None, **config_kwargs):
    """Run the ordered startup against a wired plane."""

    config = run_config or config_module.loads(
        config_text(plane.container.config, plane.container.base_url, **config_kwargs)
    )
    return start_session(
        plane.client,
        config,
        renderer_profile=renderer_profile or plane.gateway.renderer_profile,
        clock=plane.clock.run,
        sampling=_sampling(plane.container.config),
    )


# --------------------------------------------------------------------------- #
# The ordered startup
# --------------------------------------------------------------------------- #


def test_startup_runs_the_notes_order_and_stops_at_the_probe(tmp_path) -> None:
    with build_plane(scenarios.multi_turn_environment_reward(), tmp_path) as plane:
        session = open_session(plane)

        # 1..7, in order, and the probe is last.
        ordered = [name for name in plane.client.calls if name in {
            "health", "metadata", "capabilities", "taskset_tasks", "handshake"
        }]
        assert ordered[:3] == ["health", "metadata", "capabilities"]
        assert ordered.index("taskset_tasks") < ordered.index("handshake")

        assert session.handshake_id.startswith("hs_")
        assert session.agreement_digest.startswith("sha256:")
        assert session.agreement.capability_hash == session.capability.content_hash

        report = session.startup.probe
        assert report is not None
        assert report.trainable is False
        assert set(report.operations) >= {
            "submit",
            "state",
            "events",
            "renew",
            "trace",
            "reward",
            "finalize",
            "terminate",
            "idempotent_resubmit",
            "cancellation",
        }


def test_the_receipt_carries_the_handshake_pair_and_the_probe(tmp_path) -> None:
    with build_plane(scenarios.multi_turn_environment_reward(), tmp_path) as plane:
        session = open_session(plane)
        receipt = session.receipt()

        exchange = receipt["handshake"]["exchanges"][-1]
        assert exchange["request"]["schema_version"] == "cispo.handshake.v1"
        assert exchange["verdict"]["accepted"] is True
        assert exchange["outcome"] == "admissible"
        agreement = receipt["handshake"]["agreement"]
        assert agreement["handshake_id"] == session.handshake_id
        assert agreement["agreement_digest"] == session.agreement_digest
        assert agreement["obligations"]["max_concurrency"] >= 1
        assert agreement["taskset_resolution"]
        assert agreement["expires_at"]
        assert receipt["capability_hash"].startswith("sha256:")
        assert receipt["probe"]["cost"] == 0.0
        assert receipt["probe"]["cost_attribution"] == "handshake_overhead"


def test_every_attempt_carries_the_handshake_and_the_agreement(tmp_path) -> None:
    with build_plane(scenarios.multi_turn_environment_reward(), tmp_path) as plane:
        session = open_session(plane)
        submissions = list(session.submissions.values())

        assert submissions, "the probe submitted at least one attempt"
        for submitted in submissions:
            assert submitted.accepted["handshake_id"] == session.handshake_id
            fields = submitted.group_pin_fields
            assert fields["handshake_agreement_digest"] == session.agreement_digest


# --------------------------------------------------------------------------- #
# What stops a run before it spends
# --------------------------------------------------------------------------- #


def test_a_rejected_mandatory_clause_stops_before_any_session(tmp_path) -> None:
    with build_plane(scenarios.rejected_mandatory_clause(), tmp_path) as plane:
        with pytest.raises(ClauseRejected) as raised:
            open_session(plane)

        assert "evidence.behavior_logprobs" in raised.value.clause_ids
        # Nothing was bound and nothing was trained: the binder never ran.
        assert plane.binder.train_calls == []
        assert plane.binder.published == []
        assert plane.gateway.bindings == []


def test_a_renderer_mismatch_is_a_rejected_clause(tmp_path) -> None:
    config = scenarios.multi_turn_environment_reward()
    with build_plane(config, tmp_path) as plane:
        other = replace(config.renderer_profile, config_digest="sha256:cfg-b")

        with pytest.raises((ClauseRejected, PreflightRejected)) as raised:
            open_session(plane, renderer_profile=other)

        assert "policy.renderer_profile_match" in str(raised.value)
        assert plane.binder.train_calls == []


def test_an_absent_reward_is_a_failure_not_a_zero(tmp_path) -> None:
    # The unpaid probe is where this is caught: an absent reward stops the run
    # before a paid attempt exists, rather than scoring the attempt zero.
    config = replace(scenarios.absent_reward(), turns=2)
    with build_plane(config, tmp_path) as plane:
        with pytest.raises((SessionError, ProbeError)) as raised:
            open_session(plane)

        assert "reward" in str(raised.value)
        assert plane.binder.train_calls == []


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def test_tasks_carry_the_digest_the_agreement_resolved(tmp_path) -> None:
    config = scenarios.multi_turn_environment_reward()
    with build_plane(config, tmp_path) as plane:
        session = open_session(plane, task_ids=config.task_ids[:2])
        tasks = session.tasks(split="train", task_ids=config.task_ids[:2])

        assert [task.task_id for task in tasks] == list(config.task_ids[:2])
        for task in tasks:
            assert task.content_digest == session.agreement.task_digest(task.task_id)
            assert task.task_family == config.task_family

        with pytest.raises(SessionError, match="not a row"):
            session.tasks(split="train", task_ids=("row_nowhere",))


def test_evidence_returns_a_validated_episode_and_reward(tmp_path) -> None:
    config = scenarios.multi_turn_environment_reward()
    with build_plane(config, tmp_path) as plane:
        session = open_session(plane)
        rollout_id = _drive_one(session, plane)

        episode, reward = session.evidence(rollout_id)

        assert episode.rollout_id == rollout_id
        assert episode.probe is False
        assert episode.segments
        assert reward.rollout_id == rollout_id
        assert reward.trace_digest == episode.trace_digest
        assert reward.value() == pytest.approx(0.25)


def test_a_joint_episode_merges_into_one_episode_over_two_groups(tmp_path) -> None:
    config = scenarios.competitive_realtime()
    with build_plane(config, tmp_path) as plane:
        session = open_session(plane)
        rollout_id = _drive_one(session, plane)

        episode, reward = session.evidence(rollout_id)

        assert episode.agent_instance_id is None
        assert episode.team_id == "team_home"
        instances = {segment.agent_instance_id for segment in episode.segments}
        assert instances == {"home_1", "home_2"}
        assert set(episode.parameter_groups) == {"pg_alpha", "pg_beta"}
        assert reward.optimized_channel == "score::team_home"


def _drive_one(session, plane) -> str:
    """Submit one real attempt through the session and settle it."""

    from synth_optimizers.contracts.rl_identity import GroupPin
    from synth_optimizers.rl.ports import AttemptFacts, PolicyRevision

    config = plane.container.config
    revision = PolicyRevision(
        revision=0,
        revision_id="rev::pg::0",
        checkpoint_id="ckpt::pg::0",
        parameter_group_id=next(
            iter(session.topology.trainable_parameter_groups() or ("pg_solo",))
        ),
        sampler_reference="weights://ckpt",
        behavior_fingerprint=plane.binder.fingerprints(0),
    )
    task = session.tasks(split="train", task_ids=())[0]
    pin = GroupPin(
        group_id="group_probe_free",
        run_id="run_test",
        algorithm_plan_hash="plan::test",
        behavior_fingerprint=revision.behavior_fingerprint,
        policy_revision=0,
        wire_api=config.wire_api,
        sampling_transport=config.sampling_transport,
        policy_kind=config.policy_kind,
        model_family=config.model_family,
        container_image_digest=session.capability.container_image_digest,
        container_contract_hash=session.startup.contract.contract_hash,
        handshake_agreement_digest=session.agreement_digest,
        task_family=task.task_family,
        cardinality=1,
        topology_id=session.topology.topology_id,
    )
    origins = {
        revision.parameter_group_id: plane.gateway.bind(
            revision,
            pin=pin,
            sample_index=0,
            proxy_request_id="prid::one",
            attempt=AttemptFacts(rollout_id="pending", task_id=task.task_id, seed=task.seed),
        )
    }
    rollout_id = session.submit_roster(
        task, origins, pin=pin, sample_index=0, idempotency_key="key::one"
    )
    session.poll(rollout_id)
    session.finalize(rollout_id)
    return rollout_id
