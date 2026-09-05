"""The loop, end to end, against the in-process conformance fakes.

Every test below runs the whole plane: ordered startup, baseline registration
through the binder, per-sample admission under a group pin, the queue engine,
the dequeue gate, batch assembly, a real train call, an atomic publish and the
receipt directory. There is no network beyond loopback, no container runtime,
no provider and no spend, and the clock is injected so nothing sleeps.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import scenarios
from fakes.container import ContainerConfig
from plane_harness import build_plane, config_text
from synth_optimizers.contracts.rl_records import SamplingProfile
from synth_optimizers.rl import config as config_module
from synth_optimizers.rl.executor import RUN_ARTIFACTS, ContainerRunExecutor
from synth_optimizers.rl.handshake import ClauseRejected
from synth_optimizers.rl.session import start_session


def _executor(plane, tmp_path: Path, **config_kwargs) -> ContainerRunExecutor:
    """Ordered startup, then a loop wired to the ports. No spend either side."""

    config = config_module.loads(
        config_text(plane.container.config, plane.container.base_url, **config_kwargs)
    )
    session = start_session(
        plane.client,
        config,
        renderer_profile=plane.gateway.renderer_profile,
        clock=plane.clock.run,
        sampling=SamplingProfile(
            temperature=1.0, top_p=1.0, seed=plane.container.config.seed
        ),
    )
    return ContainerRunExecutor(
        config=config,
        session=session,
        gateway=plane.gateway,
        binder=plane.binder,
        clock=plane.clock.run,
        receipts=tmp_path / "receipts",
        catalog_rows=plane.binder.catalog_rows,
        lineage_rows=plane.binder.lineage_rows,
    )


def _advance(plane):
    def _tick(_executor, _report) -> None:
        plane.clock.advance(1.0)

    return _tick


def test_bounded_on_policy_admission_counts_pending_credit_groups(tmp_path):
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        executor.config = replace(executor.config, pipeline=replace(executor.config.pipeline, bounded_on_policy_batch=True, max_open_groups=3))
        executor.register_baseline()
        # A dequeued mixed group no longer counts as OPEN, but must still stop
        # speculative old-policy admission when it fills the upcoming batch.
        executor._pending_groups = ['pending'] * executor.plan.groups_per_step
        assert executor.admit_group() is None
        executor._pending_groups.clear()
        assert executor.admit_group() is not None


def _solo(**overrides) -> ContainerConfig:
    return replace(scenarios.multi_turn_environment_reward(), **overrides)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# A complete run
# --------------------------------------------------------------------------- #


def test_a_small_run_reaches_a_train_call_and_publishes_a_revision(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)

        report = executor.run(max_ticks=20, on_tick=_advance(plane))

        assert report.stop_reason == "target_train_updates_reached"
        assert len(report.updates) == 1
        update = report.updates[0]
        assert update.parameter_groups == ("pg_primary",)
        assert update.steps == 1

        # One real train call, carrying the plan hash and both advantages.
        assert len(plane.binder.train_calls) == 1
        call = plane.binder.train_calls[0]
        assert call["plan_hash"] == report.plan_hash
        assert call["examples"] == 2
        assert sorted(call["advantages"]) == pytest.approx([-0.25, 0.25])

        # A new revision was published and is what later groups would bind.
        assert plane.binder.published == ["policy-set-run::1"]
        published = report.final_revisions["pg_primary"]
        assert published.revision == 1
        assert executor.current_revision == 1
        assert published.checkpoint_id != plane.binder.baselines["pg_primary"].checkpoint_id


def test_rollout_id_settlement_does_not_serialize_dispatch(tmp_path, monkeypatch) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        original = executor._declare
        settled = []

        def declare(origins, *, rollout_id, task):
            # Both asynchronous attempts must have been submitted before any
            # settlement is allowed to wait on a provider's route lock.
            assert len(executor._rollouts) >= 2
            settled.append(rollout_id)
            return original(origins, rollout_id=rollout_id, task=task)

        monkeypatch.setattr(executor, '_declare', declare)
        report = executor.run(max_ticks=20, on_tick=_advance(plane))
        assert report.stop_reason == 'target_train_updates_reached'
        assert len(settled) == 2
        assert not executor._pending_declarations


def test_group_samples_keep_the_declared_task_seed(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=4, target_train_updates=1)
        executor.register_baseline()

        group_id = executor.admit_group()

        assert group_id is not None
        attempts = executor.store.attempts_in_group(group_id)
        assert [row.sample_index for row in attempts] == [0, 1, 2, 3]
        assert {row.seed for row in attempts} == {executor.tasks[0].seed}
        assert len({row.idempotency_key for row in attempts}) == 4


def test_a_second_preset_runs_through_the_identical_code_path(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(
            plane, tmp_path, preset="cispo_climb", group_size=2, target_train_updates=1
        )

        report = executor.run(max_ticks=20, on_tick=_advance(plane))

        assert executor.plan.preset == "cispo_climb"
        assert report.stop_reason == "target_train_updates_reached"
        assert len(plane.binder.train_calls) == 1
        # Standardized credit, same loop, same seams: only the plan differs.
        assert plane.binder.train_calls[0]["plan_hash"] == executor.plan.plan_hash


def test_a_rejected_mandatory_clause_stops_before_any_session_or_spend(tmp_path) -> None:
    with build_plane(scenarios.rejected_mandatory_clause(), tmp_path) as plane:
        with pytest.raises(ClauseRejected) as raised:
            _executor(plane, tmp_path, group_size=2)

        assert "evidence.behavior_logprobs" in raised.value.clause_ids
        assert plane.binder.baselines == {}
        assert plane.binder.train_calls == []
        assert plane.binder.published == []
        assert plane.gateway.bindings == []
        assert not (tmp_path / "receipts").exists()


# --------------------------------------------------------------------------- #
# Group dispositions
# --------------------------------------------------------------------------- #


def test_a_zero_advantage_group_is_skipped_and_replaced(tmp_path) -> None:
    config = _solo()
    tied, varied = config.task_ids[0], config.task_ids[1]

    def reward_for(task_id: str, sample_index: int) -> float:
        # The first group's rows all tie, so that group carries no ordering.
        return 0.5 if task_id == tied else 0.25 * (sample_index + 1)

    with build_plane(config, tmp_path, reward_for=reward_for) as plane:
        executor = _executor(
            plane,
            tmp_path,
            group_size=2,
            target_train_updates=1,
            maximum_sampled_groups=3,
            task_ids=(tied, varied),
        )

        report = executor.run(max_ticks=20, on_tick=_advance(plane))

        skipped = [item for item in report.groups if item.disposition == "skipped"]
        trained = [item for item in report.groups if item.disposition == "trained"]
        assert len(skipped) == 1
        assert skipped[0].zero_variance is True
        assert skipped[0].reason == "zero_advantage_group"
        assert skipped[0].rewards == (0.5, 0.5)
        # The skipped group was replaced, inside the sampled-group bound.
        assert len(trained) == 1
        assert trained[0].group_id != skipped[0].group_id
        assert report.sampled_groups == 2 <= executor.config.maximum_sampled_groups
        assert len(plane.binder.train_calls) == 1


def test_the_sampled_group_bound_ends_a_run_that_never_finds_an_ordering(tmp_path) -> None:
    with build_plane(_solo(), tmp_path, reward_for=lambda _task, _index: 0.5) as plane:
        executor = _executor(
            plane,
            tmp_path,
            group_size=2,
            target_train_updates=1,
            maximum_sampled_groups=3,
        )

        report = executor.run(max_ticks=20, on_tick=_advance(plane))

        assert report.stop_reason == "sampled_group_budget_exhausted"
        assert report.sampled_groups == 3
        assert len(report.skipped_groups) == 3
        assert plane.binder.train_calls == []
        assert plane.binder.published == []


def test_multiple_groups_keep_their_admitted_revision_across_updates(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, groups_per_step=2,
                             slots=2, max_open_groups=2, target_train_updates=3,
                             maximum_sampled_groups=30)
        report = executor.run(max_ticks=100, on_tick=_advance(plane))
        assert report.stop_reason == 'target_train_updates_reached'
        assert len(report.updates) == 3
        for group_id, pin in executor._pins.items():
            revisions = executor._group_revisions[group_id]
            assert all(revision.revision == pin.policy_revision for revision in revisions.values())


def test_a_stale_group_is_discarded_at_the_dequeue_gate(tmp_path) -> None:
    config = _solo()
    with build_plane(config, tmp_path) as plane:
        executor = _executor(
            plane,
            tmp_path,
            group_size=2,
            slots=4,
            target_train_updates=2,
            maximum_sampled_groups=6,
            max_open_groups=2,
            train_ready_capacity=1,
            maximum_policy_lag=0,
            task_ids=config.task_ids[:2],
        )

        report = executor.run(max_ticks=30, on_tick=_advance(plane))

        stale = [item for item in report.groups if item.disposition == "stale"]
        assert stale, "a group completed under revision 0 and was gated after the update"
        assert stale[0].staleness == 1
        assert "dequeue gate" in stale[0].reason
        # Its slots came back: recycled returns sample index, task, seed and
        # key, never a new task identity. They are re-admitted under a fresh
        # pin as soon as there is room for another open group.
        returned = executor._recycled or [
            {"from_group": item.group_id, "slots": item.slots}
            for item in executor._pending_recycled
        ]
        assert returned and returned[0]["from_group"] == stale[0].group_id
        assert len(returned[0]["slots"]) == 2
        assert len(report.updates) == 2


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_pause_and_resume_are_honored_mid_run(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(
            plane,
            tmp_path,
            group_size=2,
            target_train_updates=2,
            maximum_sampled_groups=4,
            task_ids=plane.container.config.task_ids[:2],
        )
        executor.register_baseline()
        executor.tick()
        assert len(executor.updates) == 1

        executor.pause()
        assert executor.lifecycle.state == "paused"
        assert executor.lifecycle.gates.admit is False
        sampled = executor.sampled_groups
        paused = executor.tick()
        assert paused["admitted_groups"] == 0
        assert paused["dispatched"] == 0
        assert executor.sampled_groups == sampled
        assert len(executor.updates) == 1

        executor.resume()
        assert executor.lifecycle.state == "admitting"
        # Resume re-handshakes before a single attempt is re-admitted.
        assert executor._rehandshakes
        assert executor._rehandshakes[-1]["agreement_digest"] == executor.session.agreement_digest

        report = executor.run(max_ticks=20, on_tick=_advance(plane))
        assert len(report.updates) == 2
        controls = [row["control"] for row in _rows(report.receipt_directory / "lifecycle.jsonl")]
        assert "pause" in controls
        assert "resume" in controls
        assert "resume_rehandshake" in controls


def test_drain_finishes_its_work_and_stop_cancels_what_is_left(tmp_path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(
            plane,
            tmp_path,
            group_size=2,
            target_train_updates=2,
            maximum_sampled_groups=4,
            task_ids=plane.container.config.task_ids[:2],
        )
        executor.register_baseline()
        executor.tick()

        executor.drain()
        assert executor.lifecycle.state == "draining"
        assert executor.lifecycle.gates.admit is False
        executor.tick()
        assert not any(executor.lifecycle.outstanding_drain_work().values())

        report = executor.finish("drained")
        assert report.lifecycle_state == "drained"
        assert report.stop_reason == "drained"
        controls = [row["control"] for row in _rows(report.receipt_directory / "lifecycle.jsonl")]
        assert "drain" in controls

    with build_plane(_solo(), tmp_path / "stop") as plane:
        executor = _executor(
            plane, tmp_path / "stop", group_size=2, target_train_updates=1
        )
        executor.register_baseline()
        executor.admit_group()
        executor.dispatch_once()
        assert executor.queues.in_flight()

        executor.stop(reason="operator")
        assert executor.lifecycle.state == "stopped"
        assert not executor.store.attempts_without_result(run_id=executor.run_id)
        report = executor.finish("stopped")
        cleanup = json.loads((report.receipt_directory / "cleanup.json").read_text())
        assert cleanup["cancelled_attempts"]


# --------------------------------------------------------------------------- #
# Joint episodes
# --------------------------------------------------------------------------- #


def test_a_joint_episode_fans_one_advantage_into_two_groups(tmp_path) -> None:
    with build_plane(scenarios.competitive_realtime(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)

        report = executor.run(max_ticks=20, on_tick=_advance(plane))

        assert report.stop_reason == "target_train_updates_reached"
        update = report.updates[0]
        assert update.parameter_groups == ("pg_alpha", "pg_beta")

        alpha, beta = sorted(plane.binder.train_calls, key=lambda row: row["parameter_group_id"])
        assert alpha["parameter_group_id"] == "pg_alpha"
        assert beta["parameter_group_id"] == "pg_beta"
        # One team advantage, fanned out: the same numbers reach both groups.
        assert alpha["advantages"] == beta["advantages"]

        # Both components published, or neither: one policy-set revision.
        assert plane.binder.published == ["policy-set-run::1"]
        assert set(report.final_revisions) == {"pg_alpha", "pg_beta"}
        for revision in report.final_revisions.values():
            assert revision.policy_set_revision_id == "policy-set-run::1"
        members = plane.binder.catalog.policy_set_members("policy-set-run::1")
        assert len(members) == 2
        for checkpoint_id in members:
            assert plane.binder.catalog.publication_status(checkpoint_id) == "published"


def test_a_one_sided_save_leaves_the_prior_set_live(tmp_path) -> None:
    from synth_optimizers.rl.policy_sets import PartialPublicationError

    with build_plane(
        scenarios.competitive_realtime(), tmp_path, fail_groups=("pg_beta",)
    ) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)

        with pytest.raises(PartialPublicationError):
            executor.run(max_ticks=20, on_tick=_advance(plane))

        assert plane.binder.published == []
        assert executor.current_revision == 0


# --------------------------------------------------------------------------- #
# The receipt directory
# --------------------------------------------------------------------------- #


def test_provider_usage_preserves_known_zero_cost(tmp_path: Path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        executor.run(max_ticks=20, on_tick=_advance(plane))

        usage = executor._provider_usage()

        assert usage["train_calls"][0]["provider_cost"] == 0.0
        assert usage["train_calls"][0]["cost_missing"] is False
        assert usage["totals"]["provider_cost"] == 0.0
        assert usage["totals"]["cost_missing"] is False


def test_provider_usage_preserves_missing_cost_as_null(tmp_path: Path) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        executor.run(max_ticks=20, on_tick=_advance(plane))
        update = executor.updates[0]
        outcome = update.outcomes["pg_primary"]
        executor.updates[0] = replace(
            update,
            outcomes={
                "pg_primary": replace(
                    outcome,
                    provider_cost=0.0,
                    metrics={**outcome.metrics, "cost_missing": True},
                )
            },
        )

        usage = executor._provider_usage()

        assert usage["train_calls"][0]["provider_cost"] is None
        assert usage["train_calls"][0]["cost_missing"] is True
        assert usage["totals"]["provider_cost"] is None
        assert usage["totals"]["cost_missing"] is True


def test_provider_usage_mixed_known_and_missing_cost_has_unknown_total(
    tmp_path: Path,
) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        executor.run(max_ticks=20, on_tick=_advance(plane))
        update = executor.updates[0]
        original = update.outcomes["pg_primary"]
        known = replace(original, provider_cost=1.25, metrics={**original.metrics})
        missing = replace(
            original,
            provider_cost=0.0,
            metrics={**original.metrics, "cost_missing": True},
        )
        executor.updates[0] = replace(
            update, outcomes={"pg_known": known, "pg_missing": missing}
        )

        usage = executor._provider_usage()

        assert [row["provider_cost"] for row in usage["train_calls"]] == [1.25, None]
        assert [row["cost_missing"] for row in usage["train_calls"]] == [False, True]
        assert usage["totals"]["provider_cost"] is None
        assert usage["totals"]["cost_missing"] is True


def test_the_receipt_directory_carries_every_artifact_the_note_lists(tmp_path) -> None:
    with build_plane(scenarios.competitive_realtime(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        report = executor.run(max_ticks=20, on_tick=_advance(plane))
        directory = report.receipt_directory

        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["plan_hash"] == report.plan_hash
        assert manifest["artifacts"] == dict(RUN_ARTIFACTS)
        for bullet, name in RUN_ARTIFACTS.items():
            assert (directory / name).is_file(), f"{bullet} -> {name}"

        config = json.loads((directory / "effective_config.json").read_text())
        assert config["plan_hash"] == report.plan_hash
        assert config["expanded_plan"]["preset"] == "cispo"
        assert "<redacted>" not in json.dumps(config["config"]["container"]["headers"])

        handshake = json.loads((directory / "handshake.json").read_text())
        assert handshake["agreement"]["handshake_id"] == executor.session.handshake_id
        assert handshake["exchanges"][0]["request"]["requirements"]

        revisions = json.loads((directory / "policy_revisions.json").read_text())
        assert set(revisions["baseline"]) == {"pg_alpha", "pg_beta"}
        assert set(revisions["trained"]) == {"pg_alpha", "pg_beta"}
        for group, baseline in revisions["baseline"].items():
            assert baseline["revision"] == 0
            assert revisions["trained"][group]["revision"] == 1
            assert baseline["checkpoint_id"] != revisions["trained"][group]["checkpoint_id"]
            assert baseline["sampler_reference"]
        # Two parameter groups, one packed provider step each.
        assert revisions["updates"][0]["provider_steps"] == 2

        probe = json.loads((directory / "probe.json").read_text())
        assert probe["trainable"] is False
        assert probe["cost"] == 0.0

        pins = _rows(directory / "group_pins.jsonl")
        assert pins and pins[0]["pin"]["algorithm_plan_hash"] == report.plan_hash
        assert pins[0]["pin_digest"]

        groups = _rows(directory / "groups.jsonl")
        assert groups[0]["rewards"] and groups[0]["advantages"]
        assert groups[0]["staleness"] == 0
        assert groups[0]["skipped"] is False

        journal = _rows(directory / "queue_journal.jsonl")
        kinds = {row["kind"] for row in journal}
        assert {"run_registered", "attempt_admitted", "group_opened"} <= kinds

        usage = json.loads((directory / "provider_usage.json").read_text())
        assert usage["totals"]["train_calls"] == 2
        assert usage["totals"]["provider_cost"] == 0.0
        assert usage["totals"]["cost_missing"] is False
        assert usage["train_calls"][0]["request_ids"]
        assert usage["train_calls"][0]["provider_cost"] == 0.0
        assert usage["train_calls"][0]["cost_missing"] is False

        catalog = _rows(directory / "checkpoint_catalog.jsonl")
        assert len(catalog) >= 4  # two baselines plus two trained components
        lineage = _rows(directory / "checkpoint_lineage.jsonl")
        assert lineage

        topology = json.loads((directory / "topology.json").read_text())
        assert topology["trainable_instances"] == ["home_1", "home_2"]
        assert topology["non_trainable_instances"] == ["away_1", "away_2"]
        assert topology["communication_channels"]

        match_set = json.loads((directory / "match_set.json").read_text())
        assert match_set["opponents"][0]["pinned_identity"] == "checkpoint::frozen-0007"

        horizon = _rows(directory / "horizon.jsonl")
        assert horizon[0]["quiescence_attested"] is True
        assert horizon[0]["horizon_kind"] == "wall_clock"

        rewards = _rows(directory / "team_rewards.jsonl")
        assert rewards[0]["optimized_channel"] == "score::team_home"

        liveness = _rows(directory / "instance_liveness.jsonl")
        assert liveness[0]["instances"]
        assert liveness[0]["entered_batch"] is True

        tps = json.loads((directory / "sampling_tps.json").read_text())
        assert tps["by_call"] and tps["generated_tokens"] > 0
        assert tps["clock_source"] == "RunClock"
        assert tps["service_time_semantics"] == "sum_of_per_call_submit_to_score_seconds"
        assert tps["makespan_semantics"] == "earliest_submit_to_latest_score_seconds"
        assert tps["service_time_seconds"] == tps["sampling_seconds"]
        assert tps["service_time_generated_tps"] == tps["weighted_aggregate_tps"]
        assert tps["makespan_seconds"] >= 0
        assert tps["rollout_count"] == len(tps["by_call"])
        if tps["makespan_seconds"] == 0:
            assert tps["end_to_end_generated_tps"] is None
            assert tps["end_to_end_rollouts_per_second"] is None

        traces = _rows(directory / "traces.jsonl")
        assert traces[0]["trace_digest"]
        receipts = _rows(directory / "reward_receipts.jsonl")
        assert receipts[0]["reward_id"]


def test_sampling_tps_distinguishes_service_time_from_concurrent_makespan(
    tmp_path: Path,
) -> None:
    with build_plane(_solo(), tmp_path) as plane:
        executor = _executor(plane, tmp_path, group_size=2, target_train_updates=1)
        executor.run(max_ticks=20, on_tick=_advance(plane))
        for index, (key, record) in enumerate(executor.evidence.items()):
            executor.evidence[key] = replace(
                record,
                submitted_at=float(index),
                scored_at=float(index + 2),
            )

        tps = executor._sampling_tps()

        assert tps["service_time_seconds"] == 4.0
        assert tps["makespan_seconds"] == 3.0
        assert tps["rollout_count"] == 2
        assert tps["service_time_generated_tps"] == pytest.approx(
            tps["generated_tokens"] / 4.0
        )
        assert tps["end_to_end_generated_tps"] == pytest.approx(
            tps["generated_tokens"] / 3.0
        )
        assert tps["end_to_end_rollouts_per_second"] == pytest.approx(2.0 / 3.0)

def test_the_receipt_says_whether_the_renderer_was_ever_verified() -> None:
    """Identity is not agreement, and a receipt must not conflate them.

    A container declares a renderer profile; whether a renderer here produces
    the same tokens is a separate question, answered only by a canary. A
    receipt that records the declaration alone reads as though the second
    question had been answered too.
    """

    from synth_optimizers.contracts.rl_records import RendererProfile, canary_digest

    unproven = RendererProfile(
        profile_id="renderers.stub.v1",
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:cfg",
        tokenizer_id="vendor/policy-20b",
        tokenizer_digest="sha256:tok",
        stop_token_ids=(2,),
    )
    assert unproven.agreement_proven is False
    assert unproven.canary_digest == ""

    proven = RendererProfile(
        profile_id="renderers.stub.v1",
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:cfg",
        tokenizer_id="vendor/policy-20b",
        tokenizer_digest="sha256:tok",
        stop_token_ids=(2,),
        canary_digest=canary_digest((1, 2, 3)),
    )
    assert proven.agreement_proven is True
    # Proving agreement does not change identity: the same profile either way.
    assert proven.fingerprint == unproven.fingerprint
