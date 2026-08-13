from __future__ import annotations

from synth_optimizers.board_server import _project_service_run, _sum_present, _usage_dict
from synth_optimizers.observability import OptimizerEvent, container_child_eval_ref


def test_missing_sequence_is_not_zero() -> None:
    heartbeat = OptimizerEvent.from_payload({"type": "optimizer.heartbeat", "run_id": "run_a"})
    started = OptimizerEvent.from_payload({"type": "optimizer.run.started", "run_id": "run_a"})
    numbered = OptimizerEvent.from_payload(
        {"type": "optimizer.run.started", "run_id": "run_a", "sequence_number": 3}
    )
    assert heartbeat.sequence_number is None
    assert started.sequence_number is None
    assert numbered.sequence_number == 3


def test_missing_usage_tokens_stay_none() -> None:
    usage = _usage_dict({})
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    present = _usage_dict({"prompt_tokens": 4, "completion_tokens": 6, "cost_usd": 0.02})
    assert present["total_tokens"] == 10
    assert present["cost_usd"] == 0.02


def test_missing_reward_stays_none() -> None:
    projected = _project_service_run({"run_id": "run_a", "usage": {}, "best_train_reward": None})
    assert projected["cost_usd"] is None
    assert projected["best_train_reward"] is None
    assert projected["usage"]["total_tokens"] is None
    assert _sum_present([None, None]) is None
    assert _sum_present([None, 1.5]) == 1.5


def test_container_child_eval_ref_has_three_fields() -> None:
    ref = container_child_eval_ref(
        "roll_abc",
        "stream_abc",
        "/reward?rollout_id=roll_abc",
    )
    assert ref["kind"] == "container_rollout"
    assert ref["id"] == "roll_abc"
    assert ref["attributes"]["stream_id"] == "stream_abc"
    assert ref["attributes"]["reward_url"] == "/reward?rollout_id=roll_abc"
    assert "frame" not in ref
    assert "nev" not in str(ref).lower()


def test_two_parallel_run_logs_do_not_share_a_spool() -> None:
    spool_a = {
        "optimizer_run_id": "gepa_luna",
        "log_id": "optimizer_event.v1:gepa_luna",
        "events": [
            OptimizerEvent.from_payload(
                {"type": "optimizer.run.started", "run_id": "gepa_luna", "sequence_number": 1}
            ),
            OptimizerEvent.from_payload(
                {"type": "optimizer.candidate.added", "run_id": "gepa_luna", "sequence_number": 2}
            ),
        ],
    }
    spool_b = {
        "optimizer_run_id": "gepa_sol",
        "log_id": "optimizer_event.v1:gepa_sol",
        "events": [
            OptimizerEvent.from_payload(
                {"type": "optimizer.run.started", "run_id": "gepa_sol", "sequence_number": 1}
            ),
            OptimizerEvent.from_payload(
                {"type": "optimizer.candidate.added", "run_id": "gepa_sol", "sequence_number": 2}
            ),
        ],
    }
    assert spool_a["log_id"] != spool_b["log_id"]
    assert [event.sequence_number for event in spool_a["events"]] == [1, 2]
    assert [event.sequence_number for event in spool_b["events"]] == [1, 2]
    assert {event.run_id for event in spool_a["events"]} == {"gepa_luna"}
    assert {event.run_id for event in spool_b["events"]} == {"gepa_sol"}


def test_dual_gepa_hub_luna_vs_sol_does_not_cross_or_stall() -> None:
    from synth_optimizers.observability import DualGepaHub, gepa_policy_ref

    hub = DualGepaHub()
    luna = hub.start("gepa_luna", gepa_policy_ref(proposer_model="openai/gpt-5.6-luna"))
    sol = hub.start("gepa_sol", gepa_policy_ref(proposer_model="openai/gpt-5.6-sol"))
    assert luna.log_id != sol.log_id
    assert luna.policy_ref["config"]["proposer_model"] != sol.policy_ref["config"]["proposer_model"]
    luna.append(
        {
            "type": "optimizer.candidate.added",
            "sequence_number": 2,
            "item": {"reward": None},
        }
    )
    sol.append({"type": "optimizer.candidate.added", "sequence_number": 2})
    first, second = hub.flip_read("gepa_luna", "gepa_sol")
    assert {event.run_id for event in first} == {"gepa_luna"}
    assert {event.run_id for event in second} == {"gepa_sol"}
    assert luna.live and sol.live
    # Missing reward on the candidate stays absent; never coerced to 0.
    assert first[1].raw.get("item", {}).get("reward") is None
    try:
        gepa_policy_ref(proposer_model="")
        raise AssertionError("empty proposer_model must fail closed")
    except ValueError:
        pass

