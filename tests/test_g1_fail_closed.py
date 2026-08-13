from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "synth_optimizers"
if "synth_optimizers" not in sys.modules:
    _pkg = types.ModuleType("synth_optimizers")
    _pkg.__path__ = [str(_SRC)]
    _pkg.__package__ = "synth_optimizers"
    sys.modules["synth_optimizers"] = _pkg

if "synth_optimizers._synth_optimizers" not in sys.modules:
    _native = types.ModuleType("synth_optimizers._synth_optimizers")
    _native.__version__ = "0.0.0-g1-test"

    class _Err(Exception):
        pass

    for _name in (
        "BudgetExceededError",
        "CacheCorruptError",
        "CacheFullError",
        "CacheMissError",
        "CancelledError",
        "ConfigError",
        "ContainerContractError",
        "EventCompareError",
        "InvariantError",
        "OptimizerDiskBudgetError",
        "OptimizerHttpError",
        "OptimizerIoError",
        "OptimizerJsonError",
        "OptimizerSqliteError",
        "OptimizerTomlDecodeError",
        "ProposerError",
        "RunFailedError",
        "StateTransitionError",
        "SynthOptimizerError",
    ):
        setattr(_native, _name, type(_name, (_Err,), {}))
    _native.GepaRunResult = object
    _native.events_compare = lambda *args, **kwargs: None
    _native.events_replay = lambda *args, **kwargs: None
    _native.gepa_compact_run_storage = lambda *args, **kwargs: None
    _native.gepa_delete_run_storage = lambda *args, **kwargs: None
    _native.gepa_inspect_run_storage = lambda *args, **kwargs: None
    _native.gepa_serve = lambda *args, **kwargs: None
    _native.gepa_workspace_storage_health = lambda *args, **kwargs: None
    sys.modules["synth_optimizers._synth_optimizers"] = _native

from synth_optimizers.board_server import _project_service_run, _sum_present, _usage_dict  # noqa: E402
from synth_optimizers.observability import (  # noqa: E402
    A3_TASK,
    CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
    DEFAULT_PROPOSER_DELTA_CHANNEL,
    DualGepaHub,
    LUNA_MED_POLICY_CONFIG,
    LUNA_MED_PROPOSER_POLICY_REF,
    PROPOSER_DELTA_EVENT_TYPE,
    SOL_MED_POLICY_CONFIG,
    SOL_MED_PROPOSER_POLICY_REF,
    OptimizerEvent,
    banking77_eval_policy_ref,
    container_child_eval_ref,
    optimizer_event_log_id,
    policy_ref,
    proposer_delta_payload,
)


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
    assert ref["role"] == "candidate_evaluation"
    assert ref["attributes"]["stream_id"] == "stream_abc"
    assert ref["attributes"]["reward_url"] == "/reward?rollout_id=roll_abc"
    assert "frame" not in ref
    assert "nev" not in str(ref).lower()
    assert "child_eval_ref" not in str(ref)
    assert "synth.stream-event.v1" not in str(ref)


def test_two_parallel_run_logs_do_not_share_a_spool() -> None:
    spool_a = {
        "optimizer_run_id": "gepa_luna",
        "log_id": optimizer_event_log_id("gepa_luna"),
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
        "log_id": optimizer_event_log_id("gepa_sol"),
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


def test_luna_and_sol_are_distinct_policy_refs_not_tasks() -> None:
    assert LUNA_MED_POLICY_CONFIG != A3_TASK
    assert SOL_MED_POLICY_CONFIG != A3_TASK
    assert LUNA_MED_PROPOSER_POLICY_REF["harness"] == "gepa_proposer"
    assert LUNA_MED_PROPOSER_POLICY_REF["config"] == "luna_med"
    assert SOL_MED_PROPOSER_POLICY_REF["config"] == "sol_med"
    assert LUNA_MED_PROPOSER_POLICY_REF != SOL_MED_PROPOSER_POLICY_REF
    assert "harness_ref" not in LUNA_MED_PROPOSER_POLICY_REF
    child = banking77_eval_policy_ref(config="candidate")
    assert child["harness"] == "banking77_eval"
    assert child != LUNA_MED_PROPOSER_POLICY_REF
    with_code = policy_ref(harness="banking77_eval", config="candidate", code="print('ok')")
    assert with_code["code"] == "print('ok')"


def test_a3_task_is_banking77_not_jsonl_smoke() -> None:
    assert A3_TASK == "banking77"
    # Bounded JSONL recipes may exist for local wiring. They are not A3 proof.
    assert A3_TASK != "jsonl_smoke"


def test_dual_gepa_hub_luna_vs_sol_does_not_cross_or_stall() -> None:
    hub = DualGepaHub()
    luna, sol = hub.start_luna_and_sol()
    assert luna.log_id != sol.log_id
    assert luna.policy_ref == LUNA_MED_PROPOSER_POLICY_REF
    assert sol.policy_ref == SOL_MED_PROPOSER_POLICY_REF
    assert luna.events[0].raw["delta"]["task"] == "banking77"
    assert luna.events[1].type == PROPOSER_DELTA_EVENT_TYPE
    assert luna.events[1].delta == proposer_delta_payload(
        0, "gepa_luna proposer sample", channel=DEFAULT_PROPOSER_DELTA_CHANNEL
    )
    assert sol.events[1].delta["text"] == "gepa_sol proposer sample"
    assert luna.events[1].delta["text"] != sol.events[1].delta["text"]

    luna.append(
        {
            "type": "optimizer.candidate.added",
            "sequence_number": 3,
            "item": {"type": "candidate", "reward": None},
        }
    )
    sol.append(
        {
            "type": "optimizer.candidate.added",
            "sequence_number": 3,
            "item": {"type": "candidate"},
        }
    )
    first, second = hub.flip_read("gepa_luna", "gepa_sol", first_after=2)
    assert [event.sequence_number for event in first] == [3]
    assert [event.sequence_number for event in second] == [1, 2, 3]
    assert {event.run_id for event in first} == {"gepa_luna"}
    assert {event.run_id for event in second} == {"gepa_sol"}
    assert luna.live and sol.live
    assert first[0].raw["item"]["reward"] is None

    luna.append({"type": "optimizer.frontier.updated", "sequence_number": 4})
    luna_tail, sol_tail = hub.flip_read("gepa_luna", "gepa_sol", first_after=3, second_after=3)
    assert [event.sequence_number for event in luna_tail] == [4]
    assert sol_tail == ()


def test_dual_gepa_hub_rejects_cross_run_and_bad_cursors() -> None:
    hub = DualGepaHub()
    luna, _sol = hub.start_luna_and_sol()
    with pytest.raises(ValueError, match="cannot be appended"):
        luna.append(
            {
                "type": "optimizer.candidate.added",
                "sequence_number": 2,
                "run_id": "gepa_sol",
            }
        )
    with pytest.raises(ValueError, match="sequence_number is required"):
        luna.append({"type": "optimizer.candidate.added"})
    with pytest.raises(ValueError, match="advance monotonically"):
        luna.append({"type": "optimizer.candidate.added", "sequence_number": 1})
    with pytest.raises(ValueError, match="policy_ref.config"):
        policy_ref(harness="gepa_proposer", config="")


def test_producer_emits_proposer_delta_on_optimizer_event() -> None:
    hub = DualGepaHub()
    luna, sol = hub.start_luna_and_sol()
    luna.append_proposer_delta(1, "Hello ", channel="reasoning")
    luna.append_proposer_delta(1, "world.", channel=DEFAULT_PROPOSER_DELTA_CHANNEL)
    deltas = [event for event in luna.events if event.type == PROPOSER_DELTA_EVENT_TYPE]
    assert [event.sequence_number for event in luna.events] == [1, 2, 3, 4]
    assert deltas[0].delta["generation"] == 0
    assert deltas[1].delta == proposer_delta_payload(1, "Hello ", channel="reasoning")
    assert deltas[2].delta["channel"] == DEFAULT_PROPOSER_DELTA_CHANNEL
    assert deltas[2].delta["text"] == "world."
    sol_deltas = [event for event in sol.events if event.type == PROPOSER_DELTA_EVENT_TYPE]
    assert [event.delta["text"] for event in sol_deltas] == ["gepa_sol proposer sample"]
    assert all(event.run_id == "gepa_luna" for event in deltas)
    assert all(event.sequence_number is not None and event.sequence_number > 0 for event in deltas)


def test_two_hubs_do_not_cross_proposer_deltas() -> None:
    hub = DualGepaHub()
    luna, sol = hub.start_luna_and_sol()
    luna.append_proposer_delta(0, "luna-only")
    first, second = hub.flip_read("gepa_luna", "gepa_sol", first_after=2, second_after=1)
    assert [event.delta.get("text") for event in first] == ["luna-only"]
    assert [event.run_id for event in first] == ["gepa_luna"]
    assert all(event.run_id == "gepa_sol" for event in second)
    assert all(
        event.delta.get("text") != "luna-only"
        for event in second
        if event.type == PROPOSER_DELTA_EVENT_TYPE
    )
    assert luna.live and sol.live


def test_child_eval_link_event_shape_is_exact() -> None:
    hub = DualGepaHub()
    luna, _sol = hub.start_luna_and_sol()
    event = luna.append_child_eval(
        "rollout_abc",
        "stream:abc",
        "/reward?rollout_id=rollout_abc",
    )
    assert event.type == CHILD_ROLLOUT_ATTACHED_EVENT_TYPE
    ref = event.delta["child_resource_ref"]
    assert ref["schema"] == "synth.resource-ref.v1"
    assert ref["kind"] == "container_rollout"
    assert ref["id"] == "rollout_abc"
    assert ref["attributes"] == {
        "stream_id": "stream:abc",
        "reward_url": "/reward?rollout_id=rollout_abc",
    }
    assert "frame" not in ref
    assert "nev" not in str(ref).lower()
    assert "child_eval_ref" not in str(ref)
    assert "harness_ref" not in str(ref)
    assert "synth.trace-stream-event.v1" not in str(event.raw)
