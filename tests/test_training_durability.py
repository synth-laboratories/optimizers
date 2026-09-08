from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import threading

import pytest

from synth_optimizers.runtime import JobStore, JobStoreError
from synth_optimizers.read_models import _events_through, _page, reduce_summary


def prepare(store):
    return store.persist_prepared(
        algorithm_id="sft",
        implementation_version="sft.tinker.v1",
        provider="tinker",
        model_id="model",
        idempotency_key="key",
        config={},
        job_id="run",
    )


@pytest.mark.parametrize("phase", ["running", "evaluating", "materializing"])
def test_two_connections_cannot_claim_active_phase(tmp_path, phase):
    path = tmp_path / "jobs.sqlite"
    first, second = JobStore(path), JobStore(path)
    prepare(first)
    first.transition("run", phase)
    barrier = threading.Barrier(2)

    def claim(pair):
        store, owner = pair
        barrier.wait()
        try:
            store.claim("run", owner)
            return True
        except JobStoreError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(claim, [(first, "one"), (second, "two")])) == [False, True]
    first.close()
    second.close()


def test_stale_owner_cannot_commit_after_takeover(tmp_path):
    path = tmp_path / "jobs.sqlite"
    first, second = JobStore(path), JobStore(path)
    prepare(first)
    first.claim("run", "old")
    second.claim("run", "new", stale_after_seconds=-1)
    with first.owned("old"):
        for write in (
            lambda: first.append_event("run", "success", {}, phase="running"),
            lambda: first.transition("run", "completed"),
            lambda: first.put_receipt("run", "request", {}),
        ):
            with pytest.raises(JobStoreError, match="fenced"):
                write()
    assert first.require("run").owner == "new"
    assert [event["kind"] for event in first.events("run")] == ["training.lifecycle"]
    first.close()
    second.close()


def test_complete_history_and_pinned_state_usage(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    prepare(store)
    store.transition("run", "running")
    for step in range(10002):
        store.append_event("run", "sft.step.metrics", {"step": step}, phase="running")
    bound = store.events("run", after_sequence=10002)[-1]["sequence"]
    assert len(_events_through(store, "run", bound)) == 10003
    store.put_receipt("run", "paid", {"request_id": "paid", "cost_usd": 2, "cost_missing": False})
    store.transition("run", "completed")
    historical = reduce_summary(store, "run", at_sequence=bound)
    assert historical["state"] == "running"
    assert historical["usage"]["cost_usd"] is None
    assert historical["progress"]["completed_units"] == 10002
    store.close()


def test_page_wire_limit_and_invalid_cursor():
    kwargs = dict(
        schema_version="test",
        projected_at_sequence=1,
        after_key=None,
        key_field="id",
        byte_limit=500,
    )
    rows = [{"id": str(i), "text": "漢字🙂" * 2} for i in range(20)]
    page = _page(rows, **kwargs)
    assert page.truncated
    assert len(json.dumps(asdict(page)).encode()) == page.bytes <= 500
    with pytest.raises(ValueError, match="row exceeds"):
        _page([{"id": "big", "text": "x" * 1000}], **kwargs)
    with pytest.raises(ValueError, match="stale"):
        _page(rows, **{**kwargs, "after_key": "missing"})


def test_sft_restart_restores_training_state_without_replaying_provider_work(tmp_path):
    from synth_optimizers.sft_executor import TinkerSftExecutor
    from synth_optimizers.recipes.banking77 import sft_recipe
    from synth_optimizers.providers.tinker import (
        FakeTinkerProvider,
        TinkerAdapter,
        TinkerCredentials,
    )

    store = JobStore(tmp_path / "jobs.sqlite")
    first = FakeTinkerProvider()
    executor = TinkerSftExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=first)
    )
    append = store.append_event_once

    class Crash(BaseException):
        pass

    def crash(job_id, kind, payload, *, phase):
        result = append(job_id, kind, payload, phase=phase)
        if kind == "sft.checkpoint.created":
            raise Crash()
        return result

    store.append_event_once = crash
    with pytest.raises(Crash):
        executor.submit(sft_recipe(steps=2).request, job_id="resume")
    store.close()
    reopened = JobStore(tmp_path / "jobs.sqlite")
    second = FakeTinkerProvider()
    resumed = TinkerSftExecutor(
        reopened, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=second)
    )
    assert resumed.resume("resume")["status"] == "completed"
    assert any(kind == "restore" for kind, _ in second.calls)
    original_ids = {request for _, request in first.calls}
    assert not original_ids.intersection(
        request for kind, request in second.calls if kind != "restore"
    )
    reopened.close()


def test_uncertain_update_is_not_retried_after_reopen(tmp_path):
    from synth_optimizers.runtime.operations import DurableProvider, UncertainOperation
    from synth_optimizers.providers.protocols import TrainingStepRequest

    store = JobStore(tmp_path / "jobs.sqlite")
    prepare(store)
    store.claim("run", "owner")
    calls = []

    class Provider:
        def train_step(self, session, request):
            calls.append(request.request_id)
            raise OSError("lost response")

    provider = DurableProvider(Provider(), store, "run", "owner")
    request = TrainingStepRequest("write", "cross_entropy", ())
    for _ in range(2):
        with pytest.raises(UncertainOperation):
            provider.train_step(None, request)
    assert calls == ["write"]
    store.close()


def test_no_evaluation_trains_single_chat_row_on_irregular_schedule(tmp_path):
    from synth_optimizers.sft_executor import TinkerSftExecutor
    from synth_optimizers.providers.tinker import (
        FakeTinkerProvider,
        TinkerAdapter,
        TinkerCredentials,
    )

    store = JobStore(tmp_path / "jobs.sqlite")
    transport = FakeTinkerProvider()
    executor = TinkerSftExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    result = executor.submit(
        {
            "model_id": "openai/gpt-oss-20b",
            "training": {"steps": 50},
            "checkpoint_schedule": {"save_steps": [10, 25, 50]},
            "checkpoint_evaluation": {"mode": "none"},
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": " question "},
                        {"role": "assistant", "content": "freeform answer"},
                    ]
                }
            ],
        },
        job_id="standalone",
    )
    assert result["status"] == "completed"
    assert not any(kind == "sample" for kind, _ in transport.calls)
    assert [
        e["payload"]["step"] for e in store.events("standalone", limit=5000) if e["kind"] == "sft.checkpoint.created"
    ] == [10, 25, 50]
    bundle = json.loads(store.artifact("standalone", "policy_bundle.json")[0])
    assert bundle["heldout"]["evaluated"] is False
    store.close()


def test_cancel_waits_for_admitted_training_call(tmp_path):
    from synth_optimizers.sft_executor import TinkerSftExecutor
    from synth_optimizers.providers.tinker import (
        FakeTinkerProvider,
        TinkerAdapter,
        TinkerCredentials,
    )

    entered, release = threading.Event(), threading.Event()

    class SlowProvider(FakeTinkerProvider):
        def train_step(self, session, request):
            entered.set()
            assert release.wait(5)
            return super().train_step(session, request)

    store = JobStore(tmp_path / "jobs.sqlite")
    transport = SlowProvider()
    executor = TinkerSftExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            executor.submit,
            {
                "training": {"steps": 2},
                "checkpoint_evaluation": {"mode": "none"},
                "examples": [{"text": "hello", "category": "world"}],
            },
            job_id="cancel",
        )
        assert entered.wait(5)
        assert executor.cancel("cancel")["status"] == "stop_requested"
        release.set()
        assert future.result()["status"] == "cancelled"
    assert len([kind for kind, _ in transport.calls if kind == "train"]) == 1
    assert len(store.receipts("cancel")) == 1
    store.close()


def test_restart_preserves_stateful_update_accumulators(tmp_path):
    from copy import deepcopy
    from synth_optimizers.sft_executor import TinkerSftExecutor
    from synth_optimizers.providers.tinker import (
        FakeTinkerProvider,
        TinkerAdapter,
        TinkerCredentials,
    )

    remote = {}

    class Stateful(FakeTinkerProvider):
        final = None

        def train_step(self, session, request):
            result = super().train_step(session, request)
            state = self.sessions[session.session_id]
            gradient = sum(sum(row["weights"]) for row in request.data)
            state["adam_m"] = 0.9 * state.get("adam_m", 0) + 0.1 * gradient
            state["adam_v"] = 0.999 * state.get("adam_v", 0) + 0.001 * gradient**2
            state["weight"] = state.get("weight", 1) - request.metadata["learning_rate"] * state[
                "adam_m"
            ] / (state["adam_v"] ** 0.5 + 1e-8)
            self.final = {key: state[key] for key in ("adam_m", "adam_v", "weight", "step")}
            return result

        def save_checkpoint(self, session_id, **kwargs):
            result = super().save_checkpoint(session_id, **kwargs)
            if kwargs["kind"] == "training":
                remote[result["resume_token"]] = deepcopy(self.sessions[session_id])
            return result

        def load_checkpoint(self, checkpoint, *, request_id):
            result = super().load_checkpoint(checkpoint, request_id=request_id)
            self.sessions[result["session_id"]] = deepcopy(remote[checkpoint.resume_token])
            return result

    config = {
        "training": {"steps": 4, "batch_size": 1},
        "checkpoint_steps": [2, 4],
        "checkpoint_evaluation": {"mode": "none"},
        "examples": [{"text": "a", "category": "x"}, {"text": "b", "category": "longer"}],
    }

    def executor(store, transport):
        return TinkerSftExecutor(
            store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
        )

    control = Stateful()
    control_store = JobStore(tmp_path / "control.sqlite")
    assert (
        executor(control_store, control).submit(config, job_id="control")["status"] == "completed"
    )
    store = JobStore(tmp_path / "restart.sqlite")
    first = Stateful()
    append = store.append_event_once

    class Crash(BaseException):
        pass

    def crash(job_id, kind, payload, *, phase):
        event = append(job_id, kind, payload, phase=phase)
        if kind == "sft.checkpoint.created" and payload["step"] == 2:
            raise Crash()
        return event

    store.append_event_once = crash
    with pytest.raises(Crash):
        executor(store, first).submit(config, job_id="restart")
    store.close()
    store = JobStore(tmp_path / "restart.sqlite")
    second = Stateful()
    assert executor(store, second).resume("restart")["status"] == "completed"
    assert second.final == control.final
    assert len([kind for kind, _ in second.calls if kind == "train"]) == 2
    store.close()
    control_store.close()


def test_failed_sampling_retains_completed_siblings_and_stops_admission(monkeypatch):
    from synth_optimizers.training_eval import evaluate_checkpoint
    from synth_optimizers.sft_dataset import parse_example
    from synth_optimizers.providers.protocols import ProviderUsage, SampleResult

    monkeypatch.setenv("SYNTH_OPTIMIZERS_SAMPLE_PARALLELISM", "2")
    calls, settled = [], []

    class Provider:
        def sample_checkpoint(self, checkpoint, request):
            calls.append(request.seed)
            if request.seed == 0:
                raise OSError("sample failed")
            return SampleResult(
                request.request_id,
                (1,),
                (-0.2,),
                "yes",
                "stop",
                ProviderUsage(output_tokens=1, cost_usd=0.01, cost_missing=False),
            )

    examples = [parse_example({"text": str(i), "category": "yes"}, index=i) for i in range(8)]
    with pytest.raises(OSError):
        evaluate_checkpoint(
            Provider(),
            {"checkpoint_id": "exact", "provider_reference": "exact"},
            examples,
            on_usage=lambda request, usage: settled.append(usage),
        )
    assert sorted(calls) == [0, 1]
    assert len(settled) == 1 and settled[0].cost_usd == 0.01


def test_public_config_preserves_canonical_save_final():
    from synth_optimizers.sft import SftConfig
    from synth_optimizers.contracts.checkpoint_plan import resolve_checkpoint_plan
    config = SftConfig.from_mapping({"run_id": "canonical", "backend": "fixture",
        "training": {"steps": 50}, "checkpoint_schedule": {"save_steps": [10, 25]},
        "checkpoint_evaluation": {"mode": "none"}})
    assert resolve_checkpoint_plan(config.config_json)["save_steps"] == [10, 25, 50]


def test_training_budget_survives_restart_and_unknown_usage(tmp_path):
    from synth_optimizers.runtime.training_budget import TrainingBudget
    from synth_optimizers.providers.protocols import TrainingStepRequest, TrainingStepResult, ProviderUsage
    from synth_optimizers.rl.budget import BudgetError
    store = JobStore(tmp_path / "budget.sqlite")
    prepare(store)
    plan = {"max_cost_usd": .003, "pricing": {"training_usd_per_million": 1000,
            "input_usd_per_million": 1000, "output_usd_per_million": 1000,
            "session_usd": 0, "save_usd": 0, "restore_usd": 0}}
    budget = TrainingBudget(store, "run", plan)
    request = TrainingStepRequest("one", "cross_entropy", ({"input_ids": [1, 2]},))
    budget.reserve("one", "train", request)
    budget.settle("one", TrainingStepResult("one", 1, {}, ProviderUsage()))
    reopened = TrainingBudget(store, "run", plan)
    assert reopened.ledger.snapshot()["counted_or_reserved_usd"] == .002
    with pytest.raises(BudgetError, match="aggregate reservation"):
        reopened.reserve("two", "train", request)
    with pytest.raises(BudgetError, match="cannot be reset"):
        TrainingBudget(store, "run", {**plan, "max_cost_usd": 1})
    store.close()


def test_pause_request_survives_checkpoint_phases(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite')
    prepare(store)
    store.transition('run', 'running')
    store.request_pause('run')
    for phase in ('running', 'materializing', 'evaluating'):
        assert store.transition('run', phase).state == 'pause_requested'
    assert store.transition('run', 'paused').state == 'paused'
    assert store.resume_prepared('run').state == 'prepared'
    store.close()


def test_training_jobs_share_aggregate_budget(tmp_path):
    from synth_optimizers.runtime.training_budget import TrainingBudget
    from synth_optimizers.rl.budget import BudgetError
    store = JobStore(tmp_path / 'jobs.sqlite')
    plan = {'experiment_id': 'authorized-canaries', 'max_cost_usd': 20,
            'pricing': {'session_usd': 12}}
    first = TrainingBudget(store, 'sft', plan)
    second = TrainingBudget(store, 'cispo', plan)
    first.reserve('sft-session', 'session')
    with pytest.raises(BudgetError):
        second.reserve('cispo-session', 'session')
    store.close()


def test_collection_cursor_pins_run_scope_and_sequence_after_restart(tmp_path):
    from synth_optimizers.read_models import sft_collections
    path = tmp_path / 'jobs.sqlite'
    store = JobStore(path)
    prepare(store)
    for step in range(150):
        store.append_event('run', 'sft.step.metrics', {'step': step}, phase='running')
    first = sft_collections(store, 'run', collection='training_metrics')
    assert first.truncated and len(first.items) == 100
    for step in range(150, 170):
        store.append_event('run', 'sft.step.metrics', {'step': step}, phase='running')
    store.close()
    store = JobStore(path)
    second = sft_collections(store, 'run', collection='training_metrics', after_key=first.next_key)
    assert len(second.items) == 50
    assert second.projected_at_sequence == first.projected_at_sequence
    assert not second.truncated
    with pytest.raises(ValueError, match='cursor'):
        sft_collections(store, 'run', collection='checkpoints', after_key=first.next_key)
    with pytest.raises(ValueError, match='cursor'):
        sft_collections(store, 'run', collection='training_metrics', after_key=first.next_key, at_sequence=170)
    store.close()


def test_resumed_claim_state_is_in_the_historical_journal(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite')
    prepare(store)
    store.request_pause('run')
    store.transition('run', 'paused')
    store.resume_prepared('run')
    store.claim('run', 'resumed')
    sequence = store.events('run')[-1]['sequence']
    assert reduce_summary(store, 'run', at_sequence=sequence)['state'] == 'running'
    store.close()


def test_cancel_drained_pause_and_preserve_blocked_recovery(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    prepare(store)
    store.transition("run", "paused")
    assert store.request_cancel("run").state == "cancelled"
    store.close()
    for state in ("blocked_budget", "blocked_evaluation", "blocked_uncertain"):
        store = JobStore(tmp_path / (state + ".sqlite"))
        prepare(store)
        store.transition("run", state, error="retained recovery reason")
        assert store.request_pause("run").state == state
        with pytest.raises(JobStoreError, match="reconciliation"):
            store.request_cancel("run")
        assert store.require("run").error == "retained recovery reason"
        store.close()
