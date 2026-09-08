import threading
import json
import sqlite3
from types import SimpleNamespace

import pytest

from synth_optimizers.rl.runtime_adapters import BoundedEpisodeRuntime, EpisodeExecutionError, FencedProvider, RuntimeOverloaded
from synth_optimizers.rl.grading import BudgetedRubricJudge
from synth_optimizers.rl.budget import ExperimentBudget


def test_episode_admission_is_bounded_and_close_drains():
    gate = threading.Event()
    started = []
    runtime = SimpleNamespace(start=lambda attempt, log: (started.append(attempt.rollout_id), gate.wait(3)),
                              poll=lambda *_: 'completed', quiesce=lambda _: ())
    wrapped = BoundedEpisodeRuntime(runtime, workers=1)
    one, two = SimpleNamespace(rollout_id='one'), SimpleNamespace(rollout_id='two')
    try:
        wrapped.start(one, None)
        wrapped.start(one, None)
        with pytest.raises(RuntimeOverloaded):
            wrapped.start(two, None)
        assert wrapped.poll(one, None) is None
        gate.set()
        assert wrapped.quiesce(one) == ()
        assert wrapped.poll(one, None) == 'completed'
    finally:
        gate.set()
        wrapped.close()
    assert started == ['one']


def test_lost_phase_ownership_prevents_provider_dispatch():
    calls = []
    def lost():
        raise RuntimeError('lease lost')
    provider = FencedProvider(SimpleNamespace(train_step=lambda *_: calls.append(1)), lost)
    with pytest.raises(RuntimeError):
        provider.train_step(None, None)
    assert not calls


def test_worker_failure_is_durable_and_raised_at_every_observation(tmp_path, caplog):
    original = ValueError('sensitive-provider-response')
    closed = []
    def fail(*_):
        raise original
    runtime = SimpleNamespace(start=fail, poll=lambda *_: pytest.fail('must not report completion'),
                              quiesce=lambda *_: pytest.fail('must not settle a failed episode'),
                              close=lambda: closed.append(True))
    path = tmp_path/'failures.sqlite3'
    wrapped = BoundedEpisodeRuntime(runtime, workers=1, failure_path=path)
    attempt = SimpleNamespace(rollout_id='failed-one')
    wrapped.start(attempt, None)
    with pytest.raises(EpisodeExecutionError) as caught:
        wrapped._futures[attempt.rollout_id].result(timeout=3)
    assert caught.value.__cause__ is original
    for observe in (lambda: wrapped.poll(attempt, None), lambda: wrapped.quiesce(attempt), wrapped.close):
        with pytest.raises(EpisodeExecutionError):
            observe()
    assert closed == [True]
    with sqlite3.connect(path) as db:
        rows = db.execute('SELECT payload FROM episode_failures').fetchall()
    assert len(rows) == 1
    payload = json.loads(rows[0][0])
    assert payload['rollout_id'] == attempt.rollout_id
    assert payload['exception_chain'][0]['type'] == 'builtins.ValueError'
    assert payload['exception_chain'][0]['frames'][-1]['function'] == 'fail'
    assert 'sensitive-provider-response' not in rows[0][0]
    assert 'sensitive-provider-response' not in caplog.text
    assert 'failed-one' in caplog.text


def test_failure_receipt_error_does_not_hide_original_worker_error(tmp_path):
    original = ValueError('worker failed')
    def fail(*_):
        raise original
    path = tmp_path/'failures.sqlite3'
    wrapped = BoundedEpisodeRuntime(SimpleNamespace(start=fail), workers=1, failure_path=path)
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE episode_failures')
    wrapped.start(SimpleNamespace(rollout_id='one'), None)
    with pytest.raises(EpisodeExecutionError, match='receipt could not be written') as caught:
        wrapped.close()
    assert caught.value.__cause__ is original


def test_cancellation_waits_for_worker_before_acknowledging_gateway_revocation():
    gate = threading.Event()
    entered = threading.Event()
    events = []
    def start(*_):
        entered.set()
        assert gate.wait(3)
        events.append('last_sample')
    def cancel(*_):
        events.append('cancel_ack')
    wrapped = BoundedEpisodeRuntime(SimpleNamespace(start=start, cancel=cancel), workers=1)
    attempt = SimpleNamespace(rollout_id='one')
    wrapped.start(attempt, None)
    assert entered.wait(3)
    waiter = threading.Thread(target=lambda: wrapped.cancel(attempt, 'stop'))
    waiter.start()
    assert events == []
    gate.set()
    waiter.join(3)
    assert not waiter.is_alive()
    events.append('revoke_gateway')
    wrapped.close()
    assert events == ['last_sample', 'cancel_ack', 'revoke_gateway']


def test_grader_preserves_text_order_and_cumulative_budget(tmp_path):
    seen = []
    class Judge:
        def grade(self, **kwargs):
            seen.append(kwargs['conversation'])
            return SimpleNamespace(index=kwargs['index'], usage={'prompt_tokens': 10, 'completion_tokens': 2})
    budget = ExperimentBudget(tmp_path/'budget.db', 'exp', 1)
    judge = BudgetedRubricJudge(Judge(), budget, input_rate=2, output_rate=8, workers=2)
    answer = '  Line ONE\n{"Action":"LEFT"}  '
    try:
        results = judge.grade_many(conversation=answer, rubrics=[{'criterion': 'a'}, {'criterion': 'b'}])
    finally:
        judge.close()
    assert [r.index for r in results] == [0, 1]
    assert seen == [answer, answer]
    assert budget.snapshot()['unsettled_operations'] == 0
    assert budget.snapshot()['counted_or_reserved_usd'] == pytest.approx(0.000072)
    settled = [e['payload'] for e in budget.events() if e['event_type'] == 'budget.settled']
    assert len(settled) == 2
    assert all(e['lane'] == 'rubric_judge' and e['duration_seconds'] >= 0 for e in settled)
