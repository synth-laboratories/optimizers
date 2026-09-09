from types import SimpleNamespace

import pytest

from synth_optimizers.rl.experiment import ExperimentStore, CoordinationError
from synth_optimizers.rl.experiment_runner import run_experiment, failure_code
from test_experiment import spec


def test_runner_completes_and_does_not_repeat(tmp_path):
    store = ExperimentStore(tmp_path/'phases.db')
    design = spec(tmp_path)
    store.submit(design)
    calls = []
    driver = SimpleNamespace(perform=lambda phase, _: calls.append(phase['id']) or {'done': True})
    assert run_experiment(store, 'exp-a', driver)['state'] == 'completed'
    run_experiment(store, 'exp-a', driver)
    assert calls == [p['id'] for p in design.phases()]


def test_uncertain_operation_blocks_restart_and_controls(tmp_path):
    store = ExperimentStore(tmp_path/'phases.db')
    store.submit(spec(tmp_path))
    calls = []
    def fail(phase, snapshot):
        calls.append(phase)
        raise TimeoutError('response lost after provider admission')
    with pytest.raises(TimeoutError):
        run_experiment(store, 'exp-a', SimpleNamespace(perform=fail))
    assert run_experiment(store, 'exp-a', SimpleNamespace(perform=fail))['state'] == 'blocked'
    assert len(calls) == 1
    with pytest.raises(CoordinationError):
        store.control('exp-a', 'pause')


@pytest.mark.parametrize(('status', 'expected'), [(402, 'provider_credit_exhausted'),
    (401, 'authentication_failed'), (429, 'provider_overloaded')])
def test_nested_provider_failures(status, expected):
    cause = RuntimeError()
    cause.status = status
    error = RuntimeError()
    error.__cause__ = cause
    assert failure_code(error) == expected
