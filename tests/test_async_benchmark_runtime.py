from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/e2e'))
from async_benchmark_runtime import enable_async, enable_world_cleanup  # noqa: E402


def test_two_episodes_start_before_either_finishes():
    gate = threading.Barrier(3)
    class Runtime:
        def __init__(self):
            self._lock = threading.RLock()
            self.done = set()
        def start(self, attempt, log):
            gate.wait(timeout=3)
            self.done.add(attempt.rollout_id)
        def poll(self, attempt, log):
            return 'completed' if attempt.rollout_id in self.done else None
        def quiesce(self, attempt):
            return ()
    enable_async(Runtime, workers=2)
    runtime = Runtime()
    attempts = [SimpleNamespace(rollout_id=str(i)) for i in range(2)]
    try:
        for attempt in attempts:
            runtime.start(attempt, None)
        gate.wait(timeout=3)
        for attempt in attempts:
            assert runtime.quiesce(attempt) == ()
            assert runtime.poll(attempt, None) == 'completed'
    finally:
        runtime._episode_pool.shutdown()


def test_failed_episode_is_not_reported_as_completed():
    class Runtime:
        def __init__(self):
            self._lock = threading.RLock()
        def start(self, attempt, log):
            raise ValueError('real episode failed')
        def poll(self, attempt, log):
            return 'completed'
        def quiesce(self, attempt):
            return ()
    enable_async(Runtime)
    runtime = Runtime()
    attempt = SimpleNamespace(rollout_id='x')
    try:
        runtime.start(attempt, None)
        runtime._episode_pool.shutdown()
        with pytest.raises(ValueError, match='real episode failed'):
            runtime.poll(attempt, None)
    finally:
        runtime._episode_pool.shutdown()


@pytest.mark.parametrize('fail', [False, True])
def test_owned_world_is_released_on_success_and_failure(fail):
    released = []
    world = SimpleNamespace(rollout_id='owned',_request=lambda *args:released.append(args))
    class Runtime:
        def __init__(self):
            self._world_factory = lambda:world
        def _run_episode(self, plan, log):
            self._world_factory()
            if fail:
                raise ValueError('episode failed')
            return 42
    enable_world_cleanup(Runtime)
    runtime = Runtime()
    log = SimpleNamespace(append=lambda *args:None)
    if fail:
        with pytest.raises(ValueError):
            runtime._run_episode(None,log)
    else:
        assert runtime._run_episode(None,log)==42
    assert released==[('DELETE','/rollouts/owned',None)]
