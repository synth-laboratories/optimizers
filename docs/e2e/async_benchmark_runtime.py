"""Bounded asynchronous dispatch for synchronous image episode runtimes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading


def enable_world_cleanup(runtime_class):
    """Release only the Rust session created by this episode, including failures."""
    original_init = runtime_class.__init__
    original_run = runtime_class._run_episode
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._managed_world = threading.local()
        factory = self._world_factory
        def tracked():
            world = factory()
            self._managed_world.current = world
            return world
        self._world_factory = tracked
    def run(self, plan, log):
        try:
            return original_run(self, plan, log)
        finally:
            world = getattr(self._managed_world, 'current', None)
            if world is not None:
                try:
                    if world.rollout_id:
                        world._request('DELETE', f'/rollouts/{world.rollout_id}', None)
                        log.append('env.session.released', {'engine_rollout_id':world.rollout_id})
                finally:
                    del self._managed_world.current
    runtime_class.__init__ = initialize
    runtime_class._run_episode = run


def enable_async(runtime_class, workers=24):
    original_start = runtime_class.start
    original_poll = runtime_class.poll
    original_quiesce = runtime_class.quiesce
    original_init = runtime_class.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._episode_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='real-episode')
        self._episode_futures = {}

    def start(self, attempt, log):
        with self._lock:
            if attempt.rollout_id not in self._episode_futures:
                self._episode_futures[attempt.rollout_id] = self._episode_pool.submit(original_start, self, attempt, log)

    def poll(self, attempt, log):
        future = self._episode_futures.get(attempt.rollout_id)
        if future is not None:
            if not future.done():
                return None
            future.result()  # A failed episode is never presented as scored.
        return original_poll(self, attempt, log)

    def quiesce(self, attempt):
        future = self._episode_futures.get(attempt.rollout_id)
        if future is not None:
            future.result(timeout=600)
        return original_quiesce(self, attempt)

    runtime_class.__init__ = initialize
    runtime_class.start = start
    runtime_class.poll = poll
    runtime_class.quiesce = quiesce
