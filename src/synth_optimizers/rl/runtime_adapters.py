"""Bounded container runtime adapters with explicit resource ownership."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
import traceback


class EpisodeExecutionError(RuntimeError):
    """A real worker failure, never a reward or recoverable status string."""


_LOGGER = logging.getLogger(__name__)


class RuntimeOverloaded(RuntimeError):
    status = 429


class FencedProvider:
    """Check phase ownership before every dispatch, including save and restore."""
    _MUTATIONS = frozenset({'sample', 'sample_checkpoint', 'train_step', 'forward',
                           'save_checkpoint', 'restore_session', 'create_session'})

    def __init__(self, provider, check):
        self.provider, self.check = provider, check

    def __getattr__(self, name):
        value = getattr(self.provider, name)
        if name not in self._MUTATIONS:
            return value
        def dispatch(*args, **kwargs):
            self.check()
            return value(*args, **kwargs)
        return dispatch


class BoundedEpisodeRuntime:
    """Compose around a synchronous RolloutRuntime; never replace its methods.

    No unbounded executor queue: admission fails before scheduling when all
    slots are occupied. Quiescence waits for the actual episode, not its submit.
    The target owner must call close during application shutdown.
    """
    def __init__(self, runtime, *, workers=24, failure_path=None):
        if type(workers) is not int or not 1 <= workers <= 128:
            raise ValueError('episode workers must be between 1 and 128')
        self.runtime = runtime
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rl-episode')
        self._slots = threading.BoundedSemaphore(workers)
        self._lock = threading.RLock()
        self._futures = {}
        self._closed = False
        self._failure_path = str(failure_path) if failure_path is not None else None
        if self._failure_path is not None:
            Path(self._failure_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._failure_path) as db:
                db.execute('CREATE TABLE IF NOT EXISTS episode_failures (rollout_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')

    def _start_episode(self, attempt, log):
        try:
            return self.runtime.start(attempt, log)
        except Exception as exc:
            # Persist at the worker boundary: the coordinator may stop polling
            # as soon as another episode fails. Do not store locals, arbitrary
            # exception messages, or HTTP bodies which can contain credentials.
            chain = []
            current = exc
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                chain.append({'type': type(current).__module__ + '.' + type(current).__qualname__,
                              'frames': [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                                         for f in traceback.extract_tb(current.__traceback__)]})
                status = getattr(current, 'code', None)
                if type(status) is int and 100 <= status <= 599:
                    chain[-1]['http_status'] = status
                current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
            payload = {'schema_version': 'rl.episode_failure.v1',
                       'rollout_id': attempt.rollout_id, 'recorded_at_unix': time.time(),
                       'exception_chain': chain}
            if self._failure_path is not None:
                try:
                    with sqlite3.connect(self._failure_path, timeout=30) as db:
                        db.execute('PRAGMA synchronous=FULL')
                        db.execute('INSERT OR IGNORE INTO episode_failures VALUES (?, ?)',
                                   (attempt.rollout_id, json.dumps(payload, sort_keys=True)))
                except Exception as receipt_error:
                    raise EpisodeExecutionError(
                        f'episode {attempt.rollout_id} failed; failure receipt could not be written '
                        f'({type(receipt_error).__name__})') from exc
            _LOGGER.error('Episode failed: %s', json.dumps(payload, sort_keys=True))
            raise EpisodeExecutionError(
                f'episode {attempt.rollout_id} failed ({type(exc).__name__}); '
                f'failure receipt: {self._failure_path}') from exc

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def start(self, attempt, log):
        with self._lock:
            if self._closed:
                raise RuntimeError('episode runtime is closed')
            if attempt.rollout_id in self._futures:
                return
            if not self._slots.acquire(blocking=False):
                raise RuntimeOverloaded('episode concurrency exhausted; admission must back off')
            try:
                future = self._pool.submit(self._start_episode, attempt, log)
                self._futures[attempt.rollout_id] = future
                future.add_done_callback(lambda _: self._slots.release())
            except BaseException:
                self._slots.release()
                raise

    def poll(self, attempt, log):
        with self._lock:
            future = self._futures.get(attempt.rollout_id)
        if future is not None:
            if not future.done():
                return None
            future.result()
        return self.runtime.poll(attempt, log)

    def quiesce(self, attempt):
        with self._lock:
            future = self._futures.get(attempt.rollout_id)
        if future is not None:
            future.result(timeout=600)
        return self.runtime.quiesce(attempt)

    def cancel(self, attempt, reason):
        # A synchronous runtime's cancel flag need not interrupt its worker.
        # Do not acknowledge termination while it can still call the gateway:
        # the executor revokes that origin immediately after this returns.
        with self._lock:
            future = self._futures.get(attempt.rollout_id)
        if future is not None:
            future.result(timeout=600)
        return self.runtime.cancel(attempt, reason)

    def close(self):
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        close = getattr(self.runtime, 'close', None)
        if callable(close):
            close()
        # Shutdown must not silently succeed for a failed worker that nobody
        # polled (for example when a different rollout stopped the executor).
        for future in self._futures.values():
            future.result()
