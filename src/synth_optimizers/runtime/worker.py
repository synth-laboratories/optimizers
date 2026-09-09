"""In-process background workers so HTTP submit can return a live run id."""

from __future__ import annotations

import threading
from collections.abc import Callable

_started: set[str] = set()
_lock = threading.Lock()


def start_job_worker(job_id: str, run: Callable[[], object]) -> threading.Thread | None:
    """Run ``run`` once per job id. Duplicate submits share the first worker."""

    with _lock:
        if job_id in _started:
            return None
        _started.add(job_id)

    def _target() -> None:
        try:
            run()
        finally:
            with _lock:
                _started.discard(job_id)

    thread = threading.Thread(target=_target, name=f"training-{job_id}", daemon=True)
    thread.start()
    return thread


def execute_owned(store, job_id: str, execute):
    """Maintain a unique lease through long provider calls and fence stale commits."""
    import uuid
    from .jobs import JobStoreError
    from ..contracts.training_schemas import TERMINAL_STATES

    owner = f"worker-{uuid.uuid4().hex}"
    job = store.claim(job_id, owner)
    if job.state in TERMINAL_STATES:
        return None
    stopped = threading.Event()

    def renew():
        while not stopped.wait(5):
            try:
                store.heartbeat(job_id, owner)
            except JobStoreError:
                return

    thread = threading.Thread(target=renew, name=f"lease-{job_id}", daemon=True)
    thread.start()
    try:
        with store.owned(owner):
            return execute(job, owner)
    finally:
        stopped.set()
        thread.join()
        store.release(job_id, owner)


class AdmissionProvider:
    """Stop new calls while allowing already admitted calls to drain."""
    _operations = frozenset({"create_session", "restore_session", "train_step", "save_checkpoint",
                             "sample", "sample_checkpoint", "forward"})

    def __init__(self, provider, store, job_id):
        self.provider, self.store, self.job_id = provider, store, job_id

    def __getattr__(self, name):
        method = getattr(self.provider, name)
        if name not in self._operations:
            return method
        def admitted(*args, **kwargs):
            from ..providers.protocols import ProviderError
            if self.store.cancellation_requested(self.job_id):
                raise ProviderError("cancel_requested", "training cancellation requested")
            return method(*args, **kwargs)
        return admitted
