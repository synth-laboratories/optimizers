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
