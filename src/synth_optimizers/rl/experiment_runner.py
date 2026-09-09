"""Fenced phase driver invocation, shared by service and CLI."""
from __future__ import annotations

import threading
import json

FAILURE_CODES = frozenset({'experiment_budget_exhausted', 'provider_credit_exhausted',
    'authentication_failed', 'provider_overloaded', 'transport_failure', 'invalid_grading',
    'invalid_evidence', 'storage_failure', 'operation_uncertain'})


def failure_code(error):
    from urllib.error import URLError
    from ..contracts.rl_records import RecordError
    current = error
    fallback = 'operation_uncertain'
    for _ in range(8):
        if current is None:
            break
        code = getattr(current, 'code', None)
        if code in FAILURE_CODES:
            return code
        if isinstance(current, OSError) and getattr(current, 'errno', None) in (28, 30):
            return 'storage_failure'
        if isinstance(current, (ConnectionError, TimeoutError, URLError)):
            fallback = 'transport_failure'
        elif isinstance(current, (ValueError, RecordError)):
            fallback = 'invalid_evidence'
        body = getattr(current, 'body', None)
        if isinstance(body, str):
            try:
                payload = json.loads(body)
                if payload.get('schema_version') == 'rl_runtime_error.v1' and payload.get('code') in FAILURE_CODES:
                    return payload['code']
            except (ValueError, AttributeError):
                pass
        status = getattr(getattr(current, 'response', None), 'status_code', None)
        status = status or getattr(current, 'status', None) or code
        if status == 402:
            return 'provider_credit_exhausted'
        if status in (401,403):
            return 'authentication_failed'
        if status in (429, 503):
            return 'provider_overloaded'
        current = current.__cause__ or current.__context__
    if isinstance(error, OSError) and getattr(error, 'errno', None) in (28,30):
        return 'storage_failure'
    return fallback


def run_experiment(store, experiment_id, driver, *, lease_seconds=60):
    """Drivers expose perform(phase, prior_snapshot). No arbitrary remote imports.

    Pause/stop prevent subsequent phase admission; a running phase drains to a
    durable boundary. Driver-level controls can implement finer executor pauses.
    """
    while claim := store.claim(experiment_id, lease_seconds=lease_seconds):
        stop = threading.Event()
        heartbeat_errors = []
        def heartbeat():
            while not stop.wait(lease_seconds / 3):
                try:
                    store.heartbeat(experiment_id, claim, lease_seconds=lease_seconds)
                except BaseException as error:
                    heartbeat_errors.append(error)
                    return
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if callable(getattr(driver, 'set_admission_check', None)):
                driver.set_admission_check(lambda: store.assert_owned(experiment_id, claim))
            result = driver.perform(claim['phase'], store.snapshot(experiment_id))
            if heartbeat_errors:
                raise heartbeat_errors[0]
            store.complete(experiment_id, claim, result)
        except BaseException as error:
            try:
                store.block(experiment_id, claim, failure_code(error))
            except Exception:
                # A lost lease cannot be used to mutate current owner state.
                pass
            raise
        finally:
            stop.set()
            thread.join()
    return store.snapshot(experiment_id)
