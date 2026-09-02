"""Live event tail for SFT and CISPO.

The sqlite journal is the record. SSE/NDJSON is a mirror. A disconnected
reader must never stop the run; a reconnect uses ``after_sequence``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from ..contracts.training_schemas import TERMINAL_STATES
from .jobs import JobStore, canonical_json


class _SseWriter(Protocol):
    def send_response(self, code: int) -> None: ...
    def send_header(self, keyword: str, value: str) -> None: ...
    def end_headers(self) -> None: ...

    wfile: Any


def wants_live_stream(path: str, query: Mapping[str, list[str]]) -> bool:
    if path.rstrip("/").endswith("/optimizer-events/stream"):
        return True
    values = [item.lower() for item in query.get("stream", [])]
    return any(item in {"1", "true", "sse", "yes"} for item in values)


def after_sequence_from(query: Mapping[str, list[str]]) -> int:
    raw = (query.get("after_sequence") or query.get("after_seq") or ["0"])[0]
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def format_sse(event: Mapping[str, Any]) -> str:
    sequence = int(event.get("sequence") or event.get("sequence_number") or 0)
    return f"id: {sequence}\nevent: optimizer\ndata: {canonical_json(event)}\n\n"


def format_sse_comment(text: str = "ping") -> str:
    return f": {text}\n\n"


def iter_live_events(
    store: JobStore,
    job_id: str,
    *,
    after_sequence: int = 0,
    live: bool = True,
    idle_timeout: float = 1.0,
) -> Iterator[dict[str, Any] | None]:
    """Yield journal events, then wait for more until the job is terminal.

    ``None`` is a heartbeat: the journal did not grow during ``idle_timeout``.
    """

    cursor = after_sequence
    while True:
        page = store.events(job_id, after_sequence=cursor, limit=500)
        for event in page:
            cursor = int(event["sequence"])
            yield event
        state = store.require(job_id).state
        if state in TERMINAL_STATES:
            leftover = store.events(job_id, after_sequence=cursor, limit=500)
            for event in leftover:
                yield event
            return
        if not live:
            return
        before = cursor
        store.wait_for_events(job_id, cursor, timeout=idle_timeout)
        if store.events(job_id, after_sequence=before, limit=1) == []:
            yield None


def write_sse(
    handler: _SseWriter,
    store: JobStore,
    job_id: str,
    *,
    after_sequence: int = 0,
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    try:
        for item in iter_live_events(store, job_id, after_sequence=after_sequence):
            payload = format_sse_comment() if item is None else format_sse(item)
            handler.wfile.write(payload.encode("utf-8"))
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        return
