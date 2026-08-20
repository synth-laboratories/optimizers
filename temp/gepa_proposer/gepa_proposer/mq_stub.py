"""Local Manderqueue HTTP stub for GEPA ascope evals.

The engine polls:
  GET  {base}/health
  GET  {base}/v1/threads/{thread_id}/messages?after_seq=0&limit=50

and writes the JSON body into `state/manderqueue_inbox.json` plus
`state/guidance.md`. Seed one operator message so proposers see guidance.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

DEFAULT_THREAD_ID = "gepa-ascope"
DEFAULT_MESSAGE = (
    "Prefer prompt edits that generalize. Do not overfit the train minibatch. "
    "Record open hypotheses in state/hypotheses.json."
)


def seed_payload(thread_id: str, message: str) -> list[dict[str, Any]]:
    return [
        {
            "seq": 1,
            "thread_id": thread_id,
            "role": "operator",
            "text": message,
            "body": message,
        }
    ]


class ManderqueueHandler(BaseHTTPRequestHandler):
    thread_id = DEFAULT_THREAD_ID
    message = DEFAULT_MESSAGE

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _write(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._write(200, {"ok": True, "service": "manderqueue-stub"})
            return
        prefix = "/v1/threads/"
        suffix = "/messages"
        if path.startswith(prefix) and path.endswith(suffix):
            thread_id = path[len(prefix) : -len(suffix)]
            if not thread_id:
                self._write(404, {"error": "missing thread_id"})
                return
            self._write(200, seed_payload(thread_id, self.message))
            return
        self._write(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/v1/threads":
            self._write(200, {"id": self.thread_id, "thread_id": self.thread_id})
            return
        self._write(404, {"error": "not found", "path": path})


def serve(host: str, port: int, *, thread_id: str, message: str) -> ThreadingHTTPServer:
    handler = type(
        "BoundManderqueueHandler",
        (ManderqueueHandler,),
        {"thread_id": thread_id, "message": message},
    )
    server = ThreadingHTTPServer((host, port), handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local manderqueue stub for GEPA ascope")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    args = parser.parse_args()
    server = serve(args.host, args.port, thread_id=args.thread_id, message=args.message)
    print(
        json.dumps(
            {
                "ok": True,
                "base_url": f"http://{args.host}:{args.port}",
                "thread_id": args.thread_id,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
