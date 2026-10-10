"""Adverse stdio boundaries; canonical API effects are qualified by tunnel E2E."""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from synth_optimizers.tunnel_custody_bridge import dispatch, close_all, MAX_MESSAGE_BYTES
from synth_optimizers.tunnel_custody_journal import FileLeaseCustodyJournal
from synth_optimizers.tunnels import TunnelError
from synth_optimizers.tunnel_custody_transport import CustodyApiClient, MAX_AUTHORITY_BYTES


class NoEffects:
    def _json_request(self, *_args, **_kwargs):
        raise AssertionError("invalid request must not reach authority")


def request(tmp_path):
    return {
        "version": 1, "action": "submit", "directory": str(tmp_path),
        "binding": {"connector_id": str(uuid4()), "route_ids": [str(uuid4())],
                    "route_name": "fixture", "gateway_url": "http://127.0.0.1:1",
                    "local_url": "http://127.0.0.1:2"},
        "payload": {"algorithm": "gepa", "run_id": "stable_job",
                    "config_json": {"run": {"run_id": "different_job"}}},
    }


def test_config_identity_and_unqualified_parent_mapping_refused_before_effects(tmp_path):
    payload = request(tmp_path)
    with pytest.raises(TunnelError, match="config job identity mismatch"):
        dispatch(payload, client=NoEffects())
    payload["payload"]["algorithm"] = "sft"
    with pytest.raises(TunnelError, match="parent/child mapping"):
        dispatch(payload, client=NoEffects())
    assert not list(tmp_path.iterdir())


def test_close_retains_unresolved_receipt_and_never_reports_termination(tmp_path):
    journal = FileLeaseCustodyJournal(tmp_path, uuid4())
    journal.save({"connector": str(uuid4()), "phase": "grant_in_doubt"})
    result = close_all(NoEffects(), tmp_path)
    assert result == {"version": 1, "closed": 0, "unresolved": 1}
    assert journal.read()["phase"] == "grant_in_doubt"


# Linux bounds each argv/environment string. Pytest exports the case ID in
# PYTEST_CURRENT_TEST; embedding the 256 KiB payload prevents child execution.
@pytest.mark.parametrize("body", [b"x" * (MAX_MESSAGE_BYTES + 1), json.dumps({"version": 1, "action": "invented", "api_key": "private-authority-canary"}).encode()], ids=["oversize", "unknown_action"])
def test_real_stdio_cli_refuses_oversize_and_sanitizes_diagnostics(body):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, "-m", "synth_optimizers.cli", "tunnel-custody"], input=body, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=10)
    assert result.returncode == 1
    assert b"private-authority-canary" not in result.stdout + result.stderr
    assert not result.stderr
    assert json.loads(result.stdout)["error"] == "custody_refused"


def test_actual_http_transport_bounds_replies_and_never_follows_redirects():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path.endswith("redirect"):
                self.send_response(302)
                self.send_header("Location", "/api/v1/secret-target")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                try:
                    self.wfile.write(b"x" * (MAX_AUTHORITY_BYTES + 1))
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api = CustodyApiClient(f"http://127.0.0.1:{server.server_port}", "private-canary", 2)
        with pytest.raises(TunnelError, match="unresolved"):
            api._json_request("GET", "/api/v1/redirect")
        with pytest.raises(TunnelError, match="response byte capacity"):
            api._json_request("GET", "/api/v1/large")
        assert calls == ["/api/v1/redirect", "/api/v1/large"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_job_identity_is_fsynced_before_the_first_grant_effect(tmp_path):
    from synth_optimizers.shared_tunnels import borrow_shared_synth_tunnel
    from synth_optimizers.tunnel_custody_journal import TunnelCustodyInDoubt

    lease = uuid4()
    effects = []

    class Backend:
        def _json_request(self, method, path, payload=None):
            if method == "GET":
                return {"revision": 1}
            retained = FileLeaseCustodyJournal(tmp_path, lease).read()
            assert retained["job_hint"] == "retained_job"
            assert retained["phase"] == "grant_in_doubt"
            assert retained["grant_request"] == payload
            effects.append(path)
            raise TimeoutError("grant reply lost")

    with pytest.raises(TunnelCustodyInDoubt, match="grant"):
        borrow_shared_synth_tunnel(
            Backend(), connector=uuid4(), routes=(uuid4(),), route_name="fixture",
            gateway_url="http://127.0.0.1:1", local_url="http://127.0.0.1:2",
            custody_directory=tmp_path, lease_identity=lease, job_identity="retained_job",
        )
    assert len(effects) == 1
    assert FileLeaseCustodyJournal(tmp_path, lease).read()["job_hint"] == "retained_job"
