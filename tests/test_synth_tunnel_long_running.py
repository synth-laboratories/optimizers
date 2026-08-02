from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_MODULE_PATH = Path(__file__).parents[1] / "src" / "synth_optimizers" / "tunnels.py"
_SPEC = importlib.util.spec_from_file_location("synth_optimizers_tunnels_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TUNNELS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TUNNELS
_SPEC.loader.exec_module(_TUNNELS)
attach_synth_tunnel_lease = _TUNNELS.attach_synth_tunnel_lease


class _FakeControlPlane:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object | None]] = []

    def _json_request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        **_: Any,
    ) -> dict[str, object]:
        self.requests.append((method, path, payload))
        if path.endswith("/attach"):
            return _credentials("worker-1", "agent-1", "2026-08-03T00:00:00Z")
        if path.endswith("/heartbeat"):
            return _credentials("worker-2", "agent-2", "2026-08-04T00:00:00Z")
        raise AssertionError(f"unexpected request: {method} {path}")


def _credentials(worker_token: str, agent_token: str, expires_at: str) -> dict[str, object]:
    return {
        "lease_id": "lease-a",
        "public_url": "https://relay.example/s/rt_a",
        "route_token": "rt_a",
        "worker_token": worker_token,
        "expires_at": expires_at,
        "agent_connect": {
            "transport": "ws",
            "url": "wss://relay.example/agent",
            "agent_token": agent_token,
        },
    }


def test_attach_by_id_preserves_url_and_heartbeat_adopts_live_credentials() -> None:
    control_plane = _FakeControlPlane()
    lease = attach_synth_tunnel_lease(
        control_plane,
        "lease-a",
        "http://127.0.0.1:8103",
        heartbeat_extend_ttl_seconds=28_800,
        wait_ready=False,
    )

    assert lease.public_url == "https://relay.example/s/rt_a"
    assert lease.worker_token == "worker-1"
    assert lease.requested_ttl_seconds == 28_800

    assert lease.send_heartbeat() == "worker-2"
    assert lease.worker_token == "worker-2"
    assert lease.expires_at == "2026-08-04T00:00:00Z"
    assert lease.agent_connect is not None
    assert lease.agent_connect["agent_token"] == "agent-2"
    assert control_plane.requests[-1][2] == {"extend_ttl_seconds": 28_800}
