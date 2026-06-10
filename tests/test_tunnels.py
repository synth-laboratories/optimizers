from __future__ import annotations

import unittest
import urllib.error
from unittest.mock import Mock, patch

from synth_optimizers import tunnels


class TunnelFailureModeTests(unittest.TestCase):
    def test_missing_binary_names_required_provider_binary(self) -> None:
        with patch.object(tunnels.shutil, "which", return_value=None):
            with self.assertRaisesRegex(
                tunnels.TunnelError,
                "cloudflared tunnel provider requires the 'cloudflared' binary",
            ):
                tunnels._required_binary("cloudflared", tunnels.TunnelProvider.CLOUDFLARED)

    def test_ngrok_exits_before_public_url_is_reported(self) -> None:
        process = Mock()
        process.poll.return_value = 17

        with self.assertRaisesRegex(
            tunnels.TunnelError,
            "ngrok process exited early with status 17",
        ):
            tunnels._discover_ngrok_public_url(
                process,
                "http://127.0.0.1:4040",
                timeout_seconds=1.0,
            )

    def test_health_timeout_reports_target_url(self) -> None:
        with (
            patch.object(
                tunnels.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            ),
            patch.object(tunnels.time, "sleep", return_value=None),
            patch.object(tunnels.time, "monotonic", side_effect=[0.0, 0.0, 0.02]),
        ):
            with self.assertRaisesRegex(
                tunnels.TunnelError,
                "timed out waiting for http://127.0.0.1:8943/health",
            ):
                tunnels._wait_for_http_ok(
                    "http://127.0.0.1:8943/health",
                    timeout_seconds=0.01,
                )

    def test_ready_failure_releases_managed_tunnel_lease(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def _json_request(
                self, method: str, path: str, *args: object, **kwargs: object
            ) -> dict:
                self.calls.append((method, path))
                if method == "POST" and path == "/v1/tunnels/lease":
                    return {
                        "lease_id": "lease_123",
                        "public_url": "https://example.com/s/route",
                        "tunnel_token": "token_123",
                        "connector_mode": "cloudflared_tunnel_token",
                    }
                if method == "POST" and path == "/v1/tunnels/lease/lease_123/release":
                    return {}
                raise AssertionError(f"unexpected request {method} {path}")

        client = FakeClient()

        with (
            patch.object(tunnels, "_required_binary", return_value="/usr/bin/cloudflared"),
            patch.object(
                tunnels.CloudflaredTunnelLease,
                "wait_ready",
                side_effect=tunnels.TunnelError("readiness failed"),
            ),
        ):
            with self.assertRaisesRegex(tunnels.TunnelError, "readiness failed"):
                tunnels.create_tunnel_lease(
                    client,
                    "http://127.0.0.1:8943",
                    provider=tunnels.TunnelProvider.CLOUDFLARED,
                    wait_ready=True,
                )

        self.assertIn(("POST", "/v1/tunnels/lease/lease_123/release"), client.calls)


if __name__ == "__main__":
    unittest.main()
