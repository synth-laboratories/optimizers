from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from synth_optimizers.sft import (
    SftConfig,
    SftService,
    SftServiceError,
    create_sft_http_server,
)


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.cancelled = False

    def request(self, method: str, path: str, payload=None) -> dict[str, Any]:
        self.requests.append((method, path, payload))
        if method == "POST" and path == "/v1/runs":
            return {"run_id": "beta_sft_123", "status": "queued"}
        if method == "POST" and path.endswith("/cancel"):
            self.cancelled = True
            return {"run_id": "beta_sft_123", "status": "cancelled"}
        if method == "GET" and path.endswith("/optimizer-events?after_sequence=0&limit=500"):
            return {
                "run_id": "beta_sft_123",
                "events": [{"sequence_number": 1, "event_type": "sft.training.queued"}],
            }
        if method == "GET" and path == "/v1/runs/beta_sft_123":
            return {
                "run_id": "beta_sft_123",
                "status": "cancelled" if self.cancelled else "queued",
                "workspace_dir": "/private/executor/workspace",
                "config_path": "/private/executor/sft.toml",
                "storage_mode": "local",
            }
        raise AssertionError((method, path, payload))


def fixture_config(run_id: str = "sft_public_123") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "backend": "fixture",
        "base_model": "openai/gpt-oss-20b",
        "checkpoint_steps": [10, 20],
    }


def test_sft_service_owns_canonical_run_and_delegates_to_beta(tmp_path) -> None:
    executor = FakeExecutor()
    service = SftService(tmp_path / "sft.sqlite", executor)

    submitted = service.submit(fixture_config())

    assert submitted == {
        "run_id": "sft_public_123",
        "algorithm": "sft",
        "status": "queued",
        "events_url": "/v1/runs/sft_public_123/optimizer-events",
        "status_url": "/v1/runs/sft_public_123",
        "artifact_base_url": "/v1/runs/sft_public_123/artifacts",
    }
    assert executor.requests[0] == (
        "POST",
        "/v1/runs",
        {
            "algorithm": "sft",
            "idempotency_key": "sft_public_123",
            "config_json": {**fixture_config(), "accelerator_slots": 1},
        },
    )
    public_run = service.get("sft_public_123")
    assert public_run["run_id"] == "sft_public_123"
    assert "workspace_dir" not in public_run
    assert "config_path" not in public_run
    assert "storage_mode" not in public_run
    assert service.optimizer_events("sft_public_123")["run_id"] == "sft_public_123"
    assert service.cancel("sft_public_123")["status"] == "cancelled"


def test_sft_service_rejects_invalid_tinker_config() -> None:
    with pytest.raises(SftServiceError, match="training_file_id"):
        SftConfig.from_mapping({"run_id": "sft_invalid", "backend": "tinker"})


def test_sft_service_creates_its_database_parent(tmp_path) -> None:
    service = SftService(tmp_path / "new" / "sft.sqlite", FakeExecutor())

    assert (tmp_path / "new" / "sft.sqlite").is_file()
    service._db.close()


def test_sft_service_serializes_same_idempotency_key(tmp_path) -> None:
    class SlowExecutor(FakeExecutor):
        def request(self, method: str, path: str, payload=None) -> dict[str, Any]:
            if method == "POST" and path == "/v1/runs":
                time.sleep(0.05)
            return super().request(method, path, payload)

    executor = SlowExecutor()
    service = SftService(tmp_path / "sft.sqlite", executor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = list(pool.map(lambda _: service.submit(fixture_config()), range(2)))

    assert [result["run_id"] for result in submitted] == ["sft_public_123", "sft_public_123"]
    assert [request[0:2] for request in executor.requests].count(("POST", "/v1/runs")) == 1


def test_sft_http_service_hides_executor_behind_public_token(tmp_path) -> None:
    service = SftService(tmp_path / "sft.sqlite", FakeExecutor())
    server = create_sft_http_server(("127.0.0.1", 0), service, service_token="public-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/runs"
        request = urllib.request.Request(
            url,
            method="POST",
            data=json.dumps({"algorithm": "sft", "config_json": fixture_config()}).encode(),
            headers={"Authorization": "Bearer public-token", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            submitted = json.loads(response.read())
        assert submitted["run_id"] == "sft_public_123"

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url)
        assert error.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
