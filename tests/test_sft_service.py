from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from synth_optimizers.providers.tinker import FakeTinkerProvider, TinkerAdapter, TinkerCredentials
from synth_optimizers.runtime import JobStore
from synth_optimizers.sft import (
    SftConfig,
    SftPublicServiceClient,
    SftService,
    SftServiceError,
    create_sft_http_server,
)
from synth_optimizers.sft_executor import TinkerSftExecutor
import json
import threading
import time
import urllib.error
import urllib.request
import pytest


def fixture_config(run_id: str = "sft_public_123") -> dict:
    return {
        "run_id": run_id,
        "backend": "fixture",
        "base_model": "openai/gpt-oss-20b",
        "checkpoint_steps": [1, 2],
        "training": {"steps": 2, "batch_size": 1, "checkpoint_every_steps": 1},
    }


def test_sft_service_owns_canonical_run_without_a_beta_executor(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    submitted = service.submit(fixture_config())
    assert submitted["run_id"] == "sft_public_123"
    assert submitted["algorithm"] == "sft"
    public_run = service.get("sft_public_123")
    assert public_run["run_id"] == "sft_public_123"
    assert public_run["status"] == "completed"
    assert "workspace_dir" not in public_run
    events = service.optimizer_events("sft_public_123")
    kinds = [event["event_type"] for event in events["events"]]
    assert "sft.dataset.validated" in kinds
    assert "sft.model.materialized" in kinds
    cancelled = service.cancel("sft_public_123")
    assert cancelled["status"] == "completed"
    service.store.close()


def test_sft_service_rejects_invalid_tinker_config() -> None:
    with pytest.raises(SftServiceError, match="training_file_id|examples|dataset"):
        SftConfig.from_mapping({"run_id": "sft_invalid", "backend": "tinker"})


def test_sft_service_creates_its_database_parent(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "new" / "sft.sqlite")
    assert (tmp_path / "new" / "sft.sqlite").is_file()
    service.store.close()


def test_sft_service_serializes_same_idempotency_key(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = list(pool.map(lambda _: service.submit(fixture_config()), range(2)))
    assert [result["run_id"] for result in submitted] == ["sft_public_123", "sft_public_123"]
    jobs = service.store._db.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0]
    assert jobs == 1
    train_events = [
        event
        for event in service.optimizer_events("sft_public_123")["events"]
        if event["event_type"] == "sft.step.metrics"
    ]
    assert len(train_events) == 2
    service.store.close()


def test_sft_service_honors_explicit_idempotency_scope_per_run(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    first = service.submit(
        fixture_config(), run_id="sft_workshop_a", idempotency_key="sft_workshop_a"
    )
    retried = service.submit(
        fixture_config(), run_id="sft_workshop_a", idempotency_key="sft_workshop_a"
    )
    second = service.submit(
        fixture_config(), run_id="sft_workshop_b", idempotency_key="sft_workshop_b"
    )
    assert first["run_id"] == retried["run_id"] == "sft_workshop_a"
    assert second["run_id"] == "sft_workshop_b"
    jobs = service.store._db.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0]
    assert jobs == 2
    service.store.close()


def test_sft_http_service_hides_executor_behind_public_token(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
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
        service.store.close()


def test_resume_after_interruption_does_not_duplicate_steps(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    submitted = service.submit(fixture_config("sft_resume"))
    assert submitted["status"] == "completed"
    resumed = service.resume("sft_resume")
    assert resumed["status"] == "completed"
    train_events = [
        event
        for event in service.optimizer_events("sft_resume")["events"]
        if event["event_type"] == "sft.step.metrics"
    ]
    assert len(train_events) == 2
    service.store.close()


class _BlockingFake(FakeTinkerProvider):
    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self._gate = gate

    def train_step(self, session, request):
        assert self._gate.wait(timeout=30)
        return super().train_step(session, request)


def test_sft_http_unknown_run_event_stream_returns_404(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    server = create_sft_http_server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = SftPublicServiceClient(f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(SftServiceError, match="404"):
            list(client.optimizer_event_stream("sft_missing"))
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/v1/runs/sft_missing/optimizer-events?stream=1"
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        service.store.close()


def test_sft_http_live_event_stream_follows_the_job(tmp_path) -> None:
    gate = threading.Event()
    store = JobStore(tmp_path / "sft.sqlite")
    executor = TinkerSftExecutor(
        store,
        TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=_BlockingFake(gate)),
        sync=False,
    )
    service = SftService(tmp_path / "sft.sqlite", executor, background=True)
    server = create_sft_http_server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        client = SftPublicServiceClient(base, timeout_seconds=30)
        request = urllib.request.Request(
            f"{base}/v1/runs",
            method="POST",
            data=json.dumps({"algorithm": "sft", "config_json": fixture_config("sft_live")}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            submitted = json.loads(response.read())
        assert submitted["status"] in {"prepared", "queued", "running"}
        assert submitted["events_url"] == "/v1/runs/sft_live/optimizer-events"
        assert submitted["events_stream_url"] == "/v1/runs/sft_live/optimizer-events/stream"

        dropped = urllib.request.urlopen(
            urllib.request.Request(
                f"{base}{submitted['events_stream_url']}",
                headers={"Accept": "text/event-stream", "Connection": "close"},
            ),
            timeout=5,
        )
        dropped.close()

        seen: list[str] = []
        follow_error: list[Exception] = []

        def _follow() -> None:
            try:
                for event in client.optimizer_event_stream("sft_live"):
                    seen.append(str(event["event_type"]))
            except Exception as exc:
                follow_error.append(exc)

        follower = threading.Thread(target=_follow)
        follower.start()
        deadline = time.time() + 10
        while time.time() < deadline and not seen:
            time.sleep(0.05)
        assert seen, "SSE connected before train_step was released"
        gate.set()
        follower.join(timeout=30)
        assert not follow_error, follow_error[0]
        assert not follower.is_alive()
        assert "sft.step.metrics" in seen or "sft.training.started" in seen
        assert "sft.completed" in seen

        page = client.optimizer_events("sft_live")
        kinds = [event["event_type"] for event in page["events"]]
        assert "sft.completed" in kinds
        assert service.get("sft_live")["status"] == "completed"

        stream_query = urllib.request.Request(
            f"{base}/v1/runs/sft_live/optimizer-events?stream=1",
            headers={"Accept": "text/event-stream", "Connection": "close"},
        )
        with urllib.request.urlopen(stream_query, timeout=10) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            chunks: list[str] = []
            for raw in response:
                line = raw.decode("utf-8", errors="replace")
                chunks.append(line)
                if "sft.completed" in line:
                    break
        assert any("sft.completed" in line for line in chunks)
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        service.store.close()
