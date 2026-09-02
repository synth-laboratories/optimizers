from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from synth_optimizers.cispo_executor import TinkerCispoExecutor
from synth_optimizers.cispo_service import (
    CispoPublicServiceClient,
    CispoService,
    _use_fixture_executor,
    cispo_service_for_serve,
    create_cispo_http_server,
)
from synth_optimizers.providers.tinker import FakeTinkerProvider, TinkerAdapter, TinkerCredentials
from synth_optimizers.recipes.banking77 import cispo_recipe
from synth_optimizers.runtime import JobStore


def _learning_signal_request() -> dict:
    return cispo_recipe(mode="learning_signal", updates=1).request


def _mixed_sample_text(request) -> str:
    return "order_physical_card" if int(request.seed or 0) % 2 == 0 else "lost_or_stolen_card"


class _GatedFakeTinkerProvider(FakeTinkerProvider):
    def __init__(self, gate: threading.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gate = gate

    def sample(self, handle, request):
        self._gate.wait(timeout=30)
        return super().sample(handle, request)


def _start_http(service, *, token: str | None = "public-token"):
    server = create_cispo_http_server(("127.0.0.1", 0), service, service_token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_cispo_service_fixture_submit_completes(tmp_path) -> None:
    service = CispoService.from_fixture(tmp_path / "cispo.sqlite")
    submitted = service.submit(_learning_signal_request(), run_id="cispo_public_123")
    assert submitted["run_id"] == "cispo_public_123"
    assert submitted["algorithm"] == "cispo"
    assert submitted["algorithm"] != "sft"
    public_run = service.get("cispo_public_123")
    assert public_run["run_id"] == "cispo_public_123"
    assert public_run["status"] == "completed"
    assert public_run["algorithm"] == "cispo"
    events = service.optimizer_events("cispo_public_123")
    kinds = [event["event_type"] for event in events["events"]]
    assert "cispo.update.completed" in kinds or "cispo.completed" in kinds
    assert not any(kind.startswith("sft.") for kind in kinds)
    service.store.close()


def test_cispo_http_rejects_algorithm_sft(tmp_path) -> None:
    service = CispoService.from_fixture(tmp_path / "cispo.sqlite")
    server = _start_http(service)
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/runs"
        request = urllib.request.Request(
            url,
            method="POST",
            data=json.dumps({"algorithm": "sft", "config_json": _learning_signal_request()}).encode(),
            headers={"Authorization": "Bearer public-token", "Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 400
        detail = error.value.read().decode("utf-8")
        assert "cispo" in detail
    finally:
        server.shutdown()
        server.server_close()
        service.store.close()


def test_cispo_http_live_sse_follow(tmp_path) -> None:
    gate = threading.Event()
    store = JobStore(tmp_path / "cispo.sqlite")
    transport = _GatedFakeTinkerProvider(
        gate, validate_cispo=True, sample_text=_mixed_sample_text
    )
    executor = TinkerCispoExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    service = CispoService(tmp_path / "cispo.sqlite", executor=executor, background=True)
    server = _start_http(service)
    client = CispoPublicServiceClient(
        f"http://127.0.0.1:{server.server_port}", "public-token", timeout_seconds=30.0
    )
    try:
        submitted = client.submit(_learning_signal_request(), run_id="cispo_live_sse")
        assert submitted["run_id"] == "cispo_live_sse"
        assert submitted["algorithm"] == "cispo"
        assert submitted["status"] not in {"completed", "failed"}
        assert submitted["events_stream_url"] == "/v1/runs/cispo_live_sse/optimizer-events/stream"

        drop_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/runs/cispo_live_sse/optimizer-events?stream=1",
            headers={"Authorization": "Bearer public-token", "Accept": "text/event-stream"},
        )
        drop_conn = urllib.request.urlopen(drop_request, timeout=5)
        drop_conn.close()

        live_kinds: list[str] = []
        terminal = threading.Event()

        def _follow() -> None:
            for event in client.optimizer_event_stream("cispo_live_sse"):
                kind = str(event.get("event_type") or "")
                live_kinds.append(kind)
                if kind in {"cispo.completed", "cispo.failed"}:
                    terminal.set()
                    return

        follower = threading.Thread(target=_follow, daemon=True)
        follower.start()
        gate.set()
        assert terminal.wait(timeout=20)
        follower.join(timeout=5)
        assert any(
            kind in {"cispo.rollout_group.completed", "cispo.update.completed"}
            or kind.startswith("cispo.")
            for kind in live_kinds
        )
        assert "cispo.completed" in live_kinds or "cispo.failed" in live_kinds
        finished = client.get("cispo_live_sse")
        assert finished["status"] in {"completed", "failed"}
        assert finished["algorithm"] == "cispo"
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        service.store.close()


def test_cispo_http_unknown_run_stream_is_404(tmp_path) -> None:
    service = CispoService.from_fixture(tmp_path / "cispo.sqlite")
    server = _start_http(service)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/runs/missing/optimizer-events/stream",
            headers={"Authorization": "Bearer public-token", "Accept": "text/event-stream"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        service.store.close()


def test_cispo_http_requires_bearer_token(tmp_path) -> None:
    service = CispoService.from_fixture(tmp_path / "cispo.sqlite")
    server = _start_http(service)
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/runs"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url)
        assert error.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        service.store.close()


def test_use_fixture_executor_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("SYNTH_OPTIMIZERS_CISPO_FIXTURE", raising=False)
    assert not _use_fixture_executor()
    monkeypatch.setenv("SYNTH_OPTIMIZERS_CISPO_FIXTURE", "1")
    assert _use_fixture_executor()
    monkeypatch.setenv("SYNTH_OPTIMIZERS_CISPO_FIXTURE", "true")
    assert not _use_fixture_executor()


def test_cispo_service_for_serve_honors_fixture_env(tmp_path, monkeypatch) -> None:
    from synth_optimizers.providers.tinker.fake import FakeTinkerProvider

    monkeypatch.setenv("SYNTH_OPTIMIZERS_CISPO_FIXTURE", "1")
    service = cispo_service_for_serve(tmp_path / "cispo.sqlite")
    try:
        assert service.executor.sync is False
        assert service.executor.provider.credentials.api_key == "fixture"
        assert isinstance(service.executor.provider._transport, FakeTinkerProvider)
    finally:
        service.store.close()


def test_cispo_fixture_env_submit_learning_signal_recipe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_OPTIMIZERS_CISPO_FIXTURE", "1")
    service = CispoService(tmp_path / "cispo.sqlite")
    try:
        submitted = service.submit(_learning_signal_request(), run_id="cispo_env_fixture")
        assert submitted["run_id"] == "cispo_env_fixture"
        assert submitted["algorithm"] == "cispo"
        assert submitted["algorithm"] != "sft"
        public_run = service.get("cispo_env_fixture")
        assert public_run["status"] == "completed"
        assert public_run["algorithm"] == "cispo"
    finally:
        service.store.close()
