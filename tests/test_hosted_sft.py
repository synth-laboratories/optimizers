from __future__ import annotations

from synth_optimizers.hosted import HostedOptimizerClient, OptimizerAlgorithmSlug, submit_sft


def test_submit_sft_uses_the_shared_hosted_run_contract(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test-key", register_usage=False)
    requests: list[tuple[str, str, dict[str, object]]] = []

    def fake_json_request(_client, method: str, path: str, payload=None, **_kwargs):
        requests.append((method, path, payload))
        return {
            "run_id": "sft_example",
            "status": "queued",
            "algorithm": "sft",
            "events_url": "/api/v1/optimizers/runs/sft_example/events",
            "status_url": "/api/v1/optimizers/runs/sft_example",
            "artifact_base_url": "/api/v1/optimizers/runs/sft_example/artifacts",
        }

    monkeypatch.setattr(HostedOptimizerClient, "_json_request", fake_json_request)

    response = client.submit_sft({"run_id": "sft_example"}, project_id="project_123")

    assert response.run_id == "sft_example"
    assert requests == [
        (
            "POST",
            "/api/v1/optimizers/runs",
            {
                "algorithm": OptimizerAlgorithmSlug.SFT.value,
                "config_json": {"run_id": "sft_example"},
                "project_id": "project_123",
            },
        )
    ]


def test_submit_sft_helper_delegates_to_client() -> None:
    class FakeClient:
        def submit_sft(self, config, **kwargs):
            return config, kwargs

    config, kwargs = submit_sft({"run_id": "sft_example"}, client=FakeClient(), run_id="sft_example")

    assert config == {"run_id": "sft_example"}
    assert kwargs == {"run_id": "sft_example"}
