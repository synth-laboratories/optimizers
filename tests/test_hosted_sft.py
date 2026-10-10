from __future__ import annotations

from synth_optimizers.hosted import HostedOptimizerClient, OptimizerAlgorithmSlug, submit_sft
from synth_optimizers.hosted import HostedOptimizerError
import pytest


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


class SftLease:
    provider = "synth_tunnel"
    lease_id = "fixture_lease"

    def __init__(self):
        self.prepared = None
        self.submitted = None

    def container_config(self):
        return {"url": "https://gateway.example/eval", "headers": {"Authorization": "private-canary"}}

    def hosted_auth_refresh(self):
        return {"provider": "synth_tunnel_v2", "lease_id": self.lease_id,
                "connector_id": "fixture_connector", "offer_id": "fixture_offer",
                "job_id": self.prepared, "refresh_interval_seconds": 30}

    def prepare_handoff(self, job):
        self.prepared = job
        return {"job_id": job}

    def submit_once(self, payload):
        self.submitted = payload
        return {"run_id": self.prepared, "algorithm": "sft", "status": "queued",
                "events_url": "/events", "status_url": "/status", "artifact_base_url": "/artifacts"}


def test_shared_sft_keeps_flat_job_and_custody_fields_without_bearer():
    lease = SftLease()
    client = HostedOptimizerClient(api_key="test-key", register_usage=False)
    result = client.submit_sft({"run_id": "sft_stable"}, container_tunnel=lease)
    assert result.run_id == lease.prepared == "sft_stable"
    config = lease.submitted["config_json"]
    assert "container" not in config
    assert config["run_id"] == "sft_stable"
    assert config["container_url"] == "https://gateway.example/eval"
    assert config["container_auth_refresh"]["job_id"] == "sft_stable"
    assert "private-canary" not in str(lease.submitted)
    assert "container_auth_bearer_env" not in config


@pytest.mark.parametrize("config", [{"run_id": "other_job"}, {"container": {}},
                                      {"container_auth_bearer_env": "foreign"},
                                      {"container_auth_refresh": {"job_id": "foreign"}}])
def test_sft_conflicting_job_or_authority_refuses_before_offer(config):
    lease = SftLease()
    client = HostedOptimizerClient(api_key="test-key", register_usage=False)
    with pytest.raises(HostedOptimizerError):
        client.submit_sft(config, run_id="retained_job", container_tunnel=lease)
    assert lease.prepared is None and lease.submitted is None
