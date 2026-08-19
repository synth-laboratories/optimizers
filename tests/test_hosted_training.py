from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from synth_optimizers.hosted import HostedOptimizerClient, HostedOptimizerError
from synth_optimizers.training import (
    BoundedTrainingCaps,
    HostedTrainingSpec,
    RequestedTraining,
    TrainingClientError,
    TrainingEvent,
    TrainingLifecycle,
    TrainingRolloutRequirement,
    apply_provider_preflight,
    apply_rollout_preflight,
    build_hosted_training_config,
    reduce_training_lifecycle,
    validate_provider_training_capabilities,
    validate_training_capabilities,
)


def _event_payload() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "training_event_v1.json").read_text())


def _capabilities() -> dict:
    from synth_optimizers.training import _canonical_sha256

    payload = {
        "schema_version": "training.rollout.capabilities.v1",
        "container_id": "banking77_classify",
        "container_digest": "sha256:container",
        "task_id": "banking77",
        "protocol_versions": ["training.rollout.request.v1"],
        "operations": ["rollout", "reward", "heartbeat"],
        "max_concurrency": 8,
        "supports_idempotency": True,
        "supports_sampler_https": True,
        "connection_modes": ["close", "keep_alive"],
    }
    payload["capability_hash"] = _canonical_sha256(payload)
    return payload


def _provider_capabilities() -> dict:
    from synth_optimizers.training import _canonical_sha256

    payload = {
        "schema_version": "training.provider_capabilities.v1",
        "provider": "tinker",
        "supported": True,
        "model": {"model_id": "gpt-oss-20b", "max_context_length": 131072},
        "algorithm": "cispo",
        "operations": [
            "sample",
            "forward",
            "train",
            "checkpoint_save",
            "checkpoint_resume",
            "cispo",
        ],
        "spend_free": True,
        "creates_training_session": False,
        "pricing": {
            "schema_version": "training.provider_pricing.v1",
            "source_url": "https://tinker-docs.thinkingmachines.ai/tinker/models.json",
            "source_entry_digest": "sha256:" + "1" * 64,
            "currency": "USD",
            "unit": "per_million_tokens",
            "prefill": 0.18,
            "cached_prefill": 0.036,
            "sample": 0.45,
            "train": 0.396,
        },
    }
    payload["capability_hash"] = _canonical_sha256(payload)
    return payload


def test_shared_event_fixture_round_trips_and_unknown_event_is_safe() -> None:
    event = TrainingEvent.from_payload(_event_payload())
    assert event.sequence == 42
    assert reduce_training_lifecycle(TrainingLifecycle.RUNNING, event) is TrainingLifecycle.RUNNING


def test_preflight_hash_is_persisted_into_submit_config() -> None:
    preflight = validate_training_capabilities(
        "http://127.0.0.1:8000",
        _capabilities(),
        TrainingRolloutRequirement(task_id="banking77", min_concurrency=4),
    )
    config = apply_rollout_preflight({"model": {"id": "gpt-oss-20b"}}, preflight)
    assert config["rollout"]["container_capability_hash"] == _capabilities()["capability_hash"]
    assert config["rollout"]["sampler_transport"] == "direct_https"


def test_tampered_capability_hash_fails_before_tunnel_or_submit() -> None:
    payload = _capabilities()
    payload["max_concurrency"] = 1
    with pytest.raises(TrainingClientError, match="hash_mismatch"):
        validate_training_capabilities(
            "http://127.0.0.1:8000",
            payload,
            TrainingRolloutRequirement(task_id="banking77"),
        )


def test_provider_preflight_is_hashed_and_persisted() -> None:
    preflight = validate_provider_training_capabilities(
        _provider_capabilities(),
        model_id="gpt-oss-20b",
        algorithm="cispo",
    )
    config = apply_provider_preflight({"model": {"id": "gpt-oss-20b"}}, preflight)
    assert config["tinker"]["capability_hash"] == _provider_capabilities()["capability_hash"]
    assert config["tinker"]["spend_free_preflight"] is True


def test_on_policy_submit_requires_preflight_and_tunnel() -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    with pytest.raises(HostedOptimizerError, match="rollout preflight"):
        client.submit_cispo({"model": {"id": "gpt-oss-20b"}})
    with pytest.raises(HostedOptimizerError, match="SynthTunnel"):
        client.submit_ppo(
            {
                "rollout": {
                    "container_capability_hash": "sha256:abc",
                },
                "tinker": {"capability_hash": "sha256:def"},
                "bounded_run_caps": {
                    "steps": 1,
                    "wall_clock_seconds": 60,
                    "cost_usd": 1,
                },
            }
        )


def test_typed_builder_materializes_exact_effective_config_and_caps() -> None:
    provider = validate_provider_training_capabilities(
        _provider_capabilities(), model_id="gpt-oss-20b", algorithm="cispo"
    )
    rollout = validate_training_capabilities(
        "http://127.0.0.1:8000",
        _capabilities(),
        TrainingRolloutRequirement(task_id="banking77"),
    )
    config = build_hosted_training_config(
        HostedTrainingSpec(
            algorithm="cispo",
            model_id="gpt-oss-20b",
            model_revision="2026-08-18",
            rank=32,
            requested_training=RequestedTraining(
                sequence_cap=1024,
                max_sample_tokens=128,
                batch_size=2,
                checkpoint_every_steps=1,
                total_training_steps=2,
            ),
            bounded_run_caps=BoundedTrainingCaps(steps=2, wall_clock_seconds=300, cost_usd=0.10),
            algorithm_config={"group_size": 2},
            repository_commits={"workshop": "workshop-sha", "optimizers": "client-sha"},
        ),
        provider=provider,
        rollout=rollout,
    )
    assert config["schema_version"] == "training.executor.v1"
    assert config["requested_training"] == config["effective_training"]
    assert config["rollout"]["task_id"] == "banking77"
    assert config["bounded_run_caps"]["cost_usd"] == 0.10
    assert config["algorithm_config"]["repository_commits"] == {
        "workshop": "workshop-sha",
        "optimizers": "client-sha",
    }


class _Lease:
    provider = "synth_tunnel"
    lease_id = "lease-1"

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def container_config(self) -> dict[str, object]:
        return {
            "url": "https://tunnel.example.invalid",
            "auth_bearer_env": "SYNTH_TUNNEL_WORKER_TOKEN",
        }


def _training_spec() -> HostedTrainingSpec:
    return HostedTrainingSpec(
        algorithm="cispo",
        model_id="gpt-oss-20b",
        model_revision="2026-08-18",
        rank=8,
        requested_training=RequestedTraining(
            sequence_cap=128,
            max_sample_tokens=16,
            batch_size=2,
            checkpoint_every_steps=1,
            total_training_steps=1,
        ),
        bounded_run_caps=BoundedTrainingCaps(steps=1, wall_clock_seconds=60, cost_usd=0.01),
        algorithm_config={"group_size": 2},
    )


def test_launch_training_owns_lease_through_wait_and_closes_once(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    provider = validate_provider_training_capabilities(
        _provider_capabilities(), model_id="gpt-oss-20b", algorithm="cispo"
    )
    rollout = validate_training_capabilities(
        "http://127.0.0.1:8000",
        _capabilities(),
        TrainingRolloutRequirement(task_id="banking77"),
    )
    lease = _Lease()
    monkeypatch.setattr(
        HostedOptimizerClient,
        "training_capabilities",
        lambda _self, **_kwargs: provider,
    )
    monkeypatch.setattr(
        HostedOptimizerClient,
        "prepare_training_rollout",
        lambda _self, *_args, **_kwargs: (rollout, lease),
    )
    monkeypatch.setattr(
        HostedOptimizerClient,
        "_submit_on_policy",
        lambda _self, *_args, **_kwargs: SimpleNamespace(run_id="run-1"),
    )
    terminal = SimpleNamespace(status="completed")
    monkeypatch.setattr(
        HostedOptimizerClient,
        "wait_for_run",
        lambda _self, *_args, **_kwargs: terminal,
    )

    launch = client.launch_training(
        _training_spec(),
        local_base_url="http://127.0.0.1:8000",
        task_id="banking77",
    )
    assert lease.closed == 0
    assert launch.wait() is terminal
    assert lease.closed == 1
    launch.close()
    assert lease.closed == 1


def test_open_synth_tunnel_forwards_wait_ready(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    lease = SimpleNamespace()
    observed: dict[str, object] = {}

    def open_tunnel(_self: HostedOptimizerClient, local_url: str, **kwargs: object) -> object:
        observed.update({"local_url": local_url, **kwargs})
        return lease

    monkeypatch.setattr(HostedOptimizerClient, "open_tunnel", open_tunnel)
    monkeypatch.setattr("synth_optimizers.hosted.SynthTunnelLease", object)

    assert (
        client.open_synth_tunnel("http://127.0.0.1:8000", wait_ready=False)
        is lease
    )
    assert observed["wait_ready"] is False


def test_launch_training_closes_lease_when_submit_fails(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    provider = validate_provider_training_capabilities(
        _provider_capabilities(), model_id="gpt-oss-20b", algorithm="cispo"
    )
    rollout = validate_training_capabilities(
        "http://127.0.0.1:8000",
        _capabilities(),
        TrainingRolloutRequirement(task_id="banking77"),
    )
    lease = _Lease()
    monkeypatch.setattr(
        HostedOptimizerClient,
        "training_capabilities",
        lambda _self, **_kwargs: provider,
    )
    monkeypatch.setattr(
        HostedOptimizerClient,
        "prepare_training_rollout",
        lambda _self, *_args, **_kwargs: (rollout, lease),
    )
    monkeypatch.setattr(
        HostedOptimizerClient,
        "_submit_on_policy",
        lambda _self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("submit failed")),
    )
    with pytest.raises(RuntimeError, match="submit failed"):
        client.launch_training(
            _training_spec(),
            local_base_url="http://127.0.0.1:8000",
            task_id="banking77",
        )
    assert lease.closed == 1


def test_resume_training_opens_fresh_lease_and_posts_checkpoint(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    provider = validate_provider_training_capabilities(
        _provider_capabilities(), model_id="gpt-oss-20b", algorithm="cispo"
    )
    rollout = validate_training_capabilities(
        "http://127.0.0.1:8000",
        _capabilities(),
        TrainingRolloutRequirement(task_id="banking77"),
    )
    lease = _Lease()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        HostedOptimizerClient,
        "training_capabilities",
        lambda _self, **_kwargs: provider,
    )
    monkeypatch.setattr(
        HostedOptimizerClient,
        "prepare_training_rollout",
        lambda _self, *_args, **_kwargs: (rollout, lease),
    )

    def request(
        _self: HostedOptimizerClient,
        method: str,
        path: str,
        payload: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        observed.update({"method": method, "path": path, "payload": payload})
        return {
            "run_id": "run-1",
            "status": "queued",
            "events_url": "/events",
            "status_url": "/status",
            "artifact_base_url": "/artifacts",
            "algorithm": "cispo",
            "attempt_id": "attempt-resume",
        }

    monkeypatch.setattr(HostedOptimizerClient, "_json_request", request)
    launch = client.resume_training(
        "run-1",
        "checkpoint:1",
        _training_spec(),
        idempotency_key="resume-1",
        local_base_url="http://127.0.0.1:8000",
        task_id="banking77",
    )
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert observed["path"] == "/api/v1/optimizers/runs/run-1/resume"
    assert payload["checkpoint_id"] == "checkpoint:1"
    assert payload["idempotency_key"] == "resume-1"
    assert payload["config_json"]["container"]["url"] == "https://tunnel.example.invalid"
    assert lease.closed == 0
    launch.close()
    assert lease.closed == 1


def _saved_lora_payload(*, status: str = "ready") -> dict:
    return {
        "schema_version": "saved_lora_checkpoint.v1",
        "checkpoint_id": "11111111-1111-1111-1111-111111111111",
        "org_id": "22222222-2222-2222-2222-222222222222",
        "owner_user_id": "33333333-3333-3333-3333-333333333333",
        "visibility": "private",
        "name": "banking77-cispo",
        "description": "saved adapter",
        "provider": "tinker",
        "checkpoint_kind": "inference",
        "optimizer_algorithm": "cispo",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "source_checkpoint_id": "policy-4",
        "provider_checkpoint_reference": "tinker://run-1/policy-4",
        "lineage": {
            "optimizer_algorithm": "cispo",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "source_checkpoint_id": "policy-4",
            "provider_checkpoint_reference": "tinker://run-1/policy-4",
        },
        "base_model": "openai/gpt-oss-20b",
        "status": status,
        "storage": {
            "backend": "wasabi",
            "bucket": "learning-models",
            "key": "optimizer-loras/test/checkpoint.tar",
            "version": None,
            "etag": "etag",
            "sha256": "ab" * 32,
            "size_bytes": 42,
            "content_type": "application/x-tar",
        },
        "tags": ["banking77", "cispo"],
        "metadata": {},
    }


def test_saved_lora_search_builds_filters_and_decodes_page(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    observed = {}

    def request(_self, method, path, *_args, **_kwargs):
        observed.update(method=method, path=path)
        return {"items": [_saved_lora_payload()], "total": 1, "limit": 20, "offset": 0}

    monkeypatch.setattr(HostedOptimizerClient, "_json_request", request)
    page = client.search_saved_lora_checkpoints(
        query="banking", scope="mine", provider="tinker", tags=["cispo"], limit=20
    )
    assert page.items[0].name == "banking77-cispo"
    assert observed["method"] == "GET"
    assert "q=banking" in observed["path"]
    assert "tags=cispo" in observed["path"]


def test_saved_loras_for_run_decodes_bidirectional_lineage(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    observed = {}

    def request(_self, method, path, *_args, **_kwargs):
        observed.update(method=method, path=path)
        return {
            "schema_version": "saved_lora_checkpoint.run_page.v1",
            "run": {
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "optimizer_algorithm": "cispo",
                "status": "succeeded",
            },
            "items": [_saved_lora_payload()],
            "counts": {"total": 1, "inference": 1, "training": 0},
            "total": 1,
            "limit": 100,
            "offset": 0,
        }

    monkeypatch.setattr(HostedOptimizerClient, "_json_request", request)
    page = client.list_saved_lora_checkpoints_for_run("run-1")
    assert page.optimizer_algorithm == "cispo"
    assert page.items[0].lineage.source_checkpoint_id == "policy-4"
    assert page.counts["inference"] == 1
    assert observed["method"] == "GET"
    assert observed["path"].startswith(
        "/api/v1/optimizers/runs/run-1/saved-checkpoints?"
    )


def test_run_outputs_lists_result_artifacts_and_model_checkpoints(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    monkeypatch.setattr(
        HostedOptimizerClient,
        "_json_request",
        lambda *_args, **_kwargs: {
            "schema_version": "optimizer.run_outputs.v1",
            "run": {
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "optimizer_algorithm": "cispo",
                "status": "succeeded",
            },
            "result": {"selected_checkpoint": "policy-4"},
            "artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "run_id": "run-1",
                    "artifact_name": "result.json",
                    "content_type": "application/json",
                    "size_bytes": 42,
                    "sha256": "ab" * 32,
                    "storage_backend": "s3",
                    "uri": "s3://bucket/result.json",
                    "download_path": "/api/v1/optimizers/runs/run-1/artifacts/result.json",
                    "metadata": {},
                }
            ],
            "model_checkpoints": [_saved_lora_payload()],
            "counts": {"artifacts": 1, "model_checkpoints": 1},
        },
    )
    outputs = client.run_outputs("run-1")
    assert outputs.result == {"selected_checkpoint": "policy-4"}
    assert outputs.artifacts[0].artifact_name == "result.json"
    assert outputs.model_checkpoints[0].lineage.run_id == "run-1"


def test_hosted_training_model_catalog_is_typed(monkeypatch) -> None:
    client = HostedOptimizerClient(api_key="test", register_usage=False)
    monkeypatch.setattr(
        HostedOptimizerClient,
        "_json_request",
        lambda *_args, **_kwargs: {
            "catalog_revision": "hosted-training-models.test",
            "live_preflight_required": True,
            "models": [
                {
                    "model_id": "openai/gpt-oss-20b",
                    "label": "GPT-OSS 20B",
                    "provider": "tinker",
                    "provider_revision": "default",
                    "architecture": "dense",
                    "max_context_length": 131072,
                    "rank": {"default": 8, "minimum": 1, "maximum": 4096},
                    "algorithms": {"cispo": {"status": "preview"}},
                }
            ],
            "total": 1,
        },
    )
    catalog = client.hosted_training_models(algorithm="cispo")
    assert catalog.live_preflight_required
    assert catalog.models[0].algorithms["cispo"]["status"] == "preview"
