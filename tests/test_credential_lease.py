from pathlib import Path

from synth_optimizers.eval.runner import WorkerManifest


def test_worker_manifest_keeps_workshop_proxy_route(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        """
        {
          "schema_version": "eval.worker-manifest.v1",
          "run_id": "opt_eval_test",
          "recipe_id": "eval.craftax.llm-policy.smoke.v1",
          "home": "/tmp/eval-home",
          "candidate_set_path": "/tmp/candidates",
          "credential_mode": "workshop_proxy",
          "inference_url": "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai",
          "provider_routes": {
            "openai": "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai/chat/completions",
            "openai_base": "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai",
            "api_key_sentinel": "workshop-proxy",
            "extra_hosts": ["host.docker.internal:host-gateway"]
          }
        }
        """,
        encoding="utf-8",
    )
    manifest = WorkerManifest.load(path)
    assert manifest.workshop_proxy
    assert "host.docker.internal" in (manifest.inference_url or "")
    assert "api.openai.com" not in (manifest.inference_url or "")


def test_worker_manifest_rejects_openai_origin(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        """
        {
          "schema_version": "eval.worker-manifest.v1",
          "run_id": "opt_eval_test",
          "recipe_id": "eval.craftax.llm-policy.smoke.v1",
          "home": "/tmp/eval-home",
          "candidate_set_path": "/tmp/candidates",
          "credential_mode": "workshop_proxy",
          "inference_url": "https://api.openai.com/v1"
        }
        """,
        encoding="utf-8",
    )
    try:
        WorkerManifest.load(path)
    except Exception as exc:
        assert "Workshop container proxy" in str(exc)
    else:
        raise AssertionError("api.openai.com must fail closed")


def test_worker_manifest_defaults_container_extra_hosts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        """
        {
          "schema_version": "eval.worker-manifest.v1",
          "run_id": "opt_eval_test",
          "recipe_id": "eval.craftax.llm-policy.smoke.v1",
          "home": "/tmp/eval-home",
          "candidate_set_path": "/tmp/candidates",
          "credential_mode": "workshop_proxy",
          "inference_url": "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai"
        }
        """,
        encoding="utf-8",
    )
    manifest = WorkerManifest.load(path)
    assert manifest.extra_hosts() == ["host.docker.internal:host-gateway"]
