from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from synth_optimizers.providers.protocols import ProviderCheckpoint


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tinker_harbor_tblite_cispo.py"
SPEC = importlib.util.spec_from_file_location("tblite_async_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checkpoint(reference: str, step: int) -> ProviderCheckpoint:
    return ProviderCheckpoint(
        checkpoint_id=f"checkpoint-{step}",
        provider_reference=reference,
        step=step,
        digest=f"digest-{step}",
        kind="inference",
    )


def test_gateway_routes_are_immutable_and_versioned() -> None:
    gateway = MODULE.RolloutGateway(object(), object(), 0)
    gateway.start()
    try:
        first = checkpoint("tinker://first", 3)
        digest = gateway.register_checkpoint("u04", first, 3)
        assert gateway._routes["u04"] == (first, digest, 3)
        assert gateway.register_checkpoint("u04", first, 3) == digest
        with pytest.raises(RuntimeError, match="immutable"):
            gateway.register_checkpoint("u04", checkpoint("tinker://second", 4), 4)
    finally:
        gateway.close()


def test_assemble_rejects_policy_switch_inside_trajectory() -> None:
    calls = [
        {
            "checkpoint_digest": "abc",
            "behavior_policy_version": version,
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_logprobs": [-0.1],
        }
        for version in (0, 1)
    ]
    with pytest.raises(RuntimeError, match="mixed or missing behavior policy version"):
        MODULE.assemble(calls)


def test_assemble_preserves_behavior_version() -> None:
    result = MODULE.assemble(
        [{
            "checkpoint_digest": "abc",
            "behavior_policy_version": 2,
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_logprobs": [-0.1],
        }]
    )
    assert result["behavior_policy_version"] == 2
    assert result["checkpoint_digest"] == "abc"


def test_async_pool_can_exceed_one_group_cardinality() -> None:
    source = SCRIPT.read_text()

    assert "ThreadPoolExecutor(max_workers=args.max_parallel)" in source
    assert "max_workers=min(args.cardinality, args.max_parallel)" not in source


def test_runner_tracks_actual_train_calls_and_target_time() -> None:
    source = SCRIPT.read_text()

    assert '"train_calls_completed"' in source
    assert '"target_reached_wall_seconds"' in source
    assert "train_calls += 1" in source


def test_docker_cleanup_is_scoped_to_platform_label() -> None:
    source = SCRIPT.read_text()

    assert 'f"label=synth.parent={args.platform_id}"' in source
    assert 'f"synth-harbor-tblite-{args.port}"' in source
    assert "shutil.rmtree(workspace_root, ignore_errors=True)" in source
    assert '["docker", "image", "prune", "-f", "--filter", "dangling=true"]' in source
