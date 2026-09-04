from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from synth_optimizers.providers.protocols import ProviderCheckpoint, ProviderSession

SCRIPT = Path(__file__).parents[1] / "docs/e2e/tinker_gradient_canary.py"
SPEC = importlib.util.spec_from_file_location("tinker_gradient_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canary)


class MovingProvider:
    def __init__(self, *, restore_offset: float = 0.0) -> None:
        self.shift = 0.0
        self.restore_offset = restore_offset
        self.training_requests = []

    def create_session(self, model_id, *, rank, seed, request_id):
        return ProviderSession("tinker", "fresh", model_id, request_id)

    def tokenize_chat(self, messages, *, add_generation_prompt):
        assert add_generation_prompt
        return {"prompt_token_ids": (10, 11, 12)}

    def sample(self, session, request):
        return SimpleNamespace(token_ids=(20, 21), logprobs=(-1.0, -2.0), text="cash_withdrawal")

    def forward(self, session, request):
        offset = self.restore_offset if session.session_id == "restored" else self.shift
        # One value per next-token prediction; only the final two are selected.
        return SimpleNamespace(logprobs=((-9.0, -9.0, -1.0 + offset, -2.0 + offset),))

    def train_step(self, session, request):
        self.training_requests.append(request)
        self.shift = 0.25
        return SimpleNamespace(step=1, metrics={"loss:sum": -0.5})

    def save_checkpoint(self, session, *, step, kind, request_id):
        reference = f"tinker://{kind}/{step}/{request_id}"
        return ProviderCheckpoint(
            checkpoint_id=f"{kind}-{step}",
            provider_reference=reference,
            step=step,
            digest=f"sha256:{kind}-{step}",
            kind=kind,
            resume_token=reference,
        )

    def restore_session(self, checkpoint, *, request_id):
        self.restore_offset = self.shift
        return ProviderSession("tinker", "restored", "model", request_id)


def test_canary_proves_executor_shaped_advantage_movement_and_restore() -> None:
    provider = MovingProvider()
    receipt = canary.run_canary(provider, run_id="proof")

    assert receipt["passed"] is True
    assert receipt["forward"]["target_sequence_sum_change"] == pytest.approx(0.5)
    request = provider.training_requests[0]
    assert request.loss_name == "cispo.slime.v1"
    assert request.data[0]["advantage"] == 1.0
    assert "advantages" not in request.data[0]
    assert request.data[0]["root_rollout_weight"] == 1.0
    assert request.data[0]["same_policy_weight"] == 1.0
    assert receipt["checkpoints"]["post_state"]["kind"] == "training_state"


class NoMovementProvider(MovingProvider):
    def train_step(self, session, request):
        self.training_requests.append(request)
        return SimpleNamespace(step=1, metrics={"loss:sum": 0.0})


def test_canary_refuses_an_optimizer_step_with_no_parameter_movement() -> None:
    with pytest.raises(RuntimeError, match="parameters_moved"):
        canary.run_canary(NoMovementProvider(), run_id="no-op")


def test_env_loader_reads_only_tinker_values(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("TINKER_API_KEY='secret'\nUNRELATED=do-not-load\n", encoding="utf-8")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED", raising=False)

    canary._load_provider_environment(env)

    assert canary.os.environ["TINKER_API_KEY"] == "secret"
    assert "UNRELATED" not in canary.os.environ
