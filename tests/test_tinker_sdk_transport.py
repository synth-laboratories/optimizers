from __future__ import annotations

from types import SimpleNamespace

import pytest

from synth_optimizers.providers.protocols import (
    ProviderError,
    ProviderSession,
    SampleRequest,
    TrainingStepRequest,
)
from synth_optimizers.providers.tinker.sdk import (
    TinkerSdkTransport,
    _tinker_loss,
    tinker_checkpoint_name,
)
from synth_optimizers.providers.tinker.validation import is_cispo_validated, write_receipt


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _Sequence:
    tokens = [7, 8]
    logprobs = [-0.1, -0.2]
    stop_reason = "stop"


class _Sampler:
    def sample(self, **kwargs):
        return _Future(SimpleNamespace(sequences=[_Sequence()]))


class _Trainer:
    model_id = "model-live"

    def get_tokenizer(self):
        return SimpleNamespace(name_or_path="openai/gpt-oss-20b", decode=lambda ids, skip_special_tokens=False: "hi")

    def forward(self, data, loss_fn):
        output = SimpleNamespace(
            loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-0.2, -0.3])}],
            metrics={"loss": 0.4},
        )
        return _Future(output)

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        self.last_loss = loss_fn
        self.last_config = loss_fn_config
        return _Future(SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

    def optim_step(self, params):
        self.last_lr = params.learning_rate
        return _Future(None)

    def save_state(self, name, ttl_seconds=None):
        self.last_state_name = name
        return _Future(SimpleNamespace(path="tinker://state"))

    def save_weights_for_sampler(self, name, ttl_seconds=None):
        self.last_sampler_name = name
        return _Future(SimpleNamespace(path="tinker://sampler"))


class _Service:
    def create_lora_training_client(self, **kwargs):
        return _Trainer()

    def create_sampling_client(self, model_path=None, base_model=None):
        return _Sampler()

    def create_training_client_from_state(self, path):
        return _Trainer()


class _Tinker:
    class ModelInput:
        @staticmethod
        def from_ints(tokens):
            return tokens

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AdamParams:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    class TensorData:
        def __init__(self, data, dtype, shape):
            self.data = data


def _transport(monkeypatch) -> TinkerSdkTransport:
    monkeypatch.setattr(
        "synth_optimizers.providers.tinker.sdk.create_prime_renderer",
        lambda tokenizer, model_id="": SimpleNamespace(
            get_stop_token_ids=lambda: [99],
            parse_response=lambda tokens: SimpleNamespace(content="order_physical_card"),
            render_ids=lambda messages, add_generation_prompt=False: [1, 2, 3],
            render=lambda messages: SimpleNamespace(token_ids=[1, 2, 3, 4], message_indices=[-1, -1, 1, 1], sampled_mask=[False, False, True, True]),
        ),
    )
    return TinkerSdkTransport(_Service(), tinker_module=_Tinker())


def test_sdk_maps_slime_to_tinker_cispo_and_refuses_generic_is(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=8, seed=0)
    session = ProviderSession(provider="tinker", session_id=handle.session_id, model_id="openai/gpt-oss-20b", request_id="s")
    with pytest.raises(ProviderError, match="not cispo.slime.v1"):
        transport.train_step(
            session,
            TrainingStepRequest(request_id="is", loss_name="importance_sampling", data=({},)),
        )
    result = transport.train_step(
        session,
        TrainingStepRequest(
            request_id="cispo",
            loss_name="cispo.slime.v1",
            data=({"token_ids": (1, 2, 3), "prompt_token_ids": (1,), "behavior_logprobs": (-0.1, -0.2), "advantages": [0.5, 0.5]},),
            metadata={"eps_clip": 1.0, "eps_clip_high": 4.0, "learning_rate": 5e-6},
        ),
    )
    trainer = transport.sessions[session.session_id]["training"]
    assert trainer.last_loss == "cispo"
    assert trainer.last_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    assert trainer.last_lr == 5e-6
    assert result["step"] == 1


def test_sdk_samples_and_parses_the_final_channel(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=4, seed=1)
    session = ProviderSession(provider="tinker", session_id=handle.session_id, model_id="openai/gpt-oss-20b", request_id="s")
    sampled = transport.sample(
        session,
        SampleRequest(request_id="roll", prompt_token_ids=(1, 2, 3), max_tokens=8, seed=0),
    )
    assert sampled["text"] == "order_physical_card"
    assert sampled["token_ids"] == [7, 8]


def test_sdk_live_sampler_name_advances_after_training(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=4, seed=1)
    session = ProviderSession(
        provider="tinker",
        session_id=handle.session_id,
        model_id="openai/gpt-oss-20b",
        request_id="session",
    )
    request = SampleRequest(
        request_id="rollout",
        prompt_token_ids=(1, 2, 3),
        max_tokens=8,
        seed=0,
    )

    transport.sample(session, request)
    trainer = transport.sessions[session.session_id]["training"]
    assert trainer.last_sampler_name.endswith("-live-0")

    transport.train_step(
        session,
        TrainingStepRequest(
            request_id="update",
            loss_name="cispo.slime.v1",
            data=(
                {
                    "token_ids": (1, 2, 3),
                    "prompt_token_ids": (1,),
                    "behavior_logprobs": (-0.1, -0.2),
                    "advantages": [0.5, 0.5],
                },
            ),
            metadata={"eps_clip": 1.0, "eps_clip_high": 4.0},
        ),
    )
    transport.sample(session, request)
    assert trainer.last_sampler_name.endswith("-live-1")


def test_validation_receipt_marks_cispo_only_after_a_paid_update(tmp_path) -> None:
    path = tmp_path / "cispo.json"
    write_receipt(
        path,
        {
            "model_id": "openai/gpt-oss-20b",
            "validated": True,
            "paid_update": True,
            "sft_job_id": "sft_1",
            "cispo_job_id": "cispo_1",
        },
    )
    assert is_cispo_validated(path, "gpt-oss-20b") is True
    write_receipt(path, {"model_id": "openai/gpt-oss-20b", "validated": True, "paid_update": False})
    assert is_cispo_validated(path, "openai/gpt-oss-20b") is False


def test_checkpoint_names_strip_colons_from_tinker_session_ids(monkeypatch) -> None:
    assert ":" not in tinker_checkpoint_name("inference", "327bf988-a131-5d14-9c38-ece24f71ae32:train:0-live")
    transport = _transport(monkeypatch)
    session_id = "327bf988-a131-5d14-9c38-ece24f71ae32:train:0"
    trainer = _Trainer()
    transport.sessions[session_id] = {"training": trainer, "step": 0}
    transport.save_checkpoint(session_id, step=0, kind="inference", request_id=f"{session_id}-live")
    assert trainer.last_sampler_name == "optimizers-inference-327bf988-a131-5d14-9c38-ece24f71ae32-train-0-live"


def test_tinker_loss_clip_bounds_match_slime() -> None:
    loss, config = _tinker_loss(
        TrainingStepRequest(
            request_id="x",
            loss_name="cispo.slime.v1",
            data=({},),
            metadata={"eps_clip": 1.0, "eps_clip_high": 4.0},
        )
    )
    assert loss == "cispo"
    assert config == {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
