from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from synth_optimizers.providers.protocols import (
    ProviderError,
    ProviderCheckpoint,
    ProviderSession,
    SampleRequest,
    TrainingStepRequest,
)
from synth_optimizers.providers.tinker.sdk import (
    TinkerSdkTransport,
    _train_datum,
    _tinker_loss,
    tinker_checkpoint_name,
)
from synth_optimizers.providers.tinker.validation import is_cispo_validated, write_receipt


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def test_checkpoint_metadata_reports_expiry_without_downloading():
    from datetime import datetime, timezone
    reference = 'tinker://run/weights/state'
    checkpoint = SimpleNamespace(tinker_path=reference,
        expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc), size_bytes=100)
    rest = SimpleNamespace(list_checkpoints=lambda run: _Future(SimpleNamespace(checkpoints=[checkpoint])))
    transport = TinkerSdkTransport(SimpleNamespace(create_rest_client=lambda: rest), tinker_module=None)
    result = transport.describe_artifact(reference)
    assert result['available'] is False
    assert result['verification'] == 'provider_listing_reference_fingerprint'
    assert result['digest'].startswith('sha256:')
    assert transport.describe_artifact('tinker://run/sampler_weights/missing')['available'] is False
    with pytest.raises(ValueError):
        transport.describe_artifact('https://example.com/weights')


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
        return _Future(SimpleNamespace(metrics={'grad_norm': 1.25}))

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
        raise AssertionError("resume must not reset Adam state")

    def create_training_client_from_state_with_optimizer(self, path):
        return _Trainer()


def test_checkpoint_sampler_is_created_once_under_concurrent_calls():
    calls = []

    def create(**kwargs):
        time.sleep(0.01)
        calls.append(kwargs)
        return object()

    transport = TinkerSdkTransport(SimpleNamespace(create_sampling_client=create), tinker_module=None)
    checkpoint = ProviderCheckpoint('checkpoint', 'tinker://fixed', 24, 'sha256:fixed', 'inference')
    with ThreadPoolExecutor(max_workers=8) as pool:
        samplers = list(pool.map(lambda _: transport._sampler_for(checkpoint), range(32)))
    assert len(calls) == 1
    assert all(sampler is samplers[0] for sampler in samplers)
    other = ProviderCheckpoint('other', 'tinker://other', 25, 'sha256:other', 'inference')
    assert transport._sampler_for(other) is not samplers[0]
    assert len(calls) == 2


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


@pytest.mark.parametrize('completion', ['Seek urgent in-person care.\nDo not drive yourself.', '["move_left", "do"]'])
def test_generic_sampling_preserves_prose_and_json(monkeypatch, completion):
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client('openai/gpt-oss-20b', rank=8, seed=0)
    transport._renderer.parse_response = lambda tokens: SimpleNamespace(content=completion)
    result = transport.sample(handle, SampleRequest(request_id='preserve-text', prompt_token_ids=(1, 2), max_tokens=64))
    assert result['text'] == completion


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
    assert result['metrics']['optimizer.grad_norm'] == 1.25
    assert result['metrics']['learning_rate'] == 5e-6


def test_sdk_consumes_the_executor_cispo_payload_and_applies_reduction_weights() -> None:
    datum = _train_datum(
        _Tinker,
        {
            "token_ids": (1, 2, 3, 4),
            "loss_mask": (0, 1, 1, 0),
            "behavior_logprobs": (-0.4, -0.3, -0.2, -0.1),
            "advantage": 0.8,
            "root_rollout_weight": 0.5,
            "same_policy_weight": 0.25,
        },
        "cispo",
    )

    # The reduced sequence share is divided across trainable target tokens.
    assert datum.loss_fn_inputs["advantages"].data == pytest.approx([0.05, 0.05, 0.0])


def test_sdk_rejects_a_cispo_payload_without_advantage() -> None:
    with pytest.raises(ProviderError, match="needs an advantage"):
        _train_datum(
            _Tinker,
            {
                "token_ids": (1, 2, 3),
                "loss_mask": (0, 1, 1),
                "behavior_logprobs": (-0.3, -0.2, -0.1),
            },
            "cispo",
        )


def test_sdk_saves_training_state_with_the_resumable_api(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=4, seed=1)

    saved = transport.save_checkpoint(
        handle.session_id, step=3, kind="training_state", request_id="resume"
    )

    trainer = transport.sessions[handle.session_id]["training"]
    assert saved["provider_reference"] == "tinker://state"
    assert trainer.last_state_name == tinker_checkpoint_name("training_state", "resume")


def test_sdk_sampler_checkpoint_is_not_advertised_as_resumable(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=4, seed=1)

    saved = transport.save_checkpoint(
        handle.session_id, step=0, kind="sampler_weights", request_id="sample"
    )

    assert saved["resume_token"] is None


def test_sdk_restore_preserves_base_model_for_renderer(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    checkpoint = ProviderCheckpoint(
        "state",
        "tinker://state",
        3,
        "sha256:state",
        "training_state",
        resume_token="tinker://state",
        model_id="openai/gpt-oss-20b",
    )

    restored = transport.load_checkpoint(checkpoint, request_id="restore")

    assert restored["model_id"] == "openai/gpt-oss-20b"
    assert transport.sessions[restored["session_id"]]["model_id"] == "openai/gpt-oss-20b"


def test_resume_refuses_weights_only_sdk(monkeypatch):
    transport = _transport(monkeypatch)
    transport._service = SimpleNamespace(create_training_client_from_state=lambda path: pytest.fail("weights-only called"))
    checkpoint = ProviderCheckpoint("state", "tinker://state", 3, "sha256:state", "training_state",
                                    resume_token="tinker://state", model_id="openai/gpt-oss-20b")
    with pytest.raises(ProviderError, match="optimizer-state restore"):
        transport.load_checkpoint(checkpoint, request_id="restore")


def test_resume_matches_uninterrupted_adam_updates(monkeypatch):
    """Stateful transport double: a weights-only reload produces another result."""
    import copy
    import math

    class AdamTrainer(_Trainer):
        def __init__(self):
            self.weight, self.m, self.v, self.t = 1.0, 0.0, 0.0, 0

        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            self.gradient = sum(sum(d.loss_fn_inputs["advantages"].data) for d in data)
            return super().forward_backward(data, loss_fn, loss_fn_config)

        def optim_step(self, params):
            self.t += 1
            self.m = 0.9*self.m + 0.1*self.gradient
            self.v = 0.99*self.v + 0.01*self.gradient**2
            self.weight -= params.learning_rate*(self.m/(1-0.9**self.t))/(math.sqrt(self.v/(1-0.99**self.t))+1e-8)
            return super().optim_step(params)

    transport = _transport(monkeypatch)
    trainer = AdamTrainer()
    transport.sessions[trainer.model_id] = {"training": trainer, "model_id": "openai/gpt-oss-20b", "step": 0}
    session = ProviderSession(provider="tinker", session_id=trainer.model_id, model_id="openai/gpt-oss-20b", request_id="s")
    def update(request_id, advantage):
        return transport.train_step(session, TrainingStepRequest(request_id=request_id,
            loss_name="cispo.slime.v1", data=({"token_ids": (1,2,3), "loss_mask": (0,1,1),
                "behavior_logprobs": (0,-0.1,-0.2), "advantage": advantage, "loss_weight": 0.5},),
            metadata={"learning_rate": 0.01}))
    update("first", 1.0)
    saved = copy.deepcopy(trainer)
    update("uninterrupted", -0.3)
    expected = (trainer.weight, trainer.m, trainer.v, trainer.t)
    calls = []
    def restore(path):
        calls.append(path)
        return copy.deepcopy(saved)
    transport._service = SimpleNamespace(create_training_client_from_state_with_optimizer=restore)
    checkpoint = ProviderCheckpoint("state", "tinker://state", 1, "sha256:state", "training_state",
                                    resume_token="tinker://state", model_id="openai/gpt-oss-20b")
    restored = transport.load_checkpoint(checkpoint, request_id="resume")
    result = update("resumed", -0.3)
    state = transport.sessions[restored["session_id"]]
    assert calls == ["tinker://state"]
    actual = state["training"]
    assert (actual.weight, actual.m, actual.v, actual.t) == pytest.approx(expected)
    assert result["step"] == state["step"] == 2


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


@pytest.mark.parametrize('logprobs', [None, [], [-0.1], [float('nan'), -0.2]])
def test_sampling_refuses_missing_or_invalid_behavior_logprobs(monkeypatch, logprobs):
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client('openai/gpt-oss-20b', rank=4, seed=1)
    monkeypatch.setattr(_Sequence, 'logprobs', logprobs)
    with pytest.raises(ProviderError, match='log-probabilit'):
        transport.sample(handle, SampleRequest(request_id='strict', prompt_token_ids=(1,), max_tokens=8))


def test_sampling_does_not_substitute_raw_text_for_empty_content(monkeypatch):
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client('openai/gpt-oss-20b', rank=4, seed=1)
    transport._renderer.parse_response = lambda tokens: SimpleNamespace(content='')
    result = transport.sample(handle, SampleRequest(request_id='empty', prompt_token_ids=(1,), max_tokens=8))
    assert result['text'] == ''


def test_sampling_requires_renderer_before_provider_call(monkeypatch):
    transport = _transport(monkeypatch)
    with pytest.raises(ProviderError, match='renderer'):
        transport.sample(None, SampleRequest(request_id='missing', prompt_token_ids=(1,), max_tokens=8))
    with pytest.raises(ProviderError, match='tokenizer'):
        transport.decode([1])


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


def test_sdk_creates_one_live_sampler_for_parallel_rollouts(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client("openai/gpt-oss-20b", rank=4, seed=1)
    session = ProviderSession(
        provider="tinker",
        session_id=handle.session_id,
        model_id="openai/gpt-oss-20b",
        request_id="session",
    )
    trainer = transport.sessions[session.session_id]["training"]
    original_save = trainer.save_weights_for_sampler
    saves: list[str] = []

    def slow_save(name, ttl_seconds=None):
        saves.append(name)
        time.sleep(0.02)
        return original_save(name, ttl_seconds=ttl_seconds)

    trainer.save_weights_for_sampler = slow_save
    with ThreadPoolExecutor(max_workers=8) as pool:
        samplers = list(pool.map(lambda _: transport._sampler_for(session), range(8)))

    assert len(saves) == 1
    assert all(sampler is samplers[0] for sampler in samplers)


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


def test_forward_targets_keep_prompt_context_and_completion_alignment(monkeypatch):
    from synth_optimizers.providers.protocols import ForwardRequest
    transport = _transport(monkeypatch)
    handle = transport.create_lora_training_client('openai/gpt-oss-20b', rank=8, seed=0)
    session = ProviderSession(provider='tinker', session_id=handle.session_id,
                              model_id='openai/gpt-oss-20b', request_id='session')
    def forward(data, loss_fn):
        assert data[0].model_input == [11, 12, 21]
        assert data[0].loss_fn_inputs['target_tokens'].data == [12, 21, 22]
        assert data[0].loss_fn_inputs['weights'].data == [0.0, 1.0, 1.0]
        return SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=[{'logprobs': [-9.0, -.3, -.4]}]))
    transport.sessions[session.session_id]['training'].forward = forward
    result = transport.forward(session, ForwardRequest(request_id='f', token_ids=((11,12,21,22),),
                                                      response_masks=((False,False,True,True),)))
    assert result['logprobs'] == ((0.0,-9.0,-.3,-.4),)


@pytest.mark.parametrize('values', [[], [-.1], [float('nan'), -.1]])
def test_forward_never_fabricates_or_truncates_logprobs(values):
    from synth_optimizers.providers.tinker.sdk import _logprob_row
    with pytest.raises(ProviderError, match='finite next-token'):
        _logprob_row({'logprobs': values}, (11,21,22))


def test_renderer_profile_freezes_actual_config_and_tokens_without_training(monkeypatch):
    import hashlib
    import json
    from dataclasses import dataclass
    @dataclass
    class Config:
        name: str = 'test-renderer'
    transport = TinkerSdkTransport(SimpleNamespace(), tinker_module=None)
    prepared = []
    monkeypatch.setattr(transport, 'prepare_renderer', prepared.append)
    transport._renderer = SimpleNamespace(config=Config())
    transport._tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'exact-tokenizer'))
    monkeypatch.setattr(transport, 'tokenize_chat', lambda *a, **k: {'prompt_token_ids': [12, 34], 'stop_token_ids': [5]})
    profile = transport.renderer_profile('openai/gpt-oss-20b')
    assert prepared == ['openai/gpt-oss-20b']
    assert profile['profile_id'] == 'renderers.test-renderer.v1'
    assert profile['tokenizer_digest'] == hashlib.sha256(b'exact-tokenizer').hexdigest()
    assert profile['config_digest'] == hashlib.sha256(json.dumps({'name':'test-renderer'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert profile['stop_token_ids'] == [5]
    assert not transport.sessions
