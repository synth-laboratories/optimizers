from __future__ import annotations

from synth_optimizers.cispo_executor import TinkerCispoExecutor
from synth_optimizers.providers.tinker import FakeTinkerProvider, TinkerAdapter, TinkerCredentials
from synth_optimizers.recipes.banking77 import cispo_recipe, evaluation_report, sft_recipe
from synth_optimizers.read_models import cispo_collections, reduce_summary, replay_equals_read_model
from synth_optimizers.runtime import JobStore
from synth_optimizers.sft_executor import TinkerSftExecutor
from synth_optimizers.sft_dataset import Example
from synth_optimizers.training_eval import evaluate_checkpoint


def test_tiny_sft_job_creates_and_evaluates_a_checkpoint(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = TinkerSftExecutor.local(store, fixture=True)
    result = executor.submit(sft_recipe(steps=1).request, job_id="sft_tiny")
    assert result["status"] == "completed"
    kinds = [event["event_type"] for event in result["events"]]
    assert "sft.checkpoint.created" in kinds
    assert "sft.checkpoint_eval.completed" in kinds
    assert "sft.heldout_eval.completed" in kinds
    body, _type, digest = store.artifact("sft_tiny", "policy_bundle.json")
    assert digest.startswith("sha256:")
    assert b"sft.tinker.v1" in body
    store.close()


def test_selection_and_heldout_requests_do_not_share_idempotency_keys() -> None:
    transport = FakeTinkerProvider(sample_text="card_arrival")
    provider = TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    checkpoint = {
        "checkpoint_id": "inference-0-reference",
        "provider_reference": "tinker://reference/inference/0",
        "step": 0,
        "digest": "sha256:" + "0" * 64,
    }
    selection = Example(
        example_id="banking77_train_00001",
        messages=(
            {"role": "user", "content": "Where is my card?"},
            {"role": "assistant", "content": "card_arrival"},
        ),
        label="card_arrival",
        text="Where is my card?",
        metadata={},
    )
    heldout = Example(
        example_id="banking77_heldout_00001",
        messages=(
            {"role": "user", "content": "Why was I charged at the cash machine?"},
            {"role": "assistant", "content": "cash_withdrawal_charge"},
        ),
        label="cash_withdrawal_charge",
        text="Why was I charged at the cash machine?",
        metadata={},
    )
    evaluate_checkpoint(provider, checkpoint, [selection])
    evaluate_checkpoint(provider, checkpoint, [heldout])
    sample_ids = [request_id for kind, request_id in transport.calls if kind == "sample"]
    assert len(sample_ids) == 2
    assert len(set(sample_ids)) == 2


def test_zero_advantage_cispo_skips_the_update(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    transport = FakeTinkerProvider(validate_cispo=True, sample_text="nope")
    executor = TinkerCispoExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    result = executor.submit(cispo_recipe(mode="canonical", updates=1).request, job_id="cispo_zero")
    assert result["status"] == "completed"
    kinds = [event["event_type"] for event in result["events"]]
    assert "cispo.zero_advantage.detected" in kinds
    assert not any(event["event_type"] == "cispo.importance_ratio.measured" for event in result["events"])
    assert ("train",) not in {call[:1] for call in transport.calls}
    store.close()


def test_cispo_performs_one_real_update_when_groups_are_mixed(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")

    def sample_text(request) -> str:
        return "order_physical_card" if int(request.seed or 0) % 2 == 0 else "lost_or_stolen_card"

    transport = FakeTinkerProvider(validate_cispo=True, sample_text=sample_text)
    executor = TinkerCispoExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    result = executor.submit(cispo_recipe(mode="learning_signal", updates=1).request, job_id="cispo_mix")
    assert result["status"] == "completed"
    kinds = [event["event_type"] for event in result["events"]]
    assert "cispo.importance_ratio.measured" in kinds
    assert "cispo.update.completed" in kinds
    assert any(kind == "train" for kind, _request in transport.calls)
    assert replay_equals_read_model(store, "cispo_mix")
    groups = cispo_collections(store, "cispo_mix", collection="rollout_groups")
    assert groups.items
    store.close()


def test_unvalidated_cispo_fails_before_paid_work(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    transport = FakeTinkerProvider(validate_cispo=False)
    executor = TinkerCispoExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    result = executor.submit(cispo_recipe().request, job_id="cispo_closed")
    assert result["status"] == "failed"
    assert "cispo.slime.v1" in str(result["error"])
    assert not any(kind == "train" for kind, _request in transport.calls)
    store.close()


def test_unvalidated_canary_is_allowed_to_train(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")

    def sample_text(request) -> str:
        return "order_physical_card" if int(request.seed or 0) % 2 == 0 else "lost_or_stolen_card"

    transport = FakeTinkerProvider(validate_cispo=False, sample_text=sample_text)
    executor = TinkerCispoExecutor(
        store,
        TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport),
        allow_unvalidated_canary=True,
    )
    request = cispo_recipe(mode="learning_signal", updates=1).request
    request["allow_unvalidated_canary"] = True
    result = executor.submit(request, job_id="cispo_canary")
    assert result["status"] == "completed"
    kinds = [event["event_type"] for event in result["events"]]
    assert "cispo.canary.started" in kinds
    assert "cispo.importance_ratio.measured" in kinds
    assert any(kind == "train" for kind, _request in transport.calls)
    store.close()


def test_cispo_restores_a_parent_sft_checkpoint(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    sft = TinkerSftExecutor.local(store, fixture=True)
    sft.submit(sft_recipe(steps=1).request, job_id="sft_parent")
    created = next(
        event for event in sft.status("sft_parent")["events"] if event["event_type"] == "sft.checkpoint.created"
    )
    transport = FakeTinkerProvider(validate_cispo=True, sample_text="nope")
    executor = TinkerCispoExecutor(
        store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport)
    )
    request = cispo_recipe(updates=1).request
    request["parent_checkpoint"] = {
        "checkpoint_id": created["payload"]["training_checkpoint_id"],
        "provider_reference": created["payload"]["training_provider_reference"],
        "resume_token": created["payload"]["resume_token"],
        "kind": "training",
        "step": created["payload"]["step"],
        "digest": created["payload"]["digest"],
    }
    result = executor.submit(request, job_id="cispo_resume")
    assert result["status"] == "completed"
    assert any(kind == "restore" for kind, _request in transport.calls)
    store.close()


def test_banking77_report_makes_regression_conspicuous() -> None:
    report = evaluation_report(
        base_accuracy=0.81,
        checkpoint_accuracy=0.50,
        heldout_accuracy=0.47,
        per_intent={"lost_or_stolen_card": {"accuracy": 0.2, "base_accuracy": 0.9, "n": 10}},
        train_loss=[1.2, 0.4],
        checkpoint_trend=[0.5],
    )
    assert report["regression_detected"] is True
    assert "0.47" in report["headline"]
    assert "regressed" in report["headline"]


def test_read_model_summary_is_bounded(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = TinkerSftExecutor.local(store, fixture=True)
    executor.submit(sft_recipe(steps=1).request, job_id="sft_summary")
    summary = reduce_summary(store, "sft_summary")
    assert summary["algorithm_id"] == "sft"
    assert summary["usage"]["cost_missing"] is True
    assert summary["usage"]["cost_usd"] is None
    store.close()
