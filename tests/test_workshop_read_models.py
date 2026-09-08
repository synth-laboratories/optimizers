from __future__ import annotations

from synth_optimizers.cispo_service import CispoService
from synth_optimizers.recipes.banking77 import cispo_recipe
from synth_optimizers.sft import SftService


def test_sft_event_page_matches_workshop_gepa_envelope(tmp_path) -> None:
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    service.submit(
        {
            "run_id": "sft_page",
            "backend": "fixture",
            "base_model": "openai/gpt-oss-20b",
            "checkpoint_steps": [1],
            "training": {"steps": 1, "batch_size": 1, "checkpoint_every_steps": 1},
        }
    )
    page = service.optimizer_events("sft_page")
    assert page["schema_version"] == "optimizer_event_page.v1"
    assert page["run_id"] == "sft_page"
    assert page["log_id"] == "sft_page"
    assert page["terminal"] is True
    assert page["next_sequence"] >= 1
    event = page["events"][0]
    assert event["attempt_id"] == "attempt-1"
    assert event["type"] == event["event_type"] == event["kind"]
    assert event["optimizer_run_id"] == "sft_page"
    assert event["algorithm_id"] == "sft"
    assert event["sequence_number"] == event["sequence"]
    metrics = [item for item in page["events"] if item["event_type"] == "sft.step.metrics"]
    assert metrics
    assert "train_loss" in metrics[0]["payload"] or "loss" in metrics[0]["payload"]
    batch = service.state_batch("sft_page", "metric_points,candidates,evaluations")
    assert batch["metric_points"]["items"]
    assert "trainLoss" in batch["metric_points"]["items"][0]["details"]
    assert batch["candidates"]["items"]
    service.store.close()


def test_cispo_workshop_collections_and_clip_identity(tmp_path) -> None:
    service = CispoService.from_fixture(tmp_path / "cispo.sqlite")
    service.submit(cispo_recipe(mode="learning_signal", updates=1).request, run_id="cispo_page")
    page = service.optimizer_events("cispo_page")
    assert page["schema_version"] == "optimizer_event_page.v1"
    kinds = [event["event_type"] for event in page["events"]]
    assert "cispo.clip.identity" in kinds
    clip = next(event for event in page["events"] if event["event_type"] == "cispo.clip.identity")
    assert clip["attempt_id"] == "attempt-1"
    assert clip["payload"]["clip"]["clip_low"] == 0.0
    assert clip["payload"]["clip"]["clip_high"] == 5.0
    batch = service.state_batch("cispo_page", "metric_points,candidates,evaluations,rollouts")
    assert batch["metric_points"]["items"]
    assert batch["rollouts"]["items"]
    assert batch["candidates"]["items"]
    service.store.close()


def test_container_evaluations_share_normal_collection_and_preserve_panel_ids(tmp_path):
    service = SftService.from_fixture(tmp_path / "sft.sqlite")
    service.submit({"run_id": "panels", "backend": "fixture", "base_model": "openai/gpt-oss-20b",
                    "checkpoint_steps": [1], "training": {"steps": 1, "batch_size": 1}})
    for role in ("selection", "final"):
        service.store.append_event("panels", "sft.child_eval.completed", {
            "eval_job_id": f"eval_{role}", "checkpoint_id": "checkpoint_same", "step": 1,
            "evaluator_id": "gsm8k", "role": role, "value": 0.5, "metric_ref": "accuracy",
            "rollouts": ["immutable-rollout-source"], "evidence_refs": ["trace-source"]}, phase="completed")
    page = service.state_batch("panels", "evaluations")["evaluations"]
    rows = [row for row in page["items"] if row.get("evaluation_id")]
    assert [row["item_id"] for row in rows] == ["eval_selection", "eval_final"]
    assert all(row["details"]["checkpointId"] == "checkpoint_same" for row in rows)
    assert [row["details"]["phase"] for row in rows] == ["selection", "final"]
    assert all(row["details"]["score"] == .5 and "rollouts" not in row["details"] for row in rows)
    service.store.close()
