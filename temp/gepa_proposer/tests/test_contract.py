from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gepa_proposer import app as app_mod
from gepa_proposer.optimizer_client import OptimizerClient
from gepa_proposer.episode import build_run_request, parse_episode, program_for_task
from gepa_proposer.fixtures import by_task_id
from gepa_proposer.scoring import score_episode
from gepa_proposer.store import JsonStore


# The catalog in load order. Archive-derived rows (checkpointed seeds minted by
# mint_checkpoint_fixtures.py) sit next to the seed-only / reconstructed rows of
# the same family.
CATALOG_TASK_IDS = [
    "train:0",
    "train:1",
    "train:2",
    "train:3",
    "train:4",
    "train:5",
    "healthbench:0",
    "healthbench:1",
    "healthbench:2",
    "healthbench:3",
    "healthbench:4",
    "crafter:0",
    "crafter:1",
    "crafter:2",
    "crafter:3",
    "crafter:4",
    "tau2:0",
    "tau2:1",
    "tau2:2",
    "minigrid:0",
    "minigrid:1",
    "minigrid:2",
    "officeqa:0",
]
CATALOG_LABELS = [
    "banking77-fresh",
    "banking77-first-checkpoint",
    "banking77-mature",
    "banking77-gen3",
    "banking77-async-fresh",
    "banking77-async-first-checkpoint",
    "healthbench-fresh",
    "healthbench-first-checkpoint",
    "healthbench-mature",
    "healthbench-openai-scored-seed",
    "healthbench-accepted-frontier",
    "crafter-fresh",
    "crafter-first-checkpoint",
    "crafter-mature",
    "crafter-archive-fresh",
    "crafter-archive-mature",
    "tau2-retail-fresh",
    "tau2-retail-first-checkpoint",
    "tau2-retail-mature",
    "minigrid-empty-fresh",
    "minigrid-empty-first-checkpoint",
    "minigrid-empty-mature",
    "officeqa-fresh",
]
# task_id -> (source run, generation, candidate count) for every fixture whose
# cursor came out of a real GEPA workspace archive at a generation_start.
ARCHIVE_FIXTURES = {
    "train:3": ("gepa_24b32fd4c2e74e96aed7ba747dcd5c55", 3, 19),
    "train:4": ("banking77_gepa_async_t50_mb20_h100_735a9c29", 0, 1),
    "train:5": ("banking77_gepa_async_t50_mb20_h100_735a9c29", 1, 7),
    "crafter:3": ("crafter_gepa_public_0fbad055", 0, 1),
    "crafter:4": ("crafter_gepa_public_0fbad055", 2, 4),
    "tau2:1": ("tau2_retail_gepa_20260819_long", 1, 3),
    "tau2:2": ("tau2_retail_gepa_20260819_long", 6, 17),
    "minigrid:1": ("minigrid_empty_gepa_20260819", 1, 3),
    "minigrid:2": ("minigrid_empty_gepa_20260819", 4, 12),
}


def client(tmp_path: Path, optimizer: Any | None = None) -> TestClient:
    app_mod.STORE = JsonStore(tmp_path / "state")
    app_mod.OPTIMIZER = optimizer if optimizer is not None else app_mod.OptimizerClient(base_url="")
    return TestClient(app_mod.app)


def test_health_and_taskset_includes_healthbench(tmp_path: Path) -> None:
    c = client(tmp_path)
    health = c.get("/health").json()
    assert health["status"] == "ok"
    assert health["contract_version"] == "2026-05-28"
    assert health["episode_requires_optimizer"] is True
    meta = c.get("/metadata").json()
    assert meta["metadata"]["optimizer_contracts"]["gepa"]["version"] == "synth_optimizers.gepa.v2"
    assert meta["metadata"]["optimizer_contracts"]["gepa"]["rollout_route"] == "/rollouts"
    assert meta["capabilities"]["rollout_modes"] == ["blocking", "async"]
    taskset = c.get("/taskset").json()
    assert taskset["splits"]["train"] == 23
    body = c.post("/taskset/tasks", json={"split": "train", "task_ids": CATALOG_TASK_IDS}).json()
    labels = [row["label"] for row in body["tasks"]]
    assert labels == CATALOG_LABELS
    rows = {row["task_id"]: row for row in body["tasks"]}
    assert rows["healthbench:0"]["downstream"]["id"] == "healthbench2"
    assert rows["healthbench:0"]["downstream"]["candidate_field"] == "system_prompt"
    assert rows["healthbench:3"]["downstream"]["policy"]["provider"] == "openai"
    assert rows["crafter:0"]["downstream"]["id"] == "crafter"
    assert rows["crafter:0"]["downstream"]["candidate_field"] == "react_system_prompt"
    assert rows["tau2:0"]["downstream"]["id"] == "tau2"
    assert rows["tau2:0"]["downstream"]["candidate_field"] == "domain_policy"
    assert rows["minigrid:0"]["downstream"]["id"] == "minigrid"
    assert rows["minigrid:0"]["downstream"]["candidate_field"] == "system_prompt"
    assert rows["officeqa:0"]["downstream"]["id"] == "officeqa"
    # Checkpointed fixtures minted from real archives route to the same inner
    # container as their seed-only / reconstructed siblings.
    assert rows["train:3"]["downstream"]["id"] == "banking77"
    assert rows["train:5"]["downstream"]["candidate_field"] == "stage2_system"
    assert rows["crafter:4"]["downstream"]["candidate_field"] == "react_system_prompt"
    assert rows["tau2:2"]["downstream"]["candidate_field"] == "domain_policy"
    assert rows["minigrid:2"]["downstream"]["id"] == "minigrid"


def test_harbor_compatibility_and_dataset(tmp_path: Path) -> None:
    c = client(tmp_path)
    harbor = c.get("/compatibility", params={"target": "harbor_proxy"}).json()
    assert harbor["target"] == "harbor_proxy"
    assert harbor["supported"] is True
    dataset = c.get("/dataset").json()
    assert dataset["splits"]["train"] == 23
    rows = c.post("/dataset/rows", json={"split": "train", "task_ids": ["healthbench:1"]}).json()
    assert rows["rows"][0]["task_id"] == "healthbench:1"
    program = c.get("/program", params={"task_id": "healthbench:1"}).json()
    assert program["target_modules"][0]["candidate_field"] == "system_prompt"


def test_two_forks_are_byte_identical(tmp_path: Path) -> None:
    c = client(tmp_path)
    info = c.post("/taskset/tasks", json={"split": "train", "task_ids": ["train:2"]}).json()
    fixture_hash = info["tasks"][0]["cursor_sha256"]
    a = c.post("/rollout", json={"task_id": "train:2", "submission_mode": "async", "policy": {"reasoning_effort": "low"}}).json()
    b = c.post("/rollout", json={"task_id": "train:2", "submission_mode": "async", "policy": {"reasoning_effort": "medium"}}).json()
    ra = c.get(f"/rollouts/{a['rollout_id']}").json()
    rb = c.get(f"/rollouts/{b['rollout_id']}").json()
    assert ra["pre_fork_cursor"]["candidates"] == rb["pre_fork_cursor"]["candidates"]
    assert ra["pre_fork_cursor"]["train_rows"] == rb["pre_fork_cursor"]["train_rows"]
    assert ra["run_id"] != rb["run_id"]
    from gepa_proposer.app import _hash_cursor

    assert _hash_cursor(ra["pre_fork_cursor"]) == _hash_cursor(rb["pre_fork_cursor"])
    assert _hash_cursor(ra["pre_fork_cursor"]) == fixture_hash


def test_pause_resume_survives_process_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    app_mod.STORE = JsonStore(state)
    app_mod.OPTIMIZER = app_mod.OptimizerClient(base_url="")
    c = TestClient(app_mod.app)
    started = c.post("/rollout", json={"task_id": "train:1", "submission_mode": "async"}).json()
    paused = c.post(f"/rollouts/{started['rollout_id']}/pause").json()
    assert paused["status"] == "paused"
    assert paused["cursor"]["phase"] == "paused"

    app_mod.STORE = JsonStore(state)
    app_mod.OPTIMIZER = app_mod.OptimizerClient(base_url="")
    c2 = TestClient(app_mod.app)
    restored = c2.get(f"/rollouts/{started['rollout_id']}").json()
    assert restored["status"] == "paused"
    resumed = c2.post(f"/rollouts/{started['rollout_id']}/resume").json()
    assert resumed["status"] == "running"
    assert resumed["paused"] is False


def test_get_ingests_terminal_gepa_run_after_proposer_restart(tmp_path: Path) -> None:
    class TerminalFake(FakeOptimizer):
        async def aget_run(self, run_id: str) -> dict[str, Any]:
            return {"run_id": run_id, "status": "succeeded"}

    fake = TerminalFake()
    c = client(tmp_path, optimizer=fake)
    spec = by_task_id("train:1")
    cursor = copy.deepcopy(spec["cursor"])
    app_mod.STORE.put_rollout(
        {
            "rollout_id": "r-live",
            "task_id": "train:1",
            "run_id": "episode-r-live",
            "status": "running",
            "success_status": "running",
            "arm": {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
            "episode": {"proposer_rounds": 1, "skip_heldout": False},
            "inner_url": "http://127.0.0.1:8765",
            "downstream": spec.get("downstream"),
            "pre_fork_cursor": copy.deepcopy(cursor),
            "cursor": cursor,
            "optimizer_run_id": "opt-ingested",
            "created_at": "2026-08-20T00:00:00Z",
            "updated_at": "2026-08-20T00:00:00Z",
            "paused": False,
        }
    )
    got = c.get("/rollouts/r-live").json()
    assert got["status"] == "completed"
    assert got["reward"] is not None
    assert got["optimizer_status"] == "succeeded"


def test_checkpoint_and_resume_async(tmp_path: Path) -> None:
    c = client(tmp_path)
    started = c.post("/rollout", json={"task_id": "train:0", "submission_mode": "async"}).json()
    pin = c.post(
        f"/rollouts/{started['rollout_id']}/checkpoints",
        json={"checkpoint_id": "pin-fresh"},
    ).json()
    assert pin["retained"] is True
    child = c.post(
        f"/rollouts/{started['rollout_id']}/resume_async",
        json={"checkpoint_id": "pin-fresh", "target_rollout_id": "child-1"},
    ).json()
    assert child["rollout_id"] == "child-1"
    loaded = c.get("/rollouts/child-1").json()
    assert loaded["parent_rollout_id"] == started["rollout_id"]
    assert loaded["cursor"]["candidates"] == pin["cursor"]["candidates"]


def test_sync_rollout_without_optimizer_is_503(tmp_path: Path) -> None:
    c = client(tmp_path)
    result = c.post(
        "/rollout",
        json={
            "task_id": "train:0",
            "submission_mode": "sync",
            "policy": {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        },
    )
    assert result.status_code == 503
    assert "GEPA_SERVICE_URL" in result.json()["detail"]


def test_v0_rejects_candidate_overlay(tmp_path: Path) -> None:
    c = client(tmp_path)
    result = c.post(
        "/rollout",
        json={
            "task_id": "train:0",
            "submission_mode": "async",
            "candidate": {"stage2_system": "a new prompt"},
        },
    )
    assert result.status_code == 422


class FakeOptimizer:
    live = True

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.run_id = "opt-fake-1"
        self.in_flight = 0
        self.max_in_flight = 0

    def create_run(self, body: dict[str, Any]) -> dict[str, Any]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.bodies.append(copy.deepcopy(body))
        return {"run_id": f"{self.run_id}-{len(self.bodies)}", "status": "running"}

    def wait_until_terminal(self, run_id: str, *, timeout_seconds: float, poll_seconds: float = 2.0) -> dict[str, Any]:
        time.sleep(0.05)
        self.in_flight = max(0, self.in_flight - 1)
        return {"run_id": run_id, "status": "succeeded"}

    def get_state(self, run_id: str) -> dict[str, Any]:
        spec = by_task_id("train:1")
        cursor = copy.deepcopy(spec["cursor"])
        parent = cursor["candidates"][0]
        example_ids = [
            str(row.get("task_id") or row.get("example_id"))
            for row in (cursor.get("minibatch_rows") or [])[:3]
        ]
        child = copy.deepcopy(parent)
        child["candidate_id"] = f"episode-child-{run_id}"
        child["parent_id"] = parent.get("candidate_id")
        child["train_scores"] = [
            {"example_id": eid, "task_id": eid, "reward": 1.0} for eid in example_ids if eid
        ]
        child["minibatch_scores"] = child["train_scores"]
        child["heldout_reward"] = 0.8
        child["heldout_scores"] = [
            {"example_id": "heldout:0", "task_id": "heldout:0", "reward": 0.8}
        ]
        parent["heldout_reward"] = 0.5
        parent["heldout_scores"] = [
            {"example_id": "heldout:0", "task_id": "heldout:0", "reward": 0.5}
        ]
        cursor["candidates"].append(child)
        cursor["best_candidate_id"] = child["candidate_id"]
        return {"run_id": run_id, "status": "succeeded", "cursor": cursor}

    def pause(self, run_id: str, timeout_seconds: int = 1800) -> Any:
        return {"run_id": run_id, "status": "paused"}

    def resume(self, run_id: str) -> Any:
        return {"run_id": run_id, "status": "running"}

    def pin(self, run_id: str, checkpoint_id: str | None = None) -> Any:
        return {"run_id": run_id, "checkpoint_id": checkpoint_id}


def test_sync_rollout_scores_terminal_optimizer_cursor(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BANKING77_URL", "http://127.0.0.1:8765")
    fake = FakeOptimizer()
    c = client(tmp_path, optimizer=fake)
    result = c.post(
        "/rollout",
        json={
            "task_id": "train:1",
            "submission_mode": "sync",
            "policy": {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
            "episode": {"proposer_rounds": 3},
        },
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "completed"
    assert body["optimizer_run_id"].startswith("opt-fake-1")
    assert "reward_info" in body
    assert body["reward_info"]["outcome_reward"] == body["reward"]
    names = {row["objective"] for row in body["objective_scores"]}
    assert names == {"train_exploration", "train_exploitation", "eval_uplift"}
    metrics = {row["objective"]: row["value"] for row in body["objective_scores"]}
    assert metrics["eval_uplift"] == pytest.approx(0.3)
    assert body["reward"] == pytest.approx(
        metrics["train_exploration"]
        + metrics["train_exploitation"]
        + metrics["eval_uplift"]
    )
    assert body["reward_details"]["heldout_evaluated"] is True
    assert any(cid.startswith("episode-child") for cid in body["reward_details"]["episode_candidate_ids"])
    created = fake.bodies[0]
    assert created["proposer"]["reasoning_effort"] == "low"
    assert created["stop_conditions"] == [
        {"kind": "episode", "proposer_rounds": 3, "skip_heldout": False}
    ]
    assert created["container_url"] == "http://127.0.0.1:8765"
    assert created["fixture"]["schema"] == "gepa_cursor_fixture.v1"


def test_parallel_async_rollouts_share_one_container(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BANKING77_URLS", "http://127.0.0.1:8765,http://127.0.0.1:8766")
    fake = FakeOptimizer()
    c = client(tmp_path, optimizer=fake)
    started = [
        c.post(
            "/rollout",
            json={
                "task_id": "train:1",
                "submission_mode": "async",
                "policy": {"reasoning_effort": effort},
            },
        ).json()
        for effort in ("low", "medium")
    ]
    deadline = time.time() + 30
    records = []
    while time.time() < deadline:
        records = [c.get(f"/rollouts/{row['rollout_id']}").json() for row in started]
        if all(row["status"] in {"completed", "failed"} for row in records):
            break
        time.sleep(0.05)
    assert all(row["status"] == "completed" for row in records), records
    urls = {body["container_url"] for body in fake.bodies}
    assert urls == {"http://127.0.0.1:8765", "http://127.0.0.1:8766"}
    efforts = {body["proposer"]["reasoning_effort"] for body in fake.bodies}
    assert efforts == {"low", "medium"}
    assert fake.max_in_flight >= 2


def test_failed_optimizer_run_does_not_yield_a_reward(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BANKING77_URL", "http://127.0.0.1:8765")

    class FailedOptimizer(FakeOptimizer):
        def wait_until_terminal(self, run_id: str, *, timeout_seconds: float, poll_seconds: float = 2.0) -> dict[str, Any]:
            return {"run_id": run_id, "status": "failed"}

    c = client(tmp_path, optimizer=FailedOptimizer())
    result = c.post(
        "/rollout",
        json={"task_id": "train:1", "submission_mode": "sync", "policy": {"reasoning_effort": "low"}},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "failed"
    assert "reward" not in body
    assert body.get("success_status") != "succeeded"


def test_build_run_request_carries_delta_horizon() -> None:
    spec = by_task_id("train:1")
    episode = parse_episode({"episode": {"proposer_rounds": 3, "max_rollouts": 200}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "medium"},
        episode=episode,
        container_url="http://127.0.0.1:8765",
    )
    assert body["proposer"]["reasoning_effort"] == "medium"
    assert body["stop_conditions"][0]["skip_heldout"] is False
    assert body["stop_conditions"][0]["max_rollouts"] == 200
    assert body["task_pools"]["pareto"]
    assert body["taskset"]["train_ids"] == body["task_pools"]["pareto"]
    assert body["advanced"]["budgets"]["max_heldout_rollouts"] == 8000
    assert body["policy"]["model"] == "gpt-4.1-nano"


def test_healthbench_run_request_uses_system_prompt_policy() -> None:
    spec = by_task_id("healthbench:1")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:8114",
    )
    assert body["policy"]["provider"] == "groq"
    assert body["policy"]["model"] == "llama-3.1-8b-instant"
    assert body["policy"]["max_tokens"] == 1536
    assert "system_prompt" in (spec["cursor"]["candidates"][0]["payload"])
    assert body["stop_conditions"][0]["skip_heldout"] is False
    # Backfilled from healthbench_groq_gepa_aug13i checkpoint 355 (the gen 1
    # generation_start this fixture was reconstructed from), not `{}`.
    assert body["fixture"]["checkpoint"]["usage"]["prompt_tokens"] == 3688916
    assert body["fixture"]["checkpoint"]["snapshot"]["usage"]["prompt_tokens"] == 3688916
    assert body["fixture"]["checkpoint"]["snapshot"]["usage"]["rollout_calls"] == 312


def test_healthbench_openai_run_request_uses_openai_inner_policy() -> None:
    spec = by_task_id("healthbench:3")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:8114",
    )
    assert spec["downstream"]["id"] == "healthbench2"
    assert body["policy"]["provider"] == "openai"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert body["policy"]["max_tokens"] == 1536
    assert len(spec["cursor"]["candidates"]) == 1
    assert spec["cursor"]["candidates"][0].get("heldout_reward") is not None
    assert len(body["taskset"]["train_ids"]) == 2
    assert len(body["taskset"]["heldout_ids"]) == 2


def test_crafter_run_request_uses_react_system_prompt() -> None:
    spec = by_task_id("crafter:1")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:20055",
    )
    assert spec["downstream"]["id"] == "crafter"
    assert spec["downstream"]["candidate_field"] == "react_system_prompt"
    assert body["policy"]["provider"] == "openai"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert "react_system_prompt" in (spec["cursor"]["candidates"][0]["payload"])
    assert len(body["taskset"]["train_ids"]) == 8
    assert len(body["taskset"]["heldout_ids"]) == 8
    mature = by_task_id("crafter:2")
    heldout = [
        candidate.get("heldout_reward")
        for candidate in mature["cursor"]["candidates"]
        if candidate.get("heldout_reward") is not None
    ]
    assert heldout


def test_tau2_run_request_uses_domain_policy() -> None:
    spec = by_task_id("tau2:0")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:8774",
    )
    assert spec["downstream"]["id"] == "tau2"
    assert spec["downstream"]["candidate_field"] == "domain_policy"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert "domain_policy" in (spec["cursor"]["candidates"][0]["payload"])
    assert len(body["taskset"]["train_ids"]) == 20
    assert len(body["taskset"]["heldout_ids"]) == 16


def test_minigrid_run_request_uses_system_prompt() -> None:
    spec = by_task_id("minigrid:0")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:8769",
    )
    assert spec["downstream"]["id"] == "minigrid"
    assert spec["downstream"]["candidate_field"] == "system_prompt"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert "system_prompt" in (spec["cursor"]["candidates"][0]["payload"])
    assert len(body["taskset"]["train_ids"]) == 8
    assert len(body["taskset"]["heldout_ids"]) == 4


def test_officeqa_run_request_uses_treasury_policy() -> None:
    spec = by_task_id("officeqa:0")
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url="http://127.0.0.1:8120",
    )
    assert spec["downstream"]["id"] == "officeqa"
    assert body["policy"]["model"] == "gpt-4.1"
    assert body["policy"]["max_tokens"] == 256
    assert "system_prompt" in (spec["cursor"]["candidates"][0]["payload"])
    assert len(body["taskset"]["train_ids"]) == 24
    assert len(body["taskset"]["heldout_ids"]) == 16


def _request_for(task_id: str, container_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = by_task_id(task_id)
    episode = parse_episode({"episode": {"proposer_rounds": 1}})
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=episode,
        container_url=container_url,
    )
    return spec, body


@pytest.mark.parametrize("task_id", ["train:3", "train:4", "train:5"])
def test_banking77_checkpointed_fixtures_send_banking77_inner_policy(task_id: str) -> None:
    spec, body = _request_for(task_id, "http://127.0.0.1:8765")
    assert spec["downstream"]["id"] == "banking77"
    assert spec["downstream"]["url_env"] == "BANKING77_URL"
    assert spec["downstream"]["url_pool_env"] == "BANKING77_URLS"
    assert spec["downstream"]["candidate_field"] == "stage2_system"
    assert body["policy"]["provider"] == "openai"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert body["policy"]["max_tokens"] == 16
    assert "stage2_system" in spec["cursor"]["candidates"][0]["payload"]
    assert len(body["taskset"]["train_ids"]) == 100
    assert len(body["taskset"]["heldout_ids"]) == 200
    assert body["taskset"]["train_ids"] == body["task_pools"]["pareto"]
    assert body["stop_conditions"][0]["skip_heldout"] is False


@pytest.mark.parametrize("task_id", ["tau2:1", "tau2:2"])
def test_tau2_checkpointed_fixtures_send_domain_policy(task_id: str) -> None:
    spec, body = _request_for(task_id, "http://127.0.0.1:8774")
    assert spec["downstream"]["id"] == "tau2"
    assert spec["downstream"]["url_pool_env"] == "TAU2_URLS"
    assert spec["downstream"]["candidate_field"] == "domain_policy"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert body["policy"]["max_tokens"] == 512
    assert "domain_policy" in spec["cursor"]["candidates"][0]["payload"]
    # The archive run used the first 20 retail train ids and 8 heldout ids,
    # which is a different heldout pool from the seed-only tau2:0 (16).
    assert len(body["taskset"]["train_ids"]) == 20
    assert len(body["taskset"]["heldout_ids"]) == 8


@pytest.mark.parametrize("task_id", ["minigrid:1", "minigrid:2"])
def test_minigrid_checkpointed_fixtures_send_system_prompt(task_id: str) -> None:
    spec, body = _request_for(task_id, "http://127.0.0.1:8769")
    assert spec["downstream"]["id"] == "minigrid"
    assert spec["downstream"]["url_pool_env"] == "MINIGRID_URLS"
    assert spec["downstream"]["candidate_field"] == "system_prompt"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert body["policy"]["max_tokens"] == 64
    assert "system_prompt" in spec["cursor"]["candidates"][0]["payload"]
    assert len(body["taskset"]["train_ids"]) == 8
    assert len(body["taskset"]["heldout_ids"]) == 4


@pytest.mark.parametrize("task_id", ["crafter:3", "crafter:4"])
def test_crafter_archive_fixtures_send_react_system_prompt(task_id: str) -> None:
    spec, body = _request_for(task_id, "http://127.0.0.1:20055")
    assert spec["downstream"]["id"] == "crafter"
    assert spec["downstream"]["url_pool_env"] == "CRAFTER_URLS"
    assert spec["downstream"]["candidate_field"] == "react_system_prompt"
    assert body["policy"]["model"] == "gpt-4.1-nano"
    assert "react_system_prompt" in spec["cursor"]["candidates"][0]["payload"]
    assert len(body["taskset"]["train_ids"]) == 8
    assert len(body["taskset"]["heldout_ids"]) == 8


@pytest.mark.parametrize("task_id", sorted(ARCHIVE_FIXTURES))
def test_archive_fixtures_are_generation_starts_with_real_usage(task_id: str) -> None:
    source_run_id, generation, candidate_count = ARCHIVE_FIXTURES[task_id]
    spec = by_task_id(task_id)
    checkpoint = spec["checkpoint"]
    assert spec["source_run_id"] == source_run_id
    assert spec["generation"] == generation
    assert len(spec["cursor"]["candidates"]) == candidate_count
    # A generation_start cursor, never a compacted generation_boundary summary.
    assert checkpoint["run_state"] == "generation_start"
    assert checkpoint["checkpoint_kind"] == "gepa_cursor"
    assert spec["cursor"]["phase"] == "generation_start"
    assert "checkpoint_summary" not in str(checkpoint["snapshot"].get("schema") or "")
    assert not checkpoint["snapshot"].get("compacted")
    # Frozen archive totals, not zeros filled in by the outer.
    usage = checkpoint["snapshot"]["usage"]
    assert usage["total_tokens"] > 0
    assert usage["rollout_calls"] > 0
    # state_history keeps the originating run id (train:3 was itself minted from
    # a proposer-eval fork, so it carries both its own and the lineage it forked
    # from). The engine rebinds these on import; the FK to optimization_runs used
    # to blow up here.
    history = spec["cursor"].get("state_history") or []
    assert history
    run_ids = {entry.get("run_id") for entry in history if isinstance(entry, dict)}
    assert source_run_id in run_ids


def test_every_fixture_ships_complete_usage_totals() -> None:
    from gepa_proposer.fixtures import all_tasks

    required = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "rollout_calls",
        "proposer_calls",
    }
    for spec in all_tasks():
        episode = parse_episode({"episode": {"proposer_rounds": 1}})
        body = build_run_request(
            spec=spec,
            cursor=spec["cursor"],
            arm={"provider": "openai", "model": "gpt-5.6-luna", "reasoning_effort": "low"},
            episode=episode,
            container_url="http://127.0.0.1:8765",
        )
        for usage in (
            body["fixture"]["checkpoint"]["usage"],
            body["fixture"]["checkpoint"]["snapshot"]["usage"],
        ):
            assert required <= set(usage), spec["task_id"]
            assert all(isinstance(usage[key], int) for key in required), spec["task_id"]
        # Only the three seed-only fixtures (no archive behind them) are zero.
        total = body["fixture"]["checkpoint"]["snapshot"]["usage"]["total_tokens"]
        if spec["task_id"] in {"tau2:0", "minigrid:0", "officeqa:0"}:
            assert total == 0
        else:
            assert total > 0, spec["task_id"]


def test_score_episode_reports_train_terms_and_eval_uplift() -> None:
    pre = [
        {
            "candidate_id": "parent",
            "train_scores": [{"example_id": "train:0", "reward": 0.4}],
            "heldout_reward": 0.5,
            "heldout_scores": [{"example_id": "heldout:0", "reward": 0.5}],
        }
    ]
    episode = [
        {
            "candidate_id": "child",
            "train_scores": [{"example_id": "train:0", "reward": 0.7}],
            "heldout_reward": 0.8,
            "heldout_scores": [{"example_id": "heldout:0", "reward": 0.8}],
        }
    ]
    scored = score_episode(pre_fork=pre, episode_candidates=episode)
    assert scored["train_exploitation"] == pytest.approx(0.3)
    assert scored["train_exploration"] == pytest.approx(0.3)
    assert scored["eval_uplift"] == pytest.approx(0.3)
    assert scored["reward"] == pytest.approx(0.9)
    assert scored["heldout_evaluated"] is True

    missing = score_episode(
        pre_fork=[{"candidate_id": "parent", "train_scores": [{"example_id": "train:0", "reward": 0.4}]}],
        episode_candidates=[
            {"candidate_id": "child", "train_scores": [{"example_id": "train:0", "reward": 0.4}]}
        ],
    )
    assert missing["eval_uplift"] == 0.0
    assert missing["heldout_evaluated"] is False
    assert missing["reward"] == missing["train_exploration"] + missing["train_exploitation"]

    live = score_episode(
        pre_fork=[
            {"candidate_id": "parent", "train_scores": [{"example_id": "train:0", "reward": 0.4}]}
        ],
        episode_candidates=[
            {"candidate_id": "child", "train_scores": [{"example_id": "train:0", "reward": 0.7}]}
        ],
        post_candidates=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            },
        ],
        best_candidate_id="child",
    )
    assert live["eval_uplift"] == pytest.approx(0.3)
    assert live["heldout_evaluated"] is True
    assert live["reward"] == pytest.approx(0.9)
    assert live["exploration_reduce"] == "mean"
    assert live["train_exploration_sum"] == pytest.approx(0.3)

    crowded = score_episode(
        pre_fork=[
            {
                "candidate_id": "parent",
                "train_scores": [
                    {"example_id": f"train:{i}", "reward": 0.0} for i in range(8)
                ],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [
                    {"example_id": f"train:{i}", "reward": 1.0 if i < 3 else 0.0}
                    for i in range(8)
                ],
                "heldout_reward": 0.5,
            }
        ],
    )
    assert crowded["train_exploration_sum"] == pytest.approx(3.0)
    assert crowded["train_exploration"] == pytest.approx(0.375)
    assert crowded["eval_uplift"] == pytest.approx(0.0)
    assert crowded["reward"] == pytest.approx(
        crowded["train_exploration"] + crowded["train_exploitation"] + crowded["eval_uplift"]
    )
    summed = score_episode(
        pre_fork=crowded and [
            {
                "candidate_id": "parent",
                "train_scores": [
                    {"example_id": f"train:{i}", "reward": 0.0} for i in range(8)
                ],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [
                    {"example_id": f"train:{i}", "reward": 1.0 if i < 3 else 0.0}
                    for i in range(8)
                ],
                "heldout_reward": 0.5,
            }
        ],
        combine={"exploration_reduce": "sum"},
    )
    assert summed["train_exploration"] == pytest.approx(3.0)


def test_metadata_advertises_optional_gepa_ascope(tmp_path: Path) -> None:
    c = client(tmp_path)
    meta = c.get("/metadata").json()
    ascope = meta["metadata"]["gepa_is_everything"]
    assert ascope["manderqueue"] == "optional"
    assert ascope["scratchpad"] == "optional"
    assert ascope["mcp_agent"] == "optional"
    assert ascope["pipeline"][-1] == "combee"
    assert meta["metadata"]["episode"]["exploration_reduce"] == "mean"
    assert meta["capabilities"]["pause_support"] is True
    assert meta["capabilities"]["resume_support"] is True


def test_eval_gepa_via_gepa_container_is_the_acceptance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("BANKING77_URL", "http://127.0.0.1:8765")
    fake = FakeOptimizer()
    c = client(tmp_path, optimizer=fake)
    result = c.post(
        "/rollout",
        json={
            "task_id": "train:1",
            "submission_mode": "sync",
            "policy": {"model": "gpt-5.6-luna", "reasoning_effort": "low"},
            "episode": {
                "proposer_rounds": 1,
                "pipeline_mode": "combee",
                "operator": {
                    "scratchpad": {"enabled": True},
                    "hypotheses": {"enabled": True, "max_open": 4},
                    "reward": {"exploration_reduce": "mean"},
                },
            },
        },
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "completed"
    assert body["reward_info"]["outcome_reward"] == body["reward"]
    details = body["reward_details"]
    assert details["exploration_reduce"] == "mean"
    assert details["train_exploration"] == pytest.approx(
        details["train_exploration_sum"] / max(1, details["scored_example_ids"])
    )
    assert body["reward"] == pytest.approx(
        details["train_exploration"]
        + details["train_exploitation"]
        + details["eval_uplift"]
    )
    created = fake.bodies[0]
    assert created["advanced"]["pipeline"]["mode"] == "flash_evolve"
    assert created["advanced"]["operator"]["scratchpad"]["enabled"] is True
    assert created["advanced"]["operator"]["hypotheses"]["max_open"] == 4


def test_operator_opt_in_is_absent_by_default(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BANKING77_URL", "http://127.0.0.1:8765")
    fake = FakeOptimizer()
    c = client(tmp_path, optimizer=fake)
    result = c.post(
        "/rollout",
        json={
            "task_id": "train:1",
            "submission_mode": "sync",
            "policy": {"reasoning_effort": "low"},
        },
    )
    assert result.status_code == 200, result.text
    assert "operator" not in fake.bodies[0]["advanced"]


def test_optional_reward_terms_use_cursor_evidence() -> None:
    scored = score_episode(
        pre_fork=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
                "acceptance_score": 0.9,
                "reward_details": {"rubric": 0.25},
            }
        ],
        post_candidates=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
                "acceptance_score": 0.9,
                "reward_details": {"rubric": 0.25},
            },
        ],
        best_candidate_id="child",
        combine={
            "include_confidence": True,
            "include_time": True,
            "include_cost": True,
            "include_milestones": True,
            "include_rubrics": True,
            "confidence_weight": 1.0,
            "time_weight": 1.0,
            "cost_weight": 1.0,
            "milestones_weight": 1.0,
            "rubrics_weight": 1.0,
        },
        context={
            "created_at": "2026-08-20T01:00:00Z",
            "completed_at": "2026-08-20T01:15:00Z",
            "pre_cursor": {"generation": 1, "usage": {"cost_usd": 0.0}},
            "post_cursor": {"generation": 2, "usage": {"cost_usd": 1.5}},
            "optimizer_finished": {"usage": {"cost_usd": 1.5}},
            "episode": {
                "max_wall_seconds": 1800,
                "max_spend_usd": 15.0,
                "proposer_rounds": 1,
            },
        },
    )
    extras = scored["optional_terms"]
    assert extras["confidence"] == pytest.approx(0.9)
    assert extras["time"] == pytest.approx(-900 / 1800)
    assert extras["cost"] == pytest.approx(-1.5 / 15.0)
    assert scored["episode_cost_usd"] == pytest.approx(1.5)
    assert extras["milestones"] == pytest.approx(1.0)
    assert extras["rubrics"] == pytest.approx(0.25)
    assert scored["reward"] == pytest.approx(
        scored["train_exploration"]
        + scored["train_exploitation"]
        + scored["eval_uplift"]
        + sum(extras.values())
    )


def test_jesterky_annotations_fill_confidence_and_rubrics(tmp_path: Path) -> None:
    state = (
        tmp_path
        / "gepa_abc"
        / "proposer_workspaces"
        / "generation_000"
        / "state"
    )
    state.mkdir(parents=True)
    (state / "jesterky_trace_annotations.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"severity": "medium", "blocker": False, "reward": 0.0}),
                json.dumps({"severity": "high", "blocker": False, "reward": 0.0}),
                json.dumps({"severity": "none", "blocker": False, "reward": 0.0}),
            ]
        )
        + "\n"
    )
    scored = score_episode(
        pre_fork=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            }
        ],
        post_candidates=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            },
        ],
        best_candidate_id="child",
        combine={
            "include_confidence": True,
            "include_rubrics": True,
        },
        context={"output_dir": str(tmp_path)},
    )
    extras = scored["optional_terms"]
    assert extras["confidence"] == pytest.approx(1.0 - 1 / 3)
    assert extras["rubrics"] == pytest.approx((0.5 + 0.25 + 1.0) / 3)
    assert extras["confidence"] != 0.0
    assert extras["rubrics"] != 0.0


def test_missing_fail_rejects_absent_confidence_and_accepts_jesterky(tmp_path: Path) -> None:
    base = {
        "pre_fork": [
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            }
        ],
        "episode_candidates": [
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            }
        ],
        "post_candidates": [
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            },
        ],
        "best_candidate_id": "child",
    }
    with pytest.raises(ValueError, match="confidence evidence missing"):
        score_episode(
            **base,
            combine={"include_confidence": True, "missing": "fail"},
            context={"output_dir": str(tmp_path)},
        )
    state = tmp_path / "gepa_abc" / "proposer_workspaces" / "generation_000" / "state"
    state.mkdir(parents=True)
    (state / "jesterky_trace_annotations.jsonl").write_text(
        json.dumps({"severity": "low", "blocker": False, "reward": 0.0}) + "\n"
    )
    scored = score_episode(
        **base,
        combine={"include_confidence": True, "include_rubrics": True, "missing": "fail"},
        context={"output_dir": str(tmp_path)},
    )
    assert scored["optional_terms"]["confidence"] == pytest.approx(1.0)
    assert scored["optional_terms"]["rubrics"] == pytest.approx(0.75)


def test_zero_cost_usd_with_tokens_is_priced_not_free() -> None:
    scored = score_episode(
        pre_fork=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            }
        ],
        post_candidates=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            },
        ],
        best_candidate_id="child",
        combine={"include_cost": True, "cost_weight": 0.0},
        context={
            "arm": {"model": "gpt-5.6-luna", "provider": "openai"},
            "downstream": {"policy": {"model": "gpt-4.1-nano", "provider": "openai"}},
            "episode": {"max_spend_usd": 15.0},
            "pre_cursor": {
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
                "usage_ledger": [
                    {
                        "boundary": "container.rollout",
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                        "usage": {
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                        },
                    }
                ],
            },
            "post_cursor": {
                "usage": {
                    "prompt_tokens": 2_000_000,
                    "completion_tokens": 1_000_000,
                    "cost_usd": 0.0,
                },
                "usage_ledger": [
                    {
                        "boundary": "container.rollout",
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                        "usage": {
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                        },
                    },
                    {
                        "boundary": "container.rollout",
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                        "usage": {
                            "prompt_tokens": 1_000_000,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                        },
                    },
                    {
                        "boundary": "proposer.codex",
                        "model": "gpt-5.6-luna",
                        "provider": "openai",
                        "prompt_tokens": 0,
                        "completion_tokens": 1_000_000,
                        "cost_usd": 0.0,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 1_000_000,
                            "cost_usd": 0.0,
                        },
                    },
                ],
            },
        },
    )
    # Episode delta: 1M nano input ($0.10) + 1M luna output ($1.20) = $1.30
    assert scored["episode_cost_usd"] == pytest.approx(1.30)
    assert scored["optional_terms"]["cost"] == pytest.approx(-1.30 / 15.0)
    assert scored["optional_terms"]["cost"] != 0.0


def test_reset_cursor_does_not_subtract_fixture_usage() -> None:
    scored = score_episode(
        pre_fork=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            }
        ],
        episode_candidates=[
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            }
        ],
        post_candidates=[
            {
                "candidate_id": "parent",
                "train_scores": [{"example_id": "train:0", "reward": 0.4}],
                "heldout_reward": 0.5,
            },
            {
                "candidate_id": "child",
                "train_scores": [{"example_id": "train:0", "reward": 0.7}],
                "heldout_reward": 0.8,
            },
        ],
        combine={"include_cost": True},
        context={
            "downstream": {"policy": {"model": "gpt-4.1-nano"}},
            "episode": {"max_spend_usd": 15.0},
            "pre_cursor": {
                "usage_ledger": [
                    {
                        "usage_ledger_id": "pre-1",
                        "boundary": "container.rollout",
                        "usage": {"prompt_tokens": 2_000_000, "completion_tokens": 0, "cost_usd": 0.0},
                    }
                ]
            },
            "post_cursor": {
                "usage_ledger": [
                    {
                        "usage_ledger_id": "post-1",
                        "boundary": "container.rollout",
                        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "cost_usd": 0.0},
                    }
                ]
            },
        },
    )
    assert scored["episode_cost_usd"] == pytest.approx(0.10)
    assert scored["cost_pricing"]["cost_source"] == "usage_ledger_delta"


def test_schema_repair_and_jesterky_bulk_reach_the_gepa_wire() -> None:
    spec = by_task_id("train:1")
    cursor = spec["cursor"]
    body = build_run_request(
        spec=spec,
        cursor=cursor,
        arm={"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode={
            "proposer_rounds": 1,
            "skip_heldout": False,
            "schema_repair_rounds": 1,
            "jesterky_bulk": True,
            "operator": {
                "scratchpad": {"enabled": True},
                "hypotheses": {"enabled": True},
                "manderqueue": {"enabled": True, "fail_closed": False},
                "reward": {
                    "exploration_reduce": "mean",
                    "confidence": True,
                    "time": True,
                    "cost": True,
                    "milestones": True,
                    "rubrics": True,
                },
            },
        },
        container_url="http://127.0.0.1:8765",
        run_id="ascope-wire",
    )
    assert body["advanced"]["proposer_io"]["schema_repair_rounds"] == 1
    assert body["advanced"]["jesterky_workflow"]["bulk"] is True
    assert body["advanced"]["jesterky_workflow"]["enabled"] is True
    assert body["advanced"]["jesterky_workflow"]["fail_closed"] is False
    assert body["advanced"]["operator"]["scratchpad"]["enabled"] is True
    assert body["advanced"]["operator"]["manderqueue"]["fail_closed"] is False
    parsed = parse_episode(
        {
            "episode": {
                "proposer_rounds": 1,
                "schema_repair_rounds": 2,
                "jesterky_bulk": True,
            }
        }
    )
    assert parsed["schema_repair_rounds"] == 2
    assert parsed["jesterky_bulk"] is True


def test_jesterky_enabled_without_bulk_survives_parse_episode() -> None:
    spec = by_task_id("train:1")
    parsed = parse_episode(
        {
            "episode": {
                "proposer_rounds": 1,
                "skip_heldout": False,
                "jesterky": True,
                "jesterky_bulk": False,
                "jesterky_workflow": {
                    "enabled": True,
                    "bulk": False,
                    "fail_closed": False,
                    "command": "/tmp/jesterky",
                },
            }
        }
    )
    assert parsed["jesterky"] is True
    assert parsed["jesterky_bulk"] is False
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode=parsed,
        container_url="http://127.0.0.1:8765",
        run_id="ascope-jesterky-cap",
    )
    assert body["advanced"]["jesterky_workflow"]["enabled"] is True
    assert body["advanced"]["jesterky_workflow"]["bulk"] is False
    assert body["advanced"]["jesterky_workflow"]["command"] == "/tmp/jesterky"


def test_ascope_mcp_and_code_lever_reach_the_gepa_wire() -> None:
    tau = by_task_id("tau2:1")
    body = build_run_request(
        spec=tau,
        cursor=tau["cursor"],
        arm={"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode={
            "proposer_rounds": 1,
            "skip_heldout": False,
            "jesterky": True,
            "operator": {
                "levers": {"prompt": True, "code": True, "harness": True},
                "mcp_agent": {
                    "enabled": True,
                    "command": "npx -y @modelcontextprotocol/server-filesystem .",
                    "server": "workspace_fs",
                },
            },
        },
        container_url="http://127.0.0.1:8774",
        run_id="ascope-mcp",
    )
    mcp = body["advanced"]["operator"]["mcp_agent"]
    assert mcp["enabled"] is True
    assert "npx" in mcp["command"]
    assert mcp["server"] == "workspace_fs"
    assert body["advanced"]["jesterky_workflow"]["enabled"] is True
    program = program_for_task(tau)
    assert program["target_modules"][0]["candidate_field"] == "domain_policy"


def test_manderqueue_stub_serves_operator_guidance() -> None:
    import socket
    import threading
    from urllib.request import urlopen

    from gepa_proposer.mq_stub import serve

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    server = serve("127.0.0.1", port, thread_id="gepa-ascope", message="prefer generalization")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        health = json.loads(urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
        assert health["ok"] is True
        messages = json.loads(
            urlopen(f"http://127.0.0.1:{port}/v1/threads/gepa-ascope/messages", timeout=2).read()
        )
        assert messages[0]["body"] == "prefer generalization"
        assert "heldout" not in messages[0]["body"].lower()
    finally:
        server.shutdown()
        server.server_close()


def test_ascope_harvest_reads_operator_workspace(tmp_path: Path) -> None:
    from gepa_proposer.ascope_harvest import harvest_episode_dir

    state = (
        tmp_path
        / "gepa_abc"
        / "proposer_workspaces"
        / "generation_000"
        / "state"
    )
    state.mkdir(parents=True)
    workspace = state.parent
    (state / "scratchpad.md").write_text("# GEPA shared scratchpad\n")
    (state / "guidance.md").write_text("# Operator guidance\n\n- prefer generalization\n")
    (state / "hypotheses.json").write_text(
        json.dumps({"schema_version": "gepa_hypotheses.v1", "open": [{"id": "h1"}], "retired": []})
    )
    (state / "manderqueue_inbox.json").write_text(
        json.dumps(
            {
                "ok": True,
                "base_url": "http://127.0.0.1:9",
                "messages": [{"body": "prefer generalization"}],
            }
        )
    )
    (state / "mcp_agent.json").write_text(
        json.dumps({"enabled": True, "server": "workspace_fs"})
    )
    (state / "operator.json").write_text(
        json.dumps({"levers": {"prompt": True, "code": True}, "control": {"pause": True}})
    )
    (workspace / ".codex_home").mkdir()
    (workspace / ".codex_home" / "config.toml").write_text("[mcp_servers.workspace_fs]\ncommand = \"npx\"\n")
    (workspace / "jesterky_workflow_receipt.json").write_text(
        json.dumps({"enabled": True, "annotated": 2})
    )
    (state / "jesterky_proposer_context.md").write_text("# jesterky themes\n")
    (state / "jesterky_theme_registry.json").write_text("[]")
    (state / "jesterky_trace_annotations.jsonl").write_text("{}\n")
    (tmp_path / "candidate_registry.json").write_text(
        json.dumps(
            [
                {
                    "candidate_id": "gepa_new",
                    "payload": {"domain_policy": "AUTH then CANCEL"},
                    "lever_bundle": {"mutated_lever_ids": ["domain_policy"]},
                }
            ]
        )
    )
    harvested = harvest_episode_dir(tmp_path)
    assert harvested["ok"] is True
    assert harvested["scratchpad"] is True
    assert harvested["guidance_has_messages"] is True
    assert harvested["hypotheses_open"] == 1
    assert harvested["manderqueue_messages"] == 1
    assert harvested["mcp_in_codex_config"] is True
    assert harvested["jesterky_annotated"] == 2
    assert harvested["jesterky_context"] is True
    assert harvested["jesterky_themes"] is True
    assert harvested["jesterky_annotations"] is True
    assert harvested["mutated_lever_ids"] == ["domain_policy"]
    assert harvested["payload_fields"] == ["domain_policy"]
    assert harvested["code_lever_mutated"] is True


def test_manderqueue_base_url_reaches_the_gepa_wire() -> None:
    spec = by_task_id("train:1")
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={"model": "gpt-5.6-luna", "reasoning_effort": "low"},
        episode={
            "proposer_rounds": 1,
            "skip_heldout": False,
            "operator": {
                "manderqueue": {
                    "enabled": True,
                    "fail_closed": False,
                    "base_url": "http://127.0.0.1:18765",
                    "thread_id": "gepa-ascope",
                }
            },
        },
        container_url="http://127.0.0.1:8765",
        run_id="ascope-mq",
    )
    mq = body["advanced"]["operator"]["manderqueue"]
    assert mq["base_url"] == "http://127.0.0.1:18765"
    assert mq["thread_id"] == "gepa-ascope"


def test_openrouter_arm_raises_jsonrpc_stall_budget(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    spec = by_task_id("train:1")
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm={
            "provider": "openrouter",
            "model": "nvidia/nemotron-3.5-lightning",
            "allow_unverified_model": True,
            "model_context_window": 1000000,
        },
        episode={"proposer_rounds": 1, "skip_heldout": False},
        container_url="http://127.0.0.1:8765",
        run_id="ascope-stall",
    )
    io = body["advanced"]["proposer_io"]
    assert io["timeout_seconds"] == 600
    assert io["message_stall_timeout_seconds"] >= 300


def test_run_poll_retries_transient_sqlite_lock(monkeypatch: Any) -> None:
    client = OptimizerClient(base_url="http://unused.test")
    responses: list[Any] = [
        RuntimeError(
            'GET /runs/run-1 -> 500: {"error":{"message":'
            '"sqlite error: database is locked"}}'
        ),
        {"status": "completed"},
    ]

    def fake_get_run(_run_id: str) -> dict[str, Any]:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client, "get_run", fake_get_run)
    result = client.wait_until_terminal("run-1", timeout_seconds=1, poll_seconds=0)
    assert result["status"] == "completed"


def test_async_run_poll_does_not_hide_unrelated_server_error(monkeypatch: Any) -> None:
    client = OptimizerClient(base_url="http://unused.test")

    async def fake_get_run(_run_id: str) -> dict[str, Any]:
        raise RuntimeError("GET /runs/run-1 -> 500: unrelated failure")

    monkeypatch.setattr(client, "aget_run", fake_get_run)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        asyncio.run(
            client.await_until_terminal("run-1", timeout_seconds=1, poll_seconds=0)
        )
