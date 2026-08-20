"""Contract tests for the GameBench lever containers.

Adapter tests run in-process and are cheap. Stack tests boot real subprocesses.
Harness tests need an LLM key and are skipped without one -- they are never
silently replaced by a stub.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamebench_levers import GAMES  # noqa: E402
from gamebench_levers.apply import apply_unified_diff, sha256_text  # noqa: E402
from gamebench_levers.inspect_script import inspect_source  # noqa: E402
from gamebench_levers.seeds import code_seed, harness_seed, prompt_seed  # noqa: E402

needs_llm = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")),
    reason="harness episodes call a model every turn; no stub exists",
)


# -- seeds -----------------------------------------------------------------
@pytest.mark.parametrize("game", GAMES)
def test_code_seed_defines_act(game: str) -> None:
    namespace: dict = {}
    exec(compile(code_seed(game), "policy.py", "exec"), namespace)  # noqa: S102
    assert callable(namespace["act"])


@pytest.mark.parametrize("game", GAMES)
def test_harness_seed_is_a_speedrunner(game: str) -> None:
    namespace: dict = {}
    exec(compile(harness_seed(game), "harness.py", "exec"), namespace)  # noqa: S102
    assert callable(namespace["run_episode"])
    skills = namespace["PUBLIC_SKILLS"]
    assert skills and all(callable(fn) for fn in skills.values())
    assert prompt_seed(game)


@pytest.mark.parametrize("game", GAMES)
def test_seed_policy_is_below_the_reference(game: str) -> None:
    """A seed already at the ceiling cannot show uplift."""
    from gamebench_levers.references import reference_policy

    assert code_seed(game) != reference_policy(game)


# -- adapters --------------------------------------------------------------
def _adapter(game: str):
    """Each game must be loaded in its own process; run these one at a time."""
    from gamebench_levers.adapters import load

    return load(game)


@pytest.mark.parametrize("game", GAMES)
def test_adapter_surface_in_subprocess(game: str) -> None:
    """gold_python collides across games, so exercise each adapter in a fresh process."""
    script = f"""
import json
from gamebench_levers.adapters import load
adapter = load({game!r})
spec = adapter.spec()
assert spec["game"] == {game!r}
assert spec["action_space"]
assert spec["train_seeds"] and spec["heldout_seeds"]
session = adapter.make_session(spec["train_seeds"][0])
obs = session.observation()
for key in ("game", "tick", "done", "ascii", "legal_actions", "score", "achievements", "state"):
    assert key in obs, key
result = session.step(obs["legal_actions"][0])
for key in ("obs", "reward", "terminated", "truncated", "info"):
    assert key in result, key
assert "outcome_reward" in result["info"]
print("ok")
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ok" in proc.stdout


def test_dungeongrid_episode_is_bounded_by_attempts() -> None:
    """Rejected actions do not advance step_index; without an attempt bound this loops."""
    script = """
from gamebench_levers.adapters import load
adapter = load("dungeongrid")
session = adapter.make_session(0, 40)
obs = session.observation()
attempts = 0
while not obs["done"] and attempts < 100000:
    obs = session.step("move:east")["obs"]
    attempts += 1
assert obs["done"], "episode never terminated"
assert attempts <= 40 * 4, attempts
print("ok", attempts)
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# -- apply / inspect -------------------------------------------------------
def test_unified_diff_applies() -> None:
    original = "def act(obs):\n    return 'up'\n"
    diff = "@@ -1,2 +1,2 @@\n def act(obs):\n-    return 'up'\n+    return 'down'\n"
    assert "down" in apply_unified_diff(original, diff)


def test_inspect_survives_unparseable_source() -> None:
    """A candidate that does not parse must not 500 the policy service's own /health."""
    report = inspect_source("def run_episode(  syntax")
    assert report["parse_ok"] is False
    assert report["syntax_error"]
    assert report["content_hash"] == sha256_text("def run_episode(  syntax")


# -- stack contract --------------------------------------------------------
@pytest.fixture(scope="module")
def code_stack():
    from gamebench_levers.stack import start_stack

    stack = start_stack("sokoban", "code")
    yield stack
    stack.stop()


def test_program_advertises_the_candidates_route(code_stack) -> None:
    metadata = httpx.get(f"{code_stack.orch_url}/metadata", timeout=30).json()
    gepa = metadata["metadata"]["optimizer_contracts"]["gepa"]
    assert gepa["candidates_route"] == "/candidates"
    program = httpx.get(f"{code_stack.orch_url}/program", timeout=30).json()
    module = program["modules"][0]
    assert module["candidate_field"] == "policy_script"
    assert module["metadata"]["lever_kind"] == "policy_script"
    assert {s["schema_id"] for s in program["side_info_schemas"]} >= {"apply_report.v1"}


def test_register_once_then_run_many(code_stack) -> None:
    from gamebench_levers.references import reference_policy

    registered = httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "t_ref", "lever_bundle": {"values": {"policy_script": reference_policy("sokoban")}}},
        timeout=180,
    ).json()
    assert registered["status"] == "registered"
    assert registered["candidate_id"] == "t_ref"

    rewards = []
    for index in range(3):
        record = httpx.post(
            f"{code_stack.orch_url}/rollout",
            json={"task_id": f"train:{index}", "candidate_id": "t_ref"},
            timeout=600,
        ).json()
        assert record["candidate_id"] == "t_ref"
        assert record["status"] == "completed"
        assert record["actionable_side_info"]
        rewards.append(record["reward"])
    assert max(rewards) > 0.0, "reference scored nothing on any train task"


def test_broken_candidate_fails_closed(code_stack) -> None:
    body = httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "t_bad", "lever_bundle": {"values": {"policy_script": "def act(obs)\n bad"}}},
        timeout=120,
    ).json()
    assert body["status"] == "apply_failed"
    assert body["candidate_id"] is None, "a failed apply must not yield a rollable candidate_id"
    report = body["apply_report"][0]
    assert report["schema_id"] == "apply_report.v1"
    assert report["compile_ok"] is False
    # the container stays usable
    assert httpx.get(f"{code_stack.orch_url}/health", timeout=30).json()["status"] == "ok"


def test_runtime_error_scores_zero_rather_than_crashing(code_stack) -> None:
    """gepa-ai's rule: never raise on one example; return a failed score plus ASI."""
    source = "def act(obs):\n    raise RuntimeError('boom')\n"
    registered = httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "t_boom", "lever_bundle": {"values": {"policy_script": source}}},
        timeout=120,
    ).json()
    assert registered["status"] == "registered", "it compiles; it only fails at call time"
    record = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate_id": "t_boom"},
        timeout=600,
    ).json()
    assert record["reward"] == 0.0
    trace = next(s for s in record["side_info"] if s["schema_id"] == "code_policy_game_trace.v1")
    assert trace["summary"]["runtime_errors"], "the proposer needs to see why it scored 0"


# -- harness ---------------------------------------------------------------
@needs_llm
def test_each_candidate_gets_its_own_worker_and_the_env_is_untouched() -> None:
    """`per_candidate_worker`: a new candidate spawns its own policy process.

    The seed's worker keeps serving the seed, so switching between candidates is a
    routing decision rather than a restart.
    """
    from gamebench_levers.stack import start_stack

    stack = start_stack("sokoban", "harness", isolation="per_candidate_worker")
    try:
        seed_health = httpx.get(f"{stack.policy_url}/health", timeout=30).json()
        assert seed_health["public_skills"]
        variant = harness_seed("sokoban").replace("MAX_LLM_CALLS = 12", "MAX_LLM_CALLS = 3")
        registered = httpx.post(
            f"{stack.orch_url}/candidates",
            json={"candidate_id": "t_h", "lever_bundle": {"values": {"harness_module": variant}}},
            timeout=300,
        ).json()
        assert registered["status"] == "registered"
        report = registered["apply_report"][0]
        assert report["protocol_id"] == "harness_restart.v1"
        assert report["restart_ok"] is True
        assert report["env_untouched"] is True
        assert report["isolation"] == "per_candidate_worker"
        assert report["worker_url"] and report["worker_url"] != stack.policy_url

        seed_after = httpx.get(f"{stack.policy_url}/health", timeout=30).json()
        assert seed_after["pid"] == seed_health["pid"], "the seed's worker must not be restarted"
        worker = httpx.get(f"{report['worker_url']}/health", timeout=30).json()
        assert worker["pid"] != seed_health["pid"]
        assert worker["compile_ok"] is True

        metadata = httpx.get(f"{stack.orch_url}/metadata", timeout=30).json()
        assert metadata["metadata"]["apply_isolation"] == "per_candidate_worker"
    finally:
        stack.stop()


@needs_llm
def test_switching_candidates_does_not_respawn_the_policy() -> None:
    """The bug this fixes: GEPA interleaves rollouts across candidates.

    With one shared worker that forced a process restart on every switch, so
    register-once/run-many degraded back into restart-per-rollout.
    """
    from gamebench_levers.stack import start_stack

    stack = start_stack("sokoban", "harness", isolation="per_candidate_worker")
    try:
        base = harness_seed("sokoban")
        for cid, calls in (("w_a", "2"), ("w_b", "3")):
            body = httpx.post(
                f"{stack.orch_url}/candidates",
                json={
                    "candidate_id": cid,
                    "lever_bundle": {"values": {"harness_module": base.replace("MAX_LLM_CALLS = 12", f"MAX_LLM_CALLS = {calls}")}},
                },
                timeout=300,
            ).json()
            assert body["status"] == "registered"
        spawns_after_register = httpx.get(f"{stack.control_url}/workers", timeout=30).json()["spawns"]

        for index in range(4):
            record = httpx.post(
                f"{stack.orch_url}/rollout",
                json={"task_id": f"train:{index % 2}", "candidate_id": "w_a" if index % 2 == 0 else "w_b"},
                timeout=600,
            ).json()
            assert record["status"] == "completed"

        stats = httpx.get(f"{stack.control_url}/workers", timeout=30).json()
        assert stats["spawns"] == spawns_after_register, "no worker may be spawned by a candidate switch"
        assert stats["reuses"] >= 4, "every interleaved rollout should reuse a warm worker"
    finally:
        stack.stop()


@needs_llm
def test_worker_pool_evicts_least_recently_used_without_losing_correctness() -> None:
    """An evicted candidate respawns from stored source: eviction costs latency only."""
    from gamebench_levers.stack import start_stack

    stack = start_stack("sokoban", "harness", isolation="per_candidate_worker", max_workers=2)
    try:
        base = harness_seed("sokoban")
        for index in range(4):
            body = httpx.post(
                f"{stack.orch_url}/candidates",
                json={
                    "candidate_id": f"e_{index}",
                    "lever_bundle": {"values": {"harness_module": base.replace("MAX_LLM_CALLS = 12", f"MAX_LLM_CALLS = {index + 2}")}},
                },
                timeout=300,
            ).json()
            assert body["status"] == "registered"
        stats = httpx.get(f"{stack.control_url}/workers", timeout=30).json()
        assert stats["evictions"] > 0, "the pool must be bounded"
        assert len(stats["workers"]) <= stats["max_workers"] + 1

        # the first candidate was evicted; it must still roll out correctly
        record = httpx.post(
            f"{stack.orch_url}/rollout",
            json={"task_id": "train:0", "candidate_id": "e_0"},
            timeout=600,
        ).json()
        assert record["status"] == "completed"
        assert record["candidate_id"] == "e_0"
    finally:
        stack.stop()


@needs_llm
def test_broken_harness_never_disturbs_the_running_policy() -> None:
    """A candidate that cannot compile is rejected before anything is written.

    Nothing is applied, so there is nothing to roll back and the live policy process
    keeps its pid. If a restart *does* fail later, the tree is rolled back instead --
    both paths must leave the stack usable.
    """
    from gamebench_levers.stack import start_stack

    stack = start_stack("sokoban", "harness")
    try:
        before = httpx.get(f"{stack.policy_url}/health", timeout=30).json()
        body = httpx.post(
            f"{stack.orch_url}/candidates",
            json={"candidate_id": "t_hbad", "lever_bundle": {"values": {"harness_module": "def run_episode(  x"}}},
            timeout=300,
        ).json()
        assert body["status"] == "apply_failed"
        assert body["candidate_id"] is None
        report = next(r for r in body["apply_report"] if r.get("compile_ok") is False)
        assert "SyntaxError" in report["reject_reason"], report.get("reject_reason")
        assert report["compile_diagnostics"]["lineno"]

        after = httpx.get(f"{stack.policy_url}/health", timeout=30).json()
        assert after["status"] == "ok"
        assert after["pid"] == before["pid"], "a non-compiling candidate must not restart the policy"
        assert after["public_skills"] == before["public_skills"]
    finally:
        stack.stop()


def test_a_candidate_id_always_resolves_to_one_policy(code_stack) -> None:
    """One candidate_id, one policy, forever.

    GEPA rolls a candidate out many times and sends the inline bundle only
    sometimes. If an unknown id silently fell back to the seed, the same id would
    score one row twice with two different policies and the engine would abort with
    `conflicting score vector material`.
    """
    from gamebench_levers.references import reference_policy

    strong = {"lever_bundle": {"values": {"policy_script": reference_policy("sokoban")}}}
    first = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:2", "candidate_id": "t_bind"},
        timeout=600,
    ).json()
    second = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:2", "candidate_id": "t_bind", **strong},
        timeout=600,
    ).json()
    third = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:2", "candidate_id": "t_bind"},
        timeout=600,
    ).json()
    assert first["reward"] == second["reward"] == third["reward"]

    fresh = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:2", "candidate_id": "t_fresh", **strong},
        timeout=600,
    ).json()
    assert fresh["reward"] > first["reward"], "a new id with a bundle must still apply it"


# -- ASI plane -------------------------------------------------------------
def test_asi_route_is_advertised(code_stack) -> None:
    metadata = httpx.get(f"{code_stack.orch_url}/metadata", timeout=30).json()
    assert metadata["metadata"]["optimizer_contracts"]["gepa"]["asi_route"] == "/asi"
    schemas = httpx.get(f"{code_stack.orch_url}/asi/schemas", timeout=30).json()
    assert {entry["schema_id"] for entry in schemas["schemas"]} >= {"apply_report.v1"}


def test_asi_is_addressable_without_reparsing_the_rollout(code_stack) -> None:
    """The engine stores the terminal record as an opaque raw_response blob.

    `/asi/{rollout_id}` is the read path that does not require digging it back out.
    """
    from gamebench_levers.references import reference_policy

    httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "t_asi", "lever_bundle": {"values": {"policy_script": reference_policy("sokoban")}}},
        timeout=180,
    )
    record = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:2", "candidate_id": "t_asi"},
        timeout=600,
    ).json()
    assert record["asi_ref"] == f"/asi/{record['rollout_id']}"

    envelope = httpx.get(f"{code_stack.orch_url}{record['asi_ref']}", timeout=30).json()
    assert envelope["schema_version"] == "asi_envelope.v1"
    assert envelope["candidate_id"] == "t_asi"
    assert envelope["reward"] == record["reward"]
    # the inline envelope still rides on the record: the engine sensor reads it there
    assert envelope["side_info"] == record["actionable_side_info"]


def test_asi_serves_one_typed_frame_and_a_thin_view(code_stack) -> None:
    record = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate_id": "t_asi"},
        timeout=600,
    ).json()
    rollout_id = record["rollout_id"]

    frame = httpx.get(
        f"{code_stack.orch_url}/asi/{rollout_id}/code_policy_game_trace.v1", timeout=30
    ).json()
    assert frame["schema_id"] == "code_policy_game_trace.v1"
    assert "runtime_errors" in frame["frames"][0]["summary"]

    thin = httpx.get(f"{code_stack.orch_url}/asi/{rollout_id}?summary_only=true", timeout=30).json()
    assert all("body" not in entry for entry in thin["side_info"]), "summary_only must drop bodies"

    assert httpx.get(f"{code_stack.orch_url}/asi/{rollout_id}/nope.v1", timeout=30).status_code == 404
    assert httpx.get(f"{code_stack.orch_url}/asi/nosuchrollout", timeout=30).status_code == 404


def test_asi_index_filters(code_stack) -> None:
    index = httpx.get(f"{code_stack.orch_url}/asi", params={"candidate_id": "t_asi"}, timeout=30).json()
    assert index["count"] >= 2
    assert {item["candidate_id"] for item in index["items"]} == {"t_asi"}
    assert all(item["asi_ref"].startswith("/asi/") for item in index["items"])

    by_schema = httpx.get(
        f"{code_stack.orch_url}/asi", params={"schema_id": "apply_report.v1"}, timeout=30
    ).json()
    assert by_schema["count"] >= 1
    assert httpx.get(
        f"{code_stack.orch_url}/asi", params={"candidate_id": "nobody"}, timeout=30
    ).json()["count"] == 0


# -- diagnostics the proposer can act on -----------------------------------
def test_compile_report_pinpoints_the_offending_line() -> None:
    """The proposer's most common failure is a raw newline inside a quoted string."""
    from gamebench_levers.orchestrator_app import compile_report

    broken = "def run_episode(env):\n    x = 'Observation:\n'\n    return {}\n"
    report = compile_report(broken, "harness.py")
    assert report["compile_ok"] is False
    assert "unterminated string literal" in report["error"]
    assert report["lineno"] == 2
    assert report["source_line"].strip().startswith("x = 'Observation:")
    assert compile_report("def act(obs):\n    return 'up'\n", "policy.py")["compile_ok"] is True


def test_apply_failure_tells_the_proposer_what_broke(code_stack) -> None:
    """`restart_failed` is not actionable; the syntax error and its line are."""
    body = httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={
            "candidate_id": "t_diag",
            "lever_bundle": {"values": {"policy_script": "def act(obs):\n    x = 'oops\n    return 'up'\n"}},
        },
        timeout=120,
    ).json()
    assert body["status"] == "apply_failed"
    report = next(r for r in body["apply_report"] if r.get("compile_ok") is False)
    assert "SyntaxError" in report["reject_reason"]
    assert report["compile_diagnostics"]["lineno"]
    assert report["compile_diagnostics"]["source_line"]

    record = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate_id": "t_diag"},
        timeout=600,
    ).json()
    verdict = httpx.get(
        f"{code_stack.orch_url}/asi/{record['rollout_id']}/episode_verdict.v1", timeout=30
    ).json()["frames"][0]["summary"]
    assert verdict["episode_ran"] is False
    assert verdict["not_scored_because"] == "apply_failed"
    assert "SyntaxError" in verdict["reject_reason"]
    assert verdict["fix_hint"]


def test_a_policy_that_never_steps_the_env_says_so(code_stack) -> None:
    """Scoring 0 is right; leaving the proposer unable to tell why is not."""
    source = "def act(obs):\n    raise RuntimeError('never steps')\n"
    httpx.post(
        f"{code_stack.orch_url}/candidates",
        json={"candidate_id": "t_nostep", "lever_bundle": {"values": {"policy_script": source}}},
        timeout=120,
    )
    record = httpx.post(
        f"{code_stack.orch_url}/rollout",
        json={"task_id": "train:0", "candidate_id": "t_nostep"},
        timeout=600,
    ).json()
    verdict = httpx.get(
        f"{code_stack.orch_url}/asi/{record['rollout_id']}/episode_verdict.v1", timeout=30
    ).json()["frames"][0]["summary"]
    assert record["reward"] == 0.0
    assert verdict["episode_ran"] is False
    assert verdict["not_scored_because"] == "no_env_steps"
    assert verdict["runtime_errors"], "the proposer needs the exception text"


def test_infra_errors_are_reported_apart_from_policy_errors(code_stack) -> None:
    """A dropped connection is not evidence about the candidate."""
    record = httpx.post(
        f"{code_stack.orch_url}/rollout", json={"task_id": "train:0"}, timeout=600
    ).json()
    verdict = httpx.get(
        f"{code_stack.orch_url}/asi/{record['rollout_id']}/episode_verdict.v1", timeout=30
    ).json()["frames"][0]["summary"]
    assert "infra_errors" in verdict and "runtime_errors" in verdict
    assert verdict["infra_errors"] == []
