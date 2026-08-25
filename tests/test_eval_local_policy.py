"""The local-inference lane: local routes, adapter candidates, snapshot pinning.

These cover the four narrow relaxations that let `eval` score a policy served
by an MLX process on the host instead of a paid provider, and — more to the
point — they cover what each relaxation still refuses.  `ModelRoute` exists so
that an agent picks a model from an allowlist rather than naming an endpoint,
so every widening of it is tested against the case it must still reject.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from synth_optimizers.eval import (
    MLX_LORA_POLICY_KIND,
    EvalContractError,
    EvalHome,
    EvalRunner,
    ModelRoute,
    WorkerManifest,
    get_recipe,
    read_mlx_lora_policy,
)
from synth_optimizers.eval.executor import TrialExecution, TrialRunRequest
from synth_optimizers.eval.staging import CandidateSource, stage_candidate_set

MLX_RECIPE = "eval.mlx.local-policy.smoke.v1"
CRAFTAX_MLX_RECIPE = "eval.craftax.mlx-local-policy.smoke.v1"
FIXTURE_RECIPE = "eval.fixture.policy-smoke.v1"
PINNED = "sha256:" + "cd" * 32
TEMPLATE_DIGEST = "sha256:" + "ef" * 32

LOCAL_ROUTE = "http://host.docker.internal:8787/v1/chat/completions"


def model_mapping(**overrides):
    """A zero-rate local route: no provider key, no per-token price."""

    payload = {
        "id": "mlx-local-base",
        "route": LOCAL_ROUTE,
        "secret": "SYNTH_MLX_RL_TOKEN",
        "efforts": [],
        "usd_per_1m_input": 0.0,
        "usd_per_1m_output": 0.0,
        "usd_per_1m_cached_input": 0.0,
        "price_source": "local-compute",
        "price_as_of": "2026-08-18",
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------ O1 route


@pytest.mark.parametrize(
    "route",
    [
        "http://127.0.0.1:8787/v1/chat/completions",
        "http://localhost:8787/v1/chat/completions",
        "http://[::1]:8787/v1/chat/completions",
        "http://host.docker.internal:8787/v1/chat/completions",
        "http://localhost:8787/v1/responses",
        "https://api.openai.com/v1/chat/completions",
        "https://api.openai.com/v1/responses",
    ],
)
def test_local_and_https_routes_are_accepted(route):
    assert ModelRoute.from_mapping(model_mapping(route=route)).route == route


@pytest.mark.parametrize(
    "route",
    [
        # A public http origin is the whole thing this schema exists to refuse.
        "http://api.openai.com/v1/chat/completions",
        # "On my LAN" is not "cannot leave this machine".
        "http://10.0.0.7:8787/v1/chat/completions",
        "http://192.168.1.20:8787/v1/chat/completions",
        # A host that merely *looks* loopback.
        "http://127.0.0.1.evil.example/v1/chat/completions",
        "http://localhost.evil.example/v1/chat/completions",
        # Credentials smuggled into the endpoint.
        "http://user:pass@127.0.0.1:8787/v1/chat/completions",
        "https://user:pass@api.openai.com/v1/chat/completions",
        # A route is an endpoint, not a request.
        "https://api.openai.com/v1/chat/completions?key=sk-live",
        "https://api.openai.com/v1/chat/completions#frag",
        # Neither API family.
        "https://api.openai.com/v1/completions",
        "http://127.0.0.1:8787/",
        # Not a transport this product speaks.
        "file:///etc/passwd",
        "ws://127.0.0.1:8787/v1/chat/completions",
    ],
)
def test_refused_routes(route):
    with pytest.raises(EvalContractError):
        ModelRoute.from_mapping(model_mapping(route=route))


def test_https_route_is_unchanged_by_the_relaxation():
    route = "https://api.openai.com/v1/chat/completions"
    model = ModelRoute.from_mapping(model_mapping(route=route, price_source="vendor"))
    assert model.to_json()["route"] == route


# ----------------------------------------------------------- O2 zero-rate


def test_zero_rate_local_route_round_trips():
    """No code change was needed for this: 0.0 was already a legal rate."""

    model = ModelRoute.from_mapping(model_mapping())
    assert (model.usd_per_1m_input, model.usd_per_1m_output, model.usd_per_1m_cached_input) == (
        0.0,
        0.0,
        0.0,
    )
    assert model.secret == "SYNTH_MLX_RL_TOKEN"
    assert model.price_source == "local-compute"
    assert ModelRoute.from_mapping(model.to_json()) == model


def test_negative_rate_is_still_refused():
    with pytest.raises(EvalContractError):
        ModelRoute.from_mapping(model_mapping(usd_per_1m_output=-0.1))


def test_local_recipe_still_declares_a_call_ceiling():
    """`max_usd` is vacuous at zero rates, so calls are the real ceiling."""

    recipe = get_recipe(MLX_RECIPE)
    assert recipe.budget is not None
    assert recipe.budget.max_llm_calls > 0


def test_craftax_local_recipe_is_bounded_and_retains_replay():
    recipe = get_recipe(CRAFTAX_MLX_RECIPE)
    assert recipe.policy_kind == "llm-policy.v1"
    assert recipe.screening_seeds == (101, 102)
    assert recipe.limits.max_parallel_trials == 1
    assert recipe.budget is not None and recipe.budget.max_llm_calls == 8
    assert set(recipe.target.required_artifacts) == {"trace", "replay"}
    assert recipe.models[0].id == "mlx-local-base"


def test_worker_manifest_can_override_only_a_declared_local_route(tmp_path):
    manifest = tmp_path / "worker.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "eval.worker-manifest.v1",
                "run_id": "opt_local",
                "recipe_id": CRAFTAX_MLX_RECIPE,
                "home": str(tmp_path / "home"),
                "candidate_set_path": str(tmp_path / "candidates.json"),
                "model_route_overrides": {
                    "mlx-local-base": "http://host.docker.internal:49152/v1/chat/completions"
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = WorkerManifest.load(manifest)
    assert loaded.model_route_overrides == {
        "mlx-local-base": "http://host.docker.internal:49152/v1/chat/completions"
    }


# --------------------------------------------------------- O3 mlx-lora.v1


def write_adapter(root: Path, *, adapter: bool = True, weights: bytes = b"lora-weights") -> Path:
    root.mkdir(parents=True)
    policy = {
        "schema_version": "eval.mlx-lora-policy.v1",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "adapter": adapter,
        "chat_template_digest": TEMPLATE_DIGEST,
        "thinking_mode": "off",
    }
    if adapter:
        policy["rank"] = 8
        (root / "adapter_config.json").write_text(
            json.dumps({"lora_layers": 8, "rank": 8}), encoding="utf-8"
        )
        (root / "adapters.safetensors").write_bytes(weights)
    (root / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return root


def test_adapter_candidate_is_read_and_summarized(tmp_path):
    policy = read_mlx_lora_policy(write_adapter(tmp_path / "ckpt"))
    assert policy.adapter and policy.rank == 8
    assert policy.base_model == "Qwen/Qwen3.5-0.8B"


def test_candidate_missing_adapter_config_is_refused(tmp_path):
    root = write_adapter(tmp_path / "ckpt")
    (root / "adapter_config.json").unlink()
    with pytest.raises(EvalContractError, match="adapter_config.json"):
        read_mlx_lora_policy(root)


def test_candidate_missing_policy_json_is_refused(tmp_path):
    root = write_adapter(tmp_path / "ckpt")
    (root / "policy.json").unlink()
    with pytest.raises(EvalContractError, match="policy.json"):
        read_mlx_lora_policy(root)


def test_adapter_free_base_is_legal(tmp_path):
    policy = read_mlx_lora_policy(write_adapter(tmp_path / "base", adapter=False))
    assert policy.adapter is False and policy.rank is None


def test_base_that_ships_adapter_bytes_is_refused(tmp_path):
    root = write_adapter(tmp_path / "base", adapter=False)
    (root / "adapters.safetensors").write_bytes(b"lora-weights")
    with pytest.raises(EvalContractError, match="adapter=false"):
        read_mlx_lora_policy(root)


def test_staging_addresses_an_adapter_by_its_bytes(tmp_path):
    home = make_home(tmp_path)
    write_adapter(tmp_path / "src" / "base", adapter=False)
    write_adapter(tmp_path / "src" / "ckpt20", weights=b"weights-20")
    write_adapter(tmp_path / "src" / "ckpt20-again", weights=b"weights-20")
    write_adapter(tmp_path / "src" / "final", weights=b"weights-40")
    candidate_set = stage_candidate_set(
        home,
        [
            CandidateSource(
                label=label,
                path=tmp_path / "src" / name,
                entrypoint="policy.json",
                kind=MLX_LORA_POLICY_KIND,
                is_baseline=label == "base",
            )
            for label, name in [
                ("base", "base"),
                ("ckpt20", "ckpt20"),
                ("ckpt20-copy", "ckpt20-again"),
                ("final", "final"),
            ]
        ],
    )
    digests = {c.label: c.artifact_digest for c in candidate_set.candidates}
    # Identical bytes collide; different adapters do not.
    assert digests["ckpt20"] == digests["ckpt20-copy"]
    assert len({digests["base"], digests["ckpt20"], digests["final"]}) == 3
    baseline = candidate_set.baseline
    assert baseline is not None and baseline.label == "base"
    assert baseline.metadata["mlx_lora"]["adapter"] is False


def test_staging_refuses_a_malformed_adapter(tmp_path):
    home = make_home(tmp_path)
    root = write_adapter(tmp_path / "src" / "ckpt")
    (root / "adapters.safetensors").unlink()
    with pytest.raises(EvalContractError, match="adapters.safetensors"):
        stage_candidate_set(
            home,
            [
                CandidateSource(
                    label="ckpt",
                    path=root,
                    entrypoint="policy.json",
                    kind=MLX_LORA_POLICY_KIND,
                )
            ],
        )


# ------------------------------------------------------ O5 catalog recipe


def test_local_recipe_is_shaped_for_a_host_proxy():
    recipe = get_recipe(MLX_RECIPE)
    assert recipe.task == "gsm8k"
    assert recipe.policy_kind == MLX_LORA_POLICY_KIND
    # `none` cannot reach a proxy on the host.
    assert recipe.target.network == "bridge"
    assert recipe.target.required_artifacts == ("trace",)
    assert recipe.models[0].route == LOCAL_ROUTE
    assert recipe.secrets == ("SYNTH_MLX_RL_TOKEN",)


def test_local_recipe_is_pinned_to_the_published_gsm8k_target():
    """Pinned to the digest the publish workflow recorded; a tag is never enough."""
    recipe = get_recipe(MLX_RECIPE)
    assert recipe.image == "ghcr.io/synth-laboratories/workshop-gsm8k-eval-target"
    assert recipe.image_digest == (
        "sha256:1954fb48382590744643a6716a897234eed8a65899297d72a1313e54c7c7ab5d"
    )
    assert recipe.available and recipe.unavailable_reason is None
    assert recipe.pinned_reference == f"{recipe.image}@{recipe.image_digest}"


# ----------------------------------------------- O4 snapshot registration


class FakeRegistrar:
    """Stands in for `synth-mlx-rl`. No socket is opened anywhere in eval."""

    def __init__(self, *, drifting: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.drifting = drifting

    def register(self, *, candidate_id: str, artifact_digest: str, policy_dir: Path) -> str:
        assert (policy_dir / "policy.json").is_file()
        self.calls.append((candidate_id, artifact_digest))
        suffix = f"_{len(self.calls)}" if self.drifting else ""
        return f"snap_{artifact_digest.split(':', 1)[1][:16]}{suffix}"


class AccuracyExecutor:
    """Writes a conforming result for the GSM8K target contract."""

    def resolve_reference(self, image: str, digest: str) -> str:
        return f"{image}@{digest}"

    def run(self, request: TrialRunRequest, *, on_event, should_cancel, heartbeat):
        started = time.time()
        trial = json.loads((request.input_dir / "trial.json").read_text(encoding="utf-8"))
        (request.output_dir / "trace.jsonl").write_text(
            json.dumps({"trial_id": trial["trial_id"], "snapshot": trial["policy_snapshot_id"]})
            + "\n",
            encoding="utf-8",
        )
        (request.output_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "eval.container-result.v1",
                    "trial_id": trial["trial_id"],
                    "status": "evaluated",
                    "benchmark_status": "passed",
                    "metrics": {
                        "accuracy": 0.5 if trial["candidate"]["label"] == "base" else 0.7,
                        "reward": 1.0,
                        "steps": 10.0,
                    },
                    "gates": [
                        {"id": "policy_loaded", "passed": True},
                        {"id": "verifier_completed", "passed": True},
                    ],
                    "usage": {"cost_usd": 0.0, "rollouts": 1, "wall_time_ms": 5},
                    "artifacts": [{"role": "trace", "path": "trace.jsonl"}],
                }
            ),
            encoding="utf-8",
        )
        return TrialExecution(
            exit_code=0,
            timed_out=False,
            cancelled=False,
            started_at=started,
            finished_at=time.time(),
            stderr_tail="",
        )


def make_home(tmp_path: Path) -> EvalHome:
    root = tmp_path / "evalhome"
    EvalHome.open(root)
    (root / "runtime.toml").write_text(
        'container_runtime = "docker"\nmax_concurrent_trials = 2\nlease_ttl_seconds = 30\n',
        encoding="utf-8",
    )
    (root / "secrets.toml").write_text(
        '[secrets]\nSYNTH_MLX_RL_TOKEN = "run-local-bearer"\n', encoding="utf-8"
    )
    home = EvalHome.open(root)
    home.write_pin(MLX_RECIPE, PINNED)
    return home


def make_runner(tmp_path: Path, home: EvalHome, *, registrar=None) -> EvalRunner:
    write_adapter(tmp_path / "src" / "base", adapter=False)
    write_adapter(tmp_path / "src" / "final", weights=b"weights-40")
    candidate_set = stage_candidate_set(
        home,
        [
            CandidateSource(
                label=label,
                path=tmp_path / "src" / label,
                entrypoint="policy.json",
                kind=MLX_LORA_POLICY_KIND,
                is_baseline=label == "base",
            )
            for label in ("base", "final")
        ],
    )
    manifest_path = tmp_path / "worker.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "eval.worker-manifest.v1",
                "run_id": "run_mlx_1",
                "recipe_id": MLX_RECIPE,
                "home": str(home.root),
                "candidate_set_path": str(
                    home.candidates_dir / candidate_set.id / "candidate_set.json"
                ),
                "session_ref": "session_test",
            }
        ),
        encoding="utf-8",
    )
    return EvalRunner(
        WorkerManifest.load(manifest_path),
        executor=AccuracyExecutor(),
        policy_registrar=registrar,
    )


def test_trial_carries_an_immutable_snapshot_id(tmp_path):
    home = make_home(tmp_path)
    registrar = FakeRegistrar()
    runner = make_runner(tmp_path, home, registrar=registrar)
    assert runner.execute() == 0

    snapshots: dict[str, set[str]] = {}
    for path in sorted((home.run_dir("run_mlx_1") / "trials").glob("*/input/trial.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        snapshot_id = trial["policy_snapshot_id"]
        assert snapshot_id
        # Derived from the candidate's content address, never a run-local name.
        assert (
            trial["candidate"]["digest"]
            .split(":", 1)[1]
            .startswith(snapshot_id.removeprefix("snap_"))
        )
        snapshots.setdefault(trial["candidate"]["label"], set()).add(snapshot_id)
    assert {label: len(ids) for label, ids in snapshots.items()} == {"base": 1, "final": 1}
    assert len(registrar.calls) == 20  # every trial re-pins before it runs


def terminal_error(home: EvalHome, run_id: str) -> str:
    path = home.run_dir(run_id) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    terminal = [e for e in events if e["event"] == "eval.run.terminal"]
    assert terminal and terminal[-1]["status"] == "failed"
    return terminal[-1]["error"]


def test_a_drifting_snapshot_id_is_refused(tmp_path):
    """A snapshot id that changes mid-run means the service handed out a name."""

    home = make_home(tmp_path)
    runner = make_runner(tmp_path, home, registrar=FakeRegistrar(drifting=True))
    assert runner.execute() == 1
    assert "must be immutable" in terminal_error(home, "run_mlx_1")


def test_adapter_candidates_need_a_registrar(tmp_path):
    """Fails before any container starts, not after scoring an unpinned policy."""

    home = make_home(tmp_path)
    runner = make_runner(tmp_path, home, registrar=None)
    assert runner.execute() == 1
    assert "policy snapshot registrar" in terminal_error(home, "run_mlx_1")
    assert not (home.run_dir("run_mlx_1") / "trials").exists()


def test_the_default_path_registers_nothing(tmp_path):
    """No registrar configured means no call and an explicitly absent id.

    The snapshot lane is opt-in: a recipe that does not need one must not
    acquire a dependency on a running service just by existing.
    """

    home = make_home(tmp_path)
    home.write_pin(FIXTURE_RECIPE, PINNED)
    source = tmp_path / "src" / "code"
    source.mkdir(parents=True)
    (source / "policy.py").write_text("class Policy:\n    pass\n", encoding="utf-8")
    candidate_set = stage_candidate_set(
        home,
        [CandidateSource(label="only", path=source, entrypoint="policy:Policy")],
    )
    manifest_path = tmp_path / "worker-fixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "eval.worker-manifest.v1",
                "run_id": "run_fixture_1",
                "recipe_id": FIXTURE_RECIPE,
                "home": str(home.root),
                "candidate_set_path": str(
                    home.candidates_dir / candidate_set.id / "candidate_set.json"
                ),
                "session_ref": "session_test",
            }
        ),
        encoding="utf-8",
    )
    runner = EvalRunner(WorkerManifest.load(manifest_path), executor=AccuracyExecutor())
    assert runner.execute() == 0
    trials = sorted((home.run_dir("run_fixture_1") / "trials").glob("*/input/trial.json"))
    assert trials
    for path in trials:
        assert json.loads(path.read_text(encoding="utf-8"))["policy_snapshot_id"] is None
