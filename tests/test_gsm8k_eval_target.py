"""The GSM8K eval target's contract, exercised without Docker or MLX.

`docker/gsm8k-eval-target/target.py` is imported with its paths redirected to
temp dirs, a stub `gsm8k_world` stands in for the vendored containers module
(the real one is vendored at build time; its parser is containers' to test),
and a local HTTP server plays the synth-mlx-rl route. What is under test is
the target's side of `eval.target.v1`: rig failure vs policy outcome, the
snapshot pin, the recipe-owned route, the bearer, and the trace.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

TARGET_PATH = Path(__file__).resolve().parents[1] / "docker" / "gsm8k-eval-target" / "target.py"
SNAPSHOT = "snap_0123456789abcdef"
SECRET = "SYNTH_MLX_RL_TOKEN"


class _Row:
    def __init__(self, question: str, answer_text: str) -> None:
        self.question = question
        self.answer_text = answer_text

    @property
    def answer(self) -> str:
        return self.answer_text.rsplit("####", 1)[-1].strip()


class _Parsed:
    def __init__(self, value, source):
        self.value, self.source, self.raw = value, source, ""

    parsed = property(lambda self: self.value is not None)
    parse_mode = property(lambda self: {"hash_marker": "exact", "trailing_number": "trailing_number"}.get(self.source, "unparsed"))
    format_compliant = property(lambda self: self.parse_mode == "exact")


def _stub_world(monkeypatch, *, rows: dict[int, _Row], declared: list) -> types.ModuleType:
    world = types.ModuleType("gsm8k_world")
    world.HELDOUT_SPLIT = "heldout"
    world.TRAIN_SPLIT = "train"
    world.HF_REVISION = "740312add88f781978c0658806c59bc2815b9866"
    world.ParsedAnswer = _Parsed

    def declare_profile(name, *, snapshot_dir=None):
        declared.append((name, str(snapshot_dir)))
        if not (Path(str(snapshot_dir)) / "test.jsonl").is_file():
            raise RuntimeError("gsm8k_snapshot_invalid:test.jsonl")

    def dataset_manifest():
        return {
            "dataset": "openai/gsm8k", "config": "main", "revision": world.HF_REVISION,
            "profile": "snapshot", "profile_source": "declared", "pinned": True,
            "splits": {"heldout": {"hf_split": "test", "rows": len(rows), "digest": "sha256:" + "0" * 64}},
            "shuffle_seed": 20260820, "parse_modes": ["exact", "trailing_number", "unparsed"],
        }

    def parse_answer(text):
        if "####" in text:
            return _Parsed(text.rsplit("####", 1)[-1].strip(), "hash_marker")
        digits = [token for token in text.replace(".", " ").split() if token.isdigit()]
        return _Parsed(digits[-1], "trailing_number") if digits else _Parsed(None, "unparsed")

    world.declare_profile = declare_profile
    world.dataset_manifest = dataset_manifest
    world.load_row = lambda split, seed: rows.get(seed)
    world.parse_answer = parse_answer
    world.public_observation = lambda row, *, seed, split: {
        "question": row.question, "seed": seed, "split": split,
        "system": "SYSTEM PROMPT", "prompt": f"Problem:\n{row.question}",
    }
    monkeypatch.setitem(sys.modules, "gsm8k_world", world)
    return world


class _Route:
    """A loopback stand-in for synth-mlx-rl's /v1/chat/completions."""

    def __init__(self, *, text: str, served_snapshot: str | None = SNAPSHOT, status: int = 200) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        route = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                route.requests.append(body)
                route.headers.append({k.lower(): v for k, v in self.headers.items()})
                if status != 200:
                    self.send_response(status); self.end_headers(); self.wfile.write(b"{}"); return
                payload = {
                    "choices": [{"message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
                    "synth": {
                        "proxy_request_ids": ["prid_test"],
                        "policy_snapshot_id": served_snapshot,
                        "training_version": 1, "api_family": "chat_completions",
                        "tokenizer_digest": "t", "template_digest": "p", "render_digest": "r",
                    },
                }
                data = json.dumps(payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def rig(monkeypatch, tmp_path):
    rows = {0: _Row("What is 2 + 2?", "2 + 2 = 4\n#### 4")}
    declared: list = []
    _stub_world(monkeypatch, rows=rows, declared=declared)
    spec = importlib.util.spec_from_file_location("gsm8k_target_under_test", TARGET_PATH)
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    snapshot_dir = tmp_path / "gsm8k"
    snapshot_dir.mkdir()
    (snapshot_dir / "test.jsonl").write_text("{}\n")
    (snapshot_dir / "train.jsonl").write_text("{}\n")
    inp, out = tmp_path / "input", tmp_path / "output"
    (inp / "policy").mkdir(parents=True)
    (inp / "policy" / "policy.json").write_text(json.dumps({
        "schema_version": "eval.mlx-lora-policy.v1", "base_model": "Qwen/Qwen3.5-2B",
        "adapter": False, "chat_template_digest": "sha256:" + "ab" * 32, "thinking_mode": "off",
    }))
    monkeypatch.setattr(target, "INPUT", inp)
    monkeypatch.setattr(target, "OUTPUT", out)
    monkeypatch.setattr(target, "SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setenv(SECRET, "local-token")
    routes: list[_Route] = []

    def write_trial(**overrides):
        route_url = overrides.pop("route_url", None)
        trial = {
            "schema_version": "eval.trial.v1", "run_id": "run", "trial_id": "trial-1", "stage": "screen",
            "seed": 0, "scenario": "gsm8k-test",
            "candidate": {"id": "cand", "label": "base", "kind": "mlx-lora.v1", "digest": "sha256:" + "cd" * 32, "entrypoint": None},
            "policy_snapshot_id": SNAPSHOT,
            "models": [{"id": "mlx-local-base", "route": route_url, "secret": SECRET, "efforts": []}],
            "budget": {"max_llm_calls": 40, "max_usd": 0.01},
            "limits": {"timeout_seconds": 60, "max_output_bytes": 1 << 20},
        }
        trial.update(overrides)
        (inp / "trial.json").write_text(json.dumps(trial))
        return trial

    def run(route: _Route | None, **overrides):
        if route is not None:
            routes.append(route)
            overrides.setdefault("route_url", route.url)
        write_trial(**overrides)
        assert target.main() == 0
        return json.loads((out / "result.json").read_text())

    yield types.SimpleNamespace(run=run, out=out, declared=declared, snapshot_dir=snapshot_dir, target=target)
    for route in routes:
        route.close()


def _gate(result, gate_id):
    return {g["id"]: g["passed"] for g in result["gates"]}.get(gate_id)


def test_a_correct_marked_answer_passes_with_the_pin_verified(rig):
    route = _Route(text="2 + 2 = 4\n#### 4")
    result = rig.run(route)
    assert (result["status"], result["benchmark_status"]) == ("evaluated", "passed")
    assert result["metrics"] == {"accuracy": 1.0}
    assert _gate(result, "policy_loaded") and _gate(result, "verifier_completed") and _gate(result, "snapshot_pinned")
    assert rig.declared == [("snapshot", str(rig.snapshot_dir))]
    # The request is pinned to the snapshot in both the body and the header,
    # carries the bearer, and never names anything but the recipe's route.
    body, headers = route.requests[0], route.headers[0]
    assert body["policy_snapshot_id"] == SNAPSHOT and headers["x-policy-pin"] == SNAPSHOT
    assert headers["authorization"] == "Bearer local-token"
    assert body["messages"][0]["content"] == "SYSTEM PROMPT" and body["temperature"] == 0.0
    trace = json.loads((rig.out / "trace.jsonl").read_text())
    assert trace["parse"] == {"value": "4", "source": "hash_marker", "mode": "exact", "format_compliant": True}
    assert trace["proxy_request_ids"] == ["prid_test"] and trace["reference"] == "4"
    assert result["evidence"]["dataset"]["revision"] == "740312add88f781978c0658806c59bc2815b9866"
    assert result["evidence"]["proxy_request_ids"] == ["prid_test"]
    assert {a["role"] for a in result["artifacts"]} == {"trace", "events", "dataset"}
    assert result["usage"]["cost_usd"] == 0.0 and result["usage"]["completion_tokens"] == 9


def test_the_fallback_parse_is_counted_but_not_format_compliance(rig):
    result = rig.run(_Route(text="Two plus two makes 4"))
    assert result["benchmark_status"] == "passed" and result["metrics"]["accuracy"] == 1.0
    assert result["evidence"]["parse_mode"] == "trailing_number"
    assert result["evidence"]["format_compliant"] is False


@pytest.mark.parametrize("text", ["2 + 2 = 5\n#### 5", "I would rather not say."])
def test_a_wrong_or_unparseable_answer_is_the_policys_failure_not_the_rigs(rig, text):
    result = rig.run(_Route(text=text))
    assert (result["status"], result["benchmark_status"]) == ("evaluated", "failed")
    assert result["metrics"] == {"accuracy": 0.0}
    assert _gate(result, "verifier_completed") is True


def test_serving_another_snapshot_invalidates_the_trial(rig):
    result = rig.run(_Route(text="#### 4", served_snapshot="snap_other"))
    assert (result["status"], result["benchmark_status"]) == ("evaluated", "invalid")
    assert result["metrics"] == {} and _gate(result, "snapshot_pinned") is False
    assert "snap_other" in result["error"]


def test_a_dead_route_is_a_rig_failure_and_never_scored(rig):
    result = rig.run(_Route(text="", status=503))
    assert result["status"] == "failed" and result["benchmark_status"] is None
    assert "HTTP 503" in result["error"] and result["metrics"] == {}


def test_no_snapshot_id_means_the_policy_was_never_loaded(rig):
    result = rig.run(_Route(text="#### 4"), policy_snapshot_id=None)
    assert (result["status"], result["benchmark_status"]) == ("evaluated", "invalid")
    assert _gate(result, "policy_loaded") is False and "PolicySnapshotRegistrar" in result["error"]


def test_the_route_must_be_the_recipes_local_one(rig):
    result = rig.run(None, route_url="http://exfil.example/v1/chat/completions")
    assert result["benchmark_status"] == "invalid" and _gate(result, "policy_loaded") is False
    assert "only name this machine" in result["error"]


def test_a_missing_secret_is_refused_before_the_network(rig, monkeypatch):
    monkeypatch.delenv(SECRET)
    route = _Route(text="#### 4")
    result = rig.run(route)
    assert result["benchmark_status"] == "invalid" and not route.requests


def test_a_broken_snapshot_dir_is_a_rig_failure(rig):
    (rig.snapshot_dir / "test.jsonl").unlink()
    result = rig.run(_Route(text="#### 4"))
    assert result["status"] == "failed" and "gsm8k_snapshot_invalid" in result["error"]


def test_a_seed_outside_the_split_is_invalid_not_failed(rig):
    result = rig.run(_Route(text="#### 4"), seed=99)
    assert (result["status"], result["benchmark_status"]) == ("evaluated", "invalid")
