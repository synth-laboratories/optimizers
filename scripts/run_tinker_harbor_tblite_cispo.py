#!/usr/bin/env python3
"""Verifier-backed Harbor TBLite CISPO training and paired evaluation on Tinker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synth_optimizers.cispo import group_advantages, is_zero_advantage_group  # noqa: E402
from synth_optimizers.providers.protocols import (  # noqa: E402
    ProviderCheckpoint,
    SampleRequest,
    TrainingStepRequest,
)
from synth_optimizers.providers.tinker.client import (  # noqa: E402
    TinkerAdapter,
    TinkerCredentials,
    new_request_id,
)

TASKS = ("jsonl-aggregator", "log-summary", "pandas-etl", "schedule-vacation", "supply-chain-fulfillment")


class RolloutGateway:
    def __init__(self, adapter: TinkerAdapter, session: object, port: int) -> None:
        self.adapter, self.session, self.port = adapter, session, port
        self.checkpoint_digest = ""
        self.sample_checkpoint: ProviderCheckpoint | None = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                try:
                    size = int(self.headers.get("content-length", "0"))
                    payload = json.loads(self.rfile.read(size))
                    messages = payload.get("messages") or []
                    rendered = owner.adapter.tokenize_chat(messages, add_generation_prompt=True)
                    prompt = tuple(int(token) for token in rendered["prompt_token_ids"])
                    started = time.monotonic()
                    sample_request = SampleRequest(
                        request_id=new_request_id("harbor", self.path, str(time.time_ns())),
                        prompt_token_ids=prompt, max_tokens=int(payload.get("max_tokens") or 4096),
                        temperature=float(payload.get("temperature", 0.8)),
                        seed=int(payload.get("seed", 0)),
                    )
                    sampled = (
                        owner.adapter.sample_checkpoint(owner.sample_checkpoint, sample_request)
                        if owner.sample_checkpoint is not None
                        else owner.adapter.sample(owner.session, sample_request)
                    )
                    elapsed = max(time.monotonic() - started, 1e-9)
                    content = owner.adapter.decode_tokens(sampled.token_ids)
                    capture = {
                        "prompt_tokens": len(prompt), "completion_tokens": len(sampled.token_ids),
                        "sampling_seconds": elapsed,
                        "completion_tokens_per_second": len(sampled.token_ids) / elapsed,
                        "prompt_token_ids": list(prompt), "generation_token_ids": list(sampled.token_ids),
                        "generation_logprobs": list(sampled.logprobs),
                        "generation_loss_mask": [1] * len(sampled.token_ids),
                        "checkpoint_digest": owner.checkpoint_digest,
                    }
                    result = {"choices": [{"message": {"role": "assistant", "content": content}}],
                              "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(sampled.token_ids),
                                        "total_tokens": len(prompt) + len(sampled.token_ids)},
                              "synth_capture": capture}
                    encoded = json.dumps(result).encode()
                    self.send_response(200); self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
                except Exception as exc:  # noqa: BLE001
                    encoded = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                    self.send_response(500); self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    def start(self) -> None:
        import threading
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown(); self.server.server_close()


def request(base: str, method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3700) as response:
        return json.loads(response.read())


def load_key(path: Path) -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    for line in path.read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit("TINKER_API_KEY missing")


def assemble(calls: list[dict]) -> dict:
    if not calls:
        raise RuntimeError("trajectory has no engine calls")
    checkpoints = {str(call.get("checkpoint_digest") or "") for call in calls}
    if len(checkpoints) != 1 or "" in checkpoints:
        raise RuntimeError("mixed or missing checkpoint identity")
    segments = []
    for call in calls:
        prompt = list(call["prompt_token_ids"])
        generated = list(call["generation_token_ids"])
        behavior = list(call["generation_logprobs"])
        if not prompt or not generated or len(generated) != len(behavior):
            raise RuntimeError("renderer token/logprob alignment failure")
        full = prompt + generated
        segments.append({"token_ids": full, "loss_mask": [0] * len(prompt) + [1] * len(generated),
                         "behavior_logprobs": [0.0] * len(prompt) + behavior})
    return {"segments": segments, "checkpoint_digest": checkpoints.pop()}


def register_policy(
    base: str, gateway_port: int, checkpoint: str, phase: str, member: int, seed: int
) -> str:
    cid = f"tblite_cispo_{phase}_m{member:02d}_policy"
    request(base, "POST", "/policy-configs", {
        "config_id": cid, "harness": "mini_swe", "config": {
            "model": "openai/gpt-oss-20b", "model_path": checkpoint,
            "base_url": f"http://host.docker.internal:{gateway_port}/v1/{phase}/m{member}",
            "api_key_env": "TINKER_API_KEY", "inference_transport": "chat_completions",
            "max_steps": 50, "max_tokens": 4096, "temperature": 0.8,
            "sampling_seed": seed, "command_timeout_seconds": 300,
            "timeout_seconds": 3600, "output_limit": 6000,
            "workspace_aliases": ["/app", "/workdir"],
            "compaction_threshold_tokens": 28000, "compaction_keep_messages": 10,
        }})
    return cid


def rollout(base: str, cid: str, task: str, phase: str, member: int) -> dict:
    rid = f"tblite_cispo_{phase}_m{member:02d}_{int(time.time())}"
    status = request(base, "POST", "/rollouts", {
        "rollout_id": rid, "task_instance_id": f"tblite/{task}",
        "policy_ref": {"harness": "mini_swe", "config": cid},
        "telemetry": {"enabled": True, "transport": "poll"},
    })
    if status.get("status") != "completed" or status.get("reward") is None:
        raise RuntimeError(f"Harbor rollout failed: {rid}: {status.get('status')}")
    events = request(base, "GET", f"/rollouts/{rid}/events?after=0&limit=1000")["events"]
    policy = next(e["payload"] for e in events if e.get("kind") == "span.policy.data" and "throughput" in e.get("payload", {}))
    calls = list(policy["throughput"]["sample_calls"])
    return {"rollout_id": rid, "reward": float(status["reward"]), "usage": status.get("usage") or {}, "calls": calls, **assemble(calls)}


def evaluate(
    base: str,
    gateway: RolloutGateway,
    checkpoint: ProviderCheckpoint,
    gateway_port: int,
    phase: str,
    seeds: list[int],
    max_parallel: int,
) -> list[dict]:
    gateway.sample_checkpoint = checkpoint
    gateway.checkpoint_digest = hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest()
    configs = [
        register_policy(base, gateway_port, checkpoint.provider_reference, phase, member, seed)
        for member, seed in enumerate(seeds)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(seeds), max_parallel)) as pool:
        futures = [
            pool.submit(rollout, base, configs[member], TASKS[member % len(TASKS)], phase, member)
            for member in range(len(seeds))
        ]
        return [future.result() for future in futures]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cardinality", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--gateway-port", type=int, default=18110)
    args = parser.parse_args()
    load_key(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    adapter = TinkerAdapter(TinkerCredentials.from_env())
    session = adapter.create_session("openai/gpt-oss-20b", rank=8, seed=0, request_id=new_request_id("tblite", "session"))
    checkpoint = adapter.save_checkpoint(session, step=0, kind="inference", request_id=new_request_id("tblite", "initial"))
    baseline_checkpoint = checkpoint
    gateway = RolloutGateway(adapter, session, args.gateway_port)
    gateway.checkpoint_digest = hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest()
    gateway.start()
    gateway.sample_checkpoint = checkpoint
    summary = {"schema_version": "harbor.tblite.cispo.v2", "steps": [], "evaluations": {}, "model": session.model_id}
    try:
        for update in range(1, args.steps + 1):
            task = TASKS[(update - 1) % len(TASKS)]
            phase = f"u{update:02d}"
            configs = [register_policy(base, args.gateway_port, checkpoint.provider_reference, phase, member, update * 1000 + member) for member in range(args.cardinality)]
            gateway.sample_checkpoint = checkpoint
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.cardinality, args.max_parallel)
            ) as pool:
                futures = [pool.submit(rollout, base, configs[member], task, phase, member) for member in range(args.cardinality)]
                rows = [future.result() for future in futures]
            rewards = [row["reward"] for row in rows]
            advantages = group_advantages(rewards)
            skipped = is_zero_advantage_group(advantages)
            if not skipped:
                data = []
                for row, advantage in zip(rows, advantages, strict=True):
                    root_weight = 1.0 / len(row["segments"])
                    for segment in row["segments"]:
                        data.append({**segment, "advantages": [advantage * root_weight]})
                adapter.train_step(session, TrainingStepRequest(
                    request_id=new_request_id("tblite", "train", str(update)), loss_name="cispo.slime.v1",
                    data=tuple(data), metadata={"eps_clip": 1.0, "eps_clip_high": 4.0, "learning_rate": 5e-6},
                ))
            checkpoint = adapter.save_checkpoint(session, step=update, kind="inference", request_id=new_request_id("tblite", "checkpoint", str(update)))
            gateway.checkpoint_digest = hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest()
            step = {"update": update, "task": task, "rewards": rewards, "advantages": list(advantages), "skipped": skipped,
                    "checkpoint_digest": hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest(),
                    "rollouts": [{k: r[k] for k in ("rollout_id", "reward", "usage")} for r in rows]}
            summary["steps"].append(step)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(step), flush=True)
        seeds = [10_000 + index for index in range(args.eval_seeds)]
        for phase, target in (("eval_baseline", baseline_checkpoint), ("eval_trained", checkpoint)):
            rows = evaluate(
                base, gateway, target, args.gateway_port, phase, seeds, args.max_parallel
            )
            summary["evaluations"][phase] = {
                "seeds": seeds,
                "checkpoint_digest": hashlib.sha256(target.provider_reference.encode()).hexdigest(),
                "rollouts": [{k: r[k] for k in ("rollout_id", "reward", "usage")} for r in rows],
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary["evaluations"][phase]), flush=True)
        print(json.dumps(summary, indent=2))
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
