#!/usr/bin/env python3
"""Verifier-backed Harbor TBLite CISPO training and paired evaluation on Tinker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
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
        self._routes: dict[str, tuple[ProviderCheckpoint, str, int]] = {}
        self._routes_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                try:
                    route = self.path.strip("/").split("/")[1]
                    with owner._routes_lock:
                        checkpoint, checkpoint_digest, policy_version = owner._routes[route]
                    size = int(self.headers.get("content-length", "0"))
                    payload = json.loads(self.rfile.read(size))
                    messages = list(payload.get("messages") or [])
                    max_tokens = int(payload.get("max_tokens") or 4096)
                    rendered = owner.adapter.tokenize_chat(messages, add_generation_prompt=True)
                    prompt = tuple(int(token) for token in rendered["prompt_token_ids"])
                    removed_messages = 0
                    while len(prompt) + max_tokens > 32_768 and len(messages) > 3:
                        del messages[2]
                        removed_messages += 1
                        rendered = owner.adapter.tokenize_chat(
                            messages, add_generation_prompt=True
                        )
                        prompt = tuple(int(token) for token in rendered["prompt_token_ids"])
                    max_tokens = min(max_tokens, max(1, 32_768 - len(prompt)))
                    started = time.monotonic()
                    sample_request = SampleRequest(
                        request_id=new_request_id("harbor", self.path, str(time.time_ns())),
                        prompt_token_ids=prompt, max_tokens=max_tokens,
                        temperature=float(payload.get("temperature", 0.8)),
                        seed=int(payload.get("seed", 0)),
                    )
                    sampled = owner.adapter.sample_checkpoint(checkpoint, sample_request)
                    elapsed = max(time.monotonic() - started, 1e-9)
                    content = owner.adapter.decode_tokens(sampled.token_ids)
                    capture = {
                        "prompt_tokens": len(prompt), "completion_tokens": len(sampled.token_ids),
                        "sampling_seconds": elapsed,
                        "completion_tokens_per_second": len(sampled.token_ids) / elapsed,
                        "prompt_token_ids": list(prompt), "generation_token_ids": list(sampled.token_ids),
                        "generation_logprobs": list(sampled.logprobs),
                        "generation_loss_mask": [1] * len(sampled.token_ids),
                        "checkpoint_digest": checkpoint_digest,
                        "behavior_policy_version": policy_version,
                        "gateway_compacted_messages": removed_messages,
                    }
                    result = {"choices": [{"message": {"role": "assistant", "content": content}}],
                              "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(sampled.token_ids),
                                        "total_tokens": len(prompt) + len(sampled.token_ids)},
                              "synth_capture": capture}
                    encoded = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                except Exception as exc:  # noqa: BLE001
                    encoded = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    def start(self) -> None:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def register_checkpoint(
        self, route: str, checkpoint: ProviderCheckpoint, policy_version: int
    ) -> str:
        digest = hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest()
        with self._routes_lock:
            existing = self._routes.get(route)
            value = (checkpoint, digest, policy_version)
            if existing is not None and existing != value:
                raise RuntimeError(f"checkpoint route {route!r} is immutable")
            self._routes[route] = value
        return digest

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


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
    policy_versions = {int(call.get("behavior_policy_version", -1)) for call in calls}
    if len(checkpoints) != 1 or "" in checkpoints:
        raise RuntimeError("mixed or missing checkpoint identity")
    if len(policy_versions) != 1 or -1 in policy_versions:
        raise RuntimeError("mixed or missing behavior policy version")
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
    return {
        "segments": segments,
        "checkpoint_digest": checkpoints.pop(),
        "behavior_policy_version": policy_versions.pop(),
    }


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
            "compaction_threshold_tokens": 18000, "compaction_keep_messages": 4,
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
    gateway.register_checkpoint(phase, checkpoint, checkpoint.step)
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
    parser.add_argument(
        "--train-calls",
        type=int,
        default=0,
        help="Stop after this many non-skipped train_step calls; --steps is the safety ceiling.",
    )
    parser.add_argument("--cardinality", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--gateway-port", type=int, default=18110)
    parser.add_argument("--cleanup-docker", action="store_true")
    parser.add_argument(
        "--platform-id",
        help="Exact synth.parent label to reap when --cleanup-docker is enabled.",
    )
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--trained-checkpoint")
    parser.add_argument("--pipeline-mode", choices=("sync", "async"), default="async")
    parser.add_argument("--max-staleness", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=2)
    args = parser.parse_args()
    load_key(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    adapter = TinkerAdapter(TinkerCredentials.from_env())
    if bool(args.baseline_checkpoint) != bool(args.trained_checkpoint):
        raise SystemExit("both --baseline-checkpoint and --trained-checkpoint are required")
    if args.baseline_checkpoint and args.trained_checkpoint:
        # Bind the base-model tokenizer and Prime renderer. Checkpoint-only
        # sampling otherwise has no model identity and falls back to ASCII
        # tokenization/decoding in the provider adapter.
        adapter.create_session(
            "openai/gpt-oss-20b",
            rank=8,
            seed=0,
            request_id=new_request_id("tblite", "eval-tokenizer"),
        )

        def external_checkpoint(reference: str, step: int) -> ProviderCheckpoint:
            digest = hashlib.sha256(reference.encode()).hexdigest()
            return ProviderCheckpoint(
                checkpoint_id=f"external-{digest[:16]}",
                provider_reference=reference,
                step=step,
                digest=f"sha256:{digest}",
                kind="inference",
                resume_token=reference,
            )

        gateway = RolloutGateway(adapter, None, args.gateway_port)
        gateway.start()
        seeds = [10_000 + index for index in range(args.eval_seeds)]
        summary = {
            "schema_version": "harbor.tblite.paired_eval.v1",
            "evaluations": {},
            "model": "openai/gpt-oss-20b",
        }
        try:
            targets = (
                ("eval_baseline", external_checkpoint(args.baseline_checkpoint, 0)),
                ("eval_trained", external_checkpoint(args.trained_checkpoint, args.steps)),
            )
            for phase, target in targets:
                rows = evaluate(
                    base, gateway, target, args.gateway_port, phase, seeds, args.max_parallel
                )
                summary["evaluations"][phase] = {
                    "seeds": seeds,
                    "checkpoint_digest": hashlib.sha256(
                        target.provider_reference.encode()
                    ).hexdigest(),
                    "rollouts": [
                        {key: row[key] for key in ("rollout_id", "reward", "usage")}
                        for row in rows
                    ],
                }
                (args.output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print(json.dumps(summary["evaluations"][phase]), flush=True)
            print(json.dumps(summary, indent=2))
        finally:
            gateway.close()
        return
    session = adapter.create_session("openai/gpt-oss-20b", rank=8, seed=0, request_id=new_request_id("tblite", "session"))
    checkpoint = adapter.save_checkpoint(session, step=0, kind="inference", request_id=new_request_id("tblite", "initial"))
    baseline_checkpoint = checkpoint
    if args.max_staleness < 0:
        raise SystemExit("--max-staleness must be non-negative")
    if args.queue_depth < 1:
        raise SystemExit("--queue-depth must be positive")
    if args.train_calls < 0:
        raise SystemExit("--train-calls must be non-negative")
    if args.train_calls > args.steps:
        raise SystemExit("--train-calls cannot exceed the --steps safety ceiling")
    if args.pipeline_mode == "async" and args.max_staleness < args.queue_depth - 1:
        raise SystemExit("async pipeline requires max staleness >= queue depth - 1")
    gateway = RolloutGateway(adapter, session, args.gateway_port)
    gateway.start()
    summary = {
        "schema_version": "harbor.tblite.cispo.v3",
        "steps": [], "evaluations": {}, "model": session.model_id,
        "pipeline": {
            "mode": args.pipeline_mode,
            "max_staleness": args.max_staleness,
            "group_queue_capacity": args.queue_depth if args.pipeline_mode == "async" else 1,
        },
    }
    try:
        run_started = time.monotonic()
        train_calls = 0
        target_reached_wall_seconds: float | None = None

        def submit_group(
            pool: concurrent.futures.ThreadPoolExecutor,
            update: int,
            behavior_checkpoint: ProviderCheckpoint,
        ) -> tuple[int, float, list[concurrent.futures.Future[dict]]]:
            task = TASKS[(update - 1) % len(TASKS)]
            phase = f"u{update:02d}"
            gateway.register_checkpoint(phase, behavior_checkpoint, behavior_checkpoint.step)
            configs = [
                register_policy(
                    base, args.gateway_port, behavior_checkpoint.provider_reference,
                    phase, member, update * 1000 + member,
                )
                for member in range(args.cardinality)
            ]
            submitted = time.monotonic()
            futures = [
                pool.submit(rollout, base, configs[member], task, phase, member)
                for member in range(args.cardinality)
            ]
            return update, submitted, futures

        # Async groups share this pool. Allow workers beyond one group's
        # cardinality so the next bounded-staleness group can occupy spare
        # Harbor leases while the current group is still finishing.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            pending = deque()
            initial_depth = min(args.steps, args.queue_depth if args.pipeline_mode == "async" else 1)
            for queued_update in range(1, initial_depth + 1):
                pending.append(submit_group(pool, queued_update, checkpoint))
            next_update_to_submit = initial_depth + 1
            for update in range(1, args.steps + 1):
                queued_update, submitted, futures = pending.popleft()
                assert queued_update == update
                rows = [future.result() for future in futures]
                rollout_completed = time.monotonic()
                behavior_versions = {row["behavior_policy_version"] for row in rows}
                if len(behavior_versions) != 1:
                    raise RuntimeError("rollout group mixed behavior policy versions")
                behavior_version = behavior_versions.pop()
                staleness = (update - 1) - behavior_version
                if staleness < 0 or staleness > args.max_staleness:
                    raise RuntimeError(
                        f"group u{update:02d} staleness {staleness} exceeds "
                        f"bound {args.max_staleness}"
                    )
                task = TASKS[(update - 1) % len(TASKS)]
                rewards = [row["reward"] for row in rows]
                advantages = group_advantages(rewards)
                skipped = is_zero_advantage_group(advantages)
                train_started = time.monotonic()
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
                    train_calls += 1
                checkpoint = adapter.save_checkpoint(session, step=update, kind="inference", request_id=new_request_id("tblite", "checkpoint", str(update)))
                train_completed = time.monotonic()
                step = {"update": update, "task": task, "rewards": rewards, "advantages": list(advantages), "skipped": skipped,
                        "checkpoint_digest": hashlib.sha256(checkpoint.provider_reference.encode()).hexdigest(),
                        "behavior_policy_version": behavior_version, "staleness": staleness,
                        "train_call": not skipped, "train_calls_completed": train_calls,
                        "rollout_wall_seconds": rollout_completed - submitted,
                        "train_checkpoint_wall_seconds": train_completed - train_started,
                        "step_wall_seconds": train_completed - submitted,
                        "cumulative_wall_seconds": train_completed - run_started,
                        "rollouts": [{k: r[k] for k in ("rollout_id", "reward", "usage")} for r in rows]}
                summary["steps"].append(step)
                (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(step), flush=True)
                if args.train_calls and train_calls >= args.train_calls:
                    target_reached_wall_seconds = train_completed - run_started
                    for _, _, queued_futures in pending:
                        for future in queued_futures:
                            future.cancel()
                    break
                if next_update_to_submit <= args.steps:
                    pending.append(submit_group(pool, next_update_to_submit, checkpoint))
                    next_update_to_submit += 1
        summary["pipeline"]["training_wall_seconds"] = time.monotonic() - run_started
        summary["pipeline"]["train_calls_completed"] = train_calls
        summary["pipeline"]["train_calls_target"] = args.train_calls or None
        summary["pipeline"]["target_reached_wall_seconds"] = target_reached_wall_seconds
        seeds = [10_000 + index for index in range(args.eval_seeds)]
        if seeds:
            for phase, target in (("eval_baseline", baseline_checkpoint), ("eval_trained", checkpoint)):
                rows = evaluate(
                    base, gateway, target, args.gateway_port, phase, seeds, args.max_parallel
                )
                summary["evaluations"][phase] = {
                    "seeds": seeds,
                    "checkpoint_digest": hashlib.sha256(target.provider_reference.encode()).hexdigest(),
                    "rollouts": [{k: r[k] for k in ("rollout_id", "reward", "usage")} for r in rows],
                }
                print(json.dumps(summary["evaluations"][phase]), flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        gateway.close()
        if args.cleanup_docker:
            if not args.platform_id:
                raise RuntimeError("--cleanup-docker requires --platform-id")
            listed = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=synth.parent={args.platform_id}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
            if ids:
                subprocess.run(
                    ["docker", "rm", "-f", *ids],
                    capture_output=True,
                    check=False,
                )
            # This runner owns the platform instance on its explicitly selected
            # port when cleanup is requested. Remove that exact container too;
            # never match other synth-containers instances by image name.
            subprocess.run(
                ["docker", "rm", "-f", f"synth-harbor-tblite-{args.port}"],
                capture_output=True,
                check=False,
            )
            workspace_root = (
                Path.home()
                / ".synth-containers"
                / "work"
                / f"harbor-tblite-{args.port}"
            )
            if workspace_root.name != f"harbor-tblite-{args.port}":
                raise RuntimeError(f"refusing unsafe workspace cleanup: {workspace_root}")
            shutil.rmtree(workspace_root, ignore_errors=True)
            # TBLite rollouts run digest-pinned images directly and do not
            # build per-rollout images. Only untagged build leftovers are safe
            # to prune globally; tagged task images are the reusable cache.
            subprocess.run(
                ["docker", "image", "prune", "-f", "--filter", "dangling=true"],
                capture_output=True,
                check=False,
            )


if __name__ == "__main__":
    main()
