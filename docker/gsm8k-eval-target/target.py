"""GSM8K `eval.target.v1` target for `mlx-lora.v1` candidates.

One trial is one policy snapshot on one seed of the pinned GSM8K test split,
scored by exact match on the parsed numeric answer with the same world module
(and therefore the same parser) the containers `gsm8k_solve` target uses.

The policy is not in this image and is never loaded here. The host registers
the candidate adapter with its synth-mlx-rl service and this container is told
only the immutable `policy_snapshot_id` and the recipe-owned route; the chat
completion it sends is pinned to that snapshot, and the response's `synth`
block is checked to say the same snapshot back. A trial that sampled some
other snapshot is reported as such, not scored.

The dataset is baked in at build time from `openai/gsm8k` at one revision and
verified against the split digests recorded in the world module before the
image exists (see stage.py); `declare_profile("snapshot", …)` verifies them
again at trial start, so a trial runs on the pinned rows or not at all.

Rig health and policy outcome travel separately: a dead route or an invalid
snapshot dir is `status: "failed"` (the rig); a wrong or unparseable answer is
`status: "evaluated"` with `benchmark_status: "failed"` (the policy). Every
trace record carries `parse_mode`, because a trial scored through the
last-number fallback is counted but is not format compliance.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, "/opt/eval")
import gsm8k_world as world  # noqa: E402

INPUT = Path("/input")
OUTPUT = Path("/output")
SNAPSHOT_DIR = Path("/opt/gsm8k")

POLICY_KIND = "mlx-lora.v1"
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 600.0
#: A cleartext route may only name this machine, as seen from inside Docker.
LOCAL_HOSTS = frozenset({"host.docker.internal", "gateway.docker.internal", "127.0.0.1", "localhost", "::1"})

SCENARIOS: dict[str, dict[str, Any]] = {
    "gsm8k-test": {"split": world.HELDOUT_SPLIT, "description": "openai/gsm8k main/test, pinned"},
}


class RigError(RuntimeError):
    """The rig could not evaluate the trial. Never scored."""


class CandidateError(ValueError):
    """The candidate does not satisfy the target's policy contract."""


def emit(event: str, **fields: Any) -> None:
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "at": time.time(), **fields}) + "\n")
        handle.flush()


def write_result(
    trial_id: str,
    *,
    status: str,
    benchmark_status: str | None,
    metrics: dict[str, float],
    gates: list[dict[str, Any]],
    started: float,
    usage: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "eval.container-result.v1",
        "trial_id": trial_id,
        "status": status,
        "benchmark_status": benchmark_status,
        "metrics": metrics,
        "gates": gates,
        "usage": {
            "rollouts": 1,
            "wall_time_ms": int((time.time() - started) * 1000),
            "cost_usd": 0.0,
            **(usage or {}),
        },
        "artifacts": [
            entry
            for entry in (
                {"role": "trace", "path": "trace.jsonl"},
                {"role": "events", "path": "events.jsonl"},
                {"role": "dataset", "path": "dataset.json"},
            )
            if (OUTPUT / entry["path"]).is_file()
        ],
        "evidence": evidence or {},
    }
    if error:
        payload["error"] = error
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_route(trial: dict[str, Any]) -> dict[str, Any]:
    """The one recipe-owned route this trial may call. Anything else is refused."""
    models = trial.get("models") or []
    if len(models) != 1:
        raise CandidateError(f"the GSM8K target expects exactly one allowlisted route, got {len(models)}")
    route = models[0]
    url = str(route.get("route") or "")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.query:
        raise CandidateError(f"route {url!r} is not a plain http(s) URL")
    if parsed.scheme == "http" and parsed.hostname.lower() not in LOCAL_HOSTS:
        raise CandidateError(f"a cleartext route may only name this machine, not {parsed.hostname!r}")
    if not parsed.path.endswith("/chat/completions"):
        raise CandidateError("the GSM8K target speaks /chat/completions")
    secret = str(route.get("secret") or "")
    if not secret or not os.environ.get(secret, "").strip():
        raise CandidateError(f"{secret or 'the route secret'} is not present in the trial container")
    return {"id": route.get("id"), "url": url, "secret": secret}


def read_policy(trial: dict[str, Any]) -> dict[str, Any]:
    candidate = trial.get("candidate") or {}
    if candidate.get("kind") != POLICY_KIND:
        raise CandidateError(f"the GSM8K target scores {POLICY_KIND} candidates, not {candidate.get('kind')!r}")
    snapshot_id = trial.get("policy_snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise CandidateError(
            "no policy_snapshot_id: the host must register the candidate with its "
            "synth-mlx-rl service before the trial (PolicySnapshotRegistrar)"
        )
    manifest_path = INPUT / "policy" / "policy.json"
    if not manifest_path.is_file():
        raise CandidateError("an mlx-lora.v1 candidate must contain policy.json")
    policy = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "snapshot_id": snapshot_id.strip(),
        "base_model": str(policy.get("base_model") or ""),
        "adapter": bool(policy.get("adapter")),
        "thinking_mode": str(policy.get("thinking_mode") or "off"),
        "rank": policy.get("rank"),
    }


def sample(route: dict[str, Any], policy: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    body = {
        "model": policy["base_model"] or "mlx-local",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "policy_snapshot_id": policy["snapshot_id"],
        "enable_thinking": policy["thinking_mode"] == "on",
    }
    request = urllib.request.Request(
        route["url"],
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ[route['secret']].strip()}",
            "X-Policy-Pin": policy["snapshot_id"],
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            raw = response.read(8_388_608)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise RigError(f"policy route answered HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RigError(f"policy route unreachable: {type(error).__name__}: {error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RigError("policy route returned a body that is not JSON") from error
    if not isinstance(payload, dict):
        raise RigError("policy route returned a body that is not an object")
    return payload


def completion_text(payload: dict[str, Any]) -> str:
    for choice in payload.get("choices") or []:
        text = ((choice or {}).get("message") or {}).get("content")
        if isinstance(text, str) and text:
            return text
    return ""


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trial = json.loads((INPUT / "trial.json").read_text(encoding="utf-8"))
    trial_id = trial["trial_id"]
    seed = int(trial["seed"])
    started = time.time()
    gates: list[dict[str, Any]] = []
    emit("trial.started", trial_id=trial_id, seed=seed, scenario=trial["scenario"])

    scenario = SCENARIOS.get(trial["scenario"])
    if scenario is None:
        write_result(
            trial_id, status="failed", benchmark_status=None, metrics={}, gates=gates, started=started,
            error=f"unknown scenario {trial['scenario']!r} for the GSM8K target",
        )
        return 0

    try:
        world.declare_profile("snapshot", snapshot_dir=SNAPSHOT_DIR)
        row = world.load_row(scenario["split"], seed)  # verifies the split digest on first load
        dataset = world.dataset_manifest()
    except Exception as error:  # noqa: BLE001 - the rig's own dataset is wrong; never score
        write_result(
            trial_id, status="failed", benchmark_status=None, metrics={}, gates=gates, started=started,
            error=f"pinned dataset unavailable: {type(error).__name__}: {error}",
        )
        return 0
    (OUTPUT / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    evidence: dict[str, Any] = {
        "dataset": {key: dataset[key] for key in ("dataset", "config", "revision", "splits", "shuffle_seed")},
        "scenario": trial["scenario"],
        "seed": seed,
    }
    if row is None:
        write_result(
            trial_id, status="evaluated", benchmark_status="invalid", metrics={}, gates=gates, started=started,
            evidence=evidence, error=f"seed {seed} is outside the pinned {scenario['split']} split",
        )
        return 0

    try:
        policy = read_policy(trial)
        route = resolve_route(trial)
        gates.append({"id": "policy_loaded", "passed": True})
    except CandidateError as error:
        gates.append({"id": "policy_loaded", "passed": False})
        write_result(
            trial_id, status="evaluated", benchmark_status="invalid", metrics={}, gates=gates, started=started,
            evidence=evidence, error=str(error),
        )
        return 0
    evidence["policy_snapshot_id"] = policy["snapshot_id"]
    evidence["route_id"] = route["id"]

    observation = world.public_observation(row, seed=seed, split=scenario["split"])
    messages = [
        {"role": "system", "content": observation["system"]},
        {"role": "user", "content": observation["prompt"]},
    ]
    emit("rollout.started", seed=seed, policy_snapshot_id=policy["snapshot_id"], route=route["id"])
    sampled_at = time.time()
    try:
        payload = sample(route, policy, messages)
    except RigError as error:
        gates.append({"id": "verifier_completed", "passed": False})
        write_result(
            trial_id, status="failed", benchmark_status=None, metrics={}, gates=gates, started=started,
            evidence=evidence, error=str(error),
        )
        return 0
    llm_seconds = round(time.time() - sampled_at, 3)

    synth = payload.get("synth") if isinstance(payload.get("synth"), dict) else {}
    served_snapshot = synth.get("policy_snapshot_id")
    pinned = served_snapshot == policy["snapshot_id"]
    gates.append({"id": "snapshot_pinned", "passed": pinned})
    evidence["proxy_request_ids"] = list(synth.get("proxy_request_ids") or [])
    evidence["served_policy_snapshot_id"] = served_snapshot
    text = completion_text(payload)
    parsed = world.parse_answer(text) if text else world.ParsedAnswer(None, "absent", "")
    reference = row.answer
    usage_block = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    usage = {
        "calls": 1,
        "prompt_tokens": int(usage_block.get("prompt_tokens") or 0),
        "completion_tokens": int(usage_block.get("completion_tokens") or 0),
        "llm_seconds": llm_seconds,
    }

    if not text:
        # Nothing was produced: an absent signal, not a zero. `accuracy` stays
        # missing and the trial is invalid rather than counted against the policy.
        gates.append({"id": "verifier_completed", "passed": False})
        correct = None
    elif not pinned:
        gates.append({"id": "verifier_completed", "passed": False})
        correct = None
    else:
        gates.append({"id": "verifier_completed", "passed": True})
        # An unparseable completion is a failed attempt and scores 0.0 — the
        # policy answered; it did not answer with a number (containers rule).
        correct = 1.0 if parsed.parsed and parsed.value == reference else 0.0

    with (OUTPUT / "trace.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "seed": seed,
                    "scenario": trial["scenario"],
                    "split": scenario["split"],
                    "policy_snapshot_id": policy["snapshot_id"],
                    "served_policy_snapshot_id": served_snapshot,
                    "proxy_request_ids": evidence["proxy_request_ids"],
                    "messages": messages,
                    "completion": text,
                    "parse": {
                        "value": parsed.value,
                        "source": parsed.source,
                        "mode": parsed.parse_mode,
                        "format_compliant": parsed.format_compliant,
                    },
                    "reference": reference,
                    "correct": correct,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "usage": usage,
                    "synth": synth,
                }
            )
            + "\n"
        )
    emit(
        "rollout.finished",
        correct=correct,
        parse_mode=parsed.parse_mode,
        format_compliant=parsed.format_compliant,
        snapshot_pinned=pinned,
        completion_tokens=usage["completion_tokens"],
    )
    evidence["parse_mode"] = parsed.parse_mode
    evidence["format_compliant"] = parsed.format_compliant

    if correct is None:
        write_result(
            trial_id, status="evaluated", benchmark_status="invalid", metrics={}, gates=gates, started=started,
            usage=usage, evidence=evidence,
            error=(
                "policy produced no completion"
                if not text
                else f"route served snapshot {served_snapshot!r}, not the pinned {policy['snapshot_id']!r}"
            ),
        )
        return 0
    write_result(
        trial_id,
        status="evaluated",
        benchmark_status="passed" if correct == 1.0 else "failed",
        metrics={"accuracy": correct},
        gates=gates,
        started=started,
        usage=usage,
        evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
