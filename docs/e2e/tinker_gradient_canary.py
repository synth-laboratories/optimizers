"""One-call paid proof that CISPO changes and restores Tinker parameters.

This is a diagnostic, not a training run.  It makes one optimizer call and
persists the exact pre/post/restore log-probability evidence needed to prove
that the executor-shaped ``advantage`` field reaches Tinker's CISPO loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from synth_optimizers.providers.protocols import (
    ForwardRequest,
    SampleRequest,
    TrainingStepRequest,
)
from synth_optimizers.providers.tinker.client import TinkerAdapter, TinkerCredentials

SCHEMA_VERSION = "tinker.cispo.gradient_canary.v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_ENV_FILE = "/Users/joshuapurtell/GitHub/frontend/.env.local"
DEFAULT_PROMPT = (
    "Classify the banking support request with one short intent label. "
    "Request: I was charged twice for the same card purchase."
)


class CanaryFailure(RuntimeError):
    """A completed canary failed one or more proof checks."""

    def __init__(self, failed: Sequence[str], evidence: Mapping[str, Any]) -> None:
        super().__init__(f"Tinker gradient canary failed checks: {list(failed)}")
        self.evidence = dict(evidence)


def _load_provider_environment(path: Path) -> None:
    """Load only Tinker variables from the already-authorized env file."""

    if os.environ.get("TINKER_API_KEY"):
        return
    allowed = {"TINKER_API_KEY", "TINKER_BASE_URL"}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        if name in allowed and name not in os.environ:
            os.environ[name] = value.strip().strip("\"'")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(f"no TINKER_API_KEY in {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _selected(values: Sequence[float], loss_mask: Sequence[bool]) -> tuple[float, ...]:
    shifted_mask = tuple(bool(value) for value in loss_mask[1:])
    if len(values) != len(shifted_mask):
        raise RuntimeError(
            f"forward result has {len(values)} values for {len(shifted_mask)} shifted tokens"
        )
    selected = tuple(float(value) for value, enabled in zip(values, shifted_mask, strict=True) if enabled)
    if not selected or not all(math.isfinite(value) for value in selected):
        raise RuntimeError("forward result has no finite selected-token log probabilities")
    return selected


def run_canary(
    provider: Any,
    *,
    run_id: str,
    model_id: str = DEFAULT_MODEL,
    rank: int = 32,
    learning_rate: float = 5e-5,
    advantage: float = 1.0,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if not math.isfinite(advantage) or advantage <= 0.0:
        raise ValueError("the movement canary requires a finite positive advantage")
    prefix = f"{run_id}-{uuid.uuid4().hex}"
    session = provider.create_session(
        model_id, rank=rank, seed=0, request_id=f"{prefix}-session"
    )
    rendered = provider.tokenize_chat(
        (
            {"role": "system", "content": "Return only the requested intent label."},
            {"role": "user", "content": prompt},
        ),
        add_generation_prompt=True,
    )
    prompt_tokens = tuple(int(value) for value in rendered["prompt_token_ids"])
    sampled = provider.sample(
        session,
        SampleRequest(
            request_id=f"{prefix}-sample",
            prompt_token_ids=prompt_tokens,
            max_tokens=32,
            temperature=0.0,
            seed=0,
        ),
    )
    completion = tuple(int(value) for value in sampled.token_ids)
    behavior_completion = tuple(float(value) for value in sampled.logprobs)
    if not completion or len(completion) != len(behavior_completion):
        raise RuntimeError("canary sampling returned missing or misaligned tokens/log probabilities")
    token_ids = prompt_tokens + completion
    loss_mask = (False,) * len(prompt_tokens) + (True,) * len(completion)
    behavior = (0.0,) * len(prompt_tokens) + behavior_completion

    def forward(request_id: str, active_session: Any) -> tuple[float, ...]:
        result = provider.forward(
            active_session,
            ForwardRequest(
                request_id=request_id,
                token_ids=(token_ids,),
                response_masks=(loss_mask,),
            ),
        )
        if len(result.logprobs) != 1:
            raise RuntimeError("canary forward returned the wrong batch width")
        return _selected(result.logprobs[0], loss_mask)

    pre = forward(f"{prefix}-forward-pre", session)
    baseline_sampler = provider.save_checkpoint(
        session, step=0, kind="sampler_weights", request_id=f"{prefix}-baseline-sampler"
    )
    baseline_state = provider.save_checkpoint(
        session, step=0, kind="training_state", request_id=f"{prefix}-baseline-state"
    )
    datum = {
        # This is the executor's canonical provider-facing shape.  In
        # particular, ``advantage`` is intentionally singular.
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "behavior_logprobs": behavior,
        "advantage": advantage,
        "root_rollout_weight": 1.0,
        "same_policy_weight": 1.0,
    }
    trained = provider.train_step(
        session,
        TrainingStepRequest(
            request_id=f"{prefix}-train",
            loss_name="cispo.slime.v1",
            data=(datum,),
            metadata={"learning_rate": learning_rate, "eps_clip": 1.0, "eps_clip_high": 4.0},
        ),
    )
    if trained.step != 1:
        raise RuntimeError(f"one optimizer call reported step {trained.step}, not 1")
    post = forward(f"{prefix}-forward-post", session)
    post_sampler = provider.save_checkpoint(
        session, step=1, kind="sampler_weights", request_id=f"{prefix}-post-sampler"
    )
    post_state = provider.save_checkpoint(
        session, step=1, kind="training_state", request_id=f"{prefix}-post-state"
    )
    restored_session = provider.restore_session(post_state, request_id=f"{prefix}-restore")
    restored = forward(f"{prefix}-forward-restored", restored_session)

    deltas = tuple(after - before for before, after in zip(pre, post, strict=True))
    restore_deltas = tuple(value - expected for value, expected in zip(restored, post, strict=True))
    maximum_change = max(abs(value) for value in deltas)
    target_sum_change = sum(post) - sum(pre)
    maximum_restore_error = max(abs(value) for value in restore_deltas)
    checks = {
        "nonzero_enabled_advantage": any(enabled and advantage != 0.0 for enabled in loss_mask),
        "optimizer_step_incremented": trained.step == 1,
        "parameters_moved": maximum_change > 1e-6,
        "positive_advantage_increased_target_logprob": target_sum_change > 1e-6,
        "restored_state_matches_live_post": maximum_restore_error <= 1e-5,
        "sampler_references_distinct": baseline_sampler.provider_reference
        != post_sampler.provider_reference,
        "state_references_distinct": baseline_state.provider_reference != post_state.provider_reference,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "rank": rank,
        "learning_rate": learning_rate,
        "advantage": advantage,
        "started_from_fresh_session": True,
        "optimizer_calls": 1,
        "sample": {
            "text": sampled.text,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completion),
        },
        "datum": {
            "schema_keys": sorted(datum),
            "uses_singular_advantage": "advantage" in datum and "advantages" not in datum,
            "selected_tokens": sum(loss_mask),
            "nonzero_advantage": advantage,
        },
        "forward": {
            "pre_selected_logprobs": list(pre),
            "post_selected_logprobs": list(post),
            "restored_selected_logprobs": list(restored),
            "selected_logprob_deltas": list(deltas),
            "maximum_absolute_change": maximum_change,
            "target_sequence_sum_change": target_sum_change,
            "maximum_restore_error": maximum_restore_error,
        },
        "training": {"step": trained.step, "metrics": dict(trained.metrics)},
        "checkpoints": {
            "baseline_sampler": _checkpoint_payload(baseline_sampler),
            "baseline_state": _checkpoint_payload(baseline_state),
            "post_sampler": _checkpoint_payload(post_sampler),
            "post_state": _checkpoint_payload(post_state),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "finished_at_unix": time.time(),
    }
    if not payload["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise CanaryFailure(failed, payload)
    return payload


def _checkpoint_payload(checkpoint: Any) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "provider_reference": checkpoint.provider_reference,
        "digest": checkpoint.digest,
        "step": checkpoint.step,
        "kind": checkpoint.kind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--advantage", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    env_file = Path(os.environ.get("SYNTH_TINKER_ENV_FILE", DEFAULT_ENV_FILE))
    _load_provider_environment(env_file)
    provider = TinkerAdapter(TinkerCredentials.from_env())
    output = Path(args.output)
    try:
        payload = run_canary(
            provider,
            run_id=args.run_id,
            model_id=args.model,
            rank=args.rank,
            learning_rate=args.learning_rate,
            advantage=args.advantage,
            prompt=args.prompt,
        )
    except CanaryFailure as error:
        _atomic_json(output, error.evidence)
        raise
    _atomic_json(output, payload)
    print(json.dumps({"passed": True, "output": str(output), "checks": payload["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
