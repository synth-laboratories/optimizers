# prompt-opt

Apache-2.0 licensed Python-only prompt optimization package that provides a local offline mirror of the public Synth GEPA/MIPRO SDK surfaces under `prompt_opt.sdk.optimization.*`.

## What is included

- `prompt_opt.sdk.optimization.policy.v1.PolicyOptimizationOfflineJob`
  - Local drop-in offline job surface for `gepa_offline` and `mipro_offline`.
- `prompt_opt.sdk.optimization.internal.prompt_learning.PromptLearningJob`
  - High-level local prompt-learning wrapper with candidate/state accessors.
- `prompt_opt.sdk.optimization.internal.configs.prompt_learning.PromptLearningConfig`
  - Canonical prompt-learning config models with local `proxied -> retrieved` normalization.
- `prompt_opt.dspy.MIPROv2`
  - DSPy-compatible local MIPRO wrapper routed through the mirrored offline SDK.
- `prompt_opt.dspy.gepa`
  - GEPA slot-in wrapper routed through the same local offline runtime.
- `prompt_opt.adapters.synth_container`
  - Container request/response helpers for local rollout integration.
- `src/gepa/__init__.py`
  - Import compatibility shim so `import gepa` works against this package.

## Install (editable, local)

```bash
cd prompt-opt
pip install -e .
```

## Quick usage

```python
from prompt_opt.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob

def task_model(prompt: str) -> str:
    if "Return exactly one of" in prompt:
        return "paris"
    return "rome"

job = PolicyOptimizationOfflineJob.create(
    kind="mipro_offline",
    system_name="cities-local",
    backend_url="local://prompt-opt",
    api_key="local",
    config={
        "prompt_learning": {
            "algorithm": "mipro",
            "execution_mode": "retrieved",
            "task_data": {
                "train_examples": [{"input": "Capital of France?", "answer": "paris"}],
                "validation_examples": [{"input": "Capital of France?", "answer": "paris"}],
            },
            "mipro": {
                "initial_candidate": {
                    "stages": [
                        {
                            "id": "main",
                            "name": "main",
                            "messages": [
                                {"role": "system", "pattern": "Answer the question.", "order": 0},
                                {"role": "user", "pattern": "{input}", "order": 1},
                            ],
                        }
                    ]
                },
                "num_candidates": 4,
                "max_iterations": 3,
            },
            "local_runtime": {"task_model": task_model},
        }
    },
)

result = job.stream_until_complete(timeout=30.0, interval=0.05)
best_candidate = job.get_state_envelope()["state"]["candidates"][result["best_candidate_id"]]
print(best_candidate["candidate_content"])
```

## GEPA shim

```python
from gepa import optimize
from prompt_opt.adapters.synth_offline import LocalEvaluator, SynthOfflineLearningAdapter

def score_fn(example, candidate):
    expected = str(example.get("answer", "")).strip().lower()
    prompt = " ".join(candidate.values()).lower()
    return 1.0 if expected and expected in prompt else 0.0

adapter = SynthOfflineLearningAdapter(LocalEvaluator(score_fn=score_fn))
result = optimize(
    seed_candidate={"system_prompt": "Answer briefly."},
    trainset=[{"input": "Capital of France?", "answer": "paris"}],
    adapter=adapter,
    max_metric_calls=8,
)

print(result.best_candidate)
print(result.val_aggregate_scores[result.best_idx])
```

## Notes

- This package is local-only and does not call Synth backend APIs.
- Runtime execution is retrieval-based only. Hosted configs that specify `execution_mode="proxied"` are accepted and normalized to `retrieved` locally.
- The local runtime mirrors candidate/state/result payloads and offline job methods; `backend_url` and `api_key` are kept for signature parity but are inert in local mode.
- Multi-stage candidates are first-class in both local GEPA and local MIPRO.

## Examples

- Local DSPy MIPRO:
  - `examples/mipro_local_example.py`
- DSPy GEPA slot-in:
  - `examples/dspy_gepa_slot_example.py`
- Local offline Banking77 via Synth `InProcessContainer`:
  - `examples/banking77_container_example.py`

Run the Banking77 local regression harness with:

```bash
PYTHONPATH=/Users/joshpurtell/Documents/Github/prompt-opt/src:/Users/joshpurtell/Documents/Github/synth-ai \
python3 /Users/joshpurtell/Documents/Github/prompt-opt/examples/banking77_container_example.py \
  --algorithms gepa mipro \
  --train-per-label 4 \
  --held-out-per-label 2 \
  --num-generations 2 \
  --children-per-generation 8 \
  --num-candidates 8 \
  --max-iterations 6
```

The script prints per-algorithm JSON with `train_*` and `held_out_*` metrics and raises if held-out improvement does not exceed `--min-held-out-delta`.
