# Lane: local MLX RL — optimizers side

Branch `agent/mlx-local-rl-20260818`, cut from `origin/main` @ `1c89092`.
Authority doc (read it before touching anything):
`~/Documents/Codex/2026-08-18/synth-mlx-rl-dev/outputs/implementation-plan-final.md`

`origin/main` was chosen over `origin/dev` deliberately: dev is 11 commits behind with
zero divergence, and `1c89092` is the exact commit every file:line reference in the plan
was audited against.

## Baseline (established 2026-08-18, before any change)

```
uv sync --group dev
uv pip install pytest          # pytest is NOT in the dev group; `uv run pytest` silently
                               # resolves an ephemeral env that cannot import synth_containers
uv run python -m pytest tests/ -q
=> 51 passed
```

Use `uv run python -m pytest`, never `uv run pytest`.

## Cross-repo coupling to watch

`pyproject.toml` pins `synth-containers==0.4.1.dev20260814` via
`[tool.uv.sources] rev = "e76f8e4ba3edae10dec24bf9e71ec1a7fb332bed"`. Every containers
change in this campaign (provider admission, `/compatibility` on the platform app,
`TokenCaptureV5` extension) requires bumping that rev here. The containers work is a
separate lane; this lane must not vendor around the pin.

**The pin is materially behind main, verified 2026-08-18.** In the pinned rev,
`runtimes/banking77.py` is 346 lines and has **no responses path whatsoever** — no
`_sample_responses`, no `_validate_responses_endpoint`; only
`_validate_remote_checkpoint_endpoint` exists. On containers `main` (a2a316b) that same
file is ~570 lines and carries both. So:

  - The endpoint-validator precedent this lane copies is `_validate_responses_endpoint`
    (banking77.py:349 on main) **and** `_validate_remote_checkpoint_endpoint`
    (banking77.py:487 on main). Only the second is visible from the pinned rev. If you
    are reading site-packages and cannot find the first, that is the pin, not an error
    in the plan.
  - D9 (`responses` first-class) does not hold on the optimizers side at all until the
    pin is bumped past the responses lane. Bumping it is a prerequisite for the first
    live GSM8K rollout, not a cleanup task.

## Ordered work

Each item names the file, the change, and the test that proves it. Nothing here needs a
GPU, a Mac-only dependency, or a running MLX service — all of it is testable with
fixtures, which is why it goes first.

### O1 — `ModelRoute` accepts a local route  [GATED: plan open decision 6]

`src/synth_optimizers/eval/models.py:244`

```python
if not route.startswith("https://"):
    raise EvalContractError("model.route must be an https endpoint")
```

Replace with: `https://` unrestricted; `http://` permitted **only** for hostnames
`127.0.0.1`, `localhost`, `::1`, `host.docker.internal`; every other `http://` origin
refused with the same error class. Path must end in `/v1/chat/completions` or
`/v1/responses` (D9 — both families are first-class).

Mirror the shape of `_validate_responses_endpoint` in the containers Banking77 runtime:
parse with `urlsplit`, reject userinfo/query/fragment, compare a normalized
`scheme://host[:port]`.

Tests: a loopback http route accepted; a `host.docker.internal` route accepted; a
public http route refused; a route with credentials in the userinfo refused; an
https route unchanged.

**Gate:** this relaxes the schema whose entire purpose is stopping an agent from naming
its own endpoint. Confirm the exact shape with Josh before landing.

### O2 — zero-rate local route shape  [GATED: plan open decision 6]

Same file, `ModelRoute.from_mapping`. A local model has no provider key and no
per-token price, but `secret` is a required identifier and the three `usd_per_1m_*`
fields are required non-negative numbers.

No schema change needed: `0.0` is already a legal rate. What is needed is a recipe that
uses `price_source = "local-compute"`, a dated `price_as_of`, and a `secret` naming the
per-run bearer token (`SYNTH_MLX_RL_TOKEN`) rather than a provider key. Add a test that
a zero-rate route round-trips and that `recipe.budget.max_llm_calls` is still enforced,
since `max_usd` becomes vacuous on this lane.

### O3 — `mlx-lora.v1` policy kind

`src/synth_optimizers/eval/models.py` (`PolicyCandidate`, `TargetManifest.policy_kinds`),
`src/synth_optimizers/eval/staging.py`.

Candidate directory layout:

```
/input/policy/
  adapter_config.json      MLX-LM adapter layout
  adapters.safetensors     rank-8 adapter (few MB)
  policy.json              base model id, rank, chat-template digest, thinking mode
```

The base model enters the same `CandidateSet` as an adapter-free candidate and is the
declared `baseline_id`, so base-vs-LoRA is a paired difference over shared seeds rather
than two runs compared by hand.

Tests: staging an adapter dir produces a stable `artifact_digest`; two candidates with
identical bytes collide on digest; a candidate missing `adapter_config.json` is refused.

### O4 — host-side snapshot registration

`src/synth_optimizers/eval/runner.py` (trial input assembly, near the `trial.json` write
around line 369).

Before each trial: register `/input/policy/` with the MLX service, receive an immutable
`policy_snapshot_id` derived from the candidate's `artifact_digest`, and write it into
`trial.json`. The container receives the snapshot id and the recipe-owned route — never
adapter bytes, never a mutable run-local name.

Test with a fake registrar; no live service.

### O5 — a pinned local fixture recipe

`src/synth_optimizers/eval/catalog/eval.mlx.local-policy.smoke.v1.toml`, modeled on
`eval.craftax.llm-policy.smoke.v1.toml`. Must set `network = "bridge"` (a `none` network
cannot reach the host proxy) and `required_artifacts = ["trace"]`.

### O6 — local SFT backend

`src/synth_optimizers/sft.py`. `SftConfig.from_mapping` currently raises
`"backend must be fixture or tinker"`. Add the local MLX value plus an executor
implementing the `SftExecutor` protocol at `sft.py:44`.

Mirror the hosted contract in optimizers-beta `crates/synth_sft/src/config.rs:39`, whose
comments record two already-paid-for lessons: `checkpoint_steps` silently setting
training length, and `max_seq_len` silently deciding which rows train at all. Carry
`dataset_digest`, `training_steps`, `max_seq_len`, and `max_dropped_fraction`.

### O7 — local `grpo` / `cispo_minimax` algorithms

New modules. Preflight (`GET /compatibility?target=pipeline_rl_logprobs`, refuse on
`supported: false`, surface `missing_features` verbatim), snapshot registration via the
container's `POST /policy-configs`, rollout grouping, logprob collect, mismatch policy,
advantages, barrier control, refresh, next round.

`cispo_minimax` must **refuse** `eps_low < 1.0` under that name — SLIME only warns
(`slime/utils/arguments.py:1874`), which is how a two-sided run gets reported as CISPO.

Blocked on the containers token-trace lane. O1–O6 are not.

### O8 — RL metric block on eval manifests

Extend `eval.run-manifest.v1` / `eval.scorecard.v1` rather than inventing a report
schema. `SELECTION_STATUSES` already supplies the honest conclusion vocabulary
(`promoted` / `no_champion` / `inconclusive` / `invalid_evidence`).

## Not in this lane

PPO (no value head in v1). Anything requiring a running MLX service or Apple-Silicon
execution — that is the `synth-mlx-rl` lane, whose M0 is getting
`scripts/real_mlx_smoke.sh` to pass at all, since nothing in the seed prototype has ever
executed on Metal.
