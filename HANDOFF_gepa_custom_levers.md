# Pickup: GEPA custom levers live search (2026-08-19 evening)

Repo: `optimizers` on `backup/last72h-optimizer-platform-20260813`. **Uncommitted.
Do not commit unless asked.**

This is **not** the proposer-eval container work. That is
[`HANDOFF_gepa_eval_pickup.md`](HANDOFF_gepa_eval_pickup.md) /
[`HANDOFF_proposer_eval_container.md`](HANDOFF_proposer_eval_container.md).
This file is the Craftax **custom lever** path: register-then-run, code
`policy_script` + ReAct prompt overlay, same GEPA loop.

Read these first, in order, and do not rediscover them:

1. [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md) — canonical contract.
2. [`.cursor/rules/gepa-custom-levers.mdc`](.cursor/rules/gepa-custom-levers.mdc) — pinned don'ts.
3. [`temp/craftax_levers/README.md`](temp/craftax_levers/README.md) — toy stack.

---

## What you are picking up

Prove GEPA can **search** (not seed-eval) on two advertised levers against the
toy Craftax orchestrator:

| Mode | Target module | Seed | Ceiling on `train:0` |
|---|---|---|---|
| code | `policy_script` | noop `act(obs)` | 2.0 (greedy collect) |
| react | `react_system_prompt` | force `action: noop` | 2.0 (collect two wood) |

Lifecycle for both (engine already does this when the container advertises
`candidates_route`):

```
GET  /program
POST /candidates     # apply + load/reload once
POST /rollout        # {candidate_id, task_id}  — repeat per seed, no re-apply
GET  /rollouts/{id}
sensor / proposer / frontier
```

Do **not** invent a second optimizer. Do **not** swap in a chat-completions
"real proposer." The proposer is GEPA's default Codex app-server
(`backend = "codex_app_server"`, `gpt-5.4-mini`, `auth_mode = api_key`).
The earlier seed-only runs never called it because `max_total_rollouts=1`.

---

## Live state when this was written

Search **finished** (`run_gepa.py` exit 0, ~463s, 2026-08-19 22:00:59Z). Both
arms got train **0.0 → 2.0**. Codex app-server proposer, default `ProposerConfig`.

| Arm | Run | Seed | Best | Notes |
|---|---|---|---|---|
| code | `gepa_craftax_code_4ad41934` | `gepa_9feb8545f4d1` 0.0 | `gepa_b56fee92f3db` 2.0 | gen-1 child `gepa_6a18eac11353` minibatch 2.0, no extra train |
| react | `gepa_craftax_react_f8502145` | `gepa_9da173c6104a` 0.0 | `gepa_1a206ba1d017` 2.0 | gen-1 child `gepa_289a7a8c7179` minibatch 2.0, no extra train |

Artifacts: `runs/craftax_levers/<run_id>/` (`result_manifest.json`,
`candidate_registry.json`, `events.jsonl`). Stacks are stopped.

Older seed-only smokes (do not treat as search):

| Run | Result |
|---|---|
| `gepa_craftax_code_de4836bc` / `86659c20` | seed noop, train 0.0, no proposals |
| `gepa_craftax_react_7bce6ddd` | seed ReAct, train **2.0** (ceiling; old wander prompt) |

---

## What already landed (do not rebuild)

### Engine (`rust/crates/synth_gepa`, `synth_optimizer_platform`)

Uncommitted. Register-then-run:

- `GepaOptimizerContract.candidates_route` (optional)
- `ContainerClient.register_candidate` / `register_candidate_at`
- After `/program` load, stamp `program.metadata.gepa_candidates_route`
- `preflight_custom_levers`: known protocols `prompt_overlay.v1`, `whole_file.v1`,
  `unified_diff.v1`, `harness_restart.v1`, `identity`; refuse `flash_evolve` +
  `apply_isolation=serial_restart`
- `gepa_container_rollout_request`: sends `candidate_id`, full `lever_bundle`
  (not string-dropped), and a `register` block when the route is advertised
- Runtime: register **once per candidate_id** before batch fan-out; strip overlay
  from `/rollout`; apply-failed → completed rollout with reward 0 + ASI
- Sensor: `actionable_side_info` falls back to `side_info`
- `prompt_assertions` skipped when `policy.enabled=false`
- Python `GepaConfig`: `target_modules` allowed without optimizer policy proxy
  (`policy=None`)

### Toy containers (`temp/craftax_levers/`)

Split env HTTP + policy subprocess + orchestrator. Toy 3×3 wood/lava grid, **not
GameBench**. GEPA talks only to the orchestrator.

- `POST /candidates`, `GET /candidates/{id}`, `GET /rollouts/{id}`
- Serial isolation: `ensure_active` reloads stored source when switching
  `candidate_id`s (optimizer does not re-send)
- Metadata: `candidates_route=/candidates`, `capabilities.metadata.policy_ready=true`
- Tests: `test_code_register_then_rollout_on_demand`,
  `test_react_register_prompt_overlay_without_episode`

### `run_gepa.py`

Now a **search** driver, not seed-eval:

- Default `ProposerConfig` (Codex). Do not add `backend=chat_completions`.
- `max_generations=2`, `proposals_per_generation=1`, `max_total_rollouts=24`
- `policy=None`, `pipeline=sync_serial`, `rollout_transport=sync`
- `cache.mode=off`
- `heldout_split` must be `"heldout"` (toy has no `"test"`)
- `usage_registration.enabled=False`
- `SYNTH_OPTIMIZERS_VL_PROJECT=0`

`GepaEpisodeConfig(skip_heldout=True)` was **removed** from the driver. The
installed native wheel rejects `[gepa.episode]` (`unknown field episode`).
Source *does* have `episode` (`synth_optimizer_platform` `GepaConfig`). Wheel
and tree are out of date with each other. Heldout will run unless you rebuild
the wheel first.

### ReAct seed prompt

Changed in `temp/craftax_levers/craftax_levers/seeds.py`:

```
Always reply with exactly: action: noop. Do not collect wood. Do not move.
```

Old wander prompt already scored **2.0** on `train:0`, so ReAct could not show
numeric uplift. WOOD_PROMPT is still the known-good overlay. Tests use
WOOD_PROMPT / "Wander randomly." and do not depend on `SEED_PROMPT` text.

---

## How to run / resume

Keys: `OPENAI_API_KEY` (and optionally `OPENROUTER_API_KEY`) in
`~/Documents/GitHub/backend/.env.local`. `run_gepa.py` loads them. Never print.

If the live job is still running, wait. If it died:

```bash
cd /Users/joshuapurtell/Documents/GitHub/optimizers
SYNTH_OPTIMIZERS_VL_PROJECT=0 PYTHONPATH=temp/craftax_levers \
  uv run python temp/craftax_levers/run_gepa.py
```

Unit tests (no search):

```bash
cd temp/craftax_levers
PYTHONPATH=. uv run --directory ../.. pytest temp/craftax_levers/tests/test_levers.py -q
# or from that package:
cd temp/craftax_levers && PYTHONPATH=. python -m pytest tests/test_levers.py -q
```

ReAct tests need the same API keys (`@needs_openai`).

Native rebuild (only if you need `[gepa.episode]` or other uncommitted Rust):

```bash
uv run --with maturin maturin build --offline \
  --manifest-path rust/crates/synth_optimizers_py/Cargo.toml \
  --out target/wheels
uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache \
  target/wheels/synth_optimizers-*.whl
```

`maturin develop` fails on `synth-containers==0.4.0.20260730` not on PyPI.
`--no-cache` is mandatory. Confirm the `.so` still contains
`gepa_candidates_route` after rebuild.

---

## Done vs remaining

**Done**

- Engine register-then-run + `lever_bundle` values + custom-lever preflight.
- Toy orch for prompt + code (+ harness kinds advertised).
- Code search: Codex child `gepa_b56fee92f3db`, train **0.0 → 2.0**.
- ReAct search: Codex child `gepa_1a206ba1d017`, train **0.0 → 2.0** (noop seed
  prompt; WOOD_PROMPT-class overlay). Same register-then-run loop.

**Not done**

1. Confirm events: `POST /candidates` once per child, multiple `/rollout`s,
   ASI present, no per-seed `/load`.
2. Harness-restart GEPA search (`harness_module` + `harness_restart.v1`) was
   **not** run. Container advertises it; `run_gepa.py` does not.
3. GameBench `gepa_codepolicy` is **not in this repo**. Do not call the toy
   stack GameBench. Monty/Rust gold is
   [`SCOPE_gepa_craftax_monty_rust.md`](SCOPE_gepa_craftax_monty_rust.md).

---

## Constraints / traps

- Prompt overlay stays the default. Do not stuff code/harness into
  `react_system_prompt`.
- Kind ≠ protocol. `policy_script` + `whole_file.v1` / `unified_diff.v1`.
  `unified_diff.v1` is the code default; `whole_file.v1` is the whole-code
  special case (what the toy apply uses for a string payload).
- Optimizer never shells into the container.
- Unknown `protocol_id` fails closed. Refuse `flash_evolve` + `serial_restart`.
- GELO `forbid_code_policy_candidates` is a different algorithm.
- Toy `heldout_split` is `"heldout"`, not `"test"`.
- Installed wheel does not decode `[gepa.episode]`. Do not put it in TOML until
  you rebuild.
- Codex propose prints nothing. Quiet stdout ≠ hung. Check `events.jsonl`
  `to":"proposing"` and `codex app-server` processes.
- Do not skip ReAct for missing keys. Load `.env.local`.
- Do not commit `.env.local` or keys.

---

## Files that matter

| Path | Why |
|---|---|
| `temp/craftax_levers/run_gepa.py` | live search driver |
| `temp/craftax_levers/craftax_levers/orchestrator_app.py` | `/candidates` + `/rollout` |
| `temp/craftax_levers/craftax_levers/seeds.py` | `SEED_POLICY`, `SEED_PROMPT`, `GREEDY_POLICY`, `WOOD_PROMPT` |
| `temp/craftax_levers/craftax_levers/stack.py` | env + policy subprocess + orch |
| `src/synth_optimizers/gepa.py` | SDK (`policy=None`, `ProposerConfig` default Codex) |
| `rust/crates/synth_gepa/src/runtime.rs` | register once, then rollout |
| `rust/crates/synth_optimizer_platform/src/levers.rs` | `lever_bundle`, known protocol ids |
| `runs/craftax_levers/gepa_craftax_code_4ad41934/` | live code search artifacts |
