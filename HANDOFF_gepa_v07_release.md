# Handoff: GEPA v0.7 ascope — finish e2e, merge, release notes

**Date:** 2026-08-20
**Repo:** `optimizers`
**Branch:** `backup/last72h-optimizer-platform-20260813` (ahead of origin by 1: `b40d598 Checkpoint working tree before disk cleanup`)
**Tree:** **uncommitted.** Do not commit unless asked. Do not force-push.

This supersedes the “open problem / still to prove” sections of
[`HANDOFF_gepa_eval_20260820.md`](HANDOFF_gepa_eval_20260820.md). Read that for
engine-bug history. Catalog / how-to-add-a-family still lives in
[`HANDOFF_gepa_eval_catalog.md`](HANDOFF_gepa_eval_catalog.md).

**v0.7 is the ascope / eval contract, not the PyPI version.** Workspace
`Cargo.toml` is still `0.2.5`; `changelog.log` last shipped `0.2.6.dev20260626`.
Decide the package version in release notes before tagging.

---

## TL;DR — what is done vs what you own

Live ascope is **proven** on `temp/gepa_proposer` via `POST /rollout`. The
remaining job is packaging: tests you can re-run, a reviewable commit, optional
`evals/` promotion, and release notes.

| You own | Status |
|---|---|
| Engine + eval ascope implementation | In the working tree (uncommitted) |
| Live N≥3 named-term pass | Done — `eval_uplift` luna-low vs luna-med |
| Code lever (`domain_policy`) | Done — `tau2:1` stamp `parallel_20260820165810` |
| Jesterky → proposer + extras in the scalar | Done — `train:1` stamp `parallel_20260820171128` |
| flash_evolve live | Done — `parallel_20260820164428` (lane overlap was 0s) |
| Contract pytest | Added; **re-run the full file before merge** |
| `cargo test -p synth_gepa --lib` | **Not re-run after the last Rust edits** — you must |
| Wheel rebuild + install | Required before any live re-run that needs engine changes |
| Promote `temp/gepa_proposer/` → `evals/containers/images/gepa-proposer/` | **Not done.** SCOPE says wait until the contract stops moving |
| Commit / PR / tag / changelog | **Not done** |
| Finalize release notes | Draft at the bottom of this file |

**Not this product:** [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md)
(GameBench / inner `policy_script` levers). Do not fold it in.

---

## Do not

- Commit unless explicitly asked.
- Kill leftovers on **8877** (old Banking77) or **8114** (HealthBench).
- Put the proposer eval on 8877.
- Invoke `uv run synth-optimizers`. Use `.venv/bin/synth-optimizers`.
- Install a wheel without `--no-cache`.
- Treat `cost_usd == 0` + tokens as free. Combiner: **`cost_usd > 0` is billed; `cost_usd == 0` with tokens is unpriced.**
- Re-rank arms on the **scalar** after `--ascope`. Extras now default to weight 1.0 when enabled, so time/confidence/rubrics swamp exploration/exploitation/uplift. Compare **named terms** (`train_exploration`, `train_exploitation`, `eval_uplift`).
- Share one inner `container_url` across parallel arms (exclusive lock).
- Print keys from `/Users/joshuapurtell/Documents/GitHub/synth-ai/.env`.

---

## Layout

| Path | Role |
|---|---|
| `rust/crates/synth_gepa/` | GEPA engine (operator workspace, jesterky, parked lanes, pause/fork, pricing) |
| `rust/crates/synth_optimizer_platform/` | Codex elicitation auto-accept, codex_home MCP append |
| `temp/gepa_proposer/` | Outer eval/RL `http_task`. **This is the product.** |
| `temp/gepa_proposer/run_luna_parallel.py` | Live driver (`--ascope`, `--serial-arms`, `--pipeline-mode`, `--task-id`) |
| `temp/gepa_proposer/gepa_proposer/scoring.py` | Three-term reward + optional extras |
| `temp/gepa_proposer/gepa_proposer/ascope_harvest.py` | Post-episode ascope receipt (`mutated_lever_ids`, jesterky, MQ, MCP) |
| `temp/gepa_proposer/gepa_proposer/mq_stub.py` | Local manderqueue for `--ascope` |
| `temp/gepa_proposer/tau2_container/` | Inner tau2 (code lever = `domain_policy`) |
| `jesterky/` (sibling repo) | Annotator. Binary: `jesterky/target/release/jesterky` |
| Credentials | `/Users/joshuapurtell/Documents/GitHub/synth-ai/.env` |

Engine: `maturin build --release --out target/wheels` from `optimizers/`, then:

```bash
uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache target/wheels/synth_optimizers-*.whl
```

Driver rebuilds the wheel unless `--skip-build`.

---

## Uncommitted surface (merge this)

**Modified**

- `rust/crates/synth_gepa/src/{codex_app_server,lane_executor,lib,operator_workspace,runtime,service}.rs`
- `rust/crates/synth_optimizer_platform/src/agent_runtime/{app_server,codex_home}.rs`
- `temp/gepa_proposer/gepa_proposer/{app,episode,scoring,store}.py`
- `temp/gepa_proposer/run_luna_parallel.py`
- `temp/gepa_proposer/tests/test_contract.py`

**Untracked (must add)**

- `temp/gepa_proposer/gepa_proposer/ascope_harvest.py`
- `temp/gepa_proposer/gepa_proposer/mq_stub.py`

Do **not** commit `temp/gepa_proposer/generated/` (live stamps, traces, Codex homes).

Suggested commit split if you want reviewable PRs:

1. Engine: operator workspace, parked lane runtime, stall timeout, adaptive-concurrency flag, jesterky, pause/fork, leak redaction, MCP elicitation.
2. Eval container: scoring extras, harvest, MQ stub, driver `--ascope`, contract tests.
3. Docs: this handoff + changelog + README catalog note.

---

## Live stamps (authoritative)

All under `temp/gepa_proposer/generated/`. Pass = `status=completed`, `skip_heldout=false`, heldout evaluated, unique `run_id`.

### Named-term ranking (the eval pass)

**`parallel_20260820145733`** + **`parallel_20260820154450`** (luna-med N=3 completed in the later stamp).
`train:1`, 6 proposals, `skip_heldout=false`.

| term | luna-low n=3 | luna-med n=3 | separates beyond each sd |
|---|---|---|---|
| train_exploration | +0.023 ± 0.021 | +0.030 ± 0.008 | no |
| train_exploitation | +0.054 ± 0.038 | +0.017 ± 0.036 | no |
| **eval_uplift** | **−0.018 ± 0.014** | **−0.002 ± 0.006** | **yes** |
| episode_cost_usd | $0.038 ± $0.015 | $0.059 ± $0.008 | yes |

Named pass term: **`eval_uplift`**. A scalar alone is not a pass. Failed-run usage is fixture-frozen totals, not spend.

Those episodes scored extras at **weight 0** (flags on, weights off). Later scoring defaults extras to weight 1.0 when `include_*` is true — **do not mix scalars across that change**.

### flash_evolve

**`parallel_20260820164428`** — luna-low, `train:1`, 1 rep, 8 rollout workers, adaptive concurrency **off**.
`mode=flash_evolve`, proposer + heldout, `gepa.run.finished`, reward 0.037, episode cost ~$0.048, `operator_control.ok=true`.
`overlap_seconds` was **0.0** / `max_concurrent_lane_jobs=1` — mode ran; lanes did not overlap wall-clock.

### Code lever (tau2 `domain_policy`)

**`parallel_20260820165810`** — `tau2:1`, luna-low, 1 rep, inner 8774.
6 episode candidates, all `mutated_lever_ids=["domain_policy"]`. Heldout evaluated. `eval_uplift` +0.0625. MQ/scratchpad/hypotheses/MCP/pause-fork ok.
Jesterky **fail-open** (`annotated=0`, 7ms): `jesterky/target/` had been deleted. Not a product bug. Binary was rebuilt and used on the next stamp. JSON: `generated/parallel_20260820165810/luna_low_vs_med.json`.

### Jesterky + reward extras in the scalar

**`parallel_20260820171128`** — `train:1`, luna-low, `--ascope`.
Jesterky command = absolute rebuilt binary, `annotated=6`, `theme_count=13`, `elapsed_ms≈89459`, `fail_open=false`. Proposer cited jesterky themes in `proposal/manifest.json`.
`optional_terms`: confidence **0.333**, rubrics **0.542**, time −0.125, cost −0.002, milestones 0 (generation 0). Scalar ~0.777 because extras now enter the reward. Pause/fork ok. Harvest `mutated_lever_ids=["stage2_system"]` (prompt, as expected).

---

## Ascope checklist (already live)

| Item | Proof |
|---|---|
| Harness / prompt / code | Operator lever filter (`operator_workspace.rs` `lever_class`). Prompt = Banking77 `stage2_system`. Code = tau2 `domain_policy`. Fixtures have no `harness_module` field; harness is enabled, not mutated. |
| Manderqueue | Local stub + `manderqueue_messages≥1`, `guidance_has_messages=true` |
| Workspace / fs / scratchpad / hypotheses | Harvest `scratchpad`, `hypotheses_open` |
| Pause / branch | Driver `prove_operator_control` fork_from + pause |
| Candidate hypotheses | `state/hypotheses.json` |
| flash_evolve / combee / pipeline | `--pipeline-mode flash_evolve` stamp above |
| Jesterky visible to proposers | Annotate artifacts + cited themes |
| Reward terms | missing=`zero` live; `fail` contract-tested. confidence, time, cost, milestones, rubrics |
| Optional MCP | `mcp_in_codex_config=true` (filesystem server) |
| Cost | `usage_ledger_delta`, `unpriced_rows=0` |

---

## Engine traps (do not rediscover)

1. **Parked lane executor.** `advance_gepa_config_once` opens a fresh `GepaRunContext` every tick. Without `ParkedLaneRuntime`, jobs complete in sqlite with **no `runtime_outcome`**. Symptom: `completed GEPA runtime job has no runtime_outcome` or livelock of `gepa.run.started`.
2. **Adaptive rollout concurrency.** Default climb (30→62 workers) tripped `rollout infra failure rate 0.28 > 0.25` on heldout. Live flash_evolve used **8 workers + `adaptive_rollout_concurrency: false`**.
3. **Resume SM.** `ensure_resumed_state_machine_allows_progress` must fast-forward `Created → Initializing → Ready` or finalize dies `created -> completed`. Do not wipe non-empty `state_history` with an empty rebuild.
4. **Jesterky `parse_episode`.** Must forward `jesterky` **and** `jesterky_workflow`. Contract: `test_jesterky_enabled_without_bulk_survives_parse_episode`. Bulk **off** (cap 6). Model `gpt-5.6-luna`.
5. **`GepaServiceRunRequest` `deny_unknown_fields`.** Operator lives under `advanced.operator`.
6. **`[dataset]` vs `[taskset]`.** Wrong table = silent empty taskset.
7. **OpenRouter / luna stall.** `advanced.proposer_io.message_stall_timeout_seconds` ≥ 300 (driver `GEPA_PROPOSER_STALL_SECONDS=300`).
8. **Codex elicitation.** `app_server.rs` auto-accepts `mcpServer/elicitation/request` → `{action: accept}`. Without this, ChatGPT/Codex arms hang.
9. **Inner Banking77 `cost_usd: 0.0`.** Combiner must not treat 0+tokens as billed-free. Catalog in `gepa_proposer/pricing.py` + `synth_gepa/src/usage_pricing.rs`.
10. **Harvest `candidate_registry.json`.** Lives under `gepa-runs/episode-<id>/gepa_<id>/`, not always the episode root. `ascope_harvest.py` globs `**/candidate_registry.json`.
11. **Tau2 inner.** Driver infers `inner_family` from `task_id`. `ensure_tau2` uses `tau2_container/.venv/bin/python synth_service_app.py`. Tau2 dialogues are slow (20 train / 8 heldout on `tau2:1`); budget ≥ 1800s.

---

## Tests to run before merge

### Contract (eval container)

```bash
cd /Users/joshuapurtell/Documents/GitHub/optimizers/temp/gepa_proposer
.venv/bin/python -m pytest tests/test_contract.py -q
```

If pytest is missing: `uv pip install --python .venv/bin/python pytest`.

Must include (added this week):

- `test_jesterky_enabled_without_bulk_survives_parse_episode`
- `test_ascope_mcp_and_code_lever_reach_the_gepa_wire`
- `test_ascope_harvest_reads_operator_workspace` (now asserts `mutated_lever_ids`)
- `test_optional_reward_terms_use_cursor_evidence`
- `test_jesterky_annotations_fill_confidence_and_rubrics`
- `test_missing_fail_rejects_absent_confidence_and_accepts_jesterky`
- `test_zero_cost_usd_with_tokens_is_priced_not_free`

Green pytest is **not** a live eval.

### Engine

```bash
cd /Users/joshuapurtell/Documents/GitHub/optimizers
cargo test -p synth_gepa --lib
cargo test -p synth_optimizer_platform --lib
```

### Live e2e (the actual eval)

Credentials: `source` is enough via the driver (`load_env_file` on `synth-ai/.env`).
Jesterky binary must exist: `ls -l ../jesterky/target/release/jesterky` (rebuild with `cargo build --release -p jesterky-cli` in `jesterky/` if missing).

**A. Ranking replay (pass bar)** — same `task_id`, isolated `run_id`s, two inners, N≥3/arm, `skip_heldout=false`, 6 proposals. Report mean ± spread of the three terms. Pass = arms separate beyond that spread **and** you can name the term.

```bash
cd temp/gepa_proposer
# pick free ports; never 8877 / 8114
GEPA_PROPOSER_ROLLOUT_WORKERS=8 GEPA_PROPOSER_STALL_SECONDS=300 \
  .venv/bin/python run_luna_parallel.py --ascope --serial-arms \
  --task-id train:1 --replicates 3 --inner-ports 8765,8766 \
  --service-port 8095 --proposer-port 8884 --max-wall-seconds 1800
```

Omit `--skip-build` unless the wheel already matches this tree.

**B. Code lever smoke (1 rep)**

```bash
GEPA_PROPOSER_ROLLOUT_WORKERS=4 .venv/bin/python run_luna_parallel.py --ascope --serial-arms \
  --task-id tau2:1 --replicates 1 --inner-ports 8775 \
  --service-port 8096 --proposer-port 8885 --max-wall-seconds 1800 \
  --arms '[{"label":"luna-low-tau2","provider":"openai","model":"gpt-5.6-luna","reasoning_effort":"low"}]'
```

Pass: `status=completed`, harvest `mutated_lever_ids` includes `domain_policy` (not just `levers.code=true`), heldout evaluated.

**C. Jesterky extras smoke (1 rep)** — confirm `jesterky_annotated≥1`, `jesterky_fail_open=false`, `optional_terms.confidence` and `.rubrics` ≠ 0.

**D. flash_evolve smoke (1 rep)**

```bash
GEPA_PROPOSER_ROLLOUT_WORKERS=8 .venv/bin/python run_luna_parallel.py --ascope --serial-arms \
  --pipeline-mode flash_evolve --task-id train:1 --replicates 1 \
  --inner-ports 8767 --service-port 8097 --proposer-port 8886 --max-wall-seconds 1800 \
  --arms '[{"label":"luna-low","provider":"openai","model":"gpt-5.6-luna","reasoning_effort":"low"}]'
```

Pass: `gepa.run.finished`, heldout evaluated, `operator_control.ok=true`. Do not require wall-clock lane overlap.

Expect ~$0.02–$0.06 per Banking77 episode, more wall on tau2. `SYNTH_OPTIMIZERS_VL_PROJECT=0` is logs-only.

---

## Promote to `evals/` (after contract freeze)

SCOPE: v0 lives in `temp/gepa_proposer/` until the contract stops moving, then copy to
`evals/containers/images/gepa-proposer/` (`contract = "http_task"`, `target_id = "gepa_proposer"`).
That directory **does not exist yet** in this repo. Do not copy `generated/`.
Do not copy GameBench / factorybench Banking77 suites.

Freeze checklist before copy:

- [ ] Full `tests/test_contract.py` green
- [ ] `cargo test -p synth_gepa --lib` green
- [ ] At least one fresh live `POST /rollout` on this tree
- [ ] README catalog + reward extras documented
- [ ] No more deny_unknown_fields surprises on `advanced.operator` / `jesterky_workflow`

---

## Merge / release procedure

1. Run contract + cargo tests above.
2. Ask before commit. Suggested message:

   ```
   Ship GEPA v0.7 operator ascope and live proposer eval.

   Park lane runtimes across service ticks, expose operator workspace
   (MQ, hypotheses, pause/fork, jesterky, MCP), and score POST /rollout
   from a real GEPA episode with named reward terms.
   ```

3. Push the existing backup branch; open PR against the intended release base (confirm with owners — this branch is a 72h backup, not `main`).
4. Bump package version if this is a real PyPI cut (today’s tree is still `0.2.5` / `0.2.6.dev*`).
5. Append `changelog.log` (style: dated bullets, “Prepared `x.y.z` …”).
6. Tag only after wheel install proof: `.venv/bin/synth-optimizers --version` matches the changelog.

---

## Draft release notes (finalize, do not publish as-is)

### GEPA v0.7 — operator ascope + proposer eval

GEPA can now optimize **prompt, code, and harness** surfaces in one run, with an
operator workspace the proposer actually reads, and an eval that is a real
optimizer episode: `POST /rollout` on the `gepa-proposer` container.

**Eval**

- Outer image: `temp/gepa_proposer` (`http_task`). Inner task containers unchanged.
- Reward terms (always): train exploration (mean reduction), train exploitation, eval uplift. Pass is **named-term separation**, not a scalar.
- Optional extras, calibrated: confidence, time, cost, milestones, rubrics; `missing=zero` or `fail`.
- Episode spend is usage-ledger set-difference vs the fixture. `cost_usd=0` with tokens is unpriced, not free.
- Catalog: 23 fixtures (Banking77, HealthBench, Crafter, tau2, MiniGrid, OfficeQA).

**Operator ascope**

- Scratchpad, candidate hypotheses, manderqueue inbox/guidance (heldout leak terms redacted).
- Pause / resume / fork (`fork_from` + `/pause`).
- Jesterky annotate-before-propose (cap 6 traces; bulk off). Themes are required reading when enabled.
- Optional MCP external agent (workspace filesystem in the live eval).
- Pipeline modes: `sync_serial`, `flash_evolve` / combee.

**Known limits**

- flash_evolve is live; overlapping lane wall-clock was not observed in the first pass.
- No fixture currently mutates a distinct `harness_module` field.
- Ranking eval (`eval_uplift` luna-low vs luna-med) used extra weights 0; ascope-on scalars are not comparable to that table.
- Eval image is still under `temp/` until the HTTP contract is frozen.

**Ops**

- Rebuild: `maturin build --release --out target/wheels` then `uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache <whl>`.
- Proposer: `.venv/bin/synth-optimizers`, not `uv run`.
- Luna needs `~/.codex`. OpenRouter arms need a stall timeout ≥ 300s.

---

## People / env

| | |
|---|---|
| Engine venv | `optimizers/.venv` |
| Eval venv | `temp/gepa_proposer/.venv` |
| Tau2 venv | `temp/gepa_proposer/tau2_container/.venv` |
| Cookbooks inner Banking77 | `synth-cookbooks-public` (driver `uv run --project banking77_container`) |
| Disk | Drivers isolate `GEPA_PROPOSER_OUTPUT_DIR` under `generated/parallel_<stamp>/gepa-runs/` — do not dump onto shared `optimizers/runs/` |
