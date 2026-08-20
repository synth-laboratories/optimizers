# Pickup: GEPA proposer eval — engine unblocked, catalog at 23, reward metric is the open problem

Repo: `optimizers` on `backup/last72h-optimizer-platform-20260813`. **Uncommitted.
Do not commit unless asked.** Session of 2026-08-19 evening → 2026-08-20 early.

Supersedes the status sections of
[`HANDOFF_gepa_eval_catalog.md`](HANDOFF_gepa_eval_catalog.md); that doc's
"How to add a family" and catalog table are still current and were updated in
this session. Read it second.

---

## TL;DR

`train:1` now scores end to end — it never did before tonight. Three proposer
arms ran at N=3 and **none of them separate**. The blocker is no longer the
engine; it is that the reward scalar cannot resolve proposers on this task.

| arm | mean | min | max | spread | sd |
|---|---|---|---|---|---|
| luna-low | 2.6243 | 2.0024 | 3.0007 | 1.00 | 0.54 |
| luna-medium | 2.6204 | 1.9132 | 2.9865 | 1.07 | 0.61 |
| `nvidia/nemotron-3.5-lightning` | 2.7332 | 1.0899 | 4.0740 | **2.98** | **1.51** |

Do not read the nemotron mean as a win. Its spread is 3x the luna arms and its
best run (4.07) sits in the same distribution as a 1.09.

**Why they can't separate:** `train_exploration` is a sum of per-seed deltas over
8 binary-reward Banking77 minibatch examples, so it is integer-valued (observed
`{1,2,3,4}`) and each unit is worth ~15x the other two terms combined.
Per-replicate exploration: luna-low `[3,3,2]`, luna-medium `[3,2,3]`, nemotron
`[4,3,1]`. One seed flip dominates the scalar.

**Fix the metric before running more arms.** Normalise exploration by minibatch
size, or reweight the three terms so heldout movement is not rounding error.
`scoring.py:score_episode` is the only place this lives.

---

## The one real signal worth chasing

Decomposed, nemotron behaves differently from both luna arms, consistently
across all three replicates:

| arm | train exploitation (mean) | eval uplift (mean) |
|---|---|---|
| luna-low | −0.0415 | −0.0008 |
| luna-medium | −0.0579 | +0.0117 |
| nemotron | **+0.0849** | **−0.0183** |

Nemotron is the only arm that *gains* train score, and the only one that *loses*
heldout. That is the shape of minibatch over-fitting, and it is exactly what the
three-term reward was built to expose. **Hypothesis, not a finding** — the
per-term gaps (~0.1 and ~0.03) are still small against run-to-run noise. Worth a
targeted follow-up that holds exploration out and replicates only these two
terms.

---

## Engine fixes landed this session (all rebuilt + installed)

### 1. `FOREIGN KEY constraint failed` on every archive-derived fixture — FIXED

The headline bug. `train:1` and `train:2` had been failing at the very end of an
otherwise-successful run.

Root cause: `restore_state_machine_from_cursor`
(`rust/crates/synth_gepa/src/lib.rs`) copied a fixture's `cursor.state_history`
verbatim. In `banking77_first.json` all 66 transitions are stamped
`run_id: "b77_gepa_eval_mfg_20260819"` — the *archive* run. At finalize,
`persist_state_history` writes each transition under `transition.run_id`, into a
workspace whose `optimization_runs` only holds the new `gepa_<uuid>`. FK
violation.

Fixed at the import site (rebind to the current run), not in
`persist_state_history`, matching how imported candidates and checkpoints are
already rebound. Originating run id stays on the cursor and in the fixture.

Consequences worth internalising:
- It fired **after** `optimizer.state.transitioned → completed`. The
  optimization had genuinely succeeded; the terminal evidence persist then failed
  and flipped the run to `failed`, so the outer refused to score it. A "failed"
  GEPA run may contain a complete, correct result.
- It was **fixture-shaped, not effort-shaped**. Whichever arm got far enough hit
  it, which is why it looked non-deterministic across arms.
- `train:0` and `healthbench:3` never hit it because they have no inherited
  state history. That is the whole reason `healthbench:3` was the only fixture
  that had ever scored.

### 2. `OptimizerError::SqliteAt` — diagnostic that found the above

Bare `sqlite error: FOREIGN KEY constraint failed` named no statement. Added a
labelled variant (same `error_code`, so nothing downstream changed) and wrapped
every FK-capable write in `persist_candidate_registry` plus each terminal
`record_*` in `finalize_completed_gepa_run`. Next occurrence reads
`sqlite error at finalize.persist_state_history: ...`. Keep this.

### 3. Service wire could not express an OpenRouter proposer

`ServiceProposerSpec` is `deny_unknown_fields` and was narrower than
`ProposerConfig`. Added `allow_unverified_model`, `model_context_window`,
`model_auto_compact_token_limit` to the wire + OpenAPI.

Two fields that looked missing were not: `backend` already defaults to
`codex_app_server` (what OpenRouter requires), and `api_key_env` is carried by
`credentials.env_var`, which `apply_service_proposer_auth` maps across.

`allow_unverified_model` was added rather than putting nemotron in
`VERIFIED_OPENROUTER_MODELS` — that allowlist means "verified, with a known
static price", which nemotron is not.

### 4. Codex proposer auth: `env_key` → command-based — **this is what fixed nemotron**

`prepare_api_key_codex_home` (`agent_runtime/codex_home.rs`) wrote
`env_key = "OPENAI_API_KEY"`. Per
[OpenRouter's Codex CLI guide](https://openrouter.ai/docs/cookbook/coding-agents/codex-cli),
that mode works but **"Codex won't fetch the OpenRouter model catalog"**, so
non-OpenAI slugs get "Unknown model" fallback metadata. Now emits:

```toml
[model_providers.gepa_proposer]
name = "GEPA proposer"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"

[model_providers.gepa_proposer.auth]
command = "sh"
args = ["-c", "echo $OPENAI_API_KEY"]
```

The command reads `OPENAI_API_KEY` because the launch path already injects the
resolved secret under that name whatever env var supplied it; the key stays out
of `config.toml`. Verified in the shipped binary that the provider struct really
has an `auth` field (`ModelProviderAuthInfo`).

This, not tokens, was the fix. Four prior nemotron episodes died on
`codex app-server stalled: no JSON-RPC progress for 120s while waiting for
turn/completed`; more tokens, a 1800s timeout and a hand-set 1M context window
all failed. Command auth made the turn terminate on the next attempt.

`wire_api = "responses"` is correct and was left alone — OpenRouter genuinely
serves `/responses` (verified by direct curl: HTTP 200, proper envelope).

### 5. Codex has no per-call output cap

Grepped the shipped `codex-cli 0.145.0` binary: its only `max_output_tokens` is
the exec-tool pragma. The model-side knobs are `model_context_window`,
`model_auto_compact_token_limit`, `model_reasoning_effort`,
`model_reasoning_summary`, `model_verbosity`. Both new plumbed fields default to
`None`, so existing runs are unchanged. With command auth working, Codex should
fetch real metadata and you should **not** need to set them.

### 6. `proposal/validate.py` — checkable proposal contract

Nemotron wrote a schema-shaped-but-wrong manifest: invented
`parent_evidence_summary` / `parent_prompt` / `top_failure_clusters`, omitted the
required `candidate_comparison` and `example_ids_used`, and used `candidates`
where the contract says `proposals`.

Now every proposer workspace ships `proposal/validate.py`, generated in Rust from
the same `REQUIRED_EVIDENCE_FIELDS` / `PRESENCE_ONLY_EVIDENCE_FIELDS` constants
`validate_manifest_contract` enforces, so the two cannot drift. The README tells
the agent to run it until exit 0.

**Caveat, verified from the transcript: nemotron never ran it.** `validate.py`
appears only in `ls` output; there is no `python3 proposal/validate.py` in the
command log and no `OK: manifest accepted` in the artifacts. The manifest came
out correct anyway — so what fixed it was the **README naming the exact contract
and its consequence**, not the verify loop. The loop is still advisory.

**Recommended next build:** close it. On `validate_manifest_contract` failure,
send a follow-up turn on the still-open thread with the specific errors instead
of failing the run. Today's repair machinery (`join_adjacent_json_strings`,
`last_json_value_from_stream`, `normalize_manifest_contract`) is JSON-syntax
salvage plus a `schema_version` backfill only — there is no schema re-prompt.
This would have salvaged a run that had 6 good proposals on disk under wrong key
names.

---

## Responses API fixes in `banking77_container` (synth-cookbooks-public)

Nothing in the eval currently uses `api_family: "responses"` — everything is
`chat_completions`. That was lucky, because the Responses branch silently dropped
the knobs this eval depends on. All fixed and verified live:

- **`max_output_tokens` now sent.** Responses does not accept `max_tokens`;
  omitting it means uncapped, not "reuse the chat value".
- **`disable_reasoning` now applied.** Chat built `extra_body`; Responses ignored
  it. Reasoning tokens bill against the same `max_output_tokens` allowance, so a
  16-token cap was being consumed entirely by a reasoning item.
- **Retry parity** with the chat branch.
- **`status != "completed"` is now a 502.** Previously an exhausted response had
  empty `output_text`, fell through to `_normalize_policy_label("")`, and scored
  as a **wrong prediction**. Infra failure counted as unsolved.

Proof (live, gpt-4.1-nano + nemotron/OpenRouter):

| test | result |
|---|---|
| responses, real row, cap 16 | reward 1.0, `request_refund`, 4 completion tokens |
| chat, same row, cap 16 | reward 1.0, `request_refund`, 3 completion tokens |
| nemotron, responses, `disable_reasoning: auto` | real label, 7 tokens |
| same, `disable_reasoning: off` | **502** `status 'incomplete' (reason='max_output_tokens')` |

Engine side needs nothing: `api_family_suffix` rewrites the policy
`inference_url` to `/responses`, the container strips it back off
(`_strip_openai_endpoint_suffix`) and branches on `api_family`. No Rust code
builds a Responses body, and none needs to.

---

## New assets

### PRBench container — `temp/gepa_proposer/prbench_container/` (port 8130)

**PRBench is fully public and ungated** (`ScaleAI/PRBench` on HuggingFace,
cc-by-4.0, arXiv:2511.11562). Not scaffolding — real data. Vendored the two Hard
subsets: **550 tasks / 9,543 expert criteria** (independently re-counted).

Scoring is the paper's Appendix D.1 formula: `s = Σ(wᵢ·Iᵢ) / Σ(wᵢ where wᵢ>0)`,
clipped to [0,1], signed numerator so detrimental criteria penalise.

**Load-bearing schema gotcha, independently verified:** weight fields are nested
under `criterion.annotations`, and **2,503 of 9,543 criteria carry more than one
non-null weight field** (annotation-history residue — e.g. both
`critically_detrimental_weight: -10` and a stale `detrimental_weight: -7`).
Summing or first-non-null gives wrong weights. Only `weight_class` disambiguates,
and it is a human-readable label (`"critically important"`) that must be
normalised to the field name (`critically_important_weight`). Resolves non-null
for all 9,543.

Seed baseline, 8 train rows, gpt-4.1-nano policy / gpt-4.1-mini judge: mean
**0.157**, range 0.000–0.312, **0 infra failures**. A legitimate 0.0 still
returned `succeeded`. Prompt-sensitive: same row, better candidate,
0.0443 → 0.0949. `payload.task` handled, with a negative control returning 422
rather than silently resolving to `train:0`.

**Caveats:**
- Judge is **not calibrated** against the official grader — the paper does not
  publish its judge prompt. Rewards are internally consistent and prompt-sensitive
  (all GEPA needs) but **not comparable to the published PRBench leaderboard**.
  Recorded in `/metadata.benchmark.scorer_caveat`.
- **Not a catalog row.** No cursor fixture, no `fixtures.py` entry, no env pool.
  Remaining work is `HANDOFF_gepa_eval_catalog.md` §"How to add a family" parts
  2 and 3.
- Never driven end-to-end through GEPA; curl-verified only.
- Only the Hard subsets are vendored; the easier `finance`/`legal` (1,100 total)
  are equally public — set `PRBENCH_SUBSETS` and drop the parquet in.

### Catalog: 14 → 23 fixtures, every `usage: {}` eliminated

9 new, all real generation-start archive cursors with real frozen usage:
`train:3` (gen 3 / 19 cands, deepest in the catalog), `train:4` / `train:5`
(second Banking77 lineage, seed 0.54 vs 0.68 — real headroom instead of a
near-ceiling parent), `crafter:3/4`, `tau2:1/2`, `minigrid:1/2`.

Existing fixtures had their usage backfilled from source checkpoints. The three
seed-only fixtures (`tau2:0`, `minigrid:0`, `officeqa:0`) now carry explicit
zeros, which is the true total. Every fixture records `usage_source`.

New minter: `temp/gepa_proposer/mint_checkpoint_fixtures.py` — offline,
idempotent, no spend. It exists because `export_checkpoints.py` requires
`metadata.retain`, a flag only the proposer-eval fork path writes, so it silently
skips every plain cookbook archive. The real bar is `generation_start` and not a
compacted `checkpoint_summary.v1`.

Contract tests **18 → 37 passed**.

**None of the 9 new fixtures has been run against a live GEPA optimizer.** Minted,
wired, contract-tested only. `train:3` is the most demanding case — its
`state_history` carries *two* run ids (its own fork plus
`b77_gepa_eval_mfg_20260819`), so it is the best test of the FK rebind fix.

---

## Gotchas that cost real time tonight — do not re-learn these

1. **A leftover container is not interchangeable with a fresh one.** The luna-low
   arm's `infra failure rate 0.28 > 0.25` was entirely caused by a Banking77
   process squatting on **8877** since 12:30, started without
   `BANKING77_POLICY_TIMEOUT_SECONDS` and therefore running the 20s default
   instead of 60s: 54 provider 504s. The medium arm on a fresh 8765 had **zero**
   timeouts. Same fixture, same shape, same engine.
   `run_luna_parallel.py` now **refuses** to reuse an inner container it did not
   start (`--allow-reused-inner` to override) and defaults to `8765,8766`.

2. **Usage on a failed run is the fixture's frozen total, not the run's spend.**
   Three failed `train:1` runs all reported
   `{prompt 482019, completion 2853, rollouts 640, proposer 1, total 484872}` —
   byte-identical to `banking77_first.json`'s
   `checkpoint.snapshot.usage`. A completed `train:1` reports
   `total 1199499 / rollouts 1536 / proposer 2`, i.e. inherited baseline plus
   actual work. **Usage is cumulative from the imported checkpoint.** Any
   per-arm cost comparison must subtract the fixture baseline, and a failed run's
   usage tells you nothing about what it spent.

3. **`_policy_arm` silently dropped unknown fields.** The outer `/rollout`
   handler hard-whitelisted `provider`/`model`/`reasoning_effort`, so
   `proposer_timeout_seconds: 900` never reached the wire and the arm ran at the
   300s default — detectable only from a 301s wall clock. Now passes known fields
   through and **422s on unknown ones**. An eval that quietly ignores a knob you
   set is worse than one that refuses it.

4. **`reasoning_effort` defaults behind your back.** The service defaults it to
   medium and `_policy_arm` defaulted it to low. For arms that differ *only* on
   effort, set it explicitly on every arm. The OpenAPI text already warns about
   this.

5. Host `~/.codex/models_cache.json` is stale — no `base_instructions` field, so
   `codex_models_manager` rejects it and the refresh times out. Killed one
   luna-medium arm. Regenerable by deleting the file; untouched because it is the
   user's live Codex home. Two chatgpt arms also share one `codex_home` and can
   race on it — `codex_home` is now per-arm overridable via `proposer_io`.

---

## Upstream issues — the turn-termination bug is not ours

- [openclaw#84076](https://github.com/openclaw/openclaw/issues/84076) — "Codex
  app-server stalls after `item/completed`, then aborts without recovery/status".
  Exact match for our symptom, **reproduces on OpenAI gpt-5.5**, open. Their
  proposed fixes are "preserve structured recovery results when idle timeout
  fires" and "retry/resume that summarizes already-completed tool calls" — i.e.
  the harvest-on-stall / repair-round idea, converged on independently.
- [openai/codex#30523](https://github.com/openai/codex/issues/30523) — custom
  OpenAI-compatible Responses provider ends turns with no function call. Open,
  unassigned, four sibling issues.

**Implication:** a luna arm can lose a completed proposal set the same way and
you would see a bare "failed". Whatever else you do, log what is on disk when a
turn stalls.

---

## Suggested order for the next session

1. **Fix the reward metric** (`scoring.py`). Nothing else in the eval means
   anything until exploration stops dominating. Normalise by minibatch size or
   reweight.
2. **Exercise the 9 new fixtures** against live GEPA — start with `train:3`
   (two run ids in its state history, best test of the FK fix) and `train:4/5`
   (weaker seed, more headroom).
3. **Close the verify loop** — schema repair round on
   `validate_manifest_contract` failure.
4. **Wire PRBench into the catalog** as `prbench:0` (fixture + `fixtures.py` +
   env pool), then a live smoke.
5. Re-run the 3-way on the fixed metric. Test the over-fitting hypothesis
   directly.

---

## File map (this session's changes)

| Path | Change |
|---|---|
| `rust/crates/synth_gepa/src/lib.rs` | `restore_state_machine_from_cursor` rebinds run_id (**the FK fix**); labelled terminal writes |
| `rust/crates/synth_gepa/src/service.rs` | `allow_unverified_model`, `model_context_window`, `model_auto_compact_token_limit` on the wire |
| `rust/crates/synth_gepa/src/codex_app_server.rs` | shared evidence-field constants; `proposal_validator_script`; README verify instruction; unit test |
| `rust/crates/synth_gepa/openapi/gepa-service-v1.yaml` | three new ProposerSpec fields documented |
| `rust/crates/synth_optimizer_platform/src/error.rs` | `OptimizerError::SqliteAt` |
| `rust/crates/synth_optimizer_platform/src/failures.rs` | `SqliteAt` → same `sqlite_error` code |
| `rust/crates/synth_optimizer_platform/src/workspace.rs` | `sql_at` / `labelled`; labelled FK-capable writes |
| `rust/crates/synth_optimizer_platform/src/config.rs` | two `ProposerConfig` token fields |
| `rust/crates/synth_optimizer_platform/src/agent_runtime/codex_home.rs` | **command-based auth block**; token config emission |
| `temp/gepa_proposer/gepa_proposer/app.py` | arm passthrough + 422 on unknown policy fields |
| `temp/gepa_proposer/gepa_proposer/episode.py` | OpenRouter proposer spec; per-arm `proposer_io` |
| `temp/gepa_proposer/gepa_proposer/fixtures.py` | catalog 14 → 23 |
| `temp/gepa_proposer/run_luna_parallel.py` | `--arms` JSON; refuses reused inners; ports `8765,8766` |
| `temp/gepa_proposer/mint_checkpoint_fixtures.py` | **new** offline fixture minter |
| `temp/gepa_proposer/prbench_container/` | **new** PRBench container |
| `temp/gepa_proposer/fixtures/*.json` | 9 new + usage backfills |
| `temp/gepa_proposer/tests/test_contract.py` | 18 → 37 |
| `synth-cookbooks-public/.../banking77_container/synth_service_app.py` | Responses branch parity + incomplete guard |

## Drivers

```bash
cd temp/gepa_proposer
# N arms, one inner port per arm (exclusive lock — not just a slowdown)
.venv/bin/python run_luna_parallel.py --skip-build --task-id train:1 \
  --proposer-port 8879 --inner-ports 8765,8766 \
  --arms '[{"label":"luna-low","provider":"openai","model":"gpt-5.6-luna","reasoning_effort":"low"},
           {"label":"luna-medium","provider":"openai","model":"gpt-5.6-luna","reasoning_effort":"medium"}]'

# OpenRouter arm
--arms '[{"label":"nemotron","provider":"openrouter","model":"nvidia/nemotron-3.5-lightning"}]'

uv run --with fastapi --with httpx --with pytest pytest -q   # expect 37 passed
```

After any Rust change: `uv run --with maturin maturin build --release --out target/wheels`
then `uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache <whl>`.
**Never reinstall mid-experiment** — each replicate spawns a fresh service from
`.venv/bin/synth-optimizers`, so a mid-matrix install silently makes replicates
incomparable.

## Don't

- Don't trust a "failed" GEPA run to mean the optimization failed — check
  `optimizer_state_history` for `→ completed` first.
- Don't compare per-arm cost from `usage` without subtracting the fixture baseline.
- Don't put the proposer on 8877, or reuse an inner container you did not start.
- Don't flip `api_family` to `responses` expecting parity outside Banking77 —
  only that container has a Responses branch, and only it has been fixed.
- Don't quote PRBench rewards as PRBench leaderboard scores.
- Don't add nemotron to `VERIFIED_OPENROUTER_MODELS`; use `allow_unverified_model`.
- Don't commit unless asked.
