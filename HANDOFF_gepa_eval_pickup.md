# Pickup: GEPA proposer eval container (2026-08-19 afternoon)

Repo: `optimizers` on `backup/last72h-optimizer-platform-20260813`. **Uncommitted.
Do not commit unless asked.**

Read these first, in order, and do not rediscover them:

1. [`aug19_gepa.md`](aug19_gepa.md) **§7** — wheel cache, `grep -c` on binaries, `[dataset]` vs `[taskset]`, hung run looks healthy.
2. [`SCOPE_gepa_eval_rl_env.md`](SCOPE_gepa_eval_rl_env.md) — eval **is** the RL env. One image.
3. [`HANDOFF_proposer_eval_container.md`](HANDOFF_proposer_eval_container.md) — engine “already there” table and original blockers.

This file is **where the work actually is right now**, not the original design.

---

## What you are picking up

A Harbor-ish `http_task` eval whose policy is the **GEPA proposer**. Task = `(downstream container, GEPA cursor fixture)`. Action = propose candidates. Reward = three terms from the **terminal optimizer cursor**:

| Term | Meaning |
|---|---|
| `train_exploration` | Σ `max(0, new_max − prior_holder)` on train `example_id`s |
| `train_exploitation` | episode train mean − pre-fork train mean (same ids) |
| `eval_uplift` | episode heldout mean − pre-fork heldout mean |

Scalar `reward` = unweighted sum of those three. `objective_scores` carries them separately. Do not invent a fourth metric. Do not infer eval uplift from train scores.

The eval **is** `POST /rollout` on `temp/gepa_proposer/`. Do not split the image, do not fold this into GELO `rlvr`, do not use `generation_boundary` fixtures, do not treat `run_luna_eval.py` as the eval, do not treat green pytest as a live eval.

---

## Live state when this was written

A **heldout-on** luna low vs medium run **failed** (~157s, 2026-08-19 18:44Z). Both arms. No rewards.

| | |
|---|---|
| Driver | `temp/gepa_proposer/.venv/bin/python run_luna_parallel.py --skip-build --proposer-port 8878` |
| Result | `temp/gepa_proposer/generated/parallel_20260819184400/luna_low_vs_med.json` |
| Optimizer runs | `gepa_88463863889f49439b251b3b676b124d` (low), `gepa_2e1bc2ab28d34bd68b95ead9e8b6f03f` (medium) |
| Cause | `heldout required but skipped due to limits` on both (`run_requests.error_json` in `gepa-service.sqlite`) |
| Fixture | `train:1` (`banking77_first.json`) |
| Episode | `proposer_rounds=1`, `skip_heldout=false`, `proposals_per_generation=6` |

`gepa_service.log` is empty (engine wrote to sqlite). Do not flip `skip_heldout` back to true. Raise `max_heldout_rollouts` first — see below.

Port **8877 was already occupied** by something else. Do not reuse a live proposer port from an earlier stack — `run_luna_parallel.py` will attach to it and score with **old code**. Pick a free `--proposer-port` or kill the leftover.

---

## What already landed (do not rebuild)

### Engine (`rust/crates/synth_gepa`, `synth_optimizer_platform`)

- Retain / pin / export / import / fork of cursor fixtures.
- `[gepa.episode]` delta-from-fork: `proposer_rounds`, `max_rollouts`, `max_wall_seconds`, `max_spend_usd`, `skip_heldout`.
- HTTP service accepts `proposer.reasoning_effort` (was hardcoded medium).
- Service **exclusive-locks one inner `container_url`**. Parallel outer arms need **two inner http_task processes**, not one shared Banking77 URL. Failure looks like `container_exclusive_conflict`.
- `GepaServiceRunRequest` is `deny_unknown_fields`. Do **not** send a `cache` field on `POST /runs`. Isolation is unique optimizer `run_id`.
- `proposer.copy_host_auth` **requires** `proposer.codex_home` or create_run is **422**. Container now sets `codex_home` to `~/.codex`.
- After an episode horizon, `skip_heldout=false` goes `generation_start` → `pre_heldout` → `heldout` → finalize. `skip_heldout=true` skips to finalizing so train-only episodes can exit 0. Earlier live scoring died with `gepa_terminal_heldout_not_evaluated` / invalid SM transition; skip was the workaround. **Heldout is on again.** If this run fails, that SM path is the first suspect (`lib.rs` `advance_generation_start` ~12281, `advance_heldout` ~12969, `finalize_completed_gepa_run` ~13443).

Wheel in `.venv` was `synth_optimizers-0.2.6.dev20260626`. Rebuild with maturin then:

```bash
uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache <whl>
```

`--no-cache` is mandatory. Invoke `.venv/bin/synth-optimizers`, not `uv run`.

`cargo test -p synth_gepa --lib` was 23 passed earlier today.

### Container (`temp/gepa_proposer/`)

Contract:

- `GET /health` → `contract_version = 2026-05-28`
- `GET /metadata` → `optimizer_contracts.gepa.v2`, `rollout_route = /rollouts`
- `GET /compatibility?target=harbor_proxy`
- `GET /dataset` + `POST /dataset/rows`
- Completed rollout includes `reward_info.outcome_reward` and ISO timestamps
- Checkpoints + `/rollouts/{parent}/resume_async` exist (eval does not need them)
- v0 `candidate` must be empty (422)
- Sync without `GEPA_SERVICE_URL` → 503
- Reward only if optimizer status is `succeeded` / `completed`

Tasks:

| task_id | fixture | downstream |
|---|---|---|
| `train:0` | banking77-fresh | banking77 `stage2_system` |
| `train:1` | banking77-first-checkpoint | banking77 |
| `train:2` | banking77-mature | banking77 |
| `healthbench:0` | reconstructed `healthbench_groq_gepa_aug13i` | healthbench2 `system_prompt`, Groq `llama-3.1-8b-instant` |

HealthBench is **catalogued, not live-scored**. Fixture: `temp/gepa_proposer/fixtures/healthbench_first.json`. Reconstruct script: `reconstruct_healthbench_fixture.py`. Needs `GROQ_API_KEY`.

Parallel: `BANKING77_URLS=http://127.0.0.1:8765,http://127.0.0.1:8766` (round-robin). Live spawn uses a daemon thread + `asyncio.run` so TestClient and uvicorn both work.

Scoring: `temp/gepa_proposer/gepa_proposer/scoring.py`. `skip_heldout` now defaults **false**. Tests: `cd temp/gepa_proposer && uv run --with fastapi --with httpx --with pytest pytest -q` → **13 passed**.

### First live luna comparison (train-only, obsolete as an eval)

`temp/gepa_proposer/generated/parallel_20260819180404/luna_low_vs_med.json`

~145s wall, `skip_heldout=true`, **2 proposals** (engine default; the container now sends 6), **N=1**, fixture `train:1`.

| arm | reward | exploitation | exploration | eval uplift |
|---|---|---|---|---|
| luna low | 0.955 | −0.045 | 1.0 | **not scored** |
| luna medium | **1.143** | **+0.143** | 1.0 | **not scored** |

They separated on exploitation; exploration tied. **This is not a pass.** SCOPE wants N≥3, 6 proposals, heldout on, isolated cache (already true via unique `run_id`).

---

## How to run (after the in-flight job)

Credentials live in `/Users/joshuapurtell/Documents/GitHub/synth-ai/.env` (driver loads it). Inner Banking77 is in `synth-cookbooks-public/cookbooks/optimizers/gepa`. Set `SYNTH_OPTIMIZERS_VL_PROJECT=0`.

```bash
cd temp/gepa_proposer
.venv/bin/python run_luna_parallel.py --skip-build --proposer-port 8879
# omit --skip-build to rebuild the engine wheel first
```

Inner Banking77 env the driver sets: `BANKING77_TRAIN_SAMPLE=100`, `BANKING77_TEST_SAMPLE=200`, `BANKING77_POLICY_CONCURRENCY=128`, `BANKING77_POLICY_TIMEOUT_SECONDS=60`.

Do not share one inner URL across arms. Do not share cache namespaces across arms for the first comparison.

---

## The heldout gap you will hit

`banking77_first.json` is a **generation-start** cursor: 7 candidates, 100 train, **200 heldout rows**, **`heldout_reward` is null on the whole archive**.

Eval uplift = episode heldout mean − pre-fork heldout mean. With a null archive that is **episode heldout vs 0**, not a scored parent delta. That is still a real inner-heldout number; it is not the same object as train exploitation. A later fixture that already finished heldout would give a true delta. Do not fake pre-fork heldout from train scores.

Worse, the container currently sets:

```python
"max_heldout_rollouts": max(1, len(pools["heldout"]))  # 200
```

Heldout eligibility (`lib.rs` `heldout_candidate_eligible`) wants **seed + best** (min 2) when they differ. Budgeted heldout is `available_rollouts / heldout_row_count`. `200/200 = 1` candidate. If the engine needs 2, it selects **none**, then `finalize_completed_gepa_run` fails with `gepa_best_candidate_missing_heldout` because `skip_heldout=false`.

**If the in-flight run fails, raise `max_heldout_rollouts` to `len(heldout) * N` (N ≥ 2, probably 8) before anything else.** 200-row heldout × several candidates is also why this run will be much slower than the 145s train-only pass. SCOPE sized heldout at 16+; the frozen fixture has 200. Shrinking the heldout pool in the fixture is a product choice, not a silent default.

Engine heldout also only writes scalar `heldout_reward` plus sensor frames. Scoring reads `heldout_reward`, `heldout_scores`, or `seed_rewards.heldout`. Scalar is enough.

---

## File map

| Path | Role |
|---|---|
| `temp/gepa_proposer/gepa_proposer/app.py` | Outer FastAPI container |
| `temp/gepa_proposer/gepa_proposer/episode.py` | `build_run_request`, `parse_episode`, inner policy, `codex_home` |
| `temp/gepa_proposer/gepa_proposer/scoring.py` | Three-term reward |
| `temp/gepa_proposer/gepa_proposer/optimizer_client.py` | GEPA HTTP; errors include response body |
| `temp/gepa_proposer/gepa_proposer/fixtures.py` | Task catalog + downstream inference |
| `temp/gepa_proposer/run_luna_parallel.py` | Live driver (heldout **on**) |
| `temp/gepa_proposer/tests/test_contract.py` | Contract + scoring tests |
| `rust/crates/synth_gepa/src/service.rs` | Exclusive lock, `reasoning_effort`, episode stop conditions |
| `rust/crates/synth_gepa/src/episode.rs` | Delta-from-fork horizon |
| `rust/crates/synth_gepa/src/lib.rs` | Heldout SM + terminalization |
| `src/synth_optimizers/gepa.py` | Python `GepaEpisodeConfig.skip_heldout` default **false** |

---

## What “done” still means

1. **This heldout live run actually completes** and prints the three terms. If it fails, fix heldout budget / SM, do not flip `skip_heldout` back to true and call it an eval.
2. **N ≥ 3** repeats per arm, same `task_id`, isolated cache, 6 proposals. Report mean ± spread of each term. Pass = arms separate by more than that spread, and you can say which term.
3. HealthBench live score is optional after Banking77 is real.
4. Promote `temp/gepa_proposer/` into `evals/containers/images/gepa-proposer/` only after the contract stops moving.

---

## Don't

- Don’t split eval vs RL into two images or two observation schemas.
- Don’t use `generation_boundary` checkpoints as fixtures.
- Don’t share one Banking77 URL across parallel arms.
- Don’t send `cache` on `POST /runs`.
- Don’t `uv pip install --reinstall` without `--no-cache`.
- Don’t `uv run synth-optimizers` after a local wheel install; use `.venv/bin/synth-optimizers`.
- Don’t score a dry fork or a failed optimizer run.
- Don’t rediscover `aug19_gepa.md` §7.
- Don’t treat the 18:04 train-only luna numbers as the eval.
- Don’t commit unless asked.
