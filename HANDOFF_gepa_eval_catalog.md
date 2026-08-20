# Pickup: expand GEPA proposer eval catalog, then nail the eval

Repo: `optimizers` on `backup/last72h-optimizer-platform-20260813`. **Uncommitted.
Do not commit unless asked.**

This is the live work: grow the outer `temp/gepa_proposer/` task set (more
downstream containers + cursor fixtures), then get a real luna-low vs luna-med
eval, not a $1 / 5-minute smoke.

Read first, in order. Do not rediscover.

1. [`SCOPE_gepa_eval_rl_env.md`](SCOPE_gepa_eval_rl_env.md) — eval **is** the RL env. One image.
2. [`HANDOFF_proposer_eval_container.md`](HANDOFF_proposer_eval_container.md) — reward terms, Banking77 shape, engine “already there”.
3. [`HANDOFF_gepa_eval_pickup.md`](HANDOFF_gepa_eval_pickup.md) — earlier same-day pickup (heldout budget, ports, wheel cache).
4. [`aug19_gepa.md`](aug19_gepa.md) **§7** — wheel cache, hung run looks healthy, `[dataset]` vs `[taskset]`.
5. [`temp/gepa_proposer/README.md`](temp/gepa_proposer/README.md) — current catalog table.

**Not this product:** [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md) /
[`HANDOFF_gepa_custom_levers.md`](HANDOFF_gepa_custom_levers.md). That is inner
apply / `policy_script` / Craftax levers. Do not fold it into this eval, and do
not reuse GELO `rlvr` flags.

---

## What you are building

One Harbor-ish `http_task` at `temp/gepa_proposer/`.

| | |
|---|---|
| Task | `(downstream container, GEPA cursor fixture)` |
| Policy under eval | the **GEPA proposer** (v0 outer `candidate` stays `{}`) |
| Action | propose candidates for one bounded episode |
| Eval | `POST /rollout` on the **outer** container |
| Reward | train exploration + train exploitation + eval uplift from the **terminal optimizer cursor** |
| Scalar | unweighted sum. `objective_scores` carries the three terms. Do not invent a fourth metric |

Isolation = unique optimizer `run_id`. GEPA service exclusive-locks **one**
inner `container_url`. Parallel arms need two inner processes.

Luna models (`gpt-5.6-luna`) are ChatGPT-home proposers. `proposer.copy_host_auth`
requires `codex_home` (`~/.codex`). Auth mode defaults to `chatgpt`.

Use `.venv/bin/synth-optimizers`, not `uv run`. After a local wheel:
`uv pip install --python .venv/bin/python --no-deps --reinstall --no-cache <whl>`.
Set `SYNTH_OPTIMIZERS_VL_PROJECT=0`. Credentials:
`/Users/joshuapurtell/Documents/GitHub/synth-ai/.env`. Do not print keys.

---

## Catalog now (23 fixtures)

Wired in `temp/gepa_proposer/gepa_proposer/fixtures.py`. Loaded from
`temp/gepa_proposer/fixtures/*.json`. Contract tests are **37 passed**
(18 before the 2026-08-19 late-evening fixture mint).

| task_id | label | field | inner | train / heldout | gen / cands | fixture kind |
|---|---|---|---|---|---|---|
| `train:0` | banking77-fresh | `stage2_system` | cookbook `banking77_container` | 100 / 200 | 0 / 1 | real cursor + usage |
| `train:1` | banking77-first-checkpoint | `stage2_system` | same | 100 / 200 | 1 / 7 | gen-start; **heldout_reward null** |
| `train:2` | banking77-mature | `stage2_system` | same | 100 / 200 | 2 / 13 | later archive |
| `train:3` | banking77-gen3 | `stage2_system` | same | 100 / 200 | 3 / 19 | deepest cursor; from a completed proposer-eval fork of the `mfg` lineage |
| `train:4` | banking77-async-fresh | `stage2_system` | same | 100 / 200 | 0 / 1 | second lineage (`..._async_t50_mb20_h100_735a9c29`), seed 0.54 |
| `train:5` | banking77-async-first-checkpoint | `stage2_system` | same | 100 / 200 | 1 / 7 | second lineage |
| `healthbench:0` | healthbench-fresh | `system_prompt` | Groq `llama-3.1-8b-instant` | 60 / 50 | 0 / 1 | reconstructed; usage backfilled |
| `healthbench:1` | first-checkpoint | `system_prompt` | Groq | 60 / 50 | 1 / 4 | reconstructed; usage backfilled |
| `healthbench:2` | mature | `system_prompt` | Groq | 60 / 50 | 1 / 6 | reconstructed; usage backfilled |
| `healthbench:3` | openai-scored-seed | `system_prompt` | OpenAI `gpt-4.1-nano` | **2 / 2** | 0 / 1 | real heldout reward; usage backfilled |
| `healthbench:4` | accepted-frontier | `system_prompt` | Groq | 60 / 50 | 1 / 2 | reconstructed; usage backfilled |
| `crafter:0/1/2` | crafter fresh/first/mature | `react_system_prompt` | cookbook `crafter_container` | 8 / 8 | 0,2,2 / 1,4,4 | compacted; usage backfilled |
| `crafter:3` | crafter-archive-fresh | `react_system_prompt` | same | 8 / 8 | 0 / 1 | archive-native cursor |
| `crafter:4` | crafter-archive-mature | `react_system_prompt` | same | 8 / 8 | 2 / 4 | archive-native cursor |
| `tau2:0` | tau2-retail-fresh | `domain_policy` | `temp/gepa_proposer/tau2_container` | 20 / 16 | 0 / 1 | seed-only |
| `tau2:1` | tau2-retail-first-checkpoint | `domain_policy` | same | 20 / **8** | 1 / 3 | archive (`tau2_retail_gepa_20260819_long`) |
| `tau2:2` | tau2-retail-mature | `domain_policy` | same | 20 / **8** | 6 / 17 | archive |
| `minigrid:0` | minigrid-empty-fresh | `system_prompt` | `temp/gepa_proposer/minigrid_container` | 8 / 4 | 0 / 1 | seed-only Empty-5x5 |
| `minigrid:1` | minigrid-empty-first-checkpoint | `system_prompt` | same | 8 / 4 | 1 / 3 | archive (`minigrid_empty_gepa_20260819`) |
| `minigrid:2` | minigrid-empty-mature | `system_prompt` | same | 8 / 4 | 4 / 12 | archive |
| `officeqa:0` | officeqa-fresh | `system_prompt` | `officeqa_container` | 24 / 16 | 0 / 1 | seed-only; needs `OFFICEQA_CSV` |

`checkpoint.snapshot.usage` is now a real frozen archive total on every fixture
except the three seed-only ones (`tau2:0`, `minigrid:0`, `officeqa:0`), which
carry explicit zeros — nothing was rolled out in those cursors. Each fixture
records where its totals came from in `usage_source`. No fixture ships `{}`.

**None of `train:3/4/5`, `crafter:3/4`, `tau2:1/2`, `minigrid:1/2` has been
exercised against a live GEPA run.** They are minted, wired, and covered by
contract tests only.

Minting more: `temp/gepa_proposer/mint_checkpoint_fixtures.py` (offline over
existing archives, idempotent, no spend). It is `export_checkpoints.py`
generalised — that script is hardcoded to three Banking77 generations and
requires `metadata.retain`, which only the proposer-eval fork path writes, so it
silently skips every plain cookbook archive. The real bar is `generation_start`
and not a compacted `checkpoint_summary.v1`.

Not catalogued (cookbook GEPA configs exist, no searchable workspace archive):
HotpotQA, HoVer, FinQA, TBLite, DungeonGrid, Harvey LAB.

Rejected archives: `crafter_gepa_public_2a373d68` (2 train rows / 1 heldout row /
2 candidates), `craftax_gepa_luna_med_*` (seed 0.0, <= 2 candidates),
`banking77_gepa_luna_med_*` and `banking77_gepa_sol_med_*` (2 candidates each),
`healthbench_groq_gepa_aug13i` snapshots (compacted `checkpoint_summary.v1` —
hence the reconstruction), and in-flight runs still in `rollout_running` /
`proposing`. `gamebench_levers/*` and `craftax_levers/*` belong to the
custom-levers product, not this eval.

`runs/gepa_<hex>` are proposer-eval fork runs. Of their Banking77
generation_start cursors, 48 carry candidate payloads identical to a catalog
fixture (the imported parent, nothing proposed before the run died) and 26 carry
novel luna-proposed children. All but two of the novel ones are gen 2 / 13
candidates — the same shape `train:2` already covers, so only the two gen 3 / 19
cursors were worth minting. `train:3` is the better of the two
(`gepa_24b32fd4…`: heldout 0.675/0.670 and a **new best** at gen 3;
`gepa_274f9683…`: heldout 0.665/0.660, best unchanged).

---

## Last live catalog comparison (budget-capped, not a pass)

Driver: `temp/gepa_proposer/run_luna_catalog.py --skip-build`.
Result: `temp/gepa_proposer/generated/catalog_20260819214935/luna_catalog_low_vs_med.json`.
Limits: `proposer_rounds=1`, `skip_heldout=false`, `max_wall_seconds=300`,
`max_spend_usd=1.0`. Wall ~28 min. Exit 1.

**Only scored pair:** `healthbench:3` (tiny OpenAI inner, heldout actually ran).

| arm | reward | exploration | exploitation | eval uplift |
|---|---|---|---|---|
| luna low | −0.257 | 0 | −0.257 | 0 |
| luna medium | **−0.172** | 0 | −0.172 | 0 |

Medium lost less train score. Neither found a new best nor moved heldout.
N=1 under a 5-minute cap. **Do not treat this as the v0 pass.**

Everything else:

| family | what happened |
|---|---|
| `train:0` | waiter timeout 390s (episode still running) |
| `train:1` / `train:2` | GEPA episode `failed`; outer refused to score |
| `healthbench:0/1/2/4` | Groq inner rollouts `infra failure rate 1.00` or missing score vectors |
| `crafter:*` | GEPA episode `failed` |
| `tau2:0` / `minigrid:0` | still `running` when the driver wrote JSON and killed children |
| `officeqa:0` | skipped; no `OFFICEQA_CSV` (inner `/rollout` is 503) |

---

## How to add a family (outer catalog)

A catalog row is three pieces. Kind is the inner load/run contract; the fixture
is the GEPA cursor. Do not add language-named kinds.

### 1. Inner http_task that GEPA can actually drive

Must speak `synth_optimizers.gepa.v2`:

- `GET /health`, `GET /metadata` with `optimizer_contracts.gepa` and
  `capabilities.metadata.policy_ready`
- `GET /program` with `target_modules` + `seed_candidate` for the mutated field
- `GET /taskset` + `POST /taskset/tasks`
- `POST /rollout` returning scalar `reward` / `reward_info.outcome_reward`

**GEPA sends `"task_id": <program id>` and `"task": <row>`.** New inners must
parse the example from `payload.task` (see TAU2 / MiniGrid). If you treat the
program id as the retail/env task id, every rollout hits task `0`.

Unsolved env episodes must still return `success_status=succeeded` with scalar
reward 0. Infra fail ≠ unsolved.

Advertise apply only if the inner owns it. This eval path is **prompt overlay**
by default (`prompt_overlay.v1`). Do not stuff policy source into
`react_system_prompt`.

### 2. Cursor fixture JSON

Put it in `temp/gepa_proposer/fixtures/`. Schema `gepa_cursor_fixture.v1`.
Must contain:

- `task_id`, `label`, `cursor` with `train_rows` + `heldout_rows`
- `checkpoint` (GEPA imports **this**, not the forked outer cursor)
- `checkpoint.snapshot.usage` with `prompt_tokens` / `completion_tokens` /
  `total_tokens` / `rollout_calls` / `proposer_calls` (empty `{}` used to 500
  the engine; `build_run_request` now fills zeros, but freeze real totals when
  you have them)

Prefer a **generation-start** cursor from a real GEPA archive
(`export_checkpoints.py`, `reconstruct_*.py`). Seed-only is OK when there is no
archive (TAU2, MiniGrid, OfficeQA) but then exploration is starting from a
single seed — say so.

Do not use `generation_boundary` summaries as fixtures.

### 3. Wire it

- Downstream dict in `fixtures.py` (`url_env`, `url_pool_env`, `candidate_field`,
  inner **policy** for the task container, not the luna proposer).
- `_infer_downstream` + load `order`.
- `program_for_task` seed text if the cursor payload is empty.
- Env pool in `run_luna_catalog.py` / README (`FOO_URLS=http://127.0.0.1:port1,port2`).
- Contract test that `build_run_request` sends the right inner policy + task ids.
- Two inner processes if you want parallel luna arms on that family.

Generators already in tree: `generate_tau2_fixture.py`,
`generate_minigrid_fixture.py`, `generate_officeqa_fixture.py`,
`reconstruct_healthbench_fixture.py`, `reconstruct_crafter_fixture.py`,
`export_checkpoints.py`.

### Inner containers that already exist

| family | where | default port | notes |
|---|---|---|---|
| Banking77 | `synth-cookbooks-public/cookbooks/optimizers/gepa/banking77_container` | 8765 / 8766 | leftover **8877** has been a live Banking77; do not put the proposer there |
| HealthBench | cookbook `healthbench_groq` | 8114 | `--storage-root` required; Groq inner + OpenAI grader |
| Crafter | cookbook `crafter_container` | 8768 | JAX; fixture policy is OpenAI nano, not cookbook Gemini |
| TAU2 retail | `temp/gepa_proposer/tau2_container` | 8774 | vendored `data/`; native τ² orchestrator |
| MiniGrid | `temp/gepa_proposer/minigrid_container` | 8769 | default `MiniGrid-Empty-5x5-v0`; DoorKey via `MINIGRID_ENV_ID` never scored with nano |
| OfficeQA | `temp/gepa_proposer/officeqa_container` | 8120 | gated HF CSV |

MiniGrid / TAU2 inner GEPA smokes already landed (separate from this eval):

- TAU2 `tau2_retail_gepa_20260819_long`: seed train 0.050 → best 0.200, heldout 0 → 0.125
- MiniGrid Empty-5x5 `minigrid_empty_gepa_20260819`: seed 0.353 / 0.226 → best 0.842 / 0.599

Those are **inner** GEPA scores. They are not outer proposer-eval rewards.

---

## File map

| Path | Role |
|---|---|
| `temp/gepa_proposer/gepa_proposer/app.py` | Outer FastAPI; pool round-robin; score only succeeded+heldout |
| `temp/gepa_proposer/gepa_proposer/episode.py` | `build_run_request`, stop conditions, inner policy, fixture checkpoint usage fill |
| `temp/gepa_proposer/gepa_proposer/fixtures.py` | Catalog + downstream inference + fork usage totals |
| `temp/gepa_proposer/gepa_proposer/scoring.py` | Three-term reward |
| `temp/gepa_proposer/run_luna_catalog.py` | All reachable seeds, luna low vs med, $1 / 5 min |
| `temp/gepa_proposer/run_luna_parallel.py` | Single `task_id` (default `train:1`) |
| `temp/gepa_proposer/tests/test_contract.py` | Contract + scoring |
| `rust/crates/synth_gepa/src/lib.rs` | `UsageTotals` serde default; fixture candidates registered before heldout |
| `rust/crates/synth_gepa/src/machines.rs` | `Registered → heldout_evaluating` via `heldout_started` |

---

## Engine / harness traps (already hit)

Fix these before expanding blindly. Several are in the working tree and need a
**rebuilt wheel** in `.venv` (`synth_optimizers-0.2.6.dev20260626` after the
2026-08-19 evening maturin).

1. **Empty fixture `usage: {}`** → `json error: missing field prompt_tokens` at
   `generation_start`. `UsageTotals` now has `#[serde(default)]`. Outer
   `build_run_request` also fills checkpoint + snapshot usage. Reconstructed
   HealthBench / Crafter / TAU2 / MiniGrid fixtures still ship `{}`.
2. **Heldout from a fixture candidate with no FSM row** →
   `invalid optimizer state transition <missing> -> heldout_evaluating`.
Banking77 `train:1/2` still failed after the FSM patch. The sqlite error was
   **`FOREIGN KEY constraint failed`**, not the missing-state transition. Fixture
   archives have parented candidates; persist wrote children (or score vectors)
   before the parent row existed. `persist_candidate_registry` now orders
   parents first, and `import_cursor_fixture` writes the snapshot registry on
   import. Rebuild the wheel before the next Banking77 live run.

   **2026-08-19 22:42 retry** (`parallel_20260819224210`, `train:1`, 30 min /
   $15, heldout on): both arms still failed. Luna-low died on **inner rollout
   infra failure rate 0.28 > 0.25** (window=32). Luna-medium still died on
   **`FOREIGN KEY constraint failed`**. Parent-order + import persist did not
   clear the FK. Next: dump the failing SQL from the run workspace DB, and
   check leftover Banking77 on **8877** (old process) vs a fresh 8765.
3. **SQLite on a 99%-full Documents volume** → `sqlite error: disk I/O error`.
   Catalog driver now puts the service DB under `/tmp/gepa-catalog-*.sqlite`.
   Keep GEPA concurrency low (catalog uses 4 workers, 2 families at a time).
4. **`skip_heldout=false` + missing heldout** → outer 409
   `refusing to score … without heldout`. Do not flip skip back to true.
   `max_heldout_rollouts` is already 8000 in `episode.py`. Waiter headroom is
   +90s (`GEPA_PROPOSER_WAIT_HEADROOM_SECONDS`). Catalog poll is wall+120s;
   `train:0` still overran 390s.
5. **`GepaServiceRunRequest` is `deny_unknown_fields`.** Do not send `cache` on
   `POST /runs`.
6. **Do not attach to leftover proposer ports.** 8877 has been Banking77.
   Catalog proposer default is **8879**.
7. **Groq HealthBench inners** failed live (`HTTPStatusError` on evaluator span,
   then `infra failure rate 1.00`). `healthbench:3` worked because inner policy
   is OpenAI. Fix Groq byok / `base_url` / key injection before treating
   `healthbench:0–2,4` as eval tasks.
8. **Crafter** failed the catalog pass; JAX + 20-turn nano episodes + 5 min is
   tight. Confirm inner `/rollout` succeeds standalone before another luna arm.
9. **OfficeQA** is not coverage until `OFFICEQA_CSV` is mounted.

---

## How to nail the eval

SCOPE pass, not the $1/5-min catalog:

1. **Banking77 full shape first.** Fixture with 100 train rows (already true for
   `train:*`). Heldout in SCOPE is 16+; frozen fixtures have **200**. Decide
   explicitly whether to shrink the heldout pool in the fixture or pay for 200.
   `train:1` has null pre-fork `heldout_reward`; eval uplift vs 0 is still a
   real inner number, not a train proxy.
2. **`skip_heldout=false`**, 6 proposals (`GEPA_PROPOSER_PROPOSALS` / pipeline
   default), `proposer_rounds=1`.
3. **N ≥ 3** independent episodes per arm, same `task_id`, unique `run_id`s,
   two inner URLs. Report mean ± spread of each of the three terms. Pass = arms
   separate by more than that spread, and you can say **which term**.
4. Size wall/spend so heldout actually finishes. 5 minutes is not enough for
   luna + Banking77 heldout. Raise `max_wall_seconds` and waiter headroom
   together.
5. Only then add a second family as a robustness check (`healthbench:3` is the
   only one that already scored end-to-end). Groq HealthBench and Crafter are
   not eval tasks until their inner rollouts succeed under GEPA.

Drivers:

```bash
cd temp/gepa_proposer
# one fixture, two arms
.venv/bin/python run_luna_parallel.py --skip-build --proposer-port 8879 --task-id train:1

# every reachable seed (budget flags)
.venv/bin/python run_luna_catalog.py --skip-build \
  --max-spend-usd 1.0 --max-wall-seconds 300
```

Omit `--skip-build` after rust changes.

Tests:

```bash
cd temp/gepa_proposer
uv run --with fastapi --with httpx --with pytest pytest -q
```

Promote `temp/gepa_proposer/` into `evals/containers/images/gepa-proposer/`
only after the contract and catalog stop moving.

---

## Don't

- Don’t split eval vs RL into two images or two observation schemas.
- Don’t treat inner TAU2 / MiniGrid GEPA scores as this eval.
- Don’t use `generation_boundary` checkpoints as fixtures.
- Don’t share one inner URL across parallel arms (`container_exclusive_conflict`).
- Don’t send `cache` on `POST /runs`.
- Don’t `uv run synth-optimizers` after a local wheel; use `.venv/bin/synth-optimizers`.
- Don’t skip `--no-cache` on wheel reinstall.
- Don’t score a dry fork or a failed optimizer run.
- Don’t flip `skip_heldout` to true to “get a number”.
- Don’t put the proposer on 8877.
- Don’t fold custom levers / Craftax apply into this container.
- Don’t commit unless asked.
