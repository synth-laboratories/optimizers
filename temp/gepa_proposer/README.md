# Temp GEPA proposer eval / RL env

One `http_task` container. Task = `(downstream container, GEPA cursor fixture)`.
Eval and RL env share this image; RL adds checkpoint + `resume_async`.

**The eval is `POST /rollout`.** A scored episode is a finished GEPA optimizer
run against the inner task container, scored from the terminal cursor.

## Contract

- `GET /health` advertises `contract_version = 2026-05-28`
- `GET /metadata` advertises `optimizer_contracts.gepa.v2` with `/rollouts`
- `GET /compatibility?target=harbor_proxy` is supported (async, catalog, verifier)
- `GET /dataset` + `POST /dataset/rows` list fixture tasks
- Sync scored rollout requires `GEPA_SERVICE_URL`; otherwise **503**
- Reward is `reward_info.outcome_reward` from a succeeded optimizer cursor:
  train exploration + train exploitation + eval uplift

## Parallel rollouts

GEPA service exclusive-locks one inner `container_url` at a time. Run two inner
http_task processes and pass a pool:

```bash
export BANKING77_URLS=http://127.0.0.1:8765,http://127.0.0.1:8766
export HEALTHBENCH_URLS=http://127.0.0.1:8114
export OFFICEQA_URLS=http://127.0.0.1:8120
export CRAFTER_URLS=http://127.0.0.1:20055
export TAU2_URLS=http://127.0.0.1:8774
export MINIGRID_URLS=http://127.0.0.1:8769
```

Each outer `POST /rollout` picks the next URL. Isolated cache namespaces come
from unique optimizer `run_id`s. Do not share one inner URL across arms.

## Tasks

`gen` is the archive generation the cursor was frozen at; `cands` is how many
candidates the fixture starts from. `usage` says whether the frozen
`checkpoint.snapshot.usage` totals are real archive numbers or zeros.

| task_id | label | downstream | gen | cands | train / heldout | usage |
|---|---|---|---|---|---|---|
| `train:0` | banking77-fresh | banking77 | 0 | 1 | 100 / 200 | real |
| `train:1` | banking77-first-checkpoint | banking77 | 1 | 7 | 100 / 200 | real |
| `train:2` | banking77-mature | banking77 | 2 | 13 | 100 / 200 | real |
| `train:3` | banking77-gen3 | banking77 | 3 | 19 | 100 / 200 | real |
| `train:4` | banking77-async-fresh | banking77 | 0 | 1 | 100 / 200 | real |
| `train:5` | banking77-async-first-checkpoint | banking77 | 1 | 7 | 100 / 200 | real |
| `healthbench:0` | healthbench-fresh | healthbench2 (`system_prompt`, Groq) | 0 | 1 | 60 / 50 | real |
| `healthbench:1` | healthbench-first-checkpoint | healthbench2 | 1 | 4 | 60 / 50 | real |
| `healthbench:2` | healthbench-mature | healthbench2 | 1 | 6 | 60 / 50 | real |
| `healthbench:3` | healthbench-openai-scored-seed | healthbench2 (`system_prompt`, OpenAI) | 0 | 1 | 2 / 2 | real |
| `healthbench:4` | healthbench-accepted-frontier | healthbench2 | 1 | 2 | 60 / 50 | real |
| `crafter:0` | crafter-fresh | crafter (`react_system_prompt`) | 0 | 1 | 8 / 8 | real |
| `crafter:1` | crafter-first-checkpoint | crafter | 2 | 4 | 8 / 8 | real |
| `crafter:2` | crafter-mature | crafter | 2 | 4 | 8 / 8 | real |
| `crafter:3` | crafter-archive-fresh | crafter | 0 | 1 | 8 / 8 | real |
| `crafter:4` | crafter-archive-mature | crafter | 2 | 4 | 8 / 8 | real |
| `tau2:0` | tau2-retail-fresh | tau2 (`domain_policy`) | 0 | 1 | 20 / 16 | zeros (seed-only) |
| `tau2:1` | tau2-retail-first-checkpoint | tau2 | 1 | 3 | 20 / 8 | real |
| `tau2:2` | tau2-retail-mature | tau2 | 6 | 17 | 20 / 8 | real |
| `minigrid:0` | minigrid-empty-fresh | minigrid (`system_prompt`) | 0 | 1 | 8 / 4 | zeros (seed-only) |
| `minigrid:1` | minigrid-empty-first-checkpoint | minigrid | 1 | 3 | 8 / 4 | real |
| `minigrid:2` | minigrid-empty-mature | minigrid | 4 | 12 | 8 / 4 | real |
| `officeqa:0` | officeqa-fresh | officeqa (`system_prompt`) | 0 | 1 | 24 / 16 | zeros (seed-only) |

Every fixture is a `generation_start` cursor. `generation_boundary` summaries
are never used. All 23 route to the inner container through the downstream dict
in `gepa_proposer/fixtures.py`, so the inner **policy** has one home: the fixture
JSON of an archive-minted row carries no embedded `downstream`.

## Minting more checkpointed fixtures

`mint_checkpoint_fixtures.py` does two offline jobs over existing archives — no
optimizer run, no spend:

```bash
cd temp/gepa_proposer
python mint_checkpoint_fixtures.py               # mint + backfill, idempotent
python mint_checkpoint_fixtures.py --mint-only
```

- **mint** exports a `generation_start` cursor straight out of a
  `workspace.sqlite`. It is `export_checkpoints.py` generalised: that script is
  hardcoded to three Banking77 generations and only accepts checkpoints carrying
  `metadata.retain`, a flag the proposer-eval fork path writes and a plain
  cookbook run does not. The real bar is `generation_start` + not a compacted
  `checkpoint_summary.v1`.
- **backfill** freezes real `usage` totals into fixtures that were reconstructed
  with `usage: {}`, read from the source archive's own checkpoint at the point
  each was reconstructed from (recorded in `usage_source` on the fixture). The
  three seed-only fixtures get explicit zeros: nothing was rolled out in those
  cursors, so zero is the true total, not a placeholder.

A minted cursor's `state_history` carries the **originating** run's `run_id`
(`train:3` carries two, since it was minted from a fork of the `mfg` lineage).
That is expected: the engine rebinds it on import. It used to end a run with
`sqlite error: FOREIGN KEY constraint failed`.

Banking77 `train:3` is the deepest cursor in the catalog (19 candidates), minted
from a completed proposer-eval fork of the `b77_gepa_eval_mfg_20260819` lineage.
`train:4/5` are a second, independent Banking77 lineage
(`banking77_gepa_async_t50_mb20_h100_735a9c29`) whose seed is much weaker
(0.54 full-train vs 0.68), so the proposer has real headroom instead of a
near-ceiling parent.

`crafter:3/4` are the archive-native counterparts of `crafter:0/1` — same run,
same generations, but the cursor is the archive's own snapshot with
`state_history`, `program`, and `rollout_task_id` intact instead of the
compacted reconstruction.

HealthBench `0/1/2/4` are reconstructed generation-start cursors from
`healthbench_groq_gepa_aug13i` (seed / first scored children / latest archive /
seed+accepted). Inner policy is Groq `llama-3.1-8b-instant`. `healthbench:3` is
the scored OpenAI `gpt-4.1-nano` seed from `healthbench2_eval_smoke_openai_v2`
(2 train / 2 heldout, real heldout reward). Same healthbench2 container, different
inner policy.

Crafter `0/1/2` are compacted generation-start cursors from
`crafter_gepa_public_0fbad055` (gen 0 start / gen 2 start / completed-with-heldout
re-exported as `generation_start`); `crafter:3/4` are the same run's gen 0 and
gen 2 cursors taken verbatim from the archive. Inner policy is OpenAI `gpt-4.1-nano`. The
inner container lives in `synth-cookbooks-public/cookbooks/optimizers/gepa/crafter_container`.
Sensor frames are stripped in `crafter:0/1/2` and kept in `crafter:3/4`;
train/heldout ids are the original `ep_train_*` / `ep_test_*` episode ids in both.

Other cookbook families (HotpotQA, HoVer, FinQA, TBLite, DungeonGrid,
Harvey LAB) have GEPA configs here but no searchable workspace archive, so
they are not catalogued. `crafter_gepa_public_2a373d68` (2 train rows, 1 heldout
row, 2 candidates) and the `craftax_gepa_luna_med_*` runs (seed 0.0, ≤ 2
candidates) are too weak and were not added. `healthbench_groq_gepa_aug13i`
stores only compacted `checkpoint_summary.v1` snapshots — that is why HealthBench
is reconstructed rather than minted. The `gamebench_levers/*` and
`craftax_levers/*` archives belong to the custom-levers / `policy_script`
product, not this eval, and are deliberately out of scope.

τ²-bench retail is Sierra's customer-service agent benchmark
([tau2-bench](https://github.com/sierra-research/tau2-bench)). The inner
container is `tau2_container/`; GEPA mutates `domain_policy`. Official retail
split is 74 train / 40 test; the fixtures use the first 20 train ids because
each rollout is a multi-turn user simulator. `tau2:0` stays the seed-only
fixture (16 heldout ids, no rollouts behind it). `tau2:1/2` are minted from the
`tau2_retail_gepa_20260819_long` inner GEPA run (train 0.050 -> 0.200, heldout
0 -> 0.125), which used 8 heldout ids — a cheaper heldout pass than `tau2:0`.

```bash
cd temp/gepa_proposer/tau2_container
uv sync                          # Python >= 3.12
set -a && source /path/to/.env && set +a   # needs OPENAI_API_KEY
python synth_service_app.py --port 8774
export TAU2_URLS=http://127.0.0.1:8774
```

Retail tasks, DB, and user-simulator guidelines are vendored under
`tau2_container/data/`. Without `OPENAI_API_KEY`, inner `/rollout` is 503.

MiniGrid is Farama `MiniGrid-Empty-5x5-v0` (DoorKey via `MINIGRID_ENV_ID`).
The inner container is `minigrid_container/`; GEPA mutates `system_prompt`.
Each rollout is a live gymnasium episode (up to 48 policy steps). All three
fixtures use 8 train seeds and 4 heldout seeds. `minigrid:0` is seed-only;
`minigrid:1/2` are minted from the `minigrid_empty_gepa_20260819` inner GEPA
run (train 0.353 -> 0.842, heldout 0.226 -> 0.599).

```bash
cd temp/gepa_proposer/minigrid_container
uv sync
set -a && source /path/to/.env && set +a   # needs OPENAI_API_KEY
python synth_service_app.py --port 8769
export MINIGRID_URLS=http://127.0.0.1:8769
```

OfficeQA is Databricks' Treasury-Bulletin grounded-reasoning benchmark
([blog](https://www.databricks.com/blog/introducing-officeqa-benchmark-end-to-end-grounded-reasoning),
dataset `databricks/officeqa`, scorer `github.com/databricks/officeqa`).
The inner container is `officeqa_container/`. Questions are gated on Hugging Face:
set `OFFICEQA_CSV` (and optionally `OFFICEQA_CORPUS_DIR` for parsed txt). Without
the CSV, `/rollout` on the inner container is 503. Fixture train/heldout ids are
`train:0..23` / `heldout:0..15` (easy / hard once the CSV is mounted).

## Required for a scored episode

```bash
export GEPA_SERVICE_URL=http://127.0.0.1:8088
export BANKING77_URLS=http://127.0.0.1:8765,http://127.0.0.1:8766
export GEPA_PROPOSER_STATE_DIR=/tmp/gepa-proposer-state
```

```json
{
  "task_id": "train:1",
  "submission_mode": "async",
  "candidate": {},
  "policy": {
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "reasoning_effort": "low"
  },
  "episode": {"proposer_rounds": 1, "skip_heldout": false}
}
```

## Run luna low vs medium in parallel

```bash
cd temp/gepa_proposer
uv run python run_luna_parallel.py --skip-build
```

Omit `--skip-build` to rebuild the local engine wheel first (`--no-cache`).

## Tests

```bash
cd temp/gepa_proposer
uv run --with fastapi --with httpx --with pytest pytest -q
```
