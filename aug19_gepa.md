# GEPA FlashEvolve — 2026-08-19 working notes

Branch: `backup/last72h-optimizer-platform-20260813`. Everything below is
**uncommitted** in `optimizers` and `synth-cookbooks-public`.

Supersedes the FlashEvolve diagnosis in the 2026-06-02 handoff. Where this file
and `src/synth_optimizers/docs/gepa/algorithms/10-pipeline-modes.md` disagree
about history, this file is newer; where either disagrees with the Rust
comments, the Rust comments win.

---

## 1. The 2026-06-02 diagnosis was incomplete

The handoff attributed FlashEvolve's 0.33s overlap to admission order in
`schedule_async_lane_transition` / `schedule_async_proposer_job`, and framed the
fix as proposer-priority scheduling.

That was not the binding constraint. `advance_gepa_once` executed **every leased
lane job inline on the driver tick**:

```rust
// lib.rs, advance_pending_runtime_job
runtime::execute_one_pending_optimizer_job_from_run_workspace(
    ..., runtime::RuntimeEffectExecutorConfig::inline_default(),
)
```

That call blocks until the job finishes. So two lanes could hold leases
simultaneously and never occupy wall clock simultaneously. **Lane leases modelled
a concurrency the executor never provided.** No admission-order change alone
could have produced overlap.

Admission order was *also* worth fixing, and was fixed — but on its own it would
have moved the needle by approximately nothing.

## 2. Second bug: unbounded duplicate enqueue (livelock)

Found by running the thing, not by reading it. FlashEvolve sat at **80% CPU
emitting zero events for 9 minutes**. Stack sample:

```
schedule_async_lane_transition
  → schedule_async_candidate_minibatches      1762/2461 samples
      → advance_proposer_waiting              1757
          → persist_gepa_run_state            1614
```

Persisted cursor at the hang: **`rollout_queue = 6218`**, `proposal_index = 0`,
`proposal_queue = 2`, one `parent_minibatch_reference` partial.

Mechanism: `advance_proposer_waiting` has a *prerequisite* return — when the
parent's minibatch scores are missing it sets `active_evaluation` to a
`parent_minibatch_reference` and returns **without advancing `proposal_index`**,
meaning "run this, then call me again". `schedule_async_candidate_minibatches`
pushed that onto `rollout_queue` and returned `Some` unconditionally, so the tick
ended before the rollout scheduler could service it. Next tick re-planned the
identical prerequisite and pushed another copy.

`async_pipelined` is immune because its branch drains rollout **first**. The
flash branch schedules minibatches ahead of rollouts — **in the original code
too**. This bug predates the 2026-08-19 work and is plausibly a second reason the
2026-06-02 flash matrix looked bad.

Fix is two-part, and the second half matters as much as the first:

1. `async_lane_work_already_queued()` — enqueue a partial's work once, checking
   `rollout_queue`, `evaluate_queue`, **and** `lane_leases`. Once dispatched an
   item leaves the queue for the lease; missing that re-opens the same path.
2. `schedule_async_candidate_minibatches` returns `Ok(None)` when nothing
   advanced and nothing was enqueued. Claiming progress on a no-op is what made
   the run loop skip its backoff and spin — that is why this presented as 80% CPU
   rather than an idle stall.

## 3. What landed

`optimizers`:

| File | Change |
|---|---|
| `rust/crates/synth_gepa/src/lane_executor.rs` | **new** — worker-thread pool, each worker with its own `WorkspaceStore` + `RequestCache` handle (rusqlite `Connection` is `Send`, not `Sync`); `LaneOverlapTracker` |
| `rust/crates/synth_gepa/src/lib.rs` | background dispatch/fold path; admit-order fix; duplicate-enqueue fix; overlap wiring; `fold_{completed,failed}_runtime_job_execution` extracted so inline and background share one bookkeeping path |
| `rust/crates/synth_gepa/src/pipeline.rs` | `background_execution` / `background_workers` on the plan |
| `rust/crates/synth_gepa/src/planner.rs` | `GepaLaneOverlapState` on the cursor |
| `rust/crates/synth_optimizer_platform/src/config.rs` | config knobs; `gpt-5.6-{luna,sol,terra}` added to `CHATGPT_PROPOSER_MODELS` |
| `rust/crates/synth_optimizer_platform/src/cache.rs` | WAL + busy timeout; `CacheCounters`; `absorb_worker_activity` so worker cache activity folds back into one boundary |
| `src/synth_optimizers/gepa.py` | passes the new knobs through (its model is `extra="ignore"`, so it would have swallowed them silently) |
| `src/synth_optimizers/victorialogs.py` | **restored** from `d1c6d27` — see §7 |

`synth-cookbooks-public`: `cookbooks/optimizers/gepa/pipeline_matrix/`
(`matrix.toml`, `run_matrix.py`, `report_matrix.py`, `README.md`).

Tests: 16 in `synth_gepa`, 27 in `synth_optimizer_platform`, all green (was
12/22). New coverage: scheduler admission (flash admits gen n+1 during gen n
full-train; pipelined refuses with `GenerationBarrier`), real-threads pool
concurrency, overlap tracker, duplicate-enqueue regression, concurrent
multi-handle cache writes, `gpt-5.6` allowlist incl. fail-closed on unknowns.

`synth_optimizers_py` does not link under `cargo test` — **pre-existing**,
confirmed against a stashed tree.

## 4. Measurements

Banking77 **smoke shape** (24 train / 16 heldout, 2 gens × 2 proposals), proposer
`gpt-5.6-luna` @ low effort, policy `gpt-4.1-nano`. Local engine wheel.

| mode | wall | propose busy | rollout busy | ceiling | max speedup | overlap | overlap(ev) | rollouts | heldout |
|---|---|---|---|---|---|---|---|---|---|
| sync_serial | 166.4 | 145.2 | 11.1 | 11.1 | 1.072x | 0.00 | 0.00 | 168 | 0.500 |
| async_pipelined | 234.9 | 205.3 | 18.1 | 18.1 | 1.084x | 0.00 | 0.00 | 168 | 0.625 |
| **flash_evolve** | 191.9 | 165.3 | 24.8 | 24.8 | 1.149x | **7.74** | **7.44** | 192 | 0.562 |

**Overlap 0.33s → 7.74s.** The two overlap figures are computed by independent
means — in-process accumulator vs. reconstruction from `runtime.job.completed`
intervals — and agree within 4%, so the number is not the instrument reporting on
itself.

`async_pipelined` showing `dispatched=0, overlap=0.00` is correct:
`background_execution` defaults false for it, deliberately, so it stays the same
control arm the 2026-06-02 matrix measured.

**flash is not faster here (0.867x), and that is not the scheduler:**

```
166.4  sync_serial
+20.1  flash's proposer lane was slower (165.3 vs 145.2) — run-to-run variance
+13.7  flash did MORE work: 192 rollouts vs 168
 -7.4  overlap saved
─────
192.8  ≈ 191.9 observed
```

## 5. Overlap is bounded by the smaller lane — know the ceiling first

Perfect propose/rollout overlap hides **at most `min(propose_busy,
rollout_busy)`**. `report_matrix.py --json` reports `overlap_ceiling_seconds` and
`max_theoretical_speedup` per run.

Real Banking77 `sync_serial` run `banking77_gepa_async_t50_mb20_h100_735a9c29`
(2026-08-19, the pre-existing long run):

```
wall            513.1 s
propose busy    434.7 s   (85% — two proposer rounds at ~190s and ~244s)
rollout busy     66.1 s   (13%)
overlap           0.0 s
ceiling          66.1 s → max theoretical speedup 1.148x
```

With `workers.propose = 1` one proposer runs at a time, so on the DAG families
**the proposer lane is the wall clock** and the rollout lane is all FlashEvolve
can hide. A measured 1.05–1.12x on Banking77 is the scheduler working, not
failing.

Corollary the smoke shape makes unavoidable: its ceiling is 11.1s (1.072x) while
observed proposer variance at *identical config* is ±50s (rounds of 75/70,
120/85, 85/80 across three runs). **Noise is ~7x the entire available signal.**
The smoke shape can prove overlap happens; it cannot demonstrate a speedup, and
no amount of re-running it will change that.

## 6. Open — resolve before the real matrix

- **flash ran 192 rollouts vs 168** for the other two modes: exactly one extra
  24-row full-train pass. Unknown whether that is legitimate staleness
  re-evaluation or duplicate work. **A wall-clock comparison between modes doing
  different amounts of work is void regardless of direction**, so this gates the
  matrix, not the other way round.
- Heldout on the smoke swings 0.5 / 0.5625 / 0.625 at fixed config — the
  proposer is nondeterministic and one row of 16 is 0.0625. Quality cannot be
  compared at this size.
- The real 3×3 matrix has **not** been run. hotpotqa and crafter cells have never
  been executed at all.
- `[dataset]` vs `[taskset]` silently defaults instead of erroring — see §7.
  This should be a hard config error.
- Catalog says luna/sol/terra support `xhigh` reasoning effort; the CLI exposes
  only `{none,low,medium,high}`, so `xhigh` is unreachable.

## 7. Infrastructure traps that cost real time

Each of these presented as something other than what it was.

- **`uv pip install --reinstall` serves a cached wheel.** Same version + same
  filename ⇒ cache wins. The rebuilt wheel had the fix, the venv `.so` did not,
  and the run failed with the *old* error message. Use `--no-cache`. Any
  local-wheel iteration loop on this repo hits this.
- **`grep -c` on a binary lies.** It reported 0 for a string that a byte search
  proved present, which sent me chasing a non-existent packaging problem. Use a
  byte search on binaries.
- **I misread a stale log** and reported failures from it after the engine was
  already fixed — the log's mtime was 25 minutes old. Timestamp log files at both
  ends.
- **`victorialogs.py` is missing on this branch** while `cli.py:36` imports it,
  so `synth-optimizers gepa run` is dead on a clean checkout. Restored from
  `d1c6d27` (stdlib-only, self-contained). Matrix runs also set
  `SYNTH_OPTIMIZERS_VL_PROJECT=0` — its network I/O is unmeasured wall clock
  inside a wall-clock comparison.
- **The public cookbooks are on the 0.2.x config schema.** They use `[dataset]` +
  `train_seeds`; this engine takes `[taskset]` with `{split}:{index}` ids plus an
  explicit `[gepa.task_pools]`, and **ignores `[dataset]` entirely**
  (`extra="ignore"`). Unmodified, every cell silently falls back to
  `train_ids = ["train:0"]` and dies on `GepaTaskPools.pareto must not be empty`.
  `run_matrix.py` translates; id format verified against the live container
  (`POST /taskset/tasks {"task_ids":["train:0"]}` →
  `{"task_id":"train:0","split":"train","seed":0}`).
- **The engine pins a `synth-containers` version that is not on PyPI**, so the
  local wheel cannot be `uv run --with`-installed normally. Use a venv with
  `--no-deps` plus the local `containers` checkout.
- **A hung run looks identical to a working one** — process up, container up, no
  error, exit code still pending. Only the event feed going quiet exposed it. The
  monitor now fires a `STALLED` event on 6 minutes of silence, and lane-overlap
  instrumentation earned its keep before it ever reported a speedup.

## 8. Verdict against the handoff's pass bar

| # | Bar | Status |
|---|---|---|
| 1 | Banking77 overlap ≫ 0.33s | **PASS** — 7.74s measured, two independent methods |
| 2 | Banking77 wall ≤ sync_serial at heldout ±0.02 | **Not demonstrated.** Not demonstrable on the smoke shape (ceiling 1.072x vs ±50s proposer noise). Needs the full shape. |
| 3 | `async_pipelined` still barrier-correct | **PASS** — enforced by unit test, and observed `dispatched=0` in a live run |
| 4 | Stale full-train follows policy | Untested live; `guarded` cell not yet run |
| 5 | Heldout gates the reported best | Unchanged; not re-verified |

**FlashEvolve's mechanism is fixed and measured. Whether it is faster is still
an open question** — and on the DAG families it can only ever be worth
`min(propose, rollout)`, which is small when a single serialized proposer round
is 85% of the run.
