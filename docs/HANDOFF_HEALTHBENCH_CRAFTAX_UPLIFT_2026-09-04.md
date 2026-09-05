# Real HealthBench and Craftax RL experiments

## Status and scope

User request: demonstrate high-throughput pipelined real-training uplift on
both benchmarks. **No uplift result is available yet.** Training and final
evaluation remain to be completed. The intended endpoint is 50 real updates
per benchmark and paired evaluation on separately frozen unused tasks.

All new artifacts are under
`/Users/joshuapurtell/GitHub/optimizers/temp/healthbench_craftax_uplift_20260904`.
No new files are written under Documents. Preserve all interrupted attempts;
they count toward spend but do not supply final-evaluation evidence.

The aggregate authorized ceiling is $49, initially estimated at $20–$40.
`dual_benchmark_budget.py` reserves at most $48 of token charges across all
processes, leaving $1 for overhead. Successful calls settle to token-based
estimates; uncertain calls retain their reservation. These are not invoices.
The direct frontend `.env.local` Tinker credential works. Its OpenRouter key
returned HTTP 401 both on rubric grading and a read-only key check. HealthBench
is paused pending explicit authorization to inspect/use evals `.env` or a
different working authorized credential. No Keychain access is permitted.

## Frozen design

`panels.json` is the pre-sampling source of task identities and selection rules.

- HealthBench: 32 training candidates, 32 validation conversations, 128 final
  conversations. Hash-selected disjoint partitions of the 5,000-row release.
  Dataset SHA-256:
  `e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f`.
  Fixed grader: GPT-4.1 snapshot 2025-04-14 via OpenRouter, one rubric call per
  physician criterion. Grader tokens must remain non-trainable.
- Craftax: 32 training world seeds 96001–96032; 16 validation seeds
  97001–97016; 64 final seeds 98001–98064. Real GameBench Rust engine, not a
  fixture world or JAX substitute. Eight policy calls / 64 environment ticks
  maximum per episode. Engine binary SHA-256:
  `656d35321ae2a9a0ca3239df7ace412963bb71300ef54e32973cb623806e28e2`.
- Screen every training candidate eight times at temperature 1; retain
  nonzero within-task reward range. Do not apply binary 1–7/8 to graded scores.
- Start fresh from `openai/gpt-oss-20b`, not the Banking77-trained checkpoint.
  Learning rate is explicitly 0.00005, chosen before training.
  Each update packs three four-sample groups. Fifty updates are segmented as
  10 + 15 + 15 + 10, resuming exact training-state artifacts between segments.
  Never load sampler weights as training weights.
- Choose among revisions 10, 25, and 50 by validation mean; ties to earliest.
  All validation occurs after training. Then score the selected checkpoint
  versus its own initial training baseline on the untouched final panel.
- Paired bootstrap uses tasks/worlds as the unit, 20,000 replicates, seed
  20260904. Rubric items are not independent evaluation examples.
- These are custom research partitions/horizons, not official leaderboard
  scores. HealthBench score uplift would not establish clinical readiness.

## Audit findings and fixes

The old paid smoke adapters were not benchmark proofs: HealthBench used a
deterministic lexical test judge, and Craftax drove a Rust-wire fixture world.
The new `serve_dual_benchmark.py` uses the real judge and Rust binary.

1. Both image runtimes executed synchronously during submission. A bounded
   asynchronous episode wrapper now permits concurrent dispatch, propagates
   errors, and waits for completion at quiescence.
2. Startup probes treated unchanged event snapshots as duplicate events.
   Snapshot de-duplication now preserves strict rejection of backwards cursors.
   Probes also select configured training IDs rather than any row in a physical
   split that might contain custom validation/final partitions.
3. Fresh Tinker checkpoint resolution needs a provider-observed digest source;
   the budgeted plane now records artifacts returned by checkpoint publication
   and supplies that map to the resolver.
4. Craftax never released completed Rust sessions, exhausting the 128-session
   engine after a bounded pilot plus screen. A `finally` cleanup deletes only
   each episode's own Rust rollout after evidence collection, including failure.
5. An illegal first action broke before its sampled call was recorded, causing
   `sealed no model call`. The image now records the actual completion with a
   zero-width environment-tick interval and ends the episode with the engine's
   existing reward. No action or reward is invented. Later illegal actions are
   retained too, instead of silently disappearing from training evidence.
6. Both images omitted declared task seeds. Task lookup then used row positions
   in request metadata even though worlds resolved their true task seeds.
   Both images now advertise the actual seed. All pre-fix screens are diagnostic
   only; the fresh full Craftax screen supplies training selection.
7. Screening now writes incremental attempt receipts as well as progress, so
   interruption no longer destroys every completed outcome.
8. Opt-in `bounded_on_policy_batch` counts open, complete, queued, and pending
   mixed groups before admission. This avoids speculative old-policy groups
   becoming stale as soon as the packed update is published, while preserving
   parallel execution within the upcoming batch.

## Observed pilot and interrupted work

Craftax's first real pilot completed 32 episodes / 253 model calls in 73.93
seconds: 25.97 episodes/minute and 205.33 model calls/minute at 12 episode
slots. Three of four worlds had reward variation. Counted sampling was about
$0.24. This proves real execution and useful variance, not uplift.

`craftax/screen_remaining_session_leak/` preserves progress from the engine
capacity failure. `craftax/screen_remaining/` preserves 73 completed incremental
attempts before the illegal-action failure. Neither contributes selections.
The replacement `craftax/screen_full/` samples all 32 frozen training seeds,
eight times each, at 24 episode slots after both fixes.

The clean full screen completed **256 episodes / 1,987 policy calls in 228.40
seconds**: **67.25 episodes/minute and 521.97 policy calls/minute**. Sixteen
worlds had nonzero reward range and were admitted. All 256 receipt seeds match
the actual world IDs. Live Rust sessions stayed at 23–24 after more than 168
completed episodes, proving the old 128-session accumulation failure was fixed.
The driver has started its first real 10-update training segment from a fresh
base-model checkpoint, `ckpt_087dbe3d9cb9c46fc080b573`.

Implementation commits: optimizers `dddcbd6`; evals `a2253fc95`. The evals fixes
are limited to actual seed declaration and preserving illegal-action evidence.

## Entry points and verification

- `prepare_dual_benchmark.py`: freezes data and configuration; refuses to
  overwrite existing panels.
- `serve_dual_benchmark.py`: real services and budgeted rubric judge.
- `pilot_dual_benchmark.py`: baseline publication and small real 8x pilot.
- `run_dual_benchmark.py <benchmark> all`: training segments, validation,
  deterministic checkpoint selection, then final evaluation.
- `evaluate_dual_benchmark.py`: bounded paired scoring, raw trace/reward
  sidecars, and task-level uncertainty estimates.

Before launching the driver, stop the screen's owned service so the driver can
start its own fresh service on 8260 (HealthBench) or 8261 (Craftax). Rust children
use the façade port plus 100. Each driver phase owns and stops its process group.
Validation uses three independent servers and eight episodes per comparison;
final evaluation uses 24 episodes. No provider retry is treated as free.

Verification so far: full optimizers suite 1,101 passed before the two added
world-cleanup cases; all four asynchronous-runtime/cleanup cases passed.
Existing Craftax CISPO image tests: 34 passed. Existing HealthBench CISPO image
tests: 25 passed. No evals test files were added or edited.

The evals checkout contains substantial unrelated user changes. Only the two
image `cispo.py` files were edited here. Keep all unrelated changes intact.
