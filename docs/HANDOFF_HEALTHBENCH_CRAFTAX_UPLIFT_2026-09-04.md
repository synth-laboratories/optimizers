# Real HealthBench and Craftax RL experiments

## Transport-integrity correction (current)

**Latest status:** clean HealthBench stopped at **36/50 durable updates** on
OpenRouter `402 Payment Required`. Read-only `/api/v1/credits` confirmed
account credits 1119.97 and usage 1120.328489891 (about $0.36 exhausted);
the key has no separate limit. This is account-wide usage, not experiment spend.
The shared experiment ledger is **$69.345383132 counted/reserved**, including
uncertain requests and all old work, under the approved $120 total cap.
All owned benchmark processes stopped. No HealthBench validation/final tasks
have been evaluated; **HealthBench uplift is not yet established**.

Recovery checkpoint: `ckpt_3b2b2e1626446de48f68587b`, revision 36.
Exact training state:
`tinker://d116981c-d8a8-5b88-96ef-b15f81004f58:train:0/weights/optimizers-training_state-save-e950c7fe38e2c5d74220b1d97f6613c2`.
Do not resume sampler weights. Preserve the interrupted `train_40` directory.
Once the authorized OpenRouter source has balance, run:

```sh
DUAL_BENCHMARK_ROOT=/Users/joshuapurtell/GitHub/optimizers/temp/healthbench_craftax_clean_transport_20260904 DUAL_PERSIST_EVIDENCE=1 caffeinate -i uv run python docs/e2e/recover_healthbench_36.py
```

The recovery refuses a changed latest checkpoint or existing recovery output,
checks credit availability without spending, then runs 4 + 10 updates using
exact saved training state, followed by frozen validation and final evaluation.
It does not reset the shared budget or repeat screening. The script passes lint
and checkpoint publication was checked read-only; recovery execution has not
been live-tested while the account is exhausted. The 20 targeted tests still pass.

The first 25 clean updates have complete segment receipts: the second segment
sustained 18.01 graded answers/min and 249.94 aggregate generated tokens/s.
All first 25 provider updates have nonzero loss. Sampled exact-answer auditing
verified 332 traces / 3,928 judge prompts; policy text was unchanged and judge
outputs were nontrainable.

A later audit found that the generic Tinker SDK transport called the Banking77
label normalizer on every sampled completion. This lowercased prose and replaced
spaces/hyphens with underscores before the benchmark consumed it. HealthBench
training was stopped at revision 17; that screening/training is diagnostic only.
Earlier Craftax numbers below describe the old wrapper, not yet a clean-text
transport proof. Do not silently carry them forward as a clean result.

The SDK now preserves parsed completion text; task-specific label normalization
stays in task evaluators. Prose and JSON-whitespace regression tests pass, along
with 20 targeted transport/budget/persistence tests. Evaluation now persists
each observed row, reward and full trace immediately, and refuses blind reruns
even after an incomplete panel.

Clean artifacts are isolated at
`/Users/joshuapurtell/GitHub/optimizers/temp/healthbench_craftax_clean_transport_20260904`.
The budget ledger remains at the original root, so this does not reset spend.
Craftax revision 50 and its original baseline are being compared on new seeds
99001–99064, frozen before that corrected evaluation. HealthBench's validation
and final tasks remain unobserved; its clean restart will use fresh screening
and fresh base-model training, not the affected 17-update state.

The user explicitly approved **$120 total**. The shared ledger enforces $119
in token reservations plus $1 overhead, including all affected prior work.
Expected combined cost is $100–115. The clean HealthBench restart is active;
20 targeted tests and lint passed before launch.

The corrected-transport Craftax recheck completed on all 64 fresh seeds:
baseline **0.203125**, trained **1.181250**, paired delta **+0.978125**,
95% paired bootstrap interval **[0.768750, 1.1859375]**; 56 wins, five losses,
three ties. The fixed revision-50 checkpoint was not reselected. Evaluation
took 232.040 seconds for 128 episodes / 1,011 policy calls (33.10 episodes/min,
261.42 calls/min). Receipt SHA-256:
`13b7c680347d847cb8eb20680193c49ef445ecb32aa3c37dd48dcadbe3245c9d`.
Evidence is in the clean root's `craftax/transport_recheck_final/` directory.
This is GameBench Rust Craftax, not the official JAX benchmark.
All 128 persisted sealed traces match their receipt/reward digest fields and
the frozen seeds. Baseline had one illegal-action termination; trained had
none. Sapling collection occurred in 7 baseline versus 63 trained worlds;
mean environment ticks were 12.859375 versus 62.03125. The gain remains largely
basic collection and better use of the action budget, not broad game mastery.

Clean HealthBench screening completed 256 answers in 198.596 seconds
(77.34 answers/min with 24 slots), selecting 28/32 tasks by nonzero reward
range. Full traces retain natural spaces/capitalization. Training starts
from a separate fresh base checkpoint, not the screening or old training state.

The sections below retain the historical sequence and old-wrapper results;
their older budget/status statements are superseded by this section.

## Status and scope

User request: demonstrate high-throughput pipelined real-training uplift on
both benchmarks. **Craftax has positive heldout return uplift after 50 real
updates. HealthBench's real-grader pilot works, and the user has now approved
the full run under a $100 combined cap.** This is not
completion of both benchmarks. All owned Craftax processes are stopped.

### Current HealthBench authorization

The user answered “yeop” to the explicit $100 combined-cap request. The guard
now reserves at most $99 of token charges with $1 held for overhead, preserving
all prior spend. Expected combined cost remains $80–95. This supersedes the
earlier pause described below; the frozen panels are not rewritten.
`budget_authorization_100.json` records the approval under the artifact root.
`run_healthbench_authorized.py` continues the remaining 28 tasks × eight using
24 episode slots and the shared 32-worker rubric pool, then invokes the frozen
50-update / validation / final procedure. It refuses existing screening output
and checks disk headroom before launch. Four budget/credential tests pass.

Protocol audit: the image uses the public HealthBench data and the same
per-example achieved-points / positive-possible-points formula, but its
`ProviderRubricJudge` prompt is shorter than the full
[reference grader template](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py).
It lacks the reference prompt's detailed multi-clause/example guidance.
The prompt remains fixed across screening, training, validation, and final
evaluation; no mid-run scorer change is made. Report this as a **fixed-judge
HealthBench research-panel comparison**, not an official HealthBench score.
The image's legacy `canonical_healthbench_grader` flag identifies its grader
model configuration, not exact prompt conformance; it must not be used to
claim full protocol equivalence. An official-template replication remains
separate work and is not silently substituted into this run.

## HealthBench credential and pilot follow-up

The user explicitly authorized checking/using `evals/.env`. Its OpenRouter key
returned HTTP 200 and was then used for actual rubric grading. No credential
value was printed, copied into the repository, or stored in receipts. Tinker
continues to use the previously authorized frontend credential source; no
Keychain access occurred.

The real pilot completed 32 answers (four frozen training tasks × eight) in
119.170 seconds: **16.11 answers/minute** with 12 episode slots. All four tasks
had nonzero reward range. The pilot added **$2.084076** to the conservative
ledger, now **$21.923190922 combined**. The 548 completed real rubric calls
averaged $0.00377517 per criterion; policy sampling was about $0.0153.
This is working real grading and screening, not HealthBench training or uplift.

Frozen partitions contain 374 training, 301 validation, and 1,476 final rubric
criteria. At the observed mean criterion price, the full original design
(8x screen, minimum 600 training answers, three paired validation comparisons,
and final paired evaluation) projects **$55.73 in grading alone**, including
the pilot. With Craftax and allowance for skipped groups / answer-length
variation, expected combined spend is approximately **$80–95**. This exceeds
the original $49 cap. All owned HealthBench processes are stopped; no further
paid work is authorized by this estimate. A **$100 combined cap** is proposed,
not applied. The existing guard remains at $48 token charges plus $1 overhead.

The next server version uses one shared 32-worker rubric pool. Each criterion
still gets the same separate model call/prompt and verdicts are recorded in
original order. The pilot above used the older serial-within-answer grader;
do not attribute its throughput to the new pool. Existing 25 image contract
tests pass both normally and with the batch hook exercised; four budget and
credential-routing tests pass. No evals test files were added or changed.

Pilot evidence: `healthbench/pilot/{manifest,attempts,summary}.json` under the
artifact root. Attempt digest:
`498225d91a5759b53e1e962526e29ab3a11765c3e9dbb2d868e0cf0f38df21fa`.
Once a larger budget is authorized, screen the remaining 28 training tasks,
then run the frozen training/validation/final design. Do not repeat the pilot
or alter the final panel.

## Completed Craftax result

On the untouched 64-world final panel, baseline mean environment return was
**0.256250** and revision 50 mean was **1.221875**: paired gain **+0.965625**,
with a task-level bootstrap 95% interval **[0.790625, 1.1421875]**.
There were 54 wins, two losses, and eight ties. Both arms used the same 64
world seeds, temperature 0, 384-token completion cap, eight policy calls and
64 environment ticks maximum. All 128 sealed traces match their receipt hashes;
all 1,023 captured calls carry the intended temperature/cap and aligned
nonempty token/logprob arrays. Neither arm had an illegal-action termination.

Validation selected revision 50 by the frozen highest-trained-mean rule:

| Revision | Validation trained mean | Repeated baseline mean |
| --- | ---: | ---: |
| 10 | 1.462500 | 0.537500 |
| 25 | 0.887500 | 0.475000 |
| 50 | 1.481250 | 0.600000 |

The winning validation margin over revision 10 was small. Repeated baseline
arms varied despite recorded temperature 0; do not claim bitwise deterministic
execution or that revision 50 is conclusively the optimal checkpoint.
No final-panel result was used for selection or further training.

### What improved—and what did not

Mean achievement count increased from 0.265625 to 1.281250. The gain is narrow:
`collect_sapling` appeared on 3 baseline worlds versus 63 trained worlds, while
`collect_wood` decreased from 10 to four. Mean executed environment ticks rose
from 13.109375 to 63.656250. The trained policy makes fuller use of the allowed
action/tick budget and repeatedly collects saplings. This is real environment
return uplift, not broad Craftax mastery, an official Craftax leaderboard
result, or proof of longer-horizon generalization. This run uses the local
GameBench Rust implementation and custom short horizons.

### Training, throughput, and cost

- 50 durable training revisions, 4,555 trainable policy-call examples, and
  543,556 reported training tokens in their catalog evidence. The provider
  executed 51 train calls: one update was abandoned after disk-full publication
  failure and is not part of the final 50-update lineage.
- The 38 updates in fully closed segments have preserved nonzero-loss metrics.
  The interrupted segment's 12 durable updates retain checkpoint/training-token
  evidence, but not the same complete end-of-segment metric export.
- Screening: 256 episodes in 228.40 seconds, **67.25 episodes/minute**.
- Training: 716 sampled episodes across about **59.02 minutes** of measured
  active windows, **12.13 episodes/minute**. This sums completed segments'
  submit-to-score windows plus the interrupted journal window; it excludes the
  user/disk-recovery pause and is not total wall-clock elapsed time.
- Final evaluation: 128 episodes / 1,023 model calls in **144.41 seconds**,
  **53.18 episodes/minute**, **425.03 policy calls/minute**.
- Combined ledger: **$19.839114922 counted/reserved**, including diagnostics,
  interrupted calls, and the rejected HealthBench grading attempt. Uncertain
  reservations remain charged to the guard; these are conservative estimates,
  not reconciled provider invoices. The original aggregate ceiling remains $49.

### Durable identities and evidence

Baseline checkpoint: `ckpt_087dbe3d9cb9c46fc080b573`.
Selected checkpoint: `ckpt_4bfc308872dc44019328ac96` (revision 50).

Selected sampler:
`tinker://d77430ff-b8de-5ace-ad44-6c444058f25f:train:0/sampler_weights/optimizers-sampler_weights-save-27a4cc6ee1ae7f5a2477e0f41848d0fa`.

Selected resumable training state:
`tinker://d77430ff-b8de-5ace-ad44-6c444058f25f:train:0/weights/optimizers-training_state-save-0dcd921a0f8ab22e07605c5e5b004478`.
Use training state for further optimization, never sampler weights.
The SDK's artifact digests hash provider reference strings, not downloaded
weight bytes; do not describe them as independently verified weight checksums.

Final receipt under the artifact root:
`craftax/final/dual_craftax_final_20260904.evaluation.json`.
SHA-256: `da36ad571b44d41d7a63ced21c2ebef59be1495684ea74ade278934ab6be2ab5`.
All 128 full traces are in `craftax/final/traces/`; engine-authored achievement
and termination details are in `craftax/final/reward_details.json`.
The committed compact result is [CRAFTAX_HELDOUT_2026-09-04.json](CRAFTAX_HELDOUT_2026-09-04.json).

### Earlier HealthBench blocker (resolved by the follow-up above)

The real dataset, disjoint frozen panels, rubric-backed runtime, screening,
training, evaluation, and shared budget guard are implemented. The authorized
frontend OpenRouter key returned 401. Permission to inspect/use
`/Users/joshuapurtell/GitHub/evals/.env` has been requested but not received.
No HealthBench uplift is established. Once a working authorized credential is
available, measure a bounded real-grader pilot and re-estimate all remaining
judge costs before launching the full 50-update experiment. Do not exceed the
original aggregate cap without fresh authorization or reuse observed final
panels to tune models.

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
9. The transport capped completion tokens after the runtime had captured its
   wire request. Actual Craftax sampling used 384 tokens, but early traces named
   the image default; HealthBench had an analogous temperature/cap mismatch.
   Image constructors now accept these settings and put them in the original
   request. This preserves actual sampling behavior. The running Craftax
   segments through revision 25 retain the old metadata; subsequent fresh
   servers use aligned metadata. Do not interpret the early request cap as the
   actual provider cap.

## Disk-full interruption and recovery

Local disk filled during the full-suite rerun (1088 tests passed; two failures and
13 setup errors included `ENOSPC`). Training stopped during publication of the
next update. Revision 22 is the last published checkpoint:
`ckpt_965fb2c7899ae2148734b8a9`. Its catalog passes `PRAGMA integrity_check`.
The unpublished update is not counted as durable progress. Original logs and
receipts remain untouched.

The user authorized removal of the disposable pytest-371 directory. On checking,
it was already absent and 76 GiB was free; the agent deleted nothing.
`recover_craftax_22.py` resumes revision 22's exact training state and runs
3 + 15 + 10 updates in distinct recovery directories before the original
validation/final procedure. It checks for at least 10 GiB free before segments
and refuses to overwrite prior recovery evidence. Metadata alignment therefore
starts at recovered revision 23, not revision 26. Six budget/async tests pass
using an explicit GitHub-local temporary directory. Do not repeat the full
suite with its default external temporary directory.
The budget guard additionally refuses new paid calls below 2 GiB free, before
reservation/provider execution. Three budget tests pass, including this refusal.

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

The first segment completed 10 real updates, all with nonzero loss, 953
trainable policy-call examples and 71,575 reported training tokens. It sampled
132 episodes in 571.63 seconds (13.86 episodes/minute including update gaps),
training 30 groups and skipping three zero-advantage groups. This throughput
is distinct from the faster screening throughput above. Exact-state resume
into the next 15-update segment is running; heldout outcomes remain unobserved.

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
