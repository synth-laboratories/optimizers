# Banking77 fast 50-update experiment

## Frozen design

This experiment resumed revision 24 (`ckpt_d02739fcc0546c017c3cfb94`) for
50 additional effective optimizer updates and reached revision 74. Its frozen
design and execution evidence are recorded below.

**Outcome: positive heldout uplift.** On the untouched 770-example final panel,
revision 74 improved over revision 24 by **+3.12 percentage points** (95% paired
bootstrap interval **+1.30 to +4.94 points**, exact McNemar **p=0.00150**).
This establishes uplift for this frozen comparison, not robustness across
independent training seeds or other datasets.

The durable root is
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_fast50_19`.
`experiment.json`, `candidates.json`, `validation_panel.json`, and
`final_panel.json` freeze the selection rules and data before sampling.

- Training candidates: 20 per intent, 1,540 total, selected by deterministic
  hashes from the 10,003-row training split. Four independent shards sample
  each candidate eight times at temperature 1, with eight slots per shard.
- Admission: retain exactly 1–7 successes out of eight. All candidate outcomes
  must be present; partial screens do not select training tasks.
- Curriculum: deterministic round-robin interleaving of eligible tasks by
  intent. Intents with no admitted examples cannot contribute, and exhausted
  intent lists drop out; this is not an assertion of perfectly equal sampling.
- Each optimizer update combines four eight-rollout groups (32 examples).
  Only effective updates count. Maximum 800 sampled groups bounds attempts
  spent seeking nonzero training signal. Policy lag remains zero.
- Validation: two fresh examples per intent, 154 total. Evaluate revisions
  34, 44, 54, 64, and 74; choose highest validation accuracy, ties to the
  earliest checkpoint. Validate only after training, avoiding training changes
  based on these scores.
- Final: ten fresh examples per intent, 770 total, disjoint from validation and
  all 955 previously recorded heldout IDs. The primary comparison is selected
  checkpoint versus revision 24; comparison with the original baseline is
  secondary context. Report both estimates and intervals honestly.
- Aggregate provider cap: $15; initial expected cost $2–$6. Sampling caps at
  256 output tokens per call. The estimated token charge is not an invoice.

The final panel is reserved for this experiment and must not be reused as an
untouched panel by future experiments.
The two final comparisons each use a complete paired evaluation; the selected
checkpoint is sampled separately in each comparison.

## Throughput fixes

1. The screener now fills available slots across task boundaries, avoiding a
   barrier after each task's eighth rollout.
2. Paired evaluation accepts `--concurrency`, capped by the container's
   admitted concurrency. Submission, polling, and collection are multiplexed
   on one owning thread; results are reordered to the frozen seed order.
   In-flight attempts and sampler routes are cleaned up if evaluation fails.
3. Tinker sampling clients are reused by immutable checkpoint reference and
   digest. Previously every checkpoint sample synchronously created a client.
4. Screening and training now settle provisional rollout IDs after completion.
   Immediate settlement acquired the same route lock held by a live sampling
   call, so dispatch could block despite available concurrency slots.

The initial screening launch exposed the last two issues. It was interrupted
before any curriculum selection; partial progress/usage files remain in
`screen_0_interrupted/` through `screen_3_interrupted/`. The replacement uses
fresh `_r2` run IDs and fresh servers. Those partial runs add cost but supply
no selection evidence. Complete new 8x outcomes are required for every task.

## Execution and recovery

`docs/e2e/prepare_banking77_fast50.py` prepares the panels and configs, then
validates complete shard outcomes and generates the curriculum. Its `prepare`
phase refuses to overwrite an already frozen experiment.

`docs/e2e/run_banking77_fast50.py all` waits for all screening manifests, runs
training, checks that all 50 new revisions exist and the target stop reason
was reached, evaluates the five validation checkpoints, and performs the two
final comparisons. It writes status, phase logs, selection, and final results
under the durable root. Training gets a fresh owned server on port 8254.
Validation comparisons use distinct ports 8254–8258, with at most four
comparisons active (32 sampling slots total). The two final comparisons run
concurrently on ports 8254–8255. Completion order cannot alter the selection
rule: ties still select the earliest revision.

Screening servers use ports 8250–8253. No macOS Keychain access is used:
credentials come from the previously authorized frontend `.env.local` through
the existing paid adapter. Do not print that file or source unrelated secrets.

If training fails after registering checkpoints, the driver refuses to start
it again blindly. Inspect the durable catalog and receipts, then explicitly
configure an exact-state resume for the remaining update count. Do not restart
the whole 50-update run or expand the budget automatically.

The sections below distinguish the predeclared design from observed execution.

## Live execution findings

Screening completed: 12,320 valid samples in 1,142.71 seconds (19.05 minutes),
averaging 646.88 samples/minute across four shards. Of 1,540 candidates,
1,165 were 8/8, 195 were 0/8, and 180 across 57 intents passed the mixed-outcome
rule. Counted uncached sampling cost is approximately $1.739; actual caching
and invoiced dollars remain unknown.

The first training attempt published revision 25, then refused a queued
revision-24 group trying to bind revision-25 weights. Admission now snapshots
each group's immutable policy revisions for later dispatch. The completed
update is preserved at `ckpt_0278ebdd569252e2f583b9a0`.

An explicit recovery (`run_banking77_fast50.py resume25`) resumes that exact
training state for the remaining 49 updates, under run ID
`b77_fast50_19_resume25`. Ten groups were admitted in the interrupted run;
the remaining cap is 790, and the frozen task order advances by ten slots.
The resumed run uses one open eight-rollout group, filling all eight execution
slots, and still accumulates four completed groups per optimizer update. This
avoids wasting work on stale prefetch at policy lag zero. The validation and
final-selection rules are unchanged. Recovery records and the original SQLite
queue journal remain under the durable experiment root.

The host entered clamshell sleep at 17:11:19 EDT and fully woke at 17:50:57
EDT, a 39m38s wall-clock interruption. Training recovered after wake. An
idle-sleep assertion was attached to the run; it does not override lid closure.
Wall-clock training duration must disclose this interruption rather than be
presented as uninterrupted compute time.

## Completed training evidence

The resumed run stopped with `target_train_updates_reached` at revision 74,
`ckpt_4ce5ae6396994e9f42c890cf`. The catalog contains every new revision 25–74.
The first interrupted run contributed one real update, 32 examples, and 3,519
training tokens; its final metrics receipt was not emitted on failure. The
resumed receipt records 49 provider train calls, 1,568 examples, and 174,599
training tokens. Each of those 49 calls reports nonzero loss and 32 nonzero
loss weights. Total: **50 additional updates, 1,600 examples, 178,118 training
tokens**, not merely rollout collection or checkpoint copying.

The resumed run sampled 4,272 rollouts in 534 groups. Of these, 196 groups
trained and 338 had zero advantage and were skipped. The fixed up-front filter
does not guarantee that a task remains mixed under later policy revisions.
Sampling makespan was 5,108.51 seconds (85.14 minutes), averaging **50.18
rollouts/minute** and 73.12 generated tokens/second, including update gaps but
excluding host sleep. Catalog baseline-to-final wall time was 124.18 minutes
(21:06:32–23:10:43 UTC), including the 39m38s sleep interruption. These timing
figures cover the resumed 49-update phase, not the earlier failed phase.

Revision 74 immutable artifacts:

- Sampler: `tinker://6a493a6e-2ed2-5ad4-bdab-3645cc5869e8:train:0/sampler_weights/optimizers-sampler_weights-save-572b6a0b00adc3265fda31b04e4264ea`
  — `sha256:1de7f78fe80a5323e9a2128f3ec295182324576a39d5a43ed2d8bb17d835a62c`.
- Training state: `tinker://6a493a6e-2ed2-5ad4-bdab-3645cc5869e8:train:0/weights/optimizers-training_state-save-fdcfef6f39ecb6b58120ff0edfd2fb39`
  — `sha256:a4ede8165878c6d19d1bd675f4993b0d4e401546d1798e4a9004d9aa753d2ed4`.

The old sequential supervisor was stopped only after training had exited and
the completed manifest was verified. Its expected KeyboardInterrupt is a
supervisor handoff, not a training failure. The independent `heldout` launcher
then ran the frozen comparisons with parallel evaluation workers.

## Validation and final results

The frozen validation selection was completed before either final comparison.

| Revision | Correct / 154 | Accuracy |
| --- | --- | --- |
| 34 | 134 | 87.01% |
| 44 | 132 | 85.71% |
| 54 | 133 | 86.36% |
| 64 | 132 | 85.71% |
| **74, selected** | **135** | **87.66%** |

| Final comparison | Baseline | Revision 74 | Delta | 95% paired interval | W / L / T | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| **Primary: revision 24** | 653/770, 84.81% | 677/770, 87.92% | **+3.12 pp** | **+1.30 to +4.94 pp** | 39 / 15 / 716 | 0.001496 |
| Secondary: original model | 632/770, 82.08% | 679/770, 88.18% | +6.10 pp | +4.03 to +8.18 pp | 57 / 10 / 703 | 4.04e-9 |

Intervals use 20,000 paired bootstrap replicates with seed 20260904. This is a
balanced ten-example-per-intent panel. Each comparison sampled the trained arm
independently; the two trained counts differ by two despite temperature zero.
They are not pooled or substituted for one another. The final panel has now
been observed and is not available for future untouched confirmation.

Final panel identity:
`sha256:a01195e6cd6e271dfd531460055bdcbc201381d74fe34b9036f5b6826db6304a`.
Primary raw receipt SHA-256:
`365e504f03d195ac8651292599c281f156289ac9d1931ca0093f0fc7a05ae324`.
Secondary raw receipt SHA-256:
`397b187bfcdd6867c7cc929330bf33400e272d3532599855f78ff6732e443a40`.

All five validation comparisons plus both final comparisons performed 4,620
rollouts in 709.65 seconds from the first evaluation start to the last finish:
**390.62 rollouts/minute**, including the selection boundary and worker startup
between phases. The driver capped validation at four parallel comparisons and
final evaluation at two.

## Cost, verification, and handoff

Recorded token usage—including completed screening, observed partial screening,
resumed training sampling, all evaluations, and all 178,118 training tokens—
estimates **$1.19–$3.17** at the recorded Tinker rates. The endpoints assume all
prefill cached versus none cached. This is **not an invoice or complete spend**:
the interrupted first training run's sampling, unrecorded in-flight/failed
calls, and checkpoint storage are not included. Provider dollar amounts and
cache-hit accounting were unavailable. The $15 experiment cap is unchanged.

Rates for `openai/gpt-oss-20b`: $0.18/M prefill, $0.036/M cached prefill,
$0.45/M sampled tokens, and $0.396/M training tokens, from
[Tinker's model rate data](https://tinker-docs.thinkingmachines.ai/tinker/models.json).
Historical checkpoint `training_evidence.provider_cost: 0.0` is a missing-cost
placeholder, **not free compute**. The completed `provider_usage.json` correctly
reports `provider_cost: null` and `cost_missing: true`; use token counts for the
estimate until billing reconciliation is available.

- Machine-readable report: `docs/receipts/banking77-fast50-20260904.json`.
- Durable raw evidence: `b77_fast50_19/` under the root documented above,
  including `final_results.json`, `selection.json`, both full final receipts,
  screening attempts, and resumed training receipts.
- Reverify locally without provider calls:
  `uv run python docs/e2e/summarize_banking77_fast50.py`.
  This checks complete mixed-outcome admission, panel separation, 50 revisions,
  provider training evidence, deterministic selection, and result receipt hashes.
- Full regression suite: **1,093 passed**; two subsequently added cost-summary
  tests also passed. Ruff and `git diff --check` passed.
- All owned training, evaluation, server, and idle-sleep assertion processes
  stopped. No listeners remain on ports 8250–8258.

No more training is required to finish this experiment. For a subsequent run,
the main efficiency opportunity is refreshing the training-only mixed-outcome
pool as the policy changes: 338/534 resumed groups were skipped despite passing
the original filter. Any further confirmation needs a newly frozen unused
panel; do not optimize against this final panel.
