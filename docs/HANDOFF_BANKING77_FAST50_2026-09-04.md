# Banking77 fast 50-update experiment

## Frozen design

This experiment resumes revision 24 (`ckpt_d02739fcc0546c017c3cfb94`) for
50 additional effective optimizer updates, targeting revision 74. It is a new
experiment; no uplift conclusion is available until its final evaluation ends.

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

The final panel is now reserved and must not be consumed by other experiments.
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
under the durable root. Each phase gets a fresh owned server on port 8254.

Screening servers use ports 8250–8253. No macOS Keychain access is used:
credentials come from the previously authorized frontend `.env.local` through
the existing paid adapter. Do not print that file or source unrelated secrets.

If training fails after registering checkpoints, the driver refuses to start
it again blindly. Inspect the durable catalog and receipts, then explicitly
configure an exact-state resume for the remaining update count. Do not restart
the whole 50-update run or expand the budget automatically.

This document records the predeclared design. Measured throughput, training
completion, cost estimates, and heldout results will be appended after execution.

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
