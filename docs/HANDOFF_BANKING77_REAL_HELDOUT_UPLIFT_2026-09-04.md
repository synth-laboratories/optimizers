# Banking77 real-training heldout-uplift engineering handoff

Date: 2026-09-04

## Result

The experiment is complete. Sixteen additional effective Tinker optimizer
updates, in two curriculum stages resumed from run 15, produced a positive
result on the sealed final 77-intent Banking77 panel:

- baseline: 64/77 = 0.8311688312;
- trained: 67/77 = 0.8701298701;
- paired uplift: **+3/77 = +3.8961 percentage points**;
- wins/losses/ties: 3/0/74;
- paired standard deviation: 0.1947710155;
- 20,000-replicate paired-bootstrap 95% percentile interval: [0, 7/77]
  = [0, 9.0909 percentage points];
- exact two-sided McNemar p = 0.25.

This is valid positive heldout evidence, but it is not conventionally
statistically significant. The interval touches zero and the exact test has
only three discordant pairs. Do not describe it as a definitive population
uplift.

The authoritative result is
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_real_uplift_18/evaluation_final/final_result.json`.
Its internal evaluation-receipt digest is
`sha256:ece1af538fc0917657ff0bfdf2e64ef756d7aad74444e650162cb2ae7a4cb3e7`;
the JSON summary file itself hashes to
`sha256:ace86a8edb041427fb85e85f56d8f7d01474146fb0f976c040780553b8921fb4`.

## Honest experiment sequence

| Checkpoint evaluated | Panel | Baseline | Trained | Delta | W/L/T | 95% paired bootstrap CI | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| run 15, `ckpt_acfd561eda65c74c9fa351ab` | validation | 67/77 | 65/77 | -2/77 | 0/2/75 | [-5/77, 0] | 0.5 |
| Stage 1, `ckpt_3b5660b22f8a8de4b82cadea` | validation | 67/77 | 66/77 | -1/77 | 1/2/74 | [-4/77, 2/77] | 1.0 |
| Stage 2, `ckpt_d02739fcc0546c017c3cfb94` | validation | 67/77 | 68/77 | +1/77 | 1/0/76 | [0, 3/77] | 1.0 |
| Stage 2, `ckpt_d02739fcc0546c017c3cfb94` | sealed final | 64/77 | 67/77 | **+3/77** | 3/0/74 | [0, 7/77] | 0.25 |

The first three rows used the same pre-frozen validation panel, digest
`sha256:0a9a0474272801300f62d09a6edd742df9423450754296fbccd6da6270e2725c`.
The final result used a separately frozen, untouched panel, digest
`sha256:5f1c672211e70f7c2686cf7951a262c53dea834a6837cbccc013901c78e5422f`.
The panel files are `validation_panel.json` and `final_test_panel.json` under
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_real_uplift_18/`;
their file SHA-256 values are respectively
`379416bcb54be6203ffcb43fe2a6d71b51c491bd0813b0e723a1adbf9678d63e`
and `294fc12f7753cd68ae27b5e38b3e6b9e471e92093656d30768eb96a37574c47c`.

## Training and screening

Both screens applied the predeclared rule exactly: sample every candidate
eight times and select only tasks with 1 through 7 successes. A 0/8 or 8/8
task was excluded. Each screen covered 56 candidates and 448 rollouts at
maximum concurrency 8.

- Stage 1 screened run 15's revision-8 checkpoint and selected 9/56 tasks:
  `8047, 2869, 8167, 8048, 2079, 8049, 1819, 3, 2872`.
- Stage 2 re-screened Stage 1's revision-16 checkpoint and selected 6/56:
  `8048, 2870, 2079, 1819, 9518, 1931`.

The screening manifests and raw attempts are in `screen_stage1/` and
`screen_stage2/` under the durable experiment root. The manifest attempt and
summary digests bind each selection to its full 8x outcomes.

Stage 1 resumed run 15 at revision 8 and completed 8 effective updates,
128 examples, and 14,447 training tokens, ending at revision 16. Stage 2
restored Stage 1's exact training state, completed another 8 effective updates,
128 examples, and 14,160 training tokens, ending at revision 24. Thus the
curriculum added 16 real optimizer updates, 256 examples, and 28,607 training
tokens. Every one of the 16 provider calls has a finite, nonzero `loss:sum`;
Stage 2 additionally records 16 nonzero loss weights per update. Both manifests
stop with `target_train_updates_reached` rather than counting zero-variance
groups as updates.

Across run 15 and the two resumed stages, sampling produced 86,338 tokens over
944 rollouts. The transport recorded `sampling_seconds: 0.0` and
`weighted_aggregate_tps: null` for all three runs, so throughput is unavailable;
null TPS must not be reported as zero throughput.

Durable training receipts:

- Stage 1: `/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_real_uplift_18/training_stage1/receipts_retry2`
- Stage 2: `/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_real_uplift_18/training_stage2/receipts`
- shared durable checkpoint catalog: `/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_variance8_gate_15/checkpoints.sqlite3`

## Immutable checkpoint identities

### Original baseline used by every paired evaluation

- checkpoint: `ckpt_b229ee0836324a7d96b0d4b0`, `pg-0@0`
- sampler: `tinker://4fae0a49-e641-5365-92d3-0a4d9f2a47ef:train:0/sampler_weights/optimizers-sampler_weights-save-e3c7b9fb8427cf8edb2b4366d2a519bd`
- sampler digest: `sha256:a4732ce4423db6060ac350a3228a3ef1e4910876592a312f2c098ee34e4dc2eb`

### Run-15 parent

- checkpoint: `ckpt_acfd561eda65c74c9fa351ab`, `pg-0@8`
- sampler: `tinker://4fae0a49-e641-5365-92d3-0a4d9f2a47ef:train:0/sampler_weights/optimizers-sampler_weights-save-5eef214380f03727d3f795fea0bb2bfd`
- sampler digest: `sha256:5c7026b85724ae867621e2e2263d6acde5cc264f81308abc75872f4b6a8cd6b8`
- training state: `tinker://4fae0a49-e641-5365-92d3-0a4d9f2a47ef:train:0/weights/optimizers-training_state-save-5727010753433ef524586df38e954377`
- training-state digest: `sha256:2336628bb3f372152bb768fb1e756d8b1b8b0bf2459cf1e5fa22bc3479e92bbc`

### Stage-1 final / Stage-2 exact parent

- checkpoint: `ckpt_3b5660b22f8a8de4b82cadea`, `pg-0@16`
- sampler: `tinker://12bc27f8-cce8-5290-980f-9f43a325a951:train:0/sampler_weights/optimizers-sampler_weights-save-e0a026763b2160aa685b1699769e0b9f`
- sampler digest: `sha256:762134748072f78367f0afc25ed04d4e7c3d86e2827d4f143d0dc7615f5b7c7f`
- training state: `tinker://12bc27f8-cce8-5290-980f-9f43a325a951:train:0/weights/optimizers-training_state-save-22c10b80244a2078ea54587ca9ccc110`
- training-state digest: `sha256:33474a59fde0a588fbe4b2d3e3f938598b42e1ba002e8a5b730c113ad9ceb894`

Stage 2's `resume_resolution.json`, `resume_artifact_identity.json`, and
`checkpoint_lineage.jsonl` independently bind that training state to the
cross-run `resumed_from` edge before the eight `trained_from` edges.

### Final trained checkpoint

- checkpoint: `ckpt_d02739fcc0546c017c3cfb94`, `pg-0@24`
- sampler: `tinker://b577d3c6-caad-5f81-b05e-d26de32c9f43:train:0/sampler_weights/optimizers-sampler_weights-save-4de8d7e9b590a800839c2b6627bdb79e`
- sampler digest: `sha256:faa5314f9e12ad2fc1b9a94dba03c07f4efb08e6e7094a204bdc8a03889b8a30`
- training state: `tinker://b577d3c6-caad-5f81-b05e-d26de32c9f43:train:0/weights/optimizers-training_state-save-ca5b3e3089c456204b25946319ab29cf`
- training-state digest: `sha256:efdf3c49b10fa464952191393ad0a8b3653dc26ddd5abe3fbf1ea4dcc5aa68f3`

The evaluation-specific immutable digest maps are in `evaluation_stage1/`,
`evaluation_stage2/`, and `evaluation_final/` under the experiment root.

## Engineering fixes proven by this sequence

- propagated executor-shaped token payloads, masks, behavior log-probabilities,
  and signed advantage-derived loss weights through the Tinker datum boundary;
- corrected sequence normalization so nonzero group coefficients remain
  nonzero at the provider;
- fixed repeated group sampling to keep the declared task seed stable while
  still obtaining rollout diversity from sample identity and the sampler;
- added immutable resume-from-training-state configuration and exact parent
  restoration, with fresh sampler materialization only after restore;
- made resumed baseline revision/idempotency and provider save-step semantics
  preserve the inherited revision;
- failed closed on base-model, parameter-group, artifact-role, compatibility,
  and independently supplied digest mismatches before paid restore;
- shared one artifact probe across prewarm and binder resolution, canonicalized
  Tinker identities, and retained only the exact resolved training-state
  ref/digest rather than an arbitrary digest map;
- recorded cross-run lineage, full checkpoint fields, resume resolution, and
  per-update loss-weight summaries in durable receipts.

Run 13 remains invalid evidence: its 60 optimizer calls had zero effective
gradients because advantages were lost at the provider boundary. Run 15 is
valid real training, but its positive training-panel probe was not a heldout
result and its first real validation was -2/77. The completed staged sequence
above is the first sealed positive final-panel result.

## Cost and operational status

Tinker reports `provider_cost: 0.0`, but every train call carries
`cost_missing: true`. Therefore monetary cost is **unknown**, not zero. Token
and call counts above are authoritative; cost attribution is not.

- Focused resume/binder/executor regressions: 48 passed; Ruff passed.
- Final repository-wide test status: **1,073 passed in 156.61 seconds**.
- Ruff passes across every file changed by this experiment. A repository-wide
  Ruff invocation still finds unrelated pre-existing findings in `.live-qa/`,
  `temp/`, and `tests/test_gsm8k_eval_target.py`; none of those files was
  modified here.
- Process check at handoff edit time: no matching trainer, evaluator, screener,
  validator, or Banking77 server process was running.
- Integrated implementation, configurations, tests, and machine-readable proof
  commit: `dc245a2`.

## Optional remaining work

No further work is required to establish this experiment's honest result. The
only scientifically useful follow-up is a predeclared, larger confirmatory
heldout panel (or multiple independently frozen panels) to obtain more
discordant pairs and enough power for a significance claim. It must preserve
the immutable baseline/final checkpoint identities above and must not be used
to tune the already completed training sequence.
