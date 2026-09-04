# Banking77 scaled-uplift handoff (2026-09-04)

## Objective

Finish the scaled Banking77 CISPO demonstration and report:

- heldout uplift against the immutable base checkpoint;
- train reward EMA uplift (`alpha = 0.2`);
- observed rollouts/minute;
- serialized-equivalent rollouts/minute and overlap uplift;
- token usage and estimated provider cost.

Run 11 is the last fully recorded success. It achieved **+2.60 percentage points**
on a fresh 77-intent heldout panel and **40.08 rollouts/minute**, but only seven
optimizer updates. Its committed report is
`docs/receipts/banking77-hard20-uplift-20260904.md`.

## Run 12 training: completed

The new immutable configuration is
`docs/e2e/configs/run_b77_hard20_paid_12.toml`.

Run 12 used:

- run id: `b77_hard20_uplift_12`;
- model: `openai/gpt-oss-20b` through Tinker;
- learning rate: `5e-5`;
- 16 samples per group, executed through eight concurrent slots;
- one group per optimizer step;
- a round-robin curriculum over four train rows for each of the 14 difficult
  intents identified from run 10;
- target: 20 durable optimizer updates;
- sampling ceiling: 56 groups.

The paid training run completed successfully with:

- **20 durable updates** (`pg-0@20`);
- **896 rollouts** (56 groups x 16);
- **20 trained groups**, 36 zero-variance/skipped groups;
- **0 stale groups**;
- stop reason: `target_train_updates_reached`.

Immutable artifacts observed at completion:

| Arm | Checkpoint | Tinker sampler reference | Digest |
|---|---|---|---|
| Baseline | `ckpt_cc7549684479d00ab6ac661c` | `tinker://d518bd2b-0e94-5e74-ba7f-2e9503bbb630:train:0/sampler_weights/optimizers-sampler_weights-save-9c6921efe9e99b390f40d0edaa6131ae` | `sha256:d051e2eff45cc28f9dbb470a4fb8ecee897bdf48a2fff955b906a17ca23d4619` |
| Final | `ckpt_3db5a6a1e7844ac6ccbe882e` | `tinker://d518bd2b-0e94-5e74-ba7f-2e9503bbb630:train:0/sampler_weights/optimizers-sampler_weights-save-96bcf18a9ab0d17cbcde9540bc383a1e` | `sha256:4e4e761e5948f6f9e39b3a3a995ba10907f785aa5054473535f42fefcebf9047` |

The provider training session embedded in both refs is
`d518bd2b-0e94-5e74-ba7f-2e9503bbb630:train:0`.

## Important artifact-loss condition

Training wrote its catalog and receipts to `/tmp` as specified by the run-12
config. After the evaluation process was interrupted, the execution environment
was replaced and `/tmp/synth-container-first-e2e` was no longer present. The
workspace config survived, but the local run-12 SQLite catalog, reward receipts,
and token receipts did not.

Therefore:

- do **not** claim run-12 throughput, train EMA, tokens, or cost from memory;
- do **not** claim that the raw run-12 evidence is locally available;
- the remote baseline/final Tinker sampler refs and their observed digests are
  recorded above and may be rehydrated if the catalog API supports importing
  immutable checkpoints;
- if rehydration is not supported, rerun the same training config after changing
  `[artifacts]` and `--receipts` to a durable workspace-external directory, then
  copy the final summarized metrics into `docs/receipts/` before stopping.

The original authorized experiment ceiling was $10, expected under $1. A new
paid rerun should follow the repository/user provider-approval rules in force for
the engineer's task.

## Evaluation attempts and the exact trap

The intended third panel is the 77 ids in run 12's `evaluation_ids`. It was
verified before execution to contain 77 unique rows and to have no overlap with
the 185 heldout seeds then found in earlier evaluation receipts.

The first evaluation invocation incorrectly forced `--reward-channel score`.
Banking77's solo-team reward is negotiated as `score::team-0`, so evaluation
failed with:

```text
error: reward reward_rollout_5b28c00a723347f10e6e has no channel 'score'
```

Omit `--reward-channel`; let the evaluator use each reward's immutable
`optimized_channel`.

Retrying immediately against the same container then reused the deterministic
probe id and failed renewal because the probe was already terminal:

```text
LifecycleError: attempt probe_43736599c4bf72030eb6 is terminal; nothing to renew
```

Restart the Banking77 server before every retry that has crossed the handshake
probe.

After restart, evaluation without `--reward-channel` ran for several minutes but
was interrupted before it emitted a receipt. Treat the third panel as
**partially observed**, not untouched. For a defensible final number, select a
fourth panel: the next unused heldout row for each of the 77 labels, excluding all
ids in the run-12 config as well as the earlier receipts/configs.

## Environment and server

Do not use macOS Keychain. The previously authorized provider environment was:

```text
SYNTH_TINKER_ENV_FILE=/Users/joshuapurtell/GitHub/frontend/.env.local
```

The working renderer canary observed from the provider was:

```text
43e18d1c29ee9cc6a849f8fc77c9efee
```

Start deterministic evaluation with a fresh process:

```sh
SYNTH_BANKING77_SOURCE=hf \
SYNTH_BANKING77_DECLARED_ROWS_PER_SPLIT=10003 \
SYNTH_CISPO_RENDERER_CANARY_DIGEST=43e18d1c29ee9cc6a849f8fc77c9efee \
SYNTH_BANKING77_TEMPERATURE=0 \
SYNTH_BANKING77_HANDSHAKE_TTL_SECONDS=7200 \
uv run --with pytest --with uvicorn python docs/e2e/serve_banking77.py 8241
```

The evaluation command should follow this shape after restoring/recreating the
catalog, digest file, and pin file:

```sh
PYTHONPATH=docs/e2e \
SYNTH_TINKER_ENV_FILE=/Users/joshuapurtell/GitHub/frontend/.env.local \
uv run synth-optimizers rl evaluate \
  --catalog PATH_TO_DURABLE_CATALOG/checkpoints.sqlite3 \
  --selector FINAL_CHECKPOINT_ID \
  --baseline BASELINE_CHECKPOINT_ID \
  --evaluation-id b77_hard20_uplift_12_fresh_heldout_77_v2 \
  --roster instance-0=pg-0:policy-0 \
  --split heldout \
  --scope-run b77_hard20_uplift_12 \
  --scope-parameter-group pg-0 \
  --scope-policy-type policy-0 \
  --metric mean_reward \
  --artifact-digests PATH_TO_ARTIFACT_DIGESTS.json \
  --pin PATH_TO_EVALUATION_PIN.json \
  --config docs/e2e/configs/run_b77_hard20_paid_12.toml \
  --plane paid_plane:paid \
  --receipts-dir PATH_TO_DURABLE_EVAL_RECEIPTS \
  --seed banking77/heldout/ID=ID \
  --json
```

Repeat `--seed` for all 77 fourth-panel rows. Do not pass `--match-set
match-set-0001`; it was not registered in the run-12 catalog. With no opponents,
the correct evaluation receipt has a null match-set revision.

## Finish criteria

1. Preserve the run-11 result; never overwrite its config or report.
2. Recover the immutable run-12 refs into a valid catalog, or rerun training to
   durable paths.
3. Use a new 77-intent panel and temperature zero for both arms.
4. Verify 77 pairs, checkpoint digests, identical seed order, two arms, and
   `score::team-0` reward binding.
5. Compute throughput from the execution window and summed provider-call
   durations, not from wall-clock guesswork.
6. Compute train means by policy revision and EMA with fixed `alpha = 0.2`.
7. Record wins/losses/ties and the honest heldout delta, even if it does not beat
   run 11's +2.60 pp.
8. Write machine-readable JSON plus a Markdown receipt under `docs/receipts/`,
   update the main container-first handoff, run validation/tests, and commit only
   the exact owned files.

## Repository state at handoff

Branch: `feat/container-first-rl`.

The only owned uncommitted file before this handoff was the new run-12 config.
`.live-qa/` and `temp/` were pre-existing untracked directories and must remain
untouched.
