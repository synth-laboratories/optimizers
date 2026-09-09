# Banking77 CISPO debugging proof (2026-09-04)

## Outcome

The zero-uplift run 13 was not evidence that RL needed more updates. It was a
zero-gradient run caused by a provider-schema mismatch. The repaired path is
now proven at three levels: deterministic unit/integration tests, a direct paid
Tinker gradient/save/restore canary, and a paid socket-container training run
with measured behavioral uplift.

## Defects found and repaired

1. The executor emitted singular `advantage`; Tinker read plural `advantages`
   and silently substituted zero.
2. Missing, empty, nonfinite, or varying-vector sequence advantages did not
   fail closed.
3. The declared loss reducer was receipted but not materialized in provider
   gradients. Assembly now emits an explicit per-token `loss_weight` and
   composes it with same-policy target-share reweighting.
4. Configured CISPO clip bounds were not forwarded to Tinker.
5. Training-token usage counted masked prompt tokens.
6. `training_state` was saved through the sampler-weights API.
7. Sampler checkpoints were incorrectly advertised as resumable.
8. Restored sessions lost their base-model/renderer identity.
9. Live sampler refresh constructed a duplicate sampling client.
10. Adapter idempotency keys could alias results across different checkpoints.
11. The CLI ignored the configured polling cadence and could exhaust its small
    tick budget while valid asynchronous work was still running.

## Paid gradient canary

The one-call canary used the executor-shaped singular-advantage datum. With one
positive CISPO update it observed:

- maximum selected-token log-probability movement: `0.677635669708252`;
- target sequence summed log-probability change: `+3.253283547020146`;
- restored-state maximum error versus live post-update: `0.0`;
- distinct baseline/post sampler references and distinct training-state refs;
- provider `loss:sum = 0.37444889545440674`.

Receipt:
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_cispo_proof_20260904/gradient_canary.json`
(SHA-256 `9c56565eb11005b971fb0ed543f9101f27acf5490b56d94724397cabec11f1db`).

## Socket pipeline behavioral gate

Run `b77_variance8_gate_15` used the frozen 15 rows selected by the 8x screen,
16 rollouts/group, eight concurrent slots, and a fresh Tinker session. It
reached 8/8 effective updates after 9 sampled groups (one zero-variance skip),
with no stale groups. All eight provider calls reported nonzero `loss:sum` and
published both sampler weights and real `/weights/` resumable state.

On the fixed deterministic 15-row training probe:

- baseline: 10/15 (`66.67%`);
- trained: 12/15 (`80.00%`);
- delta: `+13.33` percentage points;
- 3 wins, 1 loss, 11 ties.

Evaluation receipt:
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_variance8_gate_15/evaluation/b77_variance8_gate_15_train15.evaluation.json`
(SHA-256 `8345de53002972a6ad3aa1c271b923e6b61f3748f7e4ccf79028b47e8a04801e`).

## Final current-code proof

After materializing the declared reducer coefficient, fresh run
`b77_objective_proof_17` reached its one-update target on the first sampled
group. The provider trained 16 examples / 1,151 selected tokens and reported
`loss:sum = -0.06367802526801825`. It published sampler checkpoint
`ckpt_afb9ffae23d99bb2a1ee0f1f` plus a distinct resumable training-state
artifact under Tinker's `/weights/` path.

Provider receipt SHA-256:
`d368fbc49955a13eab229b2801ecbaf6fac3dc12fbf6a4fe846a35ab4cf60325`.

Provider cost attribution remained unavailable (`cost_missing=true`), so the
provider-reported zero dollars is not claimed as actual cost.

Run 16 was an exploratory 60-update continuation. It reached 34 durable
effective updates before manual interruption during a long zero-variance
sampling stretch. Its catalog is retained as partial evidence, but no completed
run receipt or heldout claim is made from it.
