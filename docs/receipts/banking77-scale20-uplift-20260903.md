# Banking77 scaled throughput and uplift run

> Follow-up: run 11 subsequently demonstrated +2.60 percentage-point uplift
> on a new untouched 77-intent heldout panel. See
> `banking77-hard20-uplift-20260904.md`. The negative run-10 result below is
> retained as development evidence rather than overwritten.

Run `b77_scale20_uplift_10` is the terminal scaled experiment. It used the
container's declared ceiling of eight concurrent attempts, group size eight,
two trainable groups per optimizer update, a fixed `2e-4` learning rate, and a
variance-bearing nine-intent training curriculum. The paired evaluation used
one heldout example for every one of Banking77's 77 intents.

## Result

| Measure | Result |
|---|---:|
| Training rollouts | 448 |
| Execution window | 898.922 s |
| Observed throughput | **29.90 rollouts/min** |
| Same calls serialized | 16.61 rollouts/min |
| Overlap throughput uplift | **1.80x / +80.0%** |
| Peak concurrency | 8 |
| Optimizer updates | 14 |
| Heldout baseline | 81.82% |
| Heldout trained | 80.52% |
| Heldout uplift | **-1.30 percentage points** |
| Train initial batch mean | 81.25% |
| Train final EMA (alpha=0.2) | 79.51% |
| Train EMA uplift | **-1.74 percentage points** |

The throughput claim is measured, not the configured width: the 448 calls had
1,618.366 seconds of summed provider duration but completed inside an 898.922
second execution window. Dividing those quantities gives 1.800x realized
overlap. The terminal-completion-only rate is 29.98 rollouts/minute.

The quality result is negative. Heldout has 3 wins, 4 losses and 70 ties over
77 paired rows. No heldout-uplift claim is supported. The train EMA definition
was fixed before execution: alpha 0.2 over chronological policy-revision batch
means, with uplift equal to the final EMA minus revision zero's batch mean.

The safety ceiling stopped the run after 56 sampled groups: 28 were trainable,
28 had zero variance, and the 28 trainable groups packed into 14 updates. No
group was stale. The requested roughly-20-update scale was reached to 14
updates without widening the predeclared paid bound after observing outcomes.

Known receipted usage for the terminal training run is 282,936 prompt tokens,
27,448 generated tokens, and 158,793 training tokens. Evaluation-token usage
and the two interrupted precursor runs are not included in that lower bound.

Machine-readable metrics are in
`docs/receipts/banking77-scale20-uplift-20260903.json`. Raw training receipts
are at `/tmp/synth-container-first-e2e/receipts_banking77_scale20_paid_10`; the
paired receipt is at
`/tmp/synth-container-first-e2e/receipts_banking77_scale20_eval_10/b77_scale20_uplift_10_heldout_77.evaluation.json`.

## Scale defects encountered

- Run 07 used one task per intent. Forty-seven of 56 groups had zero variance,
  so it stopped after four updates and identified the variance-bearing
  curriculum used by later runs.
- Run 08 published seven updates before SQLite failed because the workstation
  disk was full. Clearing only the rebuildable `uv` cache recovered space.
- Run 09 published twelve updates before the default 900-second container
  handshake expired. The durable harness now supports
  `SYNTH_BANKING77_HANDSHAKE_TTL_SECONDS`; run 10 used 7,200 seconds.
- Training and evaluation must use fresh container processes when they share a
  run id, because probe idempotency is process-scoped and stable by run id.
