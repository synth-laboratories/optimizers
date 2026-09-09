# Banking77 hard-intent uplift run

Run `b77_hard20_uplift_11` used run 10 only as development evidence. It trained
on separate corpus rows from the 14 intents missed by run 10's baseline, then
evaluated at temperature zero on a fresh heldout example for every Banking77
intent. None of those 77 evaluation rows appeared in the earlier paired runs.

## Result

| Measure | Result |
|---|---:|
| Training rollouts | 640 |
| Observed throughput | **40.08 rollouts/min** |
| Same calls serialized | 19.73 rollouts/min |
| Overlap throughput uplift | **2.03x / +103.1%** |
| Peak concurrency | 8 |
| Optimizer updates | 7 |
| Fresh heldout baseline | 74.03% |
| Fresh heldout trained | 76.62% |
| Fresh heldout uplift | **+2.60 percentage points** |
| Paired outcomes | **2 wins / 0 losses / 75 ties** |
| Train initial batch mean | 63.54% |
| Train final EMA (alpha=0.2) | 55.26% |
| Train EMA uplift | **-8.28 percentage points** |

The heldout uplift is real but small: both arms used immutable, digest-verified
checkpoints, identical task/seed ordering, temperature zero, and an untouched
77-intent panel. The training curriculum was chosen from run 10's different
heldout rows and used only Banking77 train rows.

The train EMA remains negative and is recorded as such. Exact-match reward was
too sparse: 66 of 80 groups were zero-variance, leaving 14 trainable groups and
seven optimizer updates. More paid sampling is unlikely to fix that mechanism;
the next algorithmic step should introduce a legitimate dense reward or a
supervised warm-start rather than searching for lucky seeds.

The throughput uplift is directly measured. The 640 provider calls contained
1,946.047 seconds of summed call duration but completed in a 958.066-second
execution window. That is 2.031x realized overlap, at the declared peak of
eight concurrent attempts.

Known training usage is 406,224 prompt, 39,592 generated, and 80,844 training
tokens. At the published uncached rates this counted portion estimates to
$0.123. Evaluation usage is not present in the receipt and is excluded.

Machine-readable metrics are in
`docs/receipts/banking77-hard20-uplift-20260904.json`. Raw training receipts are
at `/tmp/synth-container-first-e2e/receipts_banking77_hard20_paid_11`; the
paired receipt is at
`/tmp/synth-container-first-e2e/receipts_banking77_hard20_eval_11/b77_hard20_uplift_11_fresh_heldout_77.evaluation.json`.
