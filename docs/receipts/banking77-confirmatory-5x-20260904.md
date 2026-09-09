# Banking77: fresh 385-example confirmation

The larger evaluation completed successfully on 2026-09-04. It found a small
positive gain, but **does not establish reliable heldout uplift**.

| Metric | Result |
|---|---:|
| Baseline | 309/385 = 80.26% |
| Trained | 313/385 = 81.30% |
| Paired gain | +4/385 = +1.04 percentage points |
| Wins / losses / ties | 12 / 8 / 365 |
| Paired-bootstrap 95% interval | −1.30 to +3.38 percentage points |
| Exact two-sided McNemar p | 0.5034 |

The earlier sealed 77-example panel showed +3.90 points with p=0.25. This new
panel is separate evidence, not a replacement that makes the earlier estimate
more certain. The current checkpoint has positive observed gains on both, but
neither result supports a statistically significant uplift claim. Do not keep
adding test panels until one passes a significance threshold.

## Panel and checkpoint integrity

The panel was frozen before sampling: five distinct rows per each of 77 intents,
385 total, excluding 570 previously recorded heldout IDs. The deterministic
selection hashes rank candidates within each intent. The frozen panel digest is
`sha256:6db23335189037974cd51ed8bd067a1d0093d7d9e5d300923e69a901ae9d6fe1`.

- Baseline: `ckpt_b229ee0836324a7d96b0d4b0`, revision 0.
- Trained: `ckpt_d02739fcc0546c017c3cfb94`, revision 24.
- Both exact sampler references/digests remain those in the
  [engineering handoff](../HANDOFF_BANKING77_REAL_HELDOUT_UPLIFT_2026-09-04.md).
- Both arms ran the same task IDs, seeds, order, and `score::team-0` channel at
  temperature zero. All 770 attempts completed successfully.
- Validation checked panel balance, hashes, train/prior-panel disjointness,
  checkpoint/reference identity, attempt order, terminal state, unique rollout
  and proxy identities, and paired-summary consistency.
- Bootstrap: 20,000 replicates, seed 20260904. This is the existing paired-row
  percentile procedure; the estimand is accuracy on the balanced selected panel.

Durable evidence directory:
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_real_uplift_18/confirmatory_5x`.
It contains `panel.json`, `evaluation_pin.json`, `artifact_digests.json`,
`b77_real_uplift_18_confirmatory_5x.evaluation.json`, and `result.json`.
The raw receipt SHA-256 is
`2bbd48b658eccfd4ac923791f87520579add7c0cf7dc74cfafce64c7b429b0df`.
The [machine-readable summary](banking77-confirmatory-5x-20260904.json) records
the remaining hashes, usage, and exact statistics.

## Throughput and cost

The measured evaluation window was 19:26:21–20:29:56 UTC: **63.57 minutes**,
**12.11 attempts/minute**, and **10.74 generated tokens/second**. This runner
executes baseline then trained sequentially. Configured training concurrency
does not parallelize this evaluation path. The initial 30–40 minute estimate
was too optimistic.

The receipt counts 487,224 prompt tokens and 40,957 generated tokens across
770 calls. At [Tinker's current GPT-OSS-20B rates](https://tinker-docs.thinkingmachines.ai/tinker/models/),
the sampling estimate is **$0.036–$0.106**, spanning all-prefill-cached to
none-cached. Actual invoiced dollars and the cache fraction remain unknown;
this estimate excludes storage and prior experiment costs. The authorized
aggregate cap for this confirmation was $5.

Two startup checks failed before sampling: the default CLI assembly did not
load the authorized `.env` credential, and its renderer failed the pinned
canary check. The existing `--plane paid_plane:paid` adapter loaded the named
credential file and initialized Tinker's renderer successfully. No Keychain
access was used, no compatibility check was bypassed, and no training update
was performed during confirmation.

Read-only progress checks queried deterministic rollout IDs. The container
returns HTTP 500 with a traceback for IDs not yet created; these diagnostic
errors were separate from the 770 successful evaluation attempts. This is a
container observability issue to fix, and the measured elapsed time includes
monitoring overhead. Do not attribute the entire runtime to provider latency.
The evaluator and Banking77 server were stopped after the receipt was saved.

## What remains

1. Improve the training experiment using training/validation evidence: expand
   candidate coverage, apply the requested 8x screen and 1–7/8 admission rule,
   and predeclare the update budget and checkpoint-selection rule. The prior
   stage screened only 56 candidates and selected six tasks, so training
   coverage remains narrow. More updates alone are not proven to help.
2. Reserve new untouched confirmation data before another training cycle;
   this 385-row panel is now observed. Predeclare the effect size and power
   target, and stop treating repeated significance tests as independent proof.
3. Add bounded concurrency and durable per-attempt progress to paired evals,
   preserving deterministic order, identity, and restart semantics. Add a
   proper missing-rollout response/progress endpoint in the container.
4. Reconcile actual charges against Tinker's delayed billing feed when
   available. Token-based estimates are now possible; invoice attribution
   remains separate from the sampling receipts.

The panel freezer/validator now support configurable examples per intent while
retaining the one-example default. All **1,086 tests passed**, changed Python
files passed Ruff, and receipt validation passed. No additional paid run was
started after inspecting this result.
