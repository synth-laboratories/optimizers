# Banking77 container Eval

`python -m synth_optimizers.eval.banking77` qualifies local native evaluation of
Banking77 through Eval's existing candidate/panel/result authority and Containers'
shared OCI primitive. It reuses the HealthBench example's native event consumer,
provider reservation guard, controller, observer, cancellation and readback. The
native consumer accepts explicit world/evaluation-plan/harness identities; its
HealthBench defaults stay compatible.

Freeze a portable release before provider execution:

```sh
python -m synth_optimizers.eval.banking77 freeze \
  --evals-root /path/to/evals --wheels /inputs/linux-wheelhouse \
  --dataset /inputs/banking77-source --prices /inputs/openrouter-models.json \
  --base-image python@sha256:BASE_DIGEST --platform linux/arm64 \
  --output /releases/banking77-eval-1.0.0
synth-containers package build /releases/banking77-eval-1.0.0/container-package.toml \
  --output /receipts/build
python -m synth_optimizers.eval.banking77 run \
  --artifact /receipts/build/artifact.json --bundle /releases/banking77-eval-1.0.0 \
  --home /results/banking77 --run-id banking77-witness
```

The source directory contains `banking77-train.csv`, `banking77-heldout.csv`, and
`source-provenance.json` with the upstream repository, immutable commit, per-split
URLs and SHA256 digests. Runtime reads those frozen local CSVs. The recipe evaluates
source heldout indices0 and40; index80 is reserved and unobserved. Baseline,
selection and final are evaluation roles. Actual text-plus-label identities and
query identities must be disjoint, independently of split names. These tiny
panels establish execution support only.

The answer route is OpenRouter `openai/gpt-4.1-mini`, with128 output tokens and
truncated completions refused. Provide `OPENROUTER_API_KEY` through the authorized
environment. Credentials never enter release archives. The scorer normalizes the
first output line using the frozen Banking77 implementation and computes label
accuracy; it has zero provider calls/cost. Proposer calls are zero. Gold remains
outside public observations and the prompt candidate. Actual observations and
predictions are retained and reward is recomputed against evaluator-owned gold.

Native execution is prepare → subscribed SSE → start → terminal/reward/trace →
retain → drop-session/close. Ordered adapter custody states remain separate from
native rollout state. Scoped cursors, original envelopes, chain heads and terminal
seals are verified. A deliberate SSE disconnect resumes from its scoped ID;
identical starts replay and conflicting identities refuse. Original generation IDs,
raw provider usage and truncation status support charge reconciliation. Uncertain
provider effects retain their reservations.

```sh
python -m synth_optimizers.eval.banking77 watch --home /results/banking77 --run-id banking77-witness
python -m synth_optimizers.eval.banking77 read --home /results/banking77 --run-id banking77-witness
python -m synth_optimizers.eval.banking77 reconcile --home /results/banking77 --run-id banking77-witness --output /receipts/charges.json
python -m synth_optimizers.eval.banking77 cancel --home /results/banking77 --run-id banking77-witness
```

Progress displays at most ten rollout rows and a fixed status area. Reconnect
observes original operations; it does not restart compute or reset budgets. Native
journals live explicitly under output/native-storage. Completed operations can be
reopened in fresh network-disabled processes, while an unresolved execution restart
refuses. Cancellation belongs to the owned OCI supervisor; partial events/cursors
and outstanding reservations survive. This profile has no per-rollout native
cancellation API, so its last native state may remain running after supervisor
cancellation.

The qualification is local Linux ARM64 using an installed development snapshot,
without a new Rust build or published optimizer wheel. Hosted Rhodes/Tunnel,
optimization uplift and the separate GEPA/Banking77 matrix witness are outside
this Eval qualification.
