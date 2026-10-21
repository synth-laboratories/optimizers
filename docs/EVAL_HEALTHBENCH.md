# HealthBench container Eval

`python -m synth_optimizers.eval.healthbench` is an installed example over the
existing Eval candidate, seed, result and selection contracts. It consumes the
Containers package/build descriptor and shared OCI executor. This profile requires
the Containers candidate exposing `package_build` and `oci_trial`; the qualified
wheel and its digest are retained with the witness. It performs no Rust build.

Freeze before provider execution:

```sh
python -m synth_optimizers.eval.healthbench freeze \
  --evals-root /path/to/evals --dataset /inputs/healthbench.jsonl \
  --prices /inputs/openrouter-models.json --wheels /inputs/linux-wheelhouse \
  --base-image python@sha256:BASE_DIGEST --platform linux/arm64 \
  --output /releases/healthbench2-eval-1.0.0
synth-containers package plan /releases/healthbench2-eval-1.0.0/container-package.toml
synth-containers package build /releases/healthbench2-eval-1.0.0/container-package.toml \
  --output /receipts/build
python -m synth_optimizers.eval.healthbench run \
  --artifact /receipts/build/artifact.json --bundle /releases/healthbench2-eval-1.0.0 \
  --home /results/healthbench --run-id healthbench-witness
```

Provide `OPENROUTER_API_KEY` through the authorized environment. Plans and archives
contain a credential reference, never its value. The build uses an offline,
content-addressed wheelhouse. Dataset, prompt, local grader source/protocol,
routes, price snapshot, token ceilings and panels are frozen. No proposer runs.
The final panel remains unobserved. Row identity, rather than split labels,
establishes disjointness. The custom judge is not reference-protocol-equivalent.

Read-only observers:

```sh
python -m synth_optimizers.eval.healthbench watch --home /results/healthbench --run-id healthbench-witness
python -m synth_optimizers.eval.healthbench read --home /results/healthbench --run-id healthbench-witness
python -m synth_optimizers.eval.healthbench reconcile --home /results/healthbench \
  --run-id healthbench-witness --output /receipts/provider-charges.json
```

The TTY view reserves at most ten rows and a fixed status area on the right;
pipes have plain output. `read` checks every adopted artifact digest. Repeating
`run` on a committed result reads it without compute. An active controller holds
an exclusive home lock; another invocation observes the original run. A dead
controller with an unresolved operation refuses execution restart rather than
discarding evidence or resetting a provider budget. No detached controller or
checkpoint continuation is offered. `reconcile` reads the original generation
IDs from OpenRouter; it never submits a completion.

The retained local acceptance witness is report-only. A development installation
snapshot includes the existing native extension, which Eval does not call; this
is not a new native build, published optimizer wheel, hosted qualification or
optimization-uplift result. The broader v0.24 acceptance matrix stays independent.
