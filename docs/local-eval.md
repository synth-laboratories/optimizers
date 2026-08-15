# `eval` — local candidate evaluation

`eval` is a **local** Optimizers algorithm. It takes an immutable set of policy
candidates and an allowlisted recipe, expands them into a fair
`candidate × seed × scenario` trial matrix, runs each trial in the recipe's
pinned container, retains all evidence, scores every candidate separately, and
returns a promotable winner only when the declared gates pass.

It is not hosted. Do not add `eval` to
`synth_optimizers.hosted.OptimizerAlgorithmSlug` or to `future_algorithms.py`:
that enum describes hosted API compatibility, which this is not.

There are no Craftax-, GameBench-, or Harbor-specific execution paths in the
runner. Those are containers implementing one contract, `eval.target.v1`.

## The pieces

| Module | Owns |
| --- | --- |
| `eval/models.py` | every wire schema, and the validation that refuses partial input |
| `eval/recipes.py` | the trusted catalog: images, seeds, metrics, gates, limits, selection |
| `eval/home.py` | the app-owned home: `runtime.toml`, operator pins, candidate store, run evidence |
| `eval/staging.py` | copying policy source into immutable content-addressed artifacts |
| `eval/semaphore.py` | the machine-global trial lease store |
| `eval/executor.py` | launching one trial in the recipe-pinned OCI image |
| `eval/scoring.py` | per-candidate scorecards, elimination, paired lift, selection |
| `eval/runner.py` | sealing, planning, executing, resuming, cancelling, sealing again |

## The target contract, `eval.target.v1`

A conforming container gets exactly:

```text
/input/policy/       immutable candidate contents, read-only
/input/trial.json    candidate id/digest, seed, scenario, limits, run/trial ids
/output/             container-owned result, trace, artifacts, optional events
```

and must write `/output/result.json` (`eval.container-result.v1`). The one
distinction that matters most:

- `status: "failed"` — the **rig** could not evaluate the trial.
- `status: "evaluated", benchmark_status: "failed"` — the rig worked and the
  **policy** lost or was invalid.

Both are preserved. A missing metric stays missing; it is never scored as zero.

A target that declares `required_artifacts` must write them. `["trace"]` means
every trial writes the rollout it was scored on, so a number can be re-derived
later instead of trusted. A trial that promises a trace and omits it is
`evaluated` but not `valid`, and it cannot contribute to a decision.

## Seeds and selection

```text
screen:   every candidate + baseline on the same screening seeds
prune:    only the recipe's declared, recorded rule may remove a candidate
confirm:  baseline + survivors on fresh, shared confirmation seeds
select:   evaluate gates and paired metric lift; retain every raw result
```

Seeds are generated once, written into the sealed manifest, and never
regenerated — a restart resumes the same measurement. Candidates in a group get
the same seeds (common random numbers), so lift is a paired difference over the
seeds both arms completed, and confirmation seeds are disjoint from screening
seeds by construction.

Two statuses travel separately, and neither stands in for the other:

```text
run status:        completed | failed | cancelled
selection status:  promoted | no_champion | inconclusive | invalid_evidence
```

## The home

Everything configurable lives in TOML under the app-owned home. None of it is
an environment variable, so a run's settings can be read back afterwards.

```text
<home>/runtime.toml        container runtime + global concurrency ceiling
<home>/pins.toml           operator-pinned image digests, per catalog recipe
<home>/semaphore/          global lease store, shared by every local run
<home>/candidates/<id>/    immutable staged candidate sets
<home>/runs/<run_id>/      sealed manifests and trial evidence
```

A run's evidence tree:

```text
<run>/input_manifest.json     sealed inputs: recipe, image digest, candidates, seeds
<run>/seed_ledger.json        the explicit integers, written once
<run>/events.jsonl            append-only run lifecycle
<run>/trials/<id>/input/      exactly what the container was given
<run>/trials/<id>/output/     exactly what it wrote, including trace.jsonl
<run>/trials/<id>/job_result.json   the durable terminal record for that trial
<run>/scorecards.json         per candidate, per stage
<run>/selection.json          the decision and why
<run>/result_manifest.json    every trial and every artifact, with digests
```

The semaphore is a concurrency primitive, not a product surface. There is one
lease store per home, shared by every worker process, so the ceiling belongs to
the machine rather than to whichever run started first. A lease whose owner died
or whose heartbeat lapsed is reclaimed by the next acquirer.

## Operating it

```bash
# 1. Build a target and pin exactly what you built.
docker/eval-fixture-target/build.sh
synth-optimizers eval pin --home ~/eval --recipe eval.fixture.policy-smoke.v1 \
    --digest sha256:...

# 2. Check the runtime, the OCI runtime, and which recipes are usable.
synth-optimizers eval doctor --home ~/eval

# 3. Freeze policy source into a candidate set.
synth-optimizers eval stage --home ~/eval \
    --candidate uniform=./policies/uniform \
    --candidate greedy=./policies/greedy \
    --baseline uniform

# 4. Run it. The manifest is app-owned; the worker takes no other input.
synth-optimizers eval worker --manifest ~/eval/workers/<run>.json

# 5. Ask a running worker to stop, seal evidence, and release its leases.
synth-optimizers eval cancel --home ~/eval --run-id <run>
```

`eval worker` is an app-internal launcher. It accepts a recipe id and an
app-owned manifest path; it never accepts an image, a command, a mount, or an
environment variable.

## Shipped targets

| Recipe | Target | Notes |
| --- | --- | --- |
| `eval.fixture.policy-smoke.v1` | `docker/eval-fixture-target` | Deterministic corridor. Proves result, gates, trace, cancel, resume, semaphore without a benchmark. |
| `eval.craftax.code-policy.smoke.v1` | `docker/craftax-eval-target` | GameBench symbolic Craftax engine in-container. Report-only. |
| `eval.gamebench.craftax-code-policy.confirm.v1` | `docker/gamebench-harbor-eval-target` | Harbor workspace and verifier surface, adopted into the standard evidence tree. Promotes. |

The Craftax scenario names the board it uses (`craftax-default-48x48`). A
Craftax number that does not say which world produced it is not a result: the
48×48 default board and a 9×9 fixture room are not the same benchmark.

Both GameBench targets run the policy sweep unsupervised in a private working
directory, because GameBench's per-episode sandbox needs unprivileged user
namespaces that are not available inside a container on every host. The eval
container is the isolation boundary, and the target wrapper — not the sweep, and
not the candidate — publishes `/output` after the sweep exits. That is why the
Craftax smoke recipe is report-only.

## Adding a benchmark

Publish a container conforming to `eval.target.v1`, pin it by digest in an
allowlisted recipe, and choose the recipe-owned metric and selection policy.
That is the whole change. It is not a new algorithm, not an agent-supplied
Docker command, and not a new Workshop orchestration path.
