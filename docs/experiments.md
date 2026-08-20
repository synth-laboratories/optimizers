# `experiment` — controlled ablations over the executors that already exist

An ablation is a comparison, and a comparison is only worth anything when the
only thing that differed is the thing you meant to change. `experiment` is the
thin layer that makes that guarantee cheap: you declare what varies, and it
proves that nothing else did.

It does three things.

1. **Assign** — expand a spec into a frozen matrix of arms × blocks ×
   replicates, with a materialised dispatch order and a deterministic id per
   trial.
2. **Adapt** — hand each cell to an executor that already exists, carrying a
   correlation envelope that survives into that executor's own evidence.
3. **Reduce** — pair sealed outcome rows by block and report the comparison the
   design declared, including every reason it does not support a headline.

It is deliberately **not** a scheduler, an optimizer, or a second result store.
`eval`, the matrix runner, and GEPA already own execution, queueing, and
evidence. The moment this layer starts owning those too, there are two stories
about what ran.

## The pieces

| Module | Owns |
| --- | --- |
| `experiment/models.py` | the wire records: correlation envelope, subject reference, factor catalog, trial outcome |
| `experiment/spec.py` | `synth.experiment.v1`, and every refusal a malformed design earns |
| `experiment/plan.py` | deterministic expansion, arm ids, dispatch order, the drift guard |
| `experiment/outcomes.py` | the append-only `trial_outcomes.jsonl` |
| `experiment/analysis.py` | paired deltas, bootstrap CI, permutation p, missingness, the claim gate |
| `experiment/runner.py` | dispatch in the planned order; nothing clever |
| `experiment/adapters/eval_runtime.py` | the local `eval` container runtime |
| `experiment/adapters/gepa_cli.py` | GEPA, driven through its own config file |

## Writing a spec

```toml
schema = "synth.experiment.v1"
experiment_id = "craftax-policy-ablation-v1"
executor = "eval.runtime"
base = "eval.craftax.code-policy.smoke.v1"   # a trusted recipe id

[design]
primary_metric = "benchmark_score"
secondary_metrics = ["steps"]
pairing = "block"
counterbalance = true
missing_policy = "fail"          # or pairwise_complete; there is no imputation
min_blocks_for_claim = 8
min_completion_rate = 0.9
max_differential_missing_blocks = 0

[blocks]
kind = "seed"
values = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
replicates = 1

[factors]
"policy.candidate" = ["baseline", "champion"]

[budget]
max_trials = 20
max_cost_usd = 40
max_wall_minutes = 240

[isolation]
cache_namespace = "per_trial"
container = "fresh_per_trial"

[executor_options]
home = "~/.synth-desktop/optimizers/eval"
candidate_set = "policy_set_9c1f2a04bb31"
```

The spec says **what varies and what must not**. It never says how to run
anything: the image, the seed schedule a target may see, the metrics, and the
resource ceilings all come from the executor's own trusted configuration, and
the spec may only select among what is already declared there. That asymmetry
is the whole design — it is what makes an ablation cheap to write and still
impossible to accidentally turn into an unfair comparison.

## Factors are published, not inferred

Being in a config schema does not make a knob ablatable. A per-trial timeout is
in the schema, and varying it across arms silently changes the measurement.

So each executor publishes an **ablatable-factor catalog** — canonical path,
type, allowed values, redaction, and whether the UI may show it — and a factor
outside that catalog is refused:

```console
$ synth-optimizers experiment factors --spec craftax.toml
eval.runtime / eval.craftax.code-policy.smoke.v1
  policy.candidate  [enum]  'baseline', 'champion'
      Which staged policy artifact runs, named by its staging label.
  model.policy.effort  [enum]  'low', 'medium', 'high'
      Reasoning effort for the policy route.
```

## Three projections, one guard

Every trial's resolved configuration is split three ways:

- `fixed` — identical in every arm;
- `treatment` — the declared factor values;
- `trial_derived` — ids, paths, cache namespace, correlation, block seed.

`assert_only_treatment_differs` then refuses any inter-arm difference that is
not in `treatment`. The projections exist precisely so that comparison has
somewhere to put an *expected* difference: a diff in `trial_derived` is normal,
and a diff anywhere else is a bug.

## Correlation

Every trial carries a `synth.correlation.v1` envelope:

```json
{
  "experiment_id": "craftax-policy-ablation-v1",
  "arm_id": "arm_6edf53cf5835",
  "block_id": "seed:104",
  "replicate": 0,
  "trial_id": "t676b09f94e51e2f3",
  "plan_digest": "sha256:…",
  "subject": {
    "subject_kind": "policy-candidate",
    "subject_id": "policy_ebc737dfa7f1",
    "subject_content_digest": "sha256:…"
  },
  "candidate_id": "policy_ebc737dfa7f1"
}
```

`trial_id = hash(experiment_id, arm_id, block_id, replicate)`, so resume
recomputes the same identity from the spec alone and a trace found months later
can be matched without a lookup table.

**Absent keys, never null ones.** The envelope has to survive being written into
an executor's own config format and read back byte-identical, and TOML has no
null. An optional field emitted as `null` would silently vanish on the round
trip and take the digest with it. Both `candidate_id` and
`subject.parent_subject_id` are omitted when unset, on both sides of the
language boundary.

**The digest is recomputed, never transmitted.** The producer computes it from
its own canonical encoding and compares it against the envelope it gets back.
Sending it would mean Python and Rust agreeing byte for byte on JSON
canonicalisation forever, which is a promise not worth making for a field
nothing on the far side reads. The exact wire form is pinned as a literal in
both `tests/test_experiment_layer.py` and
`synth_optimizer_platform::correlation`, so a field added to one and not the
other fails a test rather than a run.

The envelope carries **no run id**. A service mints its own and reports it back;
a caller supplying one would be asserting authority over a namespace it does not
own, and resume would then have two candidate truths.

Downstream indexes carry a **bounded** alias subset — `experiment_id`,
`trial_id`, `candidate_id` — while the full envelope and its digest live in the
sealed evidence record. Indexing every factor would make the trace index a
second, divergent copy of the plan.

## Outcomes and analysis

Each terminal trial appends one `synth.trial-outcome.v1` row, and the file is
never rewritten. A rerun that produces a second row for the same `trial_id` is a
contradiction the reducer surfaces rather than resolves.

The reducer is a pure function of plan plus rows:

- pair by `block_id`, collapsing replicates within a block first;
- paired bootstrap CI over the **differences**, seeded from the plan digest, so
  a report is reproducible byte for byte;
- exact paired permutation test up to 14 blocks, seeded Monte Carlo beyond that;
- arm aggregates, every paired delta, expected/completed/missing blocks, cost,
  elapsed time, and the fairness facts actually observed.

Three rules do most of the work:

- a failed trial is **missing evidence, never a zero**;
- missingness is harmless only when it is **symmetric**, so it is measured per
  arm and the *difference* is what gates a claim;
- the order trials actually ran in is evidence, and nominal counterbalancing is
  not, so fairness is computed from recorded dispatch and start times.

`headline_claim_allowed` is emitted only when the declared sample, the
confidence interval, completion, missingness, image-identity, isolation, and
start-order requirements all pass. Everything that blocked it is listed.

An interval that contains zero closes the gate rather than annotating it. A null
result is a real finding and the report states it plainly — what it cannot
support is a directional headline.

## Commands

```console
synth-optimizers experiment factors --spec spec.toml    # what may vary
synth-optimizers experiment plan    --spec spec.toml --root ./run
synth-optimizers experiment aa      --spec spec.toml --root ./run
synth-optimizers experiment run     --spec spec.toml --root ./run
synth-optimizers experiment resume  --spec spec.toml --root ./run
synth-optimizers experiment report  --spec spec.toml --root ./run --json
```

Every verb recompiles the plan from the spec first. If the recipe, the target
image, or the staged candidate set has moved since the first dispatch, the
digests disagree and the command refuses rather than quietly comparing two
different measurements.

`resume --retry-rig-failures` re-dispatches trials whose sealed failure was
the **rig's**, not the arm's. A crashed container says nothing about a
treatment, so re-running it is not cherry-picking — but only `rig` and `infra`
qualify. A `policy` failure is the thing under test, and a `budget` or `timeout`
failure may itself *be* the arm difference; retrying either would be selecting
for the result you wanted.

A retry appends a new row with `attempt = N+1` and the digest of the row it
supersedes. Nothing is edited away: the superseded rows stay in the log,
`retried_trials` appears in the report totals, and the claim verdict carries a
note naming them — a rig that needed three attempts is a fact about the
comparison. Two rows for the *same* attempt remain a contradiction, not a retry.

Each retry also runs under its own executor identity (`…_r1`). Every executor
here resumes its own work from a sealed record, so re-dispatching under the
original id would replay the failure the retry exists to clear.

`aa` runs the baseline against itself on the first three blocks. It is an
**identity and isolation smoke test** — a non-zero delta there is a shared
cache, a warm container, or a leaked seed — and it never yields a headline. It
is not a noise-ceiling estimate, and three blocks could not be one.

## What it refuses

| Situation | Result |
| --- | --- |
| a factor the executor does not publish | refused at plan time |
| a seed the recipe does not declare | refused at plan time, with "widen the recipe, not the experiment" |
| a matrix over `budget.max_trials` | refused at parse time |
| a factor with one level | refused: that is a fixed value, and belongs in `[fixed]` |
| a typo'd top-level key | refused: a silently ignored table is how a control becomes a decoration |
| two arms identical in treatment *and* subject | refused: that is one arm, not two |
| an edited `plan.json` | refused: its recorded digest no longer matches its contents |
| a moved image pin mid-experiment | resume refused: different measurement |
| a trial sealed twice with different content | reported as a conflict; the claim is blocked |
| any missingness under `missing_policy = "fail"` | claim blocked |
| asymmetric missingness under `pairwise_complete` | claim blocked |
| arms that ran against different images | claim blocked |
| a confidence interval that contains zero | claim blocked; the null result is still reported |

## The `eval.runtime` adapter

`eval` already runs a fair `candidate × seed × scenario` matrix with common
random numbers, so the adapter does not reimplement any of it. It drives **one
`eval` run per cell** of the experiment matrix, which buys the two things a
single batched run cannot give: an exact dispatch order to counterbalance, and
one sealed, separately resumable receipt per trial.

Every override it sends narrows the trusted recipe — a subset of the declared
seeds, one staged candidate, one of the declared reasoning efforts. The
`plan_override` block on the worker manifest is a subset check in every branch,
so the recipe remains the only place an image, a seed, a model route, or a
ceiling can come from.

## The `gepa.cli` adapter

One rendered TOML per trial, run through the same `GepaRun` path a human would
use. GEPA keeps its run directory, manifest, run registry, and event feed; the
adapter reads the terminal record it already writes.

Its factor catalog is **curated, not reflected**. GEPA's TOML sections are
`extra="ignore"` by construction — the `[dataset]`-versus-`[taskset]` trap in the
public cookbooks is what that costs — so a schema walk would offer knobs the
engine silently drops. Publishing an explicit list makes adding a treatment a
deliberate act. What is offered varies *how* the proposer thinks:

```text
proposer.model            proposer.reasoning_effort   proposer.service_tier
proposer.backend          gepa.pipeline.mode          policy.model
jesterky_workflow.enabled
```

What is deliberately **not** offered: `gepa.max_generations`,
`gepa.minibatch_size`, `gepa.max_total_rollouts`, and the budget caps. They are
all in the schema and all change how much work an arm does, which voids any
wall-clock or cost comparison between arms.

Every arm is rendered and validated through GEPA's own typed document model at
plan time, so a value the engine would reject is a plan-time error rather than a
wasted generation. `run.run_id`, `run.output_dir`, `run.seed`,
`run.correlation`, and `cache.namespace` are derived per trial and refused as
treatments.

### Correlation through GEPA

The envelope reaches the engine three ways, and comes back on the manifest:

| Surface | Field |
| --- | --- |
| `POST /runs` (`GepaServiceRunRequest`) | `correlation`, validated before the contract handshake |
| The run config (`[run.correlation]` in TOML, `RunConfig` in Rust) | carried, never read |
| `GET /runs/{id}` | echoed, alongside the run id the service minted |
| `result_manifest.json`, success **and** failure | `correlation` |
| `run_registry.jsonl` | `correlation` |

The service still mints `run_id` itself. A caller supplying one would be
asserting authority over a namespace it does not own, and resume would then have
two candidate truths about which run a trial is — so `run_id` is not a field of
the envelope at all, and `deny_unknown_fields` refuses one that tries.

A manifest that comes back without the envelope, or with a mangled one, is
refused rather than joined to the wrong arm.

## Adding an executor

Implement `experiment.adapters.base.ExecutorAdapter` — `factor_catalog`,
`provenance`, `environment`, `fixed_projection`, `validate_blocks`,
`subject_for`, `metric_direction`, `trial_derived`, `run_trial` — plus a
`from_spec(spec, **overrides)` classmethod.

An adapter whose executor lives in another repository registers itself from
there:

```python
from synth_optimizers.experiment import register_adapter

register_adapter(EvalsMatrixAdapter)
```

The experiment layer must not grow a dependency on every executor it can drive,
so `evals.matrix` and `gepa.service` plug in this way rather than being imported
across a repository boundary. Registering a name twice is refused: two adapters
answering to one `executor` would make which executor a sealed plan actually ran
a function of import order, which is not a question a plan digest can answer.

Nothing in the experiment layer needs to learn how your executor works, which is
exactly the point.
