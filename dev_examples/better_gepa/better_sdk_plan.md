# Better SDK Implementation Plan

## Goal

Implement the paired Better Containers and Better GEPA SDK work so examples can
move away from shell-mutated TOML and GEPA-owned container launch.

Target authoring flow:

```python
with container.serve() as handle:
    config = GepaConfig(
        container=handle.connection(),
        taskset=...,
        program=None,
        objectives=None,
        policy=None,
        proposer=...,
        budgets=...,
        pipeline=...,
    )
    result = OptimizerRun(config).execute()
```

`synth-containers` owns container authoring, serving, lifecycle, and connection
creation. `synth-optimizers` owns optimizer config, execution, GEPA-specific
settings, compatibility validation, and results.

## Win Conditions

This work is done when the branch pair proves the new SDK path end to end:

- Banking77, TBLite, and Crafter can run from SDK-authored examples without
  shell-mutated TOML as the primary path.
- `synth-containers` provides the container authoring/lifecycle surface:
  `Container`, `Container.serve()`, `ContainerHandle`, and
  `ContainerConnection`.
- `synth-optimizers` provides the optimizer SDK surface: `OptimizerRun`,
  `GepaConfig`, typed config sections, `GepaConfig.from_toml()`, and a
  compatibility shim for existing `GepaRun.from_toml()` callers.
- GEPA accepts only `ContainerConnection` for optimizer execution; local process
  launch and teardown are owned by `synth-containers`.
- `program=None`, `objectives=None`, and `policy=None` mean
  container-provided/discoverable behavior, with clear validation when the
  container does not advertise enough capability.
- Non-blocking rollout is first-class: SDK examples use async submit/poll by
  default, while sync remains available for focused smoke/debug runs.
- Multi-objective runs can express objective names, directions, selection
  objective, and protected objectives from Python SDK config.
- Actionable side info flows from terminal rollout payloads into GEPA evidence
  and proposer context without being confused with objective values.
- Both repos document local dev installation with `uv`, branch expectations,
  package versions, and validation examples.
- The focused Python SDK gates pass or any skipped gate is explicitly documented
  with reason and risk.

## Implementation Status Checklist

Status key: `[x] Done`, `[~] Partial`, `[ ] TODO`.

- [x] Branches: `better-sdk` is checked out in both repos.
- [x] Dev versioning and README install: dev versions and editable `uv` notes
  exist; evidence recorded in `changelog.log` (2026-05-31).
- [x] `OptimizerRun` / `GepaConfig` SDK: Python SDK config projects to TOML and
  executes the native Rust GEPA runner.
- [x] Container lifecycle SDK: `Container`, `serve()`, `ContainerHandle`,
  `ContainerConnection`, and `ContainerRunner` exist in `synth-containers`.
- [x] Banking77 golden example: container-owned policy/program/objectives,
  `OptimizerRun(GepaConfig(...))`, task IDs, and async rollout.
- [x] TBLite SDK example: SDK runner with protected correctness objective,
  completion-time objective, typed rollout result helpers, and ASI.
- [x] Crafter SDK example: typed `GepaConfig`, in-process `Container`, two
  objectives, ASI, and proposer best-practices override.
- [x] MiniGrid SDK example: typed profile-driven SDK runner, in-process
  `Container`, task IDs, async rollout, and objective scores.
- [x] Shell runners to SDK: Banking77, TBLite, Crafter, and MiniGrid runners now
  dispatch to Python SDK scripts instead of shell-mutating TOML.
- [x] GEPA accepts `ContainerConnection` in SDK examples; container lifecycle is
  outside optimizer config.
- [x] Container-owned policy/program/objectives: SDK examples use
  `policy=None` and `program=None` where the container owns those surfaces.
- [x] Async rollout first-class: SDK-authored container default is async; SDK
  examples configure async transport explicitly.
- [x] Multi-objective SDK: `ObjectiveConfig` carries objective names,
  directions, protected objectives, and selection objective.
- [x] ASI: containers can emit typed or arbitrary JSON ASI; Rust persists it
  through sensor/evidence paths and proposer workspace files.
- [x] Proposer prompt override: SDK exposes `GepaDefaults` and
  `ProposerPromptConfig`; TOML carries `[proposer.prompt]`; Rust resolves the
  same guidance for workspace files and turn prompts.
- [x] Planning docs: `dev_examples/better_gepa/*` committed with the branch.
- [x] Pre-merge validation: gates pass; example evidence recorded in
  `changelog.log` (2026-05-31). TBLite proposer failure and MiniGrid partial
  run documented; no end-to-end re-run required for merge packet.
- [x] Commit/merge prep: both repos committed on `better-sdk`; merge packet
  and changelogs added 2026-05-31.

## Branches

Create matching branches in both repos:

```bash
cd /Users/joshpurtell/Documents/GitHub/containers
git switch -c better-sdk

cd /Users/joshpurtell/Documents/GitHub/optimizers
git switch -c better-sdk
```

Keep the branches aligned. If one repo needs a compatibility shim for the other,
land it on the matching `better-sdk` branch rather than hiding it in examples.

## Versioning

Both packages should publish/install as dev versions while this work is in
progress.

Suggested scheme:

- `synth-containers`: increment to the next `.devYYYYMMDD` or equivalent local
  dev version.
- `synth-optimizers`: increment to the next `.devYYYYMMDD` or equivalent local
  dev version.

Record the exact dev version in each repo README so examples can be reproduced.
If the package metadata does not already support daily dev versions cleanly, add
the minimal versioning path rather than inventing per-example install hacks.

## Dev Install Instructions

Add README instructions in both repos for local `uv` installs from sibling
checkouts.

Expected local install shape:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
uv sync --group dev
uv pip install -e /Users/joshpurtell/Documents/GitHub/containers
uv pip install -e /Users/joshpurtell/Documents/GitHub/optimizers
```

For examples that run from their own project directories, prefer explicit local
editable installs or `uv run --project /Users/joshpurtell/Documents/GitHub/optimizers`
so they use the dev branch, not a released wheel.

README updates should answer:

- How to install both local dev packages with `uv`.
- How to verify the installed import paths and versions.
- Which branch pair is expected (`better-sdk` in both repos).
- Which examples are the validation set.

## Style and Python SDK Gates

Follow the workspace Synth Style guidance while implementing the SDK work. The
intended source of truth is `backend/specifications/tanha/references/synthstyle.md`
in the active Synth workspace, with historical copies under `specifications/`
where present. README/API docs should use the same naming discipline as the code:
generic container concepts in `synth-containers`, optimizer-specific GEPA concepts
in `synth-optimizers`.

Run the Python SDK gates in both repos before merge:

```bash
cd /Users/joshpurtell/Documents/GitHub/containers
uv run --group dev ruff format --check src
uv run --group dev ruff check src
uv run --group dev ty check src

cd /Users/joshpurtell/Documents/GitHub/optimizers
uv run --group dev ruff format --check src
uv run --group dev ruff check src
uv run --group dev ty check src
```

## Better Containers Work

Implement in `containers`:

- `Container` authoring facade with decorators for task info, taskset,
  taskset tasks, rollout, and beta program.
- `Container.fastapi()` transport builder.
- `Container.serve()` lifecycle method.
- `ContainerRunner` for lower-level serving from a `Container`, FastAPI app,
  command, or existing URL.
- `ContainerHandle` with `url`, `connection()`, `health()`, `logs()`, `down()`,
  and context manager cleanup.
- `ContainerConnection` as the URL-only optimizer-facing value.
- Generic capability metadata and route hints without GEPA-specific names.
- Async-capable rollout support through `submission_mode="async"` and
  `/rollouts/{rollout_id}` polling routes.
- Treat non-blocking rollout as first-class SDK behavior. SDK-authored
  containers should support async submit + status polling by default unless an
  author explicitly opts out.
- Program surface under a generic beta route, not GEPA-specific metadata.

Do not put GEPA concepts in `containers`. The container should expose generic
task, rollout, reward/objective, policy, and program capabilities. Optimizers
decide whether that is enough.

## Better GEPA Work

Implement in `optimizers`:

- `OptimizerRun` generic runner.
- `OptimizerConfig` protocol/base type.
- `synth_optimizers.gepa.GepaConfig`.
- `GepaConfig.from_toml()`.
- Typed config sections:
  - `RunSettings`
  - `TasksetSelection`
  - `ProposerConfig`
  - `ProposerPromptConfig`
  - `PolicyConfig | None`
  - `ObjectiveConfig | None`
  - `BudgetConfig`
  - `GepaBudgetConfig`
  - `GepaPipeline`
  - `CacheConfig`
  - `OutputConfig`
- `PolicyType` enum:
  - `dag`
  - `react`
  - `codex`
- First-class non-blocking rollout transport:
  - SDK configs should expose rollout transport as a clear field, not bury it in
    legacy TOML.
  - Preferred default for SDK-authored examples should be async submit/poll.
  - Sync should remain available for smoke/debugging profiles.
- Container-owned policy support:
  - `policy=None` means do not inject policy override.
  - Validate that the container advertises policy readiness.
- Container-provided program/objectives support:
  - `program=None` means discover from container.
  - `objectives=None` means discover from container or fall back to reward.
- Actionable side info support:
  - Treat ASI as rollout evidence, not an objective.
  - Read top-level `actionable_side_info` from the terminal rollout payload.
  - Preserve ASI through sensor frames, evaluation cache records, and proposer
    workspace evidence.
  - Add Python SDK affordances so containers can return typed or arbitrary JSON
    ASI without coupling `synth-containers` to GEPA.
- Legacy TOML projection for Rust execution until native typed config execution
  is available.
- Backwards-compatible `GepaRun.from_toml()` shim during migration, but stop
  documenting it as the primary SDK API.
- Proposer prompt override:
  - `GepaDefaults.proposer_best_practices()` exposes the shipped guidance.
  - `ProposerPromptConfig` supports inline guidance and path-backed TOML.
  - Runtime writes the resolved guidance into each proposer workspace and uses
    it in the Codex turn prompt.

## Actionable Side Info

Actionable side info is the narrative/structured evidence that tells the
optimizer why a rollout received its objective values and what behavior might be
changed. It should not participate directly in acceptance or Pareto comparison
unless it is promoted into an explicit objective. For Crafter, examples include
failed crafts, repeated achievement events, inventory/blocker snapshots, or a
short "next blocker" field. For TBLite, examples include the final generated
diff, pytest output tails, failing test names, and verifier notes.

Target rollout shape:

```json
{
  "reward_info": {
    "outcome_reward": 4.0,
    "details": {
      "objective": "overall_achievement_count"
    }
  },
  "actionable_side_info": {
    "failed_crafts": [
      {"recipe": "wooden_pickaxe", "missing": ["crafting_table"]}
    ],
    "achievements_all": ["collect_wood", "collect_wood", "place_table"],
    "achievements_unique": ["collect_wood", "place_table"],
    "last_blocker": "crafted table but did not return to stone/coal path"
  }
}
```

TBLite should use the same split:

```json
{
  "reward_info": {
    "outcome_reward": 0.75,
    "details": {
      "objective": "correctness",
      "pytest_passed": 3,
      "pytest_total": 4,
      "completion_time_seconds": 2.41
    }
  },
  "objective_scores": [
    {
      "objective": "correctness",
      "value": 0.75,
      "source": "pytest"
    },
    {
      "objective": "completion_time_seconds",
      "value": 2.41,
      "source": "container.runtime"
    }
  ],
  "actionable_side_info": {
    "final_diff": "--- a/solution.py\n+++ b/solution.py\n...",
    "pytest_stdout_tail": "1 failed, 3 passed in 2.41s",
    "failing_tests": ["test_empty_input"],
    "implementation_notes": "solution handles normal inputs but misses empty input guard"
  }
}
```

For TBLite, `correctness` should remain the primary/selection objective.
`completion_time_seconds` can be a secondary objective with
`lower_is_better`/`minimize` direction, but it should not reward fast broken
solutions. Use objective acceptance rules or protected objectives so a candidate
cannot trade away correctness just to finish faster. The final diff is ASI:
useful for proposer repair, not a selection metric by itself.

This can come from async rollout naturally: the container accepts the rollout,
GEPA polls `/rollouts/{rollout_id}` until terminal, and the final payload carries
`reward_info`, `summary`, `trace`, and `actionable_side_info` together. No
GEPA-only ASI route is needed.

The Rust path already has the important internal hook: top-level
`actionable_side_info` is read into `SensorFrame.actionable_side_info`, persisted
with evaluation cache records, and exposed in proposer workspace evidence such
as `state/rollouts.json`, `state/evidence_frames.json`, and
`state/reflective_frames.json`. The SDK work should make that explicit and
ergonomic, then add compact ASI summaries to proposer-facing failure summaries if
the nested evidence is too hidden for proposer agents to use reliably.

## Example Migrations

### Banking77

Use Banking77 as the golden paired SDK example.

Move from:

- hand-written FastAPI routes
- `GepaRun.from_toml()`
- shell/TOML mutation
- GEPA-owned `[container.command]`

To:

- `Container(...)` facade
- `container.serve()`
- `GepaConfig(...)`
- `OptimizerRun(config).execute()`
- `policy=None`
- `program=None`
- `objectives=None`

Validation command:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash "dev_examples/banking77/run_fresh_gepa.sh"
```

After migration, replace the shell runner with a Python SDK runner and document
the new command in the Banking77 example README.

Banking77 should validate first-class non-blocking rollout:

- `Container` serves `/rollout` and `/rollouts/{rollout_id}` routes.
- GEPA uses async submit/poll transport by default.
- Keep one sync smoke command/profile for debugging container failures.

### TBLite

TBLite should prove fixed DAG-style policy ownership by the container:

- candidate prompt + task spec
- model call
- pytest verifier
- correctness objective from test pass fraction
- optional completion-time objective from measured rollout duration
- actionable side info containing final diff, pytest output tail, failing tests,
  and concise verifier notes

Validation command:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash "dev_examples/tblite/run_fresh_gepa.sh"
```

After migration, the config should be SDK-authored and GEPA should receive only
a `ContainerConnection`.

TBLite should also validate non-blocking rollout because pytest verification can
be slow and benefits from submit/poll semantics. Sync mode remains useful as a
local smoke/debug profile.

### Crafter

Crafter validates typed SDK construction for a non-classification task with
container-owned policy, ASI, and two objectives: achievement unlock rate and
turn count. It also exercises proposer best-practices override by starting from
`ProposerPromptConfig.from_defaults()` and appending Crafter-specific guidance.

Validation command:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash "dev_examples/crafter/run_fresh_gepa.sh"
```

If Crafter requires `GEMINI_API_KEY`, load it from the local env source before
running. Do not print secrets.

### MiniGrid

MiniGrid validates the SDK path for a real gymnasium environment with task IDs,
async rollout, and objective scores for task success and episode steps.

Validation command:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash "dev_examples/minigrid/run_fresh_gepa.sh"
```

## Pre-Merge Validation

Run a focused validation pass before merging `better-sdk`:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash "dev_examples/banking77/run_fresh_gepa.sh"
bash "dev_examples/tblite/run_fresh_gepa.sh"
bash "dev_examples/crafter/run_fresh_gepa.sh"
bash "dev_examples/minigrid/run_fresh_gepa.sh"
```

For each run, capture:

- command
- run id
- artifact path
- best train/heldout or reward summary
- whether the container was served via the new lifecycle API
- whether GEPA used async/non-blocking rollout transport
- package versions/import paths used

Add those results to the PR description or merge notes.

### Validation evidence (recorded 2026-05-31)

Packages: `synth-containers==0.2.0.dev20260531`, `synth-optimizers==0.2.0.dev20260531`
(editable local checkouts, branch `better-sdk`).

| Example | Command | Run ID | Outcome | Train / heldout | Lifecycle | Transport |
|---------|---------|--------|---------|-----------------|-----------|-----------|
| Banking77 | `bash dev_examples/banking77/run_fresh_gepa.sh` | `banking77_dev_20260531175216_synth` | Pass | 0.583 / 0.500 | `Container` SDK | async |
| Crafter | `bash dev_examples/crafter/run_fresh_gepa.sh` | `crafter_gepa_sdk_20260531043041` | Pass | 0.667 / 0.333 | `Container.serve()` | async |
| TBLite | `bash dev_examples/tblite/run_fresh_gepa.sh` | `tblite_gepa_public_0c961acf` | Fail (proposer) | seed OK | SDK runner | async |
| MiniGrid | `bash dev_examples/minigrid/run_fresh_gepa.sh` | `minigrid_deepseek_v4_flash_c9dcf209` | Partial | best_train 0.784 | `Container` | sync profile |

TBLite: `failure_patterns is empty` in Codex proposer — not an SDK projection bug.
MiniGrid: seed eval + proposer start validated; full generational manifest not captured.

Python gates: optimizers `src/` all pass; containers SDK delta passes (repo-wide
format debt on 20 legacy files documented in `containers/changelog.log`).

### Synth Style scan (2026-05-31)

- Boundary naming: pass — GEPA concepts in optimizers, generic container nouns in containers.
- `GepaConfig.container` accepts `ContainerConnection` only: pass.
- Interconnect sparsity: pass — lifecycle normalized before optimizer config.
- Docs drift (follow-up): README Quickstart still TOML-first; validation list incomplete.

### Customer-facing impact (2026-05-31)

**Additive:** `OptimizerRun`, `GepaConfig`, typed sections, `GepaDefaults` /
`ProposerPromptConfig`; sibling `Container` lifecycle SDK.

**Compatible:** `GepaRun.from_toml()`, CLI, raw TOML + Rust `[container.command]`.

**Behavioral:** SDK containers default async rollout; optimizers preflight
container-owned policy/program when unset.

**Non-breaking** for existing TOML callers; SDK is the new preferred authoring path.

## Merge Plan

1. Open `better-sdk` in `containers`.
2. Implement lifecycle and authoring facade there first.
3. Add README dev install instructions and dev version bump.
4. Open `better-sdk` in `optimizers`.
5. Implement GEPA SDK config and runner abstractions.
6. Add README dev install instructions and dev version bump.
7. Migrate Banking77 first.
8. Migrate TBLite second.
9. Migrate Crafter typed SDK path third.
10. Migrate MiniGrid typed SDK path fourth.
11. Implement proposer prompt override.
12. Run the pre-merge validation commands.
13. Merge `containers` first if `optimizers` depends on the new container SDK.
14. Merge `optimizers` after validation against the merged or pinned containers
    dev version.

## Risks

- Rust still owns the execution path, so Python SDK config must project cleanly
  to the current Rust config until native structured execution exists.
- Existing examples may still rely on sync `/rollout`; async/non-blocking
  rollout should become the first-class SDK path, but keep sync smoke paths until
  async behavior is stable.
- Container-owned policy requires contract metadata that current examples may
  not yet advertise.
- Avoid introducing GEPA-specific concepts into `containers` while adding program
  and policy metadata.
