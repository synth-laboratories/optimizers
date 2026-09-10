# Release: synth-optimizers

Current candidate: stable `0.2.22` (not yet published), depending on Containers
`0.4.3`. Rust and Python package versions must agree — `pyproject.toml` and
`Cargo.toml` `[workspace.package]` both read `0.2.22`.

Publication state, verified against PyPI on 2026-09-09: `synth-optimizers` is
published up to `0.2.16`, so `pip install synth-optimizers==0.2.22` cannot
resolve until this candidate ships. Containers `0.4.3` **is** published —
`pip install synth-containers==0.4.3` resolves from the public index (wheel
sha256 `eaff16ec40b6e2c9a569751f415912178f3aa0ac63396749466ec92e1d734bf5`).
The vendored wheel and lock now use these exact public bytes, verified after
protected publication run `34419184253` succeeded. The release CI separately
installs from the public index without source overrides. Re-run the package
gates after this dependency update before publishing Optimizers.

Production release acceptance covers the supported optimizer behavior below.
TBLite is eval/testing-only: it is not an installation dependency, production
release gate, or required publication. Its independent lock and blocked research
fixtures are documented in `evals/tblite/README.md`.

- fresh readwrite GEPA run writes result manifest, raw events, normalized events,
  best candidate, candidate registry, frontier, and cache profile
- immediate cached rerun makes no new proposer or rollout external calls
- readonly replay succeeds when fully cached
- readonly replay fails with a typed cache miss when the cache is incomplete
- `events compare` performs **byte equality** over `events.normalized.jsonl`
  (`rust/crates/synth_optimizer_platform/src/events.rs`), so it reports parity only
  between feeds of identical execution shape. A cached or readonly replay legitimately
  emits fewer events than the fresh run it replays, and is therefore **expected to report
  a difference** — measured here as 240 fresh versus 211 cached and 211 readonly. The
  entire 29-event gap is accounted for by exactly two event types:
  `optimizer.evaluation_result.received` (64/36/36) and
  `optimizer.limit.estimate_updated` (2/1/1). The 28 surplus evaluation records are the
  `partial: true` worker-pool progress copies emitted once per rollout that was actually
  executed — 28 in the fresh run, 0 in a replay, because a replay executes no rollout.
  The canonical non-partial record is 36 in all three feeds. Forcing byte equality would
  require deleting 28 reward-bearing records of executed rollouts and renumbering
  `sequence_number` across 199 downstream events, which would weaken the gate rather than
  normalize it.

  **The property that does hold, exactly:** the fresh, cached and readonly feeds are
  byte-identical once runtime telemetry is excluded — the
  `optimizer.rollout_queue.updated`, `optimizer.limit.estimate_updated` and
  `runtime.job.completed` events, the `partial: true` progress copy of each evaluation
  result, the worker/queue/forecast fields (`active_workers`, `semaphore_size`,
  `queued_rollouts`, `generated_at`, `sample_count`, `runtime_summary`),
  `sequence_number`, and per-execution rollout ids embedded in child resource refs. Under
  that projection all three feeds are 199 events with the identical SHA-256
  `59390639bb4d0792857fd2d8da01178ba2ae33b46fc8cdbbea63683dd0f65aac`; every one of the 42
  decision events and all 36 candidate evaluation results with their rewards match.
  Verify with `scripts/acceptance/probe_parity_property.py`.

## Validation

Release-owner exception (2026-09-09): remaining live provider-backed GEPA
end-to-end replay/cookbook validation is deferred. Offline evidence is not live
acceptance. Keep package CI, install smoke, and configuration regression gates.

The inherited shell handoff explicitly deferred 226 type diagnostics. The
production-only environment reports 228: the additional two unresolved imports
are `harbor_tblite.cispo` in optional eval paths, whose dependency was deliberately
removed from production. CI runs `scripts/check-type-debt.py` against exact
path/code/message signatures, rejects additions, and allows removals. This is
an explicit existing-debt gate, not a claim that `ty check src` is clean; TBLite
is not installed to hide the optional-import diagnostics. Run the gate script,
not bare `ty check src`: the bare command exits non-zero by design because it
still reports the baselined diagnostics. The baseline signatures live in
`scripts/ty-release-baseline.txt` (228 lines, two of them the
`harbor_tblite.cispo` unresolved imports), and
`.github/workflows/publish-pypi.yml` runs `python3 scripts/check-type-debt.py`,
which tolerates the checker's exit code 1 and fails only on a diagnostic
signature absent from the baseline.

Run from the repository root:

```bash
cargo test --workspace --exclude synth_optimizers_py
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
uv run --locked --group dev pytest tests -q
python -m py_compile src/synth_optimizers/__init__.py src/synth_optimizers/cli.py
uv run --project . --group dev ruff check src
python3 scripts/check-type-debt.py
git diff --check
```

`cargo test` was not previously listed and had stopped compiling: `identities.rs`
used `LeverBundle` without importing it. It is listed now, and passes (137).
`synth_optimizers_py` is excluded because linking the PyO3 extension as a test
binary fails in this environment; that is a harness gap, not a source defect.

The former formatting/file-size conflict is resolved by extracting four intact
test modules into separate files. Both formatting and all four file-size checks
pass. No test assertion was removed and no ceiling was raised; the oversized
production files shrink to 3,409, 22,094, 6,152, and 2,935 lines respectively.

## Cookbook acceptance

Cookbooks are **not in this repository** — there is no `cookbooks/` directory
here. They live in the separate public repo
[`synth-laboratories/synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public)
(public, default branch `main`). Clone it beside this checkout:

```bash
git clone https://github.com/synth-laboratories/synth-cookbooks-public.git
```

Every cookbook `gepa.toml` sets `cwd = ".."` and `output_dir = "../runs"`, and
`SynthOptimizerConfig::from_toml_file` absolutizes both against the config file's
own directory (`rust/crates/synth_optimizer_platform/src/config.rs:399`), not the
process working directory. So each run is issued **from its container directory**
— the convention the cookbooks' own `run_fresh_gepa.sh` follows with
`cd "$SCRIPT_DIR"`, and the directory `GepaConfig.write_toml()` drops its derived
`gepa.<run_id>.sdk.toml` into, so it must be writable. A root-relative `--config`
from either repository root resolves `cwd` and `output_dir` identically; the
earlier claim that it breaks container launch was checked here and does not hold.
Requires the package built or installed (see "Local development" in `README.md`)
plus the keys each recipe declares — Banking77 policy on `OPENAI_API_KEY`,
HotpotQA policy on `OPENROUTER_API_KEY`, Crafter policy on `GEMINI_API_KEY`, and
the Codex proposer on `OPENAI_API_KEY` in all three.

```bash
cd synth-cookbooks-public/cookbooks/optimizers/gepa/banking77_container
synth-optimizers gepa run --config gepa.toml

cd ../hotpotqa_container
synth-optimizers gepa run --config gepa.toml

cd ../crafter_container
synth-optimizers gepa run --config gepa.toml

synth-optimizers events compare --left <fresh>/events.normalized.jsonl --right <cached>/events.normalized.jsonl
```

**Flagged substitution, not a rename.** This list previously named
`cookbooks/optimizers/gepa/code_review_container/gepa.toml`. That directory
exists on no ref of the public cookbooks repo — checked against all 19 branch
and tag refs, including `main`, `dev`, and `v0.7` — so the command could never
have run for an outside reader. `hotpotqa_container` takes its place, keeping the
demonstrated count at three, but coverage is lost: per `ACCEPTANCE.md`, the
code-review fixture was the one cookbook that "preserves the private
reviewer-guidance levers", and no public cookbook replaces that lever surface.
Treat it as an open acceptance gap. The public repo also ships
`minigrid_container`, `healthbench_groq`, and `tblite_container`;
`tblite_container` is deliberately excluded because TBLite is eval/testing-only
and not a release gate.

**Blocker, verified 2026-09-09.** As the cookbook repo currently publishes them,
none of these configs loads under `0.2.22`. `GepaTomlDocument` ignores unknown
sections, and the
cookbook configs still declare the legacy `[dataset]`/`train_seeds` selection
with no `[taskset]` or `[gepa.task_pools]`, so `GepaConfig.validate()` raises
`ValueError: GepaTaskPools.pareto must not be empty` before any container or
provider call. Reproduced for all five container configs on `main` and for
Banking77/HotpotQA/Crafter on `v0.7`. The cookbook repo needs `[taskset]` and
`[gepa.task_pools]` blocks (shape shown in `README.md`'s quickstart) before this
acceptance list is executable; all five `run_fresh_gepa.sh` launchers there also
still pin `synth-optimizers==0.2.0`.

## Changelog

- Update `changelog.log` in the same change that updates package version or release docs.
- Organize entries by day: `## YYYY-MM-DD`.
- Keep the file terse: about 20 total lines for the current daily-dev window.
- Use bullets only; no paragraphs, migration snippets, or install code blocks.
- Prefer shipped user-facing changes over implementation narration.
- Link merged PRs where available, for example `[PR #3](https://github.com/synth-laboratories/optimizers/pull/3)`.
- Include the PyPI version in one bullet when a package was published.
- Keep unreleased or blocked work explicit and short.

## Publish Gate

No PyPI publish, public release tag, or production promotion is allowed without
the workspace launch checklist and evidence packet.
