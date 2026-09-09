# Release: synth-optimizers

Current candidate: stable `0.2.22` (not yet published), depending on Containers
`0.4.2`. Rust and Python package versions must agree.

Production release acceptance covers the supported optimizer behavior below.
TBLite is eval/testing-only: it is not an installation dependency, production
release gate, or required publication. Its independent lock and blocked research
fixtures are documented in `evals/tblite/README.md`.

- fresh readwrite GEPA run writes result manifest, raw events, normalized events,
  best candidate, candidate registry, frontier, and cache profile
- immediate cached rerun makes no new proposer or rollout external calls
- readonly replay succeeds when fully cached
- readonly replay fails with a typed cache miss when the cache is incomplete
- `events compare` reports parity for normalized original and cached feeds

## Validation

Run from the repository root:

```bash
cargo test --workspace --exclude synth_optimizers_py
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
uv run --locked --group dev pytest tests -q
python -m py_compile src/synth_optimizers/__init__.py src/synth_optimizers/cli.py
uv run --project . --group dev ruff check src
uv run --project . --group dev ty check src
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

Run the cookbook acceptance from the repository root when the local environment
has the package built or installed:

```bash
synth-optimizers gepa run --config cookbooks/optimizers/gepa/banking77_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/code_review_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/crafter_container/gepa.toml
synth-optimizers events compare --left <fresh>/events.normalized.jsonl --right <cached>/events.normalized.jsonl
```

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
