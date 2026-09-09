# Release: synth-optimizers

Current status: prerelease implementation for the public GEPA vertical slice.

Do not tag or publish `0.1.0` until the Banking77, TBLite, code-review, and
Crafter acceptance packet is complete:

- fresh readwrite GEPA run writes result manifest, raw events, normalized events,
  best candidate, candidate registry, frontier, and cache profile
- immediate cached rerun makes no new proposer or rollout external calls
- readonly replay succeeds when fully cached
- readonly replay fails with a typed cache miss when the cache is incomplete
- `events compare` reports parity for normalized original and cached feeds

## Validation

Run from `packages/synth-optimizers/`:

```bash
cargo test --workspace --exclude synth_optimizers_py
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
python -m py_compile src/synth_optimizers/__init__.py src/synth_optimizers/cli.py
uv run --project . --group dev ruff check src
uv run --project . --group dev ty check src
git diff --check
```

`cargo test` was not previously listed and had stopped compiling: `identities.rs`
used `LeverBundle` without importing it. It is listed now, and passes (137).
`synth_optimizers_py` is excluded because linking the PyO3 extension as a test
binary fails in this environment; that is a harness gap, not a source defect.

**`cargo fmt --check` and the file-size ratchet currently disagree.** Formatting
`synth_gepa/src/{codex_app_server,lib,service}.rs` and
`synth_optimizer_platform/src/config.rs` adds 12, 3, 6 and 6 lines respectively,
and `tests/file_size_cap.rs` allows those four files only to shrink. Running the
formatter therefore turns a green test red. These four files are left unformatted
so the ratchet holds; every other file is formatted. Resolving this needs a
decision that belongs to a review — split the files, or reformat and re-cut the
ceilings — and it must not be settled by quietly raising them.

Run the cookbook acceptance from the repository root when the local environment
has the package built or installed:

```bash
synth-optimizers gepa run --config cookbooks/optimizers/gepa/banking77_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/tblite_container/gepa.toml
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
