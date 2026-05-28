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
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
python -m py_compile src/synth_optimizers/__init__.py src/synth_optimizers/cli.py
uv run --project . --group dev ruff check src
uv run --project . --group dev ty check src
git diff --check
```

Run the cookbook acceptance from the repository root when the local environment
has the package built or installed:

```bash
synth-optimizers gepa run --config cookbooks/optimizers/gepa/banking77_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/tblite_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/code_review_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/crafter_container/gepa.toml
synth-optimizers events compare --left <fresh>/events.normalized.jsonl --right <cached>/events.normalized.jsonl
```

## Publish Gate

No PyPI publish, public release tag, or production promotion is allowed without
the workspace launch checklist and evidence packet.
