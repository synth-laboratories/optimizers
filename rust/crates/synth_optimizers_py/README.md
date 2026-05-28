# synth_optimizers_py

PyO3 crate for the public Python and CLI surface.

This crate adapts Rust result and error types into Python classes. It should not
hold optimizer logic. It is the narrow bridge between Python users and the Rust
workspace.

## Responsibilities

- Expose `_synth_optimizers` native module.
- Implement `GepaRun.from_toml(path)` and `GepaRun.execute()`.
- Return `GepaRunResult` with stable Python fields.
- Convert typed Rust errors to Python exceptions with error codes.
- Provide CLI handlers for:
  - `synth-optimizers gepa run --config gepa.toml`
  - `synth-optimizers events replay --events events.jsonl`
  - `synth-optimizers events compare --left a.jsonl --right b.jsonl`

## Python Classes

- `GepaRun`: immutable-ish handle to a parsed config path.
- `GepaRunResult`: result fields mirrored from Rust:
  - `best_candidate`
  - `manifest_path`
  - `event_feed_path`
  - `normalized_event_feed_path`
  - `cache_profile_path`
  - `candidate_registry_path`
  - `frontier_path`
  - `run_registry_path`
  - `artifact_refs`
  - `cost_usd`
  - `usage`
- `SynthOptimizerError`: base exception with stable `error_code`.
- `ConfigError`: config or TOML validation failure.
- `CacheMissError`: readonly replay miss.
- `ContainerContractError`: container boundary or GEPA contract failure.
- `ProposerError`: proposer backend failure.
- `EventCompareError`: normalized event feeds differ.

## Boundary Rules

- Python wrapper imports only the native module.
- CLI parsing can be in Python for ergonomics, but execution calls Rust.
- No Python implementation fallback when the native module is missing in normal
  package use. Import failure should be explicit.
- No private module import paths.

## Packaging

The package uses maturin. The wheel contains:

- native extension module
- thin Python wrapper under `src/synth_optimizers/`
- console entry point `synth-optimizers`

The package remains `0.1.0a0` or equivalent until the release checklist and tag
step. Banking77, TBLite, and code-review acceptance is recorded in
`packages/synth-optimizers/ACCEPTANCE.md`.
