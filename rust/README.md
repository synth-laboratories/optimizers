# Rust Workspace

The Rust workspace is the implementation home for public optimizer behavior.
Python is an API shell; Rust owns state, IO contracts, cache integrity, event
feeds, manifests, and algorithm transitions.

## Crates

```text
rust/crates/
  synth_optimizer_platform/
  synth_gepa/
  synth_optimizers_py/
```

## Dependency Direction

```text
synth_optimizers_py
        |
        v
    synth_gepa
        |
        v
synth_optimizer_platform
```

The platform crate must not depend on GEPA. It should be reusable by future
public optimizers once their contracts are ready.

## Cross-Crate Abstractions

- `SynthOptimizerConfig`: loaded from frozen public TOML sections.
- `ContainerClient`: typed HTTP access to `synth-containers`.
- `RequestCache`: SQLite request/response cache with strict readonly misses.
- `EventWriter`: raw JSONL feed plus normalized feed generation.
- `ArtifactPaths`: stable run workspace paths and public artifact refs.
- `PromptProgram`: public prompt-program contract from `/program`.
- `GepaEngine`: GEPA state machine over platform primitives.
- `GepaRunResult`: final Python and CLI result object.

## Non-Goals

- No private Python optimizer imports.
- No provider-specific prefix cache.
- No hidden service dependency.
- No MIPROv2 compatibility layer.
- No direct dataset loader in the optimizer when the container can provide rows.

## Runtime Flow

```text
TOML config
  -> platform config loader
  -> container discovery and GEPA contract check
  -> prompt program and dataset row fetch
  -> GEPA seed candidate registration
  -> minibatch rollout jobs through /rollout
  -> proposer requests through configured proposer backend
  -> full-train and heldout evaluation
  -> event normalization, cache profile, result manifest
```

Every external boundary must either be cached or explicitly uncached. In
`readonly`, any missing policy, proposer, or rollout cache entry is a typed
error.
