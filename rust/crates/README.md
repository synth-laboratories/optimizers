# Crates

The crate split is the main architecture boundary for public optimizer code.
Keep the dependency graph one-way and keep public contracts in the lowest crate
that can own them.

## Dependency Graph

```text
synth_optimizers_py
  depends on synth_gepa
  depends on synth_optimizer_platform

synth_gepa
  depends on synth_optimizer_platform

synth_optimizer_platform
  depends on external libraries only
```

## Ownership Matrix

| Concern | Owning Crate | Notes |
| --- | --- | --- |
| TOML shape | `synth_optimizer_platform` | Public config sections stay stable. |
| HTTP container calls | `synth_optimizer_platform` | Includes contract discovery and route errors. |
| Cache | `synth_optimizer_platform` | Full request/response replay cache. |
| Events | `synth_optimizer_platform` | Raw and normalized JSONL feeds. |
| Result manifests | `synth_optimizer_platform` | Paths, checksums, usage, cost. |
| Prompt candidates | `synth_optimizer_platform` | Generic prompt-program payload shape. |
| GEPA state | `synth_gepa` | Candidate registry, frontier, scheduler. |
| GEPA proposer schema | `synth_gepa` | Backend-agnostic request/response structs. |
| Codex proposer execution | Platform boundary, orchestrated by `synth_gepa` | Local process first. |
| Python API | `synth_optimizers_py` | PyO3 object conversion only. |
| CLI | `synth_optimizers_py` plus Python wrapper | Execution still calls Rust. |

## Review Rule

When adding a new type, ask which boundary owns its lifecycle:

- external IO or artifacts: platform
- algorithm state or search semantics: GEPA
- Python object conversion: PyO3 crate

If the answer is "all of them", the type is too blended and should be split.
