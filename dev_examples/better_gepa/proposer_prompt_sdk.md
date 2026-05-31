# Proposer Prompt SDK

## Requirement

Users should be able to:

1. Inspect the default GEPA proposer guidance document.
2. Use that default unchanged when they want the stock behavior.
3. Override it from Python SDK config or TOML for a run.

Today the guidance lives in Rust as
`rust/crates/synth_gepa/src/prompting_best_practices.md`, embedded via
`include_str!` in `codex_app_server.rs`. There is no SDK or config override path.

## Scope

This requirement covers the **guidance document** layer only:

- `prompting_best_practices.md`
- the content written into each proposer workspace
- the content inlined into the Codex turn prompt

It does **not** replace the full proposer orchestration owned by GEPA:

- workspace layout (`README.md`, `state/*.json`)
- manifest schema (`proposal/PROPOSAL_SCHEMA.md`)
- turn wrapper rules ("propose exactly N candidates", schema version, read order)
- task policy derived from container `task_info`

Those stay in `codex_app_server.rs`.

## Target SDK API

```python
from synth_optimizers.gepa import (
    GepaConfig,
    GepaDefaults,
    ProposerConfig,
    ProposerPromptConfig,
)

# Inspect shipped defaults
defaults = GepaDefaults.current()
print(defaults.proposer.best_practices_md)

md = GepaDefaults.proposer_best_practices()

# Use built-in default
config = GepaConfig(
    container=...,
    taskset=...,
    proposer=ProposerConfig(
        model="gpt-5.4-nano",
        prompt=None,
    ),
)

# Override with custom markdown
config = GepaConfig(
    container=...,
    taskset=...,
    proposer=ProposerConfig(
        prompt=ProposerPromptConfig(
            best_practices="""
# My proposer guidance
- Always add edge-case rules for pytest failures
- Prefer output_description fixes over broad rewrites
""",
        ),
    ),
)

# Override from file
config = GepaConfig(
    ...,
    proposer=ProposerConfig(
        prompt=ProposerPromptConfig.from_path("my_proposer_guidance.md"),
    ),
)

# Start from default, tweak, save
prompt = ProposerPromptConfig.from_defaults()
prompt.best_practices += "\n\n## Domain rule\nAlways handle empty input.\n"
```

### Types

```python
@dataclass
class ProposerPromptConfig:
    best_practices: str | None = None

    @classmethod
    def from_defaults(cls) -> "ProposerPromptConfig": ...

    @classmethod
    def from_path(cls, path: str | Path) -> "ProposerPromptConfig": ...


class GepaDefaults:
    @staticmethod
    def proposer_best_practices() -> str: ...

    @staticmethod
    def proposer_config() -> ProposerConfig: ...

    @staticmethod
    def config_template(*, container_url: str) -> GepaConfig:
        """Sensible defaults for everything except container/taskset."""
        ...
```

### Semantics

- `ProposerConfig.prompt is None` means use the built-in default guidance.
- `ProposerPromptConfig.best_practices is None` also means use the built-in default.
- A non-empty `best_practices` string replaces the shipped markdown for that run.
- `from_path()` loads markdown from disk and sets `best_practices`.

Optional later extension:

```python
ProposerPromptConfig(
    best_practices=...,
    instruction_suffix="Focus on failed crafts and core-path blockers.",
)
```

`instruction_suffix` appends to the turn prompt without replacing workspace
orchestration.

## TOML Projection

```toml
[proposer]
model = "gpt-5.4-nano"
backend = "codex_app_server"

[proposer.prompt]
best_practices_path = "prompts/my_tblite_proposer.md"
```

Rules:

- Omit `[proposer.prompt]` to use the built-in default.
- Specify at most one of `best_practices` or `best_practices_path`.
- Relative paths resolve from the config file directory.

Inline override is allowed but uncommon:

```toml
[proposer.prompt]
best_practices = """
# custom guidance
"""
```

## Rust / Runtime Changes

Add to `ProposerConfig` in `synth_optimizer_platform`:

```rust
pub best_practices: Option<String>,
pub best_practices_path: Option<String>,
```

In `codex_app_server::materialize_workspace()`:

1. Resolve guidance text from config:
   - none set -> `PROMPTING_BEST_PRACTICES`
   - `best_practices_path` -> read file
   - `best_practices` -> use inline string
2. Write resolved text to `prompting_best_practices.md`.
3. Use the same resolved text anywhere `PROMPTING_BEST_PRACTICES` is currently
   inlined or copied into `state/*.json`.

Validation:

- reject both `best_practices` and `best_practices_path`
- fail clearly if `best_practices_path` does not exist

## Python Packaging

Ship the same default markdown as package data in `synth_optimizers` so
`GepaDefaults.proposer_best_practices()` works without calling Rust.

Rust remains the runtime source of truth during execution. Release/build should
keep the Python packaged copy aligned with
`rust/crates/synth_gepa/src/prompting_best_practices.md`.

Suggested helper:

```python
GepaDefaults.proposer_best_practices() -> str
GepaDefaults.write_proposer_best_practices(path: Path) -> None
```

The second helper lets users dump the default to disk, edit it, and point
`ProposerPromptConfig.from_path()` at the edited file.

## Example Use Cases

### TBLite

Start from default guidance, add domain rules for pytest/codegen failures:

```python
prompt = ProposerPromptConfig.from_defaults()
prompt.best_practices += """
## TBLite-specific rules
- Fix signature mismatches before adding heuristics.
- When tests fail on empty input, add explicit empty-input handling rules.
"""
```

### Crafter

Override with guidance that tells the proposer to use ASI fields like failed
crafts and achievement traces when writing repair hints.

### Banking77

Use default guidance unchanged; rely on container `task_info.proposer_hints`
for closed-label classification policy.

## Win Conditions

- SDK exposes `GepaDefaults.proposer_best_practices()`.
- SDK exposes `ProposerPromptConfig` under `ProposerConfig`.
- `GepaConfig` round-trips prompt override through TOML projection.
- Rust proposer workspace uses overridden guidance when configured.
- Default behavior is unchanged when override is omitted.
- At least one dev example documents dumping/editing/overriding the default md.

## Non-Goals

- Replacing the entire Codex turn prompt or workspace contract from user config.
- Moving proposer orchestration into Python.
- Per-task automatic prompt generation without explicit user/container input.
