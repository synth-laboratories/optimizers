# Better GEPA Plan

## Goal

Make GEPA configuration easier to author, inspect, and migrate by adding SDK
abstractions over the current raw configuration shapes. The same conceptual
configuration should be clear whether it is authored through Python SDK objects
or derived from TOML.

## Configuration Abstractions

Add first-class SDK abstractions for the main GEPA configuration surfaces:

- Optimizer settings and budgets.
- Candidate program and mutable field declarations.
- Taskset selection and task identifiers.
- Container launch and route compatibility settings.
- Evaluation splits, scoring objectives, and reporting outputs.

The SDK layer should preserve the typed intent of each section instead of
treating the config as one loosely typed dictionary.

## SDK-Authored Config

SDK-authored configuration should make the common path concise:

- Use named objects for tasksets, programs, objectives, budgets, and container
  connections.
- Validate required fields before launching a run.
- Generate the same normalized internal config used by TOML-derived runs.
- Keep optimizer-specific concepts in GEPA-facing SDK names rather than leaking
  container implementation details.

## TOML-Derived Config

TOML-derived configuration should remain supported, but its projection should be
explicit:

- Parse TOML into the same typed configuration objects used by the SDK path.
- Keep section-level ownership clear, for example optimizer, taskset, program,
  container, scoring, and output/reporting sections.
- Produce useful validation errors that point back to the TOML section and key.
- Avoid one-off TOML-only behavior that cannot be represented by the SDK.

## Evidence From Banking77 Dev Example

The current Banking77 GEPA example shows several places where better SDK
abstractions would clarify intent:

- The example writes `gepa.toml` with a Python f-string. That makes config shape,
  escaping, defaults, and validation implicit instead of typed.
- The same ideas are repeated across Python constants, environment variables,
  TOML sections, route payloads, and comparison code: task ids, dataset splits,
  train/heldout ids, policy model, proposer model, budget, cache path, and
  container launch settings.
- Container launch is encoded as a raw command array with env vars inline. The
  SDK should have a typed container connection/launch abstraction that can still
  render to TOML when needed.
- Program setup is split between `/program`, `[candidate]`, and
  `[seed_candidate]`. SDK config should make mutable fields, seed candidate
  payloads, and target modules one coherent program object.
- Task selection is still expressed as `[dataset] train_seeds` and
  `heldout_seeds`. The next SDK surface should use taskset/task/task_id naming
  and derive the old TOML only as a compatibility projection while needed.
- Result reporting reaches into loosely typed result dictionaries for
  `best_candidate`, train reward, heldout reward, manifest path, and cost. A
  typed run result would make downstream comparison code less brittle.
- The example mixes four concerns in one script: task implementation, HTTP
  container, GEPA config generation, and run/comparison CLI. That is useful as a
  smoke example, but it is also evidence that the SDK should make these sections
  separately authorable.
- The comparison against `gepa-ai` has to manually map equivalent concepts:
  trainset/valset, metric calls versus rollouts, reflection model versus
  proposer model, and best-program result shape. SDK abstractions should make
  these mappings explicit enough to compare backends without bespoke glue.

## Design Intent Notes

The goal is for GEPA to work with any code that fulfills the basic container
requirements, regardless of implementation language. The Python SDK should be a
clear authoring path, not the only runtime path:

- Python users should be able to configure GEPA and containers with typed SDK
  objects.
- Non-Python containers should still work through the same normalized contract
  when they expose the required HTTP surface.
- TOML should remain a portable representation, but SDK and service APIs should
  share the same conceptual sections and validation model.

## Open Questions

- Which GEPA config sections should be stable public SDK objects first?
- Should TOML parsing live behind the SDK objects or in a separate adapter layer?
- What normalized config dump should be emitted for debugging and reproducible
  runs?
