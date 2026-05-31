# Multi-Objective SDK Support

## Goal

Expose GEPA's existing multi-objective runtime support through the Python SDK
with objective-based language.

This is not about building new multi-objective GEPA logic from scratch. The Rust
core already has the foundations. The SDK work is to make that logic easy to
configure, validate, serialize, and explain.

## Current Rust Capability

Rust already supports:

- `selection_objective`
- `objective_keys`
- `objective_directions`
- `objective_acceptance`
- `frontier_type`
- objective sets
- objective value vectors
- Pareto comparison records
- objective-aware acceptance criteria

Existing acceptance criteria include:

- `primary_improvement`
- `primary_or_objective`
- `any_objective_improved`
- `protected_objective_guard`

Existing frontier modes include:

- `per_example`
- `per_objective`
- `per_example_objective`

## Current SDK Gap

The Python SDK currently exposes only the thin run path:

```python
from synth_optimizers import GepaRun

GepaRun.from_toml("gepa.toml").execute()
```

Multi-objective behavior is only practically available by hand-authoring TOML:

```toml
[gepa]
selection_objective = "core_path_score"
objective_keys = ["overall_crafter_score", "core_path_score"]
frontier_type = "per_example_objective"
acceptance_criterion = "protected_objective_guard"

[gepa.objective_directions]
overall_crafter_score = "maximize"
core_path_score = "maximize"

[gepa.objective_acceptance]
protected_objectives = ["overall_crafter_score"]
objective_regression_tolerance = 0.02
```

The SDK needs typed objects for this surface.

## Target Python API

```python
from synth_optimizers.gepa import (
    GepaConfig,
    Objective,
    ObjectiveAcceptance,
    ObjectiveConfig,
)

config = GepaConfig(
    ...,
    objectives=ObjectiveConfig(
        selection_objective="core_path_score",
        frontier_type="per_example_objective",
        objectives=[
            Objective.reward("overall_crafter_score", direction="maximize"),
            Objective.reward("core_path_score", direction="maximize"),
        ],
        acceptance=ObjectiveAcceptance(
            criterion="protected_objective_guard",
            protected_objectives=["overall_crafter_score"],
            objective_regression_tolerance=0.02,
        ),
    ),
)
```

If `objectives=None`, GEPA should:

1. Discover objective metadata from the container contract when available.
2. Fall back to the default rollout reward objective when not available.

## Crafter Example

Crafter can expose two objectives:

- `overall_crafter_score`: number of unique achievements.
- `core_path_score`: progress on the core path:
  - collect wood
  - crafting table
  - wooden pickaxe
  - collect stone
  - stone pickaxe
  - collect coal
  - furnace
  - collect iron
  - iron pickaxe
  - diamond

The container rollout record should include objective values:

```json
{
  "rollout_id": "rollout_123",
  "status": "completed",
  "reward_info": {
    "outcome_reward": 8.0,
    "details": {
      "unique_achievements": 8,
      "core_path_achievements": 5
    }
  },
  "objective_values": {
    "overall_crafter_score": 8.0,
    "core_path_score": 5.0
  },
  "summary": {
    "core_path": {
      "completed": [
        "collect_wood",
        "crafting_table",
        "wooden_pickaxe"
      ],
      "remaining": [
        "collect_stone",
        "stone_pickaxe",
        "collect_coal",
        "furnace",
        "collect_iron",
        "iron_pickaxe",
        "diamond"
      ]
    }
  }
}
```

GEPA can then optimize primarily for `core_path_score` while protecting
`overall_crafter_score` from regression.

## SDK Naming Rules

Use objective language in public SDK APIs:

- `ObjectiveConfig`
- `Objective`
- `ObjectiveAcceptance`
- `objective_values`
- `selection_objective`
- `objective_threshold`

Avoid public SDK names like:

- `ScoringConfig`
- `score_config`
- `score_threshold`
- `score_vector`

Rust may keep internal storage names like `ScoreRecord` and `ScoreVectorRecord`
temporarily, but Python SDK and docs should speak in objectives.

## Required Implementation Work

1. Add SDK classes:
   - `Objective`
   - `ObjectiveConfig`
   - `ObjectiveAcceptance`
2. Map SDK objective config to current Rust TOML fields:
   - `gepa.selection_objective`
   - `gepa.objective_keys`
   - `gepa.objective_directions`
   - `gepa.acceptance_criterion`
   - `gepa.objective_acceptance`
   - `gepa.frontier_type`
3. Support `objectives=None` as container-provided/default reward behavior.
4. Add container contract metadata for objective declarations.
5. Add validation:
   - selection objective must be declared or discoverable
   - objective directions must be valid
   - protected objectives must be declared or discoverable
   - objective regression tolerance must be finite and non-negative
6. Update examples/docs to avoid public "score" terminology.
7. Add Crafter as the first multi-objective validation example.

## Validation

Minimum validation should include:

- Single objective default reward still works for Banking77 and TBLite.
- Explicit two-objective Crafter config renders valid TOML.
- GEPA run records objective values for both Crafter objectives.
- Selection objective drives candidate ranking.
- Protected objective guard rejects regressions beyond tolerance.

Validation commands should be added to the Better SDK pre-merge checklist once
Crafter is migrated to the typed SDK path.
