# Container-first RL plane: parallel work split

Design source: `docs/receipts/tblite-cispo-orbstack30-sync-async-benchmark-20260902.md`
(the engineering handoff section). Foundation landed in commit "Lay the
container-first RL foundation records".

## Shared, already written — import, never edit

- `src/synth_optimizers/contracts/rl_records.py` — `RendererProfile`,
  `BehaviorFingerprint`, `SamplingProfile`, `CompactionProvenance`,
  `InferenceCall`, `assert_strict_prefix`, `TrainableSegment`,
  `TrainableEpisode`, `RewardChannel`, `HorizonEvidence`, `RewardRecord`,
  `RecordError`, `EvidenceError`, `digest`, `LOGPROB_SENTINEL`.
- `src/synth_optimizers/contracts/rl_identity.py` — `GroupPin`,
  `assert_uniform_group`, `MixedGroupError`, `AgentInstance`, `Team`,
  `CommunicationChannel`, `Horizon`, `Topology`, `TopologyError`, `TaskSpec`,
  `RolloutReceipt`, state constants.
- `src/synth_optimizers/contracts/rl_clauses.py` — clause registry,
  `MANDATORY_CLAUSES`, `OPTIONAL_CLAUSES`, `VERDICTS`.

If one of these is genuinely wrong or missing a field, report it rather than
editing it: five other work streams depend on the same definitions.

## Ownership — one owner per file, no overlap

| Stream | Owns |
|---|---|
| 1 Rust contract | `rust/crates/synth_optimizer_platform/src/cispo_contract.rs`; minimal additive edits to `container_contract.rs` and `lib.rs` |
| 2 Preflight + handshake | `src/synth_optimizers/rl/{contract,capabilities,handshake,probe}.py` |
| 3 Queue engine | `src/synth_optimizers/rl/{queues,leases,lifecycle,store}.py` |
| 4 Plan + batches + replay | `src/synth_optimizers/rl/{plan,credit,objective,reducer,assembly,replay}.py` |
| 5 Checkpoint catalog | `src/synth_optimizers/rl/{catalog,policy_sets,resolver}.py` |
| 6 Conformance fakes | `tests/rl/fakes/**`, `tests/rl/test_conformance_fakes.py` |

Tests go in `tests/rl/`, named for the stream. Nobody edits
`src/synth_optimizers/rl/__init__.py`, `pyproject.toml`, `uv.lock`, the shared
contract modules, or another stream's files.

## House rules

- Python 3.11, ruff line-length 100. Frozen slotted dataclasses, module
  docstrings, `from __future__ import annotations`, validation that raises a
  typed error rather than returning a bool.
- No task, harness, environment, or model name in engine code. No literal
  dispatch on `banking77`, `healthbench`, `craftax`, `tblite`, `dungeongrid`,
  `runite`, `harbor`, `mini_swe`, `opencode`, `react`, `elf`, `barbarian`.
- CISPO is a preset over plan dimensions, never a branch in the engine.
- No git commands, no dependency changes, no Docker, no network, no paid
  provider calls. Unit tests only.
- Verify with `uv run ruff check <files>` and `uv run pytest tests/rl -q`.
