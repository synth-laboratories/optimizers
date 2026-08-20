# synth_mapo

`synth_mapo` is the Rust implementation of MAPO, the Multi-Agent Policy
Optimizer. It searches over environment-backed communication protocols,
shared-context settings, and per-role prompt suffixes while preserving the
container as the authority for rollout scoring.

The v1 loop is GEPA-shaped:

1. Materialize a seed candidate.
2. Propose candidate protocol / role / shared-context edits with the Codex
   app-server proposer (below).
3. Evaluate candidates on train seeds or train task instances.
4. Select by lexicographic MAPO score.
5. Run a terminal baseline-vs-champion heldout comparison keyed exactly by
   `(seed, episode_index, task_instance_id)`.

## Proposer

Proposal is a Codex app-server turn over a materialized evidence workspace,
following the same contract as the GEPA workspace proposer
(`synth_gepa::codex_app_server`). MAPO owns the workspace and the manifest
schema; the Codex launch, auth home, JSON-RPC transport and usage normalization
belong to `synth_optimizer_platform::agent_runtime`.

Per generation, `runs/<run_id>/proposer_workspaces/generation_NNN/` holds:

```text
README.md
coordination_ladder.md              # which coordination rung the team is failing
proposal/PROPOSAL_SCHEMA.md
proposal/manifest.json              # the proposer overwrites this
state/run_context.json
state/parent_payload.json
state/parent_candidate.json
state/candidates.json
state/candidate_deltas.json
state/rollout_examples.json         # per-rollout evidence, cited by id
state/comms_failure_summary.json    # silent_loss / chatter_loss / protocol_rejection
state/branch_checkpoints.json
state/proposal_request.json
state/workspace_pack_manifest.json
```

The manifest is `mapo_workspace_proposal_v1`. Admission is strict: a proposal
naming a protocol mode the message bus does not implement, omitting `roles`,
using an unknown `shared_context` key, or duplicating the parent payload is
refused rather than coerced. Weaker signals — reviewed files omitted, no losing
rollout cited, no ladder rung named, every proposal sharing one protocol mode —
are recorded as evidence warnings in `artifacts/mapo_proposer_receipts.json`.

**No heldout evidence enters a proposer workspace.** Candidate rows are written
without their heldout scores and a test greps the materialized workspace for a
heldout key.

`mapo.proposer_mode = "deterministic_grid"` was removed. It was a hardcoded
`match index % 8` of hand-written protocols that never read a rollout, so every
generation replayed the same eight candidates regardless of what the team
actually got wrong. Configs still setting it fail closed with a migration
message.

Configuration rejects overlap among train, selection, and held-out seeds. The
terminal comparator rejects empty arms, duplicate keys, missing keys, and any
baseline/champion key mismatch. Its artifact includes paired success and reward
effects with 95% intervals; an equal arm count alone is not pairing proof.

For DungeonGrid Plus, the launch gate is intentionally strict: the MAPO
champion must beat the baseline by at least 10 percentage points absolute on
heldout quest success over at least 20 paired episodes per arm, with
`message_chars` per success no worse than baseline.

## Validation

```bash
cargo check -p synth_mapo
cargo test -p synth_mapo
```

The public package currently exposes the library crate. Hosted and local daemon
entrypoints live in `optimizers-beta` until MAPO is signed off for merge.

## Artifacts

Each run writes:

- `artifacts/result_manifest.json`
- `artifacts/campaign_manifest.json` when a frozen campaign is bound
- `artifacts/mapo_rollout_request_preview.json`
- `artifacts/mapo_candidate_registry.json`
- `artifacts/mapo_proposer_receipts.json`
- `artifacts/mapo_rollouts.json`
- `artifacts/mapo_review_rows.json`
- `artifacts/mapo_heldout_comparison.json` when heldout proof runs
- `mapo_status.json`

`mapo_review_rows.json` uses `ohco.review_row.v1` rows for coordination
failures such as rejected messages, duplicate CLAIM evidence, and split-party
failures with no tactical communication.

For terminal Debrief materialization, set
`evidence.campaign_manifest_path` to an approved
`debrief.campaign_manifest.v1` JSON file and fill the remaining typed evidence
fields. MAPO verifies the frozen benchmark, model, split, optimizer, pairing,
metric, and spend/time boundaries against its resolved config before executing.
It copies the exact manifest bytes into the artifact root, computes the SHA-256
receipt, and derives campaign id, source digests, model snapshots, and search
budget in `debrief_evidence` from that immutable source. Missing or mismatched
authority fails closed.

Nested action/message `reason` fields and private rationale, thought, analysis,
chain-of-thought, and scratchpad fields are recursively excluded from persisted
rollout evidence. Receipts, trace identifiers, public metrics, and provenance
metadata remain available for audit.
