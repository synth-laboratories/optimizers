# synth_mapo

`synth_mapo` is the Rust implementation of MAPO, the Multi-Agent Policy
Optimizer. It searches over environment-backed communication protocols,
shared-context settings, and per-role prompt suffixes while preserving the
container as the authority for rollout scoring.

The v1 loop is GEPA-shaped:

1. Materialize a seed candidate.
2. Propose candidate protocol / role / shared-context edits.
3. Evaluate candidates on train seeds or train task instances.
4. Select by lexicographic MAPO score.
5. Run a terminal baseline-vs-champion heldout comparison keyed exactly by
   `(seed, episode_index, task_instance_id)`.

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
```

The public package currently exposes the library crate. Hosted and local daemon
entrypoints live in `optimizers-beta` until MAPO is signed off for merge.

## Artifacts

Each run writes:

- `artifacts/result_manifest.json`
- `artifacts/campaign_manifest.json` when a frozen campaign is bound
- `artifacts/mapo_rollout_request_preview.json`
- `artifacts/mapo_candidate_registry.json`
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
