//! Codex app-server proposer for MAPO.
//!
//! This mirrors the GEPA workspace proposer contract in
//! `synth_gepa::codex_app_server`: MAPO materializes a workspace of coordination
//! evidence, a Codex app-server turn inspects it and writes
//! `proposal/manifest.json`, and MAPO parses that manifest into candidates.
//!
//! MAPO owns the workspace and the schema. The generic Codex launch, auth-home
//! preparation, JSON-RPC transport and usage normalization belong to
//! `synth_optimizer_platform::agent_runtime` and are not reimplemented here.
//!
//! The hardcoded `match index % 8` grid this file used to contain was not a
//! proposer: it never read a single rollout, so every "search" replayed the same
//! eight hand-written protocols regardless of what the team actually got wrong.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Map, Value};
use synth_optimizer_platform::agent_runtime::{
    run_turn, sandbox_policy_for_mode, usage_from_messages, CodexTurnRequest,
};
use synth_optimizer_platform::{OptimizerError, Result};

use crate::candidate::{
    MapoBranchCheckpoint, MapoCandidate, MapoProtocolConfig, MapoRolloutRecord,
    MapoSharedContextConfig,
};
use crate::config::MapoConfig;

pub const MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION: &str = "mapo_workspace_proposal_v1";
pub const MAPO_PROPOSER_BACKEND: &str = "codex_app_server";

/// Communication protocols the DungeonGrid message bus actually implements.
/// A proposal naming anything else is refused rather than silently coerced.
pub const PROTOCOL_MODES: [&str; 4] = [
    "pure_decentralized",
    "no_message",
    "master_to_slaves",
    "situational_lead_taking",
];

/// Files the proposer is expected to have opened. Omission is a warning, not a
/// hard error: the manifest is still usable, but the run receipt records that
/// the proposal was made without reading the evidence it was given.
const REQUIRED_REVIEWED_FILES: [&str; 4] = [
    "state/comms_failure_summary.json",
    "state/rollout_examples.json",
    "state/parent_payload.json",
    "state/candidate_deltas.json",
];

pub struct MapoProposerInput<'a> {
    pub config: &'a MapoConfig,
    pub parent: &'a MapoCandidate,
    pub candidates: &'a [MapoCandidate],
    pub train_rollouts: &'a [MapoRolloutRecord],
    pub branch_checkpoints: &'a [MapoBranchCheckpoint],
    pub generation: usize,
    pub workspace_dir: PathBuf,
}

pub struct MapoProposal {
    pub candidates: Vec<MapoCandidate>,
    pub receipt: Value,
}

pub fn propose_candidates(input: MapoProposerInput<'_>) -> Result<MapoProposal> {
    if input.config.proposer.backend != MAPO_PROPOSER_BACKEND {
        return Err(OptimizerError::Config(format!(
            "MAPO requires proposer.backend = \"{MAPO_PROPOSER_BACKEND}\"; got {:?}",
            input.config.proposer.backend
        )));
    }
    materialize_workspace(&input)?;

    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let message_stall_timeout =
        Duration::from_secs(input.config.proposer.message_stall_timeout_seconds.max(1));

    let outcome = run_turn(CodexTurnRequest {
        run_id: &input.config.run.run_id,
        proposer: &input.config.proposer,
        workspace_dir: &input.workspace_dir,
        model: &model,
        client_name: "synth-optimizers-mapo",
        client_title: "synth-optimizers MAPO",
        client_version: env!("CARGO_PKG_VERSION"),
        thread_start_params: thread_start_params(&input, &model),
        turn_start_params: turn_start_params(&input, &model),
        timeout,
        message_stall_timeout,
        message_observer: None,
    })?;

    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let candidates = candidates_from_proposals(&input, &proposals)?;
    let warnings = manifest_evidence_warnings(&input, &manifest, &candidates);
    let usage = usage_from_messages(&outcome.received_messages, &outcome.turn_id);
    write_workspace_pack_manifest(&input.workspace_dir)?;

    let receipt = json!({
        "schema_version": "mapo_proposer_receipt.v1",
        "generation": input.generation,
        "workspace": input.workspace_dir,
        "model": model,
        "thread_id": outcome.thread_id,
        "turn_id": outcome.turn_id,
        "parent_candidate_id": input.parent.id,
        "proposal_count": candidates.len(),
        "candidate_ids": candidates.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
        "evidence_warnings": warnings,
        "usage": usage,
        "critique": manifest.get("critique"),
        "rationale": manifest.get("rationale"),
    });
    write_json(
        &input.workspace_dir.join("state").join("proposer_receipt.json"),
        &receipt,
    )?;
    Ok(MapoProposal {
        candidates,
        receipt,
    })
}

// ---------------------------------------------------------------------------
// workspace
// ---------------------------------------------------------------------------

fn materialize_workspace(input: &MapoProposerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let proposal_dir = input.workspace_dir.join("proposal");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&proposal_dir)
        .map_err(|source| OptimizerError::io(&proposal_dir, source))?;

    write_text(&input.workspace_dir.join("README.md"), &workspace_readme(input))?;
    write_text(
        &input.workspace_dir.join("coordination_ladder.md"),
        COORDINATION_LADDER,
    )?;
    write_text(&proposal_dir.join("PROPOSAL_SCHEMA.md"), &proposal_schema(input))?;
    write_json(&proposal_dir.join("manifest.json"), &manifest_stub())?;

    write_json(
        &state_dir.join("run_context.json"),
        &json!({
            "run_id": input.config.run.run_id,
            "generation": input.generation,
            "task": "MAPO multi-agent protocol proposal",
            "parent_candidate_id": input.parent.id,
            "proposals_per_generation": input.config.mapo.proposals_per_generation,
            "protocol_modes": PROTOCOL_MODES,
            "role_keys": role_keys(input),
            "max_steps": input.config.mapo.max_steps,
            "train_seeds": input.config.taskset.train_seeds,
        }),
    )?;
    write_json(&state_dir.join("parent_payload.json"), &payload_of(input.parent))?;
    write_json(
        &state_dir.join("parent_candidate.json"),
        &candidate_row(input.parent),
    )?;
    write_json(&state_dir.join("candidates.json"), &candidate_rows(input))?;
    write_json(&state_dir.join("candidate_deltas.json"), &candidate_deltas(input))?;
    write_json(&state_dir.join("rollout_examples.json"), &rollout_examples(input))?;
    write_json(
        &state_dir.join("comms_failure_summary.json"),
        &comms_failure_summary(input),
    )?;
    write_json(
        &state_dir.join("branch_checkpoints.json"),
        &branch_checkpoint_rows(input),
    )?;
    write_json(
        &state_dir.join("proposal_request.json"),
        &json!({
            "schema_version": MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
            "proposals_per_round": input.config.mapo.proposals_per_generation,
            "generation": input.generation,
            "parent_candidate_id": input.parent.id,
        }),
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)
}

/// The sealed heldout wall. Nothing written into a proposer workspace may carry
/// a heldout measurement: the proposer selects, and a selector that can read the
/// heldout is selecting on it.
fn candidate_row(candidate: &MapoCandidate) -> Value {
    json!({
        "candidate_id": candidate.id,
        "generation": candidate.generation,
        "parent_id": candidate.parent_id,
        "train_score": candidate.train_score,
        "selection_score": candidate.selection_score,
        "payload": payload_of(candidate),
    })
}

fn candidate_rows(input: &MapoProposerInput<'_>) -> Value {
    Value::Array(input.candidates.iter().map(candidate_row).collect())
}

fn payload_of(candidate: &MapoCandidate) -> Value {
    json!({
        "protocol": candidate.protocol,
        "shared_context": candidate.shared_context,
        "roles": candidate.roles,
    })
}

/// Every key the container has actually accepted in a `roles` map. These are
/// not necessarily hero ids: depending on the quest they may be hero ids, class
/// names, or the `default` fallback. Naming them `hero_ids` would invite the
/// proposer to invent ids the container will silently ignore.
fn role_keys(input: &MapoProposerInput<'_>) -> Vec<String> {
    let mut ids: BTreeSet<String> = input
        .candidates
        .iter()
        .flat_map(|candidate| candidate.roles.keys().cloned())
        .collect();
    ids.extend(input.parent.roles.keys().cloned());
    ids.into_iter().collect()
}

fn candidate_deltas(input: &MapoProposerInput<'_>) -> Value {
    let parent_payload = payload_of(input.parent);
    let rows = input
        .candidates
        .iter()
        .filter(|candidate| candidate.id != input.parent.id)
        .map(|candidate| {
            let payload = payload_of(candidate);
            json!({
                "candidate_id": candidate.id,
                "generation": candidate.generation,
                "train_score": candidate.train_score,
                "protocol_changed": payload.get("protocol") != parent_payload.get("protocol"),
                "shared_context_changed":
                    payload.get("shared_context") != parent_payload.get("shared_context"),
                "changed_roles": changed_roles(input.parent, candidate),
            })
        })
        .collect::<Vec<_>>();
    Value::Array(rows)
}

fn changed_roles(parent: &MapoCandidate, candidate: &MapoCandidate) -> Vec<String> {
    let mut changed = Vec::new();
    for (role, text) in &candidate.roles {
        if parent.roles.get(role) != Some(text) {
            changed.push(role.clone());
        }
    }
    for role in parent.roles.keys() {
        if !candidate.roles.contains_key(role) {
            changed.push(format!("-{role}"));
        }
    }
    changed
}

fn rollout_examples(input: &MapoProposerInput<'_>) -> Value {
    let rows = input
        .train_rollouts
        .iter()
        .map(|record| {
            json!({
                "rollout_id": record.rollout_id,
                "candidate_id": record.candidate_id,
                "seed": record.seed,
                "episode_index": record.episode_index,
                "success": record.success,
                "reward": record.reward,
                "messages_delivered": record.messages_delivered,
                "messages_rejected": record.messages_rejected,
                "message_chars": record.message_chars,
                "chars_per_delivered_message": if record.messages_delivered == 0 {
                    Value::Null
                } else {
                    json!(record.message_chars as f64 / record.messages_delivered as f64)
                },
                "summary": record.response.get("summary"),
            })
        })
        .collect::<Vec<_>>();
    Value::Array(rows)
}

/// Coordination failure signal derived from the rollouts already collected.
///
/// These are the only comms failure classes the current container metrics can
/// support. They are deliberately named as what they are — aggregate signal, not
/// causal attribution — because nothing here isolates the effect of one message.
fn comms_failure_summary(input: &MapoProposerInput<'_>) -> Value {
    let rollouts = input.train_rollouts;
    if rollouts.is_empty() {
        return json!({
            "schema_version": "mapo_comms_failure_summary.v1",
            "rollout_count": 0,
            "note": "no train rollouts yet; propose from the seed payload and the coordination ladder only",
        });
    }
    let losses: Vec<&MapoRolloutRecord> = rollouts.iter().filter(|r| !r.success).collect();
    let wins: Vec<&MapoRolloutRecord> = rollouts.iter().filter(|r| r.success).collect();
    let mean_delivered = mean(rollouts.iter().map(|r| r.messages_delivered as f64));

    let silent_losses: Vec<&str> = losses
        .iter()
        .filter(|r| r.messages_delivered == 0)
        .map(|r| r.rollout_id.as_str())
        .collect();
    let chatter_losses: Vec<&str> = losses
        .iter()
        .filter(|r| (r.messages_delivered as f64) > mean_delivered)
        .map(|r| r.rollout_id.as_str())
        .collect();
    let rejected: Vec<&str> = rollouts
        .iter()
        .filter(|r| r.messages_rejected > 0)
        .map(|r| r.rollout_id.as_str())
        .collect();

    let mean_chars_per_message = if mean_delivered <= 0.0 {
        Value::Null
    } else {
        json!(mean(rollouts.iter().map(|r| r.message_chars as f64)) / mean_delivered)
    };

    json!({
        "schema_version": "mapo_comms_failure_summary.v1",
        "rollout_count": rollouts.len(),
        "win_count": wins.len(),
        "loss_count": losses.len(),
        "mean_reward_win": mean(wins.iter().map(|r| r.reward)),
        "mean_reward_loss": mean(losses.iter().map(|r| r.reward)),
        "mean_messages_delivered": mean_delivered,
        "mean_messages_delivered_win": mean(wins.iter().map(|r| r.messages_delivered as f64)),
        "mean_messages_delivered_loss": mean(losses.iter().map(|r| r.messages_delivered as f64)),
        "mean_chars_per_message": mean_chars_per_message,
        "failure_classes": [
            {
                "id": "silent_loss",
                "description": "lost the episode without delivering a single message",
                "rollout_ids": silent_losses,
            },
            {
                "id": "chatter_loss",
                "description": "lost the episode while delivering more messages than the run mean",
                "rollout_ids": chatter_losses,
            },
            {
                "id": "protocol_rejection",
                "description": "the message bus rejected a message: the role tried to speak when the protocol forbade it",
                "rollout_ids": rejected,
            },
        ],
        "caveat": "aggregate signal only. No entry here isolates the causal effect of an individual message.",
    })
}

fn branch_checkpoint_rows(input: &MapoProposerInput<'_>) -> Value {
    let rows = input
        .branch_checkpoints
        .iter()
        .map(|checkpoint| {
            json!({
                "checkpoint_id": checkpoint.checkpoint_id,
                "parent_rollout_id": checkpoint.parent_rollout_id,
                "seed": checkpoint.seed,
                "step": checkpoint.step,
                "reward": checkpoint.reward,
                "messages_delivered": checkpoint.messages_delivered,
                "messages_rejected": checkpoint.messages_rejected,
            })
        })
        .collect::<Vec<_>>();
    Value::Array(rows)
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0usize;
    let mut total = 0.0;
    for value in values {
        count += 1;
        total += value;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

// ---------------------------------------------------------------------------
// prompts
// ---------------------------------------------------------------------------

const COORDINATION_LADDER: &str = r#"# MAPO coordination ladder

MAPO's analogue of an achievement ladder. Rungs are ordered: a team that fails a
low rung cannot be helped by a proposal aimed at a high one.

1. **Speak at all** — at least one message is delivered when the protocol allows it.
2. **Legal speech** — no message is rejected by the bus; roles respect the protocol
   (under `master_to_slaves` only the leader speaks; under `no_message` nobody does).
3. **Informative speech** — messages carry state a teammate could not observe locally:
   a claimed target, a blocked route, a threat, a carrier handoff.
4. **Read speech** — a teammate's behaviour changes after a message arrives.
5. **Cheap speech** — the same coordination is reached with fewer characters and
   fewer messages; message actions cost AP that could have been progress.
6. **Divided labour** — heroes stop duplicating each other's targets.
7. **Recovery** — after a blocked route or invalid action the team replans instead
   of repeating it.

`no_message` is the control rung, not a failure mode. If a candidate with
`mode = "no_message"` matches a talking candidate, the talking was decoration and
the proposal budget should go elsewhere.
"#;

fn workspace_readme(input: &MapoProposerInput<'_>) -> String {
    format!(
        r#"# MAPO proposer workspace — generation {generation}

You are proposing multi-agent coordination candidates for MAPO on a cooperative
DungeonGrid team. A candidate is a protocol mode, a shared-context configuration
and a set of per-role briefs.

Read, in this order:

1. `coordination_ladder.md` — which rung the team is actually stuck on.
2. `state/comms_failure_summary.json` — aggregate comms failure classes.
3. `state/rollout_examples.json` — per-rollout evidence you must cite by id.
4. `state/parent_payload.json` — the payload you are varying.
5. `state/candidate_deltas.json` — what has already been tried and what it scored.
6. `proposal/PROPOSAL_SCHEMA.md` — the exact manifest you must write.

Then write `proposal/manifest.json`.

There is no heldout evidence in this workspace by design. Do not ask for it and
do not infer it; proposals are selected on train evidence only.

Parent candidate: `{parent}`
Proposals required: {count}
"#,
        generation = input.generation,
        parent = input.parent.id,
        count = input.config.mapo.proposals_per_generation,
    )
}

fn manifest_stub() -> Value {
    json!({
        "schema_version": MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
        "critique": "",
        "evidence": {
            "reviewed_files": [],
            "candidate_comparison": "",
            "comms_failure_patterns": [],
            "coordination_wins": [],
            "rollout_ids_used": [],
            "ladder_rung": "",
        },
        "rationale": "",
        "proposals": [],
    })
}

fn proposal_schema(input: &MapoProposerInput<'_>) -> String {
    format!(
        r#"# MAPO Workspace Proposer Schema

Write `proposal/manifest.json` as strict JSON using this schema:

```json
{{
  "schema_version": "{schema}",
  "critique": "What the parent protocol is missing, grounded in state/ evidence.",
  "evidence": {{
    "reviewed_files": [
      "coordination_ladder.md",
      "state/comms_failure_summary.json",
      "state/rollout_examples.json",
      "state/parent_payload.json",
      "state/candidate_deltas.json",
      "state/branch_checkpoints.json"
    ],
    "candidate_comparison": "How the parent compares to what has already been tried.",
    "comms_failure_patterns": ["Named failure class plus the rollout ids that show it."],
    "coordination_wins": ["What the winning rollouts did that the losing ones did not."],
    "rollout_ids_used": ["<rollout_id>", "<rollout_id>"],
    "ladder_rung": "The lowest coordination_ladder.md rung the team is failing."
  }},
  "rationale": "Why these candidates should raise that rung.",
  "proposals": [
    {{
      "proposal_type": "parent_variation",
      "parent_candidate_ids": ["{parent}"],
      "rationale": "Which comms failure class this attacks.",
      "targets_failure_class": "silent_loss | chatter_loss | protocol_rejection | <named>",
      "proposed_payload": {{
        "protocol": {{
          "mode": "pure_decentralized",
          "max_chars": 240,
          "leader_policy": "first_hero",
          "leader_role": "",
          "followers_can_reply": false
        }},
        "shared_context": {{ "render_mode": "full", "window_tokens": 500 }},
        "roles": {{
          "default": "<brief every hero receives>",
          "<hero_id>": "<per-hero brief>"
        }}
      }}
    }}
  ]
}}
```

Rules:

- Inspect the state files with shell/python/jq before editing the manifest. Do not
  jump straight to writing it.
- `protocol.mode` must be one of: {modes}. Any other value is refused.
- `roles` must be non-empty and must contain a `default` brief. Use only keys listed
  in `state/run_context.json.role_keys`; any other key is silently ignored by the
  container, so an invented hero id is a wasted proposal.
- `shared_context` keys are optional; omitted keys inherit the parent value.
- `evidence.rollout_ids_used` must cite ids that exist in `state/rollout_examples.json`,
  including at least one losing rollout.
- `evidence.ladder_rung` must name a rung from `coordination_ladder.md`. Aim proposals
  at the lowest failing rung — a brief about efficient phrasing is wasted on a team
  that is not being read at all.
- Create exactly {count} distinct proposals. Distinct means different mechanisms,
  not paraphrases: changing only wording across all proposals wastes the generation.
- At most one proposal may be conservative. Include a `no_message` or otherwise
  stripped-down control when the evidence cannot yet show that talking helps.
- Do not duplicate `state/parent_payload.json`.
- Do not copy literal seed-specific map facts into role briefs. Briefs must
  generalize across seeds; a brief naming a specific room or door is overfitting.
"#,
        schema = MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
        parent = input.parent.id,
        modes = PROTOCOL_MODES.join(", "),
        count = input.config.mapo.proposals_per_generation,
    )
}

fn thread_start_params(input: &MapoProposerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the MAPO workspace proposer. Work only inside this workspace.".to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn turn_start_params(input: &MapoProposerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": proposer_instructions(input),
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(service_tier) = non_empty(input.config.proposer.service_tier.as_deref()) {
        params.insert(
            "serviceTier".to_string(),
            Value::String(service_tier.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Value::Object(params)
}

fn proposer_instructions(input: &MapoProposerInput<'_>) -> String {
    format!(
        "MAPO generation {generation}. Parent candidate: {parent}.\n\
         Read README.md, coordination_ladder.md, proposal/PROPOSAL_SCHEMA.md, then the state files.\n\
         Start with state/comms_failure_summary.json and state/rollout_examples.json.\n\
         Use real shell/python/jq inspection before editing proposal/manifest.json. Do not print pseudo-tool calls.\n\
         Diagnose the lowest failing rung of the coordination ladder, then propose exactly {count} \
         distinct coordination candidates that attack it.\n\
         Distinct means different mechanisms — protocol mode, leadership, shared-context window, \
         role division — not reworded briefs.\n\
         Write strict JSON to proposal/manifest.json using schema_version {schema}, including the \
         evidence block with reviewed files, named comms failure patterns, cited rollout ids and the ladder rung.",
        generation = input.generation,
        parent = input.parent.id,
        count = input.config.mapo.proposals_per_generation,
        schema = MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
    )
}

// ---------------------------------------------------------------------------
// manifest
// ---------------------------------------------------------------------------

fn read_manifest(workspace_dir: &Path) -> Result<Value> {
    let path = workspace_dir.join("proposal").join("manifest.json");
    let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
    if text.trim().is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "MAPO codex proposer wrote an empty manifest: {}",
            path.display()
        )));
    }
    Ok(serde_json::from_str(&text)?)
}

fn proposals_from_manifest(manifest: &Value) -> Result<Vec<Value>> {
    let schema_version = manifest
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema_version != MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION {
        return Err(OptimizerError::Proposer(format!(
            "MAPO proposer manifest schema_version={schema_version:?}; expected {MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION}"
        )));
    }
    if manifest.get("evidence").and_then(Value::as_object).is_none() {
        return Err(OptimizerError::Proposer(
            "MAPO proposer manifest omitted the required evidence object".to_string(),
        ));
    }
    let proposals = manifest
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if proposals.is_empty() {
        return Err(OptimizerError::Proposer(
            "MAPO proposer manifest contained no proposals".to_string(),
        ));
    }
    Ok(proposals)
}

fn candidates_from_proposals(
    input: &MapoProposerInput<'_>,
    proposals: &[Value],
) -> Result<Vec<MapoCandidate>> {
    let parent_payload = payload_of(input.parent);
    let mut candidates = Vec::with_capacity(proposals.len());
    for (index, proposal) in proposals.iter().enumerate() {
        let payload = proposal.get("proposed_payload").ok_or_else(|| {
            OptimizerError::Proposer(format!(
                "MAPO proposal {index} omitted proposed_payload"
            ))
        })?;
        if payload == &parent_payload {
            return Err(OptimizerError::Proposer(format!(
                "MAPO proposal {index} duplicates the parent payload"
            )));
        }
        let mut candidate = input.parent.clone();
        candidate.id = format!("mapo_g{}_p{index}", input.generation);
        candidate.generation = input.generation;
        candidate.parent_id = Some(input.parent.id.clone());
        candidate.train_score = None;
        candidate.selection_score = None;
        candidate.heldout_score = None;

        if let Some(protocol) = payload.get("protocol") {
            candidate.protocol = parse_protocol(protocol, index)?;
        }
        if let Some(shared_context) = payload.get("shared_context") {
            candidate.shared_context = parse_shared_context(shared_context, &candidate, index)?;
        }
        let roles = payload
            .get("roles")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                OptimizerError::Proposer(format!(
                    "MAPO proposal {index} omitted a roles object"
                ))
            })?;
        if roles.is_empty() {
            return Err(OptimizerError::Proposer(format!(
                "MAPO proposal {index} proposed an empty roles object"
            )));
        }
        candidate.roles.clear();
        for (role, text) in roles {
            let brief = text.as_str().ok_or_else(|| {
                OptimizerError::Proposer(format!(
                    "MAPO proposal {index} role {role:?} brief is not a string"
                ))
            })?;
            candidate.roles.insert(role.clone(), brief.to_string());
        }
        candidates.push(candidate);
    }
    Ok(candidates)
}

fn parse_protocol(value: &Value, index: usize) -> Result<MapoProtocolConfig> {
    let protocol: MapoProtocolConfig = serde_json::from_value(value.clone()).map_err(|source| {
        OptimizerError::Proposer(format!(
            "MAPO proposal {index} protocol did not parse: {source}"
        ))
    })?;
    if !PROTOCOL_MODES.contains(&protocol.mode.as_str()) {
        return Err(OptimizerError::Proposer(format!(
            "MAPO proposal {index} protocol.mode={:?} is not an implemented protocol; allowed: {}",
            protocol.mode,
            PROTOCOL_MODES.join(", ")
        )));
    }
    if protocol.max_chars == 0 {
        return Err(OptimizerError::Proposer(format!(
            "MAPO proposal {index} protocol.max_chars must be positive"
        )));
    }
    Ok(protocol)
}

/// Shared-context keys are optional: a proposal that only changes the window
/// should not have to restate every unrelated field, so omitted keys inherit the
/// parent value rather than snapping back to the type default.
fn parse_shared_context(
    value: &Value,
    candidate: &MapoCandidate,
    index: usize,
) -> Result<MapoSharedContextConfig> {
    let overrides = value.as_object().ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "MAPO proposal {index} shared_context must be an object"
        ))
    })?;
    let mut merged = serde_json::to_value(&candidate.shared_context)?
        .as_object()
        .cloned()
        .unwrap_or_default();
    for (key, entry) in overrides {
        if !merged.contains_key(key) {
            return Err(OptimizerError::Proposer(format!(
                "MAPO proposal {index} shared_context key {key:?} is not a shared-context field"
            )));
        }
        merged.insert(key.clone(), entry.clone());
    }
    serde_json::from_value(Value::Object(merged)).map_err(|source| {
        OptimizerError::Proposer(format!(
            "MAPO proposal {index} shared_context did not parse: {source}"
        ))
    })
}

/// Non-fatal quality signal. These land in the run receipt so a generation that
/// proposed without reading its evidence is visible afterwards instead of being
/// indistinguishable from one that did.
fn manifest_evidence_warnings(
    input: &MapoProposerInput<'_>,
    manifest: &Value,
    candidates: &[MapoCandidate],
) -> Vec<String> {
    let mut warnings = Vec::new();
    let evidence = manifest.get("evidence").and_then(Value::as_object);

    let reviewed = string_set(evidence, "reviewed_files");
    for required in REQUIRED_REVIEWED_FILES {
        if !reviewed.contains(required) {
            warnings.push(format!("evidence.reviewed_files did not include {required}"));
        }
    }

    let cited = string_set(evidence, "rollout_ids_used");
    let known: BTreeSet<&str> = input
        .train_rollouts
        .iter()
        .map(|record| record.rollout_id.as_str())
        .collect();
    let losing: BTreeSet<&str> = input
        .train_rollouts
        .iter()
        .filter(|record| !record.success)
        .map(|record| record.rollout_id.as_str())
        .collect();
    for id in &cited {
        if !known.contains(id.as_str()) {
            warnings.push(format!("evidence.rollout_ids_used cited unknown rollout {id}"));
        }
    }
    if !losing.is_empty() && !cited.iter().any(|id| losing.contains(id.as_str())) {
        warnings.push("evidence.rollout_ids_used cited no losing rollout".to_string());
    }
    if evidence
        .and_then(|evidence| evidence.get("ladder_rung"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .is_empty()
    {
        warnings.push("evidence.ladder_rung was empty".to_string());
    }

    let modes: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.protocol.mode.as_str())
        .collect();
    if candidates.len() > 1 && modes.len() == 1 {
        warnings.push(format!(
            "all {} proposals share protocol.mode={:?}",
            candidates.len(),
            modes.iter().next().copied().unwrap_or_default()
        ));
    }
    if candidates.len() != input.config.mapo.proposals_per_generation {
        warnings.push(format!(
            "requested {} proposals, manifest supplied {}",
            input.config.mapo.proposals_per_generation,
            candidates.len()
        ));
    }
    warnings
}

fn string_set(evidence: Option<&Map<String, Value>>, key: &str) -> BTreeSet<String> {
    evidence
        .and_then(|evidence| evidence.get(key))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// io
// ---------------------------------------------------------------------------

fn write_workspace_pack_manifest(workspace_dir: &Path) -> Result<()> {
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    let mut files = Vec::new();
    collect_workspace_files(workspace_dir, workspace_dir, &mut files)?;
    files.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .cmp(&right.get("path").and_then(Value::as_str))
    });
    write_json(
        &state_dir.join("workspace_pack_manifest.json"),
        &json!({
            "schema_version": "mapo_workspace_pack_manifest.v1",
            "file_count": files.len(),
            "files": files,
        }),
    )
}

fn collect_workspace_files(root: &Path, current: &Path, files: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| OptimizerError::io(current, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(current, source))?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        if relative
            .components()
            .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
        {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.is_dir() {
            collect_workspace_files(root, &path, files)?;
            continue;
        }
        if relative == Path::new("state/workspace_pack_manifest.json") {
            continue;
        }
        files.push(json!({
            "path": relative.to_string_lossy(),
            "bytes": metadata.len(),
        }));
    }
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    write_text(path, &format!("{text}\n"))
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
    }
    fs::write(path, text).map_err(|source| OptimizerError::io(path, source))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent() -> MapoCandidate {
        let mut candidate = MapoCandidate::seed("mapo_seed");
        candidate.roles.insert("default".to_string(), "seed brief".to_string());
        candidate.heldout_score = Some(crate::scoring::MapoScore::default());
        candidate
    }

    fn config() -> MapoConfig {
        let mut config = MapoConfig::default();
        config.mapo.proposals_per_generation = 2;
        config
    }

    fn input<'a>(
        config: &'a MapoConfig,
        parent: &'a MapoCandidate,
        rollouts: &'a [MapoRolloutRecord],
    ) -> MapoProposerInput<'a> {
        MapoProposerInput {
            config,
            parent,
            candidates: std::slice::from_ref(parent),
            train_rollouts: rollouts,
            branch_checkpoints: &[],
            generation: 1,
            workspace_dir: PathBuf::from("/nonexistent"),
        }
    }

    fn payload(mode: &str) -> Value {
        json!({
            "protocol": {"mode": mode, "max_chars": 200},
            "roles": {"default": "say the claim out loud"},
        })
    }

    fn rollout(id: &str, success: bool, delivered: u64) -> MapoRolloutRecord {
        MapoRolloutRecord {
            rollout_id: id.to_string(),
            candidate_id: "mapo_seed".to_string(),
            split: "train".to_string(),
            rollout_group: "g0".to_string(),
            seed: 11,
            episode_index: 0,
            task_instance_id: None,
            parent_rollout_id: None,
            parent_checkpoint_id: None,
            checkpoint_id: None,
            success,
            reward: if success { 1.0 } else { 0.0 },
            messages_delivered: delivered,
            messages_rejected: 0,
            message_chars: delivered * 20,
            response: json!({"summary": {"success": success}}),
        }
    }

    #[test]
    fn a_manifest_with_the_wrong_schema_version_is_refused() {
        let manifest = json!({"schema_version": "gepa_workspace_proposal_v3", "evidence": {}, "proposals": [{}]});
        let error = proposals_from_manifest(&manifest).unwrap_err();
        assert!(error.to_string().contains("schema_version"));
    }

    #[test]
    fn a_manifest_without_evidence_is_refused() {
        let manifest = json!({
            "schema_version": MAPO_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
            "proposals": [{"proposed_payload": payload("no_message")}],
        });
        let error = proposals_from_manifest(&manifest).unwrap_err();
        assert!(error.to_string().contains("evidence"));
    }

    #[test]
    fn an_unimplemented_protocol_mode_is_refused() {
        let config = config();
        let parent = parent();
        let proposals = vec![json!({"proposed_payload": payload("consensus_vote")})];
        let error = candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap_err();
        assert!(error.to_string().contains("consensus_vote"));
    }

    #[test]
    fn a_proposal_duplicating_the_parent_payload_is_refused() {
        let config = config();
        let parent = parent();
        let proposals = vec![json!({"proposed_payload": payload_of(&parent)})];
        let error = candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap_err();
        assert!(error.to_string().contains("duplicates the parent payload"));
    }

    #[test]
    fn an_unknown_shared_context_key_is_refused_rather_than_silently_dropped() {
        let config = config();
        let parent = parent();
        let proposals = vec![json!({
            "proposed_payload": {
                "protocol": {"mode": "no_message", "max_chars": 200},
                "shared_context": {"window_tokns": 900},
                "roles": {"default": "quiet"},
            }
        })];
        let error = candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap_err();
        assert!(error.to_string().contains("window_tokns"));
    }

    #[test]
    fn omitted_shared_context_keys_inherit_the_parent_value() {
        let config = config();
        let mut parent = parent();
        parent.shared_context.window_tokens = 777;
        let proposals = vec![json!({
            "proposed_payload": {
                "protocol": {"mode": "no_message", "max_chars": 200},
                "shared_context": {"claim_window_tokens": 42},
                "roles": {"default": "quiet"},
            }
        })];
        let candidates =
            candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap();
        assert_eq!(candidates[0].shared_context.claim_window_tokens, 42);
        assert_eq!(candidates[0].shared_context.window_tokens, 777);
    }

    #[test]
    fn a_proposal_without_roles_is_refused() {
        let config = config();
        let parent = parent();
        let proposals = vec![json!({
            "proposed_payload": {"protocol": {"mode": "no_message", "max_chars": 200}}
        })];
        let error = candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap_err();
        assert!(error.to_string().contains("roles"));
    }

    #[test]
    fn proposed_candidates_never_inherit_the_parent_scores() {
        let config = config();
        let mut parent = parent();
        parent.train_score = Some(crate::scoring::MapoScore::default());
        let proposals = vec![json!({"proposed_payload": payload("no_message")})];
        let candidates =
            candidates_from_proposals(&input(&config, &parent, &[]), &proposals).unwrap();
        assert!(candidates[0].train_score.is_none());
        assert!(candidates[0].selection_score.is_none());
        assert!(candidates[0].heldout_score.is_none());
        assert_eq!(candidates[0].parent_id.as_deref(), Some("mapo_seed"));
    }

    /// The sealed heldout wall. A proposer that can read heldout scores is
    /// selecting on them, which is the one thing the split exists to prevent.
    #[test]
    fn no_workspace_row_carries_a_heldout_score() {
        let parent = parent();
        assert!(parent.heldout_score.is_some());
        let row = candidate_row(&parent).to_string();
        assert!(!row.contains("heldout"), "workspace row leaked heldout: {row}");
    }

    #[test]
    fn evidence_warnings_flag_a_proposal_made_without_citing_a_loss() {
        let config = config();
        let parent = parent();
        let rollouts = vec![rollout("r_win", true, 3), rollout("r_loss", false, 0)];
        let manifest = json!({
            "evidence": {
                "reviewed_files": REQUIRED_REVIEWED_FILES,
                "rollout_ids_used": ["r_win"],
                "ladder_rung": "3. Informative speech",
            }
        });
        let proposals = vec![
            json!({"proposed_payload": payload("no_message")}),
            json!({"proposed_payload": payload("master_to_slaves")}),
        ];
        let candidates =
            candidates_from_proposals(&input(&config, &parent, &rollouts), &proposals).unwrap();
        let warnings =
            manifest_evidence_warnings(&input(&config, &parent, &rollouts), &manifest, &candidates);
        assert!(warnings.iter().any(|w| w.contains("no losing rollout")), "{warnings:?}");
        assert!(!warnings.iter().any(|w| w.contains("share protocol.mode")));
    }

    #[test]
    fn evidence_warnings_flag_an_uncited_rollout_and_a_single_mode_generation() {
        let config = config();
        let parent = parent();
        let rollouts = vec![rollout("r_loss", false, 0)];
        let manifest = json!({
            "evidence": {
                "reviewed_files": [],
                "rollout_ids_used": ["r_loss", "r_imaginary"],
                "ladder_rung": "",
            }
        });
        let proposals = vec![
            json!({"proposed_payload": payload("no_message")}),
            json!({"proposed_payload": {
                "protocol": {"mode": "no_message", "max_chars": 120},
                "roles": {"default": "different brief"},
            }}),
        ];
        let candidates =
            candidates_from_proposals(&input(&config, &parent, &rollouts), &proposals).unwrap();
        let warnings =
            manifest_evidence_warnings(&input(&config, &parent, &rollouts), &manifest, &candidates);
        assert!(warnings.iter().any(|w| w.contains("r_imaginary")), "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("share protocol.mode")), "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("ladder_rung")), "{warnings:?}");
        for required in REQUIRED_REVIEWED_FILES {
            assert!(warnings.iter().any(|w| w.contains(required)), "{warnings:?}");
        }
    }

    #[test]
    fn the_comms_failure_summary_separates_silent_losses_from_chatter_losses() {
        let config = config();
        let parent = parent();
        let rollouts = vec![
            rollout("r_silent", false, 0),
            rollout("r_chatter", false, 9),
            rollout("r_win", true, 2),
        ];
        let summary = comms_failure_summary(&input(&config, &parent, &rollouts));
        let classes = summary.get("failure_classes").unwrap().as_array().unwrap();
        let silent = classes.iter().find(|c| c["id"] == "silent_loss").unwrap();
        let chatter = classes.iter().find(|c| c["id"] == "chatter_loss").unwrap();
        assert_eq!(silent["rollout_ids"], json!(["r_silent"]));
        assert_eq!(chatter["rollout_ids"], json!(["r_chatter"]));
    }

    /// End-to-end check on the files the Codex turn actually reads. The
    /// row-level heldout assertion above can pass while some other writer leaks
    /// the same field, so this greps the whole materialized workspace.
    #[test]
    fn the_materialized_workspace_is_complete_and_carries_no_heldout_evidence() {
        let config = config();
        let mut parent = parent();
        parent.heldout_score = Some(crate::scoring::MapoScore::default());
        let rollouts = vec![rollout("r_win", true, 3), rollout("r_loss", false, 0)];
        let dir = std::env::temp_dir().join(format!(
            "mapo_proposer_ws_{}",
            uuid::Uuid::new_v4().simple()
        ));
        let mut input = input(&config, &parent, &rollouts);
        input.workspace_dir = dir.clone();
        materialize_workspace(&input).unwrap();

        for expected in [
            "README.md",
            "coordination_ladder.md",
            "proposal/PROPOSAL_SCHEMA.md",
            "proposal/manifest.json",
            "state/run_context.json",
            "state/parent_payload.json",
            "state/candidates.json",
            "state/candidate_deltas.json",
            "state/rollout_examples.json",
            "state/comms_failure_summary.json",
            "state/branch_checkpoints.json",
            "state/proposal_request.json",
            "state/workspace_pack_manifest.json",
        ] {
            assert!(dir.join(expected).is_file(), "missing {expected}");
        }

        let mut files = Vec::new();
        collect_workspace_files(&dir, &dir, &mut files).unwrap();
        assert!(files.len() >= 12, "{files:?}");
        // Prose may name the heldout to tell the proposer it is off limits; a
        // JSON *key* naming it is the leak.
        for file in &files {
            let relative = file["path"].as_str().unwrap();
            if !relative.ends_with(".json") {
                continue;
            }
            let value: Value =
                serde_json::from_str(&fs::read_to_string(dir.join(relative)).unwrap()).unwrap();
            assert!(
                !has_key_containing(&value, "heldout"),
                "{relative} leaked a heldout key into the proposer workspace"
            );
        }
        fs::remove_dir_all(&dir).ok();
    }

    fn has_key_containing(value: &Value, needle: &str) -> bool {
        match value {
            Value::Object(map) => map
                .iter()
                .any(|(key, entry)| key.contains(needle) || has_key_containing(entry, needle)),
            Value::Array(items) => items.iter().any(|item| has_key_containing(item, needle)),
            _ => false,
        }
    }

    #[test]
    fn the_removed_grid_is_refused_by_config_validation_with_a_migration_message() {
        let mut config = MapoConfig::default();
        config.mapo.proposer_mode = "deterministic_grid".to_string();
        config.container.url = Some("http://127.0.0.1:8080".to_string());
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("deterministic_grid"), "{error}");
        assert!(error.contains("codex_app_server"), "{error}");
    }
}
