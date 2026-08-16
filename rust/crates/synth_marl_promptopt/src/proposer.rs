use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_gepa::{
    propose_workspace_candidates, CandidateRecord, RolloutScore, WorkspaceProposerOutcome,
};
use synth_optimizer_platform::{
    LeverBundle, OptimizerError, PromptProgram, Result, SynthOptimizerConfig,
};

use crate::strategy::MarlStrategy;
use crate::types::MarlCandidate;

pub struct ProposeGenerationInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub strategy: &'a dyn MarlStrategy,
    pub parent: &'a MarlCandidate,
    pub candidates: &'a [MarlCandidate],
    pub generation: usize,
    pub reflection_rows: &'a [Value],
    pub run_dir: &'a Path,
}

pub fn propose_generation(input: ProposeGenerationInput<'_>) -> Result<Vec<MarlCandidate>> {
    let mut proposer_config = input.config.clone();
    let target_fields = input
        .strategy
        .target_fields(input.generation, input.program);
    if target_fields.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "strategy {} selected no target fields for generation {}",
            input.strategy.name(),
            input.generation
        )));
    }
    proposer_config.candidate.target_modules = target_fields.clone();

    let mut proposer_program = input.program.clone();
    proposer_program.metadata.insert(
        "marl_optimizer_dynamics".to_string(),
        input.strategy.proposer_guidance(),
    );
    proposer_program
        .metadata
        .insert("marl_variant".to_string(), json!(input.strategy.name()));

    let parent = gepa_candidate_record(input.parent);
    let candidates = input
        .candidates
        .iter()
        .map(gepa_candidate_record)
        .collect::<Vec<_>>();
    let task_pool_rows = proposer_task_pool_rows(input.reflection_rows);
    let workspace_dir = input
        .run_dir
        .join("proposer_workspaces")
        .join(format!("generation_{:03}", input.generation));
    let outcome = propose_workspace_candidates(
        &proposer_config,
        &proposer_program,
        &parent,
        &candidates,
        input.generation,
        task_pool_rows,
        workspace_dir,
    )?;
    decode_proposals(
        outcome,
        input.parent,
        input.generation,
        &target_fields,
        input.strategy.name(),
    )
}

fn decode_proposals(
    outcome: WorkspaceProposerOutcome,
    parent: &MarlCandidate,
    generation: usize,
    allowed_fields: &[String],
    variant: &str,
) -> Result<Vec<MarlCandidate>> {
    let mut candidates = Vec::new();
    for proposal in outcome.proposals {
        let proposed = proposal.resolved_payload_for_allowed_fields(allowed_fields);
        let mut payload = parent.payload.clone();
        let mut changed = Vec::new();
        for field in allowed_fields {
            let Some(value) = proposed.get(field) else {
                continue;
            };
            if value.trim().is_empty() {
                continue;
            }
            if payload.get(field) != Some(value) {
                payload.insert(field.clone(), value.clone());
                changed.push(field.clone());
            }
        }
        if changed.is_empty() {
            continue;
        }
        let candidate_id = candidate_id(variant, generation, &parent.candidate_id, &payload);
        let mut metadata = Map::new();
        metadata.insert("changed_fields".to_string(), json!(changed));
        metadata.insert("proposal_type".to_string(), json!(proposal.proposal_type));
        metadata.insert("evidence".to_string(), proposal.evidence);
        metadata.insert("proposer_backend".to_string(), json!(&outcome.backend));
        metadata.insert(
            "proposer_runtime_substrate".to_string(),
            json!(&outcome.runtime_substrate),
        );
        metadata.insert(
            "proposer_evidence_warnings".to_string(),
            json!(&outcome.evidence_warnings),
        );
        candidates.push(MarlCandidate {
            candidate_id,
            generation,
            parent_id: Some(parent.candidate_id.clone()),
            payload,
            source: format!("{}_workspace_proposer", outcome.backend),
            rationale: proposal.rationale,
            train_score: None,
            selection_score: None,
            heldout_score: None,
            sensor_frames: Vec::new(),
            metadata,
        });
    }
    if candidates.is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "{} proposer returned no candidate that changed an allowed field",
            variant
        )));
    }
    Ok(candidates)
}

fn gepa_candidate_record(candidate: &MarlCandidate) -> CandidateRecord {
    let score = candidate.selection_basis();
    let rollout_scores = candidate
        .sensor_frames
        .iter()
        .map(|frame| RolloutScore {
            example_id: frame.example_id.clone(),
            task_id: frame.task_id.clone(),
            reward: frame.reward,
        })
        .collect::<Vec<_>>();
    CandidateRecord {
        candidate_id: candidate.candidate_id.clone(),
        payload: candidate.payload.clone(),
        lever_bundle: LeverBundle::from_prompt_payload(
            &candidate.candidate_id,
            candidate.parent_id.clone(),
            &candidate.payload,
        ),
        parent_id: candidate.parent_id.clone(),
        source: candidate.source.clone(),
        status: "evaluated".to_string(),
        minibatch_reward: candidate.train_score.as_ref().map(|value| value.primary),
        train_reward: score.map(|value| value.primary),
        heldout_reward: None,
        minibatch_scores: rollout_scores.clone(),
        train_scores: rollout_scores,
        sensor_frames: candidate.sensor_frames.clone(),
        acceptance_score: score
            .map(|value| serde_json::to_value(value).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
        acceptance_metadata: candidate.metadata.clone(),
    }
}

fn proposer_task_pool_rows(reflection_rows: &[Value]) -> Value {
    let ids = reflection_rows
        .iter()
        .filter_map(|row| row.get("task_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    json!({
        "schema_version": "gepa_task_pools.v1",
        "pareto": {"row_count": 0, "task_ids": [], "rows": []},
        "minibatch": {
            "row_count": reflection_rows.len(),
            "task_ids": ids,
            "rows": reflection_rows,
        },
        "reflection": {
            "row_count": reflection_rows.len(),
            "task_ids": ids,
            "rows": reflection_rows,
        },
        "heldout": {"row_count": 0, "task_ids": [], "rows": []},
    })
}

fn candidate_id(
    variant: &str,
    generation: usize,
    parent_id: &str,
    payload: &BTreeMap<String, String>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(variant.as_bytes());
    digest.update(generation.to_le_bytes());
    digest.update(parent_id.as_bytes());
    digest.update(serde_json::to_vec(payload).unwrap_or_default());
    let hex = format!("{:x}", digest.finalize());
    format!("{variant}_g{generation}_{}", &hex[..12])
}
