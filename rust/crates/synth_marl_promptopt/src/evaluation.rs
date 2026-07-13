use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    task_identity, ContainerClient, OptimizerError, Result, SensorFrame, SynthOptimizerConfig,
};

use crate::strategy::{ArmContext, MarlStrategy};
use crate::types::{
    BudgetLedger, EvaluationArm, MarlCandidate, RolloutObservation, StrategyScore,
};

#[derive(Clone, Debug, Default)]
pub struct EvaluationBatch {
    pub observations: Vec<RolloutObservation>,
    pub sensor_frames: Vec<SensorFrame>,
}

pub struct EvaluateCandidateInput<'a> {
    pub client: &'a ContainerClient,
    pub config: &'a SynthOptimizerConfig,
    pub strategy: &'a dyn MarlStrategy,
    pub candidate: &'a MarlCandidate,
    pub parent: Option<&'a MarlCandidate>,
    pub seed_payload: &'a BTreeMap<String, String>,
    pub rows: &'a [Value],
    pub split: &'a str,
    pub stage: &'a str,
    pub run_id: &'a str,
    pub variant: &'a str,
    pub budget: &'a mut BudgetLedger,
    pub heldout: bool,
    pub primary_only: bool,
}

pub fn evaluate_candidate(input: EvaluateCandidateInput<'_>) -> Result<EvaluationBatch> {
    let mut batch = EvaluationBatch::default();
    for (row_index, row) in input.rows.iter().enumerate() {
        let task_id = task_identity(row)?;
        let mut arms = if input.primary_only {
            vec![EvaluationArm::primary(&input.candidate.payload)]
        } else {
            input.strategy.evaluation_arms(ArmContext {
                candidate: input.candidate,
                parent: input.parent,
                seed_payload: input.seed_payload,
                row,
            })
        };
        if arms.is_empty() {
            return Err(OptimizerError::Invariant(format!(
                "strategy {} returned no evaluation arms for task {task_id}",
                input.strategy.name()
            )));
        }
        if !arms.iter().any(|arm| arm.arm_id == "primary") {
            arms.insert(0, EvaluationArm::primary(&input.candidate.payload));
        }

        let remaining = if input.heldout {
            input.budget.heldout_remaining()
        } else {
            input.budget.train_remaining()
        };
        if remaining == 0 {
            break;
        }
        if arms.len() > remaining {
            arms.retain(|arm| arm.arm_id == "primary");
        }

        let start = batch.observations.len();
        for (arm_index, arm) in arms.into_iter().enumerate() {
            let budget_ordinal = if input.heldout {
                input.budget.heldout_used
            } else {
                input.budget.train_used
            };
            let admitted = if input.heldout {
                input.budget.admit_heldout()
            } else {
                input.budget.admit_train()
            };
            if !admitted {
                break;
            }
            let rollout_id = rollout_id(
                input.run_id,
                &input.candidate.candidate_id,
                input.stage,
                &task_id,
                &arm.arm_id,
                row_index,
                arm_index,
                budget_ordinal,
            );
            let request = rollout_request(
                input.config,
                input.candidate,
                row,
                input.split,
                input.stage,
                input.variant,
                &rollout_id,
                arm,
            )?;
            let response = input.client.rollout(&request)?;
            let typed = synth_optimizer_platform::RolloutResponse::from_value(response.clone())?;
            typed.validate_for_gepa()?;
            let reward = typed.outcome_reward()?;
            batch.observations.push(RolloutObservation {
                rollout_id,
                candidate_id: input.candidate.candidate_id.clone(),
                task_id: task_id.clone(),
                split: input.split.to_string(),
                stage: input.stage.to_string(),
                arm_id: request
                    .pointer("/metadata/evaluation_arm")
                    .and_then(Value::as_str)
                    .unwrap_or("primary")
                    .to_string(),
                reward,
                metrics: response_metrics(&response),
                response,
            });
        }

        let row_observations = &batch.observations[start..];
        if let Some(primary) = row_observations
            .iter()
            .find(|observation| observation.is_primary())
        {
            let mut proposer_response = primary.response.clone();
            attach_matched_diagnostics(&mut proposer_response, row_observations);
            batch.sensor_frames.push(SensorFrame::from_rollout_response(
                &input.candidate.candidate_id,
                row,
                input.stage,
                &proposer_response,
            )?);
        }
    }
    Ok(batch)
}

pub fn score_batch(
    strategy: &dyn MarlStrategy,
    batch: &EvaluationBatch,
) -> Result<StrategyScore> {
    strategy.score(&batch.observations)
}

fn rollout_request(
    config: &SynthOptimizerConfig,
    candidate: &MarlCandidate,
    row: &Value,
    split: &str,
    stage: &str,
    variant: &str,
    rollout_id: &str,
    arm: EvaluationArm,
) -> Result<Value> {
    let task_id = task_identity(row)?;
    let mut arm_metadata = arm.metadata;
    arm_metadata.insert("evaluation_arm".to_string(), json!(&arm.arm_id));
    let overlay_metadata = arm_metadata.clone();
    let payload = arm.payload;
    let policy = serde_json::to_value(&config.policy)?;
    Ok(json!({
        "rollout_id": rollout_id,
        "trace_correlation_id": rollout_id,
        "submission_mode": "sync",
        "task_id": task_id,
        "task": row,
        "task_payload": row,
        "candidate": payload,
        "candidate_overlay": {
            "candidate": payload,
            "metadata": overlay_metadata,
        },
        "policy": policy,
        "metadata": {
            "algorithm_id": "synth_marl_promptopt.v1",
            "variant": variant,
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "split": split,
            "stage": stage,
            "evaluation_arm": arm.arm_id,
            "diagnostic": arm.arm_id != "primary",
            "arm": arm_metadata,
        },
    }))
}

fn rollout_id(
    run_id: &str,
    candidate_id: &str,
    stage: &str,
    task_id: &str,
    arm_id: &str,
    row_index: usize,
    arm_index: usize,
    budget_ordinal: usize,
) -> String {
    format!(
        "{}_{}_{}_{}_{}_{}_{}_{}",
        sanitize(run_id),
        sanitize(candidate_id),
        sanitize(stage),
        sanitize(task_id),
        sanitize(arm_id),
        row_index,
        arm_index,
        budget_ordinal,
    )
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn response_metrics(response: &Value) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    for source in [
        response.get("summary"),
        response.pointer("/reward_info/metrics"),
        response.pointer("/reward_info/details"),
    ]
    .into_iter()
    .flatten()
    {
        flatten_numeric_metrics("", source, &mut metrics);
    }
    for (canonical, aliases) in [
        ("outcome_success", &["success", "task_success", "outcome_success"][..]),
        (
            "coordination_success",
            &["coordination_success", "coordination_success_rate"],
        ),
        (
            "message_action_alignment",
            &["message_action_alignment", "message_action_alignment_rate", "request_action_alignment"],
        ),
        ("role_consistency", &["role_consistency", "assignment_consistency"]),
        ("role_duplication", &["role_duplication", "duplicate_assignments"]),
        ("invalid_actions", &["invalid_actions", "invalid_action_count"]),
        ("idle_actions", &["idle_actions", "idle_action_count"]),
        (
            "interference_actions",
            &["interference_actions", "interference_action_count"],
        ),
        ("messages", &["messages", "message_count", "messages_delivered"]),
        ("message_chars", &["message_chars", "communication_chars"]),
    ] {
        if let Some(value) = aliases.iter().find_map(|alias| metrics.get(*alias).copied()) {
            metrics.insert(canonical.to_string(), value);
        }
    }
    metrics
}

fn flatten_numeric_metrics(prefix: &str, value: &Value, output: &mut BTreeMap<String, f64>) {
    let Some(object) = value.as_object() else {
        return;
    };
    for (key, child) in object {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if let Some(number) = numeric_value(child) {
            output.entry(key.clone()).or_insert(number);
            output.insert(path, number);
        } else if child.is_object() {
            flatten_numeric_metrics(&path, child, output);
        }
    }
}

fn numeric_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_bool().map(|flag| if flag { 1.0 } else { 0.0 }))
}

fn attach_matched_diagnostics(response: &mut Value, observations: &[RolloutObservation]) {
    let diagnostics = observations
        .iter()
        .map(|observation| {
            json!({
                "arm_id": observation.arm_id,
                "reward": observation.reward,
                "metrics": observation.metrics,
                "summary": observation.response.get("summary"),
            })
        })
        .collect::<Vec<_>>();
    let object = response.as_object_mut().expect("rollout response is an object");
    let actionable = object
        .entry("actionable_side_info")
        .or_insert_with(|| Value::Object(Map::new()));
    if !actionable.is_object() {
        *actionable = Value::Object(Map::new());
    }
    actionable
        .as_object_mut()
        .expect("actionable side info is an object")
        .insert("matched_marl_diagnostics".to_string(), Value::Array(diagnostics));
}
