use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::Value;
use synth_optimizer_platform::{PromptProgram, Result};

use crate::types::{EvaluationArm, MarlCandidate, RolloutObservation, StrategyScore};

pub struct ArmContext<'a> {
    pub candidate: &'a MarlCandidate,
    pub parent: Option<&'a MarlCandidate>,
    pub seed_payload: &'a BTreeMap<String, String>,
    pub row: &'a Value,
}

pub trait MarlStrategy: Send + Sync {
    fn name(&self) -> &'static str;

    fn proposer_guidance(&self) -> Value;

    fn target_fields(&self, _generation: usize, program: &PromptProgram) -> Vec<String> {
        program.mutable_field_ids()
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm>;

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore>;

    fn compare(&self, left: &StrategyScore, right: &StrategyScore) -> Ordering {
        left.primary.total_cmp(&right.primary)
    }
}

pub fn primary_only_arms(candidate: &MarlCandidate) -> Vec<EvaluationArm> {
    vec![EvaluationArm::primary(&candidate.payload)]
}

pub fn primary_mean_score(observations: &[RolloutObservation]) -> StrategyScore {
    let primary = observations
        .iter()
        .filter(|observation| observation.is_primary())
        .collect::<Vec<_>>();
    let denominator = primary.len().max(1) as f64;
    let reward = primary.iter().map(|observation| observation.reward).sum::<f64>() / denominator;
    let mut metrics = BTreeMap::new();
    metrics.insert("outcome_reward".to_string(), reward);
    for key in [
        "outcome_success",
        "coordination_success",
        "message_action_alignment",
        "role_consistency",
        "role_duplication",
        "invalid_actions",
        "idle_actions",
        "interference_actions",
        "messages",
        "message_chars",
    ] {
        let value = primary
            .iter()
            .map(|observation| observation.metric(key))
            .sum::<f64>()
            / denominator;
        metrics.insert(key.to_string(), value);
    }
    StrategyScore {
        primary: reward,
        metrics,
        diagnostics: Value::Null,
    }
}

pub fn row_roles(row: &Value) -> Vec<String> {
    for pointer in ["/roles", "/actors", "/task_payload/roles", "/metadata/roles"] {
        if let Some(values) = row.pointer(pointer).and_then(Value::as_array) {
            let roles = values
                .iter()
                .filter_map(|value| {
                    value
                        .as_str()
                        .map(str::to_string)
                        .or_else(|| value.get("role").and_then(Value::as_str).map(str::to_string))
                })
                .collect::<Vec<_>>();
            if !roles.is_empty() {
                return roles;
            }
        }
    }
    Vec::new()
}
