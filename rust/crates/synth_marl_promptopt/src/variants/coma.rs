use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{OptimizerError, Result};

use crate::strategy::{primary_mean_score, row_roles, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

const ROLE_ABLATION_PREFIX: &str = "role_ablation::";

pub struct ComaStrategy;

impl MarlStrategy for ComaStrategy {
    fn name(&self) -> &'static str {
        "coma"
    }

    fn proposer_guidance(&self) -> Value {
        json!({
            "paper_analogue": "COMA counterfactual multi-agent credit assignment",
            "mechanism": "Prompt-level same-checkpoint counterfactual credit assignment; this is not neural COMA and does not train a centralized critic.",
            "instruction": "Use factual-minus-counterfactual channel and per-role credit to identify prompt edits that materially improve the joint outcome.",
            "selection_order": [
                "primary_outcome_success",
                "primary_reward",
                "minimum_role_or_channel_success_contribution",
                "mean_contribution_vector_outcome_success_then_reward",
                "lower_messages",
                "lower_message_chars",
                "lower_invalid_actions"
            ]
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        let mut arms = vec![EvaluationArm::primary(&context.candidate.payload)];

        arms.push(EvaluationArm {
            arm_id: "channel_masked".to_string(),
            payload: context.candidate.payload.clone(),
            metadata: Map::from_iter([
                ("evaluation_arm".to_string(), json!("channel_masked")),
                (
                    "counterfactual_replay".to_string(),
                    json!("matched_checkpoint"),
                ),
            ]),
        });

        let (baseline_key, baseline_payload, baseline_candidate_id) = match context.parent {
            Some(parent) => (
                "parent_candidate",
                json!(&parent.payload),
                Some(parent.candidate_id.as_str()),
            ),
            None => ("seed_candidate", json!(context.seed_payload), None),
        };
        let mut seen_roles = BTreeSet::new();
        for role in row_roles(context.row) {
            let role = role.trim().to_ascii_lowercase();
            if role.is_empty() || !seen_roles.insert(role.clone()) {
                continue;
            }
            let arm_id = format!("{ROLE_ABLATION_PREFIX}{role}");
            let mut metadata = Map::from_iter([
                ("evaluation_arm".to_string(), json!(&arm_id)),
                ("ablate_role".to_string(), json!(&role)),
                (
                    "counterfactual_replay".to_string(),
                    json!("matched_checkpoint"),
                ),
                (
                    "ablation_baseline_source".to_string(),
                    json!(baseline_key),
                ),
                (baseline_key.to_string(), baseline_payload.clone()),
            ]);
            if let Some(candidate_id) = baseline_candidate_id {
                metadata.insert(
                    "ablation_baseline_candidate_id".to_string(),
                    json!(candidate_id),
                );
            }
            arms.push(EvaluationArm {
                arm_id,
                payload: context.candidate.payload.clone(),
                metadata,
            });
        }
        arms
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        score_observations(observations)
    }

    fn compare(&self, left: &StrategyScore, right: &StrategyScore) -> Ordering {
        outcome_success(left)
            .total_cmp(&outcome_success(right))
            .then_with(|| primary_reward(left).total_cmp(&primary_reward(right)))
            .then_with(|| {
                left.metric("minimum_role_channel_success_contribution")
                    .total_cmp(&right.metric("minimum_role_channel_success_contribution"))
            })
            .then_with(|| {
                left.metric("mean_contribution")
                    .total_cmp(&right.metric("mean_contribution"))
            })
            .then_with(|| {
                left.metric("mean_reward_contribution")
                    .total_cmp(&right.metric("mean_reward_contribution"))
            })
            .then_with(|| right.metric("messages").total_cmp(&left.metric("messages")))
            .then_with(|| {
                right
                    .metric("message_chars")
                    .total_cmp(&left.metric("message_chars"))
            })
            .then_with(|| {
                right
                    .metric("invalid_actions")
                    .total_cmp(&left.metric("invalid_actions"))
            })
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct TaskKey {
    candidate_id: String,
    split: String,
    stage: String,
    task_id: String,
}

impl TaskKey {
    fn from_observation(observation: &RolloutObservation) -> Self {
        Self {
            candidate_id: observation.candidate_id.clone(),
            split: observation.split.clone(),
            stage: observation.stage.clone(),
            task_id: observation.task_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CreditAxis {
    ChannelMasked,
    RoleAblation { role: String },
}

impl CreditAxis {
    fn arm_id(&self) -> String {
        match self {
            Self::ChannelMasked => "channel_masked".to_string(),
            Self::RoleAblation { role } => format!("{ROLE_ABLATION_PREFIX}{role}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct OutcomeVector {
    outcome_success: f64,
    reward: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ContributionVector {
    outcome_success: f64,
    reward: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AppliedInterventionReceipt {
    arm_id: String,
    intervention_applied: bool,
    checkpoint_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel_mask_applied: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ablate_role: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct CounterfactualCredit {
    axis: CreditAxis,
    counterfactual: OutcomeVector,
    contribution: ContributionVector,
    receipt: AppliedInterventionReceipt,
}

#[derive(Clone, Debug, Serialize)]
struct TaskCreditVector {
    task_key: TaskKey,
    checkpoint_digest: String,
    factual: OutcomeVector,
    counterfactuals: Vec<CounterfactualCredit>,
}

#[derive(Clone, Debug)]
struct AxisSample {
    task_key: TaskKey,
    contribution: ContributionVector,
}

#[derive(Clone, Debug, Serialize)]
struct AxisCreditSummary {
    axis: CreditAxis,
    task_count: usize,
    minimum_task_success_contribution: f64,
    mean_success_contribution: f64,
    mean_reward_contribution: f64,
}

#[derive(Clone, Debug, Serialize)]
struct FailingAxis {
    axis: CreditAxis,
    failing_task_keys: Vec<TaskKey>,
    minimum_task_success_contribution: f64,
    mean_success_contribution: f64,
    mean_reward_contribution: f64,
}

#[derive(Debug, Serialize)]
struct ComaDiagnostics {
    schema_version: &'static str,
    method: &'static str,
    neural_coma: bool,
    task_count: usize,
    axis_count: usize,
    primary: OutcomeVector,
    minimum_role_channel_success_contribution: f64,
    mean_success_contribution: f64,
    mean_reward_contribution: f64,
    credit_vectors: Vec<TaskCreditVector>,
    axis_summaries: Vec<AxisCreditSummary>,
    failing_axes: Vec<FailingAxis>,
}

fn score_observations(observations: &[RolloutObservation]) -> Result<StrategyScore> {
    let mut primary = BTreeMap::new();
    let mut channel_masked = BTreeMap::new();
    let mut role_ablations: BTreeMap<String, BTreeMap<TaskKey, &RolloutObservation>> =
        BTreeMap::new();

    for observation in observations {
        let task_key = TaskKey::from_observation(observation);
        match observation.arm_id.as_str() {
            "primary" => insert_unique(&mut primary, task_key, observation, "primary")?,
            "channel_masked" => insert_unique(
                &mut channel_masked,
                task_key,
                observation,
                "channel_masked",
            )?,
            arm_id if arm_id.starts_with(ROLE_ABLATION_PREFIX) => {
                let role = arm_id
                    .strip_prefix(ROLE_ABLATION_PREFIX)
                    .map(str::trim)
                    .filter(|role| !role.is_empty())
                    .ok_or_else(|| {
                        invariant(format!(
                            "COMA observation {} has an empty role-ablation arm",
                            observation.rollout_id
                        ))
                    })?
                    .to_ascii_lowercase();
                insert_unique(
                    role_ablations.entry(role.clone()).or_default(),
                    task_key,
                    observation,
                    &format!("role_ablation::{role}"),
                )?;
            }
            other => {
                return Err(invariant(format!(
                    "COMA score received unsupported evaluation arm {other:?} on rollout {}",
                    observation.rollout_id
                )))
            }
        }
    }

    if primary.is_empty() {
        return Err(invariant("COMA score requires at least one factual primary arm"));
    }
    if role_ablations.is_empty() {
        return Err(invariant(
            "COMA score requires at least one role-ablation intervention",
        ));
    }
    require_exact_task_keys("channel_masked", &primary, &channel_masked)?;
    require_role_coverage(&primary, &role_ablations)?;

    let mut credit_vectors = Vec::with_capacity(primary.len());
    let mut axis_samples: BTreeMap<CreditAxis, Vec<AxisSample>> = BTreeMap::new();
    for (task_key, factual_observation) in &primary {
        let factual_receipt = factual_receipt(factual_observation)?;
        let factual = outcome_vector(factual_observation)?;
        let mut counterfactuals = Vec::with_capacity(1 + role_ablations.len());

        let channel_observation = channel_masked.get(task_key).ok_or_else(|| {
            invariant(format!(
                "COMA channel intervention is missing for task key {task_key:?}"
            ))
        })?;
        let channel_axis = CreditAxis::ChannelMasked;
        let channel_credit = counterfactual_credit(
            task_key,
            &channel_axis,
            factual,
            &factual_receipt.checkpoint_digest,
            channel_observation,
        )?;
        axis_samples
            .entry(channel_axis)
            .or_default()
            .push(AxisSample {
                task_key: task_key.clone(),
                contribution: channel_credit.contribution,
            });
        counterfactuals.push(channel_credit);

        for (role, observations) in &role_ablations {
            let Some(observation) = observations.get(task_key) else {
                continue;
            };
            let axis = CreditAxis::RoleAblation { role: role.clone() };
            let credit = counterfactual_credit(
                task_key,
                &axis,
                factual,
                &factual_receipt.checkpoint_digest,
                observation,
            )?;
            axis_samples
                .entry(axis)
                .or_default()
                .push(AxisSample {
                    task_key: task_key.clone(),
                    contribution: credit.contribution,
                });
            counterfactuals.push(credit);
        }

        credit_vectors.push(TaskCreditVector {
            task_key: task_key.clone(),
            checkpoint_digest: factual_receipt.checkpoint_digest,
            factual,
            counterfactuals,
        });
    }

    let mut axis_summaries = Vec::with_capacity(axis_samples.len());
    let mut failing_axes = Vec::new();
    for (axis, samples) in axis_samples {
        let denominator = samples.len() as f64;
        let minimum_task_success_contribution = samples
            .iter()
            .map(|sample| sample.contribution.outcome_success)
            .reduce(f64::min)
            .unwrap_or(0.0);
        let mean_success_contribution = samples
            .iter()
            .map(|sample| sample.contribution.outcome_success)
            .sum::<f64>()
            / denominator;
        let mean_reward_contribution = samples
            .iter()
            .map(|sample| sample.contribution.reward)
            .sum::<f64>()
            / denominator;
        let failing_task_keys = samples
            .iter()
            .filter(|sample| {
                sample.contribution.outcome_success <= 0.0
                    || sample.contribution.reward <= 0.0
            })
            .map(|sample| sample.task_key.clone())
            .collect::<Vec<_>>();
        if !failing_task_keys.is_empty() {
            failing_axes.push(FailingAxis {
                axis: axis.clone(),
                failing_task_keys,
                minimum_task_success_contribution,
                mean_success_contribution,
                mean_reward_contribution,
            });
        }
        axis_summaries.push(AxisCreditSummary {
            axis,
            task_count: samples.len(),
            minimum_task_success_contribution,
            mean_success_contribution,
            mean_reward_contribution,
        });
    }

    let axis_count = axis_summaries.len();
    let axis_denominator = axis_count as f64;
    let minimum_role_channel_success_contribution = axis_summaries
        .iter()
        .map(|summary| summary.mean_success_contribution)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let mean_success_contribution = axis_summaries
        .iter()
        .map(|summary| summary.mean_success_contribution)
        .sum::<f64>()
        / axis_denominator;
    let mean_reward_contribution = axis_summaries
        .iter()
        .map(|summary| summary.mean_reward_contribution)
        .sum::<f64>()
        / axis_denominator;
    let task_count = credit_vectors.len();
    let task_denominator = task_count as f64;
    let primary_outcome_success = credit_vectors
        .iter()
        .map(|vector| vector.factual.outcome_success)
        .sum::<f64>()
        / task_denominator;
    let primary_reward = credit_vectors
        .iter()
        .map(|vector| vector.factual.reward)
        .sum::<f64>()
        / task_denominator;

    let diagnostics = serde_json::to_value(ComaDiagnostics {
        schema_version: "synth_marl_promptopt.coma_credit.v1",
        method: "prompt_level_same_checkpoint_factual_minus_counterfactual",
        neural_coma: false,
        task_count,
        axis_count,
        primary: OutcomeVector {
            outcome_success: primary_outcome_success,
            reward: primary_reward,
        },
        minimum_role_channel_success_contribution,
        mean_success_contribution,
        mean_reward_contribution,
        credit_vectors,
        axis_summaries,
        failing_axes,
    })?;

    let mut score = primary_mean_score(observations);
    score.primary = primary_outcome_success;
    score
        .metrics
        .insert("outcome_success".to_string(), primary_outcome_success);
    score
        .metrics
        .insert("outcome_reward".to_string(), primary_reward);
    score.metrics.insert(
        "minimum_role_channel_success_contribution".to_string(),
        minimum_role_channel_success_contribution,
    );
    score.metrics.insert(
        "minimum_success_contribution".to_string(),
        minimum_role_channel_success_contribution,
    );
    score.metrics.insert(
        "mean_contribution".to_string(),
        mean_success_contribution,
    );
    score.metrics.insert(
        "mean_success_contribution".to_string(),
        mean_success_contribution,
    );
    score.metrics.insert(
        "mean_reward_contribution".to_string(),
        mean_reward_contribution,
    );
    score.diagnostics = diagnostics;
    Ok(score)
}

fn insert_unique<'a>(
    observations: &mut BTreeMap<TaskKey, &'a RolloutObservation>,
    task_key: TaskKey,
    observation: &'a RolloutObservation,
    arm: &str,
) -> Result<()> {
    if observations.contains_key(&task_key) {
        return Err(invariant(format!(
            "COMA {arm} has duplicate observations for task key {task_key:?}; exact pairing requires one observation per arm and task"
        )));
    }
    observations.insert(task_key, observation);
    Ok(())
}

fn require_exact_task_keys(
    arm: &str,
    factual: &BTreeMap<TaskKey, &RolloutObservation>,
    counterfactual: &BTreeMap<TaskKey, &RolloutObservation>,
) -> Result<()> {
    let factual_keys = factual.keys().cloned().collect::<BTreeSet<_>>();
    let counterfactual_keys = counterfactual.keys().cloned().collect::<BTreeSet<_>>();
    if factual_keys == counterfactual_keys {
        return Ok(());
    }
    let missing = factual_keys
        .difference(&counterfactual_keys)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = counterfactual_keys
        .difference(&factual_keys)
        .cloned()
        .collect::<Vec<_>>();
    Err(invariant(format!(
        "COMA {arm} task-key set differs from factual primary: missing={missing:?} unexpected={unexpected:?}"
    )))
}

fn require_role_coverage(
    factual: &BTreeMap<TaskKey, &RolloutObservation>,
    role_ablations: &BTreeMap<String, BTreeMap<TaskKey, &RolloutObservation>>,
) -> Result<()> {
    let factual_keys = factual.keys().cloned().collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    for (role, observations) in role_ablations {
        let unexpected = observations
            .keys()
            .filter(|task_key| !factual_keys.contains(*task_key))
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected.is_empty() {
            return Err(invariant(format!(
                "COMA role_ablation::{role} contains task keys without factual primary arms: {unexpected:?}"
            )));
        }
        covered.extend(observations.keys().cloned());
    }
    let missing = factual_keys.difference(&covered).cloned().collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(invariant(format!(
            "COMA requires at least one valid role-ablation intervention per factual task; missing={missing:?}"
        )))
    }
}

fn factual_receipt(observation: &RolloutObservation) -> Result<AppliedInterventionReceipt> {
    let receipt = parse_receipt(observation)?;
    if receipt.intervention_applied {
        return Err(invariant(format!(
            "COMA factual rollout {} reports that an intervention was applied",
            observation.rollout_id
        )));
    }
    Ok(receipt)
}

fn counterfactual_credit(
    task_key: &TaskKey,
    axis: &CreditAxis,
    factual: OutcomeVector,
    factual_checkpoint: &str,
    observation: &RolloutObservation,
) -> Result<CounterfactualCredit> {
    let receipt = parse_receipt(observation)?;
    let expected_arm = axis.arm_id();
    if receipt.arm_id != expected_arm {
        return Err(invariant(format!(
            "COMA response receipt arm mismatch for task key {task_key:?}: expected={expected_arm:?} reported={:?}",
            receipt.arm_id
        )));
    }
    if !receipt.intervention_applied {
        return Err(invariant(format!(
            "COMA response receipt says intervention {expected_arm:?} was not applied for task key {task_key:?}"
        )));
    }
    match axis {
        CreditAxis::ChannelMasked => {
            if receipt.channel_mask_applied != Some(true) {
                return Err(invariant(format!(
                    "COMA channel response receipt does not say channel_mask_applied=true for task key {task_key:?}"
                )));
            }
        }
        CreditAxis::RoleAblation { role } => {
            let reported_role = receipt.ablate_role.as_deref().ok_or_else(|| {
                invariant(format!(
                    "COMA role-ablation response receipt is missing ablate_role for task key {task_key:?}"
                ))
            })?;
            if !reported_role.eq_ignore_ascii_case(role) {
                return Err(invariant(format!(
                    "COMA role-ablation response receipt role mismatch for task key {task_key:?}: expected={role:?} reported={reported_role:?}"
                )));
            }
        }
    }
    if receipt.checkpoint_digest != factual_checkpoint {
        return Err(invariant(format!(
            "COMA same-checkpoint invariant failed for task key {task_key:?} axis={expected_arm:?}: factual={factual_checkpoint:?} counterfactual={:?}",
            receipt.checkpoint_digest
        )));
    }
    let counterfactual = outcome_vector(observation)?;
    Ok(CounterfactualCredit {
        axis: axis.clone(),
        counterfactual,
        contribution: ContributionVector {
            outcome_success: factual.outcome_success - counterfactual.outcome_success,
            reward: factual.reward - counterfactual.reward,
        },
        receipt,
    })
}

fn parse_receipt(observation: &RolloutObservation) -> Result<AppliedInterventionReceipt> {
    let receipt = observation
        .response
        .get("intervention_evidence")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invariant(format!(
                "COMA rollout {} is missing the GameBench intervention_evidence receipt",
                observation.rollout_id
            ))
        })?;
    let arm_id = required_receipt_string(receipt, "arm_id", observation)?;
    if arm_id != observation.arm_id {
        return Err(invariant(format!(
            "COMA rollout {} response receipt arm {arm_id:?} does not match requested arm {:?}",
            observation.rollout_id, observation.arm_id
        )));
    }
    let intervention_applied = receipt
        .get("intervention_applied")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            invariant(format!(
                "COMA rollout {} response receipt is missing boolean intervention_applied",
                observation.rollout_id
            ))
        })?;
    let checkpoint_digest =
        required_receipt_string(receipt, "checkpoint_digest", observation)?;
    Ok(AppliedInterventionReceipt {
        arm_id,
        intervention_applied,
        checkpoint_digest,
        channel_mask_applied: receipt.get("channel_mask_applied").and_then(Value::as_bool),
        ablate_role: receipt
            .get("ablate_role")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

fn required_receipt_string(
    receipt: &Map<String, Value>,
    field: &str,
    observation: &RolloutObservation,
) -> Result<String> {
    receipt
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            invariant(format!(
                "COMA rollout {} response receipt is missing non-empty {field}",
                observation.rollout_id
            ))
        })
}

fn outcome_vector(observation: &RolloutObservation) -> Result<OutcomeVector> {
    let outcome_success = observation
        .metrics
        .get("outcome_success")
        .copied()
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            invariant(format!(
                "COMA rollout {} is missing finite outcome_success",
                observation.rollout_id
            ))
        })?;
    if !observation.reward.is_finite() {
        return Err(invariant(format!(
            "COMA rollout {} has non-finite reward",
            observation.rollout_id
        )));
    }
    Ok(OutcomeVector {
        outcome_success,
        reward: observation.reward,
    })
}

fn outcome_success(score: &StrategyScore) -> f64 {
    score
        .metrics
        .get("outcome_success")
        .copied()
        .unwrap_or(score.primary)
}

fn primary_reward(score: &StrategyScore) -> f64 {
    score
        .metrics
        .get("outcome_reward")
        .copied()
        .unwrap_or(score.primary)
}

fn invariant(message: impl Into<String>) -> OptimizerError {
    OptimizerError::Invariant(message.into())
}
