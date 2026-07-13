use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{OptimizerError, Result};

use crate::strategy::{ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

const PRIMARY_ARM: &str = "primary";
const CHANNEL_MASKED_ARM: &str = "channel_masked";

pub struct Ic3NetStrategy;

impl MarlStrategy for Ic3NetStrategy {
    fn name(&self) -> &'static str {
        "ic3net"
    }

    fn proposer_guidance(&self) -> Value {
        json!({
            "paper_analogue": "IC3Net learned communication gating",
            "status": "implemented",
            "instruction": "Use matched channel-masking evidence to choose the narrowest event-conditioned speak gate that preserves joint outcome. Remove messages with no positive causal channel value and resolve simultaneous speech with a deterministic single speaker or silence.",
            "objective_order": [
                "primary outcome reward",
                "positive causal channel value",
                "lower congestion",
                "lower interference",
                "fewer messages",
                "fewer message characters"
            ],
            "speak_gate": {
                "type": "event_conditioned",
                "allowed": ["never", "start", "info_change", "blocker", "uncertainty", "always"],
                "events": {
                    "never": "Do not speak when matched masking shows no causal benefit.",
                    "start": "Speak once to establish a load-bearing initial assignment.",
                    "info_change": "Speak only when new information changes another agent's next action.",
                    "blocker": "Speak only to report or clear a blocker that prevents progress.",
                    "uncertainty": "Speak only when uncertainty requires another agent's evidence or decision.",
                    "always": "Speak every opportunity only when every such delivery is causally justified."
                },
                "selection_rule": "Prefer the smallest event set supported by positive primary-minus-channel_masked value on exact matched tasks and checkpoints."
            },
            "collision_policy": {
                "allowed": ["single_speaker", "silence"],
                "single_speaker": "Elect one deterministic speaker with the most decision-relevant information; all others stay silent.",
                "silence": "If no unique load-bearing speaker exists, send nothing rather than colliding, duplicating, or acknowledging."
            },
            "required_evidence": {
                "arms": [PRIMARY_ARM, CHANNEL_MASKED_ARM],
                "receipt": ["arm_id", "intervention_applied", "checkpoint_digest", "channel_mask_applied"],
                "matching": ["task_id", "checkpoint_digest", "candidate_id", "split", "stage"],
                "metrics": [
                    "outcome_success",
                    "coordination_success",
                    "message_action_alignment",
                    "interference_actions",
                    "messages",
                    "message_chars"
                ]
            }
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        let primary = EvaluationArm::primary(&context.candidate.payload);
        let mut metadata = Map::from_iter([
            ("intervention_type".to_string(), json!(CHANNEL_MASKED_ARM)),
            ("matched_arm".to_string(), json!(PRIMARY_ARM)),
        ]);
        for (source, target) in [
            ("task_id", "matched_task_id"),
            ("checkpoint_key", "matched_checkpoint_key"),
        ] {
            if let Some(value) = context.row.get(source) {
                metadata.insert(target.to_string(), value.clone());
            }
        }
        let channel_masked = EvaluationArm {
            arm_id: CHANNEL_MASKED_ARM.to_string(),
            payload: context.candidate.payload.clone(),
            metadata,
        };
        vec![primary, channel_masked]
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        score_matched_pairs(observations)
    }

    fn compare(&self, left: &StrategyScore, right: &StrategyScore) -> Ordering {
        left.primary
            .total_cmp(&right.primary)
            .then_with(|| {
                positive_metric(left, "causal_channel_value")
                    .total_cmp(&positive_metric(right, "causal_channel_value"))
            })
            .then_with(|| {
                lower_metric(right, "congestion_cost")
                    .total_cmp(&lower_metric(left, "congestion_cost"))
            })
            .then_with(|| {
                lower_metric(right, "interference_cost")
                    .total_cmp(&lower_metric(left, "interference_cost"))
            })
            .then_with(|| {
                lower_metric(right, "messages").total_cmp(&lower_metric(left, "messages"))
            })
            .then_with(|| {
                lower_metric(right, "message_chars")
                    .total_cmp(&lower_metric(left, "message_chars"))
            })
    }
}

#[derive(Debug, Deserialize)]
struct InterventionReceipt {
    arm_id: String,
    intervention_applied: bool,
    checkpoint_digest: String,
    #[serde(default)]
    channel_mask_applied: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
struct CoordinationMetrics {
    outcome_success: f64,
    coordination_success: f64,
    message_action_alignment: f64,
    interference_actions: f64,
    messages: f64,
    message_chars: f64,
}

struct ValidatedArm<'a> {
    observation: &'a RolloutObservation,
    checkpoint_digest: String,
    metrics: CoordinationMetrics,
}

#[derive(Default)]
struct MatchedPair<'a> {
    primary: Option<ValidatedArm<'a>>,
    channel_masked: Option<ValidatedArm<'a>>,
}

#[derive(Default)]
struct Aggregate {
    primary_reward: f64,
    masked_reward: f64,
    outcome_success: f64,
    masked_outcome_success: f64,
    coordination_success: f64,
    masked_coordination_success: f64,
    message_action_alignment: f64,
    masked_message_action_alignment: f64,
    interference_cost: f64,
    masked_interference_actions: f64,
    congestion_cost: f64,
    messages: f64,
    message_chars: f64,
}

fn score_matched_pairs(observations: &[RolloutObservation]) -> Result<StrategyScore> {
    let Some(first) = observations.first() else {
        return Err(invariant("IC3Net scoring requires rollout observations"));
    };
    let expected_candidate = first.candidate_id.as_str();
    let expected_split = first.split.as_str();
    let expected_stage = first.stage.as_str();
    let mut pairs = BTreeMap::<String, MatchedPair<'_>>::new();

    for observation in observations {
        if observation.candidate_id != expected_candidate
            || observation.split != expected_split
            || observation.stage != expected_stage
        {
            return Err(invariant(format!(
                "IC3Net score batch mixed candidate/split/stage identities at rollout {}",
                observation.rollout_id
            )));
        }
        let arm = validate_observation(observation)?;
        let pair = pairs.entry(observation.task_id.clone()).or_default();
        let slot = match observation.arm_id.as_str() {
            PRIMARY_ARM => &mut pair.primary,
            CHANNEL_MASKED_ARM => &mut pair.channel_masked,
            other => {
                return Err(invariant(format!(
                    "IC3Net received unsupported arm {other:?} for task {}",
                    observation.task_id
                )))
            }
        };
        if slot.replace(arm).is_some() {
            return Err(invariant(format!(
                "IC3Net received duplicate {} arm for task {}",
                observation.arm_id, observation.task_id
            )));
        }
    }

    let mut aggregate = Aggregate::default();
    let mut matched_diagnostics = Vec::with_capacity(pairs.len());
    for (task_id, pair) in pairs {
        let primary = pair.primary.ok_or_else(|| {
            invariant(format!(
                "IC3Net task {task_id} is missing its required primary arm"
            ))
        })?;
        let masked = pair.channel_masked.ok_or_else(|| {
            invariant(format!(
                "IC3Net task {task_id} is missing its required channel_masked arm"
            ))
        })?;
        require_exact_match(&task_id, &primary, &masked)?;

        let causal_channel_value = primary.observation.reward - masked.observation.reward;
        let causal_outcome_value =
            primary.metrics.outcome_success - masked.metrics.outcome_success;
        let causal_coordination_value =
            primary.metrics.coordination_success - masked.metrics.coordination_success;
        let causal_alignment_value = primary.metrics.message_action_alignment
            - masked.metrics.message_action_alignment;
        let useful_aligned_messages = primary
            .metrics
            .message_action_alignment
            .clamp(0.0, 1.0)
            .min(primary.metrics.messages);
        let congestion_cost = (primary.metrics.messages - useful_aligned_messages).max(0.0);
        let gate_efficiency = if primary.metrics.messages > 0.0 {
            causal_channel_value.max(0.0) / primary.metrics.messages
        } else {
            0.0
        };

        aggregate.primary_reward += primary.observation.reward;
        aggregate.masked_reward += masked.observation.reward;
        aggregate.outcome_success += primary.metrics.outcome_success;
        aggregate.masked_outcome_success += masked.metrics.outcome_success;
        aggregate.coordination_success += primary.metrics.coordination_success;
        aggregate.masked_coordination_success += masked.metrics.coordination_success;
        aggregate.message_action_alignment += primary.metrics.message_action_alignment;
        aggregate.masked_message_action_alignment += masked.metrics.message_action_alignment;
        aggregate.interference_cost += primary.metrics.interference_actions;
        aggregate.masked_interference_actions += masked.metrics.interference_actions;
        aggregate.congestion_cost += congestion_cost;
        aggregate.messages += primary.metrics.messages;
        aggregate.message_chars += primary.metrics.message_chars;

        matched_diagnostics.push(json!({
            "task_id": task_id,
            "checkpoint_digest": primary.checkpoint_digest,
            "primary_rollout_id": primary.observation.rollout_id,
            "channel_masked_rollout_id": masked.observation.rollout_id,
            "primary": {
                "outcome_reward": primary.observation.reward,
                "outcome_success": primary.metrics.outcome_success,
                "coordination_success": primary.metrics.coordination_success,
                "message_action_alignment": primary.metrics.message_action_alignment,
                "interference_actions": primary.metrics.interference_actions,
                "messages": primary.metrics.messages,
                "message_chars": primary.metrics.message_chars
            },
            "channel_masked": {
                "outcome_reward": masked.observation.reward,
                "outcome_success": masked.metrics.outcome_success,
                "coordination_success": masked.metrics.coordination_success,
                "message_action_alignment": masked.metrics.message_action_alignment,
                "interference_actions": masked.metrics.interference_actions,
                "messages": masked.metrics.messages,
                "message_chars": masked.metrics.message_chars
            },
            "causal": {
                "channel_value": causal_channel_value,
                "positive_channel_value": causal_channel_value.max(0.0),
                "outcome_value": causal_outcome_value,
                "coordination_value": causal_coordination_value,
                "alignment_value": causal_alignment_value
            },
            "cost": {
                "congestion": congestion_cost,
                "interference": primary.metrics.interference_actions
            },
            "gate_efficiency": gate_efficiency
        }));
    }

    let denominator = matched_diagnostics.len() as f64;
    let primary_reward = aggregate.primary_reward / denominator;
    let masked_reward = aggregate.masked_reward / denominator;
    let causal_channel_value = primary_reward - masked_reward;
    let outcome_success = aggregate.outcome_success / denominator;
    let masked_outcome_success = aggregate.masked_outcome_success / denominator;
    let coordination_success = aggregate.coordination_success / denominator;
    let masked_coordination_success = aggregate.masked_coordination_success / denominator;
    let message_action_alignment = aggregate.message_action_alignment / denominator;
    let masked_message_action_alignment =
        aggregate.masked_message_action_alignment / denominator;
    let interference_cost = aggregate.interference_cost / denominator;
    let masked_interference_actions = aggregate.masked_interference_actions / denominator;
    let congestion_cost = aggregate.congestion_cost / denominator;
    let messages = aggregate.messages / denominator;
    let message_chars = aggregate.message_chars / denominator;
    let gate_efficiency = if messages > 0.0 {
        causal_channel_value.max(0.0) / messages
    } else {
        0.0
    };

    let metrics = BTreeMap::from([
        ("outcome_reward".to_string(), primary_reward),
        ("masked_outcome_reward".to_string(), masked_reward),
        ("outcome_success".to_string(), outcome_success),
        (
            "masked_outcome_success".to_string(),
            masked_outcome_success,
        ),
        (
            "causal_outcome_value".to_string(),
            outcome_success - masked_outcome_success,
        ),
        (
            "coordination_success".to_string(),
            coordination_success,
        ),
        (
            "masked_coordination_success".to_string(),
            masked_coordination_success,
        ),
        (
            "causal_coordination_value".to_string(),
            coordination_success - masked_coordination_success,
        ),
        (
            "message_action_alignment".to_string(),
            message_action_alignment,
        ),
        (
            "masked_message_action_alignment".to_string(),
            masked_message_action_alignment,
        ),
        (
            "causal_alignment_value".to_string(),
            message_action_alignment - masked_message_action_alignment,
        ),
        ("causal_channel_value".to_string(), causal_channel_value),
        (
            "positive_causal_channel_value".to_string(),
            causal_channel_value.max(0.0),
        ),
        ("congestion_cost".to_string(), congestion_cost),
        ("interference_cost".to_string(), interference_cost),
        (
            "masked_interference_actions".to_string(),
            masked_interference_actions,
        ),
        (
            "causal_interference_delta".to_string(),
            interference_cost - masked_interference_actions,
        ),
        ("gate_efficiency".to_string(), gate_efficiency),
        ("messages".to_string(), messages),
        ("message_chars".to_string(), message_chars),
    ]);
    let gate_pressure = if causal_channel_value > 0.0 {
        "retain only the event classes whose matched channel value is positive"
    } else {
        "narrow the speak gate toward never unless a specific event class proves positive value"
    };
    let collision_pressure = if congestion_cost > 0.0 {
        "elect one deterministic speaker per event; otherwise silence all speakers"
    } else {
        "preserve the current single-speaker or silence behavior"
    };

    Ok(StrategyScore {
        primary: primary_reward,
        metrics,
        diagnostics: json!({
            "schema_version": "ic3net_strategy_diagnostics.v1",
            "matched_pair_count": matched_diagnostics.len(),
            "definitions": {
                "causal_channel_value": "mean primary outcome reward minus mean channel_masked outcome reward on exact task/checkpoint pairs",
                "congestion_cost": "mean primary messages not covered by the bounded message-action-alignment signal",
                "interference_cost": "mean primary interference actions",
                "gate_efficiency": "positive causal channel value per primary message"
            },
            "aggregate": {
                "primary_outcome_reward": primary_reward,
                "channel_masked_outcome_reward": masked_reward,
                "causal_channel_value": causal_channel_value,
                "positive_causal_channel_value": causal_channel_value.max(0.0),
                "causal_outcome_value": outcome_success - masked_outcome_success,
                "causal_coordination_value": coordination_success - masked_coordination_success,
                "causal_alignment_value": message_action_alignment - masked_message_action_alignment,
                "congestion_cost": congestion_cost,
                "interference_cost": interference_cost,
                "causal_interference_delta": interference_cost - masked_interference_actions,
                "gate_efficiency": gate_efficiency,
                "messages": messages,
                "message_chars": message_chars
            },
            "proposer_feedback": {
                "gate_pressure": gate_pressure,
                "collision_pressure": collision_pressure,
                "candidate_speak_gates": ["never", "start", "info_change", "blocker", "uncertainty", "always"],
                "collision_policy": ["single_speaker", "silence"]
            },
            "matched_tasks": matched_diagnostics
        }),
    })
}

fn validate_observation(observation: &RolloutObservation) -> Result<ValidatedArm<'_>> {
    if observation.task_id.trim().is_empty() {
        return Err(invariant(format!(
            "IC3Net rollout {} has an empty task_id",
            observation.rollout_id
        )));
    }
    if !observation.reward.is_finite() {
        return Err(invariant(format!(
            "IC3Net rollout {} has a non-finite reward",
            observation.rollout_id
        )));
    }
    let response_task_id = required_string(&observation.response, "/task_id", observation)?;
    if response_task_id != observation.task_id {
        return Err(invariant(format!(
            "IC3Net rollout {} task_id mismatch: observation={:?}, response={response_task_id:?}",
            observation.rollout_id, observation.task_id
        )));
    }
    let receipt_value = observation
        .response
        .get("intervention_evidence")
        .ok_or_else(|| {
            invariant(format!(
                "IC3Net rollout {} is missing intervention_evidence",
                observation.rollout_id
            ))
        })?;
    let receipt: InterventionReceipt = serde_json::from_value(receipt_value.clone()).map_err(
        |source| {
            invariant(format!(
                "IC3Net rollout {} has an invalid typed intervention_evidence receipt: {source}",
                observation.rollout_id
            ))
        },
    )?;
    if receipt.arm_id != observation.arm_id {
        return Err(invariant(format!(
            "IC3Net rollout {} arm mismatch: observation={:?}, receipt={:?}",
            observation.rollout_id, observation.arm_id, receipt.arm_id
        )));
    }
    if receipt.checkpoint_digest.trim().is_empty() {
        return Err(invariant(format!(
            "IC3Net rollout {} receipt has an empty checkpoint_digest",
            observation.rollout_id
        )));
    }
    let summary_checkpoint = required_string(
        &observation.response,
        "/summary/checkpoint_digest",
        observation,
    )?;
    if summary_checkpoint != receipt.checkpoint_digest {
        return Err(invariant(format!(
            "IC3Net rollout {} checkpoint mismatch between summary and receipt",
            observation.rollout_id
        )));
    }
    let summary_arm = required_string(
        &observation.response,
        "/summary/evaluation_arm",
        observation,
    )?;
    if summary_arm != observation.arm_id {
        return Err(invariant(format!(
            "IC3Net rollout {} evaluation_arm mismatch between summary and observation",
            observation.rollout_id
        )));
    }
    match observation.arm_id.as_str() {
        PRIMARY_ARM if receipt.intervention_applied => {
            return Err(invariant(format!(
                "IC3Net primary rollout {} incorrectly reports intervention_applied=true",
                observation.rollout_id
            )))
        }
        PRIMARY_ARM if receipt.channel_mask_applied == Some(true) => {
            return Err(invariant(format!(
                "IC3Net primary rollout {} incorrectly reports channel_mask_applied=true",
                observation.rollout_id
            )))
        }
        CHANNEL_MASKED_ARM
            if !receipt.intervention_applied || receipt.channel_mask_applied != Some(true) =>
        {
            return Err(invariant(format!(
                "IC3Net channel_masked rollout {} requires intervention_applied=true and channel_mask_applied=true",
                observation.rollout_id
            )))
        }
        PRIMARY_ARM | CHANNEL_MASKED_ARM => {}
        other => {
            return Err(invariant(format!(
                "IC3Net rollout {} has unsupported arm {other:?}",
                observation.rollout_id
            )))
        }
    }

    Ok(ValidatedArm {
        observation,
        checkpoint_digest: receipt.checkpoint_digest,
        metrics: required_coordination_metrics(observation)?,
    })
}

fn require_exact_match(
    task_id: &str,
    primary: &ValidatedArm<'_>,
    masked: &ValidatedArm<'_>,
) -> Result<()> {
    if primary.observation.candidate_id != masked.observation.candidate_id
        || primary.observation.split != masked.observation.split
        || primary.observation.stage != masked.observation.stage
        || primary.checkpoint_digest != masked.checkpoint_digest
    {
        return Err(invariant(format!(
            "IC3Net task {task_id} primary/channel_masked arms are not an exact candidate/split/stage/checkpoint match"
        )));
    }
    if primary.metrics.messages != masked.metrics.messages
        || primary.metrics.message_chars != masked.metrics.message_chars
    {
        return Err(invariant(format!(
            "IC3Net task {task_id} channel_masked intervention changed generated message count or characters"
        )));
    }
    Ok(())
}

fn required_coordination_metrics(
    observation: &RolloutObservation,
) -> Result<CoordinationMetrics> {
    let metrics = CoordinationMetrics {
        outcome_success: required_metric(observation, "outcome_success")?,
        coordination_success: required_metric(observation, "coordination_success")?,
        message_action_alignment: required_metric(observation, "message_action_alignment")?,
        interference_actions: required_metric(observation, "interference_actions")?,
        messages: required_metric(observation, "messages")?,
        message_chars: required_metric(observation, "message_chars")?,
    };
    for (key, value) in [
        ("interference_actions", metrics.interference_actions),
        ("messages", metrics.messages),
        ("message_chars", metrics.message_chars),
    ] {
        if value < 0.0 {
            return Err(invariant(format!(
                "IC3Net rollout {} metric {key:?} must be non-negative",
                observation.rollout_id
            )));
        }
    }
    Ok(metrics)
}

fn required_metric(observation: &RolloutObservation, key: &str) -> Result<f64> {
    let value = observation.metrics.get(key).copied().ok_or_else(|| {
        invariant(format!(
            "IC3Net rollout {} is missing required metric {key:?}",
            observation.rollout_id
        ))
    })?;
    if !value.is_finite() {
        return Err(invariant(format!(
            "IC3Net rollout {} metric {key:?} is non-finite",
            observation.rollout_id
        )));
    }
    Ok(value)
}

fn required_string<'a>(
    value: &'a Value,
    pointer: &str,
    observation: &RolloutObservation,
) -> Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            invariant(format!(
                "IC3Net rollout {} is missing required string {pointer}",
                observation.rollout_id
            ))
        })
}

fn higher_metric(score: &StrategyScore, key: &str) -> f64 {
    score
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY)
}

fn positive_metric(score: &StrategyScore, key: &str) -> f64 {
    let value = higher_metric(score, key);
    if value.is_finite() {
        value.max(0.0)
    } else {
        value
    }
}

fn lower_metric(score: &StrategyScore, key: &str) -> f64 {
    score
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(f64::INFINITY)
}

fn invariant(message: impl Into<String>) -> OptimizerError {
    OptimizerError::Invariant(message.into())
}
