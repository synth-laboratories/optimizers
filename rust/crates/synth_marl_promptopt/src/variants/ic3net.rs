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
            "paper_analogue": "IC3Net communication gating",
            "status": "implemented",
            "mechanism": "Static prompt gate search inspired by IC3Net; no neural gate is trained or learned at runtime.",
            "instruction": "Use exact primary-minus-channel_masked outcome and coordination evidence to edit per-agent event gates. Cite the agent, event, matched delta, and gate diagnostic for every change; do not return benchmark menus or generic protocol families.",
            "objective_order": [
                "outcome success",
                "outcome reward",
                "positive causal channel value",
                "lower congestion and interference",
                "fewer messages",
                "fewer message characters"
            ],
            "speak_gate": {
                "type": "per_agent_event_conditioned",
                "scope": "Assign one explicit gate to each agent; role templates are valid only when they resolve to a gate per agent.",
                "allowed": ["never", "episode_start", "information_change", "blocker", "uncertainty", "always"],
                "events": {
                    "never": "Do not speak when matched masking shows no causal benefit.",
                    "episode_start": "Speak once before the first action to establish a load-bearing initial assignment.",
                    "information_change": "Speak only when newly observed information changes another agent's next action, role, or belief.",
                    "blocker": "Speak only to report or clear a blocker that prevents progress.",
                    "uncertainty": "Speak only when uncertainty requires another agent's evidence or decision.",
                    "always": "Speak every opportunity only when every such delivery is causally justified."
                },
                "selection_rule": "Prefer the smallest event set supported by positive primary-minus-channel_masked value on exact matched tasks and checkpoints."
            },
            "collision_policy": {
                "allowed": ["single_speaker", "silence_on_collision"],
                "single_speaker": "Elect one deterministic speaker with the most decision-relevant information; all others stay silent.",
                "silence_on_collision": "If more than one gate remains eligible, deliver no message for that event and record the collision."
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
                    "idle_actions",
                    "congestion_events (when reported)",
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
        higher_metric(left, "outcome_success")
            .total_cmp(&higher_metric(right, "outcome_success"))
            .then_with(|| left.primary.total_cmp(&right.primary))
            .then_with(|| {
                higher_metric(left, "positive_causal_channel_value").total_cmp(&higher_metric(
                    right,
                    "positive_causal_channel_value",
                ))
            })
            .then_with(|| {
                lower_metric(right, "congestion_cost")
                    .total_cmp(&lower_metric(left, "congestion_cost"))
            })
            .then_with(|| {
                lower_metric(right, "congestion_interference_cost")
                    .total_cmp(&lower_metric(left, "congestion_interference_cost"))
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
    idle_actions: f64,
    congestion_events: Option<f64>,
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
    interference_actions: f64,
    masked_interference_actions: f64,
    idle_actions: f64,
    masked_idle_actions: f64,
    congestion_events: f64,
    masked_congestion_events: f64,
    congestion_events_reported: f64,
    congestion_interference_cost: f64,
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
    let mut gate_diagnostics = Vec::with_capacity(pairs.len());
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

        let reward_delta = primary.observation.reward - masked.observation.reward;
        let causal_outcome_value =
            primary.metrics.outcome_success - masked.metrics.outcome_success;
        let causal_coordination_value =
            primary.metrics.coordination_success - masked.metrics.coordination_success;
        let causal_channel_value = 0.5 * (causal_outcome_value + causal_coordination_value);
        let positive_causal_channel_value = causal_channel_value.max(0.0);
        let causal_alignment_value = primary.metrics.message_action_alignment
            - masked.metrics.message_action_alignment;
        let congestion_interference_cost = primary.metrics.interference_actions
            + primary.metrics.idle_actions
            + primary.metrics.congestion_events.unwrap_or(0.0);
        let congestion_cost = congestion_interference_cost
            + primary.metrics.messages
            + primary.metrics.message_chars;
        let gating_efficiency = positive_causal_channel_value / (1.0 + congestion_cost);

        aggregate.primary_reward += primary.observation.reward;
        aggregate.masked_reward += masked.observation.reward;
        aggregate.outcome_success += primary.metrics.outcome_success;
        aggregate.masked_outcome_success += masked.metrics.outcome_success;
        aggregate.coordination_success += primary.metrics.coordination_success;
        aggregate.masked_coordination_success += masked.metrics.coordination_success;
        aggregate.message_action_alignment += primary.metrics.message_action_alignment;
        aggregate.masked_message_action_alignment += masked.metrics.message_action_alignment;
        aggregate.interference_actions += primary.metrics.interference_actions;
        aggregate.masked_interference_actions += masked.metrics.interference_actions;
        aggregate.idle_actions += primary.metrics.idle_actions;
        aggregate.masked_idle_actions += masked.metrics.idle_actions;
        aggregate.congestion_events += primary.metrics.congestion_events.unwrap_or(0.0);
        aggregate.masked_congestion_events += masked.metrics.congestion_events.unwrap_or(0.0);
        aggregate.congestion_events_reported +=
            if primary.metrics.congestion_events.is_some() {
                1.0
            } else {
                0.0
            };
        aggregate.congestion_interference_cost += congestion_interference_cost;
        aggregate.congestion_cost += congestion_cost;
        aggregate.messages += primary.metrics.messages;
        aggregate.message_chars += primary.metrics.message_chars;

        matched_diagnostics.push(json!({
            "task_id": &task_id,
            "checkpoint_digest": &primary.checkpoint_digest,
            "primary_rollout_id": &primary.observation.rollout_id,
            "channel_masked_rollout_id": &masked.observation.rollout_id,
            "primary": {
                "outcome_reward": primary.observation.reward,
                "outcome_success": primary.metrics.outcome_success,
                "coordination_success": primary.metrics.coordination_success,
                "message_action_alignment": primary.metrics.message_action_alignment,
                "interference_actions": primary.metrics.interference_actions,
                "idle_actions": primary.metrics.idle_actions,
                "congestion_events": primary.metrics.congestion_events,
                "messages": primary.metrics.messages,
                "message_chars": primary.metrics.message_chars
            },
            "channel_masked": {
                "outcome_reward": masked.observation.reward,
                "outcome_success": masked.metrics.outcome_success,
                "coordination_success": masked.metrics.coordination_success,
                "message_action_alignment": masked.metrics.message_action_alignment,
                "interference_actions": masked.metrics.interference_actions,
                "idle_actions": masked.metrics.idle_actions,
                "congestion_events": masked.metrics.congestion_events,
                "messages": masked.metrics.messages,
                "message_chars": masked.metrics.message_chars
            },
            "primary_minus_channel_masked": {
                "outcome_reward": reward_delta,
                "outcome_success": causal_outcome_value,
                "coordination_success": causal_coordination_value,
                "message_action_alignment": causal_alignment_value,
                "interference_actions": primary.metrics.interference_actions - masked.metrics.interference_actions,
                "idle_actions": primary.metrics.idle_actions - masked.metrics.idle_actions,
                "congestion_events": optional_delta(
                    primary.metrics.congestion_events,
                    masked.metrics.congestion_events,
                ),
                "messages": primary.metrics.messages - masked.metrics.messages,
                "message_chars": primary.metrics.message_chars - masked.metrics.message_chars
            },
            "causal": {
                "channel_value": causal_channel_value,
                "positive_channel_value": positive_causal_channel_value,
                "outcome_value": causal_outcome_value,
                "coordination_value": causal_coordination_value,
                "alignment_value": causal_alignment_value
            },
            "cost": {
                "congestion": congestion_cost,
                "congestion_interference": congestion_interference_cost,
                "messages": primary.metrics.messages,
                "message_chars": primary.metrics.message_chars
            },
            "gating_efficiency": gating_efficiency
        }));
        gate_diagnostics.push(json!({
            "task_id": &task_id,
            "primary": response_gate_diagnostics(&primary.observation.response),
            "channel_masked": response_gate_diagnostics(&masked.observation.response),
            "observed": {
                "channel_mask_applied": true,
                "primary_messages": primary.metrics.messages,
                "primary_message_chars": primary.metrics.message_chars,
                "primary_interference_actions": primary.metrics.interference_actions,
                "primary_idle_actions": primary.metrics.idle_actions,
                "primary_congestion_events": primary.metrics.congestion_events
            }
        }));
    }

    let denominator = matched_diagnostics.len() as f64;
    let primary_reward = aggregate.primary_reward / denominator;
    let masked_reward = aggregate.masked_reward / denominator;
    let outcome_success = aggregate.outcome_success / denominator;
    let masked_outcome_success = aggregate.masked_outcome_success / denominator;
    let coordination_success = aggregate.coordination_success / denominator;
    let masked_coordination_success = aggregate.masked_coordination_success / denominator;
    let causal_outcome_value = outcome_success - masked_outcome_success;
    let causal_coordination_value = coordination_success - masked_coordination_success;
    let causal_channel_value = 0.5 * (causal_outcome_value + causal_coordination_value);
    let positive_causal_channel_value = causal_channel_value.max(0.0);
    let message_action_alignment = aggregate.message_action_alignment / denominator;
    let masked_message_action_alignment =
        aggregate.masked_message_action_alignment / denominator;
    let interference_actions = aggregate.interference_actions / denominator;
    let masked_interference_actions = aggregate.masked_interference_actions / denominator;
    let idle_actions = aggregate.idle_actions / denominator;
    let masked_idle_actions = aggregate.masked_idle_actions / denominator;
    let congestion_events = aggregate.congestion_events / denominator;
    let masked_congestion_events = aggregate.masked_congestion_events / denominator;
    let congestion_events_reported_fraction =
        aggregate.congestion_events_reported / denominator;
    let congestion_interference_cost = aggregate.congestion_interference_cost / denominator;
    let congestion_cost = aggregate.congestion_cost / denominator;
    let messages = aggregate.messages / denominator;
    let message_chars = aggregate.message_chars / denominator;
    let gating_efficiency = positive_causal_channel_value / (1.0 + congestion_cost);

    let metrics = BTreeMap::from([
        ("outcome_reward".to_string(), primary_reward),
        ("masked_outcome_reward".to_string(), masked_reward),
        (
            "causal_reward_delta".to_string(),
            primary_reward - masked_reward,
        ),
        ("outcome_success".to_string(), outcome_success),
        (
            "masked_outcome_success".to_string(),
            masked_outcome_success,
        ),
        (
            "causal_outcome_value".to_string(),
            causal_outcome_value,
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
            causal_coordination_value,
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
            positive_causal_channel_value,
        ),
        ("congestion_cost".to_string(), congestion_cost),
        (
            "congestion_interference_cost".to_string(),
            congestion_interference_cost,
        ),
        ("interference_actions".to_string(), interference_actions),
        ("interference_cost".to_string(), interference_actions),
        (
            "masked_interference_actions".to_string(),
            masked_interference_actions,
        ),
        (
            "causal_interference_delta".to_string(),
            interference_actions - masked_interference_actions,
        ),
        ("idle_actions".to_string(), idle_actions),
        ("masked_idle_actions".to_string(), masked_idle_actions),
        (
            "causal_idle_delta".to_string(),
            idle_actions - masked_idle_actions,
        ),
        ("congestion_events".to_string(), congestion_events),
        (
            "masked_congestion_events".to_string(),
            masked_congestion_events,
        ),
        (
            "congestion_events_reported_fraction".to_string(),
            congestion_events_reported_fraction,
        ),
        ("gating_efficiency".to_string(), gating_efficiency),
        ("gate_efficiency".to_string(), gating_efficiency),
        ("messages".to_string(), messages),
        ("message_chars".to_string(), message_chars),
    ]);
    let gate_pressure = if causal_channel_value > 0.0 {
        "retain only the event classes whose matched channel value is positive"
    } else {
        "narrow the speak gate toward never unless a specific event class proves positive value"
    };
    let collision_pressure = if congestion_interference_cost > 0.0 {
        "elect one deterministic speaker per event; otherwise silence all speakers"
    } else {
        "preserve the current single-speaker or silence-on-collision behavior"
    };

    Ok(StrategyScore {
        primary: primary_reward,
        metrics,
        diagnostics: json!({
            "schema_version": "ic3net_strategy_diagnostics.v1",
            "matched_pair_count": matched_diagnostics.len(),
            "definitions": {
                "causal_channel_value": "0.5 * ((primary outcome success - masked outcome success) + (primary coordination success - masked coordination success)) on exact task/checkpoint pairs",
                "positive_causal_channel_value": "max(causal channel value, 0)",
                "congestion_interference_cost": "primary interference actions + idle actions + optional congestion events",
                "congestion_cost": "congestion/interference cost + primary messages + primary message characters",
                "gating_efficiency": "positive causal channel value / (1 + congestion cost)"
            },
            "aggregate": {
                "primary_outcome_reward": primary_reward,
                "channel_masked_outcome_reward": masked_reward,
                "causal_reward_delta": primary_reward - masked_reward,
                "causal_channel_value": causal_channel_value,
                "positive_causal_channel_value": positive_causal_channel_value,
                "causal_outcome_value": causal_outcome_value,
                "causal_coordination_value": causal_coordination_value,
                "causal_alignment_value": message_action_alignment - masked_message_action_alignment,
                "congestion_cost": congestion_cost,
                "congestion_interference_cost": congestion_interference_cost,
                "interference_actions": interference_actions,
                "causal_interference_delta": interference_actions - masked_interference_actions,
                "idle_actions": idle_actions,
                "causal_idle_delta": idle_actions - masked_idle_actions,
                "congestion_events": congestion_events,
                "congestion_events_reported_fraction": congestion_events_reported_fraction,
                "gating_efficiency": gating_efficiency,
                "messages": messages,
                "message_chars": message_chars
            },
            "proposer_feedback": {
                "gate_pressure": gate_pressure,
                "collision_pressure": collision_pressure,
                "required_edit": "Cite matched task deltas and per-agent gate diagnostics for each concrete gate change; do not emit a benchmark menu.",
                "event_gate_contract": ["never", "episode_start", "information_change", "blocker", "uncertainty", "always"],
                "collision_policy": ["single_speaker", "silence_on_collision"]
            },
            "matched_deltas": matched_diagnostics,
            "gate_diagnostics": gate_diagnostics
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
    if primary.metrics.congestion_events.is_some() != masked.metrics.congestion_events.is_some() {
        return Err(invariant(format!(
            "IC3Net task {task_id} reports congestion_events on only one matched arm"
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
        idle_actions: required_metric(observation, "idle_actions")?,
        congestion_events: optional_nonnegative_metric(observation, "congestion_events")?,
        messages: required_metric(observation, "messages")?,
        message_chars: required_metric(observation, "message_chars")?,
    };
    for (key, value) in [
        ("interference_actions", metrics.interference_actions),
        ("idle_actions", metrics.idle_actions),
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

fn optional_nonnegative_metric(
    observation: &RolloutObservation,
    key: &str,
) -> Result<Option<f64>> {
    let Some(value) = observation.metrics.get(key).copied() else {
        return Ok(None);
    };
    if !value.is_finite() || value < 0.0 {
        return Err(invariant(format!(
            "IC3Net rollout {} metric {key:?} must be finite and non-negative",
            observation.rollout_id
        )));
    }
    Ok(Some(value))
}

fn response_gate_diagnostics(response: &Value) -> Value {
    json!({
        "reported_gate_diagnostics": response.pointer("/summary/gate_diagnostics"),
        "executed_protocol": response.pointer("/trace/protocol"),
        "recognized_directives": response.pointer("/actionable_side_info/recognized_directives"),
        "ignored_directives": response.pointer("/actionable_side_info/ignored_directives"),
        "failure_signals": response.pointer("/actionable_side_info/failure_signals"),
        "per_agent_contributions": response.pointer("/reward_info/details/per_agent_contributions"),
        "masked_delivery_count": response.pointer("/summary/masked_delivery_count")
    })
}

fn optional_delta(primary: Option<f64>, masked: Option<f64>) -> Option<f64> {
    primary.zip(masked).map(|(primary, masked)| primary - masked)
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
