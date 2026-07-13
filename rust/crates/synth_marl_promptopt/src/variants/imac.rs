use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::json;
use synth_optimizer_platform::{OptimizerError, Result};

use crate::strategy::{primary_only_arms, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

const ENVELOPE_MESSAGES: f64 = 8.0;
const ENVELOPE_MESSAGE_CHARS: f64 = 1_920.0;
const ENVELOPE_MAX_MESSAGE_LENGTH: f64 = 240.0;
const ENVELOPE_LINK_TRANSMISSIONS: f64 = 24.0;
const ENVELOPE_GROUNDED_SIGNALS: f64 = 8.0;
const ENVELOPE_INTERFERENCE_ACTIONS: f64 = 0.0;
const ENVELOPE_INVALID_ACTIONS: f64 = 0.0;

const RESOURCE_SPECS: [ResourceSpec; 7] = [
    ResourceSpec {
        axis: ResourceAxis::Messages,
        aliases: &["messages", "message_count", "messages_delivered"],
        limit: ENVELOPE_MESSAGES,
        required: true,
        aggregation: Aggregation::Mean,
    },
    ResourceSpec {
        axis: ResourceAxis::MessageChars,
        aliases: &["message_chars", "communication_chars"],
        limit: ENVELOPE_MESSAGE_CHARS,
        required: true,
        aggregation: Aggregation::Mean,
    },
    ResourceSpec {
        axis: ResourceAxis::MaxMessageLength,
        aliases: &[
            "max_message_length",
            "max_message_chars_observed",
            "max_message_chars",
        ],
        limit: ENVELOPE_MAX_MESSAGE_LENGTH,
        required: false,
        aggregation: Aggregation::Maximum,
    },
    ResourceSpec {
        axis: ResourceAxis::LinkTransmissions,
        aliases: &["link_transmissions", "link_transmission_count"],
        limit: ENVELOPE_LINK_TRANSMISSIONS,
        required: false,
        aggregation: Aggregation::Mean,
    },
    ResourceSpec {
        axis: ResourceAxis::GroundedSignals,
        aliases: &[
            "grounded_signals",
            "grounded_signal_count",
            "grounded_button_activations",
        ],
        limit: ENVELOPE_GROUNDED_SIGNALS,
        required: false,
        aggregation: Aggregation::Mean,
    },
    ResourceSpec {
        axis: ResourceAxis::InterferenceActions,
        aliases: &["interference_actions", "interference_action_count"],
        limit: ENVELOPE_INTERFERENCE_ACTIONS,
        required: true,
        aggregation: Aggregation::Mean,
    },
    ResourceSpec {
        axis: ResourceAxis::InvalidActions,
        aliases: &["invalid_actions", "invalid_action_count"],
        limit: ENVELOPE_INVALID_ACTIONS,
        required: true,
        aggregation: Aggregation::Mean,
    },
];

pub struct ImacStrategy;

impl MarlStrategy for ImacStrategy {
    fn name(&self) -> &'static str {
        "imac"
    }

    fn proposer_guidance(&self) -> serde_json::Value {
        json!({
            "paper_analogue": "IMAC information bottleneck for communication",
            "status": "implemented",
            "mechanism": "Prompt-level IMAC inspiration: search bounded communication-policy and role-prompt mutations under one optimizer-wide resource envelope. This does not estimate mutual information, optimize a neural MI objective, or train a neural communication policy.",
            "instruction": "Preserve the highest outcome tier and reward first. Then apply common-envelope pressure by mutating when agents speak, message compactness, recipient fan-out, grounded request/handoff behavior, follower replies, and role ownership. Prefer event-triggered load-bearing communication; remove redundant acknowledgements, broadcasts, interference, and invalid actions.",
            "objective_order": [
                "outcome success tier",
                "outcome reward",
                "common-envelope feasibility",
                "Pareto dominance on normalized resource use"
            ],
            "communication_bottleneck_mutations": [
                "tighten SPEAK from always to event-triggered or silent only when outcome evidence permits",
                "lower MAX_CHARS while preserving the load-bearing intent",
                "replace broadcast or repeated replies with the narrowest useful recipient path",
                "make requests and handoffs grounded, single-purpose, and action-aligned",
                "remove duplicated role ownership that causes interference or invalid actions"
            ],
            "common_envelope": common_envelope_json(),
            "envelope_policy": {
                "scope": "optimizer_wide_per_rollout",
                "candidate_declared_budgets_are_authoritative": false,
                "missing_required_resource_metric": "fail_closed",
                "missing_optional_resource_metric": "report_unavailable_and_omit_that_axis"
            },
            "pareto_policy": "Do not scalarize crossed resource tradeoffs. Incomparable candidates remain nondominated so the shared core can retain and rotate them."
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        primary_only_arms(context.candidate)
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        score_primary_observations(observations)
    }

    fn compare(&self, left: &StrategyScore, right: &StrategyScore) -> Ordering {
        compare_scores(left, right)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum ResourceAxis {
    Messages,
    MessageChars,
    MaxMessageLength,
    LinkTransmissions,
    GroundedSignals,
    InterferenceActions,
    InvalidActions,
}

impl ResourceAxis {
    fn key(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::MessageChars => "message_chars",
            Self::MaxMessageLength => "max_message_length",
            Self::LinkTransmissions => "link_transmissions",
            Self::GroundedSignals => "grounded_signals",
            Self::InterferenceActions => "interference_actions",
            Self::InvalidActions => "invalid_actions",
        }
    }
}

#[derive(Clone, Copy)]
enum Aggregation {
    Mean,
    Maximum,
}

impl Aggregation {
    fn label(self) -> &'static str {
        match self {
            Self::Mean => "mean_per_primary_rollout",
            Self::Maximum => "maximum_across_primary_rollouts",
        }
    }
}

#[derive(Clone, Copy)]
struct ResourceSpec {
    axis: ResourceAxis,
    aliases: &'static [&'static str],
    limit: f64,
    required: bool,
    aggregation: Aggregation,
}

#[derive(Debug, Serialize)]
struct OutcomeDiagnostics {
    tier_metric: &'static str,
    outcome_tier: f64,
    outcome_reward: f64,
    primary_rollout_count: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FeasibilityStatus {
    Feasible,
    Infeasible,
}

#[derive(Debug, Serialize)]
struct FeasibilityDiagnostics {
    status: FeasibilityStatus,
    feasible: bool,
    required_metrics_complete: bool,
    common_envelope_satisfied: bool,
}

#[derive(Debug, Serialize)]
struct CommonEnvelopeDiagnostics {
    scope: &'static str,
    source: &'static str,
    candidate_declared_budget_used: bool,
    limits: BTreeMap<String, f64>,
}

#[derive(Debug, Serialize)]
struct ResourceDiagnostic {
    axis: ResourceAxis,
    required: bool,
    aggregation: &'static str,
    available: bool,
    observed: Option<f64>,
    envelope_limit: f64,
    normalization_scale: f64,
    normalized_use: Option<f64>,
    slack: Option<f64>,
    normalized_slack: Option<f64>,
    within_envelope: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OptionalAvailabilityDiagnostics {
    available: bool,
    present_primary_rollouts: usize,
    missing_primary_rollouts: usize,
    missing_rollout_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SlackDiagnostics {
    raw: BTreeMap<String, Option<f64>>,
    normalized: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Serialize)]
struct ImacDiagnostics {
    schema_version: &'static str,
    method: &'static str,
    prompt_level_imac_inspiration: bool,
    neural_mutual_information_optimization: bool,
    primary_arm_only: bool,
    outcome: OutcomeDiagnostics,
    feasibility: FeasibilityDiagnostics,
    common_envelope: CommonEnvelopeDiagnostics,
    resource_vector: Vec<ResourceDiagnostic>,
    slack: SlackDiagnostics,
    dominated_axes: Vec<ResourceAxis>,
    dominated_axes_definition: &'static str,
    optional_metric_availability: BTreeMap<String, OptionalAvailabilityDiagnostics>,
    missing_optional_metrics: Vec<ResourceAxis>,
}

fn score_primary_observations(observations: &[RolloutObservation]) -> Result<StrategyScore> {
    if observations.is_empty() {
        return Err(invariant(
            "IMAC score requires at least one primary rollout observation",
        ));
    }
    if let Some(observation) = observations
        .iter()
        .find(|observation| !observation.is_primary())
    {
        return Err(invariant(format!(
            "IMAC score accepts only the primary evaluation arm; rollout {} used arm {:?}",
            observation.rollout_id, observation.arm_id
        )));
    }

    let denominator = observations.len() as f64;
    let mut outcome_tier_total = 0.0;
    let mut reward_total = 0.0;
    for observation in observations {
        let outcome_tier = required_metric(
            observation,
            "outcome_success",
            &["outcome_success", "success", "task_success"],
        )?;
        if !(0.0..=1.0).contains(&outcome_tier) {
            return Err(invariant(format!(
                "IMAC rollout {} has outcome_success outside [0, 1]: {outcome_tier}",
                observation.rollout_id
            )));
        }
        if !observation.reward.is_finite() {
            return Err(invariant(format!(
                "IMAC rollout {} has non-finite outcome reward",
                observation.rollout_id
            )));
        }
        outcome_tier_total += outcome_tier;
        reward_total += observation.reward;
    }
    let outcome_tier = outcome_tier_total / denominator;
    let outcome_reward = reward_total / denominator;
    if !outcome_tier.is_finite() || !outcome_reward.is_finite() {
        return Err(invariant(
            "IMAC outcome aggregation produced a non-finite score",
        ));
    }

    let mut metrics = BTreeMap::from([
        ("outcome_tier".to_string(), outcome_tier),
        ("outcome_success".to_string(), outcome_tier),
        ("outcome_reward".to_string(), outcome_reward),
    ]);
    let mut resource_vector = Vec::with_capacity(RESOURCE_SPECS.len());
    let mut optional_metric_availability = BTreeMap::new();
    let mut missing_optional_metrics = Vec::new();
    let mut dominated_axes = Vec::new();
    let mut raw_slack = BTreeMap::new();
    let mut normalized_slack = BTreeMap::new();

    for spec in RESOURCE_SPECS {
        let mut values = Vec::with_capacity(observations.len());
        let mut missing_rollout_ids = Vec::new();
        for observation in observations {
            match resource_metric(observation, spec)? {
                Some(value) => values.push(value),
                None => missing_rollout_ids.push(observation.rollout_id.clone()),
            }
        }

        if spec.required && !missing_rollout_ids.is_empty() {
            return Err(invariant(format!(
                "IMAC required resource metric {:?} is missing from primary rollout(s) {:?}",
                spec.axis.key(),
                missing_rollout_ids
            )));
        }

        let available = missing_rollout_ids.is_empty();
        let observed = if available {
            let value = aggregate_resource(&values, spec.aggregation);
            if !value.is_finite() {
                return Err(invariant(format!(
                    "IMAC aggregate resource metric {:?} is non-finite",
                    spec.axis.key()
                )));
            }
            Some(value)
        } else {
            None
        };
        let normalization_scale = spec.limit.max(1.0);
        let normalized_use = observed.map(|value| value / normalization_scale);
        let slack = observed.map(|value| spec.limit - value);
        let axis_normalized_slack = slack.map(|value| value / normalization_scale);
        let within_envelope = observed.map(|value| value <= spec.limit);

        if let Some(value) = observed {
            metrics.insert(spec.axis.key().to_string(), value);
            metrics.insert(
                normalized_metric_key(spec.axis),
                value / normalization_scale,
            );
            metrics.insert(availability_metric_key(spec.axis), 1.0);
            if value > spec.limit {
                dominated_axes.push(spec.axis);
            }
        } else {
            metrics.insert(availability_metric_key(spec.axis), 0.0);
        }

        if !spec.required {
            optional_metric_availability.insert(
                spec.axis.key().to_string(),
                OptionalAvailabilityDiagnostics {
                    available,
                    present_primary_rollouts: values.len(),
                    missing_primary_rollouts: missing_rollout_ids.len(),
                    missing_rollout_ids,
                },
            );
            if !available {
                missing_optional_metrics.push(spec.axis);
            }
        }

        raw_slack.insert(spec.axis.key().to_string(), slack);
        normalized_slack.insert(spec.axis.key().to_string(), axis_normalized_slack);
        resource_vector.push(ResourceDiagnostic {
            axis: spec.axis,
            required: spec.required,
            aggregation: spec.aggregation.label(),
            available,
            observed,
            envelope_limit: spec.limit,
            normalization_scale,
            normalized_use,
            slack,
            normalized_slack: axis_normalized_slack,
            within_envelope,
        });
    }

    let feasible = dominated_axes.is_empty();
    metrics.insert(
        "common_envelope_feasible".to_string(),
        if feasible { 1.0 } else { 0.0 },
    );
    let diagnostics = ImacDiagnostics {
        schema_version: "marl_promptopt.imac_score.v1",
        method: "outcome_then_common_envelope_then_normalized_resource_pareto",
        prompt_level_imac_inspiration: true,
        neural_mutual_information_optimization: false,
        primary_arm_only: true,
        outcome: OutcomeDiagnostics {
            tier_metric: "mean_outcome_success",
            outcome_tier,
            outcome_reward,
            primary_rollout_count: observations.len(),
        },
        feasibility: FeasibilityDiagnostics {
            status: if feasible {
                FeasibilityStatus::Feasible
            } else {
                FeasibilityStatus::Infeasible
            },
            feasible,
            required_metrics_complete: true,
            common_envelope_satisfied: feasible,
        },
        common_envelope: CommonEnvelopeDiagnostics {
            scope: "optimizer_wide_per_rollout",
            source: "imac_strategy_static_envelope",
            candidate_declared_budget_used: false,
            limits: common_envelope_limits(),
        },
        resource_vector,
        slack: SlackDiagnostics {
            raw: raw_slack,
            normalized: normalized_slack,
        },
        dominated_axes,
        dominated_axes_definition:
            "available resource axes whose aggregate use exceeds the optimizer-wide envelope",
        optional_metric_availability,
        missing_optional_metrics,
    };

    Ok(StrategyScore {
        primary: outcome_reward,
        metrics,
        diagnostics: serde_json::to_value(diagnostics)?,
    })
}

fn compare_scores(left: &StrategyScore, right: &StrategyScore) -> Ordering {
    let Some(left_outcome_tier) = finite_score_metric(left, "outcome_tier") else {
        return Ordering::Equal;
    };
    let Some(right_outcome_tier) = finite_score_metric(right, "outcome_tier") else {
        return Ordering::Equal;
    };
    let outcome_order = left_outcome_tier.total_cmp(&right_outcome_tier);
    if outcome_order != Ordering::Equal {
        return outcome_order;
    }

    if !left.primary.is_finite() || !right.primary.is_finite() {
        return Ordering::Equal;
    }
    let reward_order = left.primary.total_cmp(&right.primary);
    if reward_order != Ordering::Equal {
        return reward_order;
    }

    let Some(left_feasible) = score_feasibility(left) else {
        return Ordering::Equal;
    };
    let Some(right_feasible) = score_feasibility(right) else {
        return Ordering::Equal;
    };
    if left_feasible != right_feasible {
        return left_feasible.cmp(&right_feasible);
    }

    let mut left_no_worse = true;
    let mut right_no_worse = true;
    let mut left_strictly_better = false;
    let mut right_strictly_better = false;
    for spec in RESOURCE_SPECS {
        if !spec.required {
            let Some(left_available) = score_availability(left, spec.axis) else {
                return Ordering::Equal;
            };
            let Some(right_available) = score_availability(right, spec.axis) else {
                return Ordering::Equal;
            };
            if left_available != right_available {
                return Ordering::Equal;
            }
            if !left_available {
                continue;
            }
        }

        let Some(left_use) = finite_score_metric(left, &normalized_metric_key(spec.axis)) else {
            return Ordering::Equal;
        };
        let Some(right_use) = finite_score_metric(right, &normalized_metric_key(spec.axis)) else {
            return Ordering::Equal;
        };
        match left_use.total_cmp(&right_use) {
            Ordering::Less => {
                left_strictly_better = true;
                right_no_worse = false;
            }
            Ordering::Greater => {
                right_strictly_better = true;
                left_no_worse = false;
            }
            Ordering::Equal => {}
        }
    }

    match (
        left_no_worse && left_strictly_better,
        right_no_worse && right_strictly_better,
    ) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => Ordering::Equal,
    }
}

fn required_metric(
    observation: &RolloutObservation,
    metric_name: &str,
    aliases: &[&str],
) -> Result<f64> {
    let value = aliases
        .iter()
        .find_map(|key| observation.metrics.get(*key).copied())
        .ok_or_else(|| {
            invariant(format!(
                "IMAC rollout {} is missing required metric {metric_name:?}",
                observation.rollout_id
            ))
        })?;
    if !value.is_finite() {
        return Err(invariant(format!(
            "IMAC rollout {} has non-finite required metric {metric_name:?}",
            observation.rollout_id
        )));
    }
    Ok(value)
}

fn resource_metric(observation: &RolloutObservation, spec: ResourceSpec) -> Result<Option<f64>> {
    let Some(value) = spec
        .aliases
        .iter()
        .find_map(|key| observation.metrics.get(*key).copied())
    else {
        return Ok(None);
    };
    if !value.is_finite() {
        return Err(invariant(format!(
            "IMAC rollout {} has non-finite resource metric {:?}",
            observation.rollout_id,
            spec.axis.key()
        )));
    }
    if value < 0.0 {
        return Err(invariant(format!(
            "IMAC rollout {} has negative resource metric {:?}: {value}",
            observation.rollout_id,
            spec.axis.key()
        )));
    }
    Ok(Some(value))
}

fn aggregate_resource(values: &[f64], aggregation: Aggregation) -> f64 {
    match aggregation {
        Aggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
        Aggregation::Maximum => values.iter().copied().reduce(f64::max).unwrap_or(0.0),
    }
}

fn score_feasibility(score: &StrategyScore) -> Option<bool> {
    match finite_score_metric(score, "common_envelope_feasible")? {
        0.0 => Some(false),
        1.0 => Some(true),
        _ => None,
    }
}

fn score_availability(score: &StrategyScore, axis: ResourceAxis) -> Option<bool> {
    match finite_score_metric(score, &availability_metric_key(axis))? {
        0.0 => Some(false),
        1.0 => Some(true),
        _ => None,
    }
}

fn finite_score_metric(score: &StrategyScore, key: &str) -> Option<f64> {
    score
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
}

fn normalized_metric_key(axis: ResourceAxis) -> String {
    format!("normalized_resource.{}", axis.key())
}

fn availability_metric_key(axis: ResourceAxis) -> String {
    format!("resource_available.{}", axis.key())
}

fn common_envelope_limits() -> BTreeMap<String, f64> {
    RESOURCE_SPECS
        .into_iter()
        .map(|spec| (spec.axis.key().to_string(), spec.limit))
        .collect()
}

fn common_envelope_json() -> serde_json::Value {
    json!({
        "scope": "optimizer_wide_per_rollout",
        "messages": ENVELOPE_MESSAGES,
        "message_chars": ENVELOPE_MESSAGE_CHARS,
        "max_message_length_if_available": ENVELOPE_MAX_MESSAGE_LENGTH,
        "link_transmissions_if_available": ENVELOPE_LINK_TRANSMISSIONS,
        "grounded_signals_if_available": ENVELOPE_GROUNDED_SIGNALS,
        "interference_actions": ENVELOPE_INTERFERENCE_ACTIONS,
        "invalid_actions": ENVELOPE_INVALID_ACTIONS
    })
}

fn invariant(message: impl Into<String>) -> OptimizerError {
    OptimizerError::Invariant(message.into())
}
