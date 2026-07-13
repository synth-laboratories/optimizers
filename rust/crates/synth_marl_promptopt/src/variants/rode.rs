use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{OptimizerError, PromptProgram, Result};

use crate::strategy::{ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

const PRIMARY_ARM: &str = "primary";
const ROLE_PERMUTED_ARM: &str = "role_permuted";
const ROLE_PROMPTS_FIELD: &str = "role_prompts";
const SHARED_INSTRUCTION_FIELD: &str = "shared_instruction";
const COMMUNICATION_POLICY_FIELD: &str = "communication_policy";

pub struct RodeStrategy;

impl MarlStrategy for RodeStrategy {
    fn name(&self) -> &'static str {
        "rode"
    }

    fn proposer_guidance(&self) -> Value {
        json!({
            "paper_analogue": "RODE role-oriented hierarchical learning",
            "status": "implemented",
            "mechanism": "Prompt-level RODE inspiration: alternate specialist-role prompt edits with shared coordination-protocol edits and validate them with matched role permutations. This is not neural action-space decomposition and does not learn role embeddings, role-conditioned action spaces, or value functions.",
            "mutation_schedule": {
                "odd_generations": {
                    "target_fields": [ROLE_PROMPTS_FIELD],
                    "objective": "Improve stable, non-duplicated specialist assignments without changing shared or communication instructions."
                },
                "even_generations": {
                    "target_fields": [SHARED_INSTRUCTION_FIELD, COMMUNICATION_POLICY_FIELD],
                    "objective": "Improve the low-frequency selector, explicit handoff protocol, and coordination rules without changing specialist role prompts."
                }
            },
            "role_selector": {
                "frequency": "low",
                "selector_interval": 4,
                "selector_ttl": 8,
                "unit": "dependent action steps",
                "rule": "Select roles at episode start, reconsider no more often than every 4 dependent action steps, and renew or replace an assignment before its 8-step TTL expires. Keep the current assignment between selector ticks unless an explicit handoff trigger fires.",
                "allowed_change_triggers": [
                    "the current owner completes its dependency",
                    "new evidence proves another specialist is required",
                    "the current owner reports a blocker or capability mismatch",
                    "the assignment TTL is about to expire with unfinished work",
                    "continuing would duplicate, abandon, or interfere with another assignment"
                ]
            },
            "handoff_protocol": {
                "request": "The current owner emits HANDOFF(from_role, to_role, assignment_id, remaining_work, evidence_or_blocker).",
                "acceptance": "The receiving role explicitly ACKs the assignment_id; ownership changes only after that ACK, and the prior owner remains accountable until then.",
                "completion": "The receiver reports completion or starts a new explicit handoff before TTL expiry.",
                "triggers": [
                    "dependency completion exposes work for another specialist",
                    "capability mismatch or blocker",
                    "new task evidence changes the required specialist",
                    "TTL expiry with remaining work"
                ],
                "credit_rule": "Credit only a valid explicit request-plus-ACK ownership transfer tied to real remaining work. A fixed leader/follower hierarchy, static role assignment, delegation without ACK, or ordinary task completion is not a successful handoff."
            },
            "selection_order": [
                "primary outcome success",
                "primary outcome reward",
                "stable role consistency and matched role-specialization signal",
                "valid explicit handoffs when reported",
                "lower selector churn when reported",
                "lower role duplication",
                "lower abandoned or unfinished assignments when reported",
                "lower interference cost",
                "lower communication cost"
            ],
            "required_evidence": {
                "arms": [PRIMARY_ARM, ROLE_PERMUTED_ARM],
                "receipt": ["arm_id", "intervention_applied", "checkpoint_digest", "role_permutation_applied"],
                "matching": ["candidate_id", "split", "stage", "task_id", "checkpoint_digest"],
                "required_metrics": [
                    "outcome_success",
                    "role_consistency",
                    "role_duplication_count",
                    "interference_action_count",
                    "message_count",
                    "message_chars"
                ],
                "optional_metrics": [
                    "abandoned_assignments",
                    "unfinished_assignments",
                    "valid_explicit_handoffs",
                    "selector_changes",
                    "communication_cost",
                    "interference_cost"
                ]
            }
        })
    }

    fn target_fields(&self, generation: usize, program: &PromptProgram) -> Vec<String> {
        program
            .mutable_field_ids()
            .into_iter()
            .filter(|field| {
                if generation % 2 == 1 {
                    field == ROLE_PROMPTS_FIELD
                } else {
                    field == SHARED_INSTRUCTION_FIELD || field == COMMUNICATION_POLICY_FIELD
                }
            })
            .collect()
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        let primary = EvaluationArm::primary(&context.candidate.payload);
        let mut metadata = Map::from_iter([
            (
                "intervention_type".to_string(),
                json!("role_permutation"),
            ),
            ("matched_arm".to_string(), json!(PRIMARY_ARM)),
            (
                "permutation_scope".to_string(),
                json!("specialist_assignment_only"),
            ),
            ("require_exact_match".to_string(), json!(true)),
        ]);
        for (source, target) in [
            ("task_id", "matched_task_id"),
            ("checkpoint_key", "matched_checkpoint_key"),
        ] {
            if let Some(value) = context.row.get(source) {
                metadata.insert(target.to_string(), value.clone());
            }
        }
        let role_permuted = EvaluationArm {
            arm_id: ROLE_PERMUTED_ARM.to_string(),
            payload: context.candidate.payload.clone(),
            metadata,
        };
        vec![primary, role_permuted]
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        score_matched_role_permutations(observations)
    }

    fn compare(&self, left: &StrategyScore, right: &StrategyScore) -> Ordering {
        higher_metric(left, "outcome_success")
            .total_cmp(&higher_metric(right, "outcome_success"))
            .then_with(|| left.primary.total_cmp(&right.primary))
            .then_with(|| {
                higher_metric(left, "role_consistency")
                    .total_cmp(&higher_metric(right, "role_consistency"))
            })
            .then_with(|| {
                higher_metric(left, "role_specialization_signal")
                    .total_cmp(&higher_metric(right, "role_specialization_signal"))
            })
            .then_with(|| {
                compare_optional_higher(left, right, "valid_explicit_handoffs")
            })
            .then_with(|| compare_optional_lower(left, right, "selector_changes"))
            .then_with(|| {
                lower_metric(right, "role_duplication")
                    .total_cmp(&lower_metric(left, "role_duplication"))
            })
            .then_with(|| {
                compare_optional_lower(left, right, "assignment_abandonment_cost")
            })
            .then_with(|| {
                lower_metric(right, "interference_cost")
                    .total_cmp(&lower_metric(left, "interference_cost"))
            })
            .then_with(|| {
                lower_metric(right, "communication_cost")
                    .total_cmp(&lower_metric(left, "communication_cost"))
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InterventionReceipt {
    arm_id: String,
    intervention_applied: bool,
    checkpoint_digest: String,
    #[serde(default)]
    role_permutation_applied: Option<bool>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct RoleMetrics {
    outcome_success: f64,
    role_consistency: f64,
    role_duplication: f64,
    interference_actions: f64,
    interference_cost: f64,
    messages: f64,
    message_chars: f64,
    communication_cost: f64,
    abandoned_assignments: Option<f64>,
    unfinished_assignments: Option<f64>,
    valid_explicit_handoffs: Option<f64>,
    selector_changes: Option<f64>,
}

impl RoleMetrics {
    fn assignment_abandonment_cost(&self) -> Option<f64> {
        match (self.abandoned_assignments, self.unfinished_assignments) {
            (None, None) => None,
            (abandoned, unfinished) => Some(abandoned.unwrap_or(0.0) + unfinished.unwrap_or(0.0)),
        }
    }
}

struct ValidatedArm<'a> {
    observation: &'a RolloutObservation,
    receipt: InterventionReceipt,
    metrics: RoleMetrics,
}

#[derive(Default)]
struct MatchedPair<'a> {
    primary: Option<ValidatedArm<'a>>,
    role_permuted: Option<ValidatedArm<'a>>,
}

#[derive(Clone, Debug, Serialize)]
struct ArmDiagnostics {
    rollout_id: String,
    outcome_reward: f64,
    metrics: RoleMetrics,
    receipt: InterventionReceipt,
}

#[derive(Clone, Debug, Serialize)]
struct MatchedRolePermutationDiagnostics {
    task_key: TaskKey,
    checkpoint_digest: String,
    primary: ArmDiagnostics,
    role_permuted: ArmDiagnostics,
    primary_minus_role_permuted_reward: f64,
    primary_minus_role_permuted_outcome_success: f64,
    primary_minus_role_permuted_role_consistency: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AggregateDiagnostics {
    primary_outcome_reward: f64,
    role_permuted_outcome_reward: f64,
    primary_outcome_success: f64,
    role_permuted_outcome_success: f64,
    primary_role_consistency: f64,
    role_permuted_role_consistency: f64,
    role_specialization_signal: f64,
    role_duplication: f64,
    interference_cost: f64,
    communication_cost: f64,
    abandoned_assignments: Option<f64>,
    unfinished_assignments: Option<f64>,
    assignment_abandonment_cost: Option<f64>,
    valid_explicit_handoffs: Option<f64>,
    selector_changes: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct OptionalMetricCoverage {
    abandoned_assignments: usize,
    unfinished_assignments: usize,
    assignment_abandonment_cost: usize,
    valid_explicit_handoffs: usize,
    selector_changes: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ProposerFeedback {
    specialization_pressure: &'static str,
    handoff_pressure: &'static str,
    selector_pressure: &'static str,
    non_credit_rule: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct RodeDiagnostics {
    schema_version: &'static str,
    method: &'static str,
    prompt_level_rode: bool,
    neural_action_space_decomposition: bool,
    matched_pair_count: usize,
    definitions: BTreeMap<&'static str, &'static str>,
    aggregate: AggregateDiagnostics,
    optional_metric_coverage: OptionalMetricCoverage,
    proposer_feedback: ProposerFeedback,
    matched_role_permutations: Vec<MatchedRolePermutationDiagnostics>,
}

#[derive(Default)]
struct OptionalTotal {
    sum: f64,
    count: usize,
}

impl OptionalTotal {
    fn add(&mut self, value: Option<f64>) {
        if let Some(value) = value {
            self.sum += value;
            self.count += 1;
        }
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

#[derive(Default)]
struct Aggregate {
    primary_reward: f64,
    role_permuted_reward: f64,
    primary_outcome_success: f64,
    role_permuted_outcome_success: f64,
    primary_role_consistency: f64,
    role_permuted_role_consistency: f64,
    role_duplication: f64,
    interference_actions: f64,
    interference_cost: f64,
    messages: f64,
    message_chars: f64,
    communication_cost: f64,
    abandoned_assignments: OptionalTotal,
    unfinished_assignments: OptionalTotal,
    assignment_abandonment_cost: OptionalTotal,
    valid_explicit_handoffs: OptionalTotal,
    selector_changes: OptionalTotal,
}

fn score_matched_role_permutations(
    observations: &[RolloutObservation],
) -> Result<StrategyScore> {
    let Some(first) = observations.first() else {
        return Err(invariant("RODE scoring requires rollout observations"));
    };
    let expected_candidate = first.candidate_id.as_str();
    let expected_split = first.split.as_str();
    let expected_stage = first.stage.as_str();
    let mut pairs = BTreeMap::<TaskKey, MatchedPair<'_>>::new();

    for observation in observations {
        if observation.candidate_id != expected_candidate
            || observation.split != expected_split
            || observation.stage != expected_stage
        {
            return Err(invariant(format!(
                "RODE score batch mixed candidate/split/stage identities at rollout {}",
                observation.rollout_id
            )));
        }
        let validated = validate_observation(observation)?;
        let task_key = TaskKey::from_observation(observation);
        let pair = pairs.entry(task_key.clone()).or_default();
        let slot = match observation.arm_id.as_str() {
            PRIMARY_ARM => &mut pair.primary,
            ROLE_PERMUTED_ARM => &mut pair.role_permuted,
            other => {
                return Err(invariant(format!(
                    "RODE received unsupported arm {other:?} for task key {task_key:?}"
                )))
            }
        };
        if slot.replace(validated).is_some() {
            return Err(invariant(format!(
                "RODE received duplicate {:?} arm for exact task key {task_key:?}",
                observation.arm_id
            )));
        }
    }

    let mut aggregate = Aggregate::default();
    let mut matched_diagnostics = Vec::with_capacity(pairs.len());
    for (task_key, pair) in pairs {
        let primary = pair.primary.ok_or_else(|| {
            invariant(format!(
                "RODE exact task key {task_key:?} is missing its required primary arm"
            ))
        })?;
        let role_permuted = pair.role_permuted.ok_or_else(|| {
            invariant(format!(
                "RODE exact task key {task_key:?} is missing its required role_permuted arm"
            ))
        })?;
        require_exact_match(&task_key, &primary, &role_permuted)?;

        aggregate.primary_reward += primary.observation.reward;
        aggregate.role_permuted_reward += role_permuted.observation.reward;
        aggregate.primary_outcome_success += primary.metrics.outcome_success;
        aggregate.role_permuted_outcome_success += role_permuted.metrics.outcome_success;
        aggregate.primary_role_consistency += primary.metrics.role_consistency;
        aggregate.role_permuted_role_consistency += role_permuted.metrics.role_consistency;
        aggregate.role_duplication += primary.metrics.role_duplication;
        aggregate.interference_actions += primary.metrics.interference_actions;
        aggregate.interference_cost += primary.metrics.interference_cost;
        aggregate.messages += primary.metrics.messages;
        aggregate.message_chars += primary.metrics.message_chars;
        aggregate.communication_cost += primary.metrics.communication_cost;
        aggregate
            .abandoned_assignments
            .add(primary.metrics.abandoned_assignments);
        aggregate
            .unfinished_assignments
            .add(primary.metrics.unfinished_assignments);
        aggregate
            .assignment_abandonment_cost
            .add(primary.metrics.assignment_abandonment_cost());
        aggregate
            .valid_explicit_handoffs
            .add(primary.metrics.valid_explicit_handoffs);
        aggregate
            .selector_changes
            .add(primary.metrics.selector_changes);

        matched_diagnostics.push(MatchedRolePermutationDiagnostics {
            task_key,
            checkpoint_digest: primary.receipt.checkpoint_digest.clone(),
            primary: ArmDiagnostics {
                rollout_id: primary.observation.rollout_id.clone(),
                outcome_reward: primary.observation.reward,
                metrics: primary.metrics,
                receipt: primary.receipt,
            },
            role_permuted: ArmDiagnostics {
                rollout_id: role_permuted.observation.rollout_id.clone(),
                outcome_reward: role_permuted.observation.reward,
                metrics: role_permuted.metrics,
                receipt: role_permuted.receipt,
            },
            primary_minus_role_permuted_reward: primary.observation.reward
                - role_permuted.observation.reward,
            primary_minus_role_permuted_outcome_success: primary.metrics.outcome_success
                - role_permuted.metrics.outcome_success,
            primary_minus_role_permuted_role_consistency: primary.metrics.role_consistency
                - role_permuted.metrics.role_consistency,
        });
    }

    let pair_count = matched_diagnostics.len();
    let denominator = pair_count as f64;
    let primary_reward = aggregate.primary_reward / denominator;
    let role_permuted_reward = aggregate.role_permuted_reward / denominator;
    let primary_outcome_success = aggregate.primary_outcome_success / denominator;
    let role_permuted_outcome_success = aggregate.role_permuted_outcome_success / denominator;
    let primary_role_consistency = aggregate.primary_role_consistency / denominator;
    let role_permuted_role_consistency =
        aggregate.role_permuted_role_consistency / denominator;
    let role_permutation_outcome_delta =
        primary_outcome_success - role_permuted_outcome_success;
    let role_permutation_consistency_delta =
        primary_role_consistency - role_permuted_role_consistency;
    let role_specialization_signal =
        0.5 * (role_permutation_outcome_delta + role_permutation_consistency_delta);
    let role_duplication = aggregate.role_duplication / denominator;
    let interference_actions = aggregate.interference_actions / denominator;
    let interference_cost = aggregate.interference_cost / denominator;
    let messages = aggregate.messages / denominator;
    let message_chars = aggregate.message_chars / denominator;
    let communication_cost = aggregate.communication_cost / denominator;
    let abandoned_assignments = aggregate.abandoned_assignments.mean();
    let unfinished_assignments = aggregate.unfinished_assignments.mean();
    let assignment_abandonment_cost = aggregate.assignment_abandonment_cost.mean();
    let valid_explicit_handoffs = aggregate.valid_explicit_handoffs.mean();
    let selector_changes = aggregate.selector_changes.mean();

    let mut metrics = BTreeMap::from([
        ("outcome_reward".to_string(), primary_reward),
        (
            "role_permuted_outcome_reward".to_string(),
            role_permuted_reward,
        ),
        (
            "role_permutation_reward_delta".to_string(),
            primary_reward - role_permuted_reward,
        ),
        ("outcome_success".to_string(), primary_outcome_success),
        (
            "role_permuted_outcome_success".to_string(),
            role_permuted_outcome_success,
        ),
        (
            "role_permutation_outcome_delta".to_string(),
            role_permutation_outcome_delta,
        ),
        ("role_consistency".to_string(), primary_role_consistency),
        (
            "role_permuted_role_consistency".to_string(),
            role_permuted_role_consistency,
        ),
        (
            "role_permutation_consistency_delta".to_string(),
            role_permutation_consistency_delta,
        ),
        (
            "role_specialization_signal".to_string(),
            role_specialization_signal,
        ),
        ("role_duplication".to_string(), role_duplication),
        (
            "interference_actions".to_string(),
            interference_actions,
        ),
        ("interference_cost".to_string(), interference_cost),
        ("messages".to_string(), messages),
        ("message_chars".to_string(), message_chars),
        ("communication_cost".to_string(), communication_cost),
        ("matched_pair_count".to_string(), pair_count as f64),
    ]);
    insert_optional_metric(&mut metrics, "abandoned_assignments", abandoned_assignments);
    insert_optional_metric(&mut metrics, "unfinished_assignments", unfinished_assignments);
    insert_optional_metric(
        &mut metrics,
        "assignment_abandonment_cost",
        assignment_abandonment_cost,
    );
    insert_optional_metric(
        &mut metrics,
        "valid_explicit_handoffs",
        valid_explicit_handoffs,
    );
    insert_optional_metric(&mut metrics, "selector_changes", selector_changes);

    let specialization_pressure = if role_specialization_signal > 0.0 {
        "Preserve the primary assignment's matched advantage while improving consistency."
    } else {
        "Make specialist ownership load-bearing and stable; the matched permutation does not yet expose a positive specialization signal."
    };
    let handoff_pressure = match valid_explicit_handoffs {
        Some(value) if value > 0.0 => {
            "Preserve only handoffs backed by explicit request-plus-ACK receipts and real remaining work."
        }
        Some(_) => {
            "Add explicit request-plus-ACK transfers at real dependency, blocker, evidence-change, or TTL triggers."
        }
        None => {
            "Handoff evidence is unreported; do not infer success from role consistency or a static hierarchy."
        }
    };
    let selector_pressure = match selector_changes {
        Some(value) if value > 0.0 => {
            "Reduce selector churn by enforcing the 4-step interval and 8-step TTL."
        }
        Some(_) => "Preserve the observed low-churn selector behavior.",
        None => "Selector-change evidence is unreported; preserve the explicit interval and TTL contract.",
    };
    let diagnostics = RodeDiagnostics {
        schema_version: "rode_strategy_diagnostics.v1",
        method: "exact matched primary versus role_permuted prompt diagnostics",
        prompt_level_rode: true,
        neural_action_space_decomposition: false,
        matched_pair_count: pair_count,
        definitions: BTreeMap::from([
            (
                "role_specialization_signal",
                "0.5 * ((primary outcome success - role-permuted outcome success) + (primary role consistency - role-permuted role consistency)) on exact task/checkpoint pairs",
            ),
            (
                "valid_explicit_handoffs",
                "Only an explicitly reported request-plus-ACK ownership transfer; static hierarchy and ordinary completion receive no handoff credit",
            ),
            (
                "assignment_abandonment_cost",
                "Reported abandoned assignments plus reported unfinished assignments",
            ),
            (
                "communication_cost",
                "Explicit communication_cost when reported, otherwise message count plus message characters",
            ),
            (
                "interference_cost",
                "Explicit interference_cost when reported, otherwise interference action count",
            ),
        ]),
        aggregate: AggregateDiagnostics {
            primary_outcome_reward: primary_reward,
            role_permuted_outcome_reward: role_permuted_reward,
            primary_outcome_success,
            role_permuted_outcome_success,
            primary_role_consistency,
            role_permuted_role_consistency,
            role_specialization_signal,
            role_duplication,
            interference_cost,
            communication_cost,
            abandoned_assignments,
            unfinished_assignments,
            assignment_abandonment_cost,
            valid_explicit_handoffs,
            selector_changes,
        },
        optional_metric_coverage: OptionalMetricCoverage {
            abandoned_assignments: aggregate.abandoned_assignments.count,
            unfinished_assignments: aggregate.unfinished_assignments.count,
            assignment_abandonment_cost: aggregate.assignment_abandonment_cost.count,
            valid_explicit_handoffs: aggregate.valid_explicit_handoffs.count,
            selector_changes: aggregate.selector_changes.count,
        },
        proposer_feedback: ProposerFeedback {
            specialization_pressure,
            handoff_pressure,
            selector_pressure,
            non_credit_rule: "Never count a fixed leader/follower hierarchy or static role assignment as a successful handoff.",
        },
        matched_role_permutations: matched_diagnostics,
    };

    Ok(StrategyScore {
        primary: primary_reward,
        metrics,
        diagnostics: serde_json::to_value(diagnostics)?,
    })
}

fn validate_observation(observation: &RolloutObservation) -> Result<ValidatedArm<'_>> {
    if observation.task_id.trim().is_empty() {
        return Err(invariant(format!(
            "RODE rollout {} has an empty task_id",
            observation.rollout_id
        )));
    }
    if !observation.reward.is_finite() {
        return Err(invariant(format!(
            "RODE rollout {} has a non-finite reward",
            observation.rollout_id
        )));
    }
    let response_task_id = required_response_string(&observation.response, "/task_id", observation)?;
    if response_task_id != observation.task_id {
        return Err(invariant(format!(
            "RODE rollout {} task key mismatch: observation={:?}, response={response_task_id:?}",
            observation.rollout_id, observation.task_id
        )));
    }
    let response_split = required_response_string(
        &observation.response,
        "/reward_info/details/split",
        observation,
    )?;
    if response_split != observation.split {
        return Err(invariant(format!(
            "RODE rollout {} split mismatch: observation={:?}, response={response_split:?}",
            observation.rollout_id, observation.split
        )));
    }

    let receipt_value = observation
        .response
        .get("intervention_evidence")
        .ok_or_else(|| {
            invariant(format!(
                "RODE rollout {} is missing intervention_evidence",
                observation.rollout_id
            ))
        })?;
    let receipt: InterventionReceipt = serde_json::from_value(receipt_value.clone()).map_err(
        |source| {
            invariant(format!(
                "RODE rollout {} has an invalid typed intervention_evidence receipt: {source}",
                observation.rollout_id
            ))
        },
    )?;
    if receipt.arm_id != observation.arm_id {
        return Err(invariant(format!(
            "RODE rollout {} arm mismatch: observation={:?}, receipt={:?}",
            observation.rollout_id, observation.arm_id, receipt.arm_id
        )));
    }
    if receipt.checkpoint_digest.trim().is_empty() {
        return Err(invariant(format!(
            "RODE rollout {} receipt has an empty checkpoint_digest",
            observation.rollout_id
        )));
    }
    for pointer in [
        "/summary/checkpoint_digest",
        "/reward_info/details/checkpoint_digest",
        "/trace/checkpoint_digest",
    ] {
        let reported = required_response_string(&observation.response, pointer, observation)?;
        if reported != receipt.checkpoint_digest {
            return Err(invariant(format!(
                "RODE rollout {} checkpoint key mismatch at {pointer}: receipt={:?}, response={reported:?}",
                observation.rollout_id, receipt.checkpoint_digest
            )));
        }
    }
    for pointer in [
        "/summary/evaluation_arm",
        "/reward_info/details/evaluation_arm",
        "/trace/evaluation_arm",
    ] {
        let reported = required_response_string(&observation.response, pointer, observation)?;
        if reported != observation.arm_id {
            return Err(invariant(format!(
                "RODE rollout {} evaluation arm mismatch at {pointer}: observation={:?}, response={reported:?}",
                observation.rollout_id, observation.arm_id
            )));
        }
    }
    match observation.arm_id.as_str() {
        PRIMARY_ARM if receipt.intervention_applied => {
            return Err(invariant(format!(
                "RODE primary rollout {} incorrectly reports intervention_applied=true",
                observation.rollout_id
            )))
        }
        PRIMARY_ARM if receipt.role_permutation_applied == Some(true) => {
            return Err(invariant(format!(
                "RODE primary rollout {} incorrectly reports role_permutation_applied=true",
                observation.rollout_id
            )))
        }
        ROLE_PERMUTED_ARM
            if !receipt.intervention_applied
                || receipt.role_permutation_applied != Some(true) =>
        {
            return Err(invariant(format!(
                "RODE role_permuted rollout {} requires typed intervention_applied=true and role_permutation_applied=true receipts",
                observation.rollout_id
            )))
        }
        PRIMARY_ARM | ROLE_PERMUTED_ARM => {}
        other => {
            return Err(invariant(format!(
                "RODE rollout {} has unsupported arm {other:?}",
                observation.rollout_id
            )))
        }
    }

    Ok(ValidatedArm {
        observation,
        receipt,
        metrics: required_role_metrics(observation)?,
    })
}

fn require_exact_match(
    task_key: &TaskKey,
    primary: &ValidatedArm<'_>,
    role_permuted: &ValidatedArm<'_>,
) -> Result<()> {
    if primary.receipt.checkpoint_digest != role_permuted.receipt.checkpoint_digest {
        return Err(invariant(format!(
            "RODE exact checkpoint invariant failed for task key {task_key:?}: primary={:?}, role_permuted={:?}",
            primary.receipt.checkpoint_digest, role_permuted.receipt.checkpoint_digest
        )));
    }
    Ok(())
}

fn required_role_metrics(observation: &RolloutObservation) -> Result<RoleMetrics> {
    let outcome_success = required_metric(observation, "outcome_success", &["outcome_success"])?;
    let role_consistency =
        required_metric(observation, "role_consistency", &["role_consistency"])?;
    for (key, value) in [
        ("outcome_success", outcome_success),
        ("role_consistency", role_consistency),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(invariant(format!(
                "RODE rollout {} metric {key:?} must be in [0, 1]",
                observation.rollout_id
            )));
        }
    }
    let role_duplication = required_nonnegative_metric(
        observation,
        "role_duplication",
        &[
            "role_duplication",
            "role_duplication_count",
            "duplicate_assignments",
        ],
    )?;
    let interference_actions = required_nonnegative_metric(
        observation,
        "interference_actions",
        &["interference_actions", "interference_action_count"],
    )?;
    let messages = required_nonnegative_metric(
        observation,
        "messages",
        &["messages", "message_count", "messages_delivered"],
    )?;
    let message_chars = required_nonnegative_metric(
        observation,
        "message_chars",
        &["message_chars", "communication_chars"],
    )?;
    let interference_cost = optional_nonnegative_metric(
        observation,
        "interference_cost",
        &["interference_cost"],
    )?
    .unwrap_or(interference_actions);
    let communication_cost = optional_nonnegative_metric(
        observation,
        "communication_cost",
        &["communication_cost"],
    )?
    .unwrap_or(messages + message_chars);

    Ok(RoleMetrics {
        outcome_success,
        role_consistency,
        role_duplication,
        interference_actions,
        interference_cost,
        messages,
        message_chars,
        communication_cost,
        abandoned_assignments: optional_nonnegative_metric(
            observation,
            "abandoned_assignments",
            &[
                "abandoned_assignments",
                "abandoned_assignment_count",
                "assignment_abandonments",
            ],
        )?,
        unfinished_assignments: optional_nonnegative_metric(
            observation,
            "unfinished_assignments",
            &["unfinished_assignments", "unfinished_assignment_count"],
        )?,
        valid_explicit_handoffs: optional_nonnegative_metric(
            observation,
            "valid_explicit_handoffs",
            &[
                "valid_explicit_handoffs",
                "valid_explicit_handoff_count",
                "successful_explicit_handoffs",
            ],
        )?,
        selector_changes: optional_nonnegative_metric(
            observation,
            "selector_changes",
            &[
                "selector_changes",
                "role_selector_changes",
                "role_assignment_changes",
            ],
        )?,
    })
}

fn required_metric(
    observation: &RolloutObservation,
    canonical: &str,
    aliases: &[&str],
) -> Result<f64> {
    optional_metric(observation, canonical, aliases)?.ok_or_else(|| {
        invariant(format!(
            "RODE rollout {} is missing required metric {canonical:?}; accepted keys={aliases:?}",
            observation.rollout_id
        ))
    })
}

fn required_nonnegative_metric(
    observation: &RolloutObservation,
    canonical: &str,
    aliases: &[&str],
) -> Result<f64> {
    let value = required_metric(observation, canonical, aliases)?;
    if value < 0.0 {
        return Err(invariant(format!(
            "RODE rollout {} metric {canonical:?} must be non-negative",
            observation.rollout_id
        )));
    }
    Ok(value)
}

fn optional_nonnegative_metric(
    observation: &RolloutObservation,
    canonical: &str,
    aliases: &[&str],
) -> Result<Option<f64>> {
    let value = optional_metric(observation, canonical, aliases)?;
    if value.is_some_and(|value| value < 0.0) {
        return Err(invariant(format!(
            "RODE rollout {} metric {canonical:?} must be non-negative",
            observation.rollout_id
        )));
    }
    Ok(value)
}

fn optional_metric(
    observation: &RolloutObservation,
    canonical: &str,
    aliases: &[&str],
) -> Result<Option<f64>> {
    let mut found = None;
    for alias in aliases {
        let Some(value) = observation.metrics.get(*alias).copied() else {
            continue;
        };
        if !value.is_finite() {
            return Err(invariant(format!(
                "RODE rollout {} metric {alias:?} is non-finite",
                observation.rollout_id
            )));
        }
        if let Some(previous) = found {
            if previous != value {
                return Err(invariant(format!(
                    "RODE rollout {} has conflicting aliases for metric {canonical:?}: previous={previous}, {alias}={value}",
                    observation.rollout_id
                )));
            }
        } else {
            found = Some(value);
        }
    }
    Ok(found)
}

fn required_response_string<'a>(
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
                "RODE rollout {} is missing required string {pointer}",
                observation.rollout_id
            ))
        })
}

fn insert_optional_metric(
    metrics: &mut BTreeMap<String, f64>,
    key: &str,
    value: Option<f64>,
) {
    if let Some(value) = value {
        metrics.insert(key.to_string(), value);
    }
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

fn compare_optional_higher(left: &StrategyScore, right: &StrategyScore, key: &str) -> Ordering {
    match (finite_metric(left, key), finite_metric(right, key)) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        _ => Ordering::Equal,
    }
}

fn compare_optional_lower(left: &StrategyScore, right: &StrategyScore, key: &str) -> Ordering {
    match (finite_metric(left, key), finite_metric(right, key)) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        _ => Ordering::Equal,
    }
}

fn finite_metric(score: &StrategyScore, key: &str) -> Option<f64> {
    score
        .metrics
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
}

fn invariant(message: impl Into<String>) -> OptimizerError {
    OptimizerError::Invariant(message.into())
}
