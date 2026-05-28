use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::cache::stable_json;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidatePayloadRecord {
    pub schema_version: String,
    pub candidate_payload_id: String,
    pub candidate_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub source: String,
    pub status: String,
    pub payload_hash: String,
    pub payload: Value,
    pub lever_bundle: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct CandidatePayloadInput<'a> {
    pub candidate_id: &'a str,
    pub parent_id: Option<String>,
    pub source: &'a str,
    pub status: &'a str,
    pub payload: &'a Value,
    pub lever_bundle: &'a Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateDeltaRecord {
    pub schema_version: String,
    pub candidate_delta_id: String,
    pub candidate_id: String,
    pub parent_candidate_id: String,
    pub operation_kind: String,
    pub source: String,
    pub status: String,
    #[serde(default)]
    pub target_levers: Vec<String>,
    #[serde(default)]
    pub changed_fields: Vec<String>,
    pub before: Value,
    pub after: Value,
    pub rationale: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct CandidateDeltaInput<'a> {
    pub candidate_id: &'a str,
    pub parent_candidate_id: &'a str,
    pub source: &'a str,
    pub status: &'a str,
    pub parent_payload: &'a Value,
    pub parent_lever_bundle: &'a Value,
    pub child_payload: &'a Value,
    pub child_lever_bundle: &'a Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanLinkRecord {
    pub schema_version: String,
    pub plan_link_id: String,
    pub source_type: String,
    pub source_id: String,
    pub target_type: String,
    pub target_id: String,
    pub relation: String,
    pub status: String,
    pub confidence: f64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct PlanLinkInput<'a> {
    pub source_type: &'a str,
    pub source_id: &'a str,
    pub target_type: &'a str,
    pub target_id: &'a str,
    pub relation: &'a str,
    pub status: &'a str,
    pub confidence: f64,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceDecisionRecord {
    pub schema_version: String,
    pub acceptance_decision_id: String,
    pub candidate_id: String,
    #[serde(default)]
    pub parent_candidate_id: Option<String>,
    pub decision: String,
    pub stage: String,
    pub status: String,
    pub reason: String,
    #[serde(default)]
    pub candidate_minibatch_reward: Option<f64>,
    #[serde(default)]
    pub parent_minibatch_reward: Option<f64>,
    #[serde(default)]
    pub candidate_train_reward: Option<f64>,
    #[serde(default)]
    pub parent_train_reward: Option<f64>,
    #[serde(default)]
    pub heldout_reward: Option<f64>,
    #[serde(default)]
    pub score: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct AcceptanceDecisionInput<'a> {
    pub candidate_id: &'a str,
    pub parent_candidate_id: Option<String>,
    pub candidate_status: &'a str,
    pub candidate_minibatch_reward: Option<f64>,
    pub parent_minibatch_reward: Option<f64>,
    pub candidate_train_reward: Option<f64>,
    pub parent_train_reward: Option<f64>,
    pub heldout_reward: Option<f64>,
    pub score: Option<Value>,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrontierCellRecord {
    pub schema_version: String,
    pub frontier_cell_id: String,
    pub frontier_name: String,
    pub candidate_id: String,
    #[serde(default)]
    pub parent_candidate_id: Option<String>,
    pub source: String,
    pub status: String,
    pub split: String,
    pub objective: String,
    pub rank: u64,
    pub score: f64,
    #[serde(default)]
    pub score_vector: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct FrontierCellInput<'a> {
    pub frontier_name: &'a str,
    pub candidate_id: &'a str,
    pub parent_candidate_id: Option<String>,
    pub source: &'a str,
    pub status: &'a str,
    pub split: &'a str,
    pub objective: &'a str,
    pub rank: u64,
    pub score: f64,
    pub score_vector: Value,
}

impl CandidatePayloadRecord {
    pub fn from_input(input: CandidatePayloadInput<'_>) -> Self {
        let payload_hash = payload_hash(input.payload, input.lever_bundle);
        Self {
            schema_version: "candidate_payload.v1".to_string(),
            candidate_payload_id: stable_id("candpayload", &[input.candidate_id]),
            candidate_id: input.candidate_id.to_string(),
            parent_id: input.parent_id,
            source: input.source.to_string(),
            status: input.status.to_string(),
            payload_hash,
            payload: input.payload.clone(),
            lever_bundle: input.lever_bundle.clone(),
            metadata: Map::new(),
        }
    }
}

impl CandidateDeltaRecord {
    pub fn from_input(input: CandidateDeltaInput<'_>) -> Self {
        let changed_fields = changed_fields(input.parent_payload, input.child_payload);
        let target_levers = target_levers(input.child_lever_bundle, &changed_fields);
        let operation_kind = if target_levers.is_empty() && changed_fields.is_empty() {
            "candidate_lineage_observed"
        } else {
            "lever_bundle_mutation"
        };
        Self {
            schema_version: "candidate_delta.v1".to_string(),
            candidate_delta_id: stable_id(
                "canddelta",
                &[input.parent_candidate_id, input.candidate_id],
            ),
            candidate_id: input.candidate_id.to_string(),
            parent_candidate_id: input.parent_candidate_id.to_string(),
            operation_kind: operation_kind.to_string(),
            source: input.source.to_string(),
            status: input.status.to_string(),
            target_levers,
            changed_fields,
            before: json!({
                "payload": input.parent_payload,
                "lever_bundle": input.parent_lever_bundle,
            }),
            after: json!({
                "payload": input.child_payload,
                "lever_bundle": input.child_lever_bundle,
            }),
            rationale: format!(
                "candidate {} derived from parent {}",
                input.candidate_id, input.parent_candidate_id
            ),
            metadata: Map::new(),
        }
    }
}

impl PlanLinkRecord {
    pub fn from_input(input: PlanLinkInput<'_>) -> Self {
        Self {
            schema_version: "plan_link.v1".to_string(),
            plan_link_id: stable_id(
                "planlink",
                &[
                    input.source_type,
                    input.source_id,
                    input.target_type,
                    input.target_id,
                    input.relation,
                ],
            ),
            source_type: input.source_type.to_string(),
            source_id: input.source_id.to_string(),
            target_type: input.target_type.to_string(),
            target_id: input.target_id.to_string(),
            relation: input.relation.to_string(),
            status: input.status.to_string(),
            confidence: input.confidence,
            metadata: input.metadata,
        }
    }
}

impl AcceptanceDecisionRecord {
    pub fn from_input(input: AcceptanceDecisionInput<'_>) -> Self {
        let (decision, stage, status, reason) = decision_parts(&input);
        Self {
            schema_version: "acceptance_decision.v1".to_string(),
            acceptance_decision_id: stable_id("acceptance", &[input.candidate_id]),
            candidate_id: input.candidate_id.to_string(),
            parent_candidate_id: input.parent_candidate_id,
            decision,
            stage,
            status,
            reason,
            candidate_minibatch_reward: input.candidate_minibatch_reward,
            parent_minibatch_reward: input.parent_minibatch_reward,
            candidate_train_reward: input.candidate_train_reward,
            parent_train_reward: input.parent_train_reward,
            heldout_reward: input.heldout_reward,
            score: input.score.unwrap_or_else(|| {
                json!({
                    "candidate_minibatch_reward": input.candidate_minibatch_reward,
                    "parent_minibatch_reward": input.parent_minibatch_reward,
                    "candidate_train_reward": input.candidate_train_reward,
                    "parent_train_reward": input.parent_train_reward,
                    "heldout_reward": input.heldout_reward,
                })
            }),
            metadata: input.metadata,
        }
    }
}

impl FrontierCellRecord {
    pub fn from_input(input: FrontierCellInput<'_>) -> Self {
        Self {
            schema_version: "frontier_cell.v1".to_string(),
            frontier_cell_id: stable_id(
                "frontier",
                &[
                    input.frontier_name,
                    input.split,
                    input.objective,
                    input.candidate_id,
                ],
            ),
            frontier_name: input.frontier_name.to_string(),
            candidate_id: input.candidate_id.to_string(),
            parent_candidate_id: input.parent_candidate_id,
            source: input.source.to_string(),
            status: input.status.to_string(),
            split: input.split.to_string(),
            objective: input.objective.to_string(),
            rank: input.rank,
            score: input.score,
            score_vector: input.score_vector,
            metadata: Map::new(),
        }
    }
}

fn decision_parts(input: &AcceptanceDecisionInput<'_>) -> (String, String, String, String) {
    match input.candidate_status {
        "accepted" => (
            "accepted".to_string(),
            "full_train".to_string(),
            "final".to_string(),
            "candidate accepted after full-train evaluation".to_string(),
        ),
        "rejected_minibatch" => (
            "rejected".to_string(),
            "minibatch".to_string(),
            "final".to_string(),
            "candidate rejected at minibatch evaluation".to_string(),
        ),
        "rejected_full_train" => (
            "rejected".to_string(),
            "full_train".to_string(),
            "final".to_string(),
            "candidate rejected at full-train evaluation".to_string(),
        ),
        "deferred_budget" => (
            "deferred".to_string(),
            "budget".to_string(),
            "final".to_string(),
            "candidate deferred because the run budget was exhausted".to_string(),
        ),
        "full_train_evaluated" if input.parent_candidate_id.is_none() => (
            "seed".to_string(),
            "seed_full_train".to_string(),
            "final".to_string(),
            "seed candidate evaluated on the full train split".to_string(),
        ),
        "full_train_evaluated" => (
            "evaluated".to_string(),
            "full_train".to_string(),
            "observed".to_string(),
            "candidate has a full-train evaluation without a terminal acceptance status"
                .to_string(),
        ),
        "minibatch_evaluated" => (
            "evaluated".to_string(),
            "minibatch".to_string(),
            "observed".to_string(),
            "candidate has a minibatch evaluation without a terminal acceptance status".to_string(),
        ),
        "registered" => (
            "pending".to_string(),
            "registered".to_string(),
            "pending".to_string(),
            "candidate registered and awaiting evaluation".to_string(),
        ),
        other => (
            "observed".to_string(),
            "unknown".to_string(),
            "observed".to_string(),
            format!("candidate status {other} observed"),
        ),
    }
}

fn payload_hash(payload: &Value, lever_bundle: &Value) -> String {
    let value = json!({
        "payload": payload,
        "lever_bundle": lever_bundle,
    });
    sha256_text(&stable_json(&value))
}

fn changed_fields(before: &Value, after: &Value) -> Vec<String> {
    let mut fields = BTreeSet::new();
    collect_changed_fields("payload", before, after, &mut fields);
    fields.into_iter().collect()
}

fn collect_changed_fields(
    prefix: &str,
    before: &Value,
    after: &Value,
    fields: &mut BTreeSet<String>,
) {
    match (before, after) {
        (Value::Object(left), Value::Object(right)) => {
            let keys = left
                .keys()
                .chain(right.keys())
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            for key in keys {
                let child_prefix = format!("{prefix}.{key}");
                match (left.get(key), right.get(key)) {
                    (Some(left_value), Some(right_value)) => {
                        collect_changed_fields(&child_prefix, left_value, right_value, fields);
                    }
                    _ => {
                        fields.insert(child_prefix);
                    }
                }
            }
        }
        _ => {
            if stable_json(before) != stable_json(after) {
                fields.insert(prefix.to_string());
            }
        }
    }
}

fn target_levers(lever_bundle: &Value, changed_fields: &[String]) -> Vec<String> {
    let mut levers = BTreeSet::new();
    if let Some(mutated) = lever_bundle
        .get("mutated_lever_ids")
        .and_then(Value::as_array)
    {
        for item in mutated {
            if let Some(value) = item.as_str() {
                levers.insert(value.to_string());
            }
        }
    }
    if !levers.is_empty() {
        return levers.into_iter().collect();
    }
    for field in changed_fields {
        if let Some(stripped) = field.strip_prefix("payload.") {
            let lever_id = stripped.split('.').next().unwrap_or(stripped);
            if !lever_id.is_empty() {
                levers.insert(lever_id.to_string());
            }
        }
    }
    if levers.is_empty() {
        if let Some(values) = lever_bundle.get("values").and_then(Value::as_object) {
            for key in values.keys() {
                levers.insert(key.clone());
            }
        }
    }
    levers.into_iter().collect()
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}

fn sha256_text(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}
