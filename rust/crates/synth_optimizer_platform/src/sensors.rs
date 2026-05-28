use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::artifacts::ArtifactRef;
use crate::cache::stable_json;
use crate::container_contract::{dataset_row_identity, RolloutResponse};
use crate::error::Result;
use crate::failures::{FailurePayload, OptimizerFailureType};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveScore {
    pub objective: String,
    pub value: f64,
    pub source: String,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDigest {
    pub sha256: String,
    pub event_count: usize,
    pub llm_request_count: usize,
    pub tool_call_count: usize,
    #[serde(default)]
    pub call_site_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorFrame {
    pub schema_version: String,
    pub sensor_frame_id: String,
    pub candidate_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub reward: f64,
    pub status: String,
    #[serde(default)]
    pub success_status: Option<String>,
    #[serde(default)]
    pub objective_scores: Vec<ObjectiveScore>,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub trace_digest: Option<TraceDigest>,
    #[serde(default)]
    pub actionable_side_info: Option<Value>,
    #[serde(default)]
    pub artifact_refs: Vec<ArtifactRef>,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl SensorFrame {
    pub fn from_rollout_response(
        candidate_id: &str,
        row: &Value,
        evaluation_stage: &str,
        response: &Value,
    ) -> Result<Self> {
        let rollout_response = RolloutResponse::from_value(response.clone())?;
        rollout_response.validate_for_gepa()?;
        let example_id = dataset_row_identity(row)?;
        let seed = row
            .get("seed")
            .or_else(|| response.get("seed"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let split = row
            .get("split")
            .or_else(|| row.get("example").and_then(|example| example.get("split")))
            .or_else(|| {
                row.get("dataset_row")
                    .and_then(|dataset_row| dataset_row.get("split"))
            })
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let reward = rollout_response.outcome_reward()?;
        let status = rollout_response.status.clone();
        let success_status = rollout_response.success_status.clone();
        let rollout_id = rollout_response.rollout_id.clone();
        let sensor_frame_id = sensor_frame_id(candidate_id, evaluation_stage, &example_id, seed);
        let mut metadata = Map::new();
        if let Some(details) = rollout_response
            .reward_info
            .as_ref()
            .and_then(|reward_info| reward_info.extra.get("details"))
        {
            metadata.insert("reward_details".to_string(), details.clone());
        }
        if !rollout_response.summary.is_empty() {
            metadata.insert(
                "summary".to_string(),
                Value::Object(rollout_response.summary.clone()),
            );
        }
        let trace_payload = rollout_trace_payload(row, response, &rollout_response, reward);
        metadata.insert("rollout_trace".to_string(), trace_payload);
        let objective = row
            .get("objective")
            .and_then(Value::as_str)
            .unwrap_or("outcome_reward")
            .to_string();
        let failure = if status == "completed" || status == "succeeded" {
            None
        } else {
            Some(
                FailurePayload::new(
                    OptimizerFailureType::Container,
                    "rollout_not_completed",
                    format!("rollout returned status {status}"),
                    false,
                )
                .with_detail("status", Value::String(status.clone())),
            )
        };
        Ok(Self {
            schema_version: "sensor_frame.v1".to_string(),
            sensor_frame_id,
            candidate_id: candidate_id.to_string(),
            rollout_id,
            example_id,
            seed,
            split,
            evaluation_stage: evaluation_stage.to_string(),
            reward,
            status,
            success_status,
            objective_scores: vec![ObjectiveScore {
                objective,
                value: reward,
                source: "container.reward_info".to_string(),
                rationale: rollout_response
                    .reward_info
                    .as_ref()
                    .and_then(|reward_info| reward_info.extra.get("details"))
                    .and_then(|details| details.get("objective"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                metadata: Map::new(),
            }],
            usage: if rollout_response.usage.is_empty() {
                json!({})
            } else {
                Value::Object(rollout_response.usage.clone())
            },
            trace_digest: rollout_response.trace.as_ref().map(trace_digest),
            actionable_side_info: response.get("actionable_side_info").cloned(),
            artifact_refs: Vec::new(),
            failure,
            metadata,
        })
    }
}

fn sensor_frame_id(
    candidate_id: &str,
    evaluation_stage: &str,
    example_id: &str,
    seed: i64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(candidate_id.as_bytes());
    digest.update(evaluation_stage.as_bytes());
    digest.update(example_id.as_bytes());
    digest.update(seed.to_string().as_bytes());
    let hex = format!("{:x}", digest.finalize());
    format!("sensor_{}", &hex[..12])
}

fn trace_digest(trace: &Value) -> TraceDigest {
    let mut digest = Sha256::new();
    digest.update(stable_json(trace).as_bytes());
    let mut call_site_ids = Vec::new();
    let mut llm_request_count = 0usize;
    let mut tool_call_count = 0usize;
    let event_history = trace
        .get("event_history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for event in &event_history {
        if event.get("llm_request").is_some()
            || event.get("type").and_then(Value::as_str) == Some("llm_request")
        {
            llm_request_count += 1;
        }
        if event.get("tool_call").is_some()
            || event.get("type").and_then(Value::as_str) == Some("tool_call")
        {
            tool_call_count += 1;
        }
        if let Some(call_site_id) = event.get("call_site_id").and_then(Value::as_str) {
            call_site_ids.push(call_site_id.to_string());
        }
    }
    call_site_ids.sort();
    call_site_ids.dedup();
    TraceDigest {
        sha256: format!("{:x}", digest.finalize()),
        event_count: event_history.len(),
        llm_request_count,
        tool_call_count,
        call_site_ids,
    }
}

fn rollout_trace_payload(
    row: &Value,
    response: &Value,
    rollout_response: &RolloutResponse,
    reward: f64,
) -> Value {
    let trace = rollout_response.trace.clone().unwrap_or_else(|| json!({}));
    let event_history = trace
        .get("event_history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    json!({
        "schema_version": "rollout_trace_payload.v1",
        "rollout_id": rollout_response.rollout_id,
        "trace_correlation_id": rollout_response.trace_correlation_id,
        "task_id": rollout_response.task_id,
        "summary": rollout_response.summary,
        "outcome": {
            "status": rollout_response.status,
            "success_status": rollout_response.success_status,
            "status_detail": rollout_response.status_detail,
            "reward": reward,
            "reward_info": rollout_response.reward_info,
        },
        "task_payload": {
            "example": row,
        },
        "request": {
            "candidate": rollout_response.metadata.get("candidate").cloned().unwrap_or(Value::Null),
            "source": rollout_response.source,
            "metadata": rollout_response.metadata,
        },
        "event_history": event_history,
        "trace": trace,
        "turns": rollout_response.turns,
        "events": rollout_response.events,
        "tool_calls": tool_calls_from_trace(response, rollout_response),
        "substitution_stats": substitution_stats_from_trace(response, rollout_response),
        "usage": rollout_response.usage,
        "raw_response": response,
    })
}

fn tool_calls_from_trace(response: &Value, rollout_response: &RolloutResponse) -> Vec<Value> {
    let mut tool_calls = Vec::new();
    collect_tool_calls(response, &mut tool_calls);
    if let Some(trace) = rollout_response.trace.as_ref() {
        collect_tool_calls(trace, &mut tool_calls);
    }
    for event in &rollout_response.events {
        collect_tool_calls(event, &mut tool_calls);
    }
    for turn in &rollout_response.turns {
        collect_tool_calls(turn, &mut tool_calls);
    }
    tool_calls.truncate(100);
    tool_calls
}

fn collect_tool_calls(value: &Value, out: &mut Vec<Value>) {
    match value {
        Value::Object(map) => {
            for key in ["tool_calls", "tools"] {
                if let Some(items) = map.get(key).and_then(Value::as_array) {
                    for item in items {
                        if item.is_object() {
                            out.push(item.clone());
                        }
                    }
                }
            }
            if map.get("tool_call").is_some()
                || map.get("type").and_then(Value::as_str) == Some("tool_call")
            {
                out.push(value.clone());
            }
            for child in map.values() {
                collect_tool_calls(child, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_tool_calls(item, out);
            }
        }
        _ => {}
    }
}

fn substitution_stats_from_trace(response: &Value, rollout_response: &RolloutResponse) -> Value {
    let mut attempts = 0i64;
    let mut applied = 0i64;
    let mut warnings = Vec::new();
    collect_substitution_stats(response, &mut attempts, &mut applied, &mut warnings);
    if let Some(trace) = rollout_response.trace.as_ref() {
        collect_substitution_stats(trace, &mut attempts, &mut applied, &mut warnings);
    }
    for event in &rollout_response.events {
        collect_substitution_stats(event, &mut attempts, &mut applied, &mut warnings);
    }
    warnings.truncate(20);
    json!({
        "attempted": attempts,
        "applied": applied,
        "warnings": warnings,
    })
}

fn collect_substitution_stats(
    value: &Value,
    attempts: &mut i64,
    applied: &mut i64,
    warnings: &mut Vec<String>,
) {
    match value {
        Value::Object(map) => {
            let mentions_substitution = map
                .get("kind")
                .or_else(|| map.get("type"))
                .and_then(Value::as_str)
                .map(|text| text.to_ascii_lowercase().contains("substitution"))
                .unwrap_or(false)
                || map.contains_key("substitution")
                || map.contains_key("substitutions");
            if mentions_substitution {
                *attempts += map
                    .get("attempted")
                    .or_else(|| map.get("attempts"))
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                *applied += map.get("applied").and_then(Value::as_i64).unwrap_or(0);
                if let Some(items) = map.get("warnings").and_then(Value::as_array) {
                    for item in items {
                        if let Some(text) = item.as_str() {
                            if !text.trim().is_empty() {
                                warnings.push(text.to_string());
                            }
                        }
                    }
                }
            }
            for child in map.values() {
                collect_substitution_stats(child, attempts, applied, warnings);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_substitution_stats(item, attempts, applied, warnings);
            }
        }
        _ => {}
    }
}
