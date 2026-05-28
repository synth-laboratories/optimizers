use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::failures::FailurePayload;
use crate::sensors::SensorFrame;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutRecord {
    pub schema_version: String,
    pub rollout_record_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub status: String,
    pub reward: f64,
    #[serde(default)]
    pub trace_sha256: Option<String>,
    pub event_count: u64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutEventRecord {
    pub schema_version: String,
    pub rollout_event_id: String,
    pub rollout_record_id: String,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub sequence_number: u64,
    pub event_type: String,
    #[serde(default)]
    pub kind: String,
    pub summary: String,
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub trace_ref: Option<String>,
    pub event: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorRolloutRecords {
    pub rollout: RolloutRecord,
    pub events: Vec<RolloutEventRecord>,
}

impl SensorRolloutRecords {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let rollout = RolloutRecord::from_sensor_frame(frame);
        let events = RolloutEventRecord::from_sensor_frame(frame, &rollout);
        Self { rollout, events }
    }
}

impl RolloutRecord {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let trace_digest = frame.trace_digest.as_ref();
        Self {
            schema_version: "rollout_record.v1".to_string(),
            rollout_record_id: stable_id(
                "rolloutrec",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    &frame.evaluation_stage,
                ],
            ),
            rollout_id: frame.rollout_id.clone(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            example_id: frame.example_id.clone(),
            seed: frame.seed,
            split: frame.split.clone(),
            evaluation_stage: frame.evaluation_stage.clone(),
            status: frame.status.clone(),
            reward: frame.reward,
            trace_sha256: trace_digest.map(|digest| digest.sha256.clone()),
            event_count: trace_digest
                .map(|digest| digest.event_count as u64)
                .unwrap_or(0),
            usage: frame.usage.clone(),
            failure: frame.failure.clone(),
            metadata: frame.metadata.clone(),
        }
    }
}

impl RolloutEventRecord {
    pub fn from_sensor_frame(frame: &SensorFrame, rollout: &RolloutRecord) -> Vec<Self> {
        let observed_payload = json!({
            "candidate_id": &frame.candidate_id,
            "sensor_frame_id": &frame.sensor_frame_id,
            "rollout_id": &frame.rollout_id,
            "example_id": &frame.example_id,
            "seed": frame.seed,
            "split": &frame.split,
            "evaluation_stage": &frame.evaluation_stage,
            "status": &frame.status,
            "success_status": &frame.success_status,
            "reward": frame.reward,
            "failure": &frame.failure,
        });
        let trace_ref = frame
            .trace_digest
            .as_ref()
            .map(|digest| format!("trace_sha256:{}", digest.sha256));
        let mut records = vec![Self {
            schema_version: "rollout_event_record.v1".to_string(),
            rollout_event_id: stable_id(
                "rolloutev",
                &[&rollout.rollout_record_id, "1", "rollout_observed"],
            ),
            rollout_record_id: rollout.rollout_record_id.clone(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            sequence_number: 1,
            event_type: "rollout_observed".to_string(),
            kind: "rollout_observed".to_string(),
            summary: format!(
                "rollout {} for {} completed with status {} and reward {:.6}",
                frame.evaluation_stage, frame.example_id, frame.status, frame.reward
            ),
            payload: observed_payload.clone(),
            trace_ref: trace_ref.clone(),
            event: observed_payload,
        }];
        let mut next_sequence = 2u64;
        if let Some(trace_payload) = frame.metadata.get("rollout_trace") {
            records.push(Self {
                schema_version: "rollout_event_record.v1".to_string(),
                rollout_event_id: stable_id(
                    "rolloutev",
                    &[
                        &rollout.rollout_record_id,
                        &next_sequence.to_string(),
                        "trace",
                    ],
                ),
                rollout_record_id: rollout.rollout_record_id.clone(),
                candidate_id: frame.candidate_id.clone(),
                sensor_frame_id: frame.sensor_frame_id.clone(),
                sequence_number: next_sequence,
                event_type: "trace".to_string(),
                kind: "trace".to_string(),
                summary: format!(
                    "rollout trace payload captured for {} on {}",
                    frame.candidate_id, frame.example_id
                ),
                payload: trace_payload.clone(),
                trace_ref: trace_ref.clone(),
                event: json!({
                    "kind": "trace",
                    "payload": trace_payload,
                    "trace_ref": trace_ref,
                }),
            });
            next_sequence += 1;

            for (index, event) in trace_payload
                .get("event_history")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let kind = event_kind(event, "trace_event");
                let event_type = if kind.is_empty() {
                    "trace_event".to_string()
                } else {
                    kind.clone()
                };
                records.push(Self {
                    schema_version: "rollout_event_record.v1".to_string(),
                    rollout_event_id: stable_id(
                        "rolloutev",
                        &[
                            &rollout.rollout_record_id,
                            &next_sequence.to_string(),
                            "trace_event",
                            &index.to_string(),
                        ],
                    ),
                    rollout_record_id: rollout.rollout_record_id.clone(),
                    candidate_id: frame.candidate_id.clone(),
                    sensor_frame_id: frame.sensor_frame_id.clone(),
                    sequence_number: next_sequence,
                    event_type,
                    kind: "trace_event".to_string(),
                    summary: event_summary(event, "rollout trace event captured"),
                    payload: json!({
                        "index": index,
                        "event": event,
                    }),
                    trace_ref: trace_ref.clone(),
                    event: event.clone(),
                });
                next_sequence += 1;
            }

            for (index, tool_call) in trace_payload
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                records.push(Self {
                    schema_version: "rollout_event_record.v1".to_string(),
                    rollout_event_id: stable_id(
                        "rolloutev",
                        &[
                            &rollout.rollout_record_id,
                            &next_sequence.to_string(),
                            "tool_call",
                            &index.to_string(),
                        ],
                    ),
                    rollout_record_id: rollout.rollout_record_id.clone(),
                    candidate_id: frame.candidate_id.clone(),
                    sensor_frame_id: frame.sensor_frame_id.clone(),
                    sequence_number: next_sequence,
                    event_type: "tool_call".to_string(),
                    kind: "tool_call".to_string(),
                    summary: tool_call_summary(tool_call),
                    payload: json!({
                        "index": index,
                        "tool_call": tool_call,
                    }),
                    trace_ref: trace_ref.clone(),
                    event: tool_call.clone(),
                });
                next_sequence += 1;
            }

            if let Some(substitution_stats) = trace_payload.get("substitution_stats") {
                records.push(Self {
                    schema_version: "rollout_event_record.v1".to_string(),
                    rollout_event_id: stable_id(
                        "rolloutev",
                        &[
                            &rollout.rollout_record_id,
                            &next_sequence.to_string(),
                            "substitution_stats",
                        ],
                    ),
                    rollout_record_id: rollout.rollout_record_id.clone(),
                    candidate_id: frame.candidate_id.clone(),
                    sensor_frame_id: frame.sensor_frame_id.clone(),
                    sequence_number: next_sequence,
                    event_type: "substitution_stats".to_string(),
                    kind: "substitution_stats".to_string(),
                    summary: "rollout substitution stats captured".to_string(),
                    payload: substitution_stats.clone(),
                    trace_ref: trace_ref.clone(),
                    event: json!({
                        "kind": "substitution_stats",
                        "payload": substitution_stats,
                    }),
                });
                next_sequence += 1;
            }
        }
        if let Some(trace_digest) = &frame.trace_digest {
            records.push(Self {
                schema_version: "rollout_event_record.v1".to_string(),
                rollout_event_id: stable_id(
                    "rolloutev",
                    &[
                        &rollout.rollout_record_id,
                        &next_sequence.to_string(),
                        "trace_digest_observed",
                    ],
                ),
                rollout_record_id: rollout.rollout_record_id.clone(),
                candidate_id: frame.candidate_id.clone(),
                sensor_frame_id: frame.sensor_frame_id.clone(),
                sequence_number: next_sequence,
                event_type: "trace_digest_observed".to_string(),
                kind: "trace_digest".to_string(),
                summary: format!(
                    "trace digest has {} events, {} llm requests, and {} tool calls",
                    trace_digest.event_count,
                    trace_digest.llm_request_count,
                    trace_digest.tool_call_count
                ),
                payload: json!({
                    "trace_sha256": &trace_digest.sha256,
                    "event_count": trace_digest.event_count,
                    "llm_request_count": trace_digest.llm_request_count,
                    "tool_call_count": trace_digest.tool_call_count,
                    "call_site_ids": &trace_digest.call_site_ids,
                }),
                trace_ref,
                event: json!({
                    "trace_sha256": &trace_digest.sha256,
                    "event_count": trace_digest.event_count,
                    "llm_request_count": trace_digest.llm_request_count,
                    "tool_call_count": trace_digest.tool_call_count,
                    "call_site_ids": &trace_digest.call_site_ids,
                }),
            });
        }
        records
    }
}

fn event_kind(event: &Value, default: &str) -> String {
    event
        .get("kind")
        .or_else(|| event.get("type"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_string()
}

fn event_summary(event: &Value, default: &str) -> String {
    event
        .get("summary")
        .or_else(|| event.get("message"))
        .or_else(|| event.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .chars()
        .take(240)
        .collect()
}

fn tool_call_summary(tool_call: &Value) -> String {
    let name = tool_call
        .get("name")
        .or_else(|| tool_call.get("tool_name"))
        .or_else(|| tool_call.pointer("/function/name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("tool");
    let call_id = tool_call
        .get("id")
        .or_else(|| tool_call.get("tool_call_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match call_id {
        Some(call_id) => format!("tool call {name} ({call_id}) captured"),
        None => format!("tool call {name} captured"),
    }
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
