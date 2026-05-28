use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::sensors::SensorFrame;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsageLedgerRecord {
    pub schema_version: String,
    pub usage_ledger_id: String,
    pub boundary: String,
    pub source_type: String,
    pub source_id: String,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub evaluation_stage: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    pub call_count: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

pub struct UsageLedgerInput<'a> {
    pub boundary: &'a str,
    pub source_type: &'a str,
    pub source_id: &'a str,
    pub candidate_id: Option<&'a str>,
    pub evaluation_stage: Option<&'a str>,
    pub model: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub call_count: u64,
    pub usage: Value,
    pub cost_usd: f64,
    pub metadata: Map<String, Value>,
}

impl UsageLedgerRecord {
    pub fn from_input(input: UsageLedgerInput<'_>) -> Self {
        Self {
            schema_version: "usage_ledger_record.v1".to_string(),
            usage_ledger_id: stable_id(
                "usage",
                &[input.boundary, input.source_type, input.source_id],
            ),
            boundary: input.boundary.to_string(),
            source_type: input.source_type.to_string(),
            source_id: input.source_id.to_string(),
            candidate_id: input.candidate_id.map(str::to_string),
            evaluation_stage: input.evaluation_stage.map(str::to_string),
            model: input.model.map(str::to_string),
            provider: input.provider.map(str::to_string),
            call_count: input.call_count,
            prompt_tokens: usage_u64(&input.usage, &["prompt_tokens", "input_tokens"]),
            completion_tokens: usage_u64(&input.usage, &["completion_tokens", "output_tokens"]),
            total_tokens: usage_u64(&input.usage, &["total_tokens"]),
            cost_usd: input.cost_usd,
            usage: input.usage,
            metadata: input.metadata,
        }
    }

    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let mut metadata = Map::new();
        metadata.insert(
            "example_id".to_string(),
            Value::String(frame.example_id.clone()),
        );
        metadata.insert("seed".to_string(), json!(frame.seed));
        metadata.insert("split".to_string(), Value::String(frame.split.clone()));
        metadata.insert("status".to_string(), Value::String(frame.status.clone()));
        if let Some(rollout_id) = &frame.rollout_id {
            metadata.insert("rollout_id".to_string(), Value::String(rollout_id.clone()));
        }
        let cost_usd = frame
            .usage
            .get("cost_usd")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Self::from_input(UsageLedgerInput {
            boundary: "container.rollout",
            source_type: "sensor_frame",
            source_id: &frame.sensor_frame_id,
            candidate_id: Some(&frame.candidate_id),
            evaluation_stage: Some(&frame.evaluation_stage),
            model: None,
            provider: None,
            call_count: 1,
            usage: frame.usage.clone(),
            cost_usd,
            metadata,
        })
    }
}

fn usage_u64(usage: &Value, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
        .unwrap_or(0)
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
