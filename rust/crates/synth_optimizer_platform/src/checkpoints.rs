use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub schema_version: String,
    pub checkpoint_id: String,
    pub sequence_number: u64,
    pub checkpoint_kind: String,
    pub status: String,
    pub run_state: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub evaluation_stage: Option<String>,
    #[serde(default)]
    pub best_candidate_id: Option<String>,
    pub candidate_count: u64,
    pub frontier_count: u64,
    pub rollout_count: u64,
    pub cost_usd: f64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub snapshot: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub created_at: String,
}

pub struct CheckpointInput<'a> {
    pub sequence_number: u64,
    pub checkpoint_kind: &'a str,
    pub status: &'a str,
    pub run_state: &'a str,
    pub reason: Option<&'a str>,
    pub generation: Option<u64>,
    pub candidate_id: Option<&'a str>,
    pub evaluation_stage: Option<&'a str>,
    pub best_candidate_id: Option<&'a str>,
    pub candidate_count: u64,
    pub frontier_count: u64,
    pub rollout_count: u64,
    pub cost_usd: f64,
    pub usage: Value,
    pub snapshot: Value,
    pub metadata: Map<String, Value>,
}

impl CheckpointRecord {
    pub fn from_input(input: CheckpointInput<'_>) -> Self {
        Self {
            schema_version: "checkpoint_record.v1".to_string(),
            checkpoint_id: stable_id(
                "checkpoint",
                &[
                    &input.sequence_number.to_string(),
                    input.checkpoint_kind,
                    input.status,
                ],
            ),
            sequence_number: input.sequence_number,
            checkpoint_kind: input.checkpoint_kind.to_string(),
            status: input.status.to_string(),
            run_state: input.run_state.to_string(),
            reason: input.reason.map(str::to_string),
            generation: input.generation,
            candidate_id: input.candidate_id.map(str::to_string),
            evaluation_stage: input.evaluation_stage.map(str::to_string),
            best_candidate_id: input.best_candidate_id.map(str::to_string),
            candidate_count: input.candidate_count,
            frontier_count: input.frontier_count,
            rollout_count: input.rollout_count,
            cost_usd: input.cost_usd,
            usage: input.usage,
            snapshot: input.snapshot,
            metadata: input.metadata,
            created_at: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
        }
    }

    pub fn summary(&self) -> Value {
        json!({
            "checkpoint_kind": self.checkpoint_kind,
            "status": self.status,
            "run_state": self.run_state,
            "reason": self.reason,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "evaluation_stage": self.evaluation_stage,
            "best_candidate_id": self.best_candidate_id,
            "candidate_count": self.candidate_count,
            "frontier_count": self.frontier_count,
            "rollout_count": self.rollout_count,
            "cost_usd": self.cost_usd,
        })
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
