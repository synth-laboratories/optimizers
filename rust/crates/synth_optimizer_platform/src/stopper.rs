use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StopperStateRecord {
    pub schema_version: String,
    pub stopper_state_id: String,
    pub sequence_number: u64,
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub evaluation_stage: Option<String>,
    pub rollout_count: u64,
    pub max_total_rollouts: u64,
    #[serde(default)]
    pub remaining_rollouts: Option<u64>,
    pub cost_usd: f64,
    pub max_cost_usd: f64,
    pub cost_budget_enabled: bool,
    pub budget_exhausted: bool,
    pub checked_at: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

pub struct StopperStateInput<'a> {
    pub sequence_number: u64,
    pub status: &'a str,
    pub reason: Option<&'a str>,
    pub generation: Option<u64>,
    pub candidate_id: Option<&'a str>,
    pub evaluation_stage: Option<&'a str>,
    pub rollout_count: u64,
    pub max_total_rollouts: u64,
    pub cost_usd: f64,
    pub max_cost_usd: f64,
    pub metadata: Map<String, Value>,
}

impl StopperStateRecord {
    pub fn from_input(input: StopperStateInput<'_>) -> Self {
        let rollout_budget_exhausted =
            input.max_total_rollouts > 0 && input.rollout_count >= input.max_total_rollouts;
        let cost_budget_enabled = input.max_cost_usd > 0.0;
        let cost_budget_exhausted = cost_budget_enabled && input.cost_usd >= input.max_cost_usd;
        let remaining_rollouts = if input.max_total_rollouts == 0 {
            None
        } else {
            Some(input.max_total_rollouts.saturating_sub(input.rollout_count))
        };
        let budget_exhausted = rollout_budget_exhausted
            || cost_budget_exhausted
            || matches!(
                input.status,
                "rollout_budget_reached" | "cost_budget_reached" | "deferred_budget"
            );
        Self {
            schema_version: "stopper_state_record.v1".to_string(),
            stopper_state_id: stable_id(
                "stopper",
                &[&input.sequence_number.to_string(), input.status],
            ),
            sequence_number: input.sequence_number,
            status: input.status.to_string(),
            reason: input.reason.map(str::to_string),
            generation: input.generation,
            candidate_id: input.candidate_id.map(str::to_string),
            evaluation_stage: input.evaluation_stage.map(str::to_string),
            rollout_count: input.rollout_count,
            max_total_rollouts: input.max_total_rollouts,
            remaining_rollouts,
            cost_usd: input.cost_usd,
            max_cost_usd: input.max_cost_usd,
            cost_budget_enabled,
            budget_exhausted,
            checked_at: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
            metadata: input.metadata,
        }
    }

    pub fn summary(&self) -> Value {
        json!({
            "status": self.status,
            "reason": self.reason,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "evaluation_stage": self.evaluation_stage,
            "rollout_count": self.rollout_count,
            "max_total_rollouts": self.max_total_rollouts,
            "remaining_rollouts": self.remaining_rollouts,
            "cost_usd": self.cost_usd,
            "max_cost_usd": self.max_cost_usd,
            "cost_budget_enabled": self.cost_budget_enabled,
            "budget_exhausted": self.budget_exhausted,
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
