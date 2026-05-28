use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionFreshnessRecord {
    pub schema_version: String,
    pub projection_id: String,
    pub run_id: String,
    pub projection_name: String,
    pub status: String,
    pub source_table: String,
    pub source_count: u64,
    pub projected_table: String,
    pub projected_count: u64,
    pub lag_count: u64,
    pub checked_at: String,
    #[serde(default)]
    pub details: Value,
}

impl ProjectionFreshnessRecord {
    pub fn derived_count(
        run_id: &str,
        projection_name: &str,
        source_table: &str,
        source_count: u64,
        projected_table: &str,
        projected_count: u64,
        checked_at: &str,
    ) -> Self {
        let lag_count = source_count.saturating_sub(projected_count);
        let status = if lag_count == 0 { "fresh" } else { "stale" };
        Self {
            schema_version: "projection_freshness.v1".to_string(),
            projection_id: stable_id("projection", &[run_id, projection_name]),
            run_id: run_id.to_string(),
            projection_name: projection_name.to_string(),
            status: status.to_string(),
            source_table: source_table.to_string(),
            source_count,
            projected_table: projected_table.to_string(),
            projected_count,
            lag_count,
            checked_at: checked_at.to_string(),
            details: json!({
                "rule": "projected_count_must_cover_source_count",
            }),
        }
    }

    pub fn exact_count(
        run_id: &str,
        projection_name: &str,
        source_table: &str,
        source_count: u64,
        projected_table: &str,
        projected_count: u64,
        checked_at: &str,
    ) -> Self {
        let lag_count = if projected_count == source_count {
            0
        } else {
            source_count.abs_diff(projected_count)
        };
        let status = if lag_count == 0 { "fresh" } else { "stale" };
        Self {
            schema_version: "projection_freshness.v1".to_string(),
            projection_id: stable_id("projection", &[run_id, projection_name]),
            run_id: run_id.to_string(),
            projection_name: projection_name.to_string(),
            status: status.to_string(),
            source_table: source_table.to_string(),
            source_count,
            projected_table: projected_table.to_string(),
            projected_count,
            lag_count,
            checked_at: checked_at.to_string(),
            details: json!({
                "rule": "projected_count_must_equal_source_count",
            }),
        }
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
