use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvariantReport {
    pub schema_version: String,
    pub report_id: String,
    pub run_id: String,
    pub status: String,
    pub checked_at: String,
    pub violation_count: u64,
    #[serde(default)]
    pub summary: Value,
    #[serde(default)]
    pub violations: Vec<InvariantViolation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvariantViolation {
    pub schema_version: String,
    pub violation_id: String,
    pub run_id: String,
    pub invariant_id: String,
    pub severity: String,
    pub subject_type: String,
    pub subject_id: String,
    pub message: String,
    #[serde(default)]
    pub repair_hint: Option<String>,
    #[serde(default)]
    pub details: Value,
}

pub struct InvariantViolationInput<'a> {
    pub run_id: &'a str,
    pub invariant_id: &'a str,
    pub severity: &'a str,
    pub subject_type: &'a str,
    pub subject_id: &'a str,
    pub message: String,
    pub repair_hint: Option<String>,
    pub details: Value,
}

pub struct CountMismatchInput<'a> {
    pub run_id: &'a str,
    pub invariant_id: &'a str,
    pub severity: &'a str,
    pub source_table: &'a str,
    pub source_count: u64,
    pub derived_table: &'a str,
    pub derived_count: u64,
    pub comparator: &'a str,
}

impl InvariantReport {
    pub fn new(
        run_id: &str,
        checked_at: &str,
        violations: Vec<InvariantViolation>,
        summary: Value,
    ) -> Self {
        let status = if violations
            .iter()
            .any(|violation| violation.severity == "error")
        {
            "fail"
        } else if violations.is_empty() {
            "pass"
        } else {
            "warn"
        };
        Self {
            schema_version: "invariant_report.v1".to_string(),
            report_id: stable_id("invreport", &[run_id, "workspace_invariants"]),
            run_id: run_id.to_string(),
            status: status.to_string(),
            checked_at: checked_at.to_string(),
            violation_count: violations.len() as u64,
            summary,
            violations,
        }
    }
}

impl InvariantViolation {
    pub fn new(input: InvariantViolationInput<'_>) -> Self {
        Self {
            schema_version: "invariant_violation.v1".to_string(),
            violation_id: stable_id(
                "invvio",
                &[
                    input.run_id,
                    input.invariant_id,
                    input.subject_type,
                    input.subject_id,
                ],
            ),
            run_id: input.run_id.to_string(),
            invariant_id: input.invariant_id.to_string(),
            severity: input.severity.to_string(),
            subject_type: input.subject_type.to_string(),
            subject_id: input.subject_id.to_string(),
            message: input.message,
            repair_hint: input.repair_hint,
            details: input.details,
        }
    }

    pub fn count_mismatch(input: CountMismatchInput<'_>) -> Self {
        Self::new(InvariantViolationInput {
            run_id: input.run_id,
            invariant_id: input.invariant_id,
            severity: input.severity,
            subject_type: "projection",
            subject_id: input.invariant_id,
            message: format!(
                "workspace invariant {} failed: {} count {} must {} {} count {}",
                input.invariant_id,
                input.derived_table,
                input.derived_count,
                input.comparator,
                input.source_table,
                input.source_count
            ),
            repair_hint: Some(
                "rerun projection repair or rebuild the run workspace from events/cache"
                    .to_string(),
            ),
            details: json!({
                "source_table": input.source_table,
                "source_count": input.source_count,
                "derived_table": input.derived_table,
                "derived_count": input.derived_count,
                "comparator": input.comparator,
            }),
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
