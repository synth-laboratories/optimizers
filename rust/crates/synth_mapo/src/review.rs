use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::candidate::MapoRolloutRecord;
use crate::runtime::MAPO_ALGORITHM_ID;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapoReviewRow {
    pub schema_version: String,
    pub algorithm_id: String,
    pub run_id: String,
    pub row_id: String,
    pub issue_kind: String,
    pub severity: String,
    pub candidate_id: String,
    pub rollout_id: String,
    pub split: String,
    pub rollout_group: String,
    pub seed: i64,
    pub episode_index: usize,
    pub summary: String,
    pub metrics: Value,
    pub evidence: Value,
}

pub fn build_mapo_review_rows<'a>(
    run_id: &str,
    rollout_groups: impl IntoIterator<Item = &'a [MapoRolloutRecord]>,
) -> Vec<MapoReviewRow> {
    let mut rows = Vec::new();
    for rollouts in rollout_groups {
        for record in rollouts {
            rows.extend(review_rows_for_rollout(run_id, record));
        }
    }
    rows
}

fn review_rows_for_rollout(run_id: &str, record: &MapoRolloutRecord) -> Vec<MapoReviewRow> {
    let mut rows = Vec::new();
    if record.messages_rejected > 0 {
        rows.push(review_row(
            run_id,
            record,
            "message_rejected",
            "medium",
            format!(
                "rollout rejected {} party message(s); candidate must obey protocol limits",
                record.messages_rejected
            ),
        ));
    }
    if duplicate_claim_signal(&record.response) {
        rows.push(review_row(
            run_id,
            record,
            "duplicate_claim",
            "medium",
            "rollout evidence reports duplicate CLAIM coordination".to_string(),
        ));
    }
    if !record.success && record.messages_delivered == 0 {
        rows.push(review_row(
            run_id,
            record,
            "split_party_stuck",
            "low",
            "failed rollout delivered no tactical messages; inspect split-party coordination"
                .to_string(),
        ));
    }
    rows
}

fn review_row(
    run_id: &str,
    record: &MapoRolloutRecord,
    issue_kind: &str,
    severity: &str,
    summary: String,
) -> MapoReviewRow {
    MapoReviewRow {
        schema_version: "ohco.review_row.v1".to_string(),
        algorithm_id: MAPO_ALGORITHM_ID.to_string(),
        run_id: run_id.to_string(),
        row_id: format!(
            "{}:{}:{}",
            record.rollout_id, issue_kind, record.episode_index
        ),
        issue_kind: issue_kind.to_string(),
        severity: severity.to_string(),
        candidate_id: record.candidate_id.clone(),
        rollout_id: record.rollout_id.clone(),
        split: record.split.clone(),
        rollout_group: record.rollout_group.clone(),
        seed: record.seed,
        episode_index: record.episode_index,
        summary,
        metrics: json!({
            "success": record.success,
            "reward": record.reward,
            "messages_delivered": record.messages_delivered,
            "messages_rejected": record.messages_rejected,
            "message_chars": record.message_chars,
        }),
        evidence: json!({
            "task_instance_id": record.task_instance_id,
            "parent_rollout_id": record.parent_rollout_id,
            "parent_checkpoint_id": record.parent_checkpoint_id,
            "checkpoint_id": record.checkpoint_id,
            "response_summary": record.response.get("summary"),
            "communication": record
                .response
                .get("summary")
                .and_then(|summary| summary.get("communication")),
        }),
    }
}

fn duplicate_claim_signal(response: &Value) -> bool {
    let text = response.to_string().to_ascii_lowercase();
    text.contains("duplicate_claim") || text.contains("duplicate claim")
}
