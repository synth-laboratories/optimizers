//! Reserved OpenAI-compatible fine-tuning DTOs.
//!
//! These types exist so Layer 3 can lock the façade contract without exposing
//! live `/v1/fine_tuning/*` HTTP routes or billing claims. Mapping helpers
//! convert between OpenAI-shaped jobs and the canonical [`OptimizerRunRecord`].

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::observability::{
    algorithm_ids, OptimizerCapabilities, OptimizerRunRecord, OptimizerUsageSummary,
    OPTIMIZER_RUN_SCHEMA_VERSION,
};
use crate::sft_backend::SftJobStatus;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SynthFineTuningExtensions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heldout_file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_binding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_eval_policy: Option<String>,
    #[serde(default)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FineTuningJobCreateRequest {
    pub model: String,
    pub training_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<Map<String, Value>>,
    /// Synth extensions. Clients may also nest these under `extra_body.synth`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synth: Option<SynthFineTuningExtensions>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FineTuningJob {
    pub id: String,
    pub object: String,
    pub model: String,
    pub status: String,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fine_tuned_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synth: Option<SynthFineTuningExtensions>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FineTuningJobEvent {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub level: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FineTuningJobCheckpoint {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub fine_tuning_job_id: String,
    pub fine_tuned_model_checkpoint: String,
    pub step_number: u64,
    #[serde(default)]
    pub metrics: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FineTuningFileObject {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: i64,
    pub filename: String,
    pub purpose: String,
}

pub fn sft_status_to_openai(status: &SftJobStatus) -> &'static str {
    match status {
        SftJobStatus::Queued | SftJobStatus::Validating => "validating_files",
        SftJobStatus::Running => "running",
        SftJobStatus::Paused => "running",
        SftJobStatus::Succeeded => "succeeded",
        SftJobStatus::Failed => "failed",
        SftJobStatus::Cancelled => "cancelled",
    }
}

pub fn openai_status_to_optimizer(status: &str) -> &'static str {
    match status {
        "validating_files" | "queued" => "queued",
        "running" => "running",
        "succeeded" => "completed",
        "failed" => "failed",
        "cancelled" => "cancelled",
        other if !other.is_empty() => "running",
        _ => "unknown",
    }
}

/// Map a create request into a local optimizer run mirror (not submitted).
pub fn fine_tuning_create_to_run_record(
    request: &FineTuningJobCreateRequest,
    run_id: impl Into<String>,
    created_at: impl Into<String>,
) -> OptimizerRunRecord {
    let run_id = run_id.into();
    let synth = request.synth.clone().unwrap_or_default();
    OptimizerRunRecord {
        schema_version: OPTIMIZER_RUN_SCHEMA_VERSION.into(),
        id: run_id,
        algorithm_id: algorithm_ids::SFT.into(),
        algorithm_version: None,
        status: "queued".into(),
        source: "cloud".into(),
        objective: Some(format!("fine-tune {}", request.model)),
        project_ref: None,
        session_ref: None,
        created_at: created_at.into(),
        started_at: None,
        finished_at: None,
        cursor_seq: 0,
        capabilities: OptimizerCapabilities::sft_defaults(),
        execution_bindings: Vec::new(),
        input_refs: Vec::new(),
        output_refs: Vec::new(),
        visual_refs: Vec::new(),
        summary: json!({
            "baseModel": request.model,
            "adapter": synth.adapter,
            "backend": synth.backend.clone().unwrap_or_else(|| "tinker".into()),
            "trainingFile": request.training_file,
            "validationFile": request.validation_file,
            "selectionFile": synth.selection_file_id,
            "heldoutFile": synth.heldout_file_id,
            "slotBinding": synth.slot_binding,
        }),
        usage: OptimizerUsageSummary::default(),
        error: None,
    }
}

pub fn run_record_to_fine_tuning_job(run: &OptimizerRunRecord) -> FineTuningJob {
    let summary = run.summary.as_object();
    let model = summary
        .and_then(|s| s.get("baseModel"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let training_file = summary
        .and_then(|s| s.get("trainingFile"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let validation_file = summary
        .and_then(|s| s.get("validationFile"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let synth = Some(SynthFineTuningExtensions {
        backend: summary
            .and_then(|s| s.get("backend"))
            .and_then(Value::as_str)
            .map(str::to_string),
        adapter: summary
            .and_then(|s| s.get("adapter"))
            .and_then(Value::as_str)
            .map(str::to_string),
        selection_file_id: summary
            .and_then(|s| s.get("selectionFile"))
            .and_then(Value::as_str)
            .map(str::to_string),
        heldout_file_id: summary
            .and_then(|s| s.get("heldoutFile"))
            .and_then(Value::as_str)
            .map(str::to_string),
        slot_binding: summary
            .and_then(|s| s.get("slotBinding"))
            .and_then(Value::as_str)
            .map(str::to_string),
        checkpoint_eval_policy: None,
        extra: Map::new(),
    });
    FineTuningJob {
        id: run.id.clone(),
        object: "fine_tuning.job".into(),
        model,
        status: match run.status.as_str() {
            "queued" => "validating_files".into(),
            "running" | "paused" => "running".into(),
            "completed" => "succeeded".into(),
            "failed" => "failed".into(),
            "cancelled" => "cancelled".into(),
            other => other.into(),
        },
        created_at: 0,
        finished_at: None,
        fine_tuned_model: summary
            .and_then(|s| s.get("promotedCheckpointId"))
            .and_then(Value::as_str)
            .map(|ckpt| format!("{}:{}", run.summary.get("baseModel").and_then(Value::as_str).unwrap_or("model"), ckpt)),
        training_file,
        validation_file,
        error: run.error.clone(),
        synth,
    }
}

pub fn optimizer_event_to_fine_tuning_event(
    event_id: &str,
    created_at: i64,
    event_type: &str,
    message: &str,
    data: Option<Value>,
) -> FineTuningJobEvent {
    FineTuningJobEvent {
        id: event_id.into(),
        object: "fine_tuning.job.event".into(),
        created_at,
        level: "info".into(),
        message: if message.is_empty() {
            event_type.into()
        } else {
            message.into()
        },
        data,
    }
}

pub fn checkpoint_to_fine_tuning_checkpoint(
    run_id: &str,
    checkpoint_id: &str,
    step: u64,
    created_at: i64,
    metrics: Map<String, Value>,
) -> FineTuningJobCheckpoint {
    FineTuningJobCheckpoint {
        id: checkpoint_id.into(),
        object: "fine_tuning.job.checkpoint".into(),
        created_at,
        fine_tuning_job_id: run_id.into(),
        fine_tuned_model_checkpoint: format!("ft:ckpt:{checkpoint_id}"),
        step_number: step,
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_request_maps_to_optimizer_run_and_back() {
        let request = FineTuningJobCreateRequest {
            model: "openai/gpt-oss-20b".into(),
            training_file: "file_train".into(),
            validation_file: Some("file_val".into()),
            suffix: Some("craftax".into()),
            hyperparameters: Some(Map::from_iter([("n_epochs".into(), json!(2))])),
            synth: Some(SynthFineTuningExtensions {
                backend: Some("tinker".into()),
                adapter: Some("lora_r16".into()),
                selection_file_id: Some("file_selection".into()),
                heldout_file_id: Some("file_heldout".into()),
                slot_binding: Some("local-mac-01".into()),
                checkpoint_eval_policy: Some("every_checkpoint".into()),
                extra: Map::from_iter([("task".into(), json!("craftax"))]),
            }),
        };
        let run = fine_tuning_create_to_run_record(&request, "ftjob_1", "2026-08-09T16:00:00Z");
        assert_eq!(run.algorithm_id, "sft");
        assert_eq!(run.capabilities.checkpoints, true);
        assert_eq!(run.summary["backend"], "tinker");
        assert_eq!(run.summary["slotBinding"], "local-mac-01");

        let job = run_record_to_fine_tuning_job(&run);
        let encoded = serde_json::to_value(&job).unwrap();
        let decoded: FineTuningJob = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.object, "fine_tuning.job");
        assert_eq!(decoded.model, "openai/gpt-oss-20b");
        assert_eq!(decoded.synth.as_ref().unwrap().backend.as_deref(), Some("tinker"));
        assert_eq!(
            decoded.synth.as_ref().unwrap().heldout_file_id.as_deref(),
            Some("file_heldout")
        );
    }

    #[test]
    fn status_mapping_round_trips_common_states() {
        assert_eq!(sft_status_to_openai(&SftJobStatus::Running), "running");
        assert_eq!(openai_status_to_optimizer("succeeded"), "completed");
        assert_eq!(openai_status_to_optimizer("cancelled"), "cancelled");
    }

    #[test]
    fn checkpoint_and_event_dto_round_trips() {
        let event = optimizer_event_to_fine_tuning_event(
            "evt_1",
            1,
            "sft.step.metrics",
            "step 10",
            Some(json!({"step": 10})),
        );
        let ckpt = checkpoint_to_fine_tuning_checkpoint(
            "ftjob_1",
            "ckpt_10",
            10,
            2,
            Map::from_iter([("train_loss".into(), json!(1.2))]),
        );
        let event_again: FineTuningJobEvent =
            serde_json::from_value(serde_json::to_value(&event).unwrap()).unwrap();
        let ckpt_again: FineTuningJobCheckpoint =
            serde_json::from_value(serde_json::to_value(&ckpt).unwrap()).unwrap();
        assert_eq!(event_again.message, "step 10");
        assert_eq!(ckpt_again.step_number, 10);
        assert_eq!(ckpt_again.object, "fine_tuning.job.checkpoint");
    }
}
