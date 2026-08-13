//! Provider-neutral SFT backend boundary.
//!
//! Standalone SFT (`algorithm_id = "sft"`) uses this trait. The GELO plugin lane
//! `goex.sft.v1` is a separate orchestration path and must not be mislabeled as
//! standalone SFT — both may later share a Tinker adapter implementation, but
//! they do not share a state machine.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::observability::{
    algorithm_ids, item_kinds, OptimizerEvent, OptimizerItem, OptimizerLogLevel,
    OPTIMIZER_EVENT_SCHEMA_VERSION,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SftJobStatus {
    Queued,
    Validating,
    Running,
    Paused,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SftSubmitRequest {
    pub run_id: String,
    pub base_model: String,
    #[serde(default)]
    pub adapter: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub training_file_id: Option<String>,
    #[serde(default)]
    pub selection_file_id: Option<String>,
    #[serde(default)]
    pub heldout_file_id: Option<String>,
    #[serde(default)]
    pub hyperparameters: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SftSubmitResponse {
    pub run_id: String,
    pub provider_job_id: String,
    pub status: SftJobStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SftCheckpointRef {
    pub checkpoint_id: String,
    pub step: u64,
    pub digest: String,
    #[serde(default)]
    pub promoted: bool,
    /// Provider-native id, namespaced (e.g. `provider.tinker:...`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_checkpoint_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SftInferenceTarget {
    pub target_id: String,
    pub checkpoint_id: String,
    pub base_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Provider-native endpoint id, namespaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_endpoint_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SftMaterializedCheckpoint {
    pub checkpoint: SftCheckpointRef,
    pub artifact_digest: String,
    #[serde(default)]
    pub artifact_refs: Vec<Value>,
}

/// Provider-neutral training backend. Implementations must never leak secrets
/// or signed URLs into [`OptimizerEvent`] payloads.
pub trait SftBackend: Send {
    fn submit(&mut self, request: SftSubmitRequest) -> Result<SftSubmitResponse, String>;
    fn poll_or_stream_events(
        &mut self,
        run_id: &str,
        after_seq: u64,
    ) -> Result<Vec<OptimizerEvent>, String>;
    fn pause(&mut self, run_id: &str) -> Result<(), String>;
    fn resume(&mut self, run_id: &str) -> Result<(), String>;
    fn cancel(&mut self, run_id: &str) -> Result<(), String>;
    fn list_checkpoints(&self, run_id: &str) -> Result<Vec<SftCheckpointRef>, String>;
    fn materialize_checkpoint(
        &mut self,
        run_id: &str,
        checkpoint_id: &str,
    ) -> Result<SftMaterializedCheckpoint, String>;
    fn create_inference_target(
        &mut self,
        run_id: &str,
        checkpoint_id: &str,
    ) -> Result<SftInferenceTarget, String>;
    fn dispose(&mut self, run_id: &str) -> Result<(), String>;
}

#[derive(Clone, Debug)]
struct FakeJob {
    request: SftSubmitRequest,
    status: SftJobStatus,
    events: Vec<OptimizerEvent>,
    next_seq: u64,
    checkpoints: Vec<SftCheckpointRef>,
    disposed: bool,
}

/// In-memory backend that emits the same canonical event shapes as the Desktop
/// SFT fixture. Used for Layer 3 affordance tests; not a hosted worker.
#[derive(Default)]
pub struct FakeSftBackend {
    jobs: std::collections::BTreeMap<String, FakeJob>,
}

impl FakeSftBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SftBackend for FakeSftBackend {
    fn submit(&mut self, request: SftSubmitRequest) -> Result<SftSubmitResponse, String> {
        if self.jobs.contains_key(&request.run_id) {
            return Err(format!("run {} already submitted", request.run_id));
        }
        let provider_job_id = format!("provider.fake:{}", request.run_id);
        let mut job = FakeJob {
            request: request.clone(),
            status: SftJobStatus::Queued,
            events: Vec::new(),
            next_seq: 1,
            checkpoints: Vec::new(),
            disposed: false,
        };
        push_event(
            &mut job,
            "optimizer.run.created",
            json!({"status":"starting"}),
            None,
            Some(json!({
                "summary": {
                    "baseModel": request.base_model,
                    "adapter": request.adapter,
                    "backend": request.backend.clone().unwrap_or_else(|| "fake".into())
                }
            })),
        );
        push_event(
            &mut job,
            "sft.dataset.validated",
            json!({}),
            None,
            Some(json!({
                "splits": {
                    "train": {"count": 100, "digest": "sha256:train"},
                    "selection": {"count": 20, "digest": "sha256:selection"},
                    "heldout": {"count": 20, "digest": "sha256:heldout"}
                }
            })),
        );
        push_event(
            &mut job,
            "sft.training.queued",
            json!({"status":"queued"}),
            None,
            None,
        );
        push_event(
            &mut job,
            "sft.training.started",
            json!({"status":"running"}),
            None,
            None,
        );
        job.status = SftJobStatus::Running;
        push_event(
            &mut job,
            "sft.step.metrics",
            json!({
                "step": 10,
                "epoch": 1,
                "train_loss": 1.5,
                "validation_loss": 1.4,
                "learning_rate": 0.0002
            }),
            None,
            None,
        );
        let ckpt = SftCheckpointRef {
            checkpoint_id: format!("ckpt_{}", request.run_id),
            step: 10,
            digest: format!("sha256:ckpt_{}", request.run_id),
            promoted: false,
            provider_checkpoint_id: Some(format!("provider.fake:ckpt:{}", request.run_id)),
        };
        push_event(
            &mut job,
            "sft.checkpoint.created",
            json!({}),
            Some(OptimizerItem {
                kind: Some(item_kinds::CHECKPOINT.into()),
                item_type: None,
                id: Some(ckpt.checkpoint_id.clone()),
                status: Some("created".into()),
                raw: json!({
                    "step": ckpt.step,
                    "digest": ckpt.digest,
                    "promoted": false
                }),
            }),
            None,
        );
        job.checkpoints.push(ckpt);
        let response = SftSubmitResponse {
            run_id: request.run_id.clone(),
            provider_job_id,
            status: job.status.clone(),
        };
        self.jobs.insert(request.run_id, job);
        Ok(response)
    }

    fn poll_or_stream_events(
        &mut self,
        run_id: &str,
        after_seq: u64,
    ) -> Result<Vec<OptimizerEvent>, String> {
        let job = self
            .jobs
            .get(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        if job.disposed {
            return Err(format!("run {run_id} was disposed"));
        }
        Ok(job
            .events
            .iter()
            .filter(|event| event.sequence_number > after_seq)
            .cloned()
            .collect())
    }

    fn pause(&mut self, run_id: &str) -> Result<(), String> {
        let job = self
            .jobs
            .get_mut(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        if !matches!(job.status, SftJobStatus::Running) {
            return Err(format!("cannot pause run in status {:?}", job.status));
        }
        job.status = SftJobStatus::Paused;
        push_event(
            job,
            "sft.training.paused",
            json!({"status":"paused"}),
            None,
            None,
        );
        Ok(())
    }

    fn resume(&mut self, run_id: &str) -> Result<(), String> {
        let job = self
            .jobs
            .get_mut(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        if !matches!(job.status, SftJobStatus::Paused) {
            return Err(format!("cannot resume run in status {:?}", job.status));
        }
        job.status = SftJobStatus::Running;
        push_event(
            job,
            "sft.training.resumed",
            json!({"status":"running"}),
            None,
            None,
        );
        Ok(())
    }

    fn cancel(&mut self, run_id: &str) -> Result<(), String> {
        let job = self
            .jobs
            .get_mut(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        if matches!(
            job.status,
            SftJobStatus::Succeeded | SftJobStatus::Failed | SftJobStatus::Cancelled
        ) {
            return Err(format!("cannot cancel terminal run {:?}", job.status));
        }
        job.status = SftJobStatus::Cancelled;
        push_event(
            job,
            "sft.training.failed",
            json!({"status":"cancelled"}),
            None,
            None,
        );
        push_event(
            job,
            "optimizer.run.completed",
            json!({"status":"cancelled"}),
            None,
            None,
        );
        Ok(())
    }

    fn list_checkpoints(&self, run_id: &str) -> Result<Vec<SftCheckpointRef>, String> {
        let job = self
            .jobs
            .get(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        Ok(job.checkpoints.clone())
    }

    fn materialize_checkpoint(
        &mut self,
        run_id: &str,
        checkpoint_id: &str,
    ) -> Result<SftMaterializedCheckpoint, String> {
        let job = self
            .jobs
            .get_mut(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        let checkpoint = job
            .checkpoints
            .iter()
            .find(|ckpt| ckpt.checkpoint_id == checkpoint_id)
            .cloned()
            .ok_or_else(|| format!("unknown checkpoint {checkpoint_id}"))?;
        let artifact_digest = format!("sha256:model_{checkpoint_id}");
        push_event(
            job,
            "sft.model.materialized",
            json!({"status":"completed"}),
            Some(OptimizerItem {
                kind: Some(item_kinds::ARTIFACT.into()),
                item_type: None,
                id: Some(format!("model_{checkpoint_id}")),
                status: Some("ready".into()),
                raw: json!({
                    "baseModel": job.request.base_model,
                    "adapter": job.request.adapter,
                    "checkpointId": checkpoint_id,
                    "digest": artifact_digest
                }),
            }),
            None,
        );
        Ok(SftMaterializedCheckpoint {
            checkpoint,
            artifact_digest: artifact_digest.clone(),
            artifact_refs: vec![json!({
                "kind": "model",
                "id": format!("model_{checkpoint_id}"),
                "digest": artifact_digest
            })],
        })
    }

    fn create_inference_target(
        &mut self,
        run_id: &str,
        checkpoint_id: &str,
    ) -> Result<SftInferenceTarget, String> {
        let job = self
            .jobs
            .get(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        if !job
            .checkpoints
            .iter()
            .any(|ckpt| ckpt.checkpoint_id == checkpoint_id)
        {
            return Err(format!("unknown checkpoint {checkpoint_id}"));
        }
        Ok(SftInferenceTarget {
            target_id: format!("infer_{checkpoint_id}"),
            checkpoint_id: checkpoint_id.into(),
            base_model: job.request.base_model.clone(),
            adapter: job.request.adapter.clone(),
            provider_endpoint_id: Some(format!("provider.fake:endpoint:{checkpoint_id}")),
        })
    }

    fn dispose(&mut self, run_id: &str) -> Result<(), String> {
        let job = self
            .jobs
            .get_mut(run_id)
            .ok_or_else(|| format!("unknown sft run {run_id}"))?;
        job.disposed = true;
        Ok(())
    }
}

fn push_event(
    job: &mut FakeJob,
    event_type: &str,
    delta: Value,
    item: Option<OptimizerItem>,
    snapshot: Option<Value>,
) {
    let sequence_number = job.next_seq;
    job.next_seq += 1;
    job.events.push(OptimizerEvent {
        schema_version: OPTIMIZER_EVENT_SCHEMA_VERSION.into(),
        event_id: Some(format!("{}:{sequence_number}", job.request.run_id)),
        event_type: event_type.into(),
        sequence_number,
        created_at: format!("2026-08-09T16:00:{:02}Z", sequence_number.min(59)),
        run_id: job.request.run_id.clone(),
        algorithm_id: Some(algorithm_ids::SFT.into()),
        algorithm: None,
        level: Some(OptimizerLogLevel::Info),
        item,
        delta: delta.as_object().cloned().unwrap_or_default(),
        snapshot: snapshot.and_then(|value| value.as_object().cloned()),
        usage_delta: None,
        artifact_refs: Vec::new(),
        error: None,
        raw: json!({}),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request(run_id: &str) -> SftSubmitRequest {
        SftSubmitRequest {
            run_id: run_id.into(),
            base_model: "openai/gpt-oss-20b".into(),
            adapter: Some("lora_r16".into()),
            backend: Some("fake".into()),
            training_file_id: Some("file_train".into()),
            selection_file_id: Some("file_selection".into()),
            heldout_file_id: Some("file_heldout".into()),
            hyperparameters: Map::new(),
            metadata: Map::from_iter([("task".into(), json!("craftax"))]),
        }
    }

    #[test]
    fn fake_backend_submit_streams_checkpoints_and_pause() {
        let mut backend = FakeSftBackend::new();
        let submitted = backend.submit(sample_request("sft_fake_1")).unwrap();
        assert_eq!(submitted.status, SftJobStatus::Running);
        let events = backend.poll_or_stream_events("sft_fake_1", 0).unwrap();
        assert!(events.iter().any(|e| e.event_type == "sft.checkpoint.created"));
        assert!(events.iter().all(|e| e.algorithm_id() == "sft"));

        backend.pause("sft_fake_1").unwrap();
        let after_pause = backend
            .poll_or_stream_events("sft_fake_1", events.len() as u64)
            .unwrap();
        assert!(after_pause
            .iter()
            .any(|e| e.event_type == "sft.training.paused"));
        backend.resume("sft_fake_1").unwrap();

        let checkpoints = backend.list_checkpoints("sft_fake_1").unwrap();
        assert_eq!(checkpoints.len(), 1);
        let materialized = backend
            .materialize_checkpoint("sft_fake_1", &checkpoints[0].checkpoint_id)
            .unwrap();
        assert!(materialized.artifact_digest.starts_with("sha256:"));
        let target = backend
            .create_inference_target("sft_fake_1", &checkpoints[0].checkpoint_id)
            .unwrap();
        assert!(target
            .provider_endpoint_id
            .as_deref()
            .unwrap()
            .starts_with("provider.fake:"));
    }

    #[test]
    fn fake_backend_cancel_is_terminal() {
        let mut backend = FakeSftBackend::new();
        backend.submit(sample_request("sft_cancel")).unwrap();
        backend.cancel("sft_cancel").unwrap();
        assert!(backend.cancel("sft_cancel").is_err());
        let events = backend.poll_or_stream_events("sft_cancel", 0).unwrap();
        assert!(events
            .iter()
            .any(|e| e.event_type == "optimizer.run.completed"));
    }
}
