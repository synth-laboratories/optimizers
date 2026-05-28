use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::artifacts::ArtifactRef;
use crate::failures::FailurePayload;
use crate::sensors::SensorFrame;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceAnnotation {
    pub schema_version: String,
    pub annotation_id: String,
    pub sensor_frame_id: String,
    pub candidate_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub example_id: String,
    pub evaluation_stage: String,
    pub backend: String,
    pub status: String,
    pub summary: String,
    #[serde(default)]
    pub trace_sha256: Option<String>,
    pub event_count: u64,
    pub llm_request_count: u64,
    pub tool_call_count: u64,
    #[serde(default)]
    pub call_site_ids: Vec<String>,
    pub support_count: u64,
    pub confidence: f64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceFrame {
    pub schema_version: String,
    pub evidence_frame_id: String,
    pub subject_type: String,
    pub subject_id: String,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub sensor_frame_id: Option<String>,
    pub kind: String,
    pub source: String,
    pub summary: String,
    #[serde(default)]
    pub score: Option<f64>,
    pub severity: String,
    pub evidence: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierJob {
    pub schema_version: String,
    pub verifier_job_id: String,
    pub verifier_id: String,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub target_type: String,
    pub target_id: String,
    pub status: String,
    #[serde(default)]
    pub score: Option<f64>,
    pub severity: String,
    pub summary: String,
    pub result: Value,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub evidence_frame_ids: Vec<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentResult {
    pub schema_version: String,
    pub result_id: String,
    pub invocation_id: String,
    pub status: String,
    pub summary: String,
    #[serde(default)]
    pub artifact_refs: Vec<ArtifactRef>,
    #[serde(default)]
    pub evidence_frame_ids: Vec<String>,
    #[serde(default)]
    pub usage: Value,
    pub cost_usd: f64,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub output: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentInvocation {
    pub schema_version: String,
    pub invocation_id: String,
    pub role: String,
    pub backend: String,
    pub trigger: String,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub target_type: String,
    pub target_id: String,
    pub status: String,
    pub input: Value,
    pub result: SubagentResult,
    #[serde(default)]
    pub usage: Value,
    pub cost_usd: f64,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorDerivedRecords {
    pub trace_annotation: TraceAnnotation,
    pub evidence_frames: Vec<EvidenceFrame>,
    pub verifier_job: VerifierJob,
    pub subagent_invocation: SubagentInvocation,
}

impl SensorDerivedRecords {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let trace_annotation = TraceAnnotation::from_sensor_frame(frame);
        let evidence_frames = EvidenceFrame::from_sensor_frame(frame, &trace_annotation);
        let evidence_frame_ids = evidence_frames
            .iter()
            .map(|frame| frame.evidence_frame_id.clone())
            .collect::<Vec<_>>();
        let verifier_job = VerifierJob::from_sensor_frame(frame, &evidence_frame_ids);
        let subagent_invocation =
            SubagentInvocation::from_sensor_frame(frame, &trace_annotation, &evidence_frame_ids);
        Self {
            trace_annotation,
            evidence_frames,
            verifier_job,
            subagent_invocation,
        }
    }
}

impl TraceAnnotation {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let trace_digest = frame.trace_digest.as_ref();
        let event_count = trace_digest
            .map(|digest| digest.event_count as u64)
            .unwrap_or(0);
        let llm_request_count = trace_digest
            .map(|digest| digest.llm_request_count as u64)
            .unwrap_or(0);
        let tool_call_count = trace_digest
            .map(|digest| digest.tool_call_count as u64)
            .unwrap_or(0);
        let call_site_ids = trace_digest
            .map(|digest| digest.call_site_ids.clone())
            .unwrap_or_default();
        let support_count = llm_request_count + tool_call_count;
        let status = if trace_digest.is_some() {
            "completed"
        } else {
            "skipped"
        };
        let summary = if trace_digest.is_some() {
            format!(
                "rollout trace for {} has {} events, {} llm requests, {} tool calls, status {}, reward {:.6}",
                frame.example_id,
                event_count,
                llm_request_count,
                tool_call_count,
                frame.status,
                frame.reward
            )
        } else {
            format!(
                "rollout for {} has no trace payload, status {}, reward {:.6}",
                frame.example_id, frame.status, frame.reward
            )
        };
        Self {
            schema_version: "trace_annotation.v1".to_string(),
            annotation_id: stable_id(
                "traceann",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    trace_digest
                        .map(|digest| digest.sha256.as_str())
                        .unwrap_or("no_trace"),
                ],
            ),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            candidate_id: frame.candidate_id.clone(),
            rollout_id: frame.rollout_id.clone(),
            example_id: frame.example_id.clone(),
            evaluation_stage: frame.evaluation_stage.clone(),
            backend: "inline_deterministic".to_string(),
            status: status.to_string(),
            summary,
            trace_sha256: trace_digest.map(|digest| digest.sha256.clone()),
            event_count,
            llm_request_count,
            tool_call_count,
            call_site_ids,
            support_count,
            confidence: if support_count > 0 { 0.85 } else { 0.35 },
            metadata: Map::new(),
        }
    }
}

impl EvidenceFrame {
    pub fn from_sensor_frame(
        frame: &SensorFrame,
        annotation: &TraceAnnotation,
    ) -> Vec<EvidenceFrame> {
        let mut frames = vec![Self {
            schema_version: "evidence_frame.v1".to_string(),
            evidence_frame_id: stable_id(
                "evidence",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    "rollout_outcome",
                ],
            ),
            subject_type: "sensor_frame".to_string(),
            subject_id: frame.sensor_frame_id.clone(),
            candidate_id: Some(frame.candidate_id.clone()),
            sensor_frame_id: Some(frame.sensor_frame_id.clone()),
            kind: "rollout_outcome".to_string(),
            source: "container.reward_info".to_string(),
            summary: format!(
                "rollout {} on {} produced reward {:.6} with status {}",
                frame.evaluation_stage, frame.example_id, frame.reward, frame.status
            ),
            score: Some(frame.reward),
            severity: severity_for_frame(frame).to_string(),
            evidence: json!({
                "reward": frame.reward,
                "status": &frame.status,
                "success_status": &frame.success_status,
                "objective_scores": &frame.objective_scores,
                "actionable_side_info": &frame.actionable_side_info,
                "failure": &frame.failure,
            }),
            metadata: Map::new(),
        }];
        frames.push(Self {
            schema_version: "evidence_frame.v1".to_string(),
            evidence_frame_id: stable_id(
                "evidence",
                &[&frame.candidate_id, &frame.sensor_frame_id, "trace_summary"],
            ),
            subject_type: "trace_annotation".to_string(),
            subject_id: annotation.annotation_id.clone(),
            candidate_id: Some(frame.candidate_id.clone()),
            sensor_frame_id: Some(frame.sensor_frame_id.clone()),
            kind: "trace_summary".to_string(),
            source: "trace_annotation.inline_deterministic".to_string(),
            summary: annotation.summary.clone(),
            score: None,
            severity: if annotation.status == "completed" {
                "info".to_string()
            } else {
                "warning".to_string()
            },
            evidence: json!({
                "annotation_id": &annotation.annotation_id,
                "trace_sha256": &annotation.trace_sha256,
                "event_count": annotation.event_count,
                "llm_request_count": annotation.llm_request_count,
                "tool_call_count": annotation.tool_call_count,
                "call_site_ids": &annotation.call_site_ids,
                "confidence": annotation.confidence,
            }),
            metadata: Map::new(),
        });
        frames
    }
}

impl VerifierJob {
    pub fn from_sensor_frame(frame: &SensorFrame, evidence_frame_ids: &[String]) -> Self {
        let outcome = if frame.failure.is_some() {
            "failed"
        } else {
            "observed"
        };
        Self {
            schema_version: "verifier_job.v1".to_string(),
            verifier_job_id: stable_id(
                "verifier",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    "container_reward_verifier",
                ],
            ),
            verifier_id: "container_reward_verifier".to_string(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            target_type: "sensor_frame".to_string(),
            target_id: frame.sensor_frame_id.clone(),
            status: "completed".to_string(),
            score: Some(frame.reward),
            severity: severity_for_frame(frame).to_string(),
            summary: format!(
                "container verifier {} reward {:.6} for {}",
                outcome, frame.reward, frame.example_id
            ),
            result: json!({
                "outcome": outcome,
                "reward": frame.reward,
                "status": &frame.status,
                "success_status": &frame.success_status,
                "objective_scores": &frame.objective_scores,
            }),
            failure: frame.failure.clone(),
            evidence_frame_ids: evidence_frame_ids.to_vec(),
            metadata: Map::new(),
        }
    }
}

impl SubagentInvocation {
    pub fn from_sensor_frame(
        frame: &SensorFrame,
        annotation: &TraceAnnotation,
        evidence_frame_ids: &[String],
    ) -> Self {
        let invocation_id = stable_id(
            "subagent",
            &[
                &frame.candidate_id,
                &frame.sensor_frame_id,
                "rollout_evidence_annotator",
            ],
        );
        let result = SubagentResult {
            schema_version: "subagent_result.v1".to_string(),
            result_id: stable_id("subagent_result", &[&invocation_id]),
            invocation_id: invocation_id.clone(),
            status: "completed".to_string(),
            summary: annotation.summary.clone(),
            artifact_refs: Vec::new(),
            evidence_frame_ids: evidence_frame_ids.to_vec(),
            usage: json!({}),
            cost_usd: 0.0,
            failure: None,
            output: json!({
                "annotation_id": &annotation.annotation_id,
                "evidence_frame_ids": evidence_frame_ids,
                "backend": "inline_deterministic",
            }),
        };
        Self {
            schema_version: "subagent_invocation.v1".to_string(),
            invocation_id,
            role: "rollout_evidence_annotator".to_string(),
            backend: "inline_deterministic".to_string(),
            trigger: "sensor_frame_persisted".to_string(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            target_type: "sensor_frame".to_string(),
            target_id: frame.sensor_frame_id.clone(),
            status: "completed".to_string(),
            input: json!({
                "candidate_id": &frame.candidate_id,
                "sensor_frame_id": &frame.sensor_frame_id,
                "rollout_id": &frame.rollout_id,
                "trace_digest": &frame.trace_digest,
                "status": &frame.status,
                "reward": frame.reward,
            }),
            result,
            usage: json!({}),
            cost_usd: 0.0,
            failure: None,
            metadata: Map::new(),
        }
    }
}

fn severity_for_frame(frame: &SensorFrame) -> &'static str {
    if frame.failure.is_some() {
        "error"
    } else if frame.reward <= 0.0 {
        "warning"
    } else {
        "info"
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
