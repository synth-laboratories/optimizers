//! Minimal, canonical JSON adapter for Containers' immutable Trace V5 evidence
//! contract.  The optimizer intentionally owns no alternate annotation schema:
//! it emits `synth.trace-evidence-bundle.v5` documents which Containers can
//! validate and project.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::{JesterkyWorkflowConfig, OptimizerError, Result};

pub const TRACE_V5_SCHEMA_VERSION: &str = "synth.trace.v5";
pub const TRACE_EVIDENCE_BUNDLE_V5_SCHEMA_VERSION: &str = "synth.trace-evidence-bundle.v5";
pub const TRACE_ANNOTATOR_V1_SCHEMA_VERSION: &str = "synth.trace-annotator.v1";
pub const ANNOTATION_V1_SCHEMA_VERSION: &str = "synth.annotation.v1";
pub const RECEIPT_V1_SCHEMA_VERSION: &str = "synth.receipt.v1";

#[derive(Clone, Debug)]
pub struct SealedTraceV5 {
    pub trace_id: String,
    pub content_digest: String,
    pub document: Value,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct TraceEvidenceBundleV5Materialization {
    pub bundle: Value,
    pub path: PathBuf,
    pub content_digest: String,
}

/// Read a sealed source trace, rejecting an unsealed or tampered document before
/// it can become annotator input.  This prevents a V4 convenience projection
/// from ever becoming the evidence authority.
pub fn load_sealed_trace_v5(path: &Path) -> Result<SealedTraceV5> {
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    let document: Value = serde_json::from_str(&text).map_err(|source| {
        OptimizerError::Config(format!(
            "invalid Trace V5 document {}: {source}",
            path.display()
        ))
    })?;
    if document.get("schema_version").and_then(Value::as_str) != Some(TRACE_V5_SCHEMA_VERSION) {
        return Err(OptimizerError::Config(format!(
            "annotation input {} is not a {} document",
            path.display(),
            TRACE_V5_SCHEMA_VERSION
        )));
    }
    let trace_id = required_string(&document, "trace_id", path)?;
    let content_digest = required_string(&document, "content_digest", path)?;
    let recomputed = content_digest_for(&document);
    if content_digest != recomputed {
        return Err(OptimizerError::Config(format!(
            "Trace V5 digest mismatch for {}: stored {content_digest}, recomputed {recomputed}",
            path.display()
        )));
    }
    Ok(SealedTraceV5 {
        trace_id,
        content_digest,
        document,
        path: path.to_path_buf(),
    })
}

/// A lossy transport projection for the existing Jesterky V4 reader.  The
/// projection explicitly carries the source V5 ref and digest; callers must
/// retain the V5 document and attach output to it via `build_*`, never treat
/// this file as a trace authority.
pub fn jesterky_v4_projection(trace: &SealedTraceV5, rollout: &Value) -> Value {
    let task_id = rollout
        .get("task_id")
        .or_else(|| rollout.get("example_id"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_id = rollout
        .get("candidate_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let summary = json!({
        "task_id": task_id,
        "candidate_id": candidate_id,
        "status": rollout.get("status").cloned().unwrap_or_else(|| json!("completed")),
        "rollout_metadata": rollout,
        "source_trace_ref": {
            "trace_id": trace.trace_id,
            "content_digest": trace.content_digest,
            "schema_version": TRACE_V5_SCHEMA_VERSION,
        },
    });
    json!({
        "schema_version": "synth_rollout_trace_v4",
        "trace_schema_version": 4,
        "rollout_id": format!("v5-{}", trace.trace_id),
        "trace_correlation_id": trace.trace_id,
        "status": rollout.get("status").cloned().unwrap_or_else(|| json!("completed")),
        "spans": [{
            "span_id": format!("v5-{}-projection", trace.trace_id),
            "call_index": 1,
            "run_id": trace.trace_id,
            "request": {"messages": [{"role":"system","content":"Trace V5 projection for GEPA annotation"}], "provider_hint":"trace_v5_projection"},
            "response": {"message": {"role":"assistant", "content": serde_json::to_string(&trace.document).unwrap_or_else(|_| "{}".to_string())}},
            "metadata": {"source_trace_ref": {"trace_id": trace.trace_id, "content_digest": trace.content_digest}}
        }],
        "events": [],
        "span_count": 1,
        "summary": summary,
        "metadata": {
            "source": "synth.trace.v5.jesterky_projection",
            "source_trace_ref": {"trace_id": trace.trace_id, "content_digest": trace.content_digest, "schema_version": TRACE_V5_SCHEMA_VERSION},
            "projection_loss": "Jesterky currently reads synth_rollout_trace_v4 transport files; the sealed V5 document remains the only annotation authority."
        }
    })
}

/// Produce a sealed V5 evidence bundle from one Jesterky result.  The result is
/// a descriptive proposer input only: no evaluator score, reward authority, or
/// benchmark verdict is ever emitted here.
pub fn build_jesterky_evidence_bundle(
    trace: &SealedTraceV5,
    scan: Option<&Value>,
    config: &JesterkyWorkflowConfig,
    config_digest: &str,
    spec_digest: &str,
    projection_digest: &str,
    started_at: &str,
    ended_at: &str,
    manifest_digest: &str,
    status: &str,
    error: Option<&str>,
) -> Value {
    let producer = json!({
        "kind": "agentic",
        "name": "jesterky",
        "version": "gepa-v5-adapter.v1",
        "model": config.model,
        "config_digest": config_digest,
    });
    let mut definition = json!({
        "annotator_id": format!("jesterky-gepa-{}", short_hash(config_digest)),
        "name": "Jesterky GEPA trace annotator",
        "purpose": "Produce descriptive, selector-grounded proposer context from a sealed Trace V5 projection.",
        "taxonomy": ["theme", "failure_mode", "reusable_rule", "blocker"],
        "version": "v1",
        "supported_trace_schemas": [TRACE_V5_SCHEMA_VERSION],
        "required_subject_scope": "trace",
        "reasoning_policy": "jesterky workflow manifest; no selection or evaluation authority",
        "grounding_requirement": "selector",
        "minimum_evidence": 1,
        "program_ref": config.spec,
        "model": config.model,
        "unavailable_evidence_behavior": "abstain",
        "confidence_semantics": "self_reported",
        "schema_version": TRACE_ANNOTATOR_V1_SCHEMA_VERSION,
        "metadata": {
            "provider": config.provider,
            "workflow_spec_digest": spec_digest,
            "workflow_config_digest": config_digest,
            "cadence": config.cadence,
            "max_targets": config.max_targets,
            "deduplicate_by_trace_digest": config.deduplicate_by_trace_digest,
            "not_authoritative_for": ["heldout_labels", "evaluation_scores", "selection_decisions", "rewards"]
        }
    });
    seal(&mut definition);
    let target = trace_selector(trace, Some("jesterky.v4-projection"));
    let scan = scan.cloned().unwrap_or_else(|| json!({}));
    let labels = scan
        .get("theme_tags")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let applied = status == "completed" && !labels.is_empty();
    let mut annotation = json!({
        "annotation_id": format!("ann_{}", short_hash(&format!("{}:{}:{}", trace.content_digest, config_digest, manifest_digest))),
        "annotator_id": definition["annotator_id"],
        "annotator_version": definition["version"],
        "annotator_digest": definition["content_digest"],
        "target": target,
        "annotation_type": "jesterky.trace_theme",
        "labels": labels,
        "author_kind": "agentic",
        "producer": producer,
        "created_at": ended_at,
        "grounding": if applied { "summary_only" } else { "source_unavailable" },
        "payload": {
            "severity": scan.get("severity").cloned().unwrap_or(Value::Null),
            "blocker": scan.get("blocker").cloned().unwrap_or(Value::Null),
            "failure_modes": scan.get("failure_modes").cloned().unwrap_or_else(|| json!([])),
            "reusable_rules": scan.get("reusable_rules").cloned().unwrap_or_else(|| json!([])),
            "prompt_harness_notes": scan.get("prompt_harness_notes").cloned().unwrap_or(Value::Null),
        },
        "confidence": scan.get("confidence").cloned().unwrap_or(Value::Null),
        "rationale": "Derived from the Jesterky V4 transport projection of the sealed source trace; descriptive proposer context only.",
        "evidence": [trace_selector(trace, Some("jesterky.v4-projection"))],
        "visibility": "private",
        "inspected_projection": "jesterky.v4-projection",
        "status": if applied { "applied" } else { "abstained" },
        "review_state": "unreviewed",
        "abstention_reason": if applied { Value::Null } else { json!(error.unwrap_or("Jesterky returned no usable descriptive labels")) },
        "inspection": {"source":"projection", "trace_body_read":false, "projection_id":"jesterky.v4-projection", "projection_digest":projection_digest},
        "schema_version": ANNOTATION_V1_SCHEMA_VERSION,
    });
    seal(&mut annotation);
    let mut receipt = json!({
        "receipt_id": format!("rcpt_{}", short_hash(&format!("{}:{}:{}", trace.content_digest, config_digest, ended_at))),
        "operation": "jesterky.annotate.trace_v5",
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "target_ids": [trace.trace_id],
        "producer": producer,
        "wall_time_seconds": Value::Null,
        "input_digests": [trace.content_digest, projection_digest, config_digest, spec_digest],
        "output_digests": [annotation["content_digest"], manifest_digest],
        "completeness": if applied { "complete" } else { "partial" },
        "errors": error.map(|message| vec![message]).unwrap_or_default(),
        "next_safe_action": if applied { Value::Null } else { json!("Wait for a new sealed Trace V5 digest before retrying annotation.") },
        "detail": {"max_spend_usd": config.max_spend_usd, "spend_usd": Value::Null, "spend_accounting": "unavailable", "not_authoritative_for": ["heldout_labels", "evaluation_scores", "selection_decisions", "rewards"]},
        "schema_version": RECEIPT_V1_SCHEMA_VERSION,
    });
    seal(&mut receipt);
    let mut bundle = json!({
        "bundle_id": format!("evb_{}", short_hash(&format!("{}:{}:{}", trace.content_digest, config_digest, manifest_digest))),
        "trace_ref": {"trace_id": trace.trace_id, "content_digest": trace.content_digest, "schema_version": TRACE_V5_SCHEMA_VERSION},
        "created_at": ended_at,
        "criteria": [], "rubrics": [], "verifier_definitions": [],
        "annotator_definitions": [definition],
        "reward_definitions": [],
        "annotations": [annotation],
        "verifier_results": [], "reward_records": [], "reward_aggregations": [], "evaluation_results": [], "benchmark_verdicts": [],
        "receipts": [receipt], "artifacts": [],
        "schema_version": TRACE_EVIDENCE_BUNDLE_V5_SCHEMA_VERSION,
        "metadata": {"source": "synth_gepa.jesterky_v5_adapter", "source_trace_path": trace.path, "annotation_policy": {"max_targets":config.max_targets, "cadence":config.cadence, "deduplicate_by_trace_digest":config.deduplicate_by_trace_digest}, "authority": "descriptive proposer context only"}
    });
    seal(&mut bundle);
    bundle
}

pub fn write_evidence_bundle_v5(
    dir: &Path,
    bundle: &Value,
) -> Result<TraceEvidenceBundleV5Materialization> {
    let trace_id = bundle
        .pointer("/trace_ref/trace_id")
        .and_then(Value::as_str)
        .unwrap_or("trace");
    let digest = bundle
        .get("content_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            OptimizerError::Config("cannot write an unsealed V5 evidence bundle".to_string())
        })?
        .to_string();
    let path = dir.join(format!("{}.evidence.v5.json", sanitize(trace_id)));
    fs::write(&path, serde_json::to_string_pretty(bundle)?)
        .map_err(|source| OptimizerError::io(&path, source))?;
    Ok(TraceEvidenceBundleV5Materialization {
        bundle: bundle.clone(),
        path,
        content_digest: digest,
    })
}

pub fn content_digest_for(value: &Value) -> String {
    let mut canonical = canonical_payload(value);
    if let Value::Object(map) = &mut canonical {
        map.remove("content_digest");
    }
    format!(
        "sha256:{:x}",
        Sha256::digest(canonical_json(&canonical).as_bytes())
    )
}

pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&canonical_payload(value)).expect("canonical JSON value")
}

fn seal(value: &mut Value) {
    let digest = content_digest_for(value);
    value["content_digest"] = json!(digest);
}

fn canonical_payload(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        Value::Array(items) => Value::Array(items.iter().map(canonical_payload).collect()),
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (key, item) in map {
                if !item.is_null() {
                    sorted.insert(key.clone(), canonical_payload(item));
                }
            }
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        primitive => primitive.clone(),
    }
}

fn trace_selector(trace: &SealedTraceV5, projection: Option<&str>) -> Value {
    json!({"trace_id":trace.trace_id, "trace_digest":trace.content_digest, "kind":"trace", "source_projection":projection, "schema_version":"synth.trace-selector.v1"})
}

fn required_string(value: &Value, key: &str, path: &Path) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            OptimizerError::Config(format!(
                "Trace V5 {} is missing non-empty {key}",
                path.display()
            ))
        })
}

fn short_hash(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))[..16].to_string()
}

fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_tampered_trace_and_keeps_annotations_non_authoritative() {
        let path = std::env::temp_dir().join(format!(
            "synth-trace-v5-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut document =
            json!({"schema_version":TRACE_V5_SCHEMA_VERSION, "trace_id":"trace_1", "events":[]});
        seal(&mut document);
        fs::write(&path, serde_json::to_string(&document).unwrap()).unwrap();
        let trace = load_sealed_trace_v5(&path).unwrap();
        let mut config = JesterkyWorkflowConfig::default();
        config.model = Some("pinned-model".to_string());
        let bundle = build_jesterky_evidence_bundle(
            &trace,
            Some(&json!({"theme_tags":["stall"]})),
            &config,
            "sha256:config",
            "sha256:spec",
            "sha256:projection",
            "2026-08-20T00:00:00Z",
            "2026-08-20T00:00:01Z",
            "sha256:manifest",
            "completed",
            None,
        );
        assert_eq!(
            bundle["schema_version"],
            TRACE_EVIDENCE_BUNDLE_V5_SCHEMA_VERSION
        );
        assert_eq!(bundle["benchmark_verdicts"], json!([]));
        assert_eq!(bundle["reward_records"], json!([]));
        document["trace_id"] = json!("tampered");
        fs::write(&path, serde_json::to_string(&document).unwrap()).unwrap();
        assert!(load_sealed_trace_v5(&path).is_err());
        let _ = fs::remove_file(path);
    }
}
