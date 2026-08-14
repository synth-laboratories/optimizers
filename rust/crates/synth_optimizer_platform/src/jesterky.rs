use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::cache::stable_json_hash;
use crate::error::{OptimizerError, Result};

pub const JESTERKY_WORKSPACE_READ_MODEL_SCHEMA_VERSION: &str =
    "synth_optimizers.jesterky_workspace_read_model.v1";

#[derive(Clone, Debug)]
struct TraceRow {
    value: Value,
    has_children: bool,
    has_score: bool,
}

pub fn read_jesterky_manifest(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    serde_json::from_str(&raw).map_err(OptimizerError::from)
}

pub fn looks_like_jesterky_manifest(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.get("trace").and_then(Value::as_object).is_some()
        && object.get("events").and_then(Value::as_array).is_some()
        && string_field(object, "run_id").is_some()
        && string_field(object, "workflow_name").is_some()
        && (string_field(object, "stop_reason").is_some()
            || string_field(object, "spec_hash").is_some())
}

pub fn jesterky_workspace_read_model(manifest: &Value) -> Result<Value> {
    if !looks_like_jesterky_manifest(manifest) {
        return Err(OptimizerError::Config(
            "jesterky manifest adapter received a value that does not match the RunManifest shape"
                .to_string(),
        ));
    }
    let trace = manifest
        .get("trace")
        .ok_or_else(|| OptimizerError::Config("jesterky manifest is missing trace".to_string()))?;
    let mut rows = Vec::new();
    collect_trace_rows(trace, 0, &mut rows)?;
    rows.sort_by(|left, right| row_addr(&left.value).cmp(row_addr(&right.value)));

    let trace_rows = rows.iter().map(|row| row.value.clone()).collect::<Vec<_>>();
    let optimizer_triples = rows
        .iter()
        .filter(|row| !row.has_children || row.has_score)
        .map(|row| optimizer_triple_row(&row.value))
        .collect::<Vec<_>>();
    let evidence_refs = evidence_refs(&trace_rows);
    let summary = manifest_summary(
        manifest,
        trace_rows.len(),
        &optimizer_triples,
        &evidence_refs,
    );

    Ok(json!({
        "schema_version": JESTERKY_WORKSPACE_READ_MODEL_SCHEMA_VERSION,
        "summary": summary,
        "trace_rows": {
            "schema_version": "synth_optimizers.jesterky_trace_rows.v1",
            "row_count": trace_rows.len(),
            "rows": trace_rows,
        },
        "optimizer_triples": {
            "schema_version": "synth_optimizers.jesterky_optimizer_triples.v1",
            "triple_count": optimizer_triples.len(),
            "contract": "inputs -> outputs -> score per leaf, plus scored interior nodes; scores are copied only from the jesterky manifest",
            "rows": optimizer_triples,
        },
        "evidence_refs": {
            "schema_version": "synth_optimizers.jesterky_evidence_refs.v1",
            "ref_count": evidence_refs.len(),
            "refs": evidence_refs,
        },
    }))
}

fn collect_trace_rows(node: &Value, depth: usize, rows: &mut Vec<TraceRow>) -> Result<()> {
    let object = node.as_object().ok_or_else(|| {
        OptimizerError::Config("jesterky trace node must be a JSON object".to_string())
    })?;
    // Annotate manifests use object Addr {run_id,node_path,iteration,local_seq};
    // older fixtures may still use a string addr. Normalize to a stable string.
    let addr = required_addr_string(object, "addr")?;
    let label = required_string(object, "label")?;
    let children = object
        .get("children")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let has_score = object.get("score").is_some_and(|value| !value.is_null());
    rows.push(TraceRow {
        value: json!({
            "addr": addr,
            "label": label,
            "depth": depth,
            "is_leaf": children.is_empty(),
            "inputs": object.get("inputs").cloned().unwrap_or(Value::Null),
            "outputs": object.get("outputs").cloned().unwrap_or(Value::Null),
            "score": object.get("score").cloned().unwrap_or(Value::Null),
            "signal": object.get("signal").cloned().unwrap_or(Value::Null),
            "artifacts": object.get("artifacts").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
            "child_count": children.len(),
        }),
        has_children: !children.is_empty(),
        has_score,
    });
    for child in children {
        collect_trace_rows(&child, depth + 1, rows)?;
    }
    Ok(())
}

fn optimizer_triple_row(row: &Value) -> Value {
    json!({
        "addr": row.get("addr").cloned().unwrap_or(Value::Null),
        "label": row.get("label").cloned().unwrap_or(Value::Null),
        "is_leaf": row.get("is_leaf").cloned().unwrap_or(Value::Bool(false)),
        "inputs": row.get("inputs").cloned().unwrap_or(Value::Null),
        "outputs": row.get("outputs").cloned().unwrap_or(Value::Null),
        "score": row.get("score").cloned().unwrap_or(Value::Null),
        "signal": row.get("signal").cloned().unwrap_or(Value::Null),
        "artifacts": row.get("artifacts").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
    })
}

fn evidence_refs(rows: &[Value]) -> Vec<Value> {
    let mut refs = Vec::new();
    for row in rows {
        let addr = row_addr(row).to_string();
        let label = row
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        refs.push(json!({
            "kind": "jesterky_process_addr",
            "addr": addr,
            "label": label,
        }));
        for artifact in row
            .get("artifacts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            refs.push(json!({
                "kind": "jesterky_artifact",
                "addr": row_addr(row),
                "label": row.get("label").cloned().unwrap_or(Value::Null),
                "artifact": artifact,
            }));
        }
    }
    refs
}

fn manifest_summary(
    manifest: &Value,
    trace_row_count: usize,
    optimizer_triples: &[Value],
    evidence_refs: &[Value],
) -> Value {
    let empty = Map::new();
    let object = manifest.as_object().unwrap_or(&empty);
    let scored_node_count = optimizer_triples
        .iter()
        .filter(|row| row.get("score").is_some_and(|value| !value.is_null()))
        .count();
    json!({
        "schema_version": "synth_optimizers.jesterky_manifest_summary.v1",
        "manifest_hash": stable_json_hash(manifest),
        "identity": {
            "run_id": string_field(object, "run_id").unwrap_or_default(),
            "workflow_name": string_field(object, "workflow_name").unwrap_or_default(),
            "spec_hash": string_field(object, "spec_hash").unwrap_or_default(),
        },
        "typed_outcome": {
            "status": string_field(object, "status").unwrap_or_default(),
            "stop_reason": string_field(object, "stop_reason").unwrap_or_default(),
        },
        "counts": {
            "events": manifest.get("events").and_then(Value::as_array).map_or(0, |items| items.len()),
            "recorded": manifest.get("recorded").and_then(Value::as_array).map_or(0, |items| items.len()),
            "trace_rows": trace_row_count,
            "optimizer_triples": optimizer_triples.len(),
            "scored_nodes": scored_node_count,
            "evidence_refs": evidence_refs.len(),
        },
        "budgets": manifest.get("budgets").cloned().unwrap_or(Value::Null),
        "goals": manifest.get("goals").cloned().unwrap_or(Value::Null),
        "invariants": manifest.get("invariants").cloned().unwrap_or(Value::Null),
        "contract_notes": [
            "Read stop_reason and typed fields; never parse failure strings.",
            "Use Addr for alignment and sorting.",
            "Grade outcomes only. This adapter never invents quality scores."
        ],
    })
}

fn row_addr(row: &Value) -> &str {
    row.get("addr").and_then(Value::as_str).unwrap_or_default()
}

fn required_string(object: &Map<String, Value>, key: &str) -> Result<String> {
    string_field(object, key).ok_or_else(|| {
        OptimizerError::Config(format!("jesterky trace node missing string field {key}"))
    })
}

fn required_addr_string(object: &Map<String, Value>, key: &str) -> Result<String> {
    let Some(value) = object.get(key) else {
        return Err(OptimizerError::Config(format!(
            "jesterky trace node missing field {key}"
        )));
    };
    addr_value_as_string(value).ok_or_else(|| {
        OptimizerError::Config(format!(
            "jesterky trace node field {key} must be a non-empty string or Addr object"
        ))
    })
}

fn addr_value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Object(addr) => {
            let run_id = addr
                .get("run_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let node_path = addr
                .get("node_path")
                .map(|path| match path {
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("/"),
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            let iteration = addr.get("iteration").and_then(Value::as_u64).unwrap_or(0);
            let local_seq = addr.get("local_seq").and_then(Value::as_u64).unwrap_or(0);
            let rendered = format!("{run_id}:{node_path}@{iteration}.{local_seq}");
            if rendered
                .trim_matches(|c| c == ':' || c == '@' || c == '.')
                .is_empty()
                && run_id.is_empty()
                && node_path.is_empty()
            {
                None
            } else {
                Some(rendered)
            }
        }
        _ => None,
    }
}

fn string_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
}
