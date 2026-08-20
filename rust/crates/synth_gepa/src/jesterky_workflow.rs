//! Optional jesterky trace-annotate hook for GEPA proposer generations.
//!
//! When `jesterky_workflow.enabled` is true, select new sealed Trace V5 inputs,
//! create explicitly lossy V4 transport projections for Jesterky, and attach
//! its descriptive output to immutable V5 evidence bundles before the next
//! proposer turn.  When disabled, force absence of those artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use synth_optimizer_platform::{
    build_jesterky_evidence_bundle, content_digest_for, jesterky_v4_projection,
    load_sealed_trace_v5, write_evidence_bundle_v5, JesterkyWorkflowConfig, OptimizerError, Result,
    SealedTraceV5, SynthOptimizerConfig,
};

pub const JESTERKY_THEME_REGISTRY_FILE: &str = "jesterky_theme_registry.json";
pub const JESTERKY_TRACE_ANNOTATIONS_FILE: &str = "jesterky_trace_annotations.jsonl";
pub const JESTERKY_PROPOSER_CONTEXT_FILE: &str = "jesterky_proposer_context.md";
pub const JESTERKY_ANNOTATE_MANIFEST_FILE: &str = "jesterky_gepa_annotate.manifest.json";
pub const JESTERKY_RECEIPT_FILE: &str = "jesterky_workflow_receipt.json";
pub const JESTERKY_RECEIPTS_JSONL: &str = "jesterky_workflow_receipts.jsonl";
pub const JESTERKY_EVIDENCE_BUNDLE_INDEX_FILE: &str = "jesterky_trace_evidence_bundles.v5.json";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JesterkyWorkflowReceipt {
    pub enabled: bool,
    pub run_id: String,
    pub generation: usize,
    pub manifest_path: String,
    pub theme_count: usize,
    pub annotated: usize,
    pub blockers: usize,
    pub command: String,
    pub actor: String,
    pub model: Option<String>,
    pub trace_dir: String,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub config_digest: String,
    #[serde(default)]
    pub spec_digest: String,
    #[serde(default)]
    pub trace_digests: Vec<String>,
    #[serde(default)]
    pub evidence_bundle_paths: Vec<String>,
    #[serde(default)]
    pub evidence_bundle_digests: Vec<String>,
}

#[derive(Clone, Debug)]
struct PreparedTrace {
    trace: SealedTraceV5,
    projection_digest: String,
    projection_trace_id: String,
}

/// Remove any prior jesterky artifacts so Arm A cannot accidentally pick them up.
pub fn clear_jesterky_workspace_artifacts(workspace_dir: &Path) -> Result<()> {
    let state_dir = workspace_dir.join("state");
    for name in [
        JESTERKY_THEME_REGISTRY_FILE,
        JESTERKY_TRACE_ANNOTATIONS_FILE,
        JESTERKY_PROPOSER_CONTEXT_FILE,
        "jesterky_manifest_summary.json",
        "jesterky_trace_rows.json",
        "jesterky_optimizer_triples.json",
        "jesterky_evidence_refs.json",
        "jesterky_read_model.json",
        JESTERKY_EVIDENCE_BUNDLE_INDEX_FILE,
    ] {
        let path = state_dir.join(name);
        if path.exists() {
            fs::remove_file(&path).map_err(|source| OptimizerError::io(&path, source))?;
        }
    }
    for name in [JESTERKY_ANNOTATE_MANIFEST_FILE, JESTERKY_RECEIPT_FILE] {
        let path = workspace_dir.join(name);
        if path.exists() {
            fs::remove_file(&path).map_err(|source| OptimizerError::io(&path, source))?;
        }
    }
    let traces = workspace_dir.join("jesterky_traces");
    if traces.exists() {
        fs::remove_dir_all(&traces).map_err(|source| OptimizerError::io(&traces, source))?;
    }
    let bundles = state_dir.join("trace_evidence_v5");
    if bundles.exists() {
        fs::remove_dir_all(&bundles).map_err(|source| OptimizerError::io(&bundles, source))?;
    }
    Ok(())
}

/// Export rollouts → run jesterky → materialize artifacts + receipt.
pub fn prepare_jesterky_workflow_for_generation(
    config: &SynthOptimizerConfig,
    rollouts: &Value,
    workspace_dir: &Path,
    generation: usize,
) -> Result<Option<JesterkyWorkflowReceipt>> {
    let wf = &config.jesterky_workflow;
    clear_jesterky_workspace_artifacts(workspace_dir)?;
    if !wf.enabled {
        return Ok(None);
    }

    let started = Instant::now();
    let result = run_enabled_jesterky_workflow(config, wf, rollouts, workspace_dir, generation);
    match result {
        Ok(mut receipt) => {
            receipt.elapsed_ms = started.elapsed().as_millis() as u64;
            write_receipt(workspace_dir, &receipt)?;
            append_run_receipt(workspace_dir, &receipt)?;
            Ok(Some(receipt))
        }
        Err(err) => {
            if wf.fail_closed {
                Err(err)
            } else {
                let receipt = JesterkyWorkflowReceipt {
                    enabled: true,
                    run_id: config.run.run_id.clone(),
                    generation,
                    manifest_path: String::new(),
                    theme_count: 0,
                    annotated: 0,
                    blockers: 0,
                    command: resolve_jesterky_command(wf),
                    actor: wf.actor.clone(),
                    model: wf.model.clone(),
                    trace_dir: String::new(),
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    status: "failed_open".to_string(),
                    reason: Some(err.to_string()),
                    provider: wf.provider.clone(),
                    config_digest: workflow_config_digest(wf),
                    spec_digest: String::new(),
                    trace_digests: Vec::new(),
                    evidence_bundle_paths: Vec::new(),
                    evidence_bundle_digests: Vec::new(),
                };
                write_receipt(workspace_dir, &receipt)?;
                append_run_receipt(workspace_dir, &receipt)?;
                Ok(Some(receipt))
            }
        }
    }
}

pub fn jesterky_workspace_rule(enabled: bool) -> Option<&'static str> {
    if enabled {
        Some(
            "jesterky_workflow.enabled=true for this run: BEFORE proposing, read \
             state/jesterky_proposer_context.md, state/jesterky_theme_registry.json, and \
             state/jesterky_trace_annotations.jsonl, and \
             state/jesterky_trace_evidence_bundles.v5.json. Use only applied V5 bundle \
             annotations as wall-safe evidence. Cite theme names / trace_ids. Do not invent \
             evaluation-split labels or selection scores.",
        )
    } else {
        None
    }
}

fn run_enabled_jesterky_workflow(
    config: &SynthOptimizerConfig,
    wf: &JesterkyWorkflowConfig,
    rollouts: &Value,
    workspace_dir: &Path,
    generation: usize,
) -> Result<JesterkyWorkflowReceipt> {
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    let trace_dir = workspace_dir.join("jesterky_traces");
    fs::create_dir_all(&trace_dir).map_err(|source| OptimizerError::io(&trace_dir, source))?;
    let config_digest = workflow_config_digest(wf);
    let seen = annotated_trace_digests(workspace_dir);
    let prepared = export_sealed_v5_rollouts(rollouts, &trace_dir, wf.max_targets, &seen)?;
    if prepared.is_empty() {
        let empty_registry = json!({
            "optimizer": "gepa",
            "themes": [],
            "traces": [],
            "headline": format!(
                "jesterky workflow enabled but no rollout evidence yet for generation {generation}"
            ),
        });
        write_theme_artifacts(&state_dir, &empty_registry, generation)?;
        return Ok(JesterkyWorkflowReceipt {
            enabled: true,
            run_id: config.run.run_id.clone(),
            generation,
            manifest_path: String::new(),
            theme_count: 0,
            annotated: 0,
            blockers: 0,
            command: resolve_jesterky_command(wf),
            actor: wf.actor.clone(),
            model: wf.model.clone(),
            trace_dir: trace_dir.display().to_string(),
            elapsed_ms: 0,
            status: "skipped".to_string(),
            reason: Some(
                "no new sealed Trace V5 evidence matched the annotation policy".to_string(),
            ),
            provider: wf.provider.clone(),
            config_digest,
            spec_digest: String::new(),
            trace_digests: Vec::new(),
            evidence_bundle_paths: Vec::new(),
            evidence_bundle_digests: Vec::new(),
        });
    }

    let manifest_path = workspace_dir.join(JESTERKY_ANNOTATE_MANIFEST_FILE);
    let command = resolve_jesterky_command(wf);
    let spec = resolve_spec_path(wf)?;
    let spec_digest = sha256_file(&spec)?;
    let args_json = json!({
        "trace_dir": trace_dir.display().to_string(),
        "artifact_dir": state_dir.display().to_string(),
    })
    .to_string();

    let mut cmd = Command::new(&command);
    // Manifest is written to --out; keep stdout null so verbose annotate trees
    // cannot fill a pipe and deadlock the timeout poller. Capture stderr only.
    cmd.arg("run")
        .arg(&spec)
        .arg("--actor")
        .arg(&wf.actor)
        .arg("--args")
        .arg(&args_json)
        .arg("--out")
        .arg(&manifest_path)
        // The generated V4 transport projection is an input to the Codex actor,
        // so make its run-local workspace the actor's readable sandbox root.
        .arg("--cd")
        .arg(workspace_dir)
        .arg("--run-id")
        .arg(format!(
            "gepa-{}-g{:03}-jesterky",
            config.run.run_id, generation
        ))
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(model) = wf.model.as_ref().filter(|m| !m.trim().is_empty()) {
        cmd.arg("--model").arg(model);
    }

    let output = run_command_with_timeout(cmd, Duration::from_secs(wf.timeout_seconds), &command)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(OptimizerError::Config(format!(
            "jesterky workflow failed (exit {:?}): stderr={}",
            output.status.code(),
            truncate_for_error(&stderr)
        )));
    }
    if !manifest_path.is_file() {
        return Err(OptimizerError::Config(format!(
            "jesterky workflow completed but missing manifest at {}",
            manifest_path.display()
        )));
    }

    let (theme_count, annotated, blockers, theme_registry) =
        materialize_jesterky_artifacts_from_manifest(&manifest_path, &state_dir)?;
    if !prepared.is_empty() && (annotated == 0 || theme_count == 0) {
        return Err(OptimizerError::Config(format!(
            "jesterky workflow produced empty annotate signal after exporting {} \
             traces (theme_count={theme_count}, annotated={annotated}, blockers={blockers}, \
             manifest={}). Refusing to continue with hollow state/jesterky_* artifacts; fix \
             gepa_trace_annotate extraction/output or disable jesterky_workflow.",
            prepared.len(),
            manifest_path.display()
        )));
    }
    let manifest_digest = sha256_file(&manifest_path)?;
    let (bundle_paths, bundle_digests) = materialize_v5_evidence_bundles(
        &prepared,
        &theme_registry,
        wf,
        &config_digest,
        &spec_digest,
        &manifest_digest,
        &state_dir,
    )?;

    Ok(JesterkyWorkflowReceipt {
        enabled: true,
        run_id: config.run.run_id.clone(),
        generation,
        manifest_path: manifest_path.display().to_string(),
        theme_count,
        annotated,
        blockers,
        command,
        actor: wf.actor.clone(),
        model: wf.model.clone(),
        trace_dir: trace_dir.display().to_string(),
        elapsed_ms: 0,
        status: "completed".to_string(),
        reason: None,
        provider: wf.provider.clone(),
        config_digest,
        spec_digest,
        trace_digests: prepared
            .into_iter()
            .map(|item| item.trace.content_digest)
            .collect(),
        evidence_bundle_paths: bundle_paths,
        evidence_bundle_digests: bundle_digests,
    })
}

fn resolve_jesterky_command(wf: &JesterkyWorkflowConfig) -> String {
    if let Ok(env_cmd) = std::env::var("STACK_JESTERKY_COMMAND") {
        let trimmed = env_cmd.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    wf.command.trim().to_string()
}

fn resolve_spec_path(wf: &JesterkyWorkflowConfig) -> Result<PathBuf> {
    let raw = PathBuf::from(wf.spec.trim());
    if raw.is_file() {
        return Ok(raw);
    }
    let candidates = [
        PathBuf::from("/Users/joshpurtell/Documents/GitHub/jesterky").join(&raw),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../jesterky")
            .join(&raw),
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(&raw),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(OptimizerError::Config(format!(
        "jesterky_workflow.spec not found: {}",
        wf.spec
    )))
}

fn run_command_with_timeout(
    mut cmd: Command,
    timeout: Duration,
    command_label: &str,
) -> Result<std::process::Output> {
    use std::io::Read;

    let mut child = cmd.spawn().map_err(|source| {
        OptimizerError::Config(format!("failed to spawn {command_label}: {source}"))
    })?;
    let stderr_handle = child.stderr.take();
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut reader) = stderr_handle {
            let _ = reader.read_to_end(&mut buf);
        }
        buf
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stderr_thread.join();
                    return Err(OptimizerError::Config(format!(
                        "{command_label} exceeded timeout of {}s",
                        timeout.as_secs()
                    )));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(source) => {
                let _ = stderr_thread.join();
                return Err(OptimizerError::Config(format!(
                    "failed waiting on {command_label}: {source}"
                )));
            }
        }
    };
    let stderr = stderr_thread.join().unwrap_or_default();
    Ok(std::process::Output {
        status,
        stdout: Vec::new(),
        stderr,
    })
}

fn export_sealed_v5_rollouts(
    rollouts: &Value,
    trace_dir: &Path,
    max_targets: usize,
    seen_digests: &BTreeSet<String>,
) -> Result<Vec<PreparedTrace>> {
    let rows = match rollouts.as_array() {
        Some(rows) => rows,
        None => return Ok(Vec::new()),
    };
    let mut ranked: Vec<(usize, &Value)> = rows.iter().enumerate().collect();
    ranked.sort_by(|(left_idx, left), (right_idx, right)| {
        let left_reward = left.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        let right_reward = right.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        let left_failed = left
            .get("success_status")
            .and_then(Value::as_str)
            .map(|s| s != "succeeded")
            .unwrap_or(false);
        let right_failed = right
            .get("success_status")
            .and_then(Value::as_str)
            .map(|s| s != "succeeded")
            .unwrap_or(false);
        right_failed
            .cmp(&left_failed)
            .then_with(|| {
                left_reward
                    .partial_cmp(&right_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right_idx.cmp(left_idx))
    });
    let selected = ranked.into_iter().collect::<Vec<_>>();
    let mut prepared = Vec::new();
    let mut selected_digests = BTreeSet::new();
    for (idx, row) in selected {
        if prepared.len() >= max_targets {
            break;
        }
        let Some(path) = trace_v5_path(row) else {
            continue;
        };
        let trace = load_sealed_trace_v5(&path)?;
        if seen_digests.contains(&trace.content_digest)
            || !selected_digests.insert(trace.content_digest.clone())
        {
            continue;
        }
        let projection = jesterky_v4_projection(&trace, row);
        let projection_digest = content_digest_for(&projection);
        let projection_trace_id = sanitize_filename(&format!("{}-{idx:04}", trace.trace_id));
        let path = trace_dir.join(format!("{projection_trace_id}.v4.json"));
        let text = serde_json::to_string_pretty(&projection)?;
        fs::write(&path, text).map_err(|source| OptimizerError::io(&path, source))?;
        prepared.push(PreparedTrace {
            trace,
            projection_digest,
            projection_trace_id,
        });
    }
    Ok(prepared)
}

fn trace_v5_path(row: &Value) -> Option<PathBuf> {
    row.get("trace_v5_path")
        .and_then(Value::as_str)
        .or_else(|| row.pointer("/trace_v5/path").and_then(Value::as_str))
        .or_else(|| {
            row.pointer("/trace_v5/document_path")
                .and_then(Value::as_str)
        })
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
}

fn annotated_trace_digests(workspace_dir: &Path) -> BTreeSet<String> {
    let Some(run_dir) = workspace_dir.parent().and_then(|path| path.parent()) else {
        return BTreeSet::new();
    };
    let path = run_dir.join(JESTERKY_RECEIPTS_JSONL);
    let Ok(text) = fs::read_to_string(path) else {
        return BTreeSet::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|receipt| receipt.get("status").and_then(Value::as_str) == Some("completed"))
        .flat_map(|receipt| {
            receipt
                .get("trace_digests")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|digest| digest.as_str().map(str::to_string))
        .collect()
}

fn workflow_config_digest(config: &JesterkyWorkflowConfig) -> String {
    content_digest_for(&serde_json::to_value(config).unwrap_or(Value::Null))
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(path).map_err(|source| OptimizerError::io(path, source))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn materialize_v5_evidence_bundles(
    prepared: &[PreparedTrace],
    theme_registry: &Value,
    config: &JesterkyWorkflowConfig,
    config_digest: &str,
    spec_digest: &str,
    manifest_digest: &str,
    state_dir: &Path,
) -> Result<(Vec<String>, Vec<String>)> {
    let scans = theme_registry
        .get("traces")
        .or_else(|| theme_registry.get("theme_matrix"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let scan_by_trace = scans
        .into_iter()
        .filter_map(|scan| {
            let trace_id = scan.get("trace_id").and_then(Value::as_str)?.to_string();
            Some((trace_id, scan))
        })
        .collect::<BTreeMap<_, _>>();
    let bundle_dir = state_dir.join("trace_evidence_v5");
    fs::create_dir_all(&bundle_dir).map_err(|source| OptimizerError::io(&bundle_dir, source))?;
    let now = rfc3339_now()?;
    let mut paths = Vec::new();
    let mut digests = Vec::new();
    let mut index = Vec::new();
    for item in prepared {
        let scan = scan_by_trace.get(&item.projection_trace_id);
        let bundle = build_jesterky_evidence_bundle(
            &item.trace,
            scan,
            config,
            config_digest,
            spec_digest,
            &item.projection_digest,
            &now,
            &now,
            manifest_digest,
            if scan.is_some() {
                "completed"
            } else {
                "abstained"
            },
            scan.is_none()
                .then_some("Jesterky manifest omitted this projected trace"),
        );
        let stored = write_evidence_bundle_v5(&bundle_dir, &bundle)?;
        paths.push(stored.path.display().to_string());
        digests.push(stored.content_digest.clone());
        index.push(json!({
            "trace_id": item.trace.trace_id,
            "trace_digest": item.trace.content_digest,
            "bundle_path": stored.path,
            "bundle_digest": stored.content_digest,
        }));
    }
    let index_path = state_dir.join(JESTERKY_EVIDENCE_BUNDLE_INDEX_FILE);
    fs::write(
        &index_path,
        serde_json::to_string_pretty(&json!({
            "schema_version": "synth_gepa.jesterky_evidence_bundle_index.v1",
            "bundles": index,
        }))?,
    )
    .map_err(|source| OptimizerError::io(&index_path, source))?;
    Ok((paths, digests))
}

fn rfc3339_now() -> Result<String> {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|error| {
            OptimizerError::Config(format!("format annotation receipt timestamp: {error}"))
        })
}

fn materialize_jesterky_artifacts_from_manifest(
    manifest_path: &Path,
    state_dir: &Path,
) -> Result<(usize, usize, usize, Value)> {
    let text = fs::read_to_string(manifest_path)
        .map_err(|source| OptimizerError::io(manifest_path, source))?;
    let manifest: Value = serde_json::from_str(&text).map_err(|source| {
        OptimizerError::Config(format!(
            "invalid jesterky annotate manifest {}: {source}",
            manifest_path.display()
        ))
    })?;

    let theme_registry = extract_theme_registry(&manifest);
    let theme_count = theme_registry
        .get("themes")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let traces = theme_registry
        .get("traces")
        .or_else(|| theme_registry.get("theme_matrix"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let annotated = traces.len();
    let blockers = traces
        .iter()
        .filter(|row| row.get("blocker").and_then(Value::as_bool).unwrap_or(false))
        .count();

    write_theme_artifacts(state_dir, &theme_registry, 0)?;
    // Overwrite context with richer counts from this materialization.
    let headline = theme_registry
        .get("headline")
        .and_then(Value::as_str)
        .unwrap_or("jesterky GEPA trace annotate completed");
    let mut context = String::new();
    context.push_str("# jesterky GEPA proposer context\n\n");
    context.push_str(headline);
    context.push_str("\n\n");
    context.push_str(&format!(
        "- annotated traces: {annotated}\n- blockers: {blockers}\n- themes: {theme_count}\n\n"
    ));
    if let Some(themes) = theme_registry.get("themes").and_then(Value::as_array) {
        context.push_str("## Themes\n");
        for theme in themes {
            let name = theme
                .get("theme")
                .or_else(|| theme.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let count = theme.get("count").and_then(Value::as_u64).unwrap_or(0);
            context.push_str(&format!("- {name} (count={count})\n"));
        }
        context.push('\n');
    }
    context.push_str(
        "Use these themes and annotations as wall-safe evidence when proposing \
         prompt candidates. Cite theme names and trace_ids; do not invent \
         evaluation-split labels or selection scores.\n",
    );
    let context_path = state_dir.join(JESTERKY_PROPOSER_CONTEXT_FILE);
    fs::write(&context_path, context)
        .map_err(|source| OptimizerError::io(&context_path, source))?;

    let annotations_path = state_dir.join(JESTERKY_TRACE_ANNOTATIONS_FILE);
    let mut lines = String::new();
    for row in &traces {
        lines.push_str(&serde_json::to_string(row)?);
        lines.push('\n');
    }
    fs::write(&annotations_path, lines)
        .map_err(|source| OptimizerError::io(&annotations_path, source))?;

    Ok((theme_count, annotated, blockers, theme_registry))
}

fn write_theme_artifacts(
    state_dir: &Path,
    theme_registry: &Value,
    generation: usize,
) -> Result<()> {
    let registry_path = state_dir.join(JESTERKY_THEME_REGISTRY_FILE);
    fs::write(
        &registry_path,
        serde_json::to_string_pretty(theme_registry)?,
    )
    .map_err(|source| OptimizerError::io(&registry_path, source))?;
    let annotations_path = state_dir.join(JESTERKY_TRACE_ANNOTATIONS_FILE);
    if !annotations_path.exists() {
        fs::write(&annotations_path, "")
            .map_err(|source| OptimizerError::io(&annotations_path, source))?;
    }
    let context_path = state_dir.join(JESTERKY_PROPOSER_CONTEXT_FILE);
    if !context_path.exists() {
        fs::write(
            &context_path,
            format!(
                "# jesterky GEPA proposer context\n\nNo search evidence exported for generation {generation} yet.\n"
            ),
        )
        .map_err(|source| OptimizerError::io(&context_path, source))?;
    }
    Ok(())
}

fn extract_theme_registry(manifest: &Value) -> Value {
    const POINTERS: &[&str] = &[
        "/trace/outputs/theme_registry",
        "/trace/outputs/summary/theme_registry",
        "/theme_registry",
    ];
    for pointer in POINTERS {
        if let Some(registry) = manifest.pointer(pointer) {
            if theme_registry_has_signal(registry) {
                return registry.clone();
            }
        }
    }
    if let Some(registry) = manifest.get("theme_registry") {
        if theme_registry_has_signal(registry) {
            return registry.clone();
        }
    }
    if let Some(children) = manifest
        .pointer("/trace/children")
        .and_then(Value::as_array)
    {
        for child in children.iter().rev() {
            if let Some(registry) = child
                .pointer("/outputs/theme_registry")
                .or_else(|| child.pointer("/outputs/summary/theme_registry"))
            {
                if theme_registry_has_signal(registry) {
                    return registry.clone();
                }
            }
        }
    }
    if let Some(recorded) = manifest.get("recorded").and_then(Value::as_array) {
        for item in recorded.iter().rev() {
            if let Some(registry) = item
                .pointer("/outputs/theme_registry")
                .or_else(|| item.pointer("/payload/theme_registry"))
                .or_else(|| item.pointer("/outputs/summary/theme_registry"))
            {
                if theme_registry_has_signal(registry) {
                    return registry.clone();
                }
            }
        }
    }
    if let Some(events) = manifest.get("events").and_then(Value::as_array) {
        for event in events.iter().rev() {
            if let Some(registry) = event
                .pointer("/payload/theme_registry")
                .or_else(|| event.pointer("/payload/summary/theme_registry"))
            {
                if theme_registry_has_signal(registry) {
                    return registry.clone();
                }
            }
        }
    }
    json!({
        "optimizer": "gepa",
        "themes": [],
        "traces": [],
        "headline": "jesterky annotate manifest lacked theme_registry; empty registry materialized",
    })
}

fn theme_registry_has_signal(registry: &Value) -> bool {
    let themes = registry
        .get("themes")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let traces = registry
        .get("traces")
        .or_else(|| registry.get("theme_matrix"))
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    themes > 0 || traces > 0
}

fn write_receipt(workspace_dir: &Path, receipt: &JesterkyWorkflowReceipt) -> Result<()> {
    let path = workspace_dir.join(JESTERKY_RECEIPT_FILE);
    fs::write(&path, serde_json::to_string_pretty(receipt)?)
        .map_err(|source| OptimizerError::io(&path, source))?;
    Ok(())
}

fn append_run_receipt(workspace_dir: &Path, receipt: &JesterkyWorkflowReceipt) -> Result<()> {
    // workspace: <run_dir>/proposer_workspaces/generation_XXX
    let Some(run_dir) = workspace_dir.parent().and_then(|p| p.parent()) else {
        return Ok(());
    };
    let path = run_dir.join(JESTERKY_RECEIPTS_JSONL);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| OptimizerError::io(&path, source))?;
    writeln!(file, "{}", serde_json::to_string(receipt)?)
        .map_err(|source| OptimizerError::io(&path, source))?;
    Ok(())
}

fn sanitize_filename(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "trace".to_string()
    } else {
        out
    }
}

fn truncate_for_error(text: &str) -> String {
    const MAX: usize = 2000;
    if text.len() <= MAX {
        text.to_string()
    } else {
        format!("{}…", &text[..MAX])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "synth-gepa-jesterky-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn selects_only_new_sealed_v5_traces_and_writes_a_transport_projection() {
        let dir = temp_path("projection");
        fs::create_dir_all(&dir).unwrap();
        let trace_path = dir.join("source.v5.json");
        let mut trace = json!({
            "schema_version": "synth.trace.v5",
            "trace_id": "trace_a",
            "events": []
        });
        trace["content_digest"] = json!(content_digest_for(&trace));
        fs::write(&trace_path, serde_json::to_string(&trace).unwrap()).unwrap();
        let rows = json!([
            {"trace_v5_path": trace_path, "candidate_id": "a", "reward": 0.0},
            {"trace_v5_path": trace_path, "candidate_id": "b", "reward": 1.0}
        ]);
        let trace_dir = dir.join("transport");
        fs::create_dir_all(&trace_dir).unwrap();
        let prepared = export_sealed_v5_rollouts(&rows, &trace_dir, 6, &BTreeSet::new()).unwrap();
        assert_eq!(
            prepared.len(),
            1,
            "duplicate trace digests must be deduplicated"
        );
        let projected = fs::read_to_string(trace_dir.join("trace_a-0001.v4.json"))
            .or_else(|_| fs::read_to_string(trace_dir.join("trace_a-0000.v4.json")))
            .unwrap();
        let projection: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(
            projection.pointer("/metadata/source_trace_ref/content_digest"),
            trace.get("content_digest")
        );
        let seen = BTreeSet::from([trace["content_digest"].as_str().unwrap().to_string()]);
        assert!(export_sealed_v5_rollouts(&rows, &trace_dir, 6, &seen)
            .unwrap()
            .is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "requires JESTERKY_LOCAL_BIN and JESTERKY_GEPA_SPEC"]
    fn local_fake_actor_fails_closed_on_hollow_annotations() {
        let command = std::env::var("JESTERKY_LOCAL_BIN")
            .expect("set JESTERKY_LOCAL_BIN to the local jesterky executable");
        let spec = std::env::var("JESTERKY_GEPA_SPEC")
            .expect("set JESTERKY_GEPA_SPEC to examples/gepa_trace_annotate.json");
        let dir = temp_path("fake-actor");
        let workspace = dir.join("proposer_workspaces/generation_001");
        fs::create_dir_all(&workspace).unwrap();
        let trace_path = dir.join("source.v5.json");
        let trace = synth_optimizer_platform::seal_gepa_rollout_trace_v5(
            "run_fake_actor",
            "sensor_fake_actor",
            &json!({
                "schema_version": "synth_rollout_trace_v4",
                "rollout_id": "rollout_fake_actor",
                "trace_correlation_id": "correlation_fake_actor",
                "status": "completed",
                "summary": {"expected":"alpha", "prediction":"beta"},
                "event_history": []
            }),
            &json!({"task_id":"task_fake_actor"}),
        )
        .unwrap();
        fs::write(&trace_path, serde_json::to_string_pretty(&trace).unwrap()).unwrap();
        let mut config = SynthOptimizerConfig::default();
        config.run.run_id = "run_fake_actor".to_string();
        config.jesterky_workflow.enabled = true;
        config.jesterky_workflow.command = command;
        config.jesterky_workflow.spec = spec;
        config.jesterky_workflow.actor = "fake".to_string();
        config.jesterky_workflow.model = Some("fake-pinned-model".to_string());
        config.jesterky_workflow.provider = "local-fake".to_string();
        config.jesterky_workflow.max_spend_usd = Some(1.0);
        let error = prepare_jesterky_workflow_for_generation(
            &config,
            &json!([{
                "trace_v5_path": trace_path,
                "candidate_id": "candidate_fake_actor",
                "task_id": "task_fake_actor",
                "reward": 0.0,
                "status": "completed"
            }]),
            &workspace,
            1,
        )
        .expect_err("the deterministic fake actor intentionally produces no themes");
        assert!(error.to_string().contains("produced empty annotate signal"));
        assert!(workspace.join(JESTERKY_ANNOTATE_MANIFEST_FILE).is_file());
        assert!(!workspace.join("state/trace_evidence_v5").exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "requires a real Craftax V4 rollout and ChatGPT-authenticated Jesterky"]
    fn local_codex_actor_materializes_v5_evidence_for_craftax() {
        let command = std::env::var("JESTERKY_LOCAL_BIN")
            .expect("set JESTERKY_LOCAL_BIN to the local jesterky executable");
        let spec = std::env::var("JESTERKY_GEPA_SPEC")
            .expect("set JESTERKY_GEPA_SPEC to examples/gepa_trace_annotate.json");
        let source_v4 = PathBuf::from(
            std::env::var("CRAFTAX_TRACE_V4")
                .expect("set CRAFTAX_TRACE_V4 to a completed Craftax V4 rollout"),
        );
        let dir = PathBuf::from(
            std::env::var("CRAFTAX_V5_ACCEPTANCE_DIR")
                .expect("set CRAFTAX_V5_ACCEPTANCE_DIR to retain the acceptance evidence"),
        );
        fs::create_dir_all(&dir).unwrap();
        let workspace = dir.join("proposer_workspaces/generation_001");
        fs::create_dir_all(&workspace).unwrap();
        let source: Value = serde_json::from_str(&fs::read_to_string(&source_v4).unwrap()).unwrap();
        let trace = synth_optimizer_platform::seal_gepa_rollout_trace_v5(
            "craftax_jesterky_v5_acceptance",
            "craftax_sensor_frame_001",
            &source,
            &source["metadata"],
        )
        .unwrap();
        let trace_path = dir.join("craftax.source.v5.json");
        fs::write(&trace_path, serde_json::to_string_pretty(&trace).unwrap()).unwrap();
        let mut config = SynthOptimizerConfig::default();
        config.run.run_id = "craftax_jesterky_v5_acceptance".to_string();
        config.jesterky_workflow.enabled = true;
        config.jesterky_workflow.command = command;
        config.jesterky_workflow.spec = spec;
        config.jesterky_workflow.actor = "codex".to_string();
        config.jesterky_workflow.model = Some("gpt-5.5".to_string());
        config.jesterky_workflow.provider = "chatgpt".to_string();
        config.jesterky_workflow.max_targets = 1;
        config.jesterky_workflow.max_spend_usd = Some(1.0);
        let receipt = prepare_jesterky_workflow_for_generation(
            &config,
            &json!([{
                "trace_v5_path": trace_path,
                "candidate_id": source.pointer("/summary/candidate_id"),
                "task_id": source.pointer("/summary/task_id"),
                "reward": source.pointer("/summary/reward"),
                "status": source.get("status")
            }]),
            &workspace,
            1,
        )
        .unwrap()
        .expect("enabled workflow returns a receipt");
        assert_eq!(receipt.status, "completed");
        assert_eq!(
            receipt.trace_digests,
            vec![trace["content_digest"].as_str().unwrap().to_string()]
        );
        assert_eq!(receipt.evidence_bundle_paths.len(), 1);
        assert!(Path::new(&receipt.manifest_path).is_file());
        let bundle: Value = serde_json::from_str(
            &fs::read_to_string(&receipt.evidence_bundle_paths[0]).unwrap(),
        )
        .unwrap();
        assert_eq!(bundle["schema_version"], "synth.trace-evidence-bundle.v5");
        assert_eq!(bundle["trace_ref"]["content_digest"], trace["content_digest"]);
        assert_eq!(bundle["reward_records"], json!([]));
        assert_eq!(bundle["benchmark_verdicts"], json!([]));
        assert!(workspace
            .join("state")
            .join(JESTERKY_EVIDENCE_BUNDLE_INDEX_FILE)
            .is_file());
    }
}
