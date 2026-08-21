//! Optional jesterky trace-annotate hook for GEPA proposer generations.
//!
//! When `jesterky_workflow.enabled` is true, export visible rollouts to
//! synth_rollout_trace_v4 files, run `jesterky`, and materialize wall-safe
//! `state/jesterky_*` artifacts into the proposer workspace before the live
//! proposer turn. When disabled, force absence of those files.

use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    JesterkyWorkflowConfig, OptimizerError, Result, SynthOptimizerConfig,
};

pub const JESTERKY_THEME_REGISTRY_FILE: &str = "jesterky_theme_registry.json";
pub const JESTERKY_TRACE_ANNOTATIONS_FILE: &str = "jesterky_trace_annotations.jsonl";
pub const JESTERKY_PROPOSER_CONTEXT_FILE: &str = "jesterky_proposer_context.md";
pub const JESTERKY_ANNOTATE_MANIFEST_FILE: &str = "jesterky_gepa_annotate.manifest.json";
pub const JESTERKY_RECEIPT_FILE: &str = "jesterky_workflow_receipt.json";
pub const JESTERKY_RECEIPTS_JSONL: &str = "jesterky_workflow_receipts.jsonl";

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
             state/jesterky_trace_annotations.jsonl. Use those themes and annotations as \
             wall-safe evidence. Cite theme names / trace_ids. Do not invent \
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
    let exported = export_gepa_rollouts_to_v4(rollouts, &trace_dir)?;
    if exported == 0 {
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
        });
    }

    let manifest_path = workspace_dir.join(JESTERKY_ANNOTATE_MANIFEST_FILE);
    let command = resolve_jesterky_command(wf);
    let spec = resolve_spec_path(wf)?;
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

    let (theme_count, annotated, blockers) =
        materialize_jesterky_artifacts_from_manifest(&manifest_path, &state_dir)?;
    if exported > 0 && (annotated == 0 || theme_count == 0) {
        return Err(OptimizerError::Config(format!(
            "jesterky workflow produced empty annotate signal after exporting {exported} \
             traces (theme_count={theme_count}, annotated={annotated}, blockers={blockers}, \
             manifest={}). Refusing to continue with hollow state/jesterky_* artifacts; fix \
             gepa_trace_annotate extraction/output or disable jesterky_workflow.",
            manifest_path.display()
        )));
    }

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
    })
}

fn resolve_jesterky_command(wf: &JesterkyWorkflowConfig) -> String {
    wf.command.trim().to_string()
}

/// The spec path is already absolute: `SynthOptimizerConfig::resolve_relative_paths`
/// absolutizes it against the TOML's own directory at load. Nothing here searches
/// a checkout, a developer home, or the process working directory.
fn resolve_spec_path(wf: &JesterkyWorkflowConfig) -> Result<PathBuf> {
    let raw = PathBuf::from(wf.spec.trim());
    if raw.is_file() {
        return Ok(raw);
    }
    Err(OptimizerError::Config(format!(
        "jesterky_workflow.spec not found: {} (paths resolve against the config TOML's \
         directory; give an absolute path or one relative to it)",
        raw.display()
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

/// Keep annotate latency bounded: gen0 typically has 4 train frames; later
/// generations accumulate many visible frames and Codex annotate can exceed
/// the workflow timeout. Prefer lowest-reward / failed frames first.
const MAX_JESTERKY_EXPORT_TRACES: usize = 6;

fn export_gepa_rollouts_to_v4(rollouts: &Value, trace_dir: &Path) -> Result<usize> {
    let rows = match rollouts.as_array() {
        Some(rows) => rows,
        None => return Ok(0),
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
    let selected = ranked
        .into_iter()
        .take(MAX_JESTERKY_EXPORT_TRACES)
        .collect::<Vec<_>>();
    let mut written = 0usize;
    for (idx, row) in selected {
        let candidate_id = row
            .get("candidate_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let task_id = row
            .get("task_id")
            .or_else(|| row.get("example_id"))
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        let status = row
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("completed");
        let rollout_id = format!("gepa-{candidate_id}-{task_id}-{idx:04}");
        let span_id = format!("{rollout_id}-span-0001");
        let summary = row.get("summary").cloned().unwrap_or(Value::Null);
        let outcome = row.get("outcome").cloned().unwrap_or(Value::Null);
        let failure = row.get("failure").cloned().unwrap_or(Value::Null);
        let mut metadata = Map::new();
        metadata.insert("source".to_string(), json!("gepa_jesterky_workflow_export"));
        metadata.insert("candidate_id".to_string(), json!(candidate_id));
        metadata.insert("task_id".to_string(), json!(task_id));
        if let Some(stage) = row.get("evaluation_stage") {
            metadata.insert("evaluation_stage".to_string(), stage.clone());
        }

        let trace = json!({
            "schema_version": "synth_rollout_trace_v4",
            "trace_schema_version": 4,
            "rollout_id": rollout_id,
            "trace_correlation_id": rollout_id,
            "status": status,
            "spans": [{
                "span_id": span_id,
                "call_index": 1,
                "run_id": rollout_id,
                "request": {
                    "messages": [
                        {"role": "system", "content": "GEPA Craftax search evidence"},
                        {"role": "user", "content": format!("task={task_id}; candidate={candidate_id}")}
                    ],
                    "provider_hint": "openai_compat"
                },
                "response": {
                    "message": {
                        "role": "assistant",
                        "content": serde_json::to_string(&json!({
                            "reward": reward,
                            "summary": summary,
                            "outcome": outcome,
                            "failure": failure,
                            "expected": row.get("expected"),
                            "prediction": row.get("prediction"),
                            "text": row.get("text"),
                            "rationale_text": row.get("rationale_text"),
                        }))?
                    },
                    "usage": row.get("usage").cloned().unwrap_or(json!({}))
                },
                "metrics": {"reward_total": reward},
                "metadata": {"candidate_id": candidate_id, "task_id": task_id}
            }],
            "events": [{
                "type": "lm_call",
                "sequence_index": 1,
                "span_id": span_id,
                "metadata": {"candidate_id": candidate_id}
            }],
            "span_count": 1,
            "summary": {
                "task_id": task_id,
                "outcome_reward": reward,
                "reward": reward,
                "candidate_id": candidate_id,
                "expected": row.get("expected"),
                "prediction": row.get("prediction"),
            },
            "metadata": metadata,
        });
        let safe_id = sanitize_filename(&rollout_id);
        let path = trace_dir.join(format!("{safe_id}.v4.json"));
        let text = serde_json::to_string_pretty(&trace)?;
        fs::write(&path, text).map_err(|source| OptimizerError::io(&path, source))?;
        written += 1;
    }
    Ok(written)
}

fn materialize_jesterky_artifacts_from_manifest(
    manifest_path: &Path,
    state_dir: &Path,
) -> Result<(usize, usize, usize)> {
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

    Ok((theme_count, annotated, blockers))
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
