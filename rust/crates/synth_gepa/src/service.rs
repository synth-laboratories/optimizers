use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    ArtifactPaths, CacheMode, CheckpointInput, CheckpointRecord, FailurePayload, OptimizerError,
    OptimizerJob, OptimizerJobKind, OptimizerJobStatus, Result, RuntimeEffectInput,
    RuntimeEffectRecord, WorkspaceRunRequestStatus, WorkspaceStore,
};

use crate::{
    advance_gepa_config_once, execute_gepa_from_toml_with_options,
    planner::{
        GepaCursor, GepaCursorPhase, GepaTickAction, GepaTickOutcome, GEPA_CURSOR_CHECKPOINT_KIND,
    },
    GepaAdvanceMode, GepaAdvanceOutcome, GepaCancellationSource, GepaExecutionOptions,
    GepaRunResult,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceConfig {
    pub db_path: PathBuf,
    pub bind_addr: String,
    pub worker_id: String,
    pub lease_seconds: u64,
}

impl GepaServiceConfig {
    pub fn new(db_path: impl Into<PathBuf>, bind_addr: impl Into<String>) -> Self {
        Self {
            db_path: db_path.into(),
            bind_addr: bind_addr.into(),
            worker_id: "synth-gepa-service".to_string(),
            lease_seconds: 3600,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceSubmitRequest {
    pub config_path: String,
    #[serde(default)]
    pub priority: i64,
    #[serde(default = "default_auto_start")]
    pub auto_start: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceSubmitResponse {
    pub request: WorkspaceRunRequestStatus,
    pub auto_started: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceWorkerOutcome {
    pub request: Option<WorkspaceRunRequestStatus>,
    pub result: Option<GepaRunResult>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaRecoveredOptimizerJob {
    pub job_id: String,
    pub status: String,
    pub attempt: u32,
    pub lease_id: Option<String>,
    pub next_retry_at: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaRunWorkspaceRecovery {
    pub run_id: String,
    pub workspace_db_path: String,
    pub recovered_optimizer_jobs: Vec<GepaRecoveredOptimizerJob>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceRecoveryOutcome {
    pub recovered_run_requests: Vec<WorkspaceRunRequestStatus>,
    pub recovered_run_workspaces: Vec<GepaRunWorkspaceRecovery>,
}

pub fn run_gepa_service(config: GepaServiceConfig) -> Result<()> {
    recover_service_state(&config.db_path)?;
    let listener = TcpListener::bind(&config.bind_addr).map_err(|source| {
        OptimizerError::io(format!("gepa service bind {}", config.bind_addr), source)
    })?;
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                thread::spawn(move || {
                    let _ = handle_connection(stream, config);
                });
            }
            Err(source) => {
                return Err(OptimizerError::io("gepa service accept", source));
            }
        }
    }
    Ok(())
}

pub fn run_next_queued_request(
    db_path: impl AsRef<Path>,
    worker_id: &str,
    lease_seconds: u64,
) -> Result<GepaServiceWorkerOutcome> {
    loop {
        let tick = tick_next_unit(&db_path, worker_id, lease_seconds)?;
        if tick.terminal || tick.request.is_none() {
            return Ok(GepaServiceWorkerOutcome {
                request: tick.request,
                result: tick.result,
                message: tick.message,
            });
        }
    }
}

pub fn tick_next_unit(
    db_path: impl AsRef<Path>,
    worker_id: &str,
    lease_seconds: u64,
) -> Result<GepaTickOutcome> {
    recover_service_state(&db_path)?;
    let lease_id = format!("lease_{}_{}", worker_id, now_millis());
    let store = WorkspaceStore::open_existing(&db_path)?;
    if let Some(active) = active_run_request(&store, worker_id)? {
        return tick_active_run_request(&store, db_path.as_ref(), active, worker_id, lease_seconds);
    }
    if let Some(cancelled) = cancelled_run_request_needing_workspace_terminalization(&store)? {
        return tick_cancelled_run_request_workspace(db_path.as_ref(), cancelled, lease_seconds);
    }
    let Some(claimed) = store.claim_next_run_request(&lease_id, Some(worker_id), lease_seconds)?
    else {
        return Ok(GepaTickOutcome {
            request: None,
            result: None,
            action: GepaTickAction::Noop,
            terminal: false,
            message: "no queued run requests".to_string(),
        });
    };
    Ok(GepaTickOutcome {
        request: Some(claimed.clone()),
        result: None,
        action: GepaTickAction::ClaimRunRequest {
            request_id: claimed.request_id.clone(),
            run_id: claimed.run_id.clone(),
        },
        terminal: false,
        message: "run request claimed".to_string(),
    })
}

fn active_run_request(
    store: &WorkspaceStore,
    worker_id: &str,
) -> Result<Option<WorkspaceRunRequestStatus>> {
    let status = store.status()?;
    let mut active = status
        .run_requests
        .into_iter()
        .filter(|request| request.status == "leased" || request.status == "running")
        .collect::<Vec<_>>();
    active.sort_by(|left, right| {
        left.submitted_at
            .cmp(&right.submitted_at)
            .then_with(|| left.request_id.cmp(&right.request_id))
    });
    if let Some(request) = active
        .iter()
        .find(|request| request.worker_id.as_deref() == Some(worker_id))
        .cloned()
    {
        return Ok(Some(request));
    }
    Ok(None)
}

fn cancelled_run_request_needing_workspace_terminalization(
    store: &WorkspaceStore,
) -> Result<Option<WorkspaceRunRequestStatus>> {
    let status = store.status()?;
    let mut cancelled = status
        .run_requests
        .into_iter()
        .filter(|request| request.status == "cancelled")
        .collect::<Vec<_>>();
    cancelled.sort_by(|left, right| {
        left.submitted_at
            .cmp(&right.submitted_at)
            .then_with(|| left.request_id.cmp(&right.request_id))
    });
    for request in cancelled {
        let workspace_db_path = request
            .run_workspace_db_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(&request.run_dir).join("workspace.sqlite"));
        if !workspace_db_path.exists() {
            continue;
        }
        let run_store = WorkspaceStore::open_existing(&workspace_db_path)?;
        let cursor = load_gepa_cursor(&run_store, &request.run_id)?;
        if cursor
            .as_ref()
            .is_none_or(|cursor| !cursor.phase.is_terminal())
        {
            return Ok(Some(request));
        }
    }
    Ok(None)
}

fn tick_cancelled_run_request_workspace(
    service_db_path: &Path,
    request: WorkspaceRunRequestStatus,
    lease_seconds: u64,
) -> Result<GepaTickOutcome> {
    let service_store = WorkspaceStore::open_existing(service_db_path)?;
    let config = service_store.run_request_config(&request.request_id)?;
    let advance = advance_gepa_config_once(
        config,
        GepaExecutionOptions {
            cancellation: Some(GepaCancellationSource {
                service_db_path: service_db_path.to_path_buf(),
                request_id: request.request_id.clone(),
                lease_id: None,
                lease_seconds,
            }),
        },
        GepaAdvanceMode::ServiceTick,
    )?;
    Ok(GepaTickOutcome {
        request: Some(request),
        result: advance.result,
        action: advance.action,
        terminal: advance.terminal,
        message: advance.message,
    })
}

fn tick_active_run_request(
    store: &WorkspaceStore,
    service_db_path: &Path,
    request: WorkspaceRunRequestStatus,
    _worker_id: &str,
    lease_seconds: u64,
) -> Result<GepaTickOutcome> {
    let lease_id = request.lease_id.clone().ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "active run request {} has no lease_id",
            request.request_id
        ))
    })?;
    if request.status == "leased" {
        let Some(started) = store.mark_run_request_started_for_lease(
            &request.request_id,
            &lease_id,
            lease_seconds,
        )?
        else {
            return Err(lost_run_request_lease_error(&request.request_id, &lease_id));
        };
        let mut run_store = ensure_run_workspace(store, &started)?;
        let mut cursor = load_gepa_cursor(&run_store, &started.run_id)?
            .unwrap_or_else(|| GepaCursor::new(started.run_id.clone()));
        cursor.phase = GepaCursorPhase::Initializing;
        cursor.metadata = json!({
            "service_request_id": started.request_id,
            "service_db_path": service_db_path,
        });
        persist_cursor_checkpoint(&mut run_store, &cursor, "running", "run request started")?;
        return Ok(GepaTickOutcome {
            request: Some(started.clone()),
            result: None,
            action: GepaTickAction::StartRunRequest {
                request_id: started.request_id.clone(),
                run_id: started.run_id.clone(),
            },
            terminal: false,
            message: "run request started".to_string(),
        });
    }

    let _ = store.heartbeat_run_request(&request.request_id, &lease_id, lease_seconds)?;
    let config = store.run_request_config(&request.request_id)?;
    let advance = match advance_gepa_config_once(
        config,
        GepaExecutionOptions {
            cancellation: Some(GepaCancellationSource {
                service_db_path: service_db_path.to_path_buf(),
                request_id: request.request_id.clone(),
                lease_id: Some(lease_id.clone()),
                lease_seconds,
            }),
        },
        GepaAdvanceMode::ServiceTick,
    ) {
        Ok(outcome) => outcome,
        Err(error) => {
            return terminalize_run_request_error(store, request, &lease_id, error);
        }
    };
    if advance.terminal {
        return terminalize_advanced_run_request(store, request, &lease_id, advance);
    }
    Ok(GepaTickOutcome {
        request: Some(request),
        result: None,
        action: advance.action,
        terminal: false,
        message: advance.message,
    })
}

fn ensure_run_workspace(
    service_store: &WorkspaceStore,
    request: &WorkspaceRunRequestStatus,
) -> Result<WorkspaceStore> {
    let workspace_db_path = request
        .run_workspace_db_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&request.run_dir).join("workspace.sqlite"));
    let store = WorkspaceStore::open(&workspace_db_path)?;
    let config = service_store.run_request_config(&request.request_id)?;
    let paths = ArtifactPaths::new(&config.run.output_dir, &config.run.run_id);
    let cache_mode = CacheMode::from(config.cache.mode);
    let cache_namespace = config
        .cache
        .namespace
        .clone()
        .unwrap_or_else(|| format!("gepa:{}", config.run.run_id));
    store.record_run_started(&paths, &config, cache_mode, &cache_namespace)?;
    Ok(store)
}

fn load_gepa_cursor(store: &WorkspaceStore, run_id: &str) -> Result<Option<GepaCursor>> {
    let Some(checkpoint) = store.latest_checkpoint(run_id, GEPA_CURSOR_CHECKPOINT_KIND)? else {
        return Ok(None);
    };
    serde_json::from_value(checkpoint.snapshot)
        .map(Some)
        .map_err(OptimizerError::from)
}

fn persist_cursor_checkpoint(
    store: &mut WorkspaceStore,
    cursor: &GepaCursor,
    status: &str,
    reason: &str,
) -> Result<()> {
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("gepa_service_tick"));
    let checkpoint = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: cursor.checkpoint_sequence,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status,
        run_state: status,
        reason: Some(reason),
        generation: Some(cursor.generation as u64),
        candidate_id: cursor.best_candidate_id.as_deref(),
        evaluation_stage: Some(cursor.phase.as_str()),
        best_candidate_id: cursor.best_candidate_id.as_deref(),
        candidate_count: cursor.candidates.as_array().map(Vec::len).unwrap_or(0) as u64,
        frontier_count: 0,
        rollout_count: cursor.rollout_count as u64,
        cost_usd: cursor.cost_usd,
        usage: cursor.usage.clone(),
        snapshot: serde_json::to_value(cursor)?,
        metadata,
    });
    store.record_checkpoint(&cursor.run_id, &checkpoint)
}

fn next_cursor_checkpoint_sequence(store: &WorkspaceStore, run_id: &str) -> Result<u64> {
    Ok(store
        .latest_checkpoint(run_id, GEPA_CURSOR_CHECKPOINT_KIND)?
        .map(|checkpoint| checkpoint.sequence_number.saturating_add(1))
        .unwrap_or(1))
}

#[allow(dead_code)]
fn plan_gepa_job(
    run_store: &mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    mut cursor: GepaCursor,
) -> Result<GepaTickOutcome> {
    let effect = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id: &request.run_id,
        effect_kind: "gepa_run",
        lane: "planner",
        status: "planned",
        subject_type: "service_request",
        subject_id: &request.request_id,
        idempotency_key: &format!("gepa-run:{}", request.request_id),
        cache_key: None,
        job_id: Some(format!("gepa-run:{}", request.request_id)),
        budget_reservation_id: None,
        attempt: 1,
        failure_class: None,
        payload: json!({
            "request_id": request.request_id,
            "config_path": request.config_path,
        }),
        metadata: Map::new(),
    });
    run_store.record_runtime_effect(&effect)?;
    let mut job = OptimizerJob::new(
        format!("gepa-run:{}", request.request_id),
        request.run_id.clone(),
        OptimizerJobKind::Checkpoint,
    );
    job.payload.insert(
        "runtime_effect_id".to_string(),
        json!(effect.runtime_effect_id),
    );
    job.payload
        .insert("request_id".to_string(), json!(request.request_id));
    job.payload
        .insert("config_path".to_string(), json!(request.config_path));
    job.payload
        .insert("queue_state".to_string(), json!("queued"));
    run_store.record_optimizer_job(&job)?;
    cursor.phase = GepaCursorPhase::ProposerWaiting;
    cursor.pending_job_id = Some(job.job_id.clone());
    cursor.pending_effect_id = Some(effect.runtime_effect_id.clone());
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(run_store, &request.run_id)?;
    cursor.metadata = json!({
        "service_request_id": request.request_id,
        "planned_job_id": job.job_id,
        "planned_effect_id": effect.runtime_effect_id,
    });
    persist_cursor_checkpoint(run_store, &cursor, "planned", "planned GEPA runtime job")?;
    Ok(GepaTickOutcome {
        request: Some(request.clone()),
        result: None,
        action: GepaTickAction::PlanRuntimeJob {
            run_id: request.run_id.clone(),
            job_id: job.job_id.clone(),
        },
        terminal: false,
        message: "planned GEPA runtime job".to_string(),
    })
}

#[allow(dead_code)]
struct ExecuteGepaJobInput<'a> {
    service_db_path: &'a Path,
    run_store: &'a mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    request_lease_id: &'a str,
    worker_id: &'a str,
    lease_seconds: u64,
    cursor: GepaCursor,
    job: OptimizerJob,
}

#[allow(dead_code)]
fn execute_gepa_job(input: ExecuteGepaJobInput<'_>) -> Result<GepaTickOutcome> {
    let ExecuteGepaJobInput {
        service_db_path,
        run_store,
        request,
        request_lease_id,
        worker_id,
        lease_seconds,
        mut cursor,
        job,
    } = input;
    let job_lease_id = format!("job_{}_{}", worker_id, now_millis());
    let Some(_) = run_store.claim_optimizer_job(
        &request.run_id,
        &job.job_id,
        &job_lease_id,
        Some(worker_id),
        lease_seconds,
    )?
    else {
        return Err(OptimizerError::Invariant(format!(
            "could not claim planned GEPA runtime job run_id={} job_id={}",
            request.run_id, job.job_id
        )));
    };
    let Some(_) = run_store.mark_optimizer_job_running(
        &request.run_id,
        &job.job_id,
        &job_lease_id,
        lease_seconds,
    )?
    else {
        return Err(OptimizerError::Invariant(format!(
            "lost GEPA runtime job lease run_id={} job_id={}",
            request.run_id, job.job_id
        )));
    };
    mark_runtime_effect_status(run_store, &request.run_id, &job, "running", None)?;
    let result = execute_gepa_from_toml_with_options(
        &request.config_path,
        GepaExecutionOptions {
            cancellation: Some(GepaCancellationSource {
                service_db_path: service_db_path.to_path_buf(),
                request_id: request.request_id.clone(),
                lease_id: Some(request_lease_id.to_string()),
                lease_seconds,
            }),
        },
    );
    let mut updated_job = run_store.optimizer_job(&request.run_id, &job.job_id)?;
    match result {
        Ok(result) => {
            let result_value = serde_json::to_value(&result)?;
            updated_job.status = OptimizerJobStatus::Completed;
            updated_job
                .payload
                .insert("queue_state".to_string(), json!("completed"));
            updated_job
                .payload
                .insert("result".to_string(), result_value.clone());
            updated_job.lease_id = None;
            updated_job.worker_id = None;
            updated_job.lease_expires_at = None;
            run_store.record_optimizer_job(&updated_job)?;
            mark_runtime_effect_status(
                run_store,
                &request.run_id,
                &updated_job,
                "completed",
                Some(result_value.clone()),
            )?;
            cursor.phase = GepaCursorPhase::Finalizing;
            cursor.metadata = json!({"result": result_value});
        }
        Err(OptimizerError::Cancelled { .. }) => {
            updated_job.status = OptimizerJobStatus::Cancelled;
            updated_job
                .payload
                .insert("queue_state".to_string(), json!("cancelled"));
            updated_job.payload.insert("error".to_string(), json!({"error_code": "synth_optimizer_cancelled", "message": "run request cancelled"}));
            updated_job.lease_id = None;
            updated_job.worker_id = None;
            updated_job.lease_expires_at = None;
            run_store.record_optimizer_job(&updated_job)?;
            mark_runtime_effect_status(
                run_store,
                &request.run_id,
                &updated_job,
                "cancelled",
                updated_job.payload.get("error").cloned(),
            )?;
            cursor.phase = GepaCursorPhase::Cancelled;
        }
        Err(error) => {
            let error_payload = json!({
                "error_code": error.error_code(),
                "message": error.to_string(),
            });
            updated_job.status = OptimizerJobStatus::Failed;
            updated_job.failure = Some(FailurePayload::from_optimizer_error(&error));
            updated_job
                .payload
                .insert("queue_state".to_string(), json!("failed"));
            updated_job
                .payload
                .insert("error".to_string(), error_payload.clone());
            updated_job.lease_id = None;
            updated_job.worker_id = None;
            updated_job.lease_expires_at = None;
            run_store.record_optimizer_job(&updated_job)?;
            mark_runtime_effect_status(
                run_store,
                &request.run_id,
                &updated_job,
                "failed",
                Some(error_payload),
            )?;
            cursor.phase = GepaCursorPhase::Failed;
        }
    }
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(run_store, &request.run_id)?;
    persist_cursor_checkpoint(run_store, &cursor, "running", "executed GEPA runtime job")?;
    Ok(GepaTickOutcome {
        request: Some(request.clone()),
        result: None,
        action: GepaTickAction::ExecuteRuntimeJob {
            run_id: request.run_id.clone(),
            job_id: job.job_id.clone(),
        },
        terminal: false,
        message: "executed GEPA runtime job".to_string(),
    })
}

#[allow(dead_code)]
fn mark_runtime_effect_status(
    run_store: &WorkspaceStore,
    run_id: &str,
    job: &OptimizerJob,
    status: &str,
    terminal_payload: Option<Value>,
) -> Result<()> {
    let Some(effect_id) = job.payload.get("runtime_effect_id").and_then(Value::as_str) else {
        return Ok(());
    };
    let existing = run_store.runtime_effect(run_id, effect_id)?;
    let mut payload = existing.payload.clone();
    if let Some(payload_object) = payload.as_object_mut() {
        payload_object.insert("completion_status".to_string(), json!(status));
        if let Some(terminal_payload) = terminal_payload {
            payload_object.insert("terminal_payload".to_string(), terminal_payload);
        }
    }
    let effect = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id,
        effect_kind: &existing.effect_kind,
        lane: &existing.lane,
        status,
        subject_type: &existing.subject_type,
        subject_id: &existing.subject_id,
        idempotency_key: &existing.idempotency_key,
        cache_key: existing.cache_key.clone(),
        job_id: existing.job_id.clone(),
        budget_reservation_id: existing.budget_reservation_id.clone(),
        attempt: existing.attempt,
        failure_class: if status == "failed" {
            Some("gepa_runtime_job_failed".to_string())
        } else {
            existing.failure_class.clone()
        },
        payload,
        metadata: existing.metadata.clone(),
    });
    run_store.record_runtime_effect(&effect)
}

fn terminalize_advanced_run_request(
    store: &WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    advance: GepaAdvanceOutcome,
) -> Result<GepaTickOutcome> {
    match advance.result {
        Some(result) => {
            let result_value = serde_json::to_value(&result)?;
            if !store.record_run_request_result_for_lease(
                &request.request_id,
                lease_id,
                &result_value,
            )? {
                return Err(lost_run_request_lease_error(&request.request_id, lease_id));
            }
            let Some(completed) =
                store.mark_run_request_completed_for_lease(&request.request_id, lease_id)?
            else {
                return Err(lost_run_request_lease_error(&request.request_id, lease_id));
            };
            Ok(GepaTickOutcome {
                request: Some(completed),
                result: Some(result),
                action: advance.action,
                terminal: true,
                message: advance.message,
            })
        }
        None => {
            let status = match &advance.action {
                GepaTickAction::TerminalizeRun { status, .. } => status.as_str(),
                _ => "failed",
            };
            if status == "cancelled" {
                let cancelled = store
                    .mark_run_request_cancelled_for_lease(
                        &request.request_id,
                        lease_id,
                        "cancelled during GEPA tick",
                    )?
                    .ok_or_else(|| lost_run_request_lease_error(&request.request_id, lease_id))?;
                Ok(GepaTickOutcome {
                    request: Some(cancelled),
                    result: None,
                    action: advance.action,
                    terminal: true,
                    message: advance.message,
                })
            } else {
                let error_payload = json!({
                    "error_code": "synth_optimizer_failed",
                    "message": advance.message,
                });
                let Some(failed) = store.mark_run_request_failed_for_lease(
                    &request.request_id,
                    lease_id,
                    &error_payload,
                )?
                else {
                    return Err(lost_run_request_lease_error(&request.request_id, lease_id));
                };
                Ok(GepaTickOutcome {
                    request: Some(failed),
                    result: None,
                    action: advance.action,
                    terminal: true,
                    message: "run request failed".to_string(),
                })
            }
        }
    }
}

fn terminalize_run_request_error(
    store: &WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    error: OptimizerError,
) -> Result<GepaTickOutcome> {
    if matches!(error, OptimizerError::Cancelled { .. }) {
        let cancelled = store
            .mark_run_request_cancelled_for_lease(
                &request.request_id,
                lease_id,
                "cancelled during GEPA tick",
            )?
            .ok_or_else(|| lost_run_request_lease_error(&request.request_id, lease_id))?;
        return Ok(GepaTickOutcome {
            request: Some(cancelled),
            result: None,
            action: GepaTickAction::TerminalizeRun {
                run_id: request.run_id,
                status: "cancelled".to_string(),
            },
            terminal: true,
            message: "run request cancelled".to_string(),
        });
    }
    let error_payload = json!({
        "error_code": error.error_code(),
        "message": error.to_string(),
    });
    let Some(failed) =
        store.mark_run_request_failed_for_lease(&request.request_id, lease_id, &error_payload)?
    else {
        return Err(lost_run_request_lease_error(&request.request_id, lease_id));
    };
    Ok(GepaTickOutcome {
        request: Some(failed),
        result: None,
        action: GepaTickAction::TerminalizeRun {
            run_id: request.run_id,
            status: "failed".to_string(),
        },
        terminal: true,
        message: "run request failed".to_string(),
    })
}

#[allow(dead_code)]
fn consume_terminal_job(
    store: &WorkspaceStore,
    run_store: &mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    mut cursor: GepaCursor,
    job: OptimizerJob,
) -> Result<GepaTickOutcome> {
    match job.status {
        OptimizerJobStatus::Completed => {
            let result_value = job.payload.get("result").cloned().ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "completed GEPA job {} has no result payload",
                    job.job_id
                ))
            })?;
            if !store.record_run_request_result_for_lease(
                &request.request_id,
                lease_id,
                &result_value,
            )? {
                return Err(lost_run_request_lease_error(&request.request_id, lease_id));
            }
            let result: GepaRunResult = serde_json::from_value(result_value.clone())?;
            let Some(completed) =
                store.mark_run_request_completed_for_lease(&request.request_id, lease_id)?
            else {
                return Err(lost_run_request_lease_error(&request.request_id, lease_id));
            };
            cursor.phase = GepaCursorPhase::Completed;
            cursor.pending_job_id = None;
            cursor.pending_effect_id = None;
            cursor.metadata = json!({"result": result_value});
            cursor.checkpoint_sequence =
                next_cursor_checkpoint_sequence(run_store, &request.run_id)?;
            persist_cursor_checkpoint(
                run_store,
                &cursor,
                "completed",
                "consumed GEPA runtime outcome",
            )?;
            Ok(GepaTickOutcome {
                request: Some(completed),
                result: Some(result),
                action: GepaTickAction::ConsumeRuntimeOutcome {
                    run_id: request.run_id,
                    job_id: job.job_id,
                },
                terminal: true,
                message: "run request completed".to_string(),
            })
        }
        OptimizerJobStatus::Cancelled => {
            terminalize_cancelled_job(store, run_store, request, lease_id, cursor, job)
        }
        OptimizerJobStatus::Failed | OptimizerJobStatus::Expired => {
            terminalize_failed_job(store, run_store, request, lease_id, cursor, job)
        }
        _ => Err(OptimizerError::Invariant(format!(
            "job {} is not terminal for consumption",
            job.job_id
        ))),
    }
}

#[allow(dead_code)]
fn consume_terminal_cursor(
    store: &WorkspaceStore,
    run_store: &mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    cursor: GepaCursor,
) -> Result<GepaTickOutcome> {
    let Some(job_id) = cursor.pending_job_id.clone() else {
        return Err(OptimizerError::Invariant(format!(
            "terminal cursor for run {} has no pending job to consume",
            request.run_id
        )));
    };
    let job = run_store.optimizer_job(&request.run_id, &job_id)?;
    consume_terminal_job(store, run_store, request, lease_id, cursor, job)
}

fn terminalize_cancelled_job(
    store: &WorkspaceStore,
    run_store: &mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    mut cursor: GepaCursor,
    job: OptimizerJob,
) -> Result<GepaTickOutcome> {
    let cancelled = match store.run_request(&request.request_id) {
        Ok(request) if request.status == "cancelled" => request,
        _ => store
            .mark_run_request_cancelled_for_lease(
                &request.request_id,
                lease_id,
                "cancelled during execution",
            )?
            .ok_or_else(|| lost_run_request_lease_error(&request.request_id, lease_id))?,
    };
    let _ = run_store.record_run_cancelled(&request.run_id);
    cursor.phase = GepaCursorPhase::Cancelled;
    cursor.pending_job_id = None;
    cursor.pending_effect_id = None;
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(run_store, &request.run_id)?;
    persist_cursor_checkpoint(
        run_store,
        &cursor,
        "cancelled",
        "consumed cancelled GEPA runtime outcome",
    )?;
    Ok(GepaTickOutcome {
        request: Some(cancelled),
        result: None,
        action: GepaTickAction::TerminalizeRun {
            run_id: request.run_id,
            status: "cancelled".to_string(),
        },
        terminal: true,
        message: format!("run request cancelled after job {}", job.job_id),
    })
}

fn terminalize_failed_job(
    store: &WorkspaceStore,
    run_store: &mut WorkspaceStore,
    request: WorkspaceRunRequestStatus,
    lease_id: &str,
    mut cursor: GepaCursor,
    job: OptimizerJob,
) -> Result<GepaTickOutcome> {
    let error_payload = job.payload.get("error").cloned().unwrap_or_else(|| {
        json!({
            "error_code": "synth_optimizer_failed",
            "message": format!("GEPA runtime job {} failed", job.job_id),
        })
    });
    let Some(failed) =
        store.mark_run_request_failed_for_lease(&request.request_id, lease_id, &error_payload)?
    else {
        return Err(lost_run_request_lease_error(&request.request_id, lease_id));
    };
    cursor.phase = GepaCursorPhase::Failed;
    cursor.pending_job_id = None;
    cursor.pending_effect_id = None;
    cursor.metadata = json!({"error": error_payload});
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(run_store, &request.run_id)?;
    persist_cursor_checkpoint(
        run_store,
        &cursor,
        "failed",
        "consumed failed GEPA runtime outcome",
    )?;
    Ok(GepaTickOutcome {
        request: Some(failed),
        result: None,
        action: GepaTickAction::TerminalizeRun {
            run_id: request.run_id,
            status: "failed".to_string(),
        },
        terminal: true,
        message: format!("run request failed after job {}", job.job_id),
    })
}

pub fn recover_service_state(db_path: impl AsRef<Path>) -> Result<GepaServiceRecoveryOutcome> {
    let store = WorkspaceStore::open(&db_path)?;
    let recovered_run_requests = store.recover_expired_run_requests()?;
    let mut recovered_run_workspaces = Vec::new();
    for request in store.status()?.run_requests {
        let workspace_db_path = request
            .run_workspace_db_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(&request.run_dir).join("workspace.sqlite"));
        if !workspace_db_path.exists() {
            continue;
        }
        let run_store = WorkspaceStore::open_existing(&workspace_db_path)?;
        let recovered = run_store.recover_expired_optimizer_jobs(&request.run_id)?;
        if recovered.is_empty() {
            continue;
        }
        recovered_run_workspaces.push(GepaRunWorkspaceRecovery {
            run_id: request.run_id,
            workspace_db_path: workspace_db_path.display().to_string(),
            recovered_optimizer_jobs: recovered
                .into_iter()
                .map(|job| GepaRecoveredOptimizerJob {
                    job_id: job.job_id,
                    status: job.status.as_str().to_string(),
                    attempt: job.attempt,
                    lease_id: job.lease_id,
                    next_retry_at: job.next_retry_at,
                })
                .collect(),
        });
    }
    Ok(GepaServiceRecoveryOutcome {
        recovered_run_requests,
        recovered_run_workspaces,
    })
}

fn lost_run_request_lease_error(request_id: &str, lease_id: &str) -> OptimizerError {
    OptimizerError::Invariant(format!(
        "run request {request_id} is no longer owned by lease {lease_id}"
    ))
}

fn handle_connection(mut stream: TcpStream, config: GepaServiceConfig) -> Result<()> {
    let request = HttpRequest::read(&mut stream)?;
    let response = route_request(request, config);
    write_response(&mut stream, response)
}

fn route_request(request: HttpRequest, config: GepaServiceConfig) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => json_response(200, &json!({"status": "ok"})),
        ("GET", "/status") => {
            let response = WorkspaceStore::open_existing(&config.db_path)
                .and_then(|store| store.status())
                .and_then(|status| serde_json::to_value(status).map_err(OptimizerError::from));
            result_response(response)
        }
        ("POST", "/runs") => {
            let submit = serde_json::from_slice::<GepaServiceSubmitRequest>(&request.body)
                .map_err(OptimizerError::from)
                .and_then(|submit| submit_run(&config, submit));
            result_response(submit)
        }
        ("POST", "/worker/run-next") => {
            let outcome =
                run_next_queued_request(&config.db_path, &config.worker_id, config.lease_seconds)
                    .and_then(|outcome| {
                        serde_json::to_value(outcome).map_err(OptimizerError::from)
                    });
            result_response(outcome)
        }
        ("POST", "/worker/tick") => {
            let outcome = tick_next_unit(&config.db_path, &config.worker_id, config.lease_seconds)
                .and_then(|outcome| serde_json::to_value(outcome).map_err(OptimizerError::from));
            result_response(outcome)
        }
        ("POST", "/worker/recover") => {
            let recovered = recover_service_state(&config.db_path)
                .and_then(|outcome| serde_json::to_value(outcome).map_err(OptimizerError::from));
            result_response(recovered)
        }
        _ => json_response(
            404,
            &json!({
                "error": "not_found",
                "method": request.method,
                "path": request.path,
            }),
        ),
    }
}

fn submit_run(config: &GepaServiceConfig, submit: GepaServiceSubmitRequest) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let request = store.submit_run_request(&submit.config_path, submit.priority)?;
    if submit.auto_start {
        let worker_config = config.clone();
        thread::spawn(move || {
            let _ = run_next_queued_request(
                &worker_config.db_path,
                &worker_config.worker_id,
                worker_config.lease_seconds,
            );
        });
    }
    serde_json::to_value(GepaServiceSubmitResponse {
        request,
        auto_started: submit.auto_start,
    })
    .map_err(OptimizerError::from)
}

fn result_response(result: Result<Value>) -> HttpResponse {
    match result {
        Ok(value) => json_response(200, &value),
        Err(error) => json_response(
            500,
            &json!({
                "error_code": error.error_code(),
                "error": error.to_string(),
            }),
        ),
    }
}

fn json_response(status: u16, value: &Value) -> HttpResponse {
    HttpResponse {
        status,
        body: serde_json::to_vec_pretty(value).unwrap_or_else(|_| b"{}".to_vec()),
    }
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> Result<()> {
    let reason = match response.status {
        200 => "OK",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        reason,
        response.body.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|source| OptimizerError::io("gepa service response", source))?;
    stream
        .write_all(&response.body)
        .map_err(|source| OptimizerError::io("gepa service response", source))?;
    stream
        .flush()
        .map_err(|source| OptimizerError::io("gepa service response", source))?;
    Ok(())
}

fn default_auto_start() -> bool {
    true
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl HttpRequest {
    fn read(stream: &mut TcpStream) -> Result<Self> {
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|source| OptimizerError::io("gepa service request", source))?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();
        if method.is_empty() || path.is_empty() {
            return Err(OptimizerError::Config(
                "malformed HTTP request line".to_string(),
            ));
        }

        let mut headers = BTreeMap::new();
        loop {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|source| OptimizerError::io("gepa service headers", source))?;
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = vec![0; content_length];
        if content_length > 0 {
            reader
                .read_exact(&mut body)
                .map_err(|source| OptimizerError::io("gepa service body", source))?;
        }
        Ok(Self { method, path, body })
    }
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}
