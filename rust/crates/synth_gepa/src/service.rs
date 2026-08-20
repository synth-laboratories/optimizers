use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::{Digest as Sha2Digest, Sha256};
use synth_optimizer_platform::{
    compact_run_storage, delete_run_storage, inspect_run_storage, inspect_workspace_storage_health,
    ArtifactPaths, CacheMode, CheckpointInput, CheckpointRecord, ContainerClient, FailurePayload,
    GepaPipelineMode, GepaStalenessPolicy, GepaTaskPoolsConfig, OptimizerError, OptimizerJob,
    OptimizerJobKind, OptimizerJobStatus, PromptProgram, Result, RunPhaseTimingRecord,
    RunStorageInspectionInput, RunStorageMaintenanceInput, RuntimeEffectInput, RuntimeEffectRecord,
    StorageHealthThresholds, StorageMaintenanceProfile, SynthOptimizerConfig, TransitionLog,
    TransitionRow, WorkspaceRunRequestStatus, WorkspaceStorageHealthInput, WorkspaceStore,
};

use crate::{
    advance_gepa_config_once, execute_gepa_from_toml_with_options, export_cursor_fixture,
    import_cursor_fixture, pin_run_checkpoint,
    planner::{
        GepaCursor, GepaCursorPhase, GepaTickAction, GepaTickOutcome, GEPA_CURSOR_CHECKPOINT_KIND,
    },
    project_gepa_limit_snapshot, record_initial_platform_snapshots, GepaAdvanceMode,
    GepaAdvanceOutcome, GepaCancellationSource, GepaExecutionOptions, GepaRunResult,
};

const DEFAULT_SERVICE_WORKER_COUNT: usize = 10;
const MAX_SERVICE_WORKER_COUNT: usize = 10;
const SERVICE_WORKER_IDLE_WAIT: Duration = Duration::from_millis(500);
const SERVICE_WORKER_ERROR_WAIT: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaServiceConfig {
    pub db_path: PathBuf,
    pub bind_addr: String,
    pub worker_id: String,
    pub lease_seconds: u64,
    pub worker_count: usize,
}

impl GepaServiceConfig {
    pub fn new(db_path: impl Into<PathBuf>, bind_addr: impl Into<String>) -> Self {
        Self {
            db_path: db_path.into(),
            bind_addr: bind_addr.into(),
            worker_id: "synth-gepa-service".to_string(),
            lease_seconds: 3600,
            worker_count: DEFAULT_SERVICE_WORKER_COUNT,
        }
    }
}

#[derive(Clone)]
struct GepaServiceRuntime {
    config: GepaServiceConfig,
    scheduler: ServiceSchedulerSignal,
    service_url: String,
    started_at: String,
}

#[derive(Clone)]
struct ServiceSchedulerSignal {
    state: Arc<(Mutex<u64>, Condvar)>,
}

impl ServiceSchedulerSignal {
    fn new() -> Self {
        Self {
            state: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }

    fn notify(&self) {
        let (lock, condvar) = &*self.state;
        if let Ok(mut generation) = lock.lock() {
            *generation = generation.saturating_add(1);
            condvar.notify_all();
        }
    }

    fn wait(&self, duration: Duration) {
        let (lock, condvar) = &*self.state;
        if let Ok(generation) = lock.lock() {
            let _ = condvar.wait_timeout(generation, duration);
        } else {
            thread::sleep(duration);
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GepaServiceRunRequest {
    container_url: String,
    #[serde(default)]
    output_dir: Option<String>,
    policy: ServicePolicySpec,
    proposer: ServiceProposerSpec,
    taskset: ServiceTasksetSpec,
    // Sibling of taskset, mirroring `GepaConfig(taskset=..., task_pools=...)` and
    // the standalone `[gepa.task_pools]` TOML — one canonical location everywhere.
    task_pools: GepaTaskPoolsConfig,
    #[serde(default)]
    manual_step: bool,
    #[serde(default)]
    stop_conditions: Vec<ServiceStopCondition>,
    #[serde(default)]
    advanced: ServiceAdvancedConfig,
    #[serde(default)]
    fork_from: Option<ServiceForkFrom>,
    #[serde(default)]
    fixture: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ServiceForkFrom {
    run_id: String,
    #[serde(default)]
    checkpoint_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServicePolicySpec {
    provider: String,
    model: String,
    #[serde(default = "default_api_family")]
    api_family: String,
    credentials: ServiceCredentials,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    inference_url: Option<String>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default = "default_disable_reasoning")]
    disable_reasoning: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceProposerSpec {
    provider: String,
    model: String,
    #[serde(default = "default_api_family")]
    api_family: String,
    #[serde(default)]
    auth_mode: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    credentials: ServiceCredentials,
    #[serde(default)]
    copy_host_auth: Option<bool>,
    #[serde(default)]
    codex_home: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Explicit opt-in for an OpenRouter proposer model outside the verified
    /// allowlist. Without it the service cannot express e.g.
    /// nvidia/nemotron-3.5-lightning, because the wire spec is
    /// deny_unknown_fields and the internal config defaults this to false.
    #[serde(default)]
    allow_unverified_model: Option<bool>,
    /// Codex `model_context_window` for the proposer's generated config.toml.
    #[serde(default)]
    model_context_window: Option<u64>,
    /// Codex `model_auto_compact_token_limit` for the proposer's generated config.toml.
    #[serde(default)]
    model_auto_compact_token_limit: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceCredentials {
    resolver: String,
    #[serde(default)]
    env_var: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceTasksetSpec {
    train_ids: Vec<String>,
    heldout_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ServiceStopCondition {
    MaxRollouts {
        n: usize,
        #[serde(default)]
        train: Option<usize>,
        #[serde(default)]
        heldout: Option<usize>,
    },
    MaxWallSeconds {
        n: u64,
    },
    MaxGenerations {
        n: usize,
    },
    MaxCostUsd {
        value: f64,
    },
    NoImprovement {
        generations: usize,
        metric: Option<String>,
    },
    ScoreThreshold {
        value: f64,
        metric: Option<String>,
    },
    Episode {
        #[serde(default)]
        proposer_rounds: Option<usize>,
        #[serde(default)]
        max_rollouts: Option<usize>,
        #[serde(default)]
        max_wall_seconds: Option<u64>,
        #[serde(default)]
        max_spend_usd: Option<f64>,
        #[serde(default)]
        skip_heldout: bool,
    },
    ExternalSignal,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceAdvancedConfig {
    #[serde(default)]
    pipeline: Option<ServicePipelineConfig>,
    #[serde(default)]
    budgets: Option<ServiceBudgetConfig>,
    #[serde(default)]
    timeouts: Option<ServiceTimeoutsConfig>,
    #[serde(default)]
    policy_io: Option<Value>,
    #[serde(default)]
    proposer_io: Option<ServiceProposerIoConfig>,
    #[serde(default)]
    adaptive_rollout_concurrency: Option<bool>,
    #[serde(default)]
    operator: Option<synth_optimizer_platform::GepaOperatorConfig>,
    #[serde(default)]
    jesterky_workflow: Option<ServiceJesterkyWorkflowConfig>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceBudgetConfig {
    #[serde(default)]
    max_train_rollouts: Option<usize>,
    #[serde(default)]
    max_heldout_rollouts: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServicePipelineConfig {
    #[serde(default)]
    mode: Option<GepaPipelineMode>,
    #[serde(default)]
    staleness_policy: Option<GepaStalenessPolicy>,
    #[serde(default)]
    delta_max: Option<u64>,
    #[serde(default)]
    max_generations: Option<usize>,
    #[serde(default)]
    proposals_per_generation: Option<usize>,
    #[serde(default)]
    minibatch_size: Option<usize>,
    #[serde(default)]
    max_in_flight_candidates: Option<usize>,
    #[serde(default)]
    rollout_workers: Option<usize>,
    #[serde(default)]
    proposer_workers: Option<usize>,
    #[serde(default)]
    evaluator_workers: Option<usize>,
    #[serde(default)]
    rollout_chunk_size: Option<usize>,
    #[serde(default)]
    speculative_alpha: Option<f64>,
    #[serde(default)]
    adaptive_stage_workers: Option<bool>,
    #[serde(default)]
    adaptive_rollout_concurrency: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceTimeoutsConfig {
    #[serde(default)]
    rollout_seconds: Option<u64>,
    #[serde(default)]
    container_http_seconds: Option<u64>,
    #[serde(default)]
    rollout_http_retries: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceProposerIoConfig {
    #[serde(default)]
    timeout_seconds: Option<u64>,
    #[serde(default)]
    message_stall_timeout_seconds: Option<u64>,
    #[serde(default)]
    codex_home: Option<String>,
    #[serde(default)]
    schema_repair_rounds: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceJesterkyWorkflowConfig {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    bulk: Option<bool>,
    #[serde(default)]
    spec: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    actor: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    fail_closed: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSocketSubscribeFrame {
    #[serde(rename = "type")]
    frame_type: String,
    #[serde(default)]
    kinds: Vec<String>,
    #[serde(default)]
    since: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSocketControlFrame {
    #[serde(rename = "type")]
    frame_type: String,
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

#[derive(Clone, Debug)]
struct ProjectedRunEvent {
    seq: u64,
    ts: String,
    kind: &'static str,
    payload: Value,
}

enum WebSocketIncoming {
    Text(String),
    Closed,
    Empty,
}

pub fn run_gepa_service(mut config: GepaServiceConfig) -> Result<()> {
    config.worker_count = normalize_service_worker_count(config.worker_count);
    recover_service_state(&config.db_path)?;
    let scheduler = ServiceSchedulerSignal::new();
    let listener = TcpListener::bind(&config.bind_addr).map_err(|source| {
        OptimizerError::io(format!("gepa service bind {}", config.bind_addr), source)
    })?;
    config.bind_addr = listener
        .local_addr()
        .map(|addr| addr.to_string())
        .unwrap_or_else(|_| config.bind_addr.clone());
    let service_url = service_url_from_bind(&config.bind_addr);
    let started_at = crate::rfc3339_now();
    start_service_workers(&config, scheduler.clone(), service_url.clone());
    let _heartbeat = ServiceHeartbeatGuard::start(&config, &service_url, &started_at)?;
    let runtime = GepaServiceRuntime {
        config,
        scheduler,
        service_url,
        started_at,
    };
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let runtime = runtime.clone();
                thread::spawn(move || {
                    let _ = handle_connection(stream, runtime);
                });
            }
            Err(source) => {
                return Err(OptimizerError::io("gepa service accept", source));
            }
        }
    }
    Ok(())
}

fn normalize_service_worker_count(worker_count: usize) -> usize {
    worker_count.clamp(1, MAX_SERVICE_WORKER_COUNT)
}

fn service_worker_id(base_worker_id: &str, worker_count: usize, slot: usize) -> String {
    if worker_count == 1 {
        base_worker_id.to_string()
    } else {
        format!("{base_worker_id}-slot-{slot:02}")
    }
}

fn start_service_workers(
    config: &GepaServiceConfig,
    scheduler: ServiceSchedulerSignal,
    service_url: String,
) {
    for slot in 0..config.worker_count {
        let db_path = config.db_path.clone();
        let lease_seconds = config.lease_seconds;
        let worker_count = config.worker_count;
        let worker_id = service_worker_id(&config.worker_id, worker_count, slot);
        let scheduler = scheduler.clone();
        let service_url = service_url.clone();
        thread::spawn(move || loop {
            match tick_next_auto_unit(&db_path, &worker_id, lease_seconds, Some(&service_url)) {
                Ok(outcome) => {
                    if outcome.request.is_none() {
                        scheduler.wait(SERVICE_WORKER_IDLE_WAIT);
                    }
                }
                Err(error) => {
                    eprintln!("[gepa-worker:{worker_id}] run loop error: {error}");
                    scheduler.wait(SERVICE_WORKER_ERROR_WAIT);
                }
            }
        });
    }
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
    tick_next_unit_inner(db_path, worker_id, lease_seconds, true, true, None)
}

fn tick_next_auto_unit(
    db_path: impl AsRef<Path>,
    worker_id: &str,
    lease_seconds: u64,
    owning_service_url: Option<&str>,
) -> Result<GepaTickOutcome> {
    tick_next_unit_inner(
        db_path,
        worker_id,
        lease_seconds,
        false,
        is_primary_service_worker(worker_id),
        owning_service_url,
    )
}

fn tick_next_unit_inner(
    db_path: impl AsRef<Path>,
    worker_id: &str,
    lease_seconds: u64,
    include_manual_step: bool,
    run_recovery: bool,
    owning_service_url: Option<&str>,
) -> Result<GepaTickOutcome> {
    if run_recovery {
        recover_service_state(&db_path)?;
    }
    let lease_id = format!("lease_{}_{}", worker_id, now_millis());
    let store = WorkspaceStore::open_existing(&db_path)?;
    if let Some(active) = active_run_request(&store, worker_id)? {
        return tick_active_run_request(
            &store,
            db_path.as_ref(),
            active,
            worker_id,
            lease_seconds,
            owning_service_url,
        );
    }
    if include_manual_step || is_primary_service_worker(worker_id) {
        if let Some(cancelled) = cancelled_run_request_needing_workspace_terminalization(&store)? {
            return tick_cancelled_run_request_workspace(
                db_path.as_ref(),
                cancelled,
                lease_seconds,
            );
        }
    }
    let claimed = if include_manual_step {
        store.claim_next_run_request(&lease_id, Some(worker_id), lease_seconds)?
    } else {
        store.claim_next_auto_run_request(&lease_id, Some(worker_id), lease_seconds)?
    };
    let Some(claimed) = claimed else {
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

fn is_primary_service_worker(worker_id: &str) -> bool {
    !worker_id.contains("-slot-") || worker_id.ends_with("-slot-00")
}

struct ServiceHeartbeatGuard {
    path: PathBuf,
    stop: Arc<AtomicBool>,
}

impl ServiceHeartbeatGuard {
    fn start(config: &GepaServiceConfig, service_url: &str, started_at: &str) -> Result<Self> {
        let home = crate::gepa_home_dir();
        let services = home.join("services");
        fs::create_dir_all(&services).map_err(|source| OptimizerError::io(&services, source))?;
        let path = services.join(format!("{}.json", service_id_for_db(&config.db_path)));
        let stop = Arc::new(AtomicBool::new(false));
        write_service_heartbeat(&path, config, service_url, started_at)?;
        let thread_path = path.clone();
        let thread_config = config.clone();
        let thread_service_url = service_url.to_string();
        let thread_started_at = started_at.to_string();
        let thread_stop = stop.clone();
        thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                let _ = write_service_heartbeat(
                    &thread_path,
                    &thread_config,
                    &thread_service_url,
                    &thread_started_at,
                );
                thread::sleep(Duration::from_secs(2));
            }
        });
        Ok(Self { path, stop })
    }
}

impl Drop for ServiceHeartbeatGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = fs::remove_file(&self.path);
    }
}

fn write_service_heartbeat(
    path: &Path,
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
) -> Result<()> {
    let payload =
        service_identity_payload(config, service_url, started_at, Some(crate::rfc3339_now()));
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&payload)?)
        .map_err(|source| OptimizerError::io(&tmp, source))?;
    fs::rename(&tmp, path).map_err(|source| OptimizerError::io(path, source))
}

fn service_identity_payload(
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
    last_seen: Option<String>,
) -> Value {
    let mut payload = json!({
        "kind": "gepa-service",
        "schema": "synth.gepa_service.whoami.v1",
        "version": env!("CARGO_PKG_VERSION"),
        "source_id": service_id_for_db(&config.db_path),
        "service_url": service_url,
        "bind": config.bind_addr.clone(),
        "pid": std::process::id(),
        "db_path": crate::absolute_path(&config.db_path).display().to_string(),
        "worker_id": config.worker_id.clone(),
        "workers": config.worker_count,
        "lease_seconds": config.lease_seconds,
        "started_at": started_at,
        "run_roots": [],
    });
    if let Some(last_seen) = last_seen {
        if let Some(object) = payload.as_object_mut() {
            object.insert("last_seen".to_string(), json!(last_seen));
        }
    }
    payload
}

fn service_id_for_db(db_path: &Path) -> String {
    let mut hasher = Sha256::new();
    Sha2Digest::update(
        &mut hasher,
        crate::absolute_path(db_path)
            .display()
            .to_string()
            .as_bytes(),
    );
    format!("{:x}", Sha2Digest::finalize(hasher))
}

fn service_url_from_bind(bind_addr: &str) -> String {
    let (host, port) = bind_addr
        .rsplit_once(':')
        .map(|(host, port)| (host.trim_matches(['[', ']']), port))
        .unwrap_or(("127.0.0.1", "8879"));
    let host = match host {
        "" | "0.0.0.0" | "::" => "127.0.0.1",
        other => other,
    };
    if host.contains(':') {
        format!("http://[{host}]:{port}")
    } else {
        format!("http://{host}:{port}")
    }
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
                in_process: None,
            }),
            owning_service_url: None,
            artifact_store: None,
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
    owning_service_url: Option<&str>,
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
        let existing_cursor = load_gepa_cursor(&run_store, &started.run_id)?;
        let mut cursor = existing_cursor.unwrap_or_else(|| {
            let mut cursor = GepaCursor::new(started.run_id.clone());
            cursor.phase = GepaCursorPhase::Initializing;
            cursor
        });
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
    ensure_run_workspace(store, &request)?;
    let config = store.run_request_config(&request.request_id)?;
    let advance = match advance_gepa_config_once(
        config,
        GepaExecutionOptions {
            cancellation: Some(GepaCancellationSource {
                service_db_path: service_db_path.to_path_buf(),
                request_id: request.request_id.clone(),
                lease_id: Some(lease_id.clone()),
                lease_seconds,
                in_process: None,
            }),
            owning_service_url: owning_service_url.map(str::to_string),
            artifact_store: None,
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
    record_initial_platform_snapshots(&store, &config, cache_mode, &cache_namespace, &paths)?;
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
    if matches!(
        cursor.phase,
        GepaCursorPhase::GenerationStart | GepaCursorPhase::Paused
    ) || status == "paused"
    {
        metadata.insert("retain".to_string(), json!(true));
    }
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
    store.record_checkpoint_compacting_previous(&cursor.run_id, &checkpoint)
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
                in_process: None,
            }),
            owning_service_url: None,
            artifact_store: None,
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
            let mut result_value = serde_json::to_value(&result)?;
            if result_value
                .get("stopped_by")
                .and_then(Value::as_object)
                .is_none()
            {
                if let Some(stopped_by) = stopped_by_for_request(&request)? {
                    if let Some(result_object) = result_value.as_object_mut() {
                        result_object.insert("stopped_by".to_string(), stopped_by);
                    }
                }
            }
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

fn handle_connection(mut stream: TcpStream, runtime: GepaServiceRuntime) -> Result<()> {
    let request = HttpRequest::read(&mut stream)?;
    let (path, _) = split_path_query(&request.path);
    let segments = path_segments(path);
    if request.method == "GET" && matches!(segments.as_slice(), ["runs", _, "ws"]) {
        if let ["runs", run_id, "ws"] = segments.as_slice() {
            return handle_websocket_connection(&mut stream, runtime, &request, run_id);
        }
    }
    let response = route_request(request, runtime);
    write_response(&mut stream, response)
}

fn route_request(request: HttpRequest, runtime: GepaServiceRuntime) -> HttpResponse {
    let (path, query) = split_path_query(&request.path);
    let segments = path_segments(path);
    let config = &runtime.config;
    match (request.method.as_str(), segments.as_slice()) {
        ("GET", ["health"]) => json_response(200, &json!({"status": "ok"})),
        ("GET", ["whoami"]) => json_response(
            200,
            &service_identity_payload(config, &runtime.service_url, &runtime.started_at, None),
        ),
        ("GET", ["openapi.yaml"]) => {
            text_response(200, "application/yaml", GEPA_SERVICE_OPENAPI_YAML)
        }
        ("GET", ["workspace"]) => result_response(workspace_summary(config), 200),
        ("GET", ["workspace", "usage"]) => result_response(workspace_usage(config), 200),
        ("GET", ["workspace", "storage"]) => result_response(workspace_storage(config), 200),
        ("POST", ["workspace", "prune"]) => result_response(prune_workspace(config, &request), 200),
        ("POST", ["workspace", "compact"]) => {
            result_response(compact_workspace_response(config, &request), 200)
        }
        ("POST", ["runs"]) => create_run_response(&runtime, &request),
        ("GET", ["runs"]) => result_response(list_runs(config, &query), 200),
        ("GET", ["runs", run_id]) => run_response(config, run_id),
        ("GET", ["runs", run_id, "state"]) => run_state_response(config, run_id),
        ("GET", ["runs", run_id, "limits"]) => run_limits_response(config, run_id),
        ("GET", ["runs", run_id, "timings"]) => run_timings_response(config, run_id),
        ("GET", ["runs", run_id, "stats"]) => run_stats_response(config, run_id),
        ("GET", ["runs", run_id, "storage"]) => run_storage_response(config, run_id),
        ("DELETE", ["runs", run_id]) => delete_run_response(config, run_id),
        ("POST", ["runs", run_id, "compact"]) => compact_run_response(config, run_id, &request),
        ("POST", ["runs", run_id, "cancel"]) => {
            control_run_response(&runtime, run_id, "cancel", &request)
        }
        ("POST", ["runs", run_id, "stop"]) => {
            control_run_response(&runtime, run_id, "stop", &request)
        }
        ("POST", ["runs", run_id, "pause"]) => {
            control_run_response(&runtime, run_id, "pause", &request)
        }
        ("POST", ["runs", run_id, "resume"]) => {
            control_run_response(&runtime, run_id, "resume", &request)
        }
        ("POST", ["runs", run_id, "checkpoints", "pin"]) => {
            pin_checkpoint_response(config, run_id, &request)
        }
        ("GET", ["runs", run_id, "checkpoints"]) => {
            result_response(list_run_checkpoints(config, run_id), 200)
        }
        ("GET", ["runs", run_id, "checkpoints", checkpoint_id, "export"]) => {
            export_checkpoint_response(config, run_id, checkpoint_id)
        }
        ("POST", ["runs", run_id, "step"]) => result_response(step_run(config, run_id), 200),
        ("GET", ["runs", run_id, "candidates"]) => {
            result_response(list_candidates(config, run_id, &query), 200)
        }
        ("GET", ["runs", run_id, "candidate-seed-rewards"]) => result_response(
            list_candidate_seed_rewards(config, run_id, None, &query),
            200,
        ),
        ("GET", ["runs", run_id, "candidates", candidate_id]) => {
            candidate_response(config, run_id, candidate_id)
        }
        ("GET", ["runs", run_id, "candidates", candidate_id, "seed-rewards"]) => result_response(
            list_candidate_seed_rewards(config, run_id, Some(candidate_id), &query),
            200,
        ),
        ("GET", ["runs", run_id, "candidates", candidate_id, "rollouts"]) => result_response(
            list_rollouts(config, run_id, Some(candidate_id), &query),
            200,
        ),
        ("GET", ["runs", run_id, "rollouts", rollout_id]) => {
            rollout_response(config, run_id, rollout_id, &query)
        }
        ("GET", ["runs", run_id, "artifacts", name]) => artifact_response(config, run_id, name),
        ("DELETE", ["runs", run_id, "artifacts"]) => drop_artifacts_response(config, run_id),
        ("GET", ["runs", run_id, "ws"]) => match WorkspaceStore::open_existing(&config.db_path)
            .and_then(|store| store.run_request_by_run_id(run_id))
        {
            Ok(Some(_)) => error_response(
                400,
                "invalid_config",
                "WebSocket upgrade headers are required for this route",
                None,
            ),
            Ok(None) => run_not_found_response(run_id),
            Err(error) => optimizer_error_response(error),
        },
        _ => error_response(
            404,
            "run_not_found",
            &format!("route not found: {} {}", request.method, path),
            None,
        ),
    }
}

fn handle_websocket_connection(
    stream: &mut TcpStream,
    runtime: GepaServiceRuntime,
    request: &HttpRequest,
    run_id: &str,
) -> Result<()> {
    match WorkspaceStore::open_existing(&runtime.config.db_path)
        .and_then(|store| store.run_request_by_run_id(run_id))
    {
        Ok(Some(_)) => {}
        Ok(None) => {
            return write_response(stream, run_not_found_response(run_id));
        }
        Err(error) => {
            return write_response(stream, optimizer_error_response(error));
        }
    }
    if !is_websocket_upgrade(request) {
        return write_response(
            stream,
            error_response(
                400,
                "invalid_config",
                "WebSocket upgrade headers are required",
                None,
            ),
        );
    }
    write_websocket_handshake(stream, request)?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|source| OptimizerError::io("gepa service websocket timeout", source))?;
    let subscribe_text = match read_websocket_frame(stream)? {
        WebSocketIncoming::Text(text) => text,
        WebSocketIncoming::Closed => return Ok(()),
        WebSocketIncoming::Empty => {
            write_websocket_json(
                stream,
                &json!({"type": "error", "error": {"code": "invalid_config", "message": "subscribe frame is required"}}),
            )?;
            return write_websocket_close(stream);
        }
    };
    let subscribe: WebSocketSubscribeFrame = serde_json::from_str(&subscribe_text)?;
    if subscribe.frame_type != "subscribe" {
        write_websocket_json(
            stream,
            &json!({"type": "error", "error": {"code": "invalid_config", "message": "first frame must be subscribe"}}),
        )?;
        return write_websocket_close(stream);
    }
    let kinds = normalize_event_kinds(&subscribe.kinds);
    write_websocket_json(
        stream,
        &json!({
            "type": "subscribed",
            "kinds": if kinds.is_empty() { vec!["*".to_string()] } else { kinds.iter().cloned().collect::<Vec<_>>() },
            "replay": subscribe.since > 0,
        }),
    )?;
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .map_err(|source| OptimizerError::io("gepa service websocket timeout", source))?;
    stream_websocket_run(stream, &runtime, run_id, &kinds, subscribe.since)
}

fn stream_websocket_run(
    stream: &mut TcpStream,
    runtime: &GepaServiceRuntime,
    run_id: &str,
    kinds: &BTreeSet<String>,
    since: u64,
) -> Result<()> {
    let mut last_sent_seq = since;
    let mut truncated_sent = false;
    loop {
        let store = WorkspaceStore::open_existing(&runtime.config.db_path)?;
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            write_websocket_json(
                stream,
                &json!({"type": "error", "error": {"code": "run_not_found", "message": format!("no Run with id {run_id:?}")}}),
            )?;
            return write_websocket_close(stream);
        };
        let events = project_run_events(&request)?;
        if let Some(first) = events.first() {
            if !truncated_sent && since > 0 && since < first.seq.saturating_sub(1) {
                write_websocket_json(
                    stream,
                    &json!({"type": "truncated", "earliest_seq": first.seq}),
                )?;
                truncated_sent = true;
            }
        }
        let prior_seq = last_sent_seq;
        let outgoing = events
            .iter()
            .filter(|event| event.seq > prior_seq && event_kind_matches(kinds, event.kind))
            .collect::<Vec<_>>();
        for event in outgoing {
            write_websocket_json(
                stream,
                &json!({
                    "type": "event",
                    "seq": event.seq,
                    "ts": event.ts,
                    "kind": event.kind,
                    "payload": event.payload,
                }),
            )?;
            last_sent_seq = event.seq;
        }
        let run = project_run(&store, &request)?;
        if is_terminal_status(project_request_status(&request.status)) {
            write_websocket_json(
                stream,
                &json!({
                    "type": "terminal",
                    "replay": since > 0,
                    "run": run,
                }),
            )?;
            return write_websocket_close(stream);
        }
        write_websocket_json(stream, &json!({"type": "status", "run": run}))?;
        match read_websocket_frame(stream)? {
            WebSocketIncoming::Text(text) => {
                if let Some(response) = handle_websocket_control(runtime, run_id, &text)? {
                    write_websocket_json(stream, &response)?;
                }
            }
            WebSocketIncoming::Closed => return Ok(()),
            WebSocketIncoming::Empty => {}
        }
    }
}

fn handle_websocket_control(
    runtime: &GepaServiceRuntime,
    run_id: &str,
    text: &str,
) -> Result<Option<Value>> {
    let frame: WebSocketControlFrame = serde_json::from_str(text)?;
    match frame.frame_type.as_str() {
        "stop" | "cancel" | "resume" => {
            let run = control_run(runtime, run_id, &frame.frame_type, None)?;
            Ok(run.map(|run| json!({"type": "status", "run": run})))
        }
        "pause" => {
            let timeout = match frame.timeout_seconds {
                Some(timeout) => validate_pause_timeout_seconds(timeout)?,
                None => 1_800,
            };
            let run = control_run(runtime, run_id, "pause", Some(timeout))?;
            Ok(run.map(|run| json!({"type": "status", "run": run})))
        }
        "step" => step_run(&runtime.config, run_id)
            .map(|value| Some(json!({"type": "status", "step": value}))),
        "subscribe" => Err(OptimizerError::Config(
            "subscribe is only valid as the first WebSocket frame".to_string(),
        )),
        other => Err(OptimizerError::Config(format!(
            "unknown WebSocket control frame type {other:?}"
        ))),
    }
}

fn create_run_response(runtime: &GepaServiceRuntime, request: &HttpRequest) -> HttpResponse {
    let idempotency_key = request.headers.get("idempotency-key").cloned();
    let body_sha256 = request_body_sha256(&request.body);
    match json_body::<GepaServiceRunRequest>(request)
        .and_then(|run_request| create_run(runtime, run_request, idempotency_key, body_sha256))
    {
        Ok((status, run)) => json_response(status, &run),
        Err(error) => optimizer_error_response(error),
    }
}

fn create_run(
    runtime: &GepaServiceRuntime,
    run_request: GepaServiceRunRequest,
    idempotency_key: Option<String>,
    request_body_sha256: String,
) -> Result<(u16, Value)> {
    let config = &runtime.config;
    let store = WorkspaceStore::open(&config.db_path)?;
    if let Some(idempotency_key) = idempotency_key.as_deref() {
        if let Some((existing, existing_body_sha256)) =
            store.run_request_by_idempotency_key(idempotency_key)?
        {
            if existing_body_sha256.as_deref() != Some(request_body_sha256.as_str()) {
                return Err(OptimizerError::Config(format!(
                    "idempotency_conflict: key {idempotency_key:?} was already used with a different body"
                )));
            }
            return project_run(&store, &existing).map(|run| (200, run));
        }
    }
    // Contract handshake FIRST: fetch the program (version-check + ingest), then
    // build the config from it so target_modules/seed_candidate are populated.
    let program = verify_container_contract(&run_request.container_url)?;
    let optimizer_config = run_request_to_optimizer_config(&run_request, &program)?;
    let request = store.submit_run_config_with_identity(
        optimizer_config,
        "http:gepa-service-v1",
        0,
        run_request.manual_step,
        Some(json!({
            "wire_contract": "gepa-service-v1",
            "container_url": run_request.container_url,
            "manual_step": run_request.manual_step,
            "idempotency_key": idempotency_key,
        })),
        idempotency_key.as_deref(),
        Some(&request_body_sha256),
    )?;
    if !request.manual_step {
        runtime.scheduler.notify();
    }
    seed_forked_cursor(&store, &request, &run_request)?;
    project_run(&store, &request).map(|run| (201, run))
}

fn seed_forked_cursor(
    service_store: &WorkspaceStore,
    request: &WorkspaceRunRequestStatus,
    run_request: &GepaServiceRunRequest,
) -> Result<()> {
    let source = if let Some(fixture_value) = &run_request.fixture {
        Some(serde_json::from_value(fixture_value.clone())?)
    } else if let Some(fork) = &run_request.fork_from {
        let Some(parent) = service_store.run_request_by_run_id(&fork.run_id)? else {
            return Err(OptimizerError::Config(format!(
                "fork parent run {} not found",
                fork.run_id
            )));
        };
        let Some(parent_store) = run_workspace_store(&parent)? else {
            return Err(OptimizerError::Config(format!(
                "fork parent run {} has no workspace yet",
                fork.run_id
            )));
        };
        let record = if let Some(checkpoint_id) = fork.checkpoint_id.as_deref() {
            parent_store
                .checkpoint_by_id(&fork.run_id, checkpoint_id)?
                .ok_or_else(|| {
                    OptimizerError::Config(format!(
                        "checkpoint {checkpoint_id} not found on run {}",
                        fork.run_id
                    ))
                })?
        } else {
            parent_store
                .latest_checkpoint(&fork.run_id, GEPA_CURSOR_CHECKPOINT_KIND)?
                .ok_or_else(|| {
                    OptimizerError::Config(format!(
                        "run {} has no gepa_cursor checkpoint to fork",
                        fork.run_id
                    ))
                })?
        };
        Some(export_cursor_fixture(&record)?)
    } else {
        None
    };
    let Some(fixture) = source else {
        return Ok(());
    };
    let mut run_store = ensure_run_workspace(service_store, request)?;
    import_cursor_fixture(&mut run_store, &request.run_id, &fixture)?;
    Ok(())
}

fn list_run_checkpoints(config: &GepaServiceConfig, run_id: &str) -> Result<Value> {
    let service_store = WorkspaceStore::open_existing(&config.db_path)?;
    let Some(request) = service_store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("no Run with id {run_id:?}")));
    };
    let Some(run_store) = run_workspace_store(&request)? else {
        return Ok(json!({"run_id": run_id, "checkpoints": []}));
    };
    let records = run_store.checkpoint_history(run_id, Some(GEPA_CURSOR_CHECKPOINT_KIND))?;
    Ok(json!({
        "run_id": run_id,
        "checkpoints": records.iter().map(|record| json!({
            "checkpoint_id": record.checkpoint_id,
            "sequence_number": record.sequence_number,
            "status": record.status,
            "run_state": record.run_state,
            "generation": record.generation,
            "retained": record.is_retained(),
            "compacted": record.is_storage_compacted(),
            "candidate_count": record.candidate_count,
            "frontier_count": record.frontier_count,
        })).collect::<Vec<_>>(),
    }))
}

fn pin_checkpoint_response(
    config: &GepaServiceConfig,
    run_id: &str,
    request: &HttpRequest,
) -> HttpResponse {
    match pin_run_checkpoint_from_request(config, run_id, request) {
        Ok(record) => json_response(
            200,
            &json!({
                "run_id": run_id,
                "checkpoint_id": record.checkpoint_id,
                "retained": true,
                "generation": record.generation,
            }),
        ),
        Err(error) => optimizer_error_response(error),
    }
}

fn pin_run_checkpoint_from_request(
    config: &GepaServiceConfig,
    run_id: &str,
    request: &HttpRequest,
) -> Result<CheckpointRecord> {
    let body: Value = if request.body.is_empty() {
        json!({})
    } else {
        serde_json::from_slice(&request.body)?
    };
    let service_store = WorkspaceStore::open_existing(&config.db_path)?;
    let Some(run_request) = service_store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("no Run with id {run_id:?}")));
    };
    let Some(mut run_store) = run_workspace_store(&run_request)? else {
        return Err(OptimizerError::Config(format!(
            "run {run_id} has no workspace yet"
        )));
    };
    let checkpoint_id = match body.get("checkpoint_id").and_then(Value::as_str) {
        Some(id) => id.to_string(),
        None => {
            run_store
                .latest_checkpoint(run_id, GEPA_CURSOR_CHECKPOINT_KIND)?
                .ok_or_else(|| OptimizerError::Config(format!("run {run_id} has no checkpoint")))?
                .checkpoint_id
        }
    };
    pin_run_checkpoint(&mut run_store, run_id, &checkpoint_id)
}

fn export_checkpoint_response(
    config: &GepaServiceConfig,
    run_id: &str,
    checkpoint_id: &str,
) -> HttpResponse {
    match export_run_checkpoint(config, run_id, checkpoint_id) {
        Ok(fixture) => json_response(200, &serde_json::to_value(fixture).unwrap_or(Value::Null)),
        Err(error) => optimizer_error_response(error),
    }
}

fn export_run_checkpoint(
    config: &GepaServiceConfig,
    run_id: &str,
    checkpoint_id: &str,
) -> Result<crate::GepaCursorFixture> {
    let service_store = WorkspaceStore::open_existing(&config.db_path)?;
    let Some(run_request) = service_store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("no Run with id {run_id:?}")));
    };
    let Some(run_store) = run_workspace_store(&run_request)? else {
        return Err(OptimizerError::Config(format!(
            "run {run_id} has no workspace yet"
        )));
    };
    let record = run_store
        .checkpoint_by_id(run_id, checkpoint_id)?
        .ok_or_else(|| OptimizerError::Config(format!("checkpoint {checkpoint_id} not found")))?;
    export_cursor_fixture(&record)
}

fn workspace_summary(config: &GepaServiceConfig) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let status = store.status()?;
    let scheduler = project_scheduler_summary(config, &status.run_requests);
    let run_status = project_run_status_summary(config, &status.run_requests)?;
    let oldest_queued_age_seconds = status
        .run_requests
        .iter()
        .filter(|request| request.status == "queued")
        .filter_map(|request| seconds_since_sqlite_timestamp(&request.submitted_at))
        .max();
    let last_progress_at = status
        .run_requests
        .iter()
        .map(|request| request.updated_at.as_str())
        .max()
        .map(str::to_string);
    Ok(json!({
        "schema_version": status.schema_version,
        "runs": {
            "total": status.run_requests.len(),
            "by_status": project_status_counts(&status.run_request_status_counts),
        },
        "liveness": {
            "running_count": status
                .run_requests
                .iter()
                .filter(|request| matches!(request.status.as_str(), "leased" | "running"))
                .count(),
            "oldest_queued_age_seconds": oldest_queued_age_seconds,
            "last_progress_at": last_progress_at,
        },
        "scheduler": scheduler,
        "run_status": run_status,
    }))
}

fn workspace_storage(config: &GepaServiceConfig) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let status = store.status()?;
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();
    for request in status.run_requests {
        let run_dir = PathBuf::from(request.run_dir);
        let Some(root) = run_dir.parent().map(Path::to_path_buf) else {
            continue;
        };
        let key = root.display().to_string();
        if seen.insert(key) {
            roots.push(root);
        }
    }
    inspect_workspace_storage_health(WorkspaceStorageHealthInput {
        roots,
        thresholds: StorageHealthThresholds::default(),
        now_unix_seconds: None,
    })
}

fn project_run_status_summary(
    config: &GepaServiceConfig,
    requests: &[WorkspaceRunRequestStatus],
) -> Result<Value> {
    let worker_count = normalize_service_worker_count(config.worker_count);
    let active = requests
        .iter()
        .filter(|request| matches!(request.status.as_str(), "leased" | "running"))
        .collect::<Vec<_>>();
    let active_workers = active.len().min(worker_count);
    let queued = requests
        .iter()
        .filter(|request| request.status == "queued")
        .collect::<Vec<_>>();
    let mut queued_reasons = Vec::new();
    let mut runnable_queued = 0usize;
    let mut blocked_queued = 0usize;
    for request in &queued {
        let block = scheduler_block_reason(request, &active, active_workers, worker_count);
        if block.is_some() {
            blocked_queued += 1;
        } else {
            runnable_queued += 1;
        }
        queued_reasons.push(json!({
            "run_id": request.run_id,
            "request_id": request.request_id,
            "submitted_at": request.submitted_at,
            "reason": block.as_ref().map(|block| block.reason.as_str()),
            "blocked_by_run_id": block.as_ref().and_then(|block| block.blocked_by_run_id.clone()),
            "blocked_by_request_id": block.as_ref().and_then(|block| block.blocked_by_request_id.clone()),
            "why_not_running": scheduler_why_not_running(block.as_ref()),
        }));
    }
    let active_leases = active
        .iter()
        .map(|request| project_active_lease(request))
        .collect::<Result<Vec<_>>>()?;
    let stale_lease_count = active_leases
        .iter()
        .filter(|lease| lease.get("stale").and_then(Value::as_bool).unwrap_or(false))
        .count();
    let worker_slots = (0..worker_count)
        .map(|slot| {
            let worker_id = service_worker_id(&config.worker_id, worker_count, slot);
            let active_request = active
                .iter()
                .find(|request| request.worker_id.as_deref() == Some(worker_id.as_str()));
            project_worker_slot(slot, worker_id, active_request.copied())
        })
        .collect::<Result<Vec<_>>>()?;
    let raw_status_counts = raw_status_counts(requests);
    let projected_status_counts = project_status_counts(&raw_status_counts);
    Ok(json!({
        "raw_status_counts": raw_status_counts,
        "projected_status_counts": projected_status_counts,
        "counts": {
            "total": requests.len(),
            "queued": raw_status_count(requests, "queued"),
            "leased": raw_status_count(requests, "leased"),
            "running": raw_status_count(requests, "running"),
            "paused": raw_status_count(requests, "paused"),
            "terminal": requests
                .iter()
                .filter(|request| is_terminal_status(project_request_status(&request.status)))
                .count(),
            "succeeded": projected_status_counts.get("succeeded").copied().unwrap_or(0),
            "failed": projected_status_counts.get("failed").copied().unwrap_or(0),
            "cancelled": projected_status_counts.get("cancelled").copied().unwrap_or(0),
            "active_workers": active_workers,
            "idle_workers": worker_count.saturating_sub(active_workers),
            "worker_count": worker_count,
            "queued_runnable": runnable_queued,
            "queued_blocked": blocked_queued,
            "stale_leases": stale_lease_count,
        },
        "active_leases": active_leases,
        "queued_reasons": queued_reasons,
        "worker_slots": worker_slots,
    }))
}

fn project_scheduler_summary(
    config: &GepaServiceConfig,
    requests: &[WorkspaceRunRequestStatus],
) -> Value {
    let worker_count = normalize_service_worker_count(config.worker_count);
    let active = requests
        .iter()
        .filter(|request| matches!(request.status.as_str(), "leased" | "running"))
        .collect::<Vec<_>>();
    let active_workers = active.len().min(worker_count);
    let queued = requests
        .iter()
        .filter(|request| request.status == "queued")
        .collect::<Vec<_>>();
    let mut runnable_queued = 0usize;
    let mut blocked_queued = 0usize;
    let mut queued_items = Vec::new();
    for request in queued {
        let block = scheduler_block_reason(request, &active, active_workers, worker_count);
        if block.is_some() {
            blocked_queued += 1;
        } else {
            runnable_queued += 1;
        }
        queued_items.push(json!({
            "run_id": request.run_id,
            "request_id": request.request_id,
            "submitted_at": request.submitted_at,
            "reason": block.as_ref().map(|block| block.reason.as_str()),
            "blocked_by_run_id": block.as_ref().and_then(|block| block.blocked_by_run_id.clone()),
            "blocked_by_request_id": block.as_ref().and_then(|block| block.blocked_by_request_id.clone()),
        }));
    }
    let workers = (0..worker_count)
        .map(|slot| {
            let worker_id = service_worker_id(&config.worker_id, worker_count, slot);
            let active_request = active
                .iter()
                .find(|request| request.worker_id.as_deref() == Some(worker_id.as_str()));
            let heartbeat = active_request.map(|request| heartbeat_projection(request));
            json!({
                "slot": slot,
                "worker_id": worker_id,
                "state": if active_request.is_some() { "active" } else { "idle" },
                "run_id": active_request.map(|request| request.run_id.clone()),
                "request_id": active_request.map(|request| request.request_id.clone()),
                "request_status": active_request.map(|request| request.status.clone()),
                "lease_expires_at": active_request.and_then(|request| request.lease_expires_at.clone()),
                "seconds_until_lease_expiry": heartbeat.as_ref().and_then(|heartbeat| heartbeat.seconds_until_lease_expiry),
                "last_heartbeat_at": active_request.and_then(|request| request.lease_expires_at.clone()),
                "last_worker_progress_at": active_request.map(|request| request.updated_at.clone()),
                "last_progress_at": active_request.map(|request| request.updated_at.clone()),
                "heartbeat_state": heartbeat.as_ref().map(|heartbeat| heartbeat.state),
                "heartbeat_detail": heartbeat.as_ref().map(|heartbeat| heartbeat.detail),
                "stale": heartbeat.as_ref().map(|heartbeat| heartbeat.stale),
                "stale_reason": heartbeat.as_ref().and_then(|heartbeat| heartbeat.stale_reason),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "worker_count": worker_count,
        "active_workers": active_workers,
        "idle_workers": worker_count.saturating_sub(active_workers),
        "queued_runnable": runnable_queued,
        "queued_blocked": blocked_queued,
        "workers": workers,
        "queued": queued_items,
    })
}

struct SchedulerBlockReason {
    reason: String,
    blocked_by_run_id: Option<String>,
    blocked_by_request_id: Option<String>,
}

fn scheduler_block_reason(
    queued: &WorkspaceRunRequestStatus,
    active: &[&WorkspaceRunRequestStatus],
    active_workers: usize,
    worker_count: usize,
) -> Option<SchedulerBlockReason> {
    if queued.manual_step {
        return Some(SchedulerBlockReason {
            reason: "manual_step".to_string(),
            blocked_by_run_id: None,
            blocked_by_request_id: None,
        });
    }
    if let Some(blocker) = active.iter().find(|request| {
        !request.container_url.is_empty() && request.container_url == queued.container_url
    }) {
        return Some(SchedulerBlockReason {
            reason: "container_exclusive_conflict".to_string(),
            blocked_by_run_id: Some(blocker.run_id.clone()),
            blocked_by_request_id: Some(blocker.request_id.clone()),
        });
    }
    if let Some(blocker) = active.iter().find(|request| {
        !request.cache_namespace.is_empty() && request.cache_namespace == queued.cache_namespace
    }) {
        return Some(SchedulerBlockReason {
            reason: "cache_namespace_conflict".to_string(),
            blocked_by_run_id: Some(blocker.run_id.clone()),
            blocked_by_request_id: Some(blocker.request_id.clone()),
        });
    }
    if active_workers >= worker_count {
        return Some(SchedulerBlockReason {
            reason: "worker_capacity".to_string(),
            blocked_by_run_id: None,
            blocked_by_request_id: None,
        });
    }
    None
}

fn scheduler_why_not_running(block: Option<&SchedulerBlockReason>) -> &'static str {
    match block.map(|block| block.reason.as_str()) {
        None => "runnable; waiting for worker tick",
        Some("manual_step") => "manual step required",
        Some("container_exclusive_conflict") => "waiting for active run using same container",
        Some("cache_namespace_conflict") => "waiting for active run using same cache namespace",
        Some("worker_capacity") => "waiting for an idle service worker",
        Some(_) => "blocked by scheduler policy",
    }
}

fn raw_status_counts(requests: &[WorkspaceRunRequestStatus]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for request in requests {
        *counts.entry(request.status.clone()).or_insert(0) += 1;
    }
    counts
}

fn raw_status_count(requests: &[WorkspaceRunRequestStatus], status: &str) -> usize {
    requests
        .iter()
        .filter(|request| request.status == status)
        .count()
}

fn project_worker_slot(
    slot: usize,
    worker_id: String,
    active_request: Option<&WorkspaceRunRequestStatus>,
) -> Result<Value> {
    let Some(request) = active_request else {
        return Ok(json!({
            "slot": slot,
            "worker_id": worker_id,
            "state": "idle",
            "heartbeat_state": "idle",
            "stale": false,
        }));
    };
    let heartbeat = heartbeat_projection(request);
    Ok(json!({
        "slot": slot,
        "worker_id": worker_id,
        "state": "active",
        "run_id": request.run_id,
        "request_id": request.request_id,
        "request_status": request.status,
        "lease_id": request.lease_id,
        "leased_at": request.leased_at,
        "lease_expires_at": request.lease_expires_at,
        "seconds_until_lease_expiry": heartbeat.seconds_until_lease_expiry,
        "last_worker_progress_at": request.updated_at,
        "last_progress_at": request.updated_at,
        "last_run_event_at": latest_run_event_at(request)?,
        "heartbeat_state": heartbeat.state,
        "heartbeat_detail": heartbeat.detail,
        "stale": heartbeat.stale,
        "stale_reason": heartbeat.stale_reason,
    }))
}

fn project_active_lease(request: &WorkspaceRunRequestStatus) -> Result<Value> {
    let heartbeat = heartbeat_projection(request);
    Ok(json!({
        "run_id": request.run_id,
        "request_id": request.request_id,
        "worker_id": request.worker_id,
        "request_status": request.status,
        "lease_id": request.lease_id,
        "leased_at": request.leased_at,
        "lease_expires_at": request.lease_expires_at,
        "seconds_until_lease_expiry": heartbeat.seconds_until_lease_expiry,
        "last_worker_progress_at": request.updated_at,
        "last_run_event_at": latest_run_event_at(request)?,
        "heartbeat_state": heartbeat.state,
        "heartbeat_detail": heartbeat.detail,
        "stale": heartbeat.stale,
        "stale_reason": heartbeat.stale_reason,
    }))
}

struct HeartbeatProjection {
    state: &'static str,
    detail: &'static str,
    seconds_until_lease_expiry: Option<i64>,
    stale: bool,
    stale_reason: Option<&'static str>,
}

fn heartbeat_projection(request: &WorkspaceRunRequestStatus) -> HeartbeatProjection {
    if !matches!(request.status.as_str(), "leased" | "running") {
        return HeartbeatProjection {
            state: "inactive",
            detail: "request is not leased or running",
            seconds_until_lease_expiry: None,
            stale: false,
            stale_reason: None,
        };
    }
    let Some(lease_expires_at) = request.lease_expires_at.as_deref() else {
        return HeartbeatProjection {
            state: "unknown",
            detail: "active request has no lease expiry",
            seconds_until_lease_expiry: None,
            stale: true,
            stale_reason: Some("missing_lease_expiry"),
        };
    };
    let Some(seconds_until_lease_expiry) = seconds_until_sqlite_timestamp(lease_expires_at) else {
        return HeartbeatProjection {
            state: "unknown",
            detail: "lease expiry timestamp could not be parsed",
            seconds_until_lease_expiry: None,
            stale: true,
            stale_reason: Some("invalid_lease_expiry"),
        };
    };
    if seconds_until_lease_expiry <= 0 {
        return HeartbeatProjection {
            state: "stale",
            detail: "active lease expired",
            seconds_until_lease_expiry: Some(seconds_until_lease_expiry),
            stale: true,
            stale_reason: Some("lease_expired"),
        };
    }
    HeartbeatProjection {
        state: "heartbeating",
        detail: "active lease has not expired",
        seconds_until_lease_expiry: Some(seconds_until_lease_expiry),
        stale: false,
        stale_reason: None,
    }
}

fn latest_run_event_at(request: &WorkspaceRunRequestStatus) -> Result<Option<String>> {
    let Some(run_store) = run_workspace_store(request)? else {
        return Ok(None);
    };
    Ok(run_store
        .event_stream_history(&request.run_id)?
        .last()
        .map(|record| record.timestamp.clone()))
}

fn workspace_usage(config: &GepaServiceConfig) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let mut runs = Vec::new();
    let mut total_bytes = 0u64;
    for request in store.status()?.run_requests {
        let bytes = directory_bytes(Path::new(&request.run_dir))?;
        total_bytes = total_bytes.saturating_add(bytes);
        runs.push(json!({
            "run_id": request.run_id,
            "bytes": bytes,
        }));
    }
    Ok(json!({
        "total_bytes": total_bytes,
        "runs": runs,
    }))
}

fn prune_workspace(config: &GepaServiceConfig, request: &HttpRequest) -> Result<Value> {
    let body = if request.body.is_empty() {
        json!({})
    } else {
        serde_json::from_slice::<Value>(&request.body)?
    };
    let dry_run = body.get("dry_run").and_then(Value::as_bool).unwrap_or(true);
    let statuses = body.get("status").and_then(Value::as_array).map(|items| {
        items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
    });
    let store = WorkspaceStore::open(&config.db_path)?;
    let mut run_ids = Vec::new();
    for request in store.status()?.run_requests {
        let projected_status = project_request_status(&request.status);
        if !is_terminal_status(projected_status) {
            continue;
        }
        if let Some(statuses) = statuses.as_ref() {
            if !statuses.contains(projected_status) {
                continue;
            }
        }
        run_ids.push(request.run_id.clone());
        if !dry_run {
            let _ = fs::remove_dir_all(&request.run_dir);
            let _ = store.delete_run_request_by_run_id(&request.run_id)?;
        }
    }
    Ok(json!({
        "run_ids": run_ids,
        "dry_run": dry_run,
    }))
}

#[derive(Clone, Debug, Deserialize)]
struct StorageMaintenanceRequest {
    #[serde(default = "default_true")]
    dry_run: bool,
    #[serde(default = "default_compact_profile")]
    profile: String,
    #[serde(default)]
    status: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

fn default_compact_profile() -> String {
    "compact".to_string()
}

fn compact_workspace_response(config: &GepaServiceConfig, request: &HttpRequest) -> Result<Value> {
    let body = storage_maintenance_request(request)?;
    let profile = StorageMaintenanceProfile::parse(&body.profile)?;
    let statuses = body
        .status
        .map(|items| items.into_iter().collect::<BTreeSet<_>>());
    let store = WorkspaceStore::open(&config.db_path)?;
    let mut runs = Vec::new();
    for request in store.status()?.run_requests {
        let projected_status = project_request_status(&request.status);
        if !is_terminal_status(projected_status) {
            continue;
        }
        if let Some(statuses) = statuses.as_ref() {
            if !statuses.contains(projected_status) {
                continue;
            }
        }
        runs.push(compact_run_request_storage(
            &request,
            profile,
            body.dry_run,
        )?);
    }
    Ok(json!({
        "schema": "synth.gepa.workspace_compaction.v1",
        "dry_run": body.dry_run,
        "profile": profile.as_str(),
        "runs": runs,
    }))
}

fn compact_run_response(
    config: &GepaServiceConfig,
    run_id: &str,
    http_request: &HttpRequest,
) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        if !is_terminal_status(project_request_status(&request.status)) {
            return Err(OptimizerError::Config(format!(
                "run {run_id} is not terminal"
            )));
        }
        let body = storage_maintenance_request(http_request)?;
        let profile = StorageMaintenanceProfile::parse(&body.profile)?;
        compact_run_request_storage(&request, profile, body.dry_run).map(Some)
    }) {
        Ok(Some(report)) => json_response(200, &report),
        Ok(None) => run_not_found_response(run_id),
        Err(OptimizerError::Config(message)) => error_response(409, "not_terminal", &message, None),
        Err(error) => optimizer_error_response(error),
    }
}

fn compact_run_request_storage(
    request: &WorkspaceRunRequestStatus,
    profile: StorageMaintenanceProfile,
    dry_run: bool,
) -> Result<Value> {
    compact_run_storage(RunStorageMaintenanceInput {
        run_dir: PathBuf::from(&request.run_dir),
        run_id: Some(request.run_id.clone()),
        profile,
        dry_run,
    })
}

fn storage_maintenance_request(request: &HttpRequest) -> Result<StorageMaintenanceRequest> {
    if request.body.is_empty() {
        return Ok(StorageMaintenanceRequest {
            dry_run: true,
            profile: default_compact_profile(),
            status: None,
        });
    }
    serde_json::from_slice(&request.body).map_err(OptimizerError::from)
}

fn list_runs(config: &GepaServiceConfig, query: &BTreeMap<String, String>) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let requested_status = query.get("status").map(String::as_str);
    let since = query.get("since").map(String::as_str);
    let until = query.get("until").map(String::as_str);
    let mut items = Vec::new();
    for request in store.status()?.run_requests {
        if requested_status.is_some_and(|status| project_request_status(&request.status) != status)
        {
            continue;
        }
        if since.is_some_and(|since| request.submitted_at.as_str() < since) {
            continue;
        }
        if until.is_some_and(|until| request.submitted_at.as_str() > until) {
            continue;
        }
        items.push(project_run(&store, &request)?);
    }
    items.sort_by(|left, right| {
        value_string(left, "submitted_at")
            .cmp(&value_string(right, "submitted_at"))
            .then_with(|| value_string(left, "run_id").cmp(&value_string(right, "run_id")))
    });
    Ok(paginate(items, query))
}

fn run_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let request = store.run_request_by_run_id(run_id)?;
        request
            .map(|request| project_run(&store, &request))
            .transpose()
    }) {
        Ok(Some(run)) => json_response(200, &run),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn run_state_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let request = store.run_request_by_run_id(run_id)?;
        request
            .map(|request| project_run_state(&store, &request))
            .transpose()
    }) {
        Ok(Some(state)) => json_response(200, &state),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn run_limits_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let config = store.run_request_config(&request.request_id)?;
        let Some(run_store) = run_workspace_store(&request)? else {
            return Ok(None);
        };
        Ok(Some(project_gepa_limit_snapshot(&run_store, &config)?))
    }) {
        Ok(Some(snapshot)) => match serde_json::to_value(snapshot) {
            Ok(value) => json_response(200, &value),
            Err(error) => optimizer_error_response(OptimizerError::from(error)),
        },
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn run_timings_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let timings = run_phase_timings_for_request(&request)?;
        Ok(Some(json!({
            "run_id": run_id,
            "summary": timing_summary_from_records(&timings),
            "timings": timings
                .iter()
                .map(project_timing_record)
                .collect::<Vec<_>>(),
        })))
    }) {
        Ok(Some(timings)) => json_response(200, &timings),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn run_stats_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        project_run_transition_stats(&request).map(Some)
    }) {
        Ok(Some(stats)) => json_response(200, &stats),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn run_storage_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let terminal = is_terminal_status(project_request_status(&request.status));
        inspect_run_storage(RunStorageInspectionInput {
            run_dir: PathBuf::from(&request.run_dir),
            run_id: Some(request.run_id.clone()),
            terminal: Some(terminal),
        })
        .map(Some)
    }) {
        Ok(Some(report)) => json_response(200, &report),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn delete_run_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        if !is_terminal_status(project_request_status(&request.status)) {
            return Err(OptimizerError::Config(format!(
                "run {run_id} is not terminal"
            )));
        }
        let _ = delete_run_storage(&request.run_dir, false)?;
        let _ = store.delete_run_request_by_run_id(run_id)?;
        Ok(Some(()))
    }) {
        Ok(_) => empty_response(204),
        Err(OptimizerError::Config(message)) => error_response(409, "not_terminal", &message, None),
        Err(error) => optimizer_error_response(error),
    }
}

fn control_run_response(
    runtime: &GepaServiceRuntime,
    run_id: &str,
    action: &str,
    http_request: &HttpRequest,
) -> HttpResponse {
    let pause_timeout = match action {
        "pause" => match validate_pause_request(http_request) {
            Ok(timeout) => Some(timeout),
            Err(error) => return optimizer_error_response(error),
        },
        _ => None,
    };
    match control_run(runtime, run_id, action, pause_timeout) {
        Ok(Some(run)) => json_response(202, &run),
        Ok(None) => run_not_found_response(run_id),
        Err(OptimizerError::Config(message)) => {
            error_response(409, "invalid_transition", &message, None)
        }
        Err(error) => optimizer_error_response(error),
    }
}

fn control_run(
    runtime: &GepaServiceRuntime,
    run_id: &str,
    action: &str,
    pause_timeout_seconds: Option<u64>,
) -> Result<Option<Value>> {
    let config = &runtime.config;
    WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let wire_status = project_request_status(&request.status);
        if is_terminal_status(wire_status) {
            return project_run(&store, &request).map(Some);
        }
        match action {
            "cancel" => {
                let request =
                    store.mark_run_request_cancelled(&request.request_id, "cancelled_by_user")?;
                runtime.scheduler.notify();
                project_run(&store, &request).map(Some)
            }
            "stop" => {
                persist_external_stop_result(&store, &request)?;
                let request = store.mark_run_request_completed(&request.request_id)?;
                project_run(&store, &request).map(Some)
            }
            "pause" => {
                if request.manual_step {
                    return Err(OptimizerError::Config(format!(
                        "run {run_id} is manual_step and cannot be paused"
                    )));
                }
                let timeout_seconds = pause_timeout_seconds.unwrap_or(1_800);
                if request.status != "paused" {
                    persist_paused_cursor(&request)?;
                }
                let request = store.mark_run_request_paused(
                    &request.request_id,
                    "pause_requested",
                    timeout_seconds,
                )?;
                project_run(&store, &request).map(Some)
            }
            "resume" => {
                if request.status != "paused" {
                    return Err(OptimizerError::Config(format!(
                        "run {run_id} is not paused"
                    )));
                }
                restore_paused_cursor(&request)?;
                let request = store.mark_run_request_resumed(&request.request_id)?;
                runtime.scheduler.notify();
                project_run(&store, &request).map(Some)
            }
            _ => Err(OptimizerError::Invariant(format!(
                "unknown run control action {action:?}"
            ))),
        }
    })
}

fn validate_pause_request(request: &HttpRequest) -> Result<u64> {
    if request.body.is_empty() {
        return Ok(1_800);
    }
    let body: Value = serde_json::from_slice(&request.body)?;
    if !body.is_object() {
        return Err(OptimizerError::Config(
            "pause request body must be a JSON object".to_string(),
        ));
    }
    let Some(timeout_value) = body.get("timeout_seconds") else {
        return Ok(1_800);
    };
    let timeout_seconds = timeout_value.as_u64().ok_or_else(|| {
        OptimizerError::Config("pause.timeout_seconds must be an integer".to_string())
    })?;
    validate_pause_timeout_seconds(timeout_seconds)
}

fn validate_pause_timeout_seconds(timeout_seconds: u64) -> Result<u64> {
    if timeout_seconds == 0 || timeout_seconds > 14_400 {
        return Err(OptimizerError::Config(
            "pause.timeout_seconds must be between 1 and 14400".to_string(),
        ));
    }
    Ok(timeout_seconds)
}

fn persist_external_stop_result(
    store: &WorkspaceStore,
    request: &WorkspaceRunRequestStatus,
) -> Result<()> {
    let result = json!({
        "stopped_by": {"kind": "external_signal"},
        "best_candidate": request.best_candidate_id.as_ref().map(|candidate_id| json!({
            "candidate_id": candidate_id,
        })),
        "cost_usd": request.cost_usd.unwrap_or(0.0),
        "usage": request.usage.clone(),
    });
    store.record_run_request_result(&request.request_id, &result)
}

fn persist_paused_cursor(request: &WorkspaceRunRequestStatus) -> Result<()> {
    let Some(mut run_store) = run_workspace_store(request)? else {
        return Ok(());
    };
    let mut cursor = load_gepa_cursor(&run_store, &request.run_id)?
        .unwrap_or_else(|| GepaCursor::new(request.run_id.clone()));
    let previous_phase = cursor.phase.as_str().to_string();
    cursor.phase = GepaCursorPhase::Paused;
    cursor.metadata = merge_json_object(
        cursor.metadata,
        json!({
            "paused": true,
            "previous_phase": previous_phase,
        }),
    );
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(&run_store, &request.run_id)?;
    persist_cursor_checkpoint(&mut run_store, &cursor, "paused", "run paused")
}

fn restore_paused_cursor(request: &WorkspaceRunRequestStatus) -> Result<()> {
    let Some(mut run_store) = run_workspace_store(request)? else {
        return Ok(());
    };
    let Some(mut cursor) = load_gepa_cursor(&run_store, &request.run_id)? else {
        return Ok(());
    };
    if !matches!(cursor.phase, GepaCursorPhase::Paused) {
        return Ok(());
    }
    let restored_phase = cursor
        .metadata
        .get("previous_phase")
        .and_then(Value::as_str)
        .and_then(gepa_cursor_phase_from_str)
        .unwrap_or(GepaCursorPhase::GenerationStart);
    cursor.phase = restored_phase;
    cursor.metadata = merge_json_object(
        cursor.metadata,
        json!({
            "paused": false,
            "resumed": true,
        }),
    );
    cursor.checkpoint_sequence = next_cursor_checkpoint_sequence(&run_store, &request.run_id)?;
    persist_cursor_checkpoint(&mut run_store, &cursor, "running", "run resumed")
}

fn gepa_cursor_phase_from_str(value: &str) -> Option<GepaCursorPhase> {
    match value {
        "initializing" => Some(GepaCursorPhase::Initializing),
        "seed_full_train" => Some(GepaCursorPhase::SeedFullTrain),
        "generation_start" => Some(GepaCursorPhase::GenerationStart),
        "proposer_waiting" => Some(GepaCursorPhase::ProposerWaiting),
        "candidate_minibatch" => Some(GepaCursorPhase::CandidateMinibatch),
        "candidate_full_train" => Some(GepaCursorPhase::CandidateFullTrain),
        "heldout" => Some(GepaCursorPhase::Heldout),
        "finalizing" => Some(GepaCursorPhase::Finalizing),
        "paused" => Some(GepaCursorPhase::Paused),
        "completed" => Some(GepaCursorPhase::Completed),
        "failed" => Some(GepaCursorPhase::Failed),
        "cancelled" => Some(GepaCursorPhase::Cancelled),
        _ => None,
    }
}

fn merge_json_object(base: Value, update: Value) -> Value {
    let mut merged = base.as_object().cloned().unwrap_or_default();
    if let Some(update) = update.as_object() {
        for (key, value) in update {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

fn step_run(config: &GepaServiceConfig, run_id: &str) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let Some(request) = store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("run not found: {run_id}")));
    };
    if !request.manual_step {
        return Err(OptimizerError::Config(format!(
            "run {run_id} was not created with manual_step=true"
        )));
    }
    if is_terminal_status(project_request_status(&request.status)) {
        return Ok(json!({
            "run_id": run_id,
            "cursor_phase": terminal_cursor_phase(&request.status),
            "terminal": true,
            "events": [],
        }));
    }
    let before_seq = project_run_events(&request)?
        .last()
        .map(|event| event.seq)
        .unwrap_or(0);
    let outcome = tick_next_unit(&config.db_path, &config.worker_id, config.lease_seconds)?;
    let refreshed = WorkspaceStore::open_existing(&config.db_path)?
        .run_request_by_run_id(run_id)?
        .ok_or_else(|| OptimizerError::Config(format!("run not found: {run_id}")))?;
    let events = project_run_events(&refreshed)?
        .into_iter()
        .filter(|event| event.seq > before_seq)
        .map(|event| {
            json!({
                "type": "event",
                "seq": event.seq,
                "ts": event.ts,
                "kind": event.kind,
                "payload": event.payload,
            })
        })
        .collect::<Vec<_>>();
    let cursor_phase = latest_cursor_phase(&refreshed)?.unwrap_or_else(|| match &outcome.action {
        GepaTickAction::ClaimRunRequest { .. } => "initializing".to_string(),
        GepaTickAction::StartRunRequest { .. } => "initializing".to_string(),
        GepaTickAction::TerminalizeRun { status, .. } => terminal_cursor_phase(status).to_string(),
        _ => "generation_start".to_string(),
    });
    Ok(json!({
        "run_id": run_id,
        "cursor_phase": cursor_phase,
        "terminal": outcome.terminal || is_terminal_status(project_request_status(&refreshed.status)),
        "events": events,
    }))
}

fn list_candidates(
    config: &GepaServiceConfig,
    run_id: &str,
    query: &BTreeMap<String, String>,
) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let Some(request) = store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("run not found: {run_id}")));
    };
    let mut items = run_workspace_store(&request)?
        .map(|run_store| {
            run_store.view().candidate_records(run_id).map(|records| {
                records
                    .into_iter()
                    .map(|record| project_candidate(run_id, record))
                    .collect::<Vec<_>>()
            })
        })
        .transpose()?
        .unwrap_or_default();
    if let Some(accepted) = query_bool(query, "accepted") {
        items.retain(|item| item.get("accepted").and_then(Value::as_bool) == Some(accepted));
    }
    if let Some(generation) = query
        .get("generation")
        .and_then(|value| value.parse::<u64>().ok())
    {
        items.retain(|item| item.get("generation").and_then(Value::as_u64) == Some(generation));
    }
    sort_candidates(&mut items, query.get("sort").map(String::as_str))?;
    Ok(paginate(items, query))
}

fn list_candidate_seed_rewards(
    config: &GepaServiceConfig,
    run_id: &str,
    candidate_id: Option<&str>,
    query: &BTreeMap<String, String>,
) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let Some(request) = store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("run not found: {run_id}")));
    };
    let mut items = run_workspace_store(&request)?
        .map(|run_store| run_store.view().candidate_seed_reward_records(run_id))
        .transpose()?
        .unwrap_or_default();
    if let Some(candidate_id) = candidate_id {
        items.retain(|item| item.get("candidate_id").and_then(Value::as_str) == Some(candidate_id));
    }
    if let Some(split) = query.get("split") {
        items.retain(|item| item.get("split").and_then(Value::as_str) == Some(split.as_str()));
    }
    if let Some(stage) = query
        .get("evaluation_stage")
        .or_else(|| query.get("stage"))
        .map(String::as_str)
    {
        items.retain(|item| item.get("evaluation_stage").and_then(Value::as_str) == Some(stage));
    }
    if let Some(seed_id) = query
        .get("seed_id")
        .or_else(|| query.get("task_id"))
        .map(String::as_str)
    {
        items.retain(|item| {
            item.get("seed_id").and_then(Value::as_str) == Some(seed_id)
                || item.get("task_id").and_then(Value::as_str) == Some(seed_id)
        });
    }
    items.sort_by(|left, right| {
        value_string(left, "candidate_id")
            .cmp(&value_string(right, "candidate_id"))
            .then_with(|| value_string(left, "split").cmp(&value_string(right, "split")))
            .then_with(|| {
                value_string(left, "evaluation_stage").cmp(&value_string(right, "evaluation_stage"))
            })
            .then_with(|| value_string(left, "seed_id").cmp(&value_string(right, "seed_id")))
    });
    Ok(paginate(items, query))
}

fn candidate_response(
    config: &GepaServiceConfig,
    run_id: &str,
    candidate_id: &str,
) -> HttpResponse {
    match list_candidates(config, run_id, &BTreeMap::new()).map(|page| {
        page.get("items")
            .and_then(Value::as_array)
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| {
                        item.get("candidate_id").and_then(Value::as_str) == Some(candidate_id)
                    })
                    .cloned()
            })
    }) {
        Ok(Some(candidate)) => json_response(200, &candidate),
        Ok(None) => run_not_found_response(run_id),
        Err(OptimizerError::Config(message)) if message.starts_with("run not found:") => {
            run_not_found_response(run_id)
        }
        Err(error) => optimizer_error_response(error),
    }
}

fn list_rollouts(
    config: &GepaServiceConfig,
    run_id: &str,
    candidate_id: Option<&str>,
    query: &BTreeMap<String, String>,
) -> Result<Value> {
    let store = WorkspaceStore::open(&config.db_path)?;
    let Some(request) = store.run_request_by_run_id(run_id)? else {
        return Err(OptimizerError::Config(format!("run not found: {run_id}")));
    };
    let mut items = run_workspace_store(&request)?
        .map(|run_store| {
            run_store.view().rollout_records(run_id).map(|records| {
                records
                    .into_iter()
                    .filter(|record| candidate_id.is_none_or(|id| record.candidate_id == id))
                    .map(|record| project_rollout(&record, None))
                    .collect::<Vec<_>>()
            })
        })
        .transpose()?
        .unwrap_or_default();
    items.sort_by(|left, right| {
        value_string(left, "candidate_id")
            .cmp(&value_string(right, "candidate_id"))
            .then_with(|| value_string(left, "task_id").cmp(&value_string(right, "task_id")))
    });
    Ok(paginate(items, query))
}

fn rollout_response(
    config: &GepaServiceConfig,
    run_id: &str,
    rollout_id: &str,
    query: &BTreeMap<String, String>,
) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let include_trajectory = query
            .get("include")
            .is_some_and(|value| value == "trajectory");
        let Some(run_store) = run_workspace_store(&request)? else {
            return Ok(None);
        };
        let rollout = run_store
            .view()
            .rollout_records(run_id)?
            .into_iter()
            .find(|record| {
                record.rollout_record_id == rollout_id
                    || record.rollout_id.as_deref() == Some(rollout_id)
            });
        let Some(rollout) = rollout else {
            return Ok(None);
        };
        let trajectory = if include_trajectory {
            Some(serde_json::to_value(
                run_store
                    .view()
                    .rollout_event_records_for_rollout(run_id, rollout_id)?,
            )?)
        } else {
            None
        };
        Ok(Some(project_rollout(&rollout, trajectory)))
    }) {
        Ok(Some(rollout)) => json_response(200, &rollout),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn artifact_response(config: &GepaServiceConfig, run_id: &str, name: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(None);
        };
        let Some(path) = artifact_path_for_name(&request, name)? else {
            return Ok(None);
        };
        let body = fs::read(&path).map_err(|source| OptimizerError::io(&path, source))?;
        let content_type = path
            .file_name()
            .and_then(|file| file.to_str())
            .map(artifact_content_type)
            .unwrap_or_else(|| artifact_content_type(name));
        Ok(Some((body, content_type)))
    }) {
        Ok(Some((body, content_type))) => binary_response(200, content_type, body),
        Ok(None) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn drop_artifacts_response(config: &GepaServiceConfig, run_id: &str) -> HttpResponse {
    match WorkspaceStore::open(&config.db_path).and_then(|store| {
        let Some(request) = store.run_request_by_run_id(run_id)? else {
            return Ok(false);
        };
        if let Some(run_store) = run_workspace_store(&request)? {
            for artifact in run_store.view().artifact_refs(run_id)? {
                if !is_public_artifact_kind(&artifact.kind) {
                    continue;
                }
                let path = PathBuf::from(artifact.path);
                let _ = fs::remove_file(path);
            }
        }
        Ok(true)
    }) {
        Ok(true) => empty_response(204),
        Ok(false) => run_not_found_response(run_id),
        Err(error) => optimizer_error_response(error),
    }
}

fn verify_container_contract(container_url: &str) -> Result<PromptProgram> {
    let client = ContainerClient::new(container_url)?;
    client.verify_gepa_contract()?;
    client.program_typed()
}

/// OpenAI-compatible base URL for a provider when the request omits one.
fn provider_default_base_url(provider: &str) -> &'static str {
    match normalize_key(provider).as_str() {
        "openrouter" => "https://openrouter.ai/api/v1",
        "anthropic" => "https://api.anthropic.com/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "together" => "https://api.together.xyz/v1",
        "fireworks" => "https://api.fireworks.ai/inference/v1",
        _ => "https://api.openai.com/v1",
    }
}

/// Endpoint suffix for the OpenAI-compatible API family.
fn api_family_suffix(api_family: &str) -> &'static str {
    match normalize_key(api_family).as_str() {
        "responses" => "responses",
        _ => "chat/completions",
    }
}

fn run_request_to_optimizer_config(
    request: &GepaServiceRunRequest,
    program: &PromptProgram,
) -> Result<SynthOptimizerConfig> {
    validate_provider("policy.provider", &request.policy.provider)?;
    validate_provider("proposer.provider", &request.proposer.provider)?;
    validate_api_family("policy.api_family", &request.policy.api_family)?;
    validate_api_family("proposer.api_family", &request.proposer.api_family)?;
    validate_taskset(&request.taskset)?;
    validate_request_task_pools(
        &request.task_pools,
        &request.taskset.train_ids,
        &request.taskset.heldout_ids,
    )?;
    let mut config = SynthOptimizerConfig::default();
    if let Some(output_dir) = request
        .output_dir
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        config.run.output_dir = PathBuf::from(output_dir);
    }
    config.container.url = Some(request.container_url.clone());
    config.taskset.train_ids = request.taskset.train_ids.clone();
    config.taskset.heldout_ids = request.taskset.heldout_ids.clone();
    config.gepa.task_pools = request.task_pools.clone();
    config.policy.provider = request.policy.provider.clone();
    config.policy.model = request.policy.model.clone();
    config.policy.api_family = request.policy.api_family.clone();
    // Resolve base_url + always-populate inference_url (byok credential handling unchanged).
    let base_url = request
        .policy
        .base_url
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| provider_default_base_url(&request.policy.provider).to_string());
    let inference_url = request
        .policy
        .inference_url
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "{}/{}",
                base_url.trim_end_matches('/'),
                api_family_suffix(&request.policy.api_family)
            )
        });
    config.policy.base_url = Some(base_url);
    config.policy.inference_url = Some(inference_url);
    config.policy.max_tokens = request.policy.max_tokens;
    config.policy.disable_reasoning = request.policy.disable_reasoning.clone();
    apply_policy_credentials(&mut config, &request.policy.credentials)?;
    config.proposer.provider = request.proposer.provider.clone();
    config.proposer.model = Some(request.proposer.model.clone());
    if let Some(allow_unverified) = request.proposer.allow_unverified_model {
        config.proposer.allow_unverified_model = allow_unverified;
    }
    config.proposer.model_context_window = request.proposer.model_context_window;
    config.proposer.model_auto_compact_token_limit =
        request.proposer.model_auto_compact_token_limit;
    config.proposer.api_family = request.proposer.api_family.clone();
    config.proposer.base_url = request
        .proposer
        .base_url
        .clone()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            if normalize_key(&request.proposer.provider) == "openai" {
                None
            } else {
                Some(provider_default_base_url(&request.proposer.provider).to_string())
            }
        });
    apply_service_proposer_auth(&mut config, &request.proposer)?;
    // Headless codex proposer config (mirrors the proven go-ex proposer in
    // synth-go-ex/core/proposers.py): never wait for approvals (the app-server has
    // no one to answer them → would hang to the turn timeout), workspace-write
    // sandbox so the agent can write the proposal manifest, bounded reasoning + a
    // generous turn timeout (go-ex uses 300s; default 120s is too short).
    config.proposer.approval_policy = Some("never".to_string());
    config.proposer.sandbox_mode = Some("workspace-write".to_string());
    if let Some(effort) = request
        .proposer
        .reasoning_effort
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        config.proposer.reasoning_effort = Some(normalize_key(effort));
    } else if config.proposer.reasoning_effort.is_none() {
        config.proposer.reasoning_effort = Some("medium".to_string());
    }
    config.proposer.timeout_seconds = 300;
    apply_stop_conditions(&mut config, &request.stop_conditions)?;
    apply_advanced_config(&mut config, &request.advanced)?;
    normalize_service_rollout_budgets(&mut config, &request.taskset)?;
    // Ingest the container's prompt program: target fields + seed candidate.
    // target_modules must be the MUTABLE candidate fields (see PromptProgram::validate_for_gepa).
    let mutable_fields: BTreeSet<String> = program
        .modules
        .iter()
        .filter(|module| module.mutable && !module.candidate_field.is_empty())
        .map(|module| module.candidate_field.clone())
        .collect();
    config.candidate.target_modules = program
        .target_modules
        .iter()
        .map(|target| {
            if target.candidate_field.is_empty() {
                target.module_id.clone()
            } else {
                target.candidate_field.clone()
            }
        })
        .filter(|field| mutable_fields.contains(field))
        .collect();
    if config.candidate.target_modules.is_empty() {
        // Fall back to all mutable fields if target_modules didn't resolve to any.
        config.candidate.target_modules = mutable_fields.into_iter().collect();
    }
    crate::operator_workspace::retain_target_modules_for_levers(
        &mut config.candidate.target_modules,
        &config.gepa.operator.levers,
    );
    if config.candidate.target_modules.is_empty() {
        return Err(OptimizerError::Config(
            "operator.levers filtered out every mutable candidate field".to_string(),
        ));
    }
    if config.seed_candidate.is_empty() {
        config.seed_candidate = program.seed_candidate.fields.clone();
    }
    config.validate()?;
    Ok(config)
}

fn validate_provider(name: &str, provider: &str) -> Result<()> {
    match normalize_key(provider).as_str() {
        "openai" | "openrouter" | "anthropic" | "groq" | "together" | "fireworks" => Ok(()),
        _ => Err(OptimizerError::Config(format!(
            "{name} must be one of openai, openrouter, anthropic, groq, together, fireworks; got {provider:?}"
        ))),
    }
}

fn validate_api_family(name: &str, api_family: &str) -> Result<()> {
    match normalize_key(api_family).as_str() {
        "chat_completions" | "responses" => Ok(()),
        _ => Err(OptimizerError::Config(format!(
            "{name} must be chat_completions or responses; got {api_family:?}"
        ))),
    }
}

fn validate_taskset(taskset: &ServiceTasksetSpec) -> Result<()> {
    if taskset.train_ids.is_empty() {
        return Err(OptimizerError::Config(
            "taskset.train_ids must be non-empty".to_string(),
        ));
    }
    if taskset.heldout_ids.is_empty() {
        return Err(OptimizerError::Config(
            "taskset.heldout_ids must be non-empty".to_string(),
        ));
    }
    Ok(())
}

/// Validate the request's `task_pools` (sibling of `taskset`): internal pool
/// invariants plus the split-local cross-reference against the taskset ids.
fn validate_request_task_pools(
    task_pools: &GepaTaskPoolsConfig,
    train_ids: &[String],
    heldout_ids: &[String],
) -> Result<()> {
    for (name, values) in [
        ("task_pools.pareto", &task_pools.pareto),
        ("task_pools.minibatch", &task_pools.minibatch),
        ("task_pools.reflection", &task_pools.reflection),
        ("task_pools.heldout", &task_pools.heldout),
    ] {
        if values.is_empty() {
            return Err(OptimizerError::Config(format!("{name} must be non-empty")));
        }
        if values.iter().any(|value| value.trim().is_empty()) {
            return Err(OptimizerError::Config(format!(
                "{name} entries must be non-empty"
            )));
        }
    }
    let minibatch = task_pools
        .minibatch
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let reflection = task_pools
        .reflection
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing = minibatch
        .difference(&reflection)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(OptimizerError::Config(format!(
            "task_pools.minibatch must be a subset of task_pools.reflection; missing from reflection: {:?}",
            missing
        )));
    }
    let heldout = task_pools.heldout.iter().cloned().collect::<BTreeSet<_>>();
    let pareto = task_pools.pareto.iter().cloned().collect::<BTreeSet<_>>();
    let overlaps = heldout
        .intersection(&pareto)
        .chain(heldout.intersection(&minibatch))
        .chain(heldout.intersection(&reflection))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !overlaps.is_empty() {
        return Err(OptimizerError::Config(format!(
            "task_pools.heldout must be disjoint from pareto/minibatch/reflection; overlaps: {:?}",
            overlaps
        )));
    }
    // Pools are split-local: search pools draw from train_ids, heldout from heldout_ids.
    let train = train_ids.iter().cloned().collect::<BTreeSet<_>>();
    let unknown_search = pareto
        .iter()
        .chain(&minibatch)
        .chain(&reflection)
        .filter(|id| !train.contains(*id))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !unknown_search.is_empty() {
        return Err(OptimizerError::Config(format!(
            "task_pools pareto/minibatch/reflection ids must come from taskset.train_ids; unknown: {:?}",
            unknown_search.into_iter().collect::<Vec<_>>()
        )));
    }
    let held = heldout_ids.iter().cloned().collect::<BTreeSet<_>>();
    let unknown_heldout = heldout
        .iter()
        .filter(|id| !held.contains(*id))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !unknown_heldout.is_empty() {
        return Err(OptimizerError::Config(format!(
            "task_pools.heldout ids must come from taskset.heldout_ids; unknown: {:?}",
            unknown_heldout.into_iter().collect::<Vec<_>>()
        )));
    }
    Ok(())
}

fn apply_policy_credentials(
    config: &mut SynthOptimizerConfig,
    credentials: &ServiceCredentials,
) -> Result<()> {
    match normalize_key(&credentials.resolver).as_str() {
        "env" => {
            let env_var = required_env_var(credentials, "policy.credentials.env_var")?;
            config.policy.credential_mode = "byok".to_string();
            config.policy.api_key_env = Some(env_var.clone());
            config
                .policy
                .config
                .insert("credential_env_var".to_string(), json!(env_var));
            Ok(())
        }
        "broker" => Err(OptimizerError::Config(
            "credentials.resolver=broker is reserved but not implemented in GEPA service v1"
                .to_string(),
        )),
        resolver => Err(OptimizerError::Config(format!(
            "credentials.resolver must be env or broker; got {resolver:?}"
        ))),
    }
}

fn apply_proposer_credentials(
    config: &mut SynthOptimizerConfig,
    credentials: &ServiceCredentials,
) -> Result<()> {
    match normalize_key(&credentials.resolver).as_str() {
        "env" => {
            config.proposer.auth_mode = "api_key".to_string();
            config.proposer.api_key_env = Some(required_env_var(
                credentials,
                "proposer.credentials.env_var",
            )?);
            Ok(())
        }
        "broker" => Err(OptimizerError::Config(
            "credentials.resolver=broker is reserved but not implemented in GEPA service v1"
                .to_string(),
        )),
        resolver => Err(OptimizerError::Config(format!(
            "credentials.resolver must be env or broker; got {resolver:?}"
        ))),
    }
}

fn apply_service_proposer_auth(
    config: &mut SynthOptimizerConfig,
    proposer: &ServiceProposerSpec,
) -> Result<()> {
    let auth_mode = proposer
        .auth_mode
        .as_deref()
        .map(normalize_key)
        .unwrap_or_else(
            || match normalize_key(&proposer.credentials.resolver).as_str() {
                "env" => "api_key".to_string(),
                _ => "chatgpt".to_string(),
            },
        );
    match auth_mode.as_str() {
        "api_key" => {
            apply_proposer_credentials(config, &proposer.credentials)?;
            config.proposer.auth_mode = "api_key".to_string();
            config.proposer.copy_host_auth = proposer.copy_host_auth.unwrap_or(false);
        }
        "chatgpt" | "host" => {
            config.proposer.auth_mode = "chatgpt".to_string();
            config.proposer.api_key_env = None;
            config.proposer.copy_host_auth = proposer.copy_host_auth.unwrap_or(true);
            config.proposer.codex_home = proposer
                .codex_home
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from);
        }
        other => {
            return Err(OptimizerError::Config(format!(
                "proposer.auth_mode must be api_key, chatgpt, or host; got {other:?}"
            )));
        }
    }
    Ok(())
}

fn required_env_var(credentials: &ServiceCredentials, field: &str) -> Result<String> {
    credentials
        .env_var
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| OptimizerError::Config(format!("{field} is required when resolver=env")))
}

fn apply_stop_conditions(
    config: &mut SynthOptimizerConfig,
    stop_conditions: &[ServiceStopCondition],
) -> Result<()> {
    for condition in stop_conditions {
        match condition {
            ServiceStopCondition::MaxRollouts { n, train, heldout } => {
                require_positive_usize("stop_conditions.max_rollouts.n", *n)?;
                config.gepa.max_total_rollouts = *n;
                if let Some(value) = *train {
                    require_positive_usize("stop_conditions.max_rollouts.train", value)?;
                    config.gepa.max_train_rollouts = Some(value);
                }
                if let Some(value) = *heldout {
                    require_positive_usize("stop_conditions.max_rollouts.heldout", value)?;
                    config.gepa.max_heldout_rollouts = Some(value);
                }
            }
            ServiceStopCondition::MaxWallSeconds { n } => {
                require_positive_u64("stop_conditions.max_wall_seconds.n", *n)?;
                config.gepa.max_time_seconds = Some(*n);
            }
            ServiceStopCondition::MaxGenerations { n } => {
                require_positive_usize("stop_conditions.max_generations.n", *n)?;
                config.gepa.max_generations = *n;
            }
            ServiceStopCondition::MaxCostUsd { value } => {
                if !value.is_finite() || *value <= 0.0 {
                    return Err(OptimizerError::Config(
                        "stop_conditions.max_cost_usd.value must be positive".to_string(),
                    ));
                }
                config.gepa.max_cost_usd = *value;
            }
            ServiceStopCondition::ExternalSignal => {}
            ServiceStopCondition::NoImprovement {
                generations,
                metric,
            } => {
                require_positive_usize("stop_conditions.no_improvement.generations", *generations)?;
                config.gepa.no_improvement_generations = Some(*generations);
                config.gepa.no_improvement_metric = Some(validate_stop_metric(
                    metric.as_deref().unwrap_or("heldout_score"),
                )?);
            }
            ServiceStopCondition::ScoreThreshold { value, metric } => {
                if !value.is_finite() {
                    return Err(OptimizerError::Config(
                        "stop_conditions.score_threshold.value must be finite".to_string(),
                    ));
                }
                config.gepa.score_threshold_value = Some(*value);
                config.gepa.score_threshold_metric = Some(validate_stop_metric(
                    metric.as_deref().unwrap_or("heldout_score"),
                )?);
            }
            ServiceStopCondition::Episode {
                proposer_rounds,
                max_rollouts,
                max_wall_seconds,
                max_spend_usd,
                skip_heldout,
            } => {
                if let Some(n) = *proposer_rounds {
                    require_positive_usize("stop_conditions.episode.proposer_rounds", n)?;
                    config.gepa.episode.proposer_rounds = Some(n);
                }
                if let Some(n) = *max_rollouts {
                    require_positive_usize("stop_conditions.episode.max_rollouts", n)?;
                    config.gepa.episode.max_rollouts = Some(n);
                }
                if let Some(n) = *max_wall_seconds {
                    require_positive_u64("stop_conditions.episode.max_wall_seconds", n)?;
                    config.gepa.episode.max_wall_seconds = Some(n);
                }
                if let Some(value) = *max_spend_usd {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(OptimizerError::Config(
                            "stop_conditions.episode.max_spend_usd must be positive".to_string(),
                        ));
                    }
                    config.gepa.episode.max_spend_usd = Some(value);
                }
                config.gepa.episode.skip_heldout = *skip_heldout;
            }
        }
    }
    Ok(())
}

fn validate_stop_metric(metric: &str) -> Result<String> {
    match normalize_key(metric).as_str() {
        "train_score" | "train_reward" => Ok("train_score".to_string()),
        "heldout_score" | "heldout_reward" => Ok("heldout_score".to_string()),
        _ => Err(OptimizerError::Config(format!(
            "stop condition metric must be train_score or heldout_score; got {metric:?}"
        ))),
    }
}

fn apply_advanced_config(
    config: &mut SynthOptimizerConfig,
    advanced: &ServiceAdvancedConfig,
) -> Result<()> {
    if let Some(budgets) = advanced.budgets.as_ref() {
        if let Some(value) = budgets.max_train_rollouts {
            require_positive_usize("advanced.budgets.max_train_rollouts", value)?;
            config.gepa.max_train_rollouts = Some(value);
        }
        if let Some(value) = budgets.max_heldout_rollouts {
            require_positive_usize("advanced.budgets.max_heldout_rollouts", value)?;
            config.gepa.max_heldout_rollouts = Some(value);
        }
    }
    if let Some(pipeline) = advanced.pipeline.as_ref() {
        if let Some(value) = pipeline.mode {
            config.gepa.pipeline.mode = value;
        }
        if let Some(value) = pipeline.staleness_policy {
            config.gepa.pipeline.staleness_policy = value;
        }
        if let Some(value) = pipeline.delta_max {
            config.gepa.pipeline.delta_max = value;
        }
        if let Some(value) = pipeline.max_generations {
            require_positive_usize("advanced.pipeline.max_generations", value)?;
            config.gepa.max_generations = value;
        }
        if let Some(value) = pipeline.proposals_per_generation {
            require_positive_usize("advanced.pipeline.proposals_per_generation", value)?;
            config.gepa.proposals_per_generation = value;
        }
        if let Some(value) = pipeline.minibatch_size {
            require_positive_usize("advanced.pipeline.minibatch_size", value)?;
            config.gepa.minibatch_size = value;
        }
        if let Some(value) = pipeline.max_in_flight_candidates {
            require_positive_usize("advanced.pipeline.max_in_flight_candidates", value)?;
            if pipeline.mode.is_none()
                && matches!(config.gepa.pipeline.mode, GepaPipelineMode::SyncSerial)
            {
                config.gepa.pipeline.mode = GepaPipelineMode::AsyncPipelined;
            }
            config.gepa.pipeline.max_in_flight_candidates = value;
        }
        if let Some(value) = pipeline.rollout_workers {
            require_positive_usize("advanced.pipeline.rollout_workers", value)?;
            config.gepa.pipeline.workers.rollout = value;
        }
        if let Some(value) = pipeline.proposer_workers {
            require_positive_usize("advanced.pipeline.proposer_workers", value)?;
            config.gepa.pipeline.workers.propose = value;
        }
        if let Some(value) = pipeline.evaluator_workers {
            require_positive_usize("advanced.pipeline.evaluator_workers", value)?;
            config.gepa.pipeline.workers.evaluate = value;
        }
        if let Some(value) = pipeline.rollout_chunk_size {
            require_positive_usize("advanced.pipeline.rollout_chunk_size", value)?;
            config.gepa.rollout_chunk_size = Some(value);
        }
        if let Some(value) = pipeline.speculative_alpha {
            if !value.is_finite() || value <= 0.0 || value > 1.0 {
                return Err(OptimizerError::Config(
                    "advanced.pipeline.speculative_alpha must be in (0, 1]".to_string(),
                ));
            }
            config.gepa.pipeline.speculative_completion.enabled = true;
            config.gepa.pipeline.speculative_completion.alpha = value;
        }
        if let Some(value) = pipeline.adaptive_stage_workers {
            config.gepa.pipeline.adaptive_stage_workers.enabled = value;
        }
        if let Some(value) = pipeline.adaptive_rollout_concurrency {
            config.gepa.pipeline.adaptive_rollout_concurrency.enabled = value;
        }
    }
    if let Some(timeouts) = advanced.timeouts.as_ref() {
        if let Some(value) = timeouts.rollout_seconds {
            require_positive_u64("advanced.timeouts.rollout_seconds", value)?;
            config.gepa.rollout_async_timeout_seconds = value;
        }
        if let Some(value) = timeouts.container_http_seconds {
            require_positive_u64("advanced.timeouts.container_http_seconds", value)?;
            config
                .policy
                .config
                .insert("container_http_seconds".to_string(), json!(value));
        }
        if let Some(value) = timeouts.rollout_http_retries {
            config
                .policy
                .config
                .insert("rollout_http_retries".to_string(), json!(value));
        }
    }
    if let Some(policy_io) = advanced.policy_io.as_ref() {
        config
            .policy
            .config
            .insert("policy_io".to_string(), policy_io.clone());
    }
    if let Some(proposer_io) = advanced.proposer_io.as_ref() {
        if let Some(value) = proposer_io.timeout_seconds {
            require_positive_u64("advanced.proposer_io.timeout_seconds", value)?;
            config.proposer.timeout_seconds = value;
        }
        if let Some(value) = proposer_io.message_stall_timeout_seconds {
            require_positive_u64("advanced.proposer_io.message_stall_timeout_seconds", value)?;
            config.proposer.message_stall_timeout_seconds = value;
        }
        if let Some(path) = proposer_io
            .codex_home
            .as_ref()
            .map(|path| path.trim())
            .filter(|path| !path.is_empty())
        {
            config.proposer.codex_home = Some(PathBuf::from(path));
        }
        if let Some(rounds) = proposer_io.schema_repair_rounds {
            config.proposer.schema_repair_rounds = rounds;
        }
    }
    if let Some(enabled) = advanced.adaptive_rollout_concurrency {
        config.gepa.pipeline.adaptive_rollout_concurrency.enabled = enabled;
    }
    if let Some(operator) = advanced.operator.clone() {
        config.gepa.operator = operator;
        if config.gepa.operator.mcp_agent.enabled && !config.proposer.mcp.enabled {
            config.proposer.mcp = config.gepa.operator.mcp_agent.clone();
        }
    }
    if let Some(wf) = advanced.jesterky_workflow.as_ref() {
        if let Some(enabled) = wf.enabled {
            config.jesterky_workflow.enabled = enabled;
        }
        if let Some(bulk) = wf.bulk {
            config.jesterky_workflow.bulk = bulk;
        }
        if let Some(spec) = wf.spec.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            config.jesterky_workflow.spec = spec.to_string();
        }
        if let Some(command) = wf
            .command
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            config.jesterky_workflow.command = command.to_string();
        }
        if let Some(actor) = wf.actor.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            config.jesterky_workflow.actor = actor.to_string();
        }
        if let Some(model) = wf.model.clone() {
            config.jesterky_workflow.model = Some(model);
        }
        if let Some(fail_closed) = wf.fail_closed {
            config.jesterky_workflow.fail_closed = fail_closed;
        }
    }
    Ok(())
}

fn normalize_service_rollout_budgets(
    config: &mut SynthOptimizerConfig,
    taskset: &ServiceTasksetSpec,
) -> Result<()> {
    let total = config.gepa.max_total_rollouts;
    let minimum_heldout_rollouts = taskset.heldout_ids.len();
    if minimum_heldout_rollouts == 0 {
        return Err(OptimizerError::Config(
            "taskset.heldout_ids must be non-empty".to_string(),
        ));
    }

    let explicit_train = config.gepa.max_train_rollouts;
    let explicit_heldout = config.gepa.max_heldout_rollouts;
    let (train, heldout) = match (explicit_train, explicit_heldout) {
        (Some(train), Some(heldout)) => (train, heldout),
        (Some(train), None) => {
            if train >= total {
                return Err(OptimizerError::Config(format!(
                    "max_rollouts total {total} leaves no heldout budget after max_train_rollouts={train}; set max_heldout_rollouts explicitly or lower train"
                )));
            }
            (train, total - train)
        }
        (None, Some(heldout)) => {
            if heldout >= total {
                return Err(OptimizerError::Config(format!(
                    "max_rollouts total {total} leaves no train budget after max_heldout_rollouts={heldout}; increase max_rollouts or lower heldout"
                )));
            }
            (total - heldout, heldout)
        }
        (None, None) => {
            if total <= minimum_heldout_rollouts {
                return Err(OptimizerError::Config(format!(
                    "max_rollouts total {total} is too small to reserve {minimum_heldout_rollouts} heldout rollouts and still run training"
                )));
            }
            (total - minimum_heldout_rollouts, minimum_heldout_rollouts)
        }
    };

    if train == 0 || heldout == 0 {
        return Err(OptimizerError::Config(
            "GEPA service rollout budgets must leave positive train and heldout buckets"
                .to_string(),
        ));
    }

    config.gepa.max_train_rollouts = Some(train);
    config.gepa.max_heldout_rollouts = Some(heldout);
    config.gepa.max_total_rollouts = train.saturating_add(heldout);
    Ok(())
}

fn require_positive_usize(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(OptimizerError::Config(format!("{name} must be positive")));
    }
    Ok(())
}

fn require_positive_u64(name: &str, value: u64) -> Result<()> {
    if value == 0 {
        return Err(OptimizerError::Config(format!("{name} must be positive")));
    }
    Ok(())
}

fn project_run(store: &WorkspaceStore, request: &WorkspaceRunRequestStatus) -> Result<Value> {
    let config = store.run_request_config(&request.request_id)?;
    if request.status != "queued" {
        ensure_run_workspace(store, request)?;
    }
    let run_store = run_workspace_store(request)?;
    let run_status = run_store
        .as_ref()
        .map(|store| store.current_optimizer_state(&request.run_id))
        .transpose()?
        .flatten();
    let cursor = run_store
        .as_ref()
        .map(|store| load_gepa_cursor(store, &request.run_id))
        .transpose()?
        .flatten();
    let usage_source = if request.usage.is_null() {
        run_status
            .as_ref()
            .map(|status| status.usage.clone())
            .or_else(|| cursor.as_ref().map(|cursor| cursor.usage.clone()))
            .unwrap_or(Value::Null)
    } else {
        request.usage.clone()
    };
    let cost_usd = request.cost_usd.or_else(|| {
        cursor
            .as_ref()
            .map(|cursor| cursor.cost_usd)
            .filter(|cost| *cost > 0.0)
    });
    let rollout_count = run_status
        .as_ref()
        .map(|status| status.counts.rollouts)
        .or_else(|| cursor.as_ref().map(|cursor| cursor.rollout_count as u64))
        .unwrap_or(0);
    let generation_count = cursor
        .as_ref()
        .map(|cursor| cursor.generation as u64)
        .unwrap_or(0);
    let best_candidate_id = request.best_candidate_id.clone().or_else(|| {
        cursor
            .as_ref()
            .and_then(|cursor| cursor.best_candidate_id.clone())
    });
    let candidate_count = cursor
        .as_ref()
        .map(|cursor| value_array_len(&cursor.candidates) as u64)
        .unwrap_or(0);
    let timing_records = run_store
        .as_ref()
        .map(|store| store.view().run_phase_timing_records(&request.run_id))
        .transpose()?
        .unwrap_or_default();
    let limit_snapshot = run_store
        .as_ref()
        .map(|store| {
            let view = store.view();
            if view.run_limit_records(&request.run_id)?.is_empty() {
                Ok(None)
            } else {
                project_gepa_limit_snapshot(store, &config).map(Some)
            }
        })
        .transpose()?
        .flatten();
    Ok(json!({
        "run_id": request.run_id,
        "status": project_request_status(&request.status),
        "worker_id": request.worker_id.as_ref(),
        "lease_expires_at": request.lease_expires_at.as_ref(),
        "phase": cursor.as_ref().map(|cursor| cursor.phase.as_str()),
        "generation": generation_count,
        "best_candidate_id": best_candidate_id,
        "candidate_count": candidate_count,
        "checkpoint_sequence": cursor.as_ref().map(|cursor| cursor.checkpoint_sequence),
        "config": project_run_config(&config, request.manual_step),
        "submitted_at": request.submitted_at,
        "started_at": request.started_at,
        "finished_at": request.finished_at,
        "usage": project_usage(&usage_source, cost_usd),
        "totals": {
            "rollouts": rollout_count,
            "generations": generation_count,
        },
        "limits": limit_snapshot,
        "timing_summary": timing_summary_from_records(&timing_records),
        "outcome": project_outcome(request),
    }))
}

fn run_phase_timings_for_request(
    request: &WorkspaceRunRequestStatus,
) -> Result<Vec<RunPhaseTimingRecord>> {
    let Some(run_store) = run_workspace_store(request)? else {
        return Ok(Vec::new());
    };
    run_store.view().run_phase_timing_records(&request.run_id)
}

fn project_timing_record(record: &RunPhaseTimingRecord) -> Value {
    serde_json::to_value(record).unwrap_or_else(|_| {
        json!({
            "run_id": record.run_id,
            "timing_id": record.timing_id,
            "lane": record.lane,
            "kind": record.kind,
            "status": record.status,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "wall_seconds": record.wall_seconds,
            "item_count": record.item_count,
        })
    })
}

fn project_run_transition_stats(request: &WorkspaceRunRequestStatus) -> Result<Value> {
    let run_dir = Path::new(&request.run_dir);
    let transition_path = run_dir.join("transitions.sqlite");
    let transitions = TransitionLog::read_run_dir(run_dir)?;
    let transition_count = transitions.len();
    let entity_counts = transition_entity_counts(&transitions);
    let latest_state_counts = transition_latest_state_counts(&transitions);
    let wall_seconds = wall_seconds_from_transitions(&transitions);
    let run_duration = run_duration_seconds(&transitions);

    let candidate_terminal_states = [
        "accepted",
        "rejected_minibatch",
        "rejected_full_train",
        "deferred_budget",
        "heldout_scored",
        "archived",
    ];
    let rollout_terminal_states = ["completed", "failed", "cached", "cancelled"];
    let rollout_count =
        distinct_entity_count_any(&transitions, "rollout", &rollout_terminal_states, false);
    let rollout_spans = state_spans(
        &transitions,
        "rollout",
        "running",
        &rollout_terminal_states,
        false,
    );
    let proposer_generation_spans = state_spans(
        &transitions,
        "proposer_round",
        "generating",
        &["returned"],
        false,
    );
    let proposer_lifecycle_spans = state_spans(
        &transitions,
        "proposer_round",
        "requested",
        &["closed", "parse_failed"],
        false,
    );
    let candidate_lifecycle_spans = state_spans(
        &transitions,
        "candidate",
        "registered",
        &candidate_terminal_states,
        true,
    );
    let candidate_minibatch_spans = state_spans(
        &transitions,
        "candidate",
        "minibatch_evaluating",
        &["accepted_minibatch", "rejected_minibatch"],
        true,
    );
    let candidate_count = distinct_generated_candidate_count(&transitions);
    let candidate_terminal_count =
        distinct_generated_entity_count_any(&transitions, "candidate", &candidate_terminal_states);
    let minibatch_passed =
        distinct_generated_entity_count(&transitions, "candidate", "accepted_minibatch");
    let minibatch_rejected =
        distinct_generated_entity_count(&transitions, "candidate", "rejected_minibatch");
    let accepted_full_train =
        distinct_generated_entity_count(&transitions, "candidate", "accepted");
    let proposer_round_count = distinct_entity_type_count(&transitions, "proposer_round");
    let upstream_429_count = transitions
        .iter()
        .filter(|row| {
            row.entity_type == "rollout"
                && rollout_terminal_states.contains(&row.to_state.as_str())
                && transition_has_429(row)
        })
        .count();
    let proposer_minutes = proposer_generation_spans.iter().sum::<f64>() / 60.0;
    let wall_minutes = wall_seconds.unwrap_or(0.0) / 60.0;
    let rollout_runtime_minutes = rollout_spans.iter().sum::<f64>() / 60.0;
    let (total_tokens, cost_usd) = transition_usage_totals(&transitions);

    Ok(json!({
        "schema_version": "gepa_run_transition_stats.v1",
        "run_id": request.run_id,
        "source": {
            "kind": "transitions.sqlite",
            "path": transition_path.display().to_string(),
            "exists": transition_path.exists(),
        },
        "transition_count": transition_count,
        "entity_counts": entity_counts,
        "latest_state_counts": latest_state_counts,
        "run": {
            "status": project_request_status(&request.status),
            "duration_seconds": run_duration,
            "wall_seconds": wall_seconds,
            "started_at": request.started_at,
            "finished_at": request.finished_at,
        },
        "candidate": {
            "count": candidate_count,
            "terminal_count": candidate_terminal_count,
            "active_count": candidate_count.saturating_sub(candidate_terminal_count),
            "duration_seconds": duration_summary(&candidate_lifecycle_spans),
            "minibatch_decision_seconds": duration_summary(&candidate_minibatch_spans),
        },
        "minibatch": {
            "passed": minibatch_passed,
            "rejected": minibatch_rejected,
            "pass_rate": ratio_value(minibatch_passed, minibatch_passed + minibatch_rejected),
        },
        "rollout": {
            "count": rollout_count,
            "duration_seconds": duration_summary(&rollout_spans),
            "max_concurrent": max_concurrent_rollouts(&transitions),
            "upstream_429_count": upstream_429_count,
            "upstream_429_rate": ratio_value(upstream_429_count, rollout_count),
        },
        "proposer": {
            "round_count": proposer_round_count,
            "llm_duration_seconds": duration_summary(&proposer_generation_spans),
            "lifecycle_seconds": duration_summary(&proposer_lifecycle_spans),
        },
        "throughput": {
            "candidates_per_proposer_minute": rate_value(candidate_count, proposer_minutes),
            "passing_candidates_per_proposer_minute": rate_value(minibatch_passed, proposer_minutes),
            "pareto_delta_per_proposer_minute": rate_value(accepted_full_train, proposer_minutes),
            "rollouts_per_wall_minute": rate_value(rollout_count, wall_minutes),
            "rollouts_per_rollout_runtime_minute": rate_value(rollout_count, rollout_runtime_minutes),
        },
        "usage": {
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        },
    }))
}

fn timing_summary_from_records(records: &[RunPhaseTimingRecord]) -> Value {
    let mut proposer_total_seconds = 0.0f64;
    let mut rollout_total_seconds = 0.0f64;
    let mut proposer_count = 0u64;
    let mut rollout_count = 0u64;
    let mut latest_proposer_seconds = None;
    let mut latest_rollout_seconds = None;
    let mut latest_proposer_finished_at: Option<&str> = None;
    let mut latest_rollout_finished_at: Option<&str> = None;
    for record in records {
        let wall_seconds = record.wall_seconds.unwrap_or(0.0);
        if record.lane == "proposer" {
            proposer_count += 1;
            proposer_total_seconds += wall_seconds;
            if record
                .finished_at
                .as_deref()
                .map(|finished_at| {
                    latest_proposer_finished_at
                        .map(|latest| finished_at > latest)
                        .unwrap_or(true)
                })
                .unwrap_or(false)
            {
                latest_proposer_finished_at = record.finished_at.as_deref();
                latest_proposer_seconds = record.wall_seconds;
            }
        } else if record.lane == "rollout" {
            rollout_count += record.item_count.unwrap_or(0);
            rollout_total_seconds += wall_seconds;
            if record
                .finished_at
                .as_deref()
                .map(|finished_at| {
                    latest_rollout_finished_at
                        .map(|latest| finished_at > latest)
                        .unwrap_or(true)
                })
                .unwrap_or(false)
            {
                latest_rollout_finished_at = record.finished_at.as_deref();
                latest_rollout_seconds = record.wall_seconds;
            }
        }
    }
    json!({
        "proposer_round_count": proposer_count,
        "proposer_total_seconds": proposer_total_seconds,
        "latest_proposer_seconds": latest_proposer_seconds,
        "latest_proposer_finished_at": latest_proposer_finished_at,
        "rollout_count": rollout_count,
        "rollout_batch_count": records.iter().filter(|record| record.lane == "rollout").count(),
        "rollout_total_seconds": rollout_total_seconds,
        "latest_rollout_seconds": latest_rollout_seconds,
        "latest_rollout_finished_at": latest_rollout_finished_at,
    })
}

fn transition_entity_counts(transitions: &[TransitionRow]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for row in transitions {
        *counts.entry(row.entity_type.clone()).or_insert(0) += 1;
    }
    counts
}

fn transition_latest_state_counts(
    transitions: &[TransitionRow],
) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut latest: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    for row in transitions {
        latest.insert(
            (row.entity_type.as_str(), row.entity_id.as_str()),
            row.to_state.as_str(),
        );
    }
    let mut counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for ((entity_type, _), state) in latest {
        *counts
            .entry(entity_type.to_string())
            .or_default()
            .entry(state.to_string())
            .or_insert(0) += 1;
    }
    counts
}

fn wall_seconds_from_transitions(transitions: &[TransitionRow]) -> Option<f64> {
    let min_ts = transitions.iter().map(|row| row.ts_unix_ms).min()?;
    let max_ts = transitions.iter().map(|row| row.ts_unix_ms).max()?;
    Some(seconds_between(min_ts, max_ts))
}

fn run_duration_seconds(transitions: &[TransitionRow]) -> Option<f64> {
    let run_rows = transitions
        .iter()
        .filter(|row| row.entity_type == "run")
        .collect::<Vec<_>>();
    let first = run_rows.first()?.ts_unix_ms;
    let terminal = run_rows
        .iter()
        .find(|row| matches!(row.to_state.as_str(), "completed" | "failed" | "cancelled"))
        .map(|row| row.ts_unix_ms);
    let last = terminal.or_else(|| run_rows.last().map(|row| row.ts_unix_ms))?;
    Some(seconds_between(first, last))
}

fn state_spans(
    transitions: &[TransitionRow],
    entity_type: &str,
    start_state: &str,
    terminal_states: &[&str],
    generated_only: bool,
) -> Vec<f64> {
    let mut by_entity: BTreeMap<&str, Vec<&TransitionRow>> = BTreeMap::new();
    for row in transitions {
        if row.entity_type == entity_type {
            by_entity
                .entry(row.entity_id.as_str())
                .or_default()
                .push(row);
        }
    }
    let mut spans = Vec::new();
    for rows in by_entity.values() {
        let mut start_ms = None;
        for row in rows {
            if generated_only && row.parent_id.is_none() {
                continue;
            }
            if row.to_state == start_state {
                start_ms = Some(row.ts_unix_ms);
            } else if let Some(start) = start_ms {
                if terminal_states.contains(&row.to_state.as_str()) {
                    spans.push(seconds_between(start, row.ts_unix_ms));
                    start_ms = None;
                }
            }
        }
    }
    spans
}

fn seconds_between(start_ms: i64, end_ms: i64) -> f64 {
    end_ms.saturating_sub(start_ms).max(0) as f64 / 1000.0
}

fn distinct_entity_type_count(transitions: &[TransitionRow], entity_type: &str) -> usize {
    transitions
        .iter()
        .filter(|row| row.entity_type == entity_type)
        .map(|row| row.entity_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_entity_count_any(
    transitions: &[TransitionRow],
    entity_type: &str,
    to_states: &[&str],
    generated_only: bool,
) -> usize {
    transitions
        .iter()
        .filter(|row| {
            row.entity_type == entity_type
                && to_states.contains(&row.to_state.as_str())
                && (!generated_only || row.parent_id.is_some())
        })
        .map(|row| row.entity_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_generated_candidate_count(transitions: &[TransitionRow]) -> usize {
    transitions
        .iter()
        .filter(|row| {
            row.entity_type == "candidate" && row.trigger == "registered" && row.parent_id.is_some()
        })
        .map(|row| row.entity_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_generated_entity_count(
    transitions: &[TransitionRow],
    entity_type: &str,
    to_state: &str,
) -> usize {
    transitions
        .iter()
        .filter(|row| {
            row.entity_type == entity_type && row.to_state == to_state && row.parent_id.is_some()
        })
        .map(|row| row.entity_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_generated_entity_count_any(
    transitions: &[TransitionRow],
    entity_type: &str,
    to_states: &[&str],
) -> usize {
    distinct_entity_count_any(transitions, entity_type, to_states, true)
}

fn max_concurrent_rollouts(transitions: &[TransitionRow]) -> u64 {
    let mut events = Vec::new();
    for row in transitions {
        if row.entity_type != "rollout" {
            continue;
        }
        if row.to_state == "running" {
            events.push((row.ts_unix_ms, 1i64));
        } else if matches!(row.to_state.as_str(), "completed" | "failed" | "cancelled") {
            events.push((row.ts_unix_ms, -1i64));
        }
    }
    events.sort_by_key(|(ts, delta)| (*ts, -*delta));
    let mut active = 0i64;
    let mut max_active = 0i64;
    for (_, delta) in events {
        active = active.saturating_add(delta);
        max_active = max_active.max(active);
    }
    u64::try_from(max_active).unwrap_or(0)
}

fn transition_has_429(row: &TransitionRow) -> bool {
    if metadata_status_is_429(row.metadata.get("status_code"))
        || metadata_status_is_429(row.metadata.get("http_status"))
    {
        return true;
    }
    row.metadata
        .get("failure")
        .and_then(Value::as_object)
        .map(|failure| {
            metadata_status_is_429(failure.get("status_code"))
                || metadata_status_is_429(failure.get("http_status"))
        })
        .unwrap_or(false)
}

fn metadata_status_is_429(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Number(number)) => number.as_u64() == Some(429),
        Some(Value::String(status)) => status == "429",
        _ => false,
    }
}

fn transition_usage_totals(transitions: &[TransitionRow]) -> (u64, f64) {
    let mut total_tokens = 0u64;
    let mut cost_usd = 0.0f64;
    for row in transitions {
        if row.entity_type == "proposer_round" && row.to_state != "closed" {
            continue;
        }
        if row.entity_type == "rollout"
            && !matches!(
                row.to_state.as_str(),
                "completed" | "failed" | "cached" | "cancelled"
            )
        {
            continue;
        }
        if !matches!(row.entity_type.as_str(), "proposer_round" | "rollout") {
            continue;
        }
        cost_usd += value_as_f64(row.metadata.get("cost_usd")).unwrap_or(0.0);
        if let Some(usage) = row.metadata.get("usage").and_then(Value::as_object) {
            total_tokens = total_tokens.saturating_add(
                value_as_u64(usage.get("total_tokens"))
                    .or_else(|| value_as_u64(usage.get("totalTokens")))
                    .unwrap_or(0),
            );
        }
    }
    (total_tokens, cost_usd)
}

fn value_as_f64(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_as_u64(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(number)) => number.as_u64(),
        Some(Value::String(text)) => text.parse::<u64>().ok(),
        _ => None,
    }
}

fn duration_summary(values: &[f64]) -> Value {
    json!({
        "count": values.len(),
        "total": values.iter().sum::<f64>(),
        "min": values.iter().copied().reduce(f64::min),
        "max": values.iter().copied().reduce(f64::max),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    })
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.total_cmp(right));
    if (quantile - 0.50).abs() < f64::EPSILON {
        let middle = ordered.len() / 2;
        if ordered.len() % 2 == 0 {
            return Some((ordered[middle - 1] + ordered[middle]) / 2.0);
        }
        return Some(ordered[middle]);
    }
    let index = ((ordered.len().saturating_sub(1)) as f64 * quantile).round() as usize;
    ordered.get(index.min(ordered.len() - 1)).copied()
}

fn ratio_value(numerator: usize, denominator: usize) -> Value {
    if denominator == 0 {
        Value::Null
    } else {
        json!(numerator as f64 / denominator as f64)
    }
}

fn rate_value(count: usize, minutes: f64) -> Value {
    if minutes <= 0.0 {
        Value::Null
    } else {
        json!(count as f64 / minutes)
    }
}

fn project_run_state(store: &WorkspaceStore, request: &WorkspaceRunRequestStatus) -> Result<Value> {
    let run = project_run(store, request)?;
    let run_store = run_workspace_store(request)?;
    let cursor = run_store
        .as_ref()
        .map(|store| load_gepa_cursor(store, &request.run_id))
        .transpose()?
        .flatten();
    let Some(cursor) = cursor else {
        return Ok(json!({
            "run_id": request.run_id,
            "status": project_request_status(&request.status),
            "run": run,
            "cursor": Value::Null,
            "phase": Value::Null,
            "generation": 0,
            "proposal_index": 0,
            "heldout_candidate_index": 0,
            "best_candidate_id": request.best_candidate_id,
            "candidate_count": 0,
            "rollout_count": 0,
            "cost_usd": request.cost_usd.unwrap_or(0.0),
            "usage": request.usage,
            "pending": {
                "job_id": Value::Null,
                "effect_id": Value::Null,
                "reservation_ids": [],
            },
            "queues": {
                "proposal": {"count": 0, "items": []},
                "propose": {"count": 0, "items": []},
                "rollout": {"count": 0, "items": []},
                "evaluate": {"count": 0, "items": []},
            },
            "pipeline": Value::Null,
            "state_history": [],
            "metadata": Value::Null,
            "terminal_summary": Value::Null,
            "error_summary": Value::Null,
        }));
    };
    let cursor_value = serde_json::to_value(&cursor)?;
    let pipeline_value = serde_json::to_value(&cursor.pipeline_state)?;
    Ok(json!({
        "run_id": request.run_id,
        "status": project_request_status(&request.status),
        "run": run,
        "cursor": cursor_value,
        "phase": cursor.phase.as_str(),
        "generation": cursor.generation,
        "proposal_index": cursor.proposal_index,
        "heldout_candidate_index": cursor.heldout_candidate_index,
        "best_candidate_id": &cursor.best_candidate_id,
        "candidate_count": value_array_len(&cursor.candidates),
        "rollout_count": cursor.rollout_count,
        "cost_usd": cursor.cost_usd,
        "usage": &cursor.usage,
        "usage_ledger": &cursor.usage_ledger,
        "pending": {
            "job_id": &cursor.pending_job_id,
            "effect_id": &cursor.pending_effect_id,
            "reservation_ids": &cursor.pending_reservation_ids,
            "rollout_task_id": &cursor.rollout_task_id,
        },
        "active_evaluation": &cursor.active_evaluation,
        "queues": {
            "proposal": {
                "count": value_array_len(&cursor.proposal_queue),
                "items": &cursor.proposal_queue,
            },
            "propose": {
                "count": cursor.pipeline_state.propose_queue.len(),
                "items": &cursor.pipeline_state.propose_queue,
            },
            "rollout": {
                "count": cursor.pipeline_state.rollout_queue.len(),
                "items": &cursor.pipeline_state.rollout_queue,
            },
            "evaluate": {
                "count": cursor.pipeline_state.evaluate_queue.len(),
                "items": &cursor.pipeline_state.evaluate_queue,
            },
        },
        "pipeline": pipeline_value,
        "checkpoint_sequence": cursor.checkpoint_sequence,
        "state_history": &cursor.state_history,
        "metadata": &cursor.metadata,
        "terminal_summary": &cursor.terminal_summary,
        "error_summary": &cursor.error_summary,
    }))
}

fn value_array_len(value: &Value) -> usize {
    value.as_array().map(Vec::len).unwrap_or(0)
}

fn project_run_config(config: &SynthOptimizerConfig, manual_step: bool) -> Value {
    let policy_env_var = config
        .policy
        .config
        .get("credential_env_var")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| config.policy.api_key_env.clone());
    json!({
        "container_url": config.container.url,
        "policy": {
            "provider": config.policy.provider,
            "model": config.policy.model,
            "api_family": config.policy.api_family,
            "credentials": credential_projection(policy_env_var),
            "base_url": config.policy.base_url,
            "max_tokens": config.policy.max_tokens,
            "disable_reasoning": config.policy.disable_reasoning,
        },
        "proposer": {
            "provider": config.proposer.provider,
            "model": config.proposer.model,
            "api_family": config.proposer.api_family,
            "reasoning_effort": config.proposer.reasoning_effort,
            "credentials": credential_projection(config.proposer.api_key_env.clone()),
        },
        "taskset": {
            "train_ids": config.taskset.train_ids,
            "heldout_ids": config.taskset.heldout_ids,
        },
        "manual_step": manual_step,
        "stop_conditions": project_stop_conditions(config),
        "advanced": project_advanced_config(config),
    })
}

fn credential_projection(env_var: Option<String>) -> Value {
    match env_var {
        Some(env_var) => json!({"resolver": "env", "env_var": env_var}),
        None => json!({"resolver": "env", "env_var": "OPENAI_API_KEY"}),
    }
}

fn project_stop_conditions(config: &SynthOptimizerConfig) -> Value {
    let rollout_limit = if config.gepa.split_rollout_budgets_enabled() {
        json!({
            "kind": "max_rollouts",
            "n": config.gepa.effective_max_total_rollouts(),
            "train": config.gepa.train_rollout_limit(),
            "heldout": config.gepa.heldout_rollout_limit(),
        })
    } else {
        json!({"kind": "max_rollouts", "n": config.gepa.max_total_rollouts})
    };
    let mut conditions = vec![
        rollout_limit,
        json!({"kind": "max_generations", "n": config.gepa.max_generations}),
    ];
    if let Some(n) = config.gepa.max_time_seconds {
        conditions.push(json!({"kind": "max_wall_seconds", "n": n}));
    }
    if config.gepa.max_cost_usd > 0.0 {
        conditions.push(json!({"kind": "max_cost_usd", "value": config.gepa.max_cost_usd}));
    }
    if let Some(generations) = config.gepa.no_improvement_generations {
        conditions.push(json!({
            "kind": "no_improvement",
            "generations": generations,
            "metric": config.gepa.no_improvement_metric.as_deref().unwrap_or("heldout_score"),
        }));
    }
    if let Some(value) = config.gepa.score_threshold_value {
        conditions.push(json!({
            "kind": "score_threshold",
            "value": value,
            "metric": config.gepa.score_threshold_metric.as_deref().unwrap_or("heldout_score"),
        }));
    }
    if config.gepa.episode.has_delta_limits() || config.gepa.episode.skip_heldout {
        let mut episode = serde_json::Map::new();
        episode.insert("kind".to_string(), json!("episode"));
        if let Some(n) = config.gepa.episode.proposer_rounds {
            episode.insert("proposer_rounds".to_string(), json!(n));
        }
        if let Some(n) = config.gepa.episode.max_rollouts {
            episode.insert("max_rollouts".to_string(), json!(n));
        }
        if let Some(n) = config.gepa.episode.max_wall_seconds {
            episode.insert("max_wall_seconds".to_string(), json!(n));
        }
        if let Some(value) = config.gepa.episode.max_spend_usd {
            episode.insert("max_spend_usd".to_string(), json!(value));
        }
        episode.insert(
            "skip_heldout".to_string(),
            json!(config.gepa.episode.skip_heldout),
        );
        conditions.push(Value::Object(episode));
    }
    Value::Array(conditions)
}

fn project_advanced_config(config: &SynthOptimizerConfig) -> Value {
    let max_train_rollouts = config
        .gepa
        .split_rollout_budgets_enabled()
        .then(|| config.gepa.train_rollout_limit());
    let max_heldout_rollouts = config
        .gepa
        .split_rollout_budgets_enabled()
        .then(|| config.gepa.heldout_rollout_limit());
    json!({
        "pipeline": {
            "mode": config.gepa.pipeline.mode.as_str(),
            "staleness_policy": config.gepa.pipeline.staleness_policy.as_str(),
            "delta_max": config.gepa.pipeline.delta_max,
            "max_generations": config.gepa.max_generations,
            "proposals_per_generation": config.gepa.proposals_per_generation,
            "minibatch_size": config.gepa.minibatch_size,
            "max_in_flight_candidates": config.gepa.pipeline.max_in_flight_candidates,
            "proposer_workers": config.gepa.pipeline.workers.propose,
            "rollout_workers": config.gepa.pipeline.workers.rollout,
            "evaluator_workers": config.gepa.pipeline.workers.evaluate,
            "rollout_chunk_size": config.gepa.rollout_chunk_size,
            "speculative_completion": config.gepa.pipeline.speculative_completion,
            "adaptive_stage_workers": config.gepa.pipeline.adaptive_stage_workers,
        },
        "budgets": {
            "max_train_rollouts": max_train_rollouts,
            "max_heldout_rollouts": max_heldout_rollouts,
        },
        "timeouts": {
            "rollout_seconds": config.gepa.rollout_async_timeout_seconds,
            "container_http_seconds": config.policy.config.get("container_http_seconds"),
            "rollout_http_retries": config.policy.config.get("rollout_http_retries"),
        },
        "policy_io": config.policy.config.get("policy_io"),
        "proposer_io": {
            "timeout_seconds": config.proposer.timeout_seconds,
        },
        "adaptive_rollout_concurrency": config.gepa.pipeline.adaptive_rollout_concurrency.enabled,
    })
}

fn project_usage(usage: &Value, cost_usd: Option<f64>) -> Value {
    json!({
        "input_tokens": usage_u64(usage, &["input_tokens", "prompt_tokens", "prompt"]),
        "output_tokens": usage_u64(usage, &["output_tokens", "completion_tokens", "completion"]),
        "cost_usd": cost_usd
            .or_else(|| usage.get("cost_usd").and_then(Value::as_f64))
            .unwrap_or(0.0),
    })
}

fn project_outcome(request: &WorkspaceRunRequestStatus) -> Value {
    match project_request_status(&request.status) {
        "succeeded" => json!({
            "result": "succeeded",
            "stopped_by": request
                .result
                .get("stopped_by")
                .cloned()
                .unwrap_or_else(|| json!({"kind": "max_generations"})),
            "best": request.best_candidate_id.as_ref().map(|candidate_id| json!({
                "candidate_id": candidate_id,
                "heldout_score": request
                    .result
                    .get("best_candidate")
                    .and_then(|candidate| candidate.get("heldout_reward"))
                    .and_then(Value::as_f64)
                    .map(|score| json!(score))
                    .unwrap_or(Value::Null),
            })),
        }),
        "failed" => json!({
            "result": "failed",
            "error": {
                "kind": failure_class(&request.error),
                "message": request.error
                    .get("message")
                    .or_else(|| request.error.get("error"))
                    .and_then(Value::as_str)
                    .unwrap_or("run failed"),
            },
        }),
        "cancelled" => {
            let reason = match request.error.get("reason").and_then(Value::as_str) {
                Some("pause_timeout") => "pause_timeout",
                _ => "cancelled_by_user",
            };
            json!({
                "result": "cancelled",
                "reason": reason,
            })
        }
        _ => Value::Null,
    }
}

fn project_candidate(run_id: &str, candidate: Value) -> Value {
    let payload = candidate.get("payload").cloned().unwrap_or(Value::Null);
    let program = payload
        .as_object()
        .and_then(|object| {
            object
                .get("instructions")
                .or_else(|| object.get("prompt"))
                .or_else(|| object.get("system"))
        })
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            if payload.is_null() {
                String::new()
            } else {
                serde_json::to_string(&payload).unwrap_or_default()
            }
        });
    let status = candidate
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("");
    let train_score = candidate.get("train_reward").and_then(Value::as_f64);
    let heldout_score = candidate.get("heldout_reward").and_then(Value::as_f64);
    let seed_rewards = candidate
        .get("seed_rewards")
        .cloned()
        .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
        .unwrap_or_else(|| candidate_seed_rewards_from_value(&candidate));
    let seed_counts = candidate
        .get("seed_counts")
        .cloned()
        .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
        .unwrap_or_else(|| seed_counts_from_seed_rewards_value(&seed_rewards));
    let generation = candidate
        .get("generation")
        .and_then(Value::as_u64)
        .or_else(|| {
            candidate
                .get("acceptance_metadata")
                .and_then(|metadata| metadata.get("generation"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    json!({
        "candidate_id": candidate.get("candidate_id").and_then(Value::as_str).unwrap_or_default(),
        "run_id": candidate.get("run_id").and_then(Value::as_str).unwrap_or(run_id),
        "generation": generation,
        "parent_id": candidate.get("parent_id").cloned().unwrap_or(Value::Null),
        "accepted": matches!(status, "accepted" | "best" | "frontier"),
        "rejection_reason": candidate.get("rejection_reason").cloned().unwrap_or(Value::Null),
        "train_score": train_score,
        "heldout_score": heldout_score,
        "seed_counts": seed_counts,
        "seed_rewards": seed_rewards,
        "program": program,
    })
}

fn candidate_seed_rewards_from_value(candidate: &Value) -> Value {
    let mut grouped = BTreeMap::<String, Vec<Value>>::new();
    if let Some(frames) = candidate.get("sensor_frames").and_then(Value::as_array) {
        for frame in frames {
            let stage = frame
                .get("evaluation_stage")
                .and_then(Value::as_str)
                .map(seed_reward_stage)
                .unwrap_or_else(|| "unknown".to_string());
            grouped
                .entry(stage)
                .or_default()
                .push(seed_reward_row_from_frame_value(frame));
        }
    }
    if !grouped.contains_key("minibatch") {
        let rows = candidate
            .get("minibatch_scores")
            .and_then(Value::as_array)
            .map(|scores| seed_reward_rows_from_score_values("minibatch", scores))
            .unwrap_or_default();
        if !rows.is_empty() {
            grouped.insert("minibatch".to_string(), rows);
        }
    }
    if !grouped.contains_key("train") {
        let rows = candidate
            .get("train_scores")
            .and_then(Value::as_array)
            .map(|scores| seed_reward_rows_from_score_values("train", scores))
            .unwrap_or_default();
        if !rows.is_empty() {
            grouped.insert("train".to_string(), rows);
        }
    }
    grouped
        .values_mut()
        .for_each(|rows| rows.sort_by(seed_reward_row_cmp));
    Value::Object(
        grouped
            .into_iter()
            .map(|(stage, rows)| (stage, Value::Array(rows)))
            .collect(),
    )
}

fn seed_reward_stage(evaluation_stage: &str) -> String {
    match evaluation_stage {
        "candidate_minibatch" => "minibatch".to_string(),
        "candidate_full_train" | "seed_full_train" => "train".to_string(),
        "parent_minibatch_reference" => "parent_minibatch_reference".to_string(),
        "heldout" => "heldout".to_string(),
        other => other.to_string(),
    }
}

fn seed_reward_row_from_frame_value(frame: &Value) -> Value {
    let mut row = Map::new();
    let task_id = frame
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    row.insert("seed_id".to_string(), json!(task_id));
    for key in [
        "task_id",
        "example_id",
        "split",
        "evaluation_stage",
        "reward",
        "status",
        "success_status",
        "sensor_frame_id",
        "rollout_id",
    ] {
        if let Some(value) = frame.get(key) {
            row.insert(key.to_string(), value.clone());
        }
    }
    let metadata = compact_seed_reward_metadata_value(frame);
    if metadata
        .as_object()
        .is_some_and(|object| !object.is_empty())
    {
        row.insert("metadata".to_string(), metadata);
    }
    Value::Object(row)
}

fn compact_seed_reward_metadata_value(frame: &Value) -> Value {
    let mut metadata = Map::new();
    for key in [
        "actionable_side_info",
        "objective_scores",
        "usage",
        "trace_digest",
        "artifact_refs",
    ] {
        if let Some(value) = frame.get(key) {
            if !value.is_null() {
                metadata.insert(key.to_string(), value.clone());
            }
        }
    }
    if let Some(frame_metadata) = frame.get("metadata").and_then(Value::as_object) {
        for key in [
            "summary",
            "reward_details",
            "rollout_trace",
            "rollout_trace_artifact_refs",
        ] {
            if let Some(value) = frame_metadata.get(key) {
                metadata.insert(key.to_string(), value.clone());
            }
        }
    }
    Value::Object(metadata)
}

fn seed_reward_rows_from_score_values(stage: &str, scores: &[Value]) -> Vec<Value> {
    scores
        .iter()
        .map(|score| {
            let task_id = score
                .get("task_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            json!({
                "seed_id": task_id,
                "task_id": task_id,
                "example_id": score.get("example_id").cloned().unwrap_or(Value::Null),
                "evaluation_stage": stage,
                "reward": score.get("reward").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn seed_counts_from_seed_rewards_value(seed_rewards: &Value) -> Value {
    let mut counts = Map::new();
    if let Some(object) = seed_rewards.as_object() {
        for (stage, rows) in object {
            counts.insert(
                stage.clone(),
                json!(rows.as_array().map(Vec::len).unwrap_or(0)),
            );
        }
    }
    Value::Object(counts)
}

fn seed_reward_row_cmp(left: &Value, right: &Value) -> std::cmp::Ordering {
    value_string(left, "task_id")
        .cmp(&value_string(right, "task_id"))
        .then_with(|| value_string(left, "example_id").cmp(&value_string(right, "example_id")))
        .then_with(|| {
            value_string(left, "evaluation_stage").cmp(&value_string(right, "evaluation_stage"))
        })
}

fn project_rollout(
    record: &synth_optimizer_platform::RolloutRecord,
    trajectory: Option<Value>,
) -> Value {
    let tokens = usage_u64(
        &record.usage,
        &["total_tokens", "tokens", "input_tokens", "prompt_tokens"],
    );
    json!({
        "rollout_id": record.rollout_id.as_ref().unwrap_or(&record.rollout_record_id),
        "candidate_id": record.candidate_id,
        "task_id": record.task_id,
        "score": record.reward,
        "tokens": tokens,
        "latency_ms": record
            .metadata
            .get("latency_ms")
            .and_then(Value::as_u64),
        "trajectory": trajectory,
    })
}

fn run_workspace_store(request: &WorkspaceRunRequestStatus) -> Result<Option<WorkspaceStore>> {
    let path = request
        .run_workspace_db_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&request.run_dir).join("workspace.sqlite"));
    if path.exists() {
        WorkspaceStore::open_existing(path).map(Some)
    } else {
        Ok(None)
    }
}

fn latest_cursor_phase(request: &WorkspaceRunRequestStatus) -> Result<Option<String>> {
    let Some(run_store) = run_workspace_store(request)? else {
        return Ok(None);
    };
    Ok(load_gepa_cursor(&run_store, &request.run_id)?
        .map(|cursor| cursor.phase.as_str().to_string()))
}

fn stopped_by_for_request(request: &WorkspaceRunRequestStatus) -> Result<Option<Value>> {
    let Some(run_store) = run_workspace_store(request)? else {
        return Ok(None);
    };
    Ok(load_gepa_cursor(&run_store, &request.run_id)?
        .and_then(|cursor| cursor.terminal_summary)
        .and_then(|summary| summary.get("stopped_by").cloned()))
}

fn artifact_path_for_name(
    request: &WorkspaceRunRequestStatus,
    name: &str,
) -> Result<Option<PathBuf>> {
    let known = match name {
        "manifest.json" | "result_manifest.json" => request
            .result_manifest_path
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| {
                request
                    .result
                    .get("manifest_path")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
            }),
        "event_feed.jsonl" | "events.jsonl" => request
            .result
            .get("event_feed_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        "events.normalized.jsonl" | "normalized_event_feed.jsonl" => request
            .result
            .get("normalized_event_feed_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        "cache_profile.json" => request
            .result
            .get("cache_profile_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        "candidate_registry.json" => request
            .result
            .get("candidate_registry_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        "frontier.json" => request
            .result
            .get("frontier_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        "score_chart.svg" | "score_chart.png" => request
            .result
            .get("score_chart_path")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        _ => None,
    };
    if known.as_ref().is_some_and(|path| path.exists()) {
        return Ok(known);
    }
    let Some(run_store) = run_workspace_store(request)? else {
        return Ok(None);
    };
    Ok(run_store
        .view()
        .artifact_refs(&request.run_id)?
        .into_iter()
        .map(|artifact| PathBuf::from(artifact.path))
        .find(|path| path.file_name().and_then(|file| file.to_str()) == Some(name)))
}

fn artifact_content_type(name: &str) -> &'static str {
    if name.ends_with(".json") {
        "application/json"
    } else if name.ends_with(".jsonl") {
        "application/x-ndjson"
    } else if name.ends_with(".svg") {
        "image/svg+xml"
    } else if name.ends_with(".png") {
        "image/png"
    } else {
        "application/octet-stream"
    }
}

fn is_public_artifact_kind(kind: &str) -> bool {
    matches!(
        kind,
        "manifest"
            | "result_manifest"
            | "events_jsonl"
            | "events_normalized_jsonl"
            | "candidate_registry"
            | "frontier"
            | "score_chart"
            | "score_chart_svg"
    )
}

fn project_run_events(request: &WorkspaceRunRequestStatus) -> Result<Vec<ProjectedRunEvent>> {
    let path = event_feed_path_for_request(request);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(&path).map_err(|source| OptimizerError::io(&path, source))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| OptimizerError::io(&path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        let raw = serde_json::from_str::<Value>(&line)?;
        let raw_type = raw.get("type").and_then(Value::as_str).unwrap_or("event");
        let Some(kind) = public_event_kind(raw_type, &raw) else {
            continue;
        };
        let ts = raw
            .get("ts")
            .and_then(Value::as_str)
            .unwrap_or("1970-01-01T00:00:00Z")
            .to_string();
        let mut payload = scrub_event_payload(raw.get("fields").cloned().unwrap_or(Value::Null));
        if let Some(payload_object) = payload.as_object_mut() {
            payload_object.insert("source_event_type".to_string(), json!(raw_type));
            if let Some(message) = raw.get("message").and_then(Value::as_str) {
                payload_object.insert("message".to_string(), json!(message));
            }
        } else {
            payload = json!({
                "value": payload,
                "source_event_type": raw_type,
                "message": raw.get("message").and_then(Value::as_str).unwrap_or(""),
            });
        }
        events.push(ProjectedRunEvent {
            seq: events.len() as u64 + 1,
            ts,
            kind,
            payload,
        });
    }
    if events.len() > 50_000 {
        let keep_from = events.len() - 50_000;
        events.drain(0..keep_from);
    }
    Ok(events)
}

fn event_feed_path_for_request(request: &WorkspaceRunRequestStatus) -> PathBuf {
    request
        .result
        .get("event_feed_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&request.run_dir).join("events.jsonl"))
}

fn public_event_kind(event_type: &str, raw: &Value) -> Option<&'static str> {
    let event_type = normalize_key(event_type);
    if event_type.contains("run_finished")
        || event_type.contains("run_finished")
        || event_type.contains("run_failed")
        || event_type.contains("run_cancelled")
        || event_type.contains("run_completed")
        || event_type.contains("gepa_run_finished")
    {
        return Some("run.terminal");
    }
    if event_type.contains("transition")
        || event_type.contains("checkpoint")
        || event_type.contains("run_started")
        || event_type.contains("run_start")
    {
        return Some("run.status_changed");
    }
    if event_type.contains("generation") && event_type.contains("start") {
        return Some("generation.started");
    }
    if (event_type.contains("proposer") || event_type.contains("proposal"))
        && (event_type.contains("finished")
            || event_type.contains("completed")
            || event_type.contains("generated"))
    {
        return Some("proposer.completed");
    }
    if event_type.contains("candidate_accepted") || event_type.contains("candidate.accepted") {
        return Some("candidate.accepted");
    }
    if event_type.contains("candidate_rejected") || event_type.contains("candidate.rejected") {
        return Some("candidate.rejected");
    }
    if event_type.contains("candidate_evaluated")
        || event_type.contains("candidate.evaluated")
        || event_type.contains("candidate_scored")
        || event_type.contains("candidate.scored")
    {
        return Some("candidate.scored");
    }
    if event_type.contains("heldout") && event_type.contains("start") {
        return Some("heldout.started");
    }
    if event_type.contains("heldout") && event_type.contains("completed") {
        return Some("heldout.completed");
    }
    if event_type.contains("usage")
        || raw
            .get("fields")
            .is_some_and(|fields| fields.get("usage").is_some())
    {
        return Some("usage.tick");
    }
    if event_type.contains("frontier_updated") || event_type.contains("frontier.updated") {
        return Some("frontier.updated");
    }
    None
}

fn scrub_event_payload(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut scrubbed = Map::new();
            for (key, value) in object {
                let normalized = normalize_key(&key);
                if normalized.ends_with("_path")
                    || normalized.ends_with("_dir")
                    || normalized.contains("workspace_db")
                    || normalized.contains("registry_path")
                {
                    continue;
                }
                scrubbed.insert(key, scrub_event_payload(value));
            }
            Value::Object(scrubbed)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(scrub_event_payload).collect()),
        other => other,
    }
}

fn normalize_event_kinds(kinds: &[String]) -> BTreeSet<String> {
    kinds
        .iter()
        .map(|kind| kind.trim())
        .filter(|kind| !kind.is_empty())
        .map(str::to_string)
        .collect()
}

fn event_kind_matches(kinds: &BTreeSet<String>, kind: &str) -> bool {
    kinds.is_empty() || kinds.contains("*") || kinds.contains(kind)
}

fn project_request_status(status: &str) -> &'static str {
    match status {
        "queued" => "queued",
        "leased" | "running" => "running",
        "paused" => "paused",
        "completed" => "succeeded",
        "failed" => "failed",
        "cancelled" => "cancelled",
        _ => "failed",
    }
}

fn project_status_counts(counts: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
    let mut projected = BTreeMap::new();
    for (status, count) in counts {
        *projected
            .entry(project_request_status(status).to_string())
            .or_insert(0) += count;
    }
    projected
}

fn is_terminal_status(status: &str) -> bool {
    matches!(status, "succeeded" | "failed" | "cancelled")
}

fn terminal_cursor_phase(status: &str) -> &'static str {
    match project_request_status(status) {
        "succeeded" => "completed",
        "cancelled" => "cancelled",
        "failed" => "failed",
        _ => "initializing",
    }
}

fn failure_class(error: &Value) -> &'static str {
    let text = error
        .get("error_code")
        .or_else(|| error.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if text.contains("container") {
        "container_unreachable"
    } else if text.contains("budget") {
        "budget_exhausted"
    } else if text.contains("config") {
        "invalid_config"
    } else {
        "internal"
    }
}

fn sort_candidates(items: &mut [Value], sort: Option<&str>) -> Result<()> {
    let sort = sort.unwrap_or("candidate_id");
    let (descending, key) = sort
        .strip_prefix('-')
        .map(|key| (true, key))
        .unwrap_or((false, sort));
    if !matches!(
        key,
        "candidate_id" | "generation" | "train_score" | "heldout_score"
    ) {
        return Err(OptimizerError::Config(format!(
            "candidate sort must be candidate_id, generation, train_score, or heldout_score; got {sort:?}"
        )));
    }
    items.sort_by(|left, right| {
        let ordering = match key {
            "train_score" => value_f64(left, key)
                .partial_cmp(&value_f64(right, key))
                .unwrap_or(std::cmp::Ordering::Equal),
            "heldout_score" => value_f64(left, key)
                .partial_cmp(&value_f64(right, key))
                .unwrap_or(std::cmp::Ordering::Equal),
            "generation" => value_i64(left, key).cmp(&value_i64(right, key)),
            _ => value_string(left, "candidate_id").cmp(&value_string(right, "candidate_id")),
        };
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    Ok(())
}

fn paginate(mut items: Vec<Value>, query: &BTreeMap<String, String>) -> Value {
    let limit = query
        .get("limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50)
        .clamp(1, 200);
    let cursor = query
        .get("cursor")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let total = items.len();
    let page = if cursor >= total {
        Vec::new()
    } else {
        items
            .drain(cursor..usize::min(cursor + limit, total))
            .collect()
    };
    let next_cursor = if cursor + limit < total {
        Value::String((cursor + limit).to_string())
    } else {
        Value::Null
    };
    json!({
        "items": page,
        "next_cursor": next_cursor,
    })
}

fn query_bool(query: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    query
        .get(key)
        .and_then(|value| match normalize_key(value).as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        })
}

fn usage_u64(usage: &Value, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
        .unwrap_or(0)
}

fn value_string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn value_i64(value: &Value, key: &str) -> i64 {
    value.get(key).and_then(Value::as_i64).unwrap_or_default()
}

fn value_f64(value: &Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or(f64::NEG_INFINITY)
}

fn directory_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    if path.is_file() {
        return Ok(fs::metadata(path)
            .map_err(|source| OptimizerError::io(path, source))?
            .len());
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path).map_err(|source| OptimizerError::io(path, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(path, source))?;
        total = total.saturating_add(directory_bytes(&entry.path())?);
    }
    Ok(total)
}

fn seconds_since_sqlite_timestamp(timestamp: &str) -> Option<u64> {
    let timestamp_seconds = sqlite_timestamp_seconds(timestamp)?;
    let now_seconds = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    Some(now_seconds.saturating_sub(timestamp_seconds).max(0) as u64)
}

fn seconds_until_sqlite_timestamp(timestamp: &str) -> Option<i64> {
    let timestamp_seconds = sqlite_timestamp_seconds(timestamp)?;
    let now_seconds = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    Some(timestamp_seconds.saturating_sub(now_seconds))
}

fn sqlite_timestamp_seconds(timestamp: &str) -> Option<i64> {
    let (date, time) = timestamp.split_once(' ')?;
    let mut date_parts = date.split('-').filter_map(|part| part.parse::<i64>().ok());
    let year = date_parts.next()?;
    let month = date_parts.next()?;
    let day = date_parts.next()?;
    let mut time_parts = time.split(':').filter_map(|part| part.parse::<i64>().ok());
    let hour = time_parts.next()?;
    let minute = time_parts.next()?;
    let second = time_parts.next()?;
    Some(
        days_from_civil(year, month, day)?
            .saturating_mul(86_400)
            .saturating_add(hour.saturating_mul(3_600))
            .saturating_add(minute.saturating_mul(60))
            .saturating_add(second),
    )
}

fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let adjusted_year = year - i64::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

fn json_body<T>(request: &HttpRequest) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_slice(&request.body).map_err(OptimizerError::from)
}

fn request_body_sha256(body: &[u8]) -> String {
    let mut digest = Sha256::new();
    Sha2Digest::update(&mut digest, body);
    format!("{:x}", Sha2Digest::finalize(digest))
}

fn result_response(result: Result<Value>, success_status: u16) -> HttpResponse {
    match result {
        Ok(value) => json_response(success_status, &value),
        Err(OptimizerError::Config(message)) if message.starts_with("run not found:") => {
            let run_id = message.trim_start_matches("run not found:").trim();
            run_not_found_response(run_id)
        }
        Err(error) => optimizer_error_response(error),
    }
}

fn optimizer_error_response(error: OptimizerError) -> HttpResponse {
    if let OptimizerError::Config(message) = &error {
        if message.starts_with("idempotency_conflict:") {
            return error_response(409, "idempotency_conflict", message, None);
        }
    }
    let (status, code) = match &error {
        OptimizerError::Config(_) | OptimizerError::Json(_) | OptimizerError::TomlDecode(_) => {
            (422, "invalid_config")
        }
        OptimizerError::Container(_) | OptimizerError::Http(_) => (422, "invalid_config"),
        OptimizerError::StateTransition { .. } => (409, "invalid_transition"),
        OptimizerError::Cancelled { .. } => (409, "invalid_transition"),
        _ => (500, "internal"),
    };
    error_response(status, code, &error.to_string(), None)
}

fn run_not_found_response(run_id: &str) -> HttpResponse {
    error_response(
        404,
        "run_not_found",
        &format!("no Run with id {run_id:?}"),
        None,
    )
}

fn error_response(status: u16, code: &str, message: &str, details: Option<Value>) -> HttpResponse {
    json_response(
        status,
        &json!({
            "error": {
                "code": code,
                "message": message,
                "detail": details.unwrap_or(Value::Null),
            }
        }),
    )
}

fn json_response(status: u16, value: &Value) -> HttpResponse {
    HttpResponse {
        status,
        content_type: "application/json".to_string(),
        body: serde_json::to_vec_pretty(value).unwrap_or_else(|_| b"{}".to_vec()),
    }
}

fn text_response(status: u16, content_type: &str, text: &str) -> HttpResponse {
    HttpResponse {
        status,
        content_type: content_type.to_string(),
        body: text.as_bytes().to_vec(),
    }
}

fn binary_response(status: u16, content_type: &str, body: Vec<u8>) -> HttpResponse {
    HttpResponse {
        status,
        content_type: content_type.to_string(),
        body,
    }
}

fn empty_response(status: u16) -> HttpResponse {
    HttpResponse {
        status,
        content_type: "application/octet-stream".to_string(),
        body: Vec::new(),
    }
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> Result<()> {
    let reason = match response.status {
        400 => "Bad Request",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Entity",
        501 => "Not Implemented",
        _ => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        reason,
        response.content_type,
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

fn is_websocket_upgrade(request: &HttpRequest) -> bool {
    request
        .headers
        .get("upgrade")
        .is_some_and(|value| value.eq_ignore_ascii_case("websocket"))
        && request
            .headers
            .get("connection")
            .is_some_and(|value| value.to_ascii_lowercase().contains("upgrade"))
        && request.headers.contains_key("sec-websocket-key")
}

fn write_websocket_handshake(stream: &mut TcpStream, request: &HttpRequest) -> Result<()> {
    let key = request
        .headers
        .get("sec-websocket-key")
        .ok_or_else(|| OptimizerError::Config("missing sec-websocket-key".to_string()))?;
    let mut digest = Sha1::new();
    Sha1Digest::update(&mut digest, key.as_bytes());
    Sha1Digest::update(&mut digest, b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let accept = base64::engine::general_purpose::STANDARD.encode(Sha1Digest::finalize(digest));
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|source| OptimizerError::io("gepa service websocket handshake", source))?;
    stream
        .flush()
        .map_err(|source| OptimizerError::io("gepa service websocket handshake", source))
}

fn read_websocket_frame(stream: &mut TcpStream) -> Result<WebSocketIncoming> {
    let mut header = [0u8; 2];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(source)
            if matches!(
                source.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) =>
        {
            return Ok(WebSocketIncoming::Empty);
        }
        Err(source) if source.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Ok(WebSocketIncoming::Closed);
        }
        Err(source) => {
            return Err(OptimizerError::io("gepa service websocket frame", source));
        }
    }
    let opcode = header[0] & 0x0f;
    let masked = header[1] & 0x80 != 0;
    let mut len = (header[1] & 0x7f) as u64;
    if len == 126 {
        let mut extended = [0u8; 2];
        stream
            .read_exact(&mut extended)
            .map_err(|source| OptimizerError::io("gepa service websocket frame", source))?;
        len = u16::from_be_bytes(extended) as u64;
    } else if len == 127 {
        let mut extended = [0u8; 8];
        stream
            .read_exact(&mut extended)
            .map_err(|source| OptimizerError::io("gepa service websocket frame", source))?;
        len = u64::from_be_bytes(extended);
    }
    if len > 65_536 {
        return Err(OptimizerError::Config(
            "WebSocket client frame exceeds 64 KiB".to_string(),
        ));
    }
    let mut mask = [0u8; 4];
    if masked {
        stream
            .read_exact(&mut mask)
            .map_err(|source| OptimizerError::io("gepa service websocket mask", source))?;
    } else if matches!(opcode, 0x1 | 0x8 | 0x9 | 0xA) {
        return Err(OptimizerError::Config(
            "WebSocket client frames must be masked".to_string(),
        ));
    }
    let mut payload = vec![0u8; len as usize];
    if len > 0 {
        stream
            .read_exact(&mut payload)
            .map_err(|source| OptimizerError::io("gepa service websocket payload", source))?;
    }
    if masked {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[index % 4];
        }
    }
    match opcode {
        0x1 => String::from_utf8(payload)
            .map(WebSocketIncoming::Text)
            .map_err(|error| OptimizerError::Config(error.to_string())),
        0x8 => Ok(WebSocketIncoming::Closed),
        0x9 => {
            write_websocket_frame(stream, 0xA, &payload)?;
            Ok(WebSocketIncoming::Empty)
        }
        0xA => Ok(WebSocketIncoming::Empty),
        _ => Err(OptimizerError::Config(format!(
            "unsupported WebSocket opcode {opcode}"
        ))),
    }
}

fn write_websocket_json(stream: &mut TcpStream, value: &Value) -> Result<()> {
    let text = serde_json::to_vec(value)?;
    write_websocket_frame(stream, 0x1, &text)
}

fn write_websocket_close(stream: &mut TcpStream) -> Result<()> {
    write_websocket_frame(stream, 0x8, &[])
}

fn write_websocket_frame(stream: &mut TcpStream, opcode: u8, payload: &[u8]) -> Result<()> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x80 | (opcode & 0x0f));
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    stream
        .write_all(&frame)
        .map_err(|source| OptimizerError::io("gepa service websocket write", source))?;
    stream
        .flush()
        .map_err(|source| OptimizerError::io("gepa service websocket write", source))
}

fn default_auto_start() -> bool {
    true
}

fn default_api_family() -> String {
    "chat_completions".to_string()
}

fn default_disable_reasoning() -> String {
    "auto".to_string()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn split_path_query(path: &str) -> (&str, BTreeMap<String, String>) {
    let (path, raw_query) = path.split_once('?').unwrap_or((path, ""));
    let mut query = BTreeMap::new();
    for pair in raw_query.split('&').filter(|item| !item.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        query.insert(key.to_string(), value.to_string());
    }
    (path, query)
}

fn path_segments(path: &str) -> Vec<&str> {
    path.trim_matches('/')
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
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
        Ok(Self {
            method,
            path,
            headers,
            body,
        })
    }
}

struct HttpResponse {
    status: u16,
    content_type: String,
    body: Vec<u8>,
}

const GEPA_SERVICE_OPENAPI_YAML: &str = include_str!("../openapi/gepa-service-v1.yaml");
