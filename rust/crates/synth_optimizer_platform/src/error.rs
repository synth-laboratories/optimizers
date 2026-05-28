use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, OptimizerError>;

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("container error: {0}")]
    Container(String),
    #[error("proposer error: {0}")]
    Proposer(String),
    #[error("cache miss namespace={namespace} key={cache_key}")]
    CacheMiss {
        namespace: String,
        cache_key: String,
    },
    #[error("cache full path={path} size={size_bytes} max={max_bytes}")]
    CacheFull {
        path: PathBuf,
        size_bytes: u64,
        max_bytes: u64,
    },
    #[error("cache corrupt operation={operation} path={path}: {message}")]
    CacheCorrupt {
        path: PathBuf,
        operation: String,
        message: String,
    },
    #[error(
        "budget exceeded run_id={run_id} limit={limit} requested={requested} available={available}"
    )]
    BudgetExceeded {
        run_id: String,
        limit: String,
        requested: String,
        available: String,
    },
    #[error("optimizer run cancelled request_id={request_id}")]
    Cancelled { request_id: String },
    #[error("event feed compare failed: {0}")]
    EventCompare(String),
    #[error("optimizer run failed: {0}")]
    Failed(String),
    #[error("optimizer invariant violated: {0}")]
    Invariant(String),
    #[error("invalid optimizer state transition {from} -> {to} for trigger {trigger}")]
    StateTransition {
        from: String,
        to: String,
        trigger: String,
    },
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "disk budget {limit_kind} limit exceeded path={path} used_bytes={used_bytes} limit_bytes={limit_bytes}"
    )]
    DiskBudgetExceeded {
        path: PathBuf,
        used_bytes: u64,
        limit_bytes: u64,
        limit_kind: &'static str,
    },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("toml decode error: {0}")]
    TomlDecode(#[from] toml::de::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

impl OptimizerError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Config(_) => "synth_optimizer_config_error",
            Self::Container(_) => "synth_optimizer_container_error",
            Self::Proposer(_) => "synth_optimizer_proposer_error",
            Self::CacheMiss { .. } => "synth_optimizer_cache_miss",
            Self::CacheFull { .. } => "synth_optimizer_cache_full",
            Self::CacheCorrupt { .. } => "synth_optimizer_cache_corrupt",
            Self::BudgetExceeded { .. } => "synth_optimizer_budget_exceeded",
            Self::Cancelled { .. } => "synth_optimizer_cancelled",
            Self::EventCompare(_) => "synth_optimizer_event_compare_failed",
            Self::Failed(_) => "synth_optimizer_failed",
            Self::Invariant(_) => "synth_optimizer_invariant_error",
            Self::StateTransition { .. } => "synth_optimizer_state_transition_error",
            Self::Io { .. } => "synth_optimizer_io_error",
            Self::DiskBudgetExceeded { .. } => "synth_optimizer_disk_budget_exceeded",
            Self::Json(_) => "synth_optimizer_json_error",
            Self::TomlDecode(_) => "synth_optimizer_toml_decode_error",
            Self::Http(_) => "synth_optimizer_http_error",
            Self::Sqlite(_) => "synth_optimizer_sqlite_error",
        }
    }
}
