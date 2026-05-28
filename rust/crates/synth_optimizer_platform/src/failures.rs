use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::error::OptimizerError;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerFailureType {
    Validation,
    Auth,
    Conflict,
    Quota,
    Dependency,
    Timeout,
    Backpressure,
    Upstream,
    Container,
    ContainerFailed,
    ContainerError,
    ContainerTerminated,
    ContainerCancelled,
    PromptOverlay,
    Cache,
    CacheMiss,
    CacheFull,
    CacheCorrupt,
    Budget,
    Cancelled,
    Proposer,
    Verifier,
    StateTransition,
    Invariant,
    Io,
    DiskBudget,
    Json,
    Toml,
    Http,
    Sqlite,
    Internal,
}

impl OptimizerFailureType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Auth => "auth",
            Self::Conflict => "conflict",
            Self::Quota => "quota",
            Self::Dependency => "dependency",
            Self::Timeout => "timeout",
            Self::Backpressure => "backpressure",
            Self::Upstream => "upstream",
            Self::Container => "container",
            Self::ContainerFailed => "container_failed",
            Self::ContainerError => "container_error",
            Self::ContainerTerminated => "container_terminated",
            Self::ContainerCancelled => "container_cancelled",
            Self::PromptOverlay => "prompt_overlay",
            Self::Cache => "cache",
            Self::CacheMiss => "cache_miss",
            Self::CacheFull => "cache_full",
            Self::CacheCorrupt => "cache_corrupt",
            Self::Budget => "budget",
            Self::Cancelled => "cancelled",
            Self::Proposer => "proposer",
            Self::Verifier => "verifier",
            Self::StateTransition => "state_transition",
            Self::Invariant => "invariant",
            Self::Io => "io",
            Self::DiskBudget => "disk_budget",
            Self::Json => "json",
            Self::Toml => "toml",
            Self::Http => "http",
            Self::Sqlite => "sqlite",
            Self::Internal => "internal",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailurePayload {
    pub failure_type: OptimizerFailureType,
    pub reason_code: String,
    pub message: String,
    pub retryable: bool,
    pub mutated: bool,
    #[serde(default)]
    pub details: Map<String, Value>,
}

impl FailurePayload {
    pub fn new(
        failure_type: OptimizerFailureType,
        reason_code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Self {
        Self {
            failure_type,
            reason_code: reason_code.into(),
            message: message.into(),
            retryable,
            mutated: false,
            details: Map::new(),
        }
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: Value) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    pub fn with_mutated(mut self, mutated: bool) -> Self {
        self.mutated = mutated;
        self
    }

    pub fn failure_class(&self) -> &'static str {
        self.failure_type.as_str()
    }

    pub fn from_optimizer_error(error: &OptimizerError) -> Self {
        match error {
            OptimizerError::Config(message) => Self::new(
                OptimizerFailureType::Validation,
                "config_error",
                message,
                false,
            ),
            OptimizerError::Container(message) => {
                let failure_type = classify_container_message(message);
                let reason_code = failure_type.as_str();
                Self::new(
                    failure_type,
                    reason_code,
                    message,
                    retryable_container_message(message),
                )
            }
            OptimizerError::Proposer(message) => {
                let failure_type = if contains_any(message, &["timeout", "timed out", "deadline"]) {
                    OptimizerFailureType::Timeout
                } else {
                    OptimizerFailureType::Proposer
                };
                let reason_code = failure_type.as_str();
                Self::new(failure_type, reason_code, message, true)
            }
            OptimizerError::CacheMiss {
                namespace,
                cache_key,
            } => Self::new(
                OptimizerFailureType::CacheMiss,
                "cache_miss",
                error.to_string(),
                false,
            )
            .with_detail("namespace", json!(namespace))
            .with_detail("cache_key", json!(cache_key)),
            OptimizerError::CacheFull {
                path,
                size_bytes,
                max_bytes,
            } => Self::new(
                OptimizerFailureType::CacheFull,
                "cache_full",
                error.to_string(),
                false,
            )
            .with_detail("path", json!(path.display().to_string()))
            .with_detail("size_bytes", json!(size_bytes))
            .with_detail("max_bytes", json!(max_bytes)),
            OptimizerError::CacheCorrupt {
                path,
                operation,
                message,
            } => Self::new(
                OptimizerFailureType::CacheCorrupt,
                "cache_corrupt",
                message,
                false,
            )
            .with_detail("path", json!(path.display().to_string()))
            .with_detail("operation", json!(operation)),
            OptimizerError::BudgetExceeded { .. } => Self::new(
                OptimizerFailureType::Budget,
                "budget_exceeded",
                error.to_string(),
                false,
            ),
            OptimizerError::Cancelled { request_id } => Self::new(
                OptimizerFailureType::Cancelled,
                "cancelled",
                error.to_string(),
                false,
            )
            .with_detail("request_id", json!(request_id)),
            OptimizerError::EventCompare(message) => Self::new(
                OptimizerFailureType::Validation,
                "event_compare_failed",
                message,
                false,
            ),
            OptimizerError::Failed(message) => {
                Self::new(OptimizerFailureType::Internal, "run_failed", message, false)
            }
            OptimizerError::Invariant(message) => Self::new(
                OptimizerFailureType::Invariant,
                "invariant_violation",
                message,
                false,
            ),
            OptimizerError::StateTransition { from, to, trigger } => Self::new(
                OptimizerFailureType::StateTransition,
                "invalid_state_transition",
                error.to_string(),
                false,
            )
            .with_detail("from", json!(from))
            .with_detail("to", json!(to))
            .with_detail("trigger", json!(trigger)),
            OptimizerError::Io { path, source } => Self::new(
                OptimizerFailureType::Io,
                "io_error",
                source.to_string(),
                true,
            )
            .with_detail("path", json!(path.display().to_string())),
            OptimizerError::DiskBudgetExceeded {
                path,
                used_bytes,
                limit_bytes,
                limit_kind,
            } => Self::new(
                OptimizerFailureType::DiskBudget,
                "disk_budget_exceeded",
                error.to_string(),
                // Hard-limit failures are not retryable in-process — the
                // operator must prune the runs dir or raise the limit
                // before the next launch can succeed. Soft-limit hits
                // are launch-only and equally non-retryable.
                false,
            )
            .with_detail("path", json!(path.display().to_string()))
            .with_detail("used_bytes", json!(used_bytes))
            .with_detail("limit_bytes", json!(limit_bytes))
            .with_detail("limit_kind", json!(limit_kind)),
            OptimizerError::Json(error) => Self::new(
                OptimizerFailureType::Json,
                "json_error",
                error.to_string(),
                false,
            ),
            OptimizerError::TomlDecode(error) => Self::new(
                OptimizerFailureType::Toml,
                "toml_decode_error",
                error.to_string(),
                false,
            ),
            OptimizerError::Http(error) => Self::new(
                OptimizerFailureType::Http,
                "http_error",
                error.to_string(),
                true,
            )
            .with_detail(
                "status",
                json!(error.status().map(|status| status.as_u16())),
            ),
            OptimizerError::Sqlite(error) => Self::new(
                OptimizerFailureType::Sqlite,
                "sqlite_error",
                error.to_string(),
                true,
            ),
        }
    }
}

fn classify_container_message(message: &str) -> OptimizerFailureType {
    let normalized = message.to_ascii_lowercase();
    if contains_any(&normalized, &["timeout", "timed out", "deadline"]) {
        OptimizerFailureType::Timeout
    } else if contains_any(&normalized, &["cancelled", "canceled"]) {
        OptimizerFailureType::ContainerCancelled
    } else if contains_any(&normalized, &["terminated", "killed", "signal"]) {
        OptimizerFailureType::ContainerTerminated
    } else if contains_any(&normalized, &["failed", "non-zero", "nonzero"]) {
        OptimizerFailureType::ContainerFailed
    } else if contains_any(&normalized, &["overlay", "patch", "template"]) {
        OptimizerFailureType::PromptOverlay
    } else {
        OptimizerFailureType::ContainerError
    }
}

fn retryable_container_message(message: &str) -> bool {
    !contains_any(
        &message.to_ascii_lowercase(),
        &["validation", "invalid", "schema", "contract", "unsupported"],
    )
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}
