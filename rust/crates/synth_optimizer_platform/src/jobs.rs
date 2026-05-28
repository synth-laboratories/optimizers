use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::failures::FailurePayload;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerJobKind {
    Rollout,
    Annotation,
    Verification,
    Subagent,
    Proposer,
    Checkpoint,
}

impl OptimizerJobKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rollout => "rollout",
            Self::Annotation => "annotation",
            Self::Verification => "verification",
            Self::Subagent => "subagent",
            Self::Proposer => "proposer",
            Self::Checkpoint => "checkpoint",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "rollout" => Some(Self::Rollout),
            "annotation" => Some(Self::Annotation),
            "verification" => Some(Self::Verification),
            "subagent" => Some(Self::Subagent),
            "proposer" => Some(Self::Proposer),
            "checkpoint" => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerJobStatus {
    Pending,
    Leased,
    Running,
    Annotating,
    Verifying,
    Completed,
    Failed,
    Cancelled,
    Expired,
    RetryScheduled,
}

impl OptimizerJobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Leased => "leased",
            Self::Running => "running",
            Self::Annotating => "annotating",
            Self::Verifying => "verifying",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
            Self::RetryScheduled => "retry_scheduled",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "leased" => Some(Self::Leased),
            "running" => Some(Self::Running),
            "annotating" => Some(Self::Annotating),
            "verifying" => Some(Self::Verifying),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" | "canceled" => Some(Self::Cancelled),
            "expired" => Some(Self::Expired),
            "retry_scheduled" => Some(Self::RetryScheduled),
            _ => None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Expired
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_seconds: u64,
    #[serde(default)]
    pub retryable_failure_types: Vec<String>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            backoff_seconds: 0,
            retryable_failure_types: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerJob {
    pub job_id: String,
    pub run_id: String,
    pub kind: OptimizerJobKind,
    pub status: OptimizerJobStatus,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub attempt: u32,
    #[serde(default)]
    pub lease_id: Option<String>,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub leased_at: Option<String>,
    #[serde(default)]
    pub lease_expires_at: Option<String>,
    #[serde(default)]
    pub heartbeat_at: Option<String>,
    #[serde(default)]
    pub next_retry_at: Option<String>,
    #[serde(default)]
    pub retry_policy: RetryPolicy,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub payload: Map<String, Value>,
}

impl OptimizerJob {
    pub fn new(
        job_id: impl Into<String>,
        run_id: impl Into<String>,
        kind: OptimizerJobKind,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            run_id: run_id.into(),
            kind,
            status: OptimizerJobStatus::Pending,
            candidate_id: None,
            attempt: 0,
            lease_id: None,
            worker_id: None,
            leased_at: None,
            lease_expires_at: None,
            heartbeat_at: None,
            next_retry_at: None,
            retry_policy: RetryPolicy::default(),
            failure: None,
            payload: Map::new(),
        }
    }
}
