use serde::{Deserialize, Serialize};

use crate::{OptimizerError, Result};

/// Where the proposer agent process runs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSubstrate {
    #[default]
    Local,
    Docker,
    Daytona,
}

impl ExecutionSubstrate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Docker => "docker",
            Self::Daytona => "daytona",
        }
    }
}

impl std::fmt::Display for ExecutionSubstrate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub fn validate_execution_mode_compat(execution_mode: &str) -> Result<()> {
    if normalize_execution_mode(execution_mode).is_some() {
        return Ok(());
    }
    Err(OptimizerError::Config(format!(
        "unsupported proposer.execution_mode {execution_mode:?}; expected stdio/local_process \
         or websocket/ws"
    )))
}

pub fn normalize_execution_mode(execution_mode: &str) -> Option<&'static str> {
    match execution_mode.trim().to_ascii_lowercase().as_str() {
        "stdio" | "local_process" => Some("stdio"),
        "websocket" | "ws" => Some("websocket"),
        _ => None,
    }
}
