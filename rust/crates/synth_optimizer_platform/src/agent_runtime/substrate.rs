use serde::{Deserialize, Serialize};

use crate::{OptimizerError, Result};

/// Where the proposer agent process runs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSubstrate {
    #[default]
    Local,
    Docker,
}

impl ExecutionSubstrate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Docker => "docker",
        }
    }
}

impl std::fmt::Display for ExecutionSubstrate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub fn validate_execution_mode_compat(execution_mode: &str) -> Result<()> {
    if execution_mode.trim() == "local_process" {
        return Ok(());
    }
    Err(OptimizerError::Config(format!(
        "unsupported proposer.execution_mode {execution_mode:?}; use \
         proposer.runtime_substrate = \"local\" or \"docker\" and leave \
         execution_mode = \"local_process\" during migration"
    )))
}
