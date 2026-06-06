use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::{OptimizerError, ProposerConfig, Result, RuntimeEffectBudgetEstimate};

use super::session::CodexTurnRequest;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleAgentConfig {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub output_schema: Option<String>,
    #[serde(default)]
    pub budget_estimate: RuntimeEffectBudgetEstimate,
    #[serde(flatten)]
    pub proposer: ProposerConfig,
}

impl Default for RoleAgentConfig {
    fn default() -> Self {
        Self {
            role: String::new(),
            output_schema: None,
            budget_estimate: RuntimeEffectBudgetEstimate::default(),
            proposer: ProposerConfig::default(),
        }
    }
}

impl RoleAgentConfig {
    pub fn resolve(
        &self,
        role: impl Into<String>,
        fallback_model: impl AsRef<str>,
    ) -> Result<ResolvedRoleAgentConfig> {
        let mut proposer = self.proposer.clone();
        let role = non_empty(Some(self.role.as_str()))
            .map(str::to_string)
            .unwrap_or_else(|| role.into());
        if role.trim().is_empty() {
            return Err(OptimizerError::Config(
                "role agent config requires a non-empty role".to_string(),
            ));
        }
        let model = non_empty(proposer.model.as_deref())
            .map(str::to_string)
            .or_else(|| non_empty(Some(fallback_model.as_ref())).map(str::to_string))
            .ok_or_else(|| {
                OptimizerError::Config(format!(
                    "{role} role agent requires a model or fallback model"
                ))
            })?;
        proposer.model = Some(model.clone());
        Ok(ResolvedRoleAgentConfig {
            role,
            proposer,
            model,
            output_schema: self.output_schema.clone(),
            budget_estimate: self.budget_estimate,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedRoleAgentConfig {
    pub role: String,
    pub proposer: ProposerConfig,
    pub model: String,
    pub output_schema: Option<String>,
    pub budget_estimate: RuntimeEffectBudgetEstimate,
}

impl ResolvedRoleAgentConfig {
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.proposer.timeout_seconds.max(1))
    }

    pub fn codex_turn_request<'a>(
        &'a self,
        input: RoleAgentTurnRequestInput<'a>,
    ) -> CodexTurnRequest<'a> {
        CodexTurnRequest {
            run_id: input.run_id,
            proposer: &self.proposer,
            workspace_dir: input.workspace_dir,
            model: &self.model,
            client_name: input.client_name,
            client_title: input.client_title,
            client_version: input.client_version,
            thread_start_params: self
                .thread_start_params(input.thread_instructions, input.thread_metadata),
            turn_start_params: self.turn_start_params(input.turn_input),
            timeout: input.timeout.unwrap_or_else(|| self.timeout()),
        }
    }

    pub fn thread_start_params(
        &self,
        instructions: impl AsRef<str>,
        metadata: Option<Value>,
    ) -> Value {
        let mut params = Map::new();
        params.insert("model".to_string(), Value::String(self.model.clone()));
        params.insert(
            "instructions".to_string(),
            Value::String(instructions.as_ref().to_string()),
        );
        if let Some(approval_policy) = non_empty(self.proposer.approval_policy.as_deref()) {
            params.insert(
                "approvalPolicy".to_string(),
                Value::String(approval_policy.to_string()),
            );
        }
        if let Some(sandbox_mode) = non_empty(self.proposer.sandbox_mode.as_deref()) {
            params.insert(
                "sandbox".to_string(),
                Value::String(sandbox_mode.to_string()),
            );
        }
        if let Some(metadata) = metadata {
            params.insert("metadata".to_string(), metadata);
        }
        Value::Object(params)
    }

    pub fn turn_start_params(&self, input: Value) -> Value {
        let mut params = Map::new();
        params.insert("model".to_string(), Value::String(self.model.clone()));
        params.insert("input".to_string(), normalize_turn_input(input));
        if let Some(reasoning_effort) = non_empty(self.proposer.reasoning_effort.as_deref()) {
            params.insert(
                "effort".to_string(),
                Value::String(reasoning_effort.to_string()),
            );
        }
        if let Some(service_tier) = non_empty(self.proposer.service_tier.as_deref()) {
            params.insert(
                "serviceTier".to_string(),
                Value::String(service_tier.to_string()),
            );
        }
        if let Some(approval_policy) = non_empty(self.proposer.approval_policy.as_deref()) {
            params.insert(
                "approvalPolicy".to_string(),
                Value::String(approval_policy.to_string()),
            );
        }
        if let Some(sandbox_mode) = non_empty(self.proposer.sandbox_mode.as_deref()) {
            params.insert(
                "sandboxPolicy".to_string(),
                sandbox_policy_for_mode(sandbox_mode),
            );
        }
        Value::Object(params)
    }
}

pub struct RoleAgentTurnRequestInput<'a> {
    pub run_id: &'a str,
    pub workspace_dir: &'a Path,
    pub client_name: &'a str,
    pub client_title: &'a str,
    pub client_version: &'a str,
    pub thread_instructions: &'a str,
    pub thread_metadata: Option<Value>,
    pub turn_input: Value,
    pub timeout: Option<Duration>,
}

pub fn text_turn_input(text: impl Into<String>) -> Value {
    Value::Array(vec![json!({
        "type": "text",
        "text": text.into(),
        "textElements": [],
    })])
}

pub fn sandbox_policy_for_mode(mode: &str) -> Value {
    match mode.trim() {
        "read-only" | "read_only" => json!({
            "type": "readOnly",
            "readOnlyAccess": {"type": "fullAccess"},
        }),
        "workspace-write" | "workspace_write" => json!({
            "type": "workspaceWrite",
            "readOnlyAccess": {"type": "fullAccess"},
            "networkAccess": true,
        }),
        _ => Value::String(mode.to_string()),
    }
}

fn normalize_turn_input(input: Value) -> Value {
    match input {
        Value::String(text) => text_turn_input(text),
        Value::Array(_) => input,
        other => Value::Array(vec![other]),
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
