use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{OptimizerError, Result};

pub const PROMPT_PROGRAM_VERSION: &str = "prompt_program.v1";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PromptProgram {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub program_id: String,
    #[serde(default)]
    pub modules: Vec<PromptModule>,
    #[serde(default)]
    pub target_modules: Vec<TargetModule>,
    #[serde(default)]
    pub seed_candidate: PromptCandidatePayload,
    #[serde(default)]
    pub rollout_overlay_schema: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl PromptProgram {
    pub fn from_value(value: Value) -> serde_json::Result<Self> {
        serde_json::from_value(value)
    }

    pub fn mutable_field_ids(&self) -> Vec<String> {
        self.modules
            .iter()
            .filter(|module| module.mutable)
            .map(|module| module.candidate_field.clone())
            .filter(|field| !field.trim().is_empty())
            .collect()
    }

    pub fn validate_for_gepa(
        &self,
        configured_target_fields: &[String],
        configured_seed_fields: &BTreeMap<String, String>,
    ) -> Result<()> {
        if self.version.trim() != PROMPT_PROGRAM_VERSION {
            return Err(OptimizerError::Container(format!(
                "/program.version must be {PROMPT_PROGRAM_VERSION:?}, got {:?}",
                self.version
            )));
        }
        if self.program_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "/program.program_id is required".to_string(),
            ));
        }
        if self.modules.is_empty() {
            return Err(OptimizerError::Container(
                "/program.modules must contain at least one module".to_string(),
            ));
        }
        if self.target_modules.is_empty() {
            return Err(OptimizerError::Container(
                "/program.target_modules must contain at least one target".to_string(),
            ));
        }

        let mut mutable_fields = BTreeSet::new();
        for module in &self.modules {
            if module.module_id.trim().is_empty() {
                return Err(OptimizerError::Container(
                    "/program.modules[].module_id is required".to_string(),
                ));
            }
            if module.mutable {
                if module.candidate_field.trim().is_empty() {
                    return Err(OptimizerError::Container(format!(
                        "/program module {:?} is mutable but has no candidate_field",
                        module.module_id
                    )));
                }
                mutable_fields.insert(module.candidate_field.clone());
            }
        }
        if mutable_fields.is_empty() {
            return Err(OptimizerError::Container(
                "/program must declare at least one mutable module with candidate_field"
                    .to_string(),
            ));
        }

        for target in &self.target_modules {
            if target.module_id.trim().is_empty() {
                return Err(OptimizerError::Container(
                    "/program.target_modules[].module_id is required".to_string(),
                ));
            }
            if target.candidate_field.trim().is_empty() {
                return Err(OptimizerError::Container(format!(
                    "/program target module {:?} has no candidate_field",
                    target.module_id
                )));
            }
            if !mutable_fields.contains(&target.candidate_field) {
                return Err(OptimizerError::Container(format!(
                    "/program target field {:?} is not declared as a mutable module candidate_field",
                    target.candidate_field
                )));
            }
        }

        for field in configured_target_fields {
            if !mutable_fields.contains(field) {
                return Err(OptimizerError::Config(format!(
                    "candidate.target_modules contains {field:?}, but /program does not declare it as mutable"
                )));
            }
        }

        let seed_fields = if configured_seed_fields.is_empty() {
            &self.seed_candidate.fields
        } else {
            configured_seed_fields
        };
        if seed_fields.is_empty() {
            return Err(OptimizerError::Config(
                "seed candidate must be provided by [seed_candidate] or /program.seed_candidate"
                    .to_string(),
            ));
        }
        for (field, value) in seed_fields {
            if !mutable_fields.contains(field) {
                return Err(OptimizerError::Config(format!(
                    "seed candidate field {field:?} is not declared as mutable by /program"
                )));
            }
            if value.trim().is_empty() {
                return Err(OptimizerError::Config(format!(
                    "seed candidate field {field:?} must be non-empty"
                )));
            }
        }

        if let Some(candidate_fields) = self
            .rollout_overlay_schema
            .get("candidate_fields")
            .and_then(Value::as_array)
        {
            for value in candidate_fields {
                let Some(field) = value.as_str() else {
                    return Err(OptimizerError::Container(
                        "/program.rollout_overlay_schema.candidate_fields must be strings"
                            .to_string(),
                    ));
                };
                if !mutable_fields.contains(field) {
                    return Err(OptimizerError::Container(format!(
                        "/program.rollout_overlay_schema candidate field {field:?} is not mutable"
                    )));
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PromptModule {
    pub module_id: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub mutable: bool,
    #[serde(default)]
    pub candidate_field: String,
    #[serde(default)]
    pub template_variables: Vec<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TargetModule {
    pub module_id: String,
    #[serde(default)]
    pub candidate_field: String,
    #[serde(default)]
    pub objective: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PromptCandidatePayload {
    #[serde(flatten)]
    pub fields: BTreeMap<String, String>,
}

impl PromptCandidatePayload {
    pub fn from_map(fields: BTreeMap<String, String>) -> Self {
        Self { fields }
    }

    pub fn to_value(&self) -> Value {
        let mut map = Map::new();
        for (key, value) in &self.fields {
            map.insert(key.clone(), Value::String(value.clone()));
        }
        Value::Object(map)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CandidateOverlay {
    #[serde(default)]
    pub candidate: PromptCandidatePayload,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}
