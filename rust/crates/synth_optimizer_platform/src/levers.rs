use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::prompt_program::PromptProgram;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeverKind {
    TextPrompt,
    SystemPrompt,
    UserPrompt,
    AgentsMd,
    SkillMd,
    WorkspaceFile,
    ToolPolicy,
    ConfigAppend,
    VerifierRubric,
    ActionPolicy,
    PolicyScript,
    SourcedPython,
    HarnessModule,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverSpec {
    pub lever_id: String,
    pub kind: LeverKind,
    pub mutable: bool,
    pub required: bool,
    #[serde(default)]
    pub candidate_field: Option<String>,
    #[serde(default)]
    pub module_id: Option<String>,
    #[serde(default)]
    pub template_variables: Vec<String>,
    #[serde(default)]
    pub constraints: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverManifest {
    pub version: String,
    pub program_id: String,
    #[serde(default)]
    pub levers: Vec<LeverSpec>,
    #[serde(default)]
    pub target_levers: Vec<String>,
    #[serde(default)]
    pub seed_bundle: Option<LeverBundle>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl LeverManifest {
    pub fn from_prompt_program(program: &PromptProgram) -> Self {
        let levers = program
            .modules
            .iter()
            .map(|module| {
                let candidate_field = if module.candidate_field.trim().is_empty() {
                    module.module_id.clone()
                } else {
                    module.candidate_field.clone()
                };
                let kind = module
                    .metadata
                    .get("lever_kind")
                    .or_else(|| module.metadata.get("kind"))
                    .and_then(Value::as_str)
                    .and_then(parse_lever_kind)
                    .unwrap_or_else(|| prompt_role_to_lever_kind(&module.role));
                let constraints = module
                    .metadata
                    .get("constraints")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                LeverSpec {
                    lever_id: candidate_field.clone(),
                    kind,
                    mutable: module.mutable,
                    required: module.mutable,
                    candidate_field: Some(candidate_field),
                    module_id: Some(module.module_id.clone()),
                    template_variables: module.template_variables.clone(),
                    constraints,
                    metadata: module.metadata.clone(),
                }
            })
            .collect::<Vec<_>>();
        let target_levers = program
            .target_modules
            .iter()
            .map(|target| {
                if target.candidate_field.trim().is_empty() {
                    target.module_id.clone()
                } else {
                    target.candidate_field.clone()
                }
            })
            .collect::<Vec<_>>();
        let seed_bundle = if program.seed_candidate.fields.is_empty() {
            None
        } else {
            Some(LeverBundle::from_prompt_payload(
                "seed",
                None,
                &program.seed_candidate.fields,
            ))
        };
        Self {
            version: "lever_manifest.v1".to_string(),
            program_id: program.program_id.clone(),
            levers,
            target_levers,
            seed_bundle,
            metadata: program.metadata.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverBundle {
    pub schema_version: String,
    pub bundle_id: String,
    #[serde(default)]
    pub parent_ids: Vec<String>,
    #[serde(default)]
    pub values: BTreeMap<String, Value>,
    #[serde(default)]
    pub mutated_lever_ids: Vec<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl LeverBundle {
    pub fn from_prompt_payload(
        bundle_id: impl Into<String>,
        parent_id: Option<String>,
        payload: &BTreeMap<String, String>,
    ) -> Self {
        let values = payload
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<BTreeMap<_, _>>();
        Self {
            schema_version: "lever_bundle.v1".to_string(),
            bundle_id: bundle_id.into(),
            parent_ids: parent_id.into_iter().collect(),
            values,
            mutated_lever_ids: payload.keys().cloned().collect(),
            metadata: Map::new(),
        }
    }

    pub fn to_prompt_payload(&self) -> BTreeMap<String, String> {
        self.values
            .iter()
            .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
            .collect()
    }
}

pub const GEPA_KNOWN_PROTOCOL_IDS: &[&str] = &[
    "prompt_overlay.v1",
    "whole_file.v1",
    "unified_diff.v1",
    "harness_restart.v1",
    "identity",
];

impl LeverSpec {
    pub fn protocol_id(&self) -> Option<&str> {
        self.metadata
            .get("protocol_id")
            .or_else(|| self.constraints.get("protocol_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }
}

impl LeverManifest {
    pub fn advertised_protocol_ids(&self) -> Vec<String> {
        self.levers
            .iter()
            .filter_map(|lever| lever.protocol_id().map(str::to_string))
            .collect()
    }
}

fn prompt_role_to_lever_kind(role: &str) -> LeverKind {
    match role.trim().to_ascii_lowercase().as_str() {
        "system" => LeverKind::SystemPrompt,
        "user" => LeverKind::UserPrompt,
        _ => LeverKind::TextPrompt,
    }
}

fn parse_lever_kind(value: &str) -> Option<LeverKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "text" | "text_prompt" => Some(LeverKind::TextPrompt),
        "system" | "system_prompt" => Some(LeverKind::SystemPrompt),
        "user" | "user_prompt" => Some(LeverKind::UserPrompt),
        "agents" | "agents_md" | "agents.md" => Some(LeverKind::AgentsMd),
        "skill" | "skill_md" | "skill.md" => Some(LeverKind::SkillMd),
        "workspace_file" | "file" => Some(LeverKind::WorkspaceFile),
        "tool_policy" | "tools" => Some(LeverKind::ToolPolicy),
        "config_append" | "config" => Some(LeverKind::ConfigAppend),
        "verifier_rubric" | "rubric" => Some(LeverKind::VerifierRubric),
        "action_policy" | "policy" => Some(LeverKind::ActionPolicy),
        "policy_script" => Some(LeverKind::PolicyScript),
        "sourced_python" => Some(LeverKind::SourcedPython),
        "harness_module" => Some(LeverKind::HarnessModule),
        "other" => Some(LeverKind::Other),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_id_reads_module_metadata() {
        let spec: LeverSpec = serde_json::from_value(serde_json::json!({
            "lever_id": "policy_script",
            "kind": "policy_script",
            "mutable": true,
            "required": true,
            "metadata": { "protocol_id": "whole_file.v1" }
        }))
        .expect("lever spec");
        assert_eq!(spec.protocol_id(), Some("whole_file.v1"));
        assert!(GEPA_KNOWN_PROTOCOL_IDS.contains(&spec.protocol_id().unwrap()));
    }

    #[test]
    fn unknown_protocol_is_not_in_the_allowlist() {
        assert!(!GEPA_KNOWN_PROTOCOL_IDS.contains(&"flash_evolve.v0"));
    }
}
