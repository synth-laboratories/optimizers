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
        "other" => Some(LeverKind::Other),
        _ => None,
    }
}
