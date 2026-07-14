use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::scoring::MapoScore;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoCandidate {
    pub id: String,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub protocol: MapoProtocolConfig,
    #[serde(default)]
    pub shared_context: MapoSharedContextConfig,
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
    #[serde(default)]
    pub train_score: Option<MapoScore>,
    #[serde(default)]
    pub selection_score: Option<MapoScore>,
    #[serde(default)]
    pub heldout_score: Option<MapoScore>,
}

impl MapoCandidate {
    pub fn seed(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            generation: 0,
            parent_id: None,
            protocol: MapoProtocolConfig::default(),
            shared_context: MapoSharedContextConfig::default(),
            roles: BTreeMap::new(),
            train_score: None,
            selection_score: None,
            heldout_score: None,
        }
    }

    pub fn protocol_value(&self) -> Value {
        serde_json::to_value(&self.protocol).unwrap_or(Value::Null)
    }
}

impl Default for MapoCandidate {
    fn default() -> Self {
        Self::seed("mapo_seed")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoProtocolConfig {
    #[serde(default = "default_protocol_mode")]
    pub mode: String,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default = "default_leader_policy")]
    pub leader_policy: String,
    #[serde(default)]
    pub leader_role: String,
    #[serde(default)]
    pub followers_can_reply: bool,
}

impl Default for MapoProtocolConfig {
    fn default() -> Self {
        Self {
            mode: default_protocol_mode(),
            max_chars: default_max_chars(),
            leader_policy: default_leader_policy(),
            leader_role: String::new(),
            followers_can_reply: false,
        }
    }
}

fn default_protocol_mode() -> String {
    "pure_decentralized".to_string()
}

fn default_max_chars() -> usize {
    240
}

fn default_leader_policy() -> String {
    "first_hero".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoSharedContextConfig {
    #[serde(default = "default_render_mode")]
    pub render_mode: String,
    #[serde(default = "default_window_tokens")]
    pub window_tokens: usize,
    #[serde(default = "default_claim_window_tokens")]
    pub claim_window_tokens: usize,
    #[serde(default = "default_patch_summary_window_tokens")]
    pub patch_summary_window_tokens: usize,
    #[serde(default = "default_planner_window_tokens")]
    pub planner_window_tokens: usize,
    #[serde(default = "default_implementer_window_tokens")]
    pub implementer_window_tokens: usize,
    #[serde(default = "default_self_policy")]
    pub self_policy: String,
    #[serde(default = "default_true")]
    pub tried_admission_filter: bool,
    #[serde(default = "default_true")]
    pub claims_enabled: bool,
    #[serde(default = "default_true")]
    pub patch_summary_enabled: bool,
}

impl Default for MapoSharedContextConfig {
    fn default() -> Self {
        Self {
            render_mode: default_render_mode(),
            window_tokens: default_window_tokens(),
            claim_window_tokens: default_claim_window_tokens(),
            patch_summary_window_tokens: default_patch_summary_window_tokens(),
            planner_window_tokens: default_planner_window_tokens(),
            implementer_window_tokens: default_implementer_window_tokens(),
            self_policy: default_self_policy(),
            tried_admission_filter: true,
            claims_enabled: true,
            patch_summary_enabled: true,
        }
    }
}

fn default_render_mode() -> String {
    "full".to_string()
}

fn default_window_tokens() -> usize {
    500
}

fn default_claim_window_tokens() -> usize {
    100
}

fn default_patch_summary_window_tokens() -> usize {
    200
}

fn default_planner_window_tokens() -> usize {
    1500
}

fn default_implementer_window_tokens() -> usize {
    800
}

fn default_self_policy() -> String {
    "include".to_string()
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapoRolloutRecord {
    pub rollout_id: String,
    pub candidate_id: String,
    pub split: String,
    pub rollout_group: String,
    pub seed: i64,
    pub episode_index: usize,
    #[serde(default)]
    pub task_instance_id: Option<String>,
    #[serde(default)]
    pub parent_rollout_id: Option<String>,
    #[serde(default)]
    pub parent_checkpoint_id: Option<String>,
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    pub success: bool,
    pub reward: f64,
    pub messages_delivered: u64,
    pub messages_rejected: u64,
    pub message_chars: u64,
    pub response: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapoBranchCheckpoint {
    pub checkpoint_id: String,
    pub parent_rollout_id: String,
    pub seed: i64,
    #[serde(default)]
    pub task_instance_id: Option<String>,
    pub step: usize,
    pub reward: f64,
    #[serde(default)]
    pub messages_delivered: u64,
    #[serde(default)]
    pub messages_rejected: u64,
    #[serde(default)]
    pub message_chars: u64,
}
