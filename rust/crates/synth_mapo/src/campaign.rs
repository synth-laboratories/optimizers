use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use synth_optimizer_platform::{OptimizerError, Result};

use crate::config::MapoConfig;
use crate::runtime::MAPO_ALGORITHM_ID;

pub const CAMPAIGN_MANIFEST_SCHEMA: &str = "debrief.campaign_manifest.v1";
pub const CAMPAIGN_MANIFEST_ARTIFACT: &str = "campaign_manifest.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CampaignAdapter {
    #[serde(rename = "mapo-craftax-multiagent")]
    CraftaxMultiagent,
    #[serde(rename = "mapo-dungeongrid-plus")]
    DungeongridPlus,
    #[serde(rename = "mapo-alem")]
    Alem,
    #[serde(rename = "mapo-overcooked-v2")]
    OvercookedV2,
}

impl CampaignAdapter {
    fn benchmark_id(self) -> &'static str {
        match self {
            Self::CraftaxMultiagent => "craftax-multiagent",
            Self::DungeongridPlus => "dungeongrid-plus",
            Self::Alem => "alem",
            Self::OvercookedV2 => "overcooked-v2",
        }
    }

    fn heldout_count(self) -> usize {
        match self {
            Self::Alem => 60,
            Self::CraftaxMultiagent | Self::DungeongridPlus | Self::OvercookedV2 => 20,
        }
    }

    fn rollout_cap(self) -> usize {
        match self {
            Self::Alem => 648,
            Self::CraftaxMultiagent | Self::DungeongridPlus | Self::OvercookedV2 => 568,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebriefCampaignManifest {
    pub schema_version: String,
    pub status: String,
    pub campaign_id: String,
    pub adapter_id: CampaignAdapter,
    pub benchmark_id: String,
    pub frozen_at: String,
    pub approved_by: String,
    pub authority: CampaignAuthority,
    pub models: CampaignModels,
    pub optimizer: CampaignOptimizer,
    pub splits: CampaignSplits,
    pub pairing: CampaignPairing,
    pub metrics: CampaignMetrics,
    pub budget: CampaignBudget,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignAuthority {
    pub environment_digest: String,
    pub source_contract: String,
    pub source_digests: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignModels {
    pub baseline: String,
    pub proposer: String,
    pub executor: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignOptimizer {
    pub algorithm_id: String,
    pub generations: usize,
    pub proposals_per_generation: usize,
    pub selection_top_k: usize,
    pub train_repeats: usize,
    pub selection_repeats: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSplits {
    pub train_ids: Vec<String>,
    pub selection_ids: Vec<String>,
    pub heldout_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignPairing {
    pub key_fields: Vec<String>,
    pub terminal_arms: Vec<String>,
    pub heldout_pairs_per_arm: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignMetrics {
    pub primary_metric: String,
    pub noninferiority_margin: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignBudget {
    pub max_rollouts: usize,
    pub max_model_cost_usd: f64,
    pub rate_cap_usd_per_hour: f64,
    pub max_wall_time_seconds: usize,
}

#[derive(Clone, Debug)]
pub struct CampaignBinding {
    pub manifest: DebriefCampaignManifest,
    pub sha256: String,
    pub bytes: Vec<u8>,
}

impl CampaignBinding {
    pub fn load(config: &MapoConfig) -> Result<Option<Self>> {
        let Some(path) = config.evidence.campaign_manifest_path.as_deref() else {
            return Ok(None);
        };
        let bytes = fs::read(path).map_err(|source| OptimizerError::io(path, source))?;
        let manifest: DebriefCampaignManifest = serde_json::from_slice(&bytes)?;
        validate_manifest(config, &manifest)?;
        Ok(Some(Self {
            manifest,
            sha256: sha256_bytes(&bytes),
            bytes,
        }))
    }

    pub fn evidence_value(&self, config: &MapoConfig) -> Value {
        json!({
            "benchmark_id": &self.manifest.benchmark_id,
            "campaign_id": &self.manifest.campaign_id,
            "campaign_manifest_digest": &self.sha256,
            "paper_reference": &config.evidence.paper_reference,
            "environment_digest": &self.manifest.authority.environment_digest,
            "model_snapshot": &self.manifest.models,
            "team_topology": &config.evidence.team_topology,
            "agent_roles": &config.evidence.agent_roles,
            "prompt_protocol_digest": &config.evidence.prompt_protocol_digest,
            "search_budget": &self.manifest.optimizer,
            "source_digests": &self.manifest.authority.source_digests,
            "primary_metric": &self.manifest.metrics.primary_metric,
            "noninferiority_margin": self.manifest.metrics.noninferiority_margin,
            "scorer_version": &config.evidence.scorer_version,
            "token_cost": &config.evidence.token_cost,
            "latency": &config.evidence.latency,
            "known_differences": &config.evidence.known_differences,
            "claim_label": &config.evidence.claim_label,
        })
    }

    pub fn receipt_value(&self) -> Value {
        json!({
            "schema_version": "debrief.campaign_manifest_receipt.v1",
            "campaign_id": &self.manifest.campaign_id,
            "path": format!("artifacts/{CAMPAIGN_MANIFEST_ARTIFACT}"),
            "sha256": &self.sha256,
        })
    }

    pub fn write_artifact(&self, artifact_dir: &Path) -> Result<()> {
        let path = artifact_dir.join(CAMPAIGN_MANIFEST_ARTIFACT);
        fs::write(&path, &self.bytes).map_err(|source| OptimizerError::io(&path, source))
    }
}

fn validate_manifest(config: &MapoConfig, manifest: &DebriefCampaignManifest) -> Result<()> {
    require_equal(
        "schema_version",
        &manifest.schema_version,
        CAMPAIGN_MANIFEST_SCHEMA,
    )?;
    require_equal("status", &manifest.status, "frozen")?;
    require_equal("approved_by", &manifest.approved_by, "Josh")?;
    require_approved("campaign_id", &manifest.campaign_id)?;
    require_approved("benchmark_id", &manifest.benchmark_id)?;
    require_approved("frozen_at", &manifest.frozen_at)?;
    require_approved(
        "authority.environment_digest",
        &manifest.authority.environment_digest,
    )?;
    require_approved(
        "authority.source_contract",
        &manifest.authority.source_contract,
    )?;
    if manifest.benchmark_id != manifest.adapter_id.benchmark_id() {
        return config_error("campaign manifest adapter_id and benchmark_id do not match");
    }
    if manifest.benchmark_id != config.evidence.benchmark_id {
        return config_error("campaign manifest benchmark_id does not match evidence.benchmark_id");
    }
    for (field, value) in [
        ("evidence.paper_reference", &config.evidence.paper_reference),
        ("evidence.team_topology", &config.evidence.team_topology),
        (
            "evidence.prompt_protocol_digest",
            &config.evidence.prompt_protocol_digest,
        ),
        ("evidence.primary_metric", &config.evidence.primary_metric),
        ("evidence.scorer_version", &config.evidence.scorer_version),
    ] {
        require_approved(field, value)?;
    }
    if config.evidence.agent_roles.is_empty() {
        return config_error("evidence.agent_roles cannot be empty for a frozen campaign");
    }
    for (agent, role) in &config.evidence.agent_roles {
        require_approved("evidence.agent_roles agent", agent)?;
        require_approved("evidence.agent_roles role", role)?;
    }
    for difference in &config.evidence.known_differences {
        require_approved("evidence.known_differences entry", difference)?;
    }
    if value_is_empty(&config.evidence.token_cost) {
        return config_error("evidence.token_cost is required for a frozen campaign");
    }
    if value_is_empty(&config.evidence.latency) {
        return config_error("evidence.latency is required for a frozen campaign");
    }
    if manifest.authority.environment_digest != config.evidence.environment_digest {
        return config_error(
            "campaign manifest environment_digest does not match evidence.environment_digest",
        );
    }
    if manifest.authority.source_digests.is_empty() {
        return config_error("campaign manifest authority.source_digests cannot be empty");
    }
    for (source, digest) in &manifest.authority.source_digests {
        require_approved("authority.source_digests path", source)?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return config_error("campaign manifest source digests must be lowercase SHA-256 hex");
        }
    }
    for (field, model) in [
        ("models.baseline", &manifest.models.baseline),
        ("models.proposer", &manifest.models.proposer),
        ("models.executor", &manifest.models.executor),
    ] {
        require_approved(field, model)?;
    }
    if manifest.models.baseline != config.policy.model
        || manifest.models.executor != config.policy.model
    {
        return config_error(
            "campaign manifest baseline/executor models must match policy.model exactly",
        );
    }
    if config.proposer.model.as_deref() != Some(manifest.models.proposer.as_str()) {
        return config_error("campaign manifest proposer model must match proposer.model exactly");
    }
    if manifest.optimizer.algorithm_id != MAPO_ALGORITHM_ID
        || manifest.optimizer.generations != 4
        || manifest.optimizer.proposals_per_generation != 4
        || manifest.optimizer.selection_top_k != 3
        || manifest.optimizer.train_repeats != 3
        || manifest.optimizer.selection_repeats != 1
    {
        return config_error("campaign manifest optimizer does not match the plan-approved search");
    }
    if manifest.optimizer.generations != config.mapo.max_generations
        || manifest.optimizer.proposals_per_generation != config.mapo.proposals_per_generation
        || manifest.optimizer.selection_top_k != config.mapo.selection_top_k
        || manifest.optimizer.train_repeats != config.mapo.rollouts_per_candidate
        || manifest.optimizer.selection_repeats != config.mapo.selection_rollouts_per_candidate
    {
        return config_error("campaign manifest optimizer does not match the resolved MAPO config");
    }

    validate_splits(config, manifest)?;
    if manifest.pairing.key_fields != ["seed", "episode_index", "task_instance_id"] {
        return config_error("campaign manifest pairing key_fields do not match debrief.v1");
    }
    if manifest.pairing.terminal_arms != ["baseline", "champion"] {
        return config_error("campaign manifest terminal arms must be baseline and champion");
    }
    let heldout_count = manifest.adapter_id.heldout_count();
    if manifest.pairing.heldout_pairs_per_arm != heldout_count
        || config.mapo.heldout_rollouts_per_arm != heldout_count
        || config.mapo.heldout_min_paired_episodes_per_arm != heldout_count
    {
        return config_error(
            "campaign manifest heldout pairing does not match the resolved MAPO config",
        );
    }
    if manifest.metrics.primary_metric != config.evidence.primary_metric {
        return config_error(
            "campaign manifest primary_metric does not match evidence.primary_metric",
        );
    }
    if config.evidence.noninferiority_margin.as_f64()
        != Some(manifest.metrics.noninferiority_margin)
    {
        return config_error(
            "campaign manifest noninferiority_margin does not match evidence.noninferiority_margin",
        );
    }
    let expected_rollouts = manifest.adapter_id.rollout_cap();
    if manifest.budget.max_rollouts != expected_rollouts {
        return config_error("campaign manifest max_rollouts does not match the plan-approved cap");
    }
    if !manifest.budget.max_model_cost_usd.is_finite()
        || manifest.budget.max_model_cost_usd <= 0.0
        || !manifest.budget.rate_cap_usd_per_hour.is_finite()
        || manifest.budget.rate_cap_usd_per_hour <= 0.0
        || manifest.budget.max_wall_time_seconds == 0
    {
        return config_error("campaign manifest budget values must be positive and finite");
    }
    Ok(())
}

fn validate_splits(config: &MapoConfig, manifest: &DebriefCampaignManifest) -> Result<()> {
    let expected = [
        (
            "train",
            &manifest.splits.train_ids,
            split_ids(&config.taskset.train_seeds, &[]),
            8,
        ),
        (
            "selection",
            &manifest.splits.selection_ids,
            split_ids(
                &config.taskset.selection_seeds,
                &config.taskset.selection_task_instance_ids,
            ),
            4,
        ),
        (
            "heldout",
            &manifest.splits.heldout_ids,
            split_ids(&config.taskset.heldout_seeds, &[]),
            manifest.adapter_id.heldout_count(),
        ),
    ];
    let mut all = BTreeSet::new();
    for (name, ids, config_ids, count) in expected {
        if ids.len() != count {
            return config_error(&format!(
                "campaign manifest {name} split must contain exactly {count} ids"
            ));
        }
        let manifest_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
        if manifest_ids.len() != ids.len() {
            return config_error(&format!(
                "campaign manifest {name} split contains duplicates"
            ));
        }
        if manifest_ids
            .iter()
            .any(|id| !id.starts_with("seed:") && !id.starts_with("task:"))
        {
            return config_error(&format!(
                "campaign manifest {name} ids must start with seed: or task:"
            ));
        }
        if manifest_ids != config_ids {
            return config_error(&format!(
                "campaign manifest {name} split does not match the resolved MAPO config"
            ));
        }
        if manifest_ids.iter().any(|id| !all.insert(id.clone())) {
            return config_error("campaign manifest train/selection/heldout splits overlap");
        }
    }
    Ok(())
}

fn split_ids(seeds: &[i64], task_ids: &[String]) -> BTreeSet<String> {
    seeds
        .iter()
        .map(|seed| format!("seed:{seed}"))
        .chain(task_ids.iter().map(|task_id| format!("task:{task_id}")))
        .collect()
}

fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(value) => value.trim().is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
        Value::Bool(_) | Value::Number(_) => false,
    }
}

fn require_approved(field: &str, value: &str) -> Result<()> {
    let normalized = value.trim();
    if normalized.is_empty()
        || ["todo", "tbd", "placeholder", "pending"]
            .iter()
            .any(|token| normalized.to_ascii_lowercase().contains(token))
    {
        return config_error(&format!(
            "campaign manifest {field} must be an approved value"
        ));
    }
    Ok(())
}

fn require_equal(field: &str, value: &str, expected: &str) -> Result<()> {
    if value != expected {
        return config_error(&format!("campaign manifest {field} must be {expected}"));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn config_error<T>(message: &str) -> Result<T> {
    Err(OptimizerError::Config(message.to_string()))
}
