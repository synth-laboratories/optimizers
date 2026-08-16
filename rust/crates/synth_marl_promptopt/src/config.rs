use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use synth_optimizer_platform::{OptimizerError, Result, SynthOptimizerConfig};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarlPromptoptConfig {
    pub gepa_profile: PathBuf,
    pub variant: String,
    #[serde(default)]
    pub run: MarlRunConfig,
    #[serde(default)]
    pub experiment: MarlExperimentConfig,
}

impl MarlPromptoptConfig {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
        let mut config: Self = toml::from_str(&text)?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        if config.gepa_profile.is_relative() {
            config.gepa_profile = base.join(&config.gepa_profile);
        }
        if config.run.output_dir.is_relative() {
            config.run.output_dir = base.join(&config.run.output_dir);
        }
        config.validate()?;
        Ok(config)
    }

    pub fn load_gepa_config(&self) -> Result<SynthOptimizerConfig> {
        let mut config = SynthOptimizerConfig::from_toml_file(&self.gepa_profile)?;
        config.run.run_id = self.run.run_id.clone();
        config.run.output_dir = self.run.output_dir.clone();
        config.run.seed = self.run.seed;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.variant.trim().is_empty() {
            return Err(OptimizerError::Config(
                "variant must be coma, ic3net, imac, or rode".to_string(),
            ));
        }
        if self.run.run_id.trim().is_empty() {
            return Err(OptimizerError::Config("run.run_id is required".to_string()));
        }
        if self.experiment.selection_candidates_per_generation == 0 {
            return Err(OptimizerError::Config(
                "experiment.selection_candidates_per_generation must be positive".to_string(),
            ));
        }
        if self.experiment.minimum_rows_per_candidate == 0 {
            return Err(OptimizerError::Config(
                "experiment.minimum_rows_per_candidate must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarlRunConfig {
    #[serde(default = "default_run_id")]
    pub run_id: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default)]
    pub seed: u64,
}

impl Default for MarlRunConfig {
    fn default() -> Self {
        Self {
            run_id: default_run_id(),
            output_dir: default_output_dir(),
            seed: 0,
        }
    }
}

fn default_run_id() -> String {
    format!("marl_promptopt_{}", Uuid::new_v4().simple())
}

fn default_output_dir() -> PathBuf {
    PathBuf::from(".out/marl_promptopt")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarlExperimentConfig {
    #[serde(default = "default_selection_candidates_per_generation")]
    pub selection_candidates_per_generation: usize,
    #[serde(default = "default_minimum_rows_per_candidate")]
    pub minimum_rows_per_candidate: usize,
    #[serde(default = "default_true")]
    pub require_disjoint_splits: bool,
    #[serde(default = "default_true")]
    pub require_exact_rollout_budget: bool,
    #[serde(default = "default_true")]
    pub compare_seed_on_heldout: bool,
}

impl Default for MarlExperimentConfig {
    fn default() -> Self {
        Self {
            selection_candidates_per_generation: default_selection_candidates_per_generation(),
            minimum_rows_per_candidate: default_minimum_rows_per_candidate(),
            require_disjoint_splits: true,
            require_exact_rollout_budget: true,
            compare_seed_on_heldout: true,
        }
    }
}

fn default_selection_candidates_per_generation() -> usize {
    1
}

fn default_minimum_rows_per_candidate() -> usize {
    1
}

fn default_true() -> bool {
    true
}
