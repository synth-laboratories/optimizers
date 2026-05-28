use serde::{Deserialize, Serialize};

use crate::config::{GepaConfig, SynthOptimizerConfig};
use crate::error::{OptimizerError, Result};
use crate::limits::{RunLimitPolicy, RuntimeEffectBudgetEstimate};

pub const GEPA_LIMIT_STOP_POLICY: &str = "graceful_finish_generation";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GepaRuntimeEffectBudgetEstimates {
    pub proposer: RuntimeEffectBudgetEstimate,
    pub rollout: RuntimeEffectBudgetEstimate,
}

#[derive(Clone, Debug)]
pub struct ConfiguredGepaRunLimits {
    max_total_rollouts: Option<u64>,
    max_cost_usd: Option<f64>,
    max_time_seconds: Option<u64>,
    max_prompt_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    max_total_tokens: Option<u64>,
    hard_limit: bool,
    stop_policy: &'static str,
    budget_estimates: GepaRuntimeEffectBudgetEstimates,
}

impl ConfiguredGepaRunLimits {
    pub fn from_config(config: &SynthOptimizerConfig) -> Self {
        Self {
            max_total_rollouts: Some(config.gepa.effective_max_total_rollouts() as u64),
            max_cost_usd: gepa_cost_limit_usd(&config.gepa),
            max_time_seconds: config.gepa.max_time_seconds,
            max_prompt_tokens: config.gepa.max_prompt_tokens,
            max_completion_tokens: config.gepa.max_completion_tokens,
            max_total_tokens: config.gepa.max_total_tokens,
            hard_limit: true,
            stop_policy: GEPA_LIMIT_STOP_POLICY,
            budget_estimates: GepaRuntimeEffectBudgetEstimates {
                proposer: gepa_proposer_budget_estimate(
                    &config.gepa,
                    config.proposer.timeout_seconds,
                ),
                rollout: gepa_rollout_budget_estimate(&config.gepa),
            },
        }
    }

    pub fn budget_estimates(&self) -> GepaRuntimeEffectBudgetEstimates {
        self.budget_estimates
    }

    pub fn proposer_budget_estimate(&self) -> RuntimeEffectBudgetEstimate {
        self.budget_estimates.proposer
    }

    pub fn rollout_budget_estimate(&self) -> RuntimeEffectBudgetEstimate {
        self.budget_estimates.rollout
    }
}

impl RunLimitPolicy for ConfiguredGepaRunLimits {
    fn max_total_rollouts(&self) -> Option<u64> {
        self.max_total_rollouts
    }

    fn max_cost_usd(&self) -> Option<f64> {
        self.max_cost_usd
    }

    fn max_time_seconds(&self) -> Option<u64> {
        self.max_time_seconds
    }

    fn max_prompt_tokens(&self) -> Option<u64> {
        self.max_prompt_tokens
    }

    fn max_completion_tokens(&self) -> Option<u64> {
        self.max_completion_tokens
    }

    fn max_total_tokens(&self) -> Option<u64> {
        self.max_total_tokens
    }

    fn hard_limit(&self) -> bool {
        self.hard_limit
    }

    fn stop_policy(&self) -> &str {
        self.stop_policy
    }
}

pub(crate) fn validate_gepa_limit_config(config: &GepaConfig) -> Result<()> {
    if gepa_cost_limit_usd(config).is_some() {
        require_positive_f64(
            "gepa.proposer_estimated_cost_usd",
            config.proposer_estimated_cost_usd,
        )?;
        require_positive_f64(
            "gepa.rollout_estimated_cost_usd",
            config.rollout_estimated_cost_usd,
        )?;
    }
    if config.max_prompt_tokens.is_some() {
        require_positive(
            "gepa.proposer_estimated_prompt_tokens",
            config.proposer_estimated_prompt_tokens,
        )?;
        require_positive(
            "gepa.rollout_estimated_prompt_tokens",
            config.rollout_estimated_prompt_tokens,
        )?;
    }
    if config.max_completion_tokens.is_some() {
        require_positive(
            "gepa.proposer_estimated_completion_tokens",
            config.proposer_estimated_completion_tokens,
        )?;
        require_positive(
            "gepa.rollout_estimated_completion_tokens",
            config.rollout_estimated_completion_tokens,
        )?;
    }
    if config.max_total_tokens.is_some() {
        require_total_token_estimate(
            "gepa proposer token estimate",
            config.proposer_estimated_total_tokens,
            config.proposer_estimated_prompt_tokens,
            config.proposer_estimated_completion_tokens,
        )?;
        require_total_token_estimate(
            "gepa rollout token estimate",
            config.rollout_estimated_total_tokens,
            config.rollout_estimated_prompt_tokens,
            config.rollout_estimated_completion_tokens,
        )?;
    }
    if config.max_time_seconds.is_some() {
        require_positive(
            "gepa.rollout_estimated_wall_seconds",
            config.rollout_estimated_wall_seconds,
        )?;
    }
    Ok(())
}

fn gepa_cost_limit_usd(config: &GepaConfig) -> Option<f64> {
    (config.max_cost_usd > 0.0).then_some(config.max_cost_usd)
}

fn gepa_proposer_budget_estimate(
    config: &GepaConfig,
    timeout_seconds: u64,
) -> RuntimeEffectBudgetEstimate {
    RuntimeEffectBudgetEstimate {
        max_cost_usd: config.proposer_estimated_cost_usd,
        max_prompt_tokens: config.proposer_estimated_prompt_tokens,
        max_completion_tokens: config.proposer_estimated_completion_tokens,
        max_total_tokens: config.proposer_estimated_total_tokens,
        max_rollouts: None,
        max_wall_seconds: Some(timeout_seconds),
    }
}

fn gepa_rollout_budget_estimate(config: &GepaConfig) -> RuntimeEffectBudgetEstimate {
    RuntimeEffectBudgetEstimate {
        max_cost_usd: config.rollout_estimated_cost_usd,
        max_prompt_tokens: config.rollout_estimated_prompt_tokens,
        max_completion_tokens: config.rollout_estimated_completion_tokens,
        max_total_tokens: config.rollout_estimated_total_tokens,
        max_rollouts: Some(1),
        max_wall_seconds: config.rollout_estimated_wall_seconds,
    }
}

fn require_positive(name: &str, value: Option<u64>) -> Result<()> {
    if value.unwrap_or(0) == 0 {
        return Err(OptimizerError::Config(format!(
            "{name} is required and must be positive when the corresponding hard limit is set"
        )));
    }
    Ok(())
}

fn require_positive_f64(name: &str, value: Option<f64>) -> Result<()> {
    match value {
        Some(item) if item.is_finite() && item > 0.0 => Ok(()),
        _ => Err(OptimizerError::Config(format!(
            "{name} is required and must be positive when the corresponding hard limit is set",
        ))),
    }
}

fn require_total_token_estimate(
    name: &str,
    total: Option<u64>,
    prompt: Option<u64>,
    completion: Option<u64>,
) -> Result<()> {
    if total.unwrap_or(0) > 0 || prompt.unwrap_or(0) + completion.unwrap_or(0) > 0 {
        return Ok(());
    }
    Err(OptimizerError::Config(format!(
        "{name} requires either a positive total token estimate or positive prompt/completion estimates when gepa.max_total_tokens is set"
    )))
}
