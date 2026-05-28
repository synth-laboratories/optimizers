use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use synth_optimizer_platform::{
    GepaAdaptiveRolloutConcurrencyConfig, GepaPipelineConfig, GepaPipelineMode,
    GepaStalenessPolicy, OptimizerError, Result, SynthOptimizerConfig,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "runtime", rename_all = "snake_case")]
pub enum GepaPipelineRuntimePlan {
    SyncSerial(GepaSyncSerialPlan),
    AsyncPipelined(GepaAsyncPipelinedPlan),
}

impl GepaPipelineRuntimePlan {
    pub fn from_config(config: &SynthOptimizerConfig) -> Result<Self> {
        match config.gepa.pipeline.mode {
            GepaPipelineMode::SyncSerial => Ok(Self::SyncSerial(GepaSyncSerialPlan {
                rollout_transport: config.gepa.rollout_submission_mode.clone(),
            })),
            GepaPipelineMode::AsyncPipelined => {
                Ok(Self::AsyncPipelined(GepaAsyncPipelinedPlan::from_config(
                    &config.gepa.pipeline,
                    &config.gepa.rollout_submission_mode,
                    config.gepa.rollout_chunk_size,
                )?))
            }
        }
    }

    pub fn mode(&self) -> GepaPipelineMode {
        match self {
            Self::SyncSerial(_) => GepaPipelineMode::SyncSerial,
            Self::AsyncPipelined(_) => GepaPipelineMode::AsyncPipelined,
        }
    }

    pub fn metadata(&self) -> Value {
        match self {
            Self::SyncSerial(plan) => json!({
                "mode": GepaPipelineMode::SyncSerial.as_str(),
                "rollout_transport": plan.rollout_transport,
            }),
            Self::AsyncPipelined(plan) => json!({
                "mode": GepaPipelineMode::AsyncPipelined.as_str(),
                "rollout_transport": plan.rollout_transport,
                "staleness_policy": plan.staleness_policy.as_str(),
                "workers": {
                    "propose": plan.propose_workers,
                    "rollout": plan.rollout_workers,
                    "evaluate": plan.evaluate_workers,
                },
                "max_in_flight_candidates": plan.max_in_flight_candidates,
                "rollout_chunk_size": plan.rollout_chunk_size,
                "adaptive_rollout_concurrency": plan.adaptive_rollout_concurrency,
                "lanes": plan.lanes(),
            }),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaSyncSerialPlan {
    pub rollout_transport: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaAsyncPipelinedPlan {
    pub rollout_transport: String,
    pub staleness_policy: GepaStalenessPolicy,
    pub propose_workers: usize,
    pub rollout_workers: usize,
    pub evaluate_workers: usize,
    pub max_in_flight_candidates: usize,
    pub rollout_chunk_size: usize,
    pub adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig,
}

impl GepaAsyncPipelinedPlan {
    fn from_config(
        config: &GepaPipelineConfig,
        rollout_transport: &str,
        rollout_chunk_size: Option<usize>,
    ) -> Result<Self> {
        if !matches!(config.staleness_policy, GepaStalenessPolicy::Full) {
            return Err(OptimizerError::Config(format!(
                "gepa.pipeline.staleness_policy = {:?} is reserved for a later async-pipelined phase; use full for the first runtime slice",
                config.staleness_policy
            )));
        }
        Ok(Self {
            rollout_transport: rollout_transport.to_string(),
            staleness_policy: config.staleness_policy,
            propose_workers: config.workers.propose,
            rollout_workers: config.workers.rollout,
            evaluate_workers: config.workers.evaluate,
            max_in_flight_candidates: config.max_in_flight_candidates,
            adaptive_rollout_concurrency: config.adaptive_rollout_concurrency.clone(),
            rollout_chunk_size: rollout_chunk_size.unwrap_or_else(|| {
                if config.adaptive_rollout_concurrency.enabled {
                    config.adaptive_rollout_concurrency.initial.clamp(1, 128)
                } else {
                    config.workers.rollout.clamp(1, 128)
                }
            }),
        })
    }

    pub fn lanes(&self) -> Vec<GepaPipelineLane> {
        vec![
            GepaPipelineLane::Propose,
            GepaPipelineLane::Rollout,
            GepaPipelineLane::Evaluate,
        ]
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GepaPipelineLane {
    Propose,
    Rollout,
    Evaluate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaPipelineWorkItem {
    pub lane: GepaPipelineLane,
    pub subject_id: String,
    pub generation: usize,
    pub parent_pool_version: u64,
    pub current_pool_version: Option<u64>,
    pub stale_gap: Option<u64>,
}

impl GepaPipelineWorkItem {
    pub fn with_current_pool_version(mut self, current_pool_version: u64) -> Self {
        self.stale_gap = Some(current_pool_version.saturating_sub(self.parent_pool_version));
        self.current_pool_version = Some(current_pool_version);
        self
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GepaStaleItemDisposition {
    AcceptAsIs,
    Discard,
    ReflectivePatch,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaAsyncPipelineSketch {
    pub producer: &'static str,
    pub proposer_lane: &'static str,
    pub rollout_lane: &'static str,
    pub evaluate_lane: &'static str,
    pub consumer: &'static str,
}

impl Default for GepaAsyncPipelineSketch {
    fn default() -> Self {
        Self {
            producer: "select parents while in_flight_candidates < max_in_flight_candidates",
            proposer_lane: "invoke proposer subagents and enqueue candidate minibatch work",
            rollout_lane:
                "execute rollout jobs through existing sync_post or async_post_poll transport",
            evaluate_lane: "fold scored candidates, including heldout validation shards",
            consumer:
                "admit accepted candidates, bump pool_version, and handle stale work by policy",
        }
    }
}
