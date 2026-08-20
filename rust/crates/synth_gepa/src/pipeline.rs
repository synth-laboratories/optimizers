use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use synth_optimizer_platform::{
    GepaAdaptiveRolloutConcurrencyConfig, GepaAdaptiveStageWorkersConfig, GepaPipelineConfig,
    GepaPipelineMode, GepaSpeculativeCompletionConfig, GepaStalenessPolicy, OptimizerError, Result,
    SynthOptimizerConfig,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "runtime", rename_all = "snake_case")]
pub enum GepaPipelineRuntimePlan {
    SyncSerial(GepaSyncSerialPlan),
    AsyncPipelined(GepaAsyncPipelinePlan),
    /// Shares `GepaAsyncPipelinePlan` with `AsyncPipelined` but drops the
    /// generation barrier so proposer reflection overlaps policy rollouts.
    ///
    /// Overlap needs two things and for a long time only had one. Dropping the
    /// barrier (`uses_generation_barrier() == false`) lets gen `n+1` propose be
    /// *admitted* while gen `n` full-train work is still outstanding, but until
    /// `background_execution` landed the driver still executed each leased job
    /// inline on the tick, so admitted-but-not-yet-run lanes queued behind each
    /// other and measured overlap was ~0.33s on the 2026-06-02 Banking77 matrix
    /// (741s vs 509s for `SyncSerial`, 0.687x, heldout tied at 0.750).
    ///
    /// `background_execution` defaults to true for this mode; see
    /// `GepaPipelineMode::FlashEvolve`.
    FlashEvolve(GepaAsyncPipelinePlan),
}

impl GepaPipelineRuntimePlan {
    pub fn from_config(config: &SynthOptimizerConfig) -> Result<Self> {
        match config.gepa.pipeline.mode {
            GepaPipelineMode::SyncSerial => Ok(Self::SyncSerial(GepaSyncSerialPlan {
                rollout_transport: config.gepa.rollout_submission_mode.clone(),
            })),
            GepaPipelineMode::AsyncPipelined => {
                Ok(Self::AsyncPipelined(GepaAsyncPipelinePlan::from_config(
                    GepaPipelineMode::AsyncPipelined,
                    &config.gepa.pipeline,
                    &config.gepa.rollout_submission_mode,
                    config.gepa.rollout_chunk_size,
                )?))
            }
            GepaPipelineMode::FlashEvolve => {
                Ok(Self::FlashEvolve(GepaAsyncPipelinePlan::from_config(
                    GepaPipelineMode::FlashEvolve,
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
            Self::AsyncPipelined(plan) | Self::FlashEvolve(plan) => plan.mode,
        }
    }

    pub fn metadata(&self) -> Value {
        match self {
            Self::SyncSerial(plan) => json!({
                "mode": GepaPipelineMode::SyncSerial.as_str(),
                "rollout_transport": plan.rollout_transport,
            }),
            Self::AsyncPipelined(plan) | Self::FlashEvolve(plan) => json!({
                "mode": plan.mode.as_str(),
                "rollout_transport": plan.rollout_transport,
                "staleness_policy": plan.staleness_policy.as_str(),
                "delta_max": plan.delta_max,
                "workers": {
                    "propose": plan.propose_workers,
                    "rollout": plan.rollout_workers,
                    "evaluate": plan.evaluate_workers,
                },
                "max_in_flight_candidates": plan.max_in_flight_candidates,
                "rollout_chunk_size": plan.rollout_chunk_size,
                "adaptive_rollout_concurrency": plan.adaptive_rollout_concurrency,
                "adaptive_stage_workers": plan.adaptive_stage_workers,
                "speculative_completion": plan.speculative_completion,
                "background_execution": plan.background_execution,
                "background_workers": plan.background_workers,
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
pub struct GepaAsyncPipelinePlan {
    pub mode: GepaPipelineMode,
    pub rollout_transport: String,
    pub staleness_policy: GepaStalenessPolicy,
    pub delta_max: u64,
    pub propose_workers: usize,
    pub rollout_workers: usize,
    pub evaluate_workers: usize,
    pub max_in_flight_candidates: usize,
    pub rollout_chunk_size: usize,
    pub adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig,
    pub adaptive_stage_workers: GepaAdaptiveStageWorkersConfig,
    pub speculative_completion: GepaSpeculativeCompletionConfig,
    /// Dispatch leased lane jobs to worker threads instead of running them
    /// inline on the driver tick. Without this, lane leases are bookkeeping
    /// only and no two lanes can overlap in wall clock.
    pub background_execution: bool,
    /// Worker threads backing `background_execution`.
    pub background_workers: usize,
}

pub type GepaAsyncPipelinedPlan = GepaAsyncPipelinePlan;

impl GepaAsyncPipelinePlan {
    fn from_config(
        mode: GepaPipelineMode,
        config: &GepaPipelineConfig,
        rollout_transport: &str,
        rollout_chunk_size: Option<usize>,
    ) -> Result<Self> {
        match (mode, config.staleness_policy) {
            (GepaPipelineMode::AsyncPipelined, GepaStalenessPolicy::Full) => {}
            (GepaPipelineMode::AsyncPipelined, policy) => {
                return Err(OptimizerError::Config(format!(
                    "gepa.pipeline.staleness_policy = {policy:?} is reserved for flash_evolve; use full with async_pipelined"
                )));
            }
            (
                GepaPipelineMode::FlashEvolve,
                GepaStalenessPolicy::Full
                | GepaStalenessPolicy::Guarded
                | GepaStalenessPolicy::Reflective,
            ) => {}
            (GepaPipelineMode::SyncSerial, _) => {
                return Err(OptimizerError::Config(
                    "sync_serial does not use an async pipeline plan".to_string(),
                ));
            }
        }
        let background_execution = config
            .background_execution
            .unwrap_or(matches!(mode, GepaPipelineMode::FlashEvolve));
        let background_workers = config
            .background_workers
            .unwrap_or_else(|| {
                config
                    .workers
                    .propose
                    .saturating_add(config.workers.rollout)
                    .saturating_add(config.workers.evaluate)
            })
            .clamp(1, 64);
        Ok(Self {
            mode,
            background_execution,
            background_workers,
            rollout_transport: rollout_transport.to_string(),
            staleness_policy: config.staleness_policy,
            delta_max: config.delta_max,
            propose_workers: config.workers.propose,
            rollout_workers: config.workers.rollout,
            evaluate_workers: config.workers.evaluate,
            max_in_flight_candidates: config.max_in_flight_candidates,
            adaptive_rollout_concurrency: config.adaptive_rollout_concurrency.clone(),
            adaptive_stage_workers: config.adaptive_stage_workers.clone(),
            speculative_completion: config.speculative_completion.clone(),
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

    pub fn label(&self) -> &'static str {
        self.mode.as_str()
    }

    pub fn uses_generation_barrier(&self) -> bool {
        matches!(self.mode, GepaPipelineMode::AsyncPipelined)
    }

    pub fn stale_item_disposition(
        &self,
        parent_pool_version: u64,
        current_pool_version: u64,
    ) -> GepaStaleItemDecision {
        let stale_gap = current_pool_version.saturating_sub(parent_pool_version);
        match (self.mode, self.staleness_policy) {
            (GepaPipelineMode::FlashEvolve, GepaStalenessPolicy::Guarded)
                if stale_gap > self.delta_max =>
            {
                GepaStaleItemDecision {
                    disposition: GepaStaleItemDisposition::Discard,
                    current_pool_version,
                    stale_gap,
                    reason: format!(
                        "stale gap {stale_gap} exceeds guarded delta_max {}",
                        self.delta_max
                    ),
                }
            }
            (GepaPipelineMode::FlashEvolve, GepaStalenessPolicy::Reflective) if stale_gap > 0 => {
                GepaStaleItemDecision {
                    disposition: GepaStaleItemDisposition::ReflectivePatch,
                    current_pool_version,
                    stale_gap,
                    reason: "stale item requires reflective review".to_string(),
                }
            }
            _ => GepaStaleItemDecision {
                disposition: GepaStaleItemDisposition::AcceptAsIs,
                current_pool_version,
                stale_gap,
                reason: if stale_gap == 0 {
                    "item is current".to_string()
                } else {
                    "staleness policy accepts stale item".to_string()
                },
            },
        }
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
pub struct GepaStaleItemDecision {
    pub disposition: GepaStaleItemDisposition,
    pub current_pool_version: u64,
    pub stale_gap: u64,
    pub reason: String,
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

#[cfg(test)]
mod tests {
    use super::*;
    use synth_optimizer_platform::{GepaPipelineConfig, GepaPipelineMode, GepaStalenessPolicy};

    fn pipeline_config(mode: GepaPipelineMode, policy: GepaStalenessPolicy) -> GepaPipelineConfig {
        let mut config = GepaPipelineConfig::default();
        config.mode = mode;
        config.staleness_policy = policy;
        config
    }

    #[test]
    fn flash_evolve_reflective_plan_is_supported() {
        let config = pipeline_config(
            GepaPipelineMode::FlashEvolve,
            GepaStalenessPolicy::Reflective,
        );

        let plan = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::FlashEvolve,
            &config,
            "async",
            Some(4),
        )
        .expect("flash_evolve reflective plan should build");

        assert_eq!(plan.mode, GepaPipelineMode::FlashEvolve);
        assert_eq!(plan.staleness_policy, GepaStalenessPolicy::Reflective);
        assert!(!plan.uses_generation_barrier());
    }

    #[test]
    fn background_execution_defaults_on_for_flash_and_off_for_pipelined() {
        // Lane leases only model concurrency; `background_execution` is what
        // actually delivers it. FlashEvolve gets it by default. `async_pipelined`
        // deliberately does not, so it stays the same control arm the 2026-06-02
        // matrix measured.
        let flash = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::FlashEvolve,
            &pipeline_config(GepaPipelineMode::FlashEvolve, GepaStalenessPolicy::Full),
            "async",
            None,
        )
        .expect("flash plan should build");
        assert!(flash.background_execution);
        assert_eq!(
            flash.background_workers,
            flash.propose_workers + flash.rollout_workers + flash.evaluate_workers
        );

        let pipelined = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::AsyncPipelined,
            &pipeline_config(GepaPipelineMode::AsyncPipelined, GepaStalenessPolicy::Full),
            "async",
            None,
        )
        .expect("pipelined plan should build");
        assert!(!pipelined.background_execution);
    }

    #[test]
    fn background_execution_is_explicitly_overridable() {
        let mut config = pipeline_config(GepaPipelineMode::FlashEvolve, GepaStalenessPolicy::Full);
        config.background_execution = Some(false);
        config.background_workers = Some(3);
        let plan = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::FlashEvolve,
            &config,
            "async",
            None,
        )
        .expect("flash plan should build");
        assert!(!plan.background_execution);
        assert_eq!(plan.background_workers, 3);
    }

    #[test]
    fn async_pipelined_rejects_non_full_staleness() {
        let config = pipeline_config(
            GepaPipelineMode::AsyncPipelined,
            GepaStalenessPolicy::Guarded,
        );

        let error = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::AsyncPipelined,
            &config,
            "async",
            None,
        )
        .expect_err("async_pipelined should reject guarded staleness");

        assert!(error.to_string().contains("reserved for flash_evolve"));
    }

    #[test]
    fn guarded_policy_discards_items_beyond_delta() {
        let mut config =
            pipeline_config(GepaPipelineMode::FlashEvolve, GepaStalenessPolicy::Guarded);
        config.delta_max = 2;
        let plan = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::FlashEvolve,
            &config,
            "async",
            None,
        )
        .expect("flash_evolve guarded plan should build");

        let current = plan.stale_item_disposition(3, 5);
        assert_eq!(current.disposition, GepaStaleItemDisposition::AcceptAsIs);
        assert_eq!(current.stale_gap, 2);

        let stale = plan.stale_item_disposition(2, 5);
        assert_eq!(stale.disposition, GepaStaleItemDisposition::Discard);
        assert_eq!(stale.stale_gap, 3);
    }

    #[test]
    fn reflective_policy_reviews_only_stale_items() {
        let config = pipeline_config(
            GepaPipelineMode::FlashEvolve,
            GepaStalenessPolicy::Reflective,
        );
        let plan = GepaAsyncPipelinePlan::from_config(
            GepaPipelineMode::FlashEvolve,
            &config,
            "async",
            None,
        )
        .expect("flash_evolve reflective plan should build");

        let current = plan.stale_item_disposition(5, 5);
        assert_eq!(current.disposition, GepaStaleItemDisposition::AcceptAsIs);

        let stale = plan.stale_item_disposition(4, 5);
        assert_eq!(stale.disposition, GepaStaleItemDisposition::ReflectivePatch);
        assert_eq!(stale.stale_gap, 1);
    }
}
