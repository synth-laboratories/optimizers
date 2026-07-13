use serde_json::json;
use synth_optimizer_platform::Result;

use crate::strategy::{primary_mean_score, primary_only_arms, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

pub struct Ic3NetStrategy;

impl MarlStrategy for Ic3NetStrategy {
    fn name(&self) -> &'static str {
        "ic3net"
    }

    fn proposer_guidance(&self) -> serde_json::Value {
        json!({
            "paper_analogue": "IC3Net learned communication gating",
            "status": "shared-core placeholder",
            "instruction": "Use matched channel masking to reward speaking only when delivery changes coordinated action."
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        primary_only_arms(context.candidate)
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        Ok(primary_mean_score(observations))
    }
}
