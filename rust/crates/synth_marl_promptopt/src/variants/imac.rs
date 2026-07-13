use serde_json::json;
use synth_optimizer_platform::Result;

use crate::strategy::{primary_mean_score, primary_only_arms, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

pub struct ImacStrategy;

impl MarlStrategy for ImacStrategy {
    fn name(&self) -> &'static str {
        "imac"
    }

    fn proposer_guidance(&self) -> serde_json::Value {
        json!({
            "paper_analogue": "IMAC information bottleneck for communication",
            "status": "shared-core placeholder",
            "instruction": "Preserve successful coordination while minimizing messages and characters on a Pareto frontier."
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        primary_only_arms(context.candidate)
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        Ok(primary_mean_score(observations))
    }
}
