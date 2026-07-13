use serde_json::json;
use synth_optimizer_platform::Result;

use crate::strategy::{primary_mean_score, primary_only_arms, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

pub struct RodeStrategy;

impl MarlStrategy for RodeStrategy {
    fn name(&self) -> &'static str {
        "rode"
    }

    fn proposer_guidance(&self) -> serde_json::Value {
        json!({
            "paper_analogue": "RODE role-oriented hierarchical learning",
            "status": "shared-core placeholder",
            "instruction": "Alternate high-level role assignment edits with lower-level shared and communication policy edits."
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        primary_only_arms(context.candidate)
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        Ok(primary_mean_score(observations))
    }
}
