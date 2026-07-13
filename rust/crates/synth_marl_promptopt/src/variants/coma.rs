use serde_json::json;
use synth_optimizer_platform::Result;

use crate::strategy::{primary_mean_score, primary_only_arms, ArmContext, MarlStrategy};
use crate::types::{EvaluationArm, RolloutObservation, StrategyScore};

pub struct ComaStrategy;

impl MarlStrategy for ComaStrategy {
    fn name(&self) -> &'static str {
        "coma"
    }

    fn proposer_guidance(&self) -> serde_json::Value {
        json!({
            "paper_analogue": "COMA counterfactual multi-agent credit assignment",
            "status": "shared-core placeholder",
            "instruction": "Use matched role ablations to identify which prompt edits causally improve the joint outcome."
        })
    }

    fn evaluation_arms(&self, context: ArmContext<'_>) -> Vec<EvaluationArm> {
        primary_only_arms(context.candidate)
    }

    fn score(&self, observations: &[RolloutObservation]) -> Result<StrategyScore> {
        Ok(primary_mean_score(observations))
    }
}
