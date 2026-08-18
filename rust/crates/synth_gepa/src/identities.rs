use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    GepaCandidateIdentity, GepaDeploymentCandidate, GepaHeldoutMeasurement,
    GepaReconciliationStatus, GepaRunResult, LeverBundle,
};

use crate::CandidateRecord;

pub const DEPLOYMENT_RULE_OPTIMIZATION_SELECTED: &str = "optimization_selected";
pub const DEPLOYMENT_RULE_HELDOUT_BEST: &str = "heldout_best";
pub const VERDICT_IMPROVEMENT_DEMONSTRATED: &str = "improvement_demonstrated";
pub const VERDICT_NO_MEASURED_IMPROVEMENT: &str = "no_measured_improvement";
pub const VERDICT_INCONCLUSIVE: &str = "inconclusive";

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateIdentitySet {
    pub optimization_selected_idx: Option<usize>,
    pub heldout_best_idx: Option<usize>,
    pub deployment_idx: Option<usize>,
    pub deployment_rule: String,
}

impl Default for CandidateIdentitySet {
    fn default() -> Self {
        Self {
            optimization_selected_idx: None,
            heldout_best_idx: None,
            deployment_idx: None,
            deployment_rule: DEPLOYMENT_RULE_OPTIMIZATION_SELECTED.to_string(),
        }
    }
}

impl CandidateIdentitySet {
    pub fn freeze_optimization_selected(&mut self, train_best_idx: Option<usize>) {
        if self.optimization_selected_idx.is_none() {
            self.optimization_selected_idx = train_best_idx;
        }
    }

    pub fn record_heldout_best(&mut self, heldout_best_idx: Option<usize>) {
        self.heldout_best_idx = heldout_best_idx;
    }

    pub fn apply_deployment_rule(&mut self, rule: &str) {
        let (idx, resolved_rule) =
            resolve_deployment_idx(rule, self.optimization_selected_idx, self.heldout_best_idx);
        self.deployment_idx = idx;
        self.deployment_rule = resolved_rule;
    }

    pub fn best_idx_alias(&self) -> Option<usize> {
        self.deployment_idx
            .or(self.optimization_selected_idx)
            .or(self.heldout_best_idx)
    }
}

pub fn normalize_deployment_rule(rule: &str) -> String {
    let normalized = rule.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        DEPLOYMENT_RULE_HELDOUT_BEST | "heldout" | "heldout_winner" => {
            DEPLOYMENT_RULE_HELDOUT_BEST.to_string()
        }
        _ => DEPLOYMENT_RULE_OPTIMIZATION_SELECTED.to_string(),
    }
}

pub fn resolve_deployment_idx(
    rule: &str,
    optimization_selected_idx: Option<usize>,
    heldout_best_idx: Option<usize>,
) -> (Option<usize>, String) {
    let rule = normalize_deployment_rule(rule);
    let idx = if rule == DEPLOYMENT_RULE_HELDOUT_BEST {
        heldout_best_idx.or(optimization_selected_idx)
    } else {
        optimization_selected_idx.or(heldout_best_idx)
    };
    (idx, rule)
}

pub fn candidate_identity(
    candidates: &[CandidateRecord],
    idx: Option<usize>,
    split: &str,
    score: impl Fn(&CandidateRecord) -> Option<f64>,
) -> Option<GepaCandidateIdentity> {
    let idx = idx?;
    let candidate = candidates.get(idx)?;
    Some(GepaCandidateIdentity {
        id: candidate.candidate_id.clone(),
        score: score(candidate),
        split: split.to_string(),
    })
}

pub fn heldout_measurements(candidates: &[CandidateRecord]) -> Vec<GepaHeldoutMeasurement> {
    candidates
        .iter()
        .filter(|candidate| candidate.heldout_reward.is_some())
        .map(|candidate| GepaHeldoutMeasurement {
            id: candidate.candidate_id.clone(),
            score: candidate.heldout_reward,
        })
        .collect()
}

pub fn seed_train_reward(candidates: &[CandidateRecord]) -> Option<f64> {
    candidates
        .iter()
        .find(|candidate| candidate.source == "seed")
        .and_then(|candidate| candidate.train_reward)
        .or_else(|| {
            candidates
                .first()
                .and_then(|candidate| candidate.train_reward)
        })
}

pub fn improvement_verdict(seed_train: Option<f64>, selected_train: Option<f64>) -> &'static str {
    match (seed_train, selected_train) {
        (Some(seed), Some(selected)) if selected > seed => VERDICT_IMPROVEMENT_DEMONSTRATED,
        (Some(_), Some(_)) => VERDICT_NO_MEASURED_IMPROVEMENT,
        (None, Some(_)) => VERDICT_INCONCLUSIVE,
        _ => VERDICT_NO_MEASURED_IMPROVEMENT,
    }
}

pub fn selection_authority_value(
    candidates: &[CandidateRecord],
    identities: &CandidateIdentitySet,
    rollout_count: usize,
    ledger_rollout_count: Option<u64>,
    reported_cost: Option<f64>,
    ledger_cost: Option<f64>,
) -> Value {
    let optimization_selected = candidate_identity(
        candidates,
        identities.optimization_selected_idx,
        "train",
        |candidate| candidate.train_reward,
    );
    let heldout_best = candidate_identity(
        candidates,
        identities.heldout_best_idx,
        "heldout",
        |candidate| candidate.heldout_reward,
    );
    let deployment_id = identities
        .deployment_idx
        .and_then(|idx| candidates.get(idx))
        .map(|candidate| candidate.candidate_id.clone());
    let selected_train = optimization_selected
        .as_ref()
        .and_then(|identity| identity.score);
    json!({
        "optimization_selected_candidate": optimization_selected,
        "heldout_measurements_by_candidate": heldout_measurements(candidates),
        "heldout_best_candidate": heldout_best,
        "deployment_candidate": deployment_id.as_ref().map(|id| GepaDeploymentCandidate {
            id: id.clone(),
            rule: identities.deployment_rule.clone(),
        }),
        "improvement_verdict": improvement_verdict(seed_train_reward(candidates), selected_train),
        "rollout_count_authority": {
            "source": "run_state",
            "value": rollout_count,
        },
        "cost_authority": {
            "source": if reported_cost.is_some() { "manifest" } else { "incomplete" },
            "value": reported_cost,
        },
        "reconciliation_status": reconciliation_status(
            rollout_count,
            ledger_rollout_count,
            reported_cost,
            ledger_cost,
        ),
    })
}

pub fn reconciliation_status(
    run_rollouts: usize,
    ledger_rollouts: Option<u64>,
    reported_cost: Option<f64>,
    ledger_cost: Option<f64>,
) -> GepaReconciliationStatus {
    let mut authorities = Map::new();
    authorities.insert(
        "run_state".to_string(),
        json!({"rollouts": run_rollouts, "cost_usd": reported_cost}),
    );
    authorities.insert(
        "usage_ledger".to_string(),
        json!({"rollouts": ledger_rollouts, "cost_usd": ledger_cost}),
    );
    let rollout_divergent = ledger_rollouts.is_some_and(|value| value != run_rollouts as u64);
    let cost_divergent = match (reported_cost, ledger_cost) {
        (Some(left), Some(right)) => (left - right).abs() > 1e-9,
        _ => false,
    };
    let incomplete = reported_cost.is_none() || ledger_cost.is_none() || ledger_rollouts.is_none();
    let status = if rollout_divergent || cost_divergent {
        "divergent"
    } else if incomplete {
        "incomplete"
    } else {
        "aligned"
    };
    GepaReconciliationStatus {
        status: status.to_string(),
        authorities: Value::Object(authorities),
    }
}

pub fn apply_selection_to_result(
    result: &mut GepaRunResult,
    candidates: &[CandidateRecord],
    identities: &CandidateIdentitySet,
    rollout_count: usize,
    ledger_rollout_count: Option<u64>,
    reported_cost: Option<f64>,
    ledger_cost: Option<f64>,
    resolved_pipeline: Value,
) {
    let authority = selection_authority_value(
        candidates,
        identities,
        rollout_count,
        ledger_rollout_count,
        reported_cost,
        ledger_cost,
    );
    result.optimization_selected_candidate = authority
        .get("optimization_selected_candidate")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    result.heldout_measurements_by_candidate = authority
        .get("heldout_measurements_by_candidate")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    result.heldout_best_candidate = authority
        .get("heldout_best_candidate")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    result.deployment_candidate = authority
        .get("deployment_candidate")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    result.improvement_verdict = authority
        .get("improvement_verdict")
        .and_then(Value::as_str)
        .map(str::to_string);
    result.rollout_count_authority = authority
        .get("rollout_count_authority")
        .cloned()
        .unwrap_or(Value::Null);
    result.cost_authority = authority
        .get("cost_authority")
        .cloned()
        .unwrap_or(Value::Null);
    result.reconciliation_status = authority
        .get("reconciliation_status")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    result.resolved_pipeline = resolved_pipeline;
}

pub fn idx_for_candidate_id(
    candidates: &[CandidateRecord],
    candidate_id: Option<&str>,
) -> Option<usize> {
    let candidate_id = candidate_id?;
    candidates
        .iter()
        .position(|candidate| candidate.candidate_id == candidate_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn candidate(
        id: &str,
        source: &str,
        train: Option<f64>,
        heldout: Option<f64>,
    ) -> CandidateRecord {
        CandidateRecord {
            candidate_id: id.to_string(),
            payload: BTreeMap::new(),
            lever_bundle: LeverBundle::from_prompt_payload(id, None, &BTreeMap::new()),
            parent_id: None,
            source: source.to_string(),
            status: "accepted".to_string(),
            minibatch_reward: None,
            train_reward: train,
            heldout_reward: heldout,
            minibatch_scores: Vec::new(),
            train_scores: Vec::new(),
            sensor_frames: Vec::new(),
            acceptance_score: Value::Null,
            acceptance_metadata: Map::new(),
        }
    }

    #[test]
    fn heldout_winner_does_not_replace_optimization_selected() {
        let candidates = vec![
            candidate("seed", "seed", Some(0.4), Some(0.3)),
            candidate("train_best", "proposer", Some(0.8), Some(0.2)),
            candidate("heldout_best", "proposer", Some(0.5), Some(0.9)),
        ];
        let mut identities = CandidateIdentitySet::default();
        identities.freeze_optimization_selected(Some(1));
        identities.record_heldout_best(Some(2));
        identities.apply_deployment_rule(DEPLOYMENT_RULE_OPTIMIZATION_SELECTED);
        assert_eq!(identities.optimization_selected_idx, Some(1));
        assert_eq!(identities.heldout_best_idx, Some(2));
        assert_eq!(identities.deployment_idx, Some(1));
        assert_eq!(identities.best_idx_alias(), Some(1));
        assert_ne!(
            identities.optimization_selected_idx,
            identities.heldout_best_idx
        );
        assert_ne!(identities.deployment_idx, identities.heldout_best_idx);
        let selected = candidate_identity(
            &candidates,
            identities.optimization_selected_idx,
            "train",
            |c| c.train_reward,
        )
        .unwrap();
        let heldout =
            candidate_identity(&candidates, identities.heldout_best_idx, "heldout", |c| {
                c.heldout_reward
            })
            .unwrap();
        let deployment = identities
            .deployment_idx
            .and_then(|idx| candidates.get(idx))
            .unwrap();
        assert_eq!(selected.id, "train_best");
        assert_eq!(heldout.id, "heldout_best");
        assert_eq!(deployment.candidate_id, "train_best");
    }

    #[test]
    fn zero_uplift_is_no_measured_improvement() {
        assert_eq!(
            improvement_verdict(Some(0.5), Some(0.5)),
            VERDICT_NO_MEASURED_IMPROVEMENT
        );
        assert_eq!(
            improvement_verdict(Some(0.5), Some(0.4)),
            VERDICT_NO_MEASURED_IMPROVEMENT
        );
        assert_eq!(
            improvement_verdict(Some(0.5), Some(0.6)),
            VERDICT_IMPROVEMENT_DEMONSTRATED
        );
    }

    #[test]
    fn freeze_is_write_once() {
        let mut identities = CandidateIdentitySet::default();
        identities.freeze_optimization_selected(Some(3));
        identities.freeze_optimization_selected(Some(9));
        assert_eq!(identities.optimization_selected_idx, Some(3));
    }

    #[test]
    fn concurrent_identity_sets_are_isolated() {
        let mut left = CandidateIdentitySet::default();
        let mut right = CandidateIdentitySet::default();
        left.freeze_optimization_selected(Some(1));
        right.freeze_optimization_selected(Some(2));
        left.record_heldout_best(Some(3));
        right.record_heldout_best(Some(4));
        left.apply_deployment_rule(DEPLOYMENT_RULE_OPTIMIZATION_SELECTED);
        right.apply_deployment_rule(DEPLOYMENT_RULE_HELDOUT_BEST);
        assert_eq!(left.optimization_selected_idx, Some(1));
        assert_eq!(right.optimization_selected_idx, Some(2));
        assert_eq!(left.best_idx_alias(), Some(1));
        assert_eq!(right.best_idx_alias(), Some(4));
    }
}
