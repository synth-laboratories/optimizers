use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use synth_optimizer_platform::SensorFrame;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarlCandidate {
    pub candidate_id: String,
    pub generation: usize,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub payload: BTreeMap<String, String>,
    pub source: String,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub train_score: Option<StrategyScore>,
    #[serde(default)]
    pub selection_score: Option<StrategyScore>,
    #[serde(default)]
    pub heldout_score: Option<StrategyScore>,
    #[serde(default)]
    pub sensor_frames: Vec<SensorFrame>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl MarlCandidate {
    pub fn selection_basis(&self) -> Option<&StrategyScore> {
        self.selection_score.as_ref().or(self.train_score.as_ref())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StrategyScore {
    pub primary: f64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub diagnostics: Value,
}

impl StrategyScore {
    pub fn metric(&self, key: &str) -> f64 {
        self.metrics.get(key).copied().unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationArm {
    pub arm_id: String,
    pub payload: BTreeMap<String, String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl EvaluationArm {
    pub fn primary(payload: &BTreeMap<String, String>) -> Self {
        Self {
            arm_id: "primary".to_string(),
            payload: payload.clone(),
            metadata: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutObservation {
    pub rollout_id: String,
    pub candidate_id: String,
    pub task_id: String,
    pub split: String,
    pub stage: String,
    pub arm_id: String,
    pub reward: f64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    pub response: Value,
}

impl RolloutObservation {
    pub fn metric(&self, key: &str) -> f64 {
        self.metrics.get(key).copied().unwrap_or(0.0)
    }

    pub fn is_primary(&self) -> bool {
        self.arm_id == "primary"
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BudgetLedger {
    pub train_limit: usize,
    pub heldout_limit: usize,
    pub train_used: usize,
    pub heldout_used: usize,
    pub proposer_calls: usize,
}

impl BudgetLedger {
    pub fn train_remaining(&self) -> usize {
        self.train_limit.saturating_sub(self.train_used)
    }

    pub fn heldout_remaining(&self) -> usize {
        self.heldout_limit.saturating_sub(self.heldout_used)
    }

    pub fn admit_train(&mut self) -> bool {
        if self.train_remaining() == 0 {
            return false;
        }
        self.train_used += 1;
        true
    }

    pub fn admit_heldout(&mut self) -> bool {
        if self.heldout_remaining() == 0 {
            return false;
        }
        self.heldout_used += 1;
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSplits {
    pub train_rows: Vec<Value>,
    pub selection_rows: Vec<Value>,
    pub heldout_rows: Vec<Value>,
}

impl DatasetSplits {
    pub fn assert_disjoint(&self) -> Result<(), String> {
        let train = task_ids(&self.train_rows)?;
        let selection = task_ids(&self.selection_rows)?;
        let heldout = task_ids(&self.heldout_rows)?;
        assert_no_overlap("train", &train, "selection", &selection)?;
        assert_no_overlap("train", &train, "heldout", &heldout)?;
        assert_no_overlap("selection", &selection, "heldout", &heldout)
    }
}

fn task_ids(rows: &[Value]) -> Result<BTreeSet<String>, String> {
    rows.iter()
        .map(|row| {
            row.get("task_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| "dataset row is missing task_id".to_string())
        })
        .collect()
}

fn assert_no_overlap(
    left_name: &str,
    left: &BTreeSet<String>,
    right_name: &str,
    right: &BTreeSet<String>,
) -> Result<(), String> {
    let overlap = left.intersection(right).cloned().collect::<Vec<_>>();
    if overlap.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{left_name} and {right_name} task ids overlap: {overlap:?}"
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarlRunResult {
    pub schema_version: String,
    pub run_id: String,
    pub variant: String,
    pub environment: String,
    pub seed_candidate_id: String,
    pub champion_candidate_id: String,
    pub heldout_seed_score: Option<StrategyScore>,
    pub heldout_champion_score: Option<StrategyScore>,
    pub heldout_uplift: Option<f64>,
    pub budget: BudgetLedger,
    pub candidate_count: usize,
    pub rollout_count: usize,
    pub manifest_path: String,
}
