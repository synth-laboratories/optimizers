use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::cache::stable_value_hash;
use crate::sensors::{ObjectiveScore, SensorFrame};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveSetRecord {
    pub schema_version: String,
    pub objective_set_id: String,
    pub objective_set_hash: String,
    pub selection_objective: String,
    pub frontier_type: String,
    pub objectives: Vec<ObjectiveSpec>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    pub schema_version: String,
    pub objective_id: String,
    pub name: String,
    pub direction: String,
    pub source: String,
    pub aggregation: String,
    pub split_policy: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreRecord {
    pub schema_version: String,
    pub score_id: String,
    pub objective_id: String,
    pub objective: String,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub source: String,
    pub value: f64,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreVectorRecord {
    pub schema_version: String,
    pub score_vector_id: String,
    pub objective_set_id: String,
    pub objective_set_hash: String,
    pub candidate_id: String,
    pub split: String,
    pub evaluation_stage: String,
    pub status: String,
    pub selection_objective: String,
    #[serde(default)]
    pub selection_score: Option<f64>,
    #[serde(default)]
    pub mean_reward: Option<f64>,
    pub score_count: u64,
    #[serde(default)]
    pub objective_values: Map<String, Value>,
    #[serde(default)]
    pub covered_objectives: Vec<String>,
    #[serde(default)]
    pub missing_objectives: Vec<String>,
    #[serde(default)]
    pub example_ids: Vec<String>,
    #[serde(default)]
    pub seeds: Vec<i64>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoComparisonRecord {
    pub schema_version: String,
    pub pareto_comparison_id: String,
    pub objective_set_id: String,
    pub objective_set_hash: String,
    pub frontier_type: String,
    pub split: String,
    pub evaluation_stage: String,
    pub challenger_candidate_id: String,
    pub incumbent_candidate_id: String,
    pub challenger_score_vector_id: String,
    pub incumbent_score_vector_id: String,
    pub result: String,
    #[serde(default)]
    pub dominance: Value,
    pub rationale: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorScoreRecords {
    pub objectives: Vec<ObjectiveSpec>,
    pub scores: Vec<ScoreRecord>,
}

impl ObjectiveSetRecord {
    pub fn from_specs(
        selection_objective: &str,
        frontier_type: &str,
        mut objectives: Vec<ObjectiveSpec>,
        metadata: Map<String, Value>,
    ) -> Self {
        objectives.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.source.cmp(&right.source))
                .then_with(|| left.objective_id.cmp(&right.objective_id))
        });
        objectives.dedup_by(|left, right| {
            left.name == right.name
                && left.source == right.source
                && left.direction == right.direction
        });
        let identity = json!({
            "schema_version": "objective_set_identity.v1",
            "selection_objective": selection_objective,
            "frontier_type": frontier_type,
            "objectives": objectives.clone(),
        });
        let objective_set_hash = stable_value_hash(&identity);
        Self {
            schema_version: "objective_set.v1".to_string(),
            objective_set_id: format!("objective_set_{}", &objective_set_hash[..16]),
            objective_set_hash,
            selection_objective: selection_objective.to_string(),
            frontier_type: frontier_type.to_string(),
            objectives,
            metadata,
        }
    }
}

impl SensorScoreRecords {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let objective_scores = if frame.objective_scores.is_empty() {
            vec![ObjectiveScore {
                objective: "outcome_reward".to_string(),
                value: frame.reward,
                source: "sensor_frame.reward".to_string(),
                rationale: None,
                metadata: Map::new(),
            }]
        } else {
            frame.objective_scores.clone()
        };
        let mut objectives = Vec::new();
        let mut scores = Vec::new();
        for objective_score in objective_scores {
            let objective = ObjectiveSpec::from_objective_score(&objective_score);
            let score = ScoreRecord::from_sensor_frame(frame, &objective, &objective_score);
            if !objectives
                .iter()
                .any(|existing: &ObjectiveSpec| existing.objective_id == objective.objective_id)
            {
                objectives.push(objective);
            }
            scores.push(score);
        }
        Self { objectives, scores }
    }
}

impl ObjectiveSpec {
    pub fn from_objective_score(score: &ObjectiveScore) -> Self {
        Self {
            schema_version: "objective_spec.v1".to_string(),
            objective_id: stable_id("objective", &[&score.objective, &score.source]),
            name: score.objective.clone(),
            direction: "maximize".to_string(),
            source: score.source.clone(),
            aggregation: "mean".to_string(),
            split_policy: "per_split_then_overall".to_string(),
            metadata: score.metadata.clone(),
        }
    }
}

impl ScoreRecord {
    pub fn from_sensor_frame(
        frame: &SensorFrame,
        objective: &ObjectiveSpec,
        score: &ObjectiveScore,
    ) -> Self {
        Self {
            schema_version: "score_record.v1".to_string(),
            score_id: stable_id(
                "score",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    &objective.objective_id,
                    &score.source,
                ],
            ),
            objective_id: objective.objective_id.clone(),
            objective: score.objective.clone(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            rollout_id: frame.rollout_id.clone(),
            example_id: frame.example_id.clone(),
            seed: frame.seed,
            split: frame.split.clone(),
            evaluation_stage: frame.evaluation_stage.clone(),
            source: score.source.clone(),
            value: score.value,
            rationale: score.rationale.clone(),
            metadata: score.metadata.clone(),
        }
    }
}

impl ScoreVectorRecord {
    pub fn from_scores(
        objective_set: &ObjectiveSetRecord,
        candidate_id: &str,
        split: &str,
        evaluation_stage: &str,
        scores: &[ScoreRecord],
        metadata: Map<String, Value>,
    ) -> Self {
        let mut totals = BTreeMap::<String, (f64, u64)>::new();
        let mut example_ids = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        for score in scores {
            let entry = totals.entry(score.objective.clone()).or_insert((0.0, 0));
            entry.0 += score.value;
            entry.1 += 1;
            example_ids.insert(score.example_id.clone());
            seeds.insert(score.seed);
        }

        let mut objective_values = Map::new();
        for (objective, (total, count)) in &totals {
            if *count > 0 {
                objective_values.insert(objective.clone(), json!(total / *count as f64));
            }
        }

        let declared = objective_set
            .objectives
            .iter()
            .map(|objective| objective.name.clone())
            .collect::<BTreeSet<_>>();
        let covered_objectives = declared
            .iter()
            .filter(|objective| objective_values.contains_key(*objective))
            .cloned()
            .collect::<Vec<_>>();
        let missing_objectives = declared
            .iter()
            .filter(|objective| !objective_values.contains_key(*objective))
            .cloned()
            .collect::<Vec<_>>();
        let selection_score = objective_values
            .get(&objective_set.selection_objective)
            .and_then(Value::as_f64);
        let mean_reward = objective_values
            .get("outcome_reward")
            .and_then(Value::as_f64)
            .or(selection_score);
        let status = if scores.is_empty() {
            "empty"
        } else if missing_objectives.is_empty() {
            "complete"
        } else {
            "partial"
        };
        Self {
            schema_version: "score_vector.v1".to_string(),
            score_vector_id: stable_id(
                "scorevector",
                &[
                    &objective_set.objective_set_id,
                    candidate_id,
                    split,
                    evaluation_stage,
                ],
            ),
            objective_set_id: objective_set.objective_set_id.clone(),
            objective_set_hash: objective_set.objective_set_hash.clone(),
            candidate_id: candidate_id.to_string(),
            split: split.to_string(),
            evaluation_stage: evaluation_stage.to_string(),
            status: status.to_string(),
            selection_objective: objective_set.selection_objective.clone(),
            selection_score,
            mean_reward,
            score_count: scores.len() as u64,
            objective_values,
            covered_objectives,
            missing_objectives,
            example_ids: example_ids.into_iter().collect(),
            seeds: seeds.into_iter().collect(),
            metadata,
        }
    }

    pub fn objective_value(&self, objective: &str) -> Option<f64> {
        self.objective_values.get(objective).and_then(Value::as_f64)
    }
}

impl ParetoComparisonRecord {
    pub fn from_vectors(
        objective_set: &ObjectiveSetRecord,
        frontier_type: &str,
        split: &str,
        evaluation_stage: &str,
        challenger: &ScoreVectorRecord,
        incumbent: &ScoreVectorRecord,
        metadata: Map<String, Value>,
    ) -> Self {
        let mut better = Vec::new();
        let mut worse = Vec::new();
        let mut equal = Vec::new();
        let mut missing = Vec::new();
        for objective in &objective_set.objectives {
            let left = challenger.objective_value(&objective.name);
            let right = incumbent.objective_value(&objective.name);
            let (Some(left), Some(right)) = (left, right) else {
                missing.push(objective.name.clone());
                continue;
            };
            let direction = objective_direction(&objective.direction);
            let delta = (left - right) * direction;
            if delta > f64::EPSILON {
                better.push(objective.name.clone());
            } else if delta < -f64::EPSILON {
                worse.push(objective.name.clone());
            } else {
                equal.push(objective.name.clone());
            }
        }
        let result = if better.is_empty() && worse.is_empty() && !equal.is_empty() {
            "tie"
        } else if !better.is_empty() && worse.is_empty() {
            "challenger_dominates"
        } else if better.is_empty() && !worse.is_empty() {
            "incumbent_dominates"
        } else if !better.is_empty() && !worse.is_empty() {
            "mixed"
        } else {
            "incomparable"
        };
        let dominance = json!({
            "better_objectives": better,
            "worse_objectives": worse,
            "equal_objectives": equal,
            "missing_objectives": missing,
            "challenger_selection_score": challenger.selection_score,
            "incumbent_selection_score": incumbent.selection_score,
        });
        let rationale = match result {
            "challenger_dominates" => "challenger is at least as good on all comparable objectives and better on at least one",
            "incumbent_dominates" => "incumbent is at least as good on all comparable objectives and better on at least one",
            "tie" => "challenger and incumbent tie on comparable objectives",
            "mixed" => "challenger improves some objectives and regresses others",
            _ => "comparison has no comparable objective values",
        };
        Self {
            schema_version: "pareto_comparison.v1".to_string(),
            pareto_comparison_id: stable_id(
                "pareto",
                &[
                    &objective_set.objective_set_id,
                    frontier_type,
                    split,
                    evaluation_stage,
                    &challenger.candidate_id,
                    &incumbent.candidate_id,
                ],
            ),
            objective_set_id: objective_set.objective_set_id.clone(),
            objective_set_hash: objective_set.objective_set_hash.clone(),
            frontier_type: frontier_type.to_string(),
            split: split.to_string(),
            evaluation_stage: evaluation_stage.to_string(),
            challenger_candidate_id: challenger.candidate_id.clone(),
            incumbent_candidate_id: incumbent.candidate_id.clone(),
            challenger_score_vector_id: challenger.score_vector_id.clone(),
            incumbent_score_vector_id: incumbent.score_vector_id.clone(),
            result: result.to_string(),
            dominance,
            rationale: rationale.to_string(),
            metadata,
        }
    }
}

fn objective_direction(direction: &str) -> f64 {
    match direction.trim().to_ascii_lowercase().as_str() {
        "min" | "minimize" | "lower" | "lower_is_better" | "down" => -1.0,
        _ => 1.0,
    }
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}
