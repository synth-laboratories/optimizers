use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_optimizer_platform::{
    task_identity, ContainerClient, OptimizerError, Result, SensorFrame, SynthOptimizerConfig,
    TasksetTasksRequest,
};

use crate::config::MarlPromptoptConfig;
use crate::evaluation::{evaluate_candidate, score_batch, EvaluateCandidateInput};
use crate::proposer::{propose_generation, ProposeGenerationInput};
use crate::strategy::{primary_mean_score, MarlStrategy};
use crate::types::{BudgetLedger, MarlCandidate, MarlRunResult, RolloutObservation, StrategyScore};
use crate::variants::strategy_by_name;

#[derive(Clone, Debug)]
struct SearchSplits {
    train_rows: Vec<Value>,
    selection_rows: Vec<Value>,
    heldout_ids: Vec<String>,
}

pub fn execute_marl_promptopt_from_toml(path: impl AsRef<Path>) -> Result<MarlRunResult> {
    execute_marl_promptopt(MarlPromptoptConfig::from_toml_file(path)?)
}

pub fn execute_marl_promptopt(config: MarlPromptoptConfig) -> Result<MarlRunResult> {
    let gepa_config = config.load_gepa_config()?;
    let strategy = strategy_by_name(&config.variant)?;
    let client = container_client(&gepa_config)?;
    client.health_typed()?;
    client.verify_gepa_contract()?;

    let task_info = client.task_info()?;
    let environment = task_info
        .get("environment")
        .or_else(|| task_info.get("task_id"))
        .or_else(|| task_info.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let mut program = client.program_typed()?;
    program
        .metadata
        .insert("task_info".to_string(), task_info.clone());
    program.validate_for_gepa(
        &gepa_config.candidate.target_modules,
        &gepa_config.seed_candidate,
    )?;

    let splits = load_search_splits(&client, &gepa_config)?;
    if config.experiment.require_disjoint_splits {
        assert_search_splits_disjoint(&splits).map_err(OptimizerError::Config)?;
    }
    if splits.train_rows.is_empty()
        || splits.selection_rows.is_empty()
        || splits.heldout_ids.is_empty()
    {
        return Err(OptimizerError::Config(
            "MARL comparison requires non-empty train, selection, and heldout rows".to_string(),
        ));
    }

    let run_dir = config.run.output_dir.join(&config.run.run_id);
    fs::create_dir_all(&run_dir).map_err(|source| OptimizerError::io(&run_dir, source))?;
    write_json(&run_dir.join("resolved_variant_config.json"), &config)?;
    write_json(
        &run_dir.join("dataset_snapshot.json"),
        &json!({
            "schema_version": "marl_promptopt_dataset_snapshot.v1",
            "environment": environment,
            "train": public_split_snapshot(&splits.train_rows)?,
            "selection": public_split_snapshot(&splits.selection_rows)?,
            "heldout": {
                "row_count": splits.heldout_ids.len(),
                "task_id_digest": string_list_digest(&splits.heldout_ids),
                "rows_loaded": false,
            },
        }),
    )?;

    let seed_payload = if gepa_config.seed_candidate.is_empty() {
        program.seed_candidate.fields.clone()
    } else {
        gepa_config.seed_candidate.clone()
    };
    let mut seed = MarlCandidate {
        candidate_id: "seed".to_string(),
        generation: 0,
        parent_id: None,
        payload: seed_payload.clone(),
        source: "container_seed".to_string(),
        rationale: "Frozen seed prompt program".to_string(),
        train_score: None,
        selection_score: None,
        heldout_score: None,
        sensor_frames: Vec::new(),
        metadata: Map::new(),
    };
    let mut budget = BudgetLedger {
        train_limit: gepa_config.gepa.train_rollout_limit(),
        heldout_limit: gepa_config.gepa.heldout_rollout_limit(),
        ..BudgetLedger::default()
    };
    if config.experiment.compare_seed_on_heldout && budget.heldout_limit % 2 != 0 {
        return Err(OptimizerError::Config(
            "equal paired heldout comparison requires an even heldout rollout limit".to_string(),
        ));
    }

    let seed_rows = sample_rows(
        &splits.train_rows,
        0,
        gepa_config.gepa.minibatch_size.max(1),
    );
    let mut seed_batch = evaluate_candidate(EvaluateCandidateInput {
        client: &client,
        config: &gepa_config,
        strategy: strategy.as_ref(),
        candidate: &seed,
        parent: None,
        seed_payload: &seed_payload,
        rows: &seed_rows,
        split: &gepa_config.taskset.train_split,
        stage: "seed_full_train",
        run_id: &config.run.run_id,
        variant: strategy.name(),
        budget: &mut budget,
        heldout: false,
        primary_only: false,
    })?;
    let seed_score = score_batch(strategy.as_ref(), &seed_batch)?;
    attach_strategy_score(&mut seed_batch.sensor_frames, &seed_score)?;
    seed.train_score = Some(seed_score);
    seed.sensor_frames = seed_batch.sensor_frames.clone();
    let mut observations = seed_batch.observations;
    let mut candidates = vec![seed];

    for generation in 1..=gepa_config.gepa.max_generations {
        if budget.train_remaining() < config.experiment.minimum_rows_per_candidate {
            break;
        }
        let frontier = nondominated_candidate_indices(strategy.as_ref(), &candidates);
        let parent_index = frontier[(generation - 1) % frontier.len()];
        let parent = candidates[parent_index].clone();
        let proposed = propose_generation(ProposeGenerationInput {
            config: &gepa_config,
            program: &program,
            strategy: strategy.as_ref(),
            parent: &parent,
            candidates: &candidates,
            generation,
            reflection_rows: &splits.train_rows,
            run_dir: &run_dir,
        })?;
        budget.proposer_calls += 1;

        let train_rows = sample_rows(
            &splits.train_rows,
            generation,
            gepa_config.gepa.minibatch_size.max(1),
        );
        let generation_start = candidates.len();
        for mut candidate in proposed {
            if budget.train_remaining() < config.experiment.minimum_rows_per_candidate {
                break;
            }
            let mut batch = evaluate_candidate(EvaluateCandidateInput {
                client: &client,
                config: &gepa_config,
                strategy: strategy.as_ref(),
                candidate: &candidate,
                parent: Some(&parent),
                seed_payload: &seed_payload,
                rows: &train_rows,
                split: &gepa_config.taskset.train_split,
                stage: "candidate_minibatch",
                run_id: &config.run.run_id,
                variant: strategy.name(),
                budget: &mut budget,
                heldout: false,
                primary_only: false,
            })?;
            if batch.observations.is_empty() {
                break;
            }
            let score = score_batch(strategy.as_ref(), &batch)?;
            attach_strategy_score(&mut batch.sensor_frames, &score)?;
            candidate.train_score = Some(score);
            candidate.sensor_frames = batch.sensor_frames.clone();
            observations.extend(batch.observations);
            candidates.push(candidate);
        }

        let generation_end = candidates.len();
        if generation_end == generation_start {
            break;
        }
        let mut generation_indices = (generation_start..generation_end).collect::<Vec<_>>();
        generation_indices.sort_by(|left, right| {
            compare_candidates(strategy.as_ref(), &candidates[*right], &candidates[*left])
                .then_with(|| {
                    candidates[*left]
                        .candidate_id
                        .cmp(&candidates[*right].candidate_id)
                })
        });
        generation_indices.truncate(
            config
                .experiment
                .selection_candidates_per_generation
                .min(generation_indices.len()),
        );
        for index in generation_indices {
            if budget.train_remaining() < config.experiment.minimum_rows_per_candidate {
                break;
            }
            let candidate_snapshot = candidates[index].clone();
            let parent_snapshot = candidate_snapshot
                .parent_id
                .as_deref()
                .and_then(|parent_id| {
                    candidates
                        .iter()
                        .find(|candidate| candidate.candidate_id == parent_id)
                })
                .cloned();
            let mut batch = evaluate_candidate(EvaluateCandidateInput {
                client: &client,
                config: &gepa_config,
                strategy: strategy.as_ref(),
                candidate: &candidate_snapshot,
                parent: parent_snapshot.as_ref(),
                seed_payload: &seed_payload,
                rows: &splits.selection_rows,
                split: &gepa_config.taskset.train_split,
                stage: "selection",
                run_id: &config.run.run_id,
                variant: strategy.name(),
                budget: &mut budget,
                heldout: false,
                primary_only: false,
            })?;
            if !batch.observations.is_empty() {
                let score = score_batch(strategy.as_ref(), &batch)?;
                attach_strategy_score(&mut batch.sensor_frames, &score)?;
                candidates[index].selection_score = Some(score);
                candidates[index]
                    .sensor_frames
                    .extend(batch.sensor_frames.clone());
                observations.extend(batch.observations);
            }
        }
        let frontier_candidate_ids = nondominated_candidate_indices(strategy.as_ref(), &candidates)
            .into_iter()
            .map(|index| candidates[index].candidate_id.clone())
            .collect::<Vec<_>>();
        write_json(
            &run_dir.join(format!("generation_{generation:03}.json")),
            &json!({
                "generation": generation,
                "budget": budget,
                "frontier_candidate_ids": frontier_candidate_ids,
                "candidates": candidates,
            }),
        )?;
    }

    let champion_index = best_candidate_index(strategy.as_ref(), &candidates)?;
    let champion_id = candidates[champion_index].candidate_id.clone();
    if config.experiment.require_exact_rollout_budget {
        fill_train_budget(
            &client,
            &gepa_config,
            strategy.as_ref(),
            &candidates[champion_index],
            &seed_payload,
            &splits.train_rows,
            &config.run.run_id,
            &mut budget,
            &mut observations,
        )?;
    }

    // Fetch heldout payloads only after search and parent selection have ended. The public GEPA
    // proposer runs below `run_dir`, so pre-search artifacts contain only a count and digest.
    let heldout_rows = fetch_rows(
        &client,
        &gepa_config.taskset.heldout_split,
        &splits.heldout_ids,
        Value::Object(gepa_config.taskset.filters.clone()),
    )?;
    if heldout_rows.is_empty() {
        return Err(OptimizerError::Config(
            "MARL comparison requires non-empty heldout rows".to_string(),
        ));
    }

    let (heldout_seed, heldout_champion, heldout_observations) = paired_heldout(
        &client,
        &gepa_config,
        strategy.as_ref(),
        &candidates[0],
        &candidates[champion_index],
        &seed_payload,
        &heldout_rows,
        &config.run.run_id,
        &mut budget,
    )?;
    observations.extend(heldout_observations);
    candidates[0].heldout_score = heldout_seed.clone();
    candidates[champion_index].heldout_score = heldout_champion.clone();

    if config.experiment.require_exact_rollout_budget
        && (budget.train_used != budget.train_limit || budget.heldout_used != budget.heldout_limit)
    {
        return Err(OptimizerError::Invariant(format!(
            "exact rollout budget not consumed: train={}/{}, heldout={}/{}",
            budget.train_used, budget.train_limit, budget.heldout_used, budget.heldout_limit
        )));
    }

    let heldout_uplift =
        heldout_seed
            .as_ref()
            .zip(heldout_champion.as_ref())
            .map(|(seed_score, champion_score)| {
                let seed_outcome = score_outcome(seed_score);
                let champion_outcome = score_outcome(champion_score);
                champion_outcome - seed_outcome
            });
    let manifest_path = run_dir.join("result_manifest.json");
    let frontier_candidate_ids = nondominated_candidate_indices(strategy.as_ref(), &candidates)
        .into_iter()
        .map(|index| candidates[index].candidate_id.clone())
        .collect::<Vec<_>>();
    let result = MarlRunResult {
        schema_version: "marl_promptopt_result.v1".to_string(),
        run_id: config.run.run_id.clone(),
        variant: strategy.name().to_string(),
        environment,
        seed_candidate_id: candidates[0].candidate_id.clone(),
        champion_candidate_id: champion_id,
        frontier_candidate_ids,
        heldout_seed_score: heldout_seed,
        heldout_champion_score: heldout_champion,
        heldout_uplift,
        budget: budget.clone(),
        candidate_count: candidates.len(),
        rollout_count: observations.len(),
        manifest_path: manifest_path.display().to_string(),
    };
    write_json(&run_dir.join("candidate_registry.json"), &candidates)?;
    write_json(&run_dir.join("rollouts.json"), &observations)?;
    write_json(&manifest_path, &result)?;
    Ok(result)
}

fn container_client(config: &SynthOptimizerConfig) -> Result<ContainerClient> {
    let url = config
        .container
        .url
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| OptimizerError::Config("container.url is required".to_string()))?;
    ContainerClient::with_headers_and_bearer_env(
        url,
        config.container.headers.clone(),
        config.container.auth_bearer_env.as_deref(),
    )
}

fn load_search_splits(
    client: &ContainerClient,
    config: &SynthOptimizerConfig,
) -> Result<SearchSplits> {
    let selection_ids = if config.gepa.task_pools.pareto.is_empty() {
        return Err(OptimizerError::Config(
            "gepa.task_pools.pareto must name the frozen selection task ids".to_string(),
        ));
    } else {
        config.gepa.task_pools.pareto.clone()
    };
    let selection_set = selection_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut train_ids = if config.gepa.task_pools.minibatch.is_empty()
        && config.gepa.task_pools.reflection.is_empty()
    {
        config.taskset.train_ids.clone()
    } else {
        config
            .gepa
            .task_pools
            .minibatch
            .iter()
            .chain(config.gepa.task_pools.reflection.iter())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    train_ids.retain(|task_id| !selection_set.contains(task_id));
    let heldout_ids = if config.gepa.task_pools.heldout.is_empty() {
        config.taskset.heldout_ids.clone()
    } else {
        config.gepa.task_pools.heldout.clone()
    };
    Ok(SearchSplits {
        train_rows: fetch_rows(
            client,
            &config.taskset.train_split,
            &train_ids,
            Value::Object(config.taskset.filters.clone()),
        )?,
        selection_rows: fetch_rows(
            client,
            &config.taskset.train_split,
            &selection_ids,
            Value::Object(config.taskset.filters.clone()),
        )?,
        heldout_ids,
    })
}

fn assert_search_splits_disjoint(splits: &SearchSplits) -> std::result::Result<(), String> {
    let train_ids = row_task_ids(&splits.train_rows)?;
    let selection_ids = row_task_ids(&splits.selection_rows)?;
    let heldout_ids = splits.heldout_ids.iter().cloned().collect::<BTreeSet<_>>();
    assert_no_task_overlap("train", &train_ids, "selection", &selection_ids)?;
    assert_no_task_overlap("train", &train_ids, "heldout", &heldout_ids)?;
    assert_no_task_overlap("selection", &selection_ids, "heldout", &heldout_ids)
}

fn row_task_ids(rows: &[Value]) -> std::result::Result<BTreeSet<String>, String> {
    rows.iter()
        .map(|row| task_identity(row).map_err(|error| error.to_string()))
        .collect()
}

fn assert_no_task_overlap(
    left_name: &str,
    left: &BTreeSet<String>,
    right_name: &str,
    right: &BTreeSet<String>,
) -> std::result::Result<(), String> {
    let overlap = left.intersection(right).cloned().collect::<Vec<_>>();
    if overlap.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{left_name} and {right_name} task ids overlap: {overlap:?}"
        ))
    }
}

fn public_split_snapshot(rows: &[Value]) -> Result<Value> {
    let task_ids = rows.iter().map(task_identity).collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "row_count": rows.len(),
        "task_ids": task_ids,
        "row_digest": json_digest(rows)?,
    }))
}

fn string_list_digest(values: &[String]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn json_digest(value: &(impl Serialize + ?Sized)) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(serde_json::to_vec(value)?);
    Ok(format!("{:x}", digest.finalize()))
}

fn fetch_rows(
    client: &ContainerClient,
    split: &str,
    task_ids: &[String],
    filters: Value,
) -> Result<Vec<Value>> {
    if task_ids.is_empty() {
        return Err(OptimizerError::Config(format!(
            "no task ids configured for split {split:?}"
        )));
    }
    let expected = task_ids.iter().cloned().collect::<BTreeSet<_>>();
    if expected.len() != task_ids.len() {
        return Err(OptimizerError::Config(format!(
            "duplicate task ids configured for split {split:?}"
        )));
    }
    let request = TasksetTasksRequest::new(split, task_ids, filters);
    let rows = client.taskset_tasks_typed(&request)?.tasks;
    let actual = rows
        .iter()
        .map(task_identity)
        .collect::<Result<BTreeSet<_>>>()?;
    if rows.len() != task_ids.len() || actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(OptimizerError::Container(format!(
            "taskset returned a non-exact split {split:?}: requested={}, returned={}, missing={missing:?}, unexpected={unexpected:?}",
            task_ids.len(),
            rows.len(),
        )));
    }
    Ok(rows)
}

fn sample_rows(rows: &[Value], generation: usize, count: usize) -> Vec<Value> {
    if rows.is_empty() || count == 0 {
        return Vec::new();
    }
    let count = count.min(rows.len());
    let start = (generation.saturating_mul(count)) % rows.len();
    (0..count)
        .map(|offset| rows[(start + offset) % rows.len()].clone())
        .collect()
}

fn best_candidate_index(
    strategy: &dyn MarlStrategy,
    candidates: &[MarlCandidate],
) -> Result<usize> {
    candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.selection_basis().is_some())
        .max_by(|(_, left), (_, right)| {
            compare_candidates(strategy, left, right)
                .then_with(|| right.candidate_id.cmp(&left.candidate_id))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| OptimizerError::Invariant("no scored MARL candidate".to_string()))
}

fn nondominated_candidate_indices(
    strategy: &dyn MarlStrategy,
    candidates: &[MarlCandidate],
) -> Vec<usize> {
    let scored = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.selection_basis().is_some())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut frontier = scored
        .iter()
        .copied()
        .filter(|candidate_index| {
            !scored.iter().copied().any(|other_index| {
                other_index != *candidate_index
                    && compare_candidates(
                        strategy,
                        &candidates[other_index],
                        &candidates[*candidate_index],
                    ) == Ordering::Greater
            })
        })
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        candidates[*left]
            .candidate_id
            .cmp(&candidates[*right].candidate_id)
    });
    if frontier.is_empty() {
        vec![0]
    } else {
        frontier
    }
}

fn attach_strategy_score(frames: &mut [SensorFrame], score: &StrategyScore) -> Result<()> {
    let score_value = serde_json::to_value(score)?;
    for frame in frames {
        frame
            .metadata
            .insert("marl_strategy_score".to_string(), score_value.clone());
        let actionable = frame
            .actionable_side_info
            .get_or_insert_with(|| Value::Object(Map::new()));
        if !actionable.is_object() {
            *actionable = Value::Object(Map::new());
        }
        actionable
            .as_object_mut()
            .expect("actionable side info is an object")
            .insert("marl_strategy_score".to_string(), score_value.clone());
    }
    Ok(())
}

fn compare_candidates(
    strategy: &dyn MarlStrategy,
    left: &MarlCandidate,
    right: &MarlCandidate,
) -> Ordering {
    match (left.selection_basis(), right.selection_basis()) {
        (Some(left), Some(right)) => strategy.compare(left, right),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_train_budget(
    client: &ContainerClient,
    config: &SynthOptimizerConfig,
    strategy: &dyn MarlStrategy,
    champion: &MarlCandidate,
    seed_payload: &BTreeMap<String, String>,
    rows: &[Value],
    run_id: &str,
    budget: &mut BudgetLedger,
    observations: &mut Vec<RolloutObservation>,
) -> Result<()> {
    let mut index = 0usize;
    while budget.train_remaining() > 0 {
        let row = rows[index % rows.len()].clone();
        let batch = evaluate_candidate(EvaluateCandidateInput {
            client,
            config,
            strategy,
            candidate: champion,
            parent: None,
            seed_payload,
            rows: &[row],
            split: &config.taskset.train_split,
            stage: "budget_audit",
            run_id,
            variant: strategy.name(),
            budget,
            heldout: false,
            primary_only: true,
        })?;
        if batch.observations.is_empty() {
            return Err(OptimizerError::Invariant(
                "train budget audit made no progress".to_string(),
            ));
        }
        observations.extend(batch.observations);
        index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn paired_heldout(
    client: &ContainerClient,
    config: &SynthOptimizerConfig,
    strategy: &dyn MarlStrategy,
    seed: &MarlCandidate,
    champion: &MarlCandidate,
    seed_payload: &BTreeMap<String, String>,
    rows: &[Value],
    run_id: &str,
    budget: &mut BudgetLedger,
) -> Result<(
    Option<StrategyScore>,
    Option<StrategyScore>,
    Vec<RolloutObservation>,
)> {
    if budget.heldout_limit == 0 {
        return Ok((None, None, Vec::new()));
    }
    let mut seed_observations = Vec::new();
    let mut champion_observations = Vec::new();
    let mut index = 0usize;
    while budget.heldout_remaining() >= 2 {
        let row = rows[index % rows.len()].clone();
        let seed_batch = evaluate_candidate(EvaluateCandidateInput {
            client,
            config,
            strategy,
            candidate: seed,
            parent: None,
            seed_payload,
            rows: std::slice::from_ref(&row),
            split: &config.taskset.heldout_split,
            stage: "heldout",
            run_id,
            variant: strategy.name(),
            budget,
            heldout: true,
            primary_only: true,
        })?;
        let champion_batch = evaluate_candidate(EvaluateCandidateInput {
            client,
            config,
            strategy,
            candidate: champion,
            parent: None,
            seed_payload,
            rows: &[row],
            split: &config.taskset.heldout_split,
            stage: "heldout",
            run_id,
            variant: strategy.name(),
            budget,
            heldout: true,
            primary_only: true,
        })?;
        seed_observations.extend(seed_batch.observations);
        champion_observations.extend(champion_batch.observations);
        index += 1;
    }
    let seed_score =
        (!seed_observations.is_empty()).then(|| primary_mean_score(&seed_observations));
    let champion_score =
        (!champion_observations.is_empty()).then(|| primary_mean_score(&champion_observations));
    let mut all = seed_observations;
    all.extend(champion_observations);
    Ok((seed_score, champion_score, all))
}

fn score_outcome(score: &StrategyScore) -> f64 {
    score
        .metrics
        .get("outcome_success")
        .copied()
        .unwrap_or(score.primary)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{text}\n")).map_err(|source| OptimizerError::io(path, source))
}
