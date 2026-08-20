use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use synth_optimizer_platform::{ContainerClient, OptimizerError, Result};

use crate::campaign::CampaignBinding;
use crate::candidate::{MapoBranchCheckpoint, MapoCandidate, MapoRolloutRecord};
use crate::config::{MapoConfig, MapoExecutionOptions};
use crate::executor::{
    execute_branch_discovery_seed_rollouts, execute_branch_discovery_task_rollouts,
    execute_candidate_branch_rollouts, execute_candidate_rollouts, execute_candidate_task_rollouts,
    rollout_request,
};
use crate::proposer::{propose_candidates, MapoProposerInput};
use crate::review::build_mapo_review_rows;
use crate::scoring::{mapo_score_better, MapoHeldoutComparison, MapoScore};

pub const MAPO_ALGORITHM_ID: &str = "synth_mapo.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapoRunResult {
    pub algorithm_id: String,
    pub run_id: String,
    pub run_dir: String,
    pub champion: MapoCandidate,
    pub candidates: Vec<MapoCandidate>,
    pub train_rollouts: Vec<MapoRolloutRecord>,
    pub selection_rollouts: Vec<MapoRolloutRecord>,
    pub branch_rollouts: Vec<MapoRolloutRecord>,
    pub branch_checkpoints: Vec<MapoBranchCheckpoint>,
    pub baseline_heldout_rollouts: Vec<MapoRolloutRecord>,
    pub heldout_rollouts: Vec<MapoRolloutRecord>,
    pub heldout_comparison: Option<MapoHeldoutComparison>,
    pub proposer_receipts: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debrief_evidence: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub campaign_manifest_receipt: Option<Value>,
    pub rollout_request_preview: Value,
    pub dry_run: bool,
}

pub fn execute_mapo_with_options(
    config: MapoConfig,
    options: MapoExecutionOptions,
) -> Result<MapoRunResult> {
    let run_dir = config.run_dir();
    let run_id = config.run.run_id.clone();
    let dry_run = options.dry_run;
    let resume_requested = options.resume;
    match execute_mapo_with_options_inner(config, options) {
        Ok(result) => Ok(result),
        Err(error) => {
            if let Err(persist_error) =
                persist_mapo_failure_artifacts(&run_dir, &run_id, dry_run, resume_requested, &error)
            {
                eprintln!(
                    "failed to persist MAPO failure artifacts for run_id={run_id}: {persist_error}"
                );
            }
            Err(error)
        }
    }
}

fn execute_mapo_with_options_inner(
    config: MapoConfig,
    options: MapoExecutionOptions,
) -> Result<MapoRunResult> {
    config.validate()?;
    let campaign_binding = CampaignBinding::load(&config)?;
    let run_dir = config.run_dir();
    let artifact_dir = run_dir.join("artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    if let Some(binding) = &campaign_binding {
        binding.write_artifact(&artifact_dir)?;
    }
    write_json_pretty(
        &artifact_dir.join("resolved_config.json"),
        &config.resolved_config_value()?,
    )?;

    let mut baseline_candidate = config.seed_candidate.clone();
    if baseline_candidate.id.trim().is_empty() {
        baseline_candidate.id = "mapo_seed".to_string();
    }
    let mut champion = baseline_candidate.clone();
    if champion.id.trim().is_empty() {
        champion.id = "mapo_seed".to_string();
    }
    let mut candidates = vec![champion.clone()];
    let mut train_rollouts = Vec::new();
    let mut selection_rollouts = Vec::new();
    let mut branch_rollouts = Vec::new();
    let mut branch_checkpoints = Vec::new();
    let mut baseline_heldout_rollouts = Vec::new();
    let mut heldout_rollouts = Vec::new();
    let mut heldout_comparison = None;
    let mut preview_requests = Vec::new();
    let mut proposer_receipts: Vec<Value> = Vec::new();

    let container_url = config
        .container
        .url
        .as_deref()
        .ok_or_else(|| OptimizerError::Config("container.url is required".to_string()))?;
    let client = ContainerClient::with_headers_bearer_env_and_timeout(
        container_url,
        config.container.headers.clone(),
        config.container.auth_bearer_env.as_deref(),
        config.mapo.request_timeout_seconds,
    )?;

    preview_requests.push(rollout_request(
        &config,
        &champion,
        "train",
        "preview",
        config.taskset.train_seeds[0],
        0,
        "mapo_preview_train",
    ));
    let rollout_request_preview = json!({
        "schema_version": "mapo_rollout_request_preview.v1",
        "run_id": &config.run.run_id,
        "requests": preview_requests,
    });
    persist_mapo_artifacts(
        &run_dir,
        &artifact_dir,
        &config,
        campaign_binding.as_ref(),
        &champion,
        &candidates,
        &train_rollouts,
        &selection_rollouts,
        &branch_rollouts,
        &branch_checkpoints,
        &baseline_heldout_rollouts,
        &heldout_rollouts,
        heldout_comparison.as_ref(),
        &rollout_request_preview,
        options.dry_run,
        options.resume,
    )?;

    if !options.dry_run {
        wait_for_container_ready(
            &client,
            container_url,
            config.mapo.container_connect_timeout_seconds,
        )?;
        // Evaluate the seed before the first proposal. The grid proposer ignored
        // evidence, so it did not matter that generation 1 ran before any
        // rollout existed; a proposer that reads rollouts would have been handed
        // an empty workspace and asked to diagnose a team it had never seen.
        {
            let records = execute_candidate_rollouts(
                &client,
                &config,
                &champion,
                "train",
                "g0_seed",
                &config.taskset.train_seeds,
                config.mapo.rollouts_per_candidate,
            )?;
            champion.train_score = Some(MapoScore::from_rollouts(&records));
            train_rollouts.extend(records);
            candidates.clear();
            candidates.push(champion.clone());
        }

        for generation in 1..=config.mapo.max_generations {
            let proposal = propose_candidates(MapoProposerInput {
                config: &config,
                parent: &champion,
                candidates: &candidates,
                train_rollouts: &train_rollouts,
                branch_checkpoints: &branch_checkpoints,
                generation,
                workspace_dir: run_dir
                    .join("proposer_workspaces")
                    .join(format!("generation_{generation:03}")),
            })?;
            proposer_receipts.push(proposal.receipt);
            write_json_pretty(
                &artifact_dir.join("mapo_proposer_receipts.json"),
                &json!({
                    "schema_version": "mapo_proposer_receipts.v1",
                    "run_id": &config.run.run_id,
                    "receipts": &proposer_receipts,
                }),
            )?;
            let mut generation_candidates = proposal.candidates;
            generation_candidates.push(champion.clone());
            let mut evaluated_generation = Vec::new();
            for (candidate_index, candidate) in generation_candidates.into_iter().enumerate() {
                let rollout_group = format!("g{generation}_c{candidate_index}");
                let records = execute_candidate_rollouts(
                    &client,
                    &config,
                    &candidate,
                    "train",
                    &rollout_group,
                    &config.taskset.train_seeds,
                    config.mapo.rollouts_per_candidate,
                )?;
                let score = MapoScore::from_rollouts(&records);
                let mut evaluated = candidate;
                evaluated.train_score = Some(score);
                train_rollouts.extend(records);
                let current_score = evaluated.train_score.as_ref().ok_or_else(|| {
                    OptimizerError::Invariant("MAPO candidate missing train score".to_string())
                })?;
                let champion_score = champion.train_score.as_ref();
                if champion_score
                    .map(|score| mapo_score_better(current_score, score))
                    .unwrap_or(true)
                {
                    champion = evaluated.clone();
                }
                evaluated_generation.push(evaluated.clone());
                candidates.push(evaluated);
                persist_mapo_artifacts(
                    &run_dir,
                    &artifact_dir,
                    &config,
                    campaign_binding.as_ref(),
                    &champion,
                    &candidates,
                    &train_rollouts,
                    &selection_rollouts,
                    &branch_rollouts,
                    &branch_checkpoints,
                    &baseline_heldout_rollouts,
                    &heldout_rollouts,
                    heldout_comparison.as_ref(),
                    &rollout_request_preview,
                    options.dry_run,
                    options.resume,
                )?;
            }
            if !config.taskset.selection_seeds.is_empty()
                || !config.taskset.selection_task_instance_ids.is_empty()
            {
                let generation_branch_checkpoints = if config.mapo.branch_selection_enabled {
                    let rollout_group = format!("g{generation}_branch_discovery");
                    let (discovery_records, discovered_checkpoints) =
                        if config.taskset.selection_task_instance_ids.is_empty() {
                            execute_branch_discovery_seed_rollouts(
                                &client,
                                &config,
                                &champion,
                                "branch_discovery",
                                &rollout_group,
                                &config.taskset.selection_seeds,
                            )?
                        } else {
                            execute_branch_discovery_task_rollouts(
                                &client,
                                &config,
                                &champion,
                                "branch_discovery",
                                &rollout_group,
                                &config.taskset.selection_task_instance_ids,
                            )?
                        };
                    branch_rollouts.extend(discovery_records);
                    branch_checkpoints.extend(discovered_checkpoints.clone());
                    persist_mapo_artifacts(
                        &run_dir,
                        &artifact_dir,
                        &config,
                        campaign_binding.as_ref(),
                        &champion,
                        &candidates,
                        &train_rollouts,
                        &selection_rollouts,
                        &branch_rollouts,
                        &branch_checkpoints,
                        &baseline_heldout_rollouts,
                        &heldout_rollouts,
                        heldout_comparison.as_ref(),
                        &rollout_request_preview,
                        options.dry_run,
                        options.resume,
                    )?;
                    discovered_checkpoints
                } else {
                    Vec::new()
                };
                let selection_candidates = top_train_candidates(
                    evaluated_generation,
                    config.mapo.selection_top_k,
                    &champion.id,
                );
                let mut selection_champion: Option<MapoCandidate> = None;
                for (candidate_index, candidate) in selection_candidates.into_iter().enumerate() {
                    let rollout_group = format!("g{generation}_s{candidate_index}");
                    let records = if config.mapo.branch_selection_enabled {
                        execute_candidate_branch_rollouts(
                            &client,
                            &config,
                            &candidate,
                            "branch_selection",
                            &rollout_group,
                            &generation_branch_checkpoints,
                        )?
                    } else if config.taskset.selection_task_instance_ids.is_empty() {
                        execute_candidate_rollouts(
                            &client,
                            &config,
                            &candidate,
                            "selection",
                            &rollout_group,
                            &config.taskset.selection_seeds,
                            config.mapo.selection_rollouts_per_candidate,
                        )?
                    } else {
                        execute_candidate_task_rollouts(
                            &client,
                            &config,
                            &candidate,
                            "selection",
                            &rollout_group,
                            &config.taskset.selection_task_instance_ids,
                            config.mapo.selection_rollouts_per_candidate,
                        )?
                    };
                    let score = MapoScore::from_rollouts(&records);
                    let mut evaluated = candidate;
                    evaluated.selection_score = Some(score);
                    if config.mapo.branch_selection_enabled {
                        branch_rollouts.extend(records.clone());
                    }
                    selection_rollouts.extend(records);
                    let current_score = evaluated.selection_score.as_ref().ok_or_else(|| {
                        OptimizerError::Invariant(
                            "MAPO candidate missing selection score".to_string(),
                        )
                    })?;
                    if current_score.messages_delivered
                        >= config.mapo.selection_min_messages_delivered
                    {
                        let champion_score = selection_champion
                            .as_ref()
                            .and_then(|candidate| candidate.selection_score.as_ref());
                        if champion_score
                            .map(|score| mapo_score_better(current_score, score))
                            .unwrap_or(true)
                        {
                            selection_champion = Some(evaluated.clone());
                        }
                    }
                    candidates.push(evaluated);
                    persist_mapo_artifacts(
                        &run_dir,
                        &artifact_dir,
                        &config,
                        campaign_binding.as_ref(),
                        &champion,
                        &candidates,
                        &train_rollouts,
                        &selection_rollouts,
                        &branch_rollouts,
                        &branch_checkpoints,
                        &baseline_heldout_rollouts,
                        &heldout_rollouts,
                        heldout_comparison.as_ref(),
                        &rollout_request_preview,
                        options.dry_run,
                        options.resume,
                    )?;
                }
                if let Some(selected) = selection_champion {
                    champion = selected;
                } else if config.mapo.selection_min_messages_delivered > 0 {
                    return Err(OptimizerError::Invariant(format!(
                        "no MAPO selection candidate delivered at least {} messages",
                        config.mapo.selection_min_messages_delivered
                    )));
                }
            }
            persist_mapo_artifacts(
                &run_dir,
                &artifact_dir,
                &config,
                campaign_binding.as_ref(),
                &champion,
                &candidates,
                &train_rollouts,
                &selection_rollouts,
                &branch_rollouts,
                &branch_checkpoints,
                &baseline_heldout_rollouts,
                &heldout_rollouts,
                heldout_comparison.as_ref(),
                &rollout_request_preview,
                options.dry_run,
                options.resume,
            )?;
        }

        baseline_heldout_rollouts = execute_candidate_rollouts(
            &client,
            &config,
            &baseline_candidate,
            "heldout_baseline",
            "heldout_baseline",
            &config.taskset.heldout_seeds,
            config.mapo.heldout_rollouts_per_arm,
        )?;
        persist_mapo_artifacts(
            &run_dir,
            &artifact_dir,
            &config,
            campaign_binding.as_ref(),
            &champion,
            &candidates,
            &train_rollouts,
            &selection_rollouts,
            &branch_rollouts,
            &branch_checkpoints,
            &baseline_heldout_rollouts,
            &heldout_rollouts,
            heldout_comparison.as_ref(),
            &rollout_request_preview,
            options.dry_run,
            options.resume,
        )?;
        let heldout_records = execute_candidate_rollouts(
            &client,
            &config,
            &champion,
            "heldout",
            "heldout_champion",
            &config.taskset.heldout_seeds,
            config.mapo.heldout_rollouts_per_arm,
        )?;
        champion.heldout_score = Some(MapoScore::from_rollouts(&heldout_records));
        heldout_rollouts = heldout_records;
        heldout_comparison = Some(MapoHeldoutComparison::new(
            baseline_candidate.id.clone(),
            champion.id.clone(),
            &baseline_heldout_rollouts,
            &heldout_rollouts,
            config.mapo.heldout_min_paired_episodes_per_arm,
            config.mapo.heldout_min_success_delta_pp,
        )?);
    }

    let debrief_evidence = campaign_binding
        .as_ref()
        .map(|binding| binding.evidence_value(&config));
    let campaign_manifest_receipt = campaign_binding
        .as_ref()
        .map(CampaignBinding::receipt_value);
    let result = MapoRunResult {
        algorithm_id: MAPO_ALGORITHM_ID.to_string(),
        run_id: config.run.run_id.clone(),
        run_dir: run_dir.display().to_string(),
        champion: champion.clone(),
        candidates,
        train_rollouts,
        selection_rollouts,
        branch_rollouts,
        branch_checkpoints,
        baseline_heldout_rollouts,
        heldout_rollouts,
        heldout_comparison,
        proposer_receipts,
        debrief_evidence,
        campaign_manifest_receipt,
        rollout_request_preview,
        dry_run: options.dry_run,
    };
    persist_mapo_artifacts(
        &run_dir,
        &artifact_dir,
        &config,
        campaign_binding.as_ref(),
        &result.champion,
        &result.candidates,
        &result.train_rollouts,
        &result.selection_rollouts,
        &result.branch_rollouts,
        &result.branch_checkpoints,
        &result.baseline_heldout_rollouts,
        &result.heldout_rollouts,
        result.heldout_comparison.as_ref(),
        &result.rollout_request_preview,
        options.dry_run,
        options.resume,
    )?;
    Ok(result)
}

fn persist_mapo_failure_artifacts(
    run_dir: &Path,
    run_id: &str,
    dry_run: bool,
    resume_requested: bool,
    error: &OptimizerError,
) -> Result<()> {
    let artifact_dir = run_dir.join("artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    let failure = json!({
        "schema_version": "mapo_failure_manifest.v1",
        "algorithm_id": MAPO_ALGORITHM_ID,
        "run_id": run_id,
        "run_dir": run_dir.display().to_string(),
        "status": "failed",
        "terminal": true,
        "dry_run": dry_run,
        "resume_requested": resume_requested,
        "error_code": error.error_code(),
        "error": error.to_string(),
    });
    write_json_pretty(&artifact_dir.join("mapo_failure_manifest.json"), &failure)?;
    write_json_pretty(&artifact_dir.join("result_manifest.json"), &failure)?;
    write_json_pretty(
        &run_dir.join("mapo_status.json"),
        &json!({
            "schema_version": "mapo_status.v1",
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": run_id,
            "status": "failed",
            "terminal": true,
            "dry_run": dry_run,
            "resume_requested": resume_requested,
            "error_code": error.error_code(),
            "error": error.to_string(),
        }),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn persist_mapo_artifacts(
    run_dir: &Path,
    artifact_dir: &Path,
    config: &MapoConfig,
    campaign_binding: Option<&CampaignBinding>,
    champion: &MapoCandidate,
    candidates: &[MapoCandidate],
    train_rollouts: &[MapoRolloutRecord],
    selection_rollouts: &[MapoRolloutRecord],
    branch_rollouts: &[MapoRolloutRecord],
    branch_checkpoints: &[MapoBranchCheckpoint],
    baseline_heldout_rollouts: &[MapoRolloutRecord],
    heldout_rollouts: &[MapoRolloutRecord],
    heldout_comparison: Option<&MapoHeldoutComparison>,
    rollout_request_preview: &Value,
    dry_run: bool,
    resume_requested: bool,
) -> Result<()> {
    if let Some(binding) = campaign_binding {
        binding.write_artifact(artifact_dir)?;
    }
    let debrief_evidence = campaign_binding.map(|binding| binding.evidence_value(config));
    let campaign_manifest_receipt = campaign_binding.map(CampaignBinding::receipt_value);
    write_json_pretty(
        &artifact_dir.join("mapo_rollout_request_preview.json"),
        rollout_request_preview,
    )?;
    write_json_pretty(
        &artifact_dir.join("mapo_candidate_registry.json"),
        &json!({
            "schema_version": "mapo_candidate_registry.v1",
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": &config.run.run_id,
            "champion_candidate_id": &champion.id,
            "candidates": candidates,
        }),
    )?;
    write_json_pretty(
        &artifact_dir.join("mapo_rollouts.json"),
        &json!({
            "schema_version": "mapo_rollouts.v1",
            "run_id": &config.run.run_id,
            "heldout_baseline": baseline_heldout_rollouts,
            "train": train_rollouts,
            "selection": selection_rollouts,
            "branch": branch_rollouts,
            "branch_checkpoints": branch_checkpoints,
            "heldout": heldout_rollouts,
        }),
    )?;
    let review_rows = build_mapo_review_rows(
        &config.run.run_id,
        [
            train_rollouts,
            selection_rollouts,
            branch_rollouts,
            baseline_heldout_rollouts,
            heldout_rollouts,
        ],
    );
    write_json_pretty(
        &artifact_dir.join("mapo_review_rows.json"),
        &json!({
            "schema_version": "mapo_review_rows.v1",
            "row_schema_version": "ohco.review_row.v1",
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": &config.run.run_id,
            "rows": review_rows,
        }),
    )?;
    if let Some(comparison) = heldout_comparison {
        write_json_pretty(
            &artifact_dir.join("mapo_heldout_comparison.json"),
            comparison,
        )?;
    }
    write_json_pretty(
        &artifact_dir.join("result_manifest.json"),
        &json!({
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": &config.run.run_id,
            "run_dir": run_dir.display().to_string(),
            "champion": champion,
            "candidates": candidates,
            "train_rollouts": train_rollouts,
            "selection_rollouts": selection_rollouts,
            "branch_rollouts": branch_rollouts,
            "branch_checkpoints": branch_checkpoints,
            "baseline_heldout_rollouts": baseline_heldout_rollouts,
            "heldout_rollouts": heldout_rollouts,
            "review_rows": review_rows,
            "heldout_comparison": heldout_comparison,
            "debrief_evidence": debrief_evidence,
            "campaign_manifest_receipt": campaign_manifest_receipt,
            "rollout_request_preview": rollout_request_preview,
            "dry_run": dry_run,
        }),
    )?;
    write_json_pretty(
        &run_dir.join("mapo_status.json"),
        &json!({
            "schema_version": "mapo_status.v1",
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": &config.run.run_id,
            "champion_candidate_id": &champion.id,
            "dry_run": dry_run,
            "resume_requested": resume_requested,
            "rollout_counts": {
                "train": train_rollouts.len(),
                "selection": selection_rollouts.len(),
                "branch": branch_rollouts.len(),
                "branch_checkpoints": branch_checkpoints.len(),
                "heldout_baseline": baseline_heldout_rollouts.len(),
                "heldout": heldout_rollouts.len(),
                "review_rows": review_rows.len(),
            },
            "heldout_comparison_written": heldout_comparison.is_some(),
        }),
    )?;
    Ok(())
}

fn top_train_candidates(
    mut candidates: Vec<MapoCandidate>,
    top_k: usize,
    champion_id: &str,
) -> Vec<MapoCandidate> {
    candidates.sort_by(|left, right| {
        let left_score = left.train_score.as_ref();
        let right_score = right.train_score.as_ref();
        match (left_score, right_score) {
            (Some(left_score), Some(right_score)) => {
                if mapo_score_better(left_score, right_score) {
                    std::cmp::Ordering::Less
                } else if mapo_score_better(right_score, left_score) {
                    std::cmp::Ordering::Greater
                } else {
                    left.id.cmp(&right.id)
                }
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.id.cmp(&right.id),
        }
    });

    let mut selected = Vec::new();
    for candidate in candidates.iter() {
        if candidate.id == champion_id
            && selected
                .iter()
                .all(|existing: &MapoCandidate| existing.id != candidate.id)
        {
            selected.push(candidate.clone());
            break;
        }
    }
    for candidate in candidates {
        if selected.len() >= top_k {
            break;
        }
        if selected
            .iter()
            .all(|existing: &MapoCandidate| existing.id != candidate.id)
        {
            selected.push(candidate);
        }
    }
    selected
}

fn wait_for_container_ready(
    client: &ContainerClient,
    container_url: &str,
    timeout_seconds: f64,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds.max(0.0));
    let mut backoff = Duration::from_millis(500);
    loop {
        match client.health() {
            Ok(_) => return Ok(()),
            Err(error) => {
                if timeout_seconds <= 0.0 || Instant::now() >= deadline {
                    return Err(OptimizerError::Invariant(format!(
                        "MAPO rollout container at {container_url} unreachable after \
                         {timeout_seconds:.0}s (last error: {error})"
                    )));
                }
            }
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(Duration::from_secs(5));
    }
}

fn write_json_pretty(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).map_err(|source| OptimizerError::io(path, source))
}
