use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::{CandidateRecord, RolloutScore};
use reqwest::blocking::Client;
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    jesterky_workspace_read_model, looks_like_jesterky_manifest,
    proposer_delta_chunks_from_protocol, read_jesterky_manifest, record_manifest_validation,
    run_turn, AgentTurnOutcome, CodexTurnRequest, NanoAgentTurnIdentity, NanoCodexExecution,
    NanoCodexSessionPool, NanoCodexTurnRequest, OptimizerError, PromptProgram, Result,
    SynthOptimizerConfig,
};

const GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION: &str = "gepa_reflective_frame.v1";
const CONTAINER_SENSOR_ADAPTER_ID: &str = "synth.container_sensor_frame_adapter";
const CONTAINER_SENSOR_ADAPTER_VERSION: &str = "v1";
const GEPA_ADAPTER_SOURCE: &str = "https://gepa-ai.github.io/gepa/guides/adapters/";
const GEPA_ALGORITHM_ID: &str = "synth_gepa.v1";
const GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION: &str = "gepa_workspace_proposal_v3";
const OPENROUTER_GROK43_MODEL: &str = "x-ai/grok-4.3";
const OPENROUTER_GROK43_INPUT_USD_PER_MILLION: f64 = 1.25;
const OPENROUTER_GROK43_CACHED_INPUT_USD_PER_MILLION: f64 = 0.20;
const OPENROUTER_GROK43_OUTPUT_USD_PER_MILLION: f64 = 2.50;
const DEEPSEEK_INPUT_USD_PER_MILLION: f64 = 0.27;
const DEEPSEEK_OUTPUT_USD_PER_MILLION: f64 = 1.10;
const CHAT_COMPLETIONS_PROPOSER_MAX_TOKENS: u64 = 8_192;
const CHAT_COMPLETIONS_PROPOSER_MAX_EVIDENCE_CHARS_PER_FILE: usize = 32_000;
const CHAT_COMPLETIONS_PROPOSER_MAX_TOTAL_EVIDENCE_CHARS: usize = 160_000;

static NANO_CODEX_SESSIONS: OnceLock<Mutex<NanoCodexSessionPool>> = OnceLock::new();

pub(crate) struct CodexProposerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub parent: &'a CandidateRecord,
    pub candidates: &'a [CandidateRecord],
    pub generation: usize,
    pub task_pool_rows: Value,
    pub workspace_dir: PathBuf,
}

pub(crate) struct CodexStalenessReviewerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub item: Value,
    pub stale_candidates: Vec<CandidateRecord>,
    pub current_best: Option<CandidateRecord>,
    pub pool_summary: Value,
    pub workspace_dir: PathBuf,
}

#[derive(Default)]
struct ProposerCandidateEvidenceStats {
    rollouts: usize,
    wins: usize,
    losses: usize,
    reward_sum: f64,
}

pub(crate) fn run_codex_app_server_proposer(input: CodexProposerInput<'_>) -> Result<Value> {
    materialize_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let message_stall_timeout =
        Duration::from_secs(input.config.proposer.message_stall_timeout_seconds.max(1));
    let turn_request = CodexTurnRequest {
        run_id: &input.config.run.run_id,
        proposer: &input.config.proposer,
        workspace_dir: &input.workspace_dir,
        model: &model,
        client_name: "synth-optimizers-gepa",
        client_title: "synth-optimizers GEPA",
        client_version: env!("CARGO_PKG_VERSION"),
        thread_start_params: thread_start_params(&input, &model),
        turn_start_params: turn_start_params(&input, &model)?,
        timeout,
        message_stall_timeout,
        message_observer: None,
    };
    if input.config.proposer.nano_codex.enabled {
        let execution = run_nano_codex_proposer(&input, turn_request)?;
        return build_response_from_outcome(
            &input,
            &model,
            execution.outcome,
            Some(execution.receipt_path),
        );
    }
    let outcome = run_turn(turn_request)?;
    build_response_from_outcome(&input, &model, outcome, None)
}

pub(crate) fn run_codex_staleness_reviewer(
    input: CodexStalenessReviewerInput<'_>,
) -> Result<Value> {
    if input.config.proposer.backend != "codex_app_server" {
        return Err(OptimizerError::Config(
            "gepa.pipeline.staleness_policy = reflective requires proposer.backend = \"codex_app_server\" for the staleness reviewer"
                .to_string(),
        ));
    }
    if input.config.proposer.nano_codex.enabled {
        return Err(OptimizerError::Config(
            "nano-Codex does not yet support the reflective staleness reviewer; refusing to fall back to a one-shot proposer process".to_string(),
        ));
    }
    materialize_staleness_review_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let message_stall_timeout =
        Duration::from_secs(input.config.proposer.message_stall_timeout_seconds.max(1));
    let outcome = run_turn(CodexTurnRequest {
        run_id: &input.config.run.run_id,
        proposer: &input.config.proposer,
        workspace_dir: &input.workspace_dir,
        model: &model,
        client_name: "synth-optimizers-gepa",
        client_title: "synth-optimizers GEPA Staleness Reviewer",
        client_version: env!("CARGO_PKG_VERSION"),
        thread_start_params: staleness_thread_start_params(&input, &model),
        turn_start_params: staleness_turn_start_params(&input, &model),
        timeout,
        message_stall_timeout,
        message_observer: None,
    })?;
    build_staleness_review_response(&input, &model, outcome)
}

/// Direct OpenAI-compatible Chat Completions proposer. Works for any provider whose
/// `/chat/completions` endpoint matches the OpenAI shape.
/// This is the path NVIDIA must use: the codex_app_server route speaks the Responses
/// wire, which `integrate.api.nvidia.com` does not serve.
pub(crate) fn run_deepseek_chat_proposer(input: CodexProposerInput<'_>) -> Result<Value> {
    let provider = input.config.proposer.provider.trim().to_ascii_lowercase();
    // (default base_url, default api_key_env, default model, send DeepSeek `thinking` field)
    let (default_base_url, default_api_key_env, default_model, deepseek_thinking) =
        match provider.as_str() {
            "deepseek" => (
                "https://api.deepseek.com",
                "DEEPSEEK_API_KEY",
                "deepseek-v4-flash",
                true,
            ),
            "nvidia" => (
                "https://integrate.api.nvidia.com/v1",
                "NVIDIA_API_KEY",
                "nvidia/nemotron-3-ultra-550b-a55b",
                false,
            ),
            "openai" => (
                "https://api.openai.com/v1",
                "OPENAI_API_KEY",
                "gpt-4.1-mini",
                false,
            ),
            other => {
                return Err(OptimizerError::Config(format!(
                    "chat-completions proposer backend requires proposer.provider = \"deepseek\", \"nvidia\", or \"openai\"; got {other:?}"
                )))
            }
        };
    materialize_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());
    let api_key_env =
        non_empty(input.config.proposer.api_key_env.as_deref()).unwrap_or(default_api_key_env);
    let api_key = env::var(api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Proposer(format!(
                "chat-completions proposer ({provider}) requires non-empty {api_key_env}"
            ))
        })?;
    let base_url = input
        .config
        .proposer
        .base_url
        .as_deref()
        .unwrap_or(default_base_url)
        .trim_end_matches('/')
        .to_string();
    let mut request = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are the GEPA workspace proposer. Return only a JSON object that matches the requested manifest schema."
            },
            {
                "role": "user",
                "content": deepseek_chat_prompt(&input)?
            }
        ],
        "response_format": {"type": "json_object"},
        // Keep OpenRouter-compatible requests inside common 128k context windows.
        // Two GEPA proposals fit comfortably in 8k while 32k can make the
        // provider reject otherwise-valid evidence packets before generation.
        "max_tokens": CHAT_COMPLETIONS_PROPOSER_MAX_TOKENS,
        "stream": false
    });
    // DeepSeek-specific switch that suppresses its reasoning channel so `content` is the
    // bare JSON manifest. NVIDIA rejects unknown request fields, so only send it for DeepSeek.
    if deepseek_thinking {
        request["thinking"] = json!({"type": "disabled"});
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(
            input.config.proposer.timeout_seconds.max(1),
        ))
        .build()?;
    // Resilience: retry the whole request when the response can't be decoded,
    // the content is missing, or the model's manifest JSON is malformed/truncated
    // (transient DeepSeek errors). HTTP 429/5xx are already retried inside
    // post_deepseek_chat_completion; this outer loop also covers decode/parse.
    let (chat_response, manifest) = {
        let mut last_err: Option<String> = None;
        let mut result: Option<(Value, Value)> = None;
        for attempt in 1..=3usize {
            match post_deepseek_chat_completion(&client, &base_url, &api_key, &request, 4) {
                Ok(resp) => {
                    match resp
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                    {
                        Some(content) => match serde_json::from_str::<Value>(content.trim()) {
                            Ok(m) => {
                                result = Some((resp, m));
                                break;
                            }
                            Err(e) => last_err = Some(format!("manifest JSON parse failed: {e}")),
                        },
                        None => {
                            last_err =
                                Some("response missing choices[0].message.content".to_string())
                        }
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
            if attempt < 3 {
                std::thread::sleep(Duration::from_secs(2 * attempt as u64));
            }
        }
        match result {
            Some(pair) => pair,
            None => {
                return Err(OptimizerError::Proposer(format!(
                    "chat-completions proposer failed after 3 attempts: {}",
                    last_err.unwrap_or_else(|| "unknown error".to_string())
                )))
            }
        }
    };
    let manifest_path = input.workspace_dir.join("proposal").join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let evidence_warnings = manifest_evidence_warnings(&input, &manifest, &proposals);
    let usage = chat_response.get("usage").cloned().ok_or_else(|| {
        OptimizerError::Proposer("chat-completions proposer response missing usage".to_string())
    })?;
    let usage = normalize_proposer_usage(input.config, &model, usage);
    write_deepseek_chat_artifacts(&input, &request, &chat_response)?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(json!({
        "backend": input.config.proposer.backend,
        "runtime_substrate": "local",
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
        "evidence_warnings": evidence_warnings,
        "proposer_stream_chunks": chat_content_stream_chunks(&chat_response),
    }))
}

fn deepseek_chat_prompt(input: &CodexProposerInput<'_>) -> Result<String> {
    let best_practices = resolved_prompting_best_practices(input)?;
    let mut prompt = String::new();
    prompt.push_str(&proposer_prompt_context(input));
    prompt.push_str("\n\nPrompting best practices:\n");
    prompt.push_str(best_practices.trim());
    prompt.push_str("\n\nManifest schema:\n");
    prompt.push_str(&proposal_schema(input));
    prompt.push_str("\n\nUse these workspace files as the complete evidence packet.\n");
    let mut remaining_evidence_chars = CHAT_COMPLETIONS_PROPOSER_MAX_TOTAL_EVIDENCE_CHARS;
    for path in [
        "state/proposer_metadata.json",
        "state/task_info.json",
        "state/program_contract.json",
        "state/parent_payload.json",
        "state/proposer_failure_summary.json",
        "state/proposer_repair_hints.json",
        "state/proposer_examples.json",
        "state/reflective_frames.json",
        "state/proposal_request.json",
        "state/reflector_input.json",
    ] {
        let file_path = input.workspace_dir.join(path);
        let text = fs::read_to_string(&file_path)
            .map_err(|source| OptimizerError::io(&file_path, source))?;
        let file_budget =
            remaining_evidence_chars.min(CHAT_COMPLETIONS_PROPOSER_MAX_EVIDENCE_CHARS_PER_FILE);
        let text_char_count = text.chars().count();
        let text = if file_budget == 0 {
            "[omitted: chat-completions proposer evidence budget exhausted]".to_string()
        } else if text_char_count > file_budget {
            let head: String = text.chars().take(file_budget).collect();
            remaining_evidence_chars = remaining_evidence_chars.saturating_sub(file_budget);
            format!("{head}\n…[truncated to {file_budget} chars to fit context budget]…")
        } else {
            remaining_evidence_chars = remaining_evidence_chars.saturating_sub(text_char_count);
            text
        };
        prompt.push_str("\n\n--- ");
        prompt.push_str(path);
        prompt.push_str(" ---\n");
        prompt.push_str(&text);
    }
    prompt.push_str(
        "\n\nReturn strict JSON only. Do not wrap it in Markdown. The JSON object must have \
         schema_version, critique, evidence, rationale, and proposals. Each proposed_payload \
         must contain full replacement text for every target module.",
    );
    Ok(prompt)
}

fn post_deepseek_chat_completion(
    client: &Client,
    base_url: &str,
    api_key: &str,
    request: &Value,
    max_attempts: usize,
) -> Result<Value> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    for attempt in 1..=max_attempts {
        let response = client
            .post(&url)
            .bearer_auth(api_key)
            .json(request)
            .send()?;
        let status = response.status();
        let text = response.text()?;
        if status.is_success() {
            return Ok(serde_json::from_str(&text)?);
        }
        if attempt < max_attempts && matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) {
            let delay = match attempt {
                1 => Duration::from_secs(2),
                2 => Duration::from_secs(5),
                _ => Duration::from_secs(10),
            };
            std::thread::sleep(delay);
            continue;
        }
        return Err(OptimizerError::Proposer(format!(
            "DeepSeek chat proposer failed after {attempt}/{max_attempts} attempts with status \
             {}: {}",
            status,
            text.chars().take(1000).collect::<String>()
        )));
    }
    Err(OptimizerError::Proposer(
        "DeepSeek chat proposer retry loop exited unexpectedly".to_string(),
    ))
}

fn write_deepseek_chat_artifacts(
    input: &CodexProposerInput<'_>,
    request: &Value,
    response: &Value,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("deepseek_chat_request.json"),
        &json!({
            "model": request.get("model"),
            "message_count": request.get("messages").and_then(Value::as_array).map(Vec::len),
            "max_tokens": request.get("max_tokens"),
            "response_format": request.get("response_format"),
            "thinking": request.get("thinking"),
        }),
    )?;
    write_json(&artifact_dir.join("deepseek_chat_response.json"), response)
}

fn proposer_stream_chunks_from_messages(messages: &[Value]) -> Vec<Value> {
    proposer_delta_chunks_from_protocol(messages)
        .into_iter()
        .map(|(channel, text)| json!({"channel": channel, "text": text}))
        .collect()
}

fn chat_content_stream_chunks(chat_response: &Value) -> Vec<Value> {
    chat_response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| vec![json!({"channel": "content", "text": text})])
        .unwrap_or_default()
}

fn build_response_from_outcome(
    input: &CodexProposerInput<'_>,
    model: &str,
    outcome: AgentTurnOutcome,
    nano_receipt_path: Option<PathBuf>,
) -> Result<Value> {
    let usage = normalize_proposer_usage(
        input.config,
        model,
        outcome.usage.clone().unwrap_or_else(
            || json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        ),
    );
    let mut prevalidation_response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "usage": usage.clone(),
        "manifest_validation": "pending",
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        prevalidation_response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        prevalidation_response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    write_agent_artifacts(
        input,
        model,
        &outcome.thread_id,
        &outcome.turn_id,
        &outcome.thread_response,
        &outcome.final_turn,
        &prevalidation_response,
        &outcome,
    )?;
    let validation_started = std::time::Instant::now();
    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let manifest_validation_ms = validation_started.elapsed().as_millis();
    let mut evidence_warnings = manifest_evidence_warnings(input, &manifest, &proposals);
    if outcome.usage.is_none() {
        evidence_warnings.push(
            "proposer usage missing from codex turn payload; token counts may be incomplete"
                .to_string(),
        );
    }
    let mut response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
        "evidence_warnings": evidence_warnings,
        "received_messages": outcome.received_messages,
        "proposer_stream_chunks": proposer_stream_chunks_from_messages(&outcome.received_messages),
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    if let Some(receipt_path) = nano_receipt_path {
        let receipt = record_manifest_validation(&receipt_path, manifest_validation_ms)?;
        response["nano_codex"] = json!({
            "receipt_path": receipt_path,
            "receipt": receipt,
        });
    }
    write_agent_artifacts(
        input,
        model,
        &outcome.thread_id,
        &outcome.turn_id,
        &outcome.thread_response,
        &outcome.final_turn,
        &response,
        &outcome,
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(response)
}

fn run_nano_codex_proposer(
    input: &CodexProposerInput<'_>,
    turn: CodexTurnRequest<'_>,
) -> Result<NanoCodexExecution> {
    let task_info_path = input.workspace_dir.join("state").join("task_info.json");
    let task_info = read_json_value(&task_info_path)?;
    let artifact_dir = input.workspace_dir.join(".nano_codex");
    let identity = NanoAgentTurnIdentity {
        request_id: format!(
            "{}-generation-{:03}",
            input.config.run.run_id, input.generation
        ),
        run_id: input.config.run.run_id.clone(),
        role: "gepa_proposer".to_string(),
        round: input.generation.to_string(),
        treatment_preset: "prompt".to_string(),
        parent_candidate_id: input.parent.candidate_id.clone(),
        workspace_id: format!("generation_{:03}", input.generation),
    };
    let request = NanoCodexTurnRequest {
        turn,
        identity,
        static_context: json!({
            "schema_version": "gepa.nano_codex.static_context.v1",
            "task_info": task_info,
            "program": input.program,
            "target_modules": input.config.candidate.target_modules,
            "proposer_prompt": input.config.proposer.prompt,
        }),
        replay_artifact_paths: vec![PathBuf::from("proposal/manifest.json")],
        artifact_dir: &artifact_dir,
        cancel_before_start: false,
    };
    let pool = NANO_CODEX_SESSIONS.get_or_init(|| Mutex::new(NanoCodexSessionPool::default()));
    pool.lock()
        .map_err(|_| {
            OptimizerError::Invariant("nano-Codex session pool mutex poisoned".to_string())
        })?
        .run(request)
}

fn normalize_proposer_usage(config: &SynthOptimizerConfig, model: &str, usage: Value) -> Value {
    let Some(mut usage_map) = usage.as_object().cloned() else {
        return usage;
    };
    let provider = config.proposer.provider.trim().to_ascii_lowercase();
    let model_lower = model.trim().to_ascii_lowercase();
    let reported_cost = usage_f64_from_map(&usage_map, "cost_usd")
        .filter(|value| value.is_finite() && *value > 0.0);
    if provider.eq_ignore_ascii_case("openrouter") && model_lower == OPENROUTER_GROK43_MODEL {
        return normalize_openrouter_grok43_usage(model, usage_map);
    }
    if provider.eq_ignore_ascii_case("deepseek") || model_lower.contains("deepseek") {
        usage_map.insert(
            "provider".to_string(),
            Value::String("deepseek".to_string()),
        );
        usage_map.insert("model".to_string(), Value::String(model.to_string()));
        if reported_cost.is_none() {
            let prompt_tokens = usage_u64_from_any(&usage_map, &["prompt_tokens", "input_tokens"]);
            let completion_tokens =
                usage_u64_from_any(&usage_map, &["completion_tokens", "output_tokens"]);
            if prompt_tokens == 0 && completion_tokens == 0 {
                return Value::Object(usage_map);
            }
            let cost_usd = prompt_tokens as f64 * DEEPSEEK_INPUT_USD_PER_MILLION / 1_000_000.0
                + completion_tokens as f64 * DEEPSEEK_OUTPUT_USD_PER_MILLION / 1_000_000.0;
            usage_map.insert("cost_usd".to_string(), json!(cost_usd));
            usage_map.insert(
                "cost_source".to_string(),
                Value::String("deepseek_static_price".to_string()),
            );
            usage_map.insert(
                "cost_pricing".to_string(),
                json!({
                    "input_usd_per_million": DEEPSEEK_INPUT_USD_PER_MILLION,
                    "output_usd_per_million": DEEPSEEK_OUTPUT_USD_PER_MILLION,
                }),
            );
        }
        return Value::Object(usage_map);
    }
    Value::Object(usage_map)
}

fn normalize_openrouter_grok43_usage(model: &str, mut usage_map: Map<String, Value>) -> Value {
    usage_map.insert(
        "provider".to_string(),
        Value::String("openrouter".to_string()),
    );
    usage_map.insert("model".to_string(), Value::String(model.to_string()));
    if usage_f64_from_map(&usage_map, "cost_usd")
        .filter(|value| value.is_finite() && *value > 0.0)
        .is_none()
    {
        let prompt_tokens = usage_u64_from_any(&usage_map, &["prompt_tokens", "input_tokens"]);
        let completion_tokens =
            usage_u64_from_any(&usage_map, &["completion_tokens", "output_tokens"]);
        let cached_prompt_tokens =
            usage_u64_from_map(&usage_map, "cached_prompt_tokens").min(prompt_tokens);
        let billable_prompt_tokens = prompt_tokens.saturating_sub(cached_prompt_tokens);
        let cost_usd = billable_prompt_tokens as f64 * OPENROUTER_GROK43_INPUT_USD_PER_MILLION
            / 1_000_000.0
            + cached_prompt_tokens as f64 * OPENROUTER_GROK43_CACHED_INPUT_USD_PER_MILLION
                / 1_000_000.0
            + completion_tokens as f64 * OPENROUTER_GROK43_OUTPUT_USD_PER_MILLION / 1_000_000.0;
        usage_map.insert("cost_usd".to_string(), json!(cost_usd));
        usage_map.insert(
            "cost_source".to_string(),
            Value::String("openrouter_xai_grok43_static_price".to_string()),
        );
        usage_map.insert(
            "cost_pricing".to_string(),
            json!({
                "input_usd_per_million": OPENROUTER_GROK43_INPUT_USD_PER_MILLION,
                "cached_input_usd_per_million": OPENROUTER_GROK43_CACHED_INPUT_USD_PER_MILLION,
                "output_usd_per_million": OPENROUTER_GROK43_OUTPUT_USD_PER_MILLION,
            }),
        );
        if usage_u64_from_map(&usage_map, "total_tokens") > 200_000 {
            usage_map.insert(
                "cost_warning".to_string(),
                Value::String(
                    "OpenRouter Grok 4.3 uses tiered pricing above 200k total tokens; static \
                     estimate uses base-tier pricing"
                        .to_string(),
                ),
            );
        }
    }
    Value::Object(usage_map)
}

fn usage_u64_from_any(map: &Map<String, Value>, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| {
            let value = map.get(*key)?;
            value
                .as_u64()
                .or_else(|| {
                    value
                        .as_f64()
                        .filter(|v| v.is_finite() && *v >= 0.0)
                        .map(|v| v as u64)
                })
                .or_else(|| {
                    let text = value.as_str()?;
                    text.parse::<u64>().ok().or_else(|| {
                        text.parse::<f64>()
                            .ok()
                            .filter(|v| v.is_finite() && *v >= 0.0)
                            .map(|v| v as u64)
                    })
                })
        })
        .unwrap_or(0)
}

fn usage_u64_from_map(map: &Map<String, Value>, key: &str) -> u64 {
    usage_u64_from_any(map, &[key])
}

fn usage_f64_from_map(map: &Map<String, Value>, key: &str) -> Option<f64> {
    map.get(key)
        .and_then(|value| value.as_f64().or_else(|| value.as_str()?.parse().ok()))
}

fn build_staleness_review_response(
    input: &CodexStalenessReviewerInput<'_>,
    model: &str,
    outcome: AgentTurnOutcome,
) -> Result<Value> {
    let usage = normalize_proposer_usage(
        input.config,
        model,
        outcome.usage.clone().unwrap_or_else(
            || json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        ),
    );
    let verdict_path = input.workspace_dir.join("review").join("verdict.json");
    let verdict = read_staleness_verdict_json(&verdict_path)?;
    validate_staleness_verdict(&verdict)?;
    write_json(&verdict_path, &verdict)?;
    let mut response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "model": model,
        "usage": usage,
        "verdict": verdict,
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    write_staleness_agent_artifacts(input, model, &outcome, &response)?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(response)
}

fn validate_staleness_verdict(verdict: &Value) -> Result<()> {
    let verdict_text = verdict
        .get("verdict")
        .or_else(|| verdict.get("decision"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        verdict_text.as_str(),
        "accept" | "accept_as_is" | "discard" | "drop" | "patch" | "repair"
    ) {
        return Err(OptimizerError::Proposer(format!(
            "staleness reviewer verdict must be accept, discard, or patch; got {verdict_text:?}"
        )));
    }
    if matches!(verdict_text.as_str(), "patch" | "repair")
        && !verdict
            .get("patched_payload")
            .or_else(|| verdict.get("payload"))
            .is_some_and(Value::is_object)
    {
        return Err(OptimizerError::Proposer(
            "staleness reviewer patch verdict must include patched_payload".to_string(),
        ));
    }
    Ok(())
}

fn write_staleness_agent_artifacts(
    input: &CodexStalenessReviewerInput<'_>,
    model: &str,
    outcome: &AgentTurnOutcome,
    response: &Value,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("opencode_session.json"),
        &json!({
            "schema_version": "gepa_codex_staleness_review_session.v1",
            "backend": "codex_app_server",
            "model": model,
            "thread_id": outcome.thread_id,
            "turn_id": outcome.turn_id,
            "workspace": input.workspace_dir,
            "supervisor_receipt": outcome.supervisor_receipt,
            "thread_response": outcome.thread_response,
            "final_turn": outcome.final_turn,
        }),
    )?;
    write_json(
        &artifact_dir.join("opencode_messages.json"),
        &json!({
            "schema_version": "gepa_codex_staleness_review_messages.v1",
            "sent": outcome.sent_messages,
            "received": outcome.received_messages,
        }),
    )?;
    write_json(&artifact_dir.join("opencode_response.json"), response)?;
    let mut events = String::new();
    for message in &outcome.received_messages {
        events.push_str(&serde_json::to_string(message)?);
        events.push('\n');
    }
    write_text(&artifact_dir.join("opencode_sse_events.jsonl"), &events)?;
    Ok(())
}

fn materialize_staleness_review_workspace(input: &CodexStalenessReviewerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let review_dir = input.workspace_dir.join("review");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&review_dir).map_err(|source| OptimizerError::io(&review_dir, source))?;
    write_text(
        &input.workspace_dir.join("README.md"),
        &staleness_review_readme(input),
    )?;
    write_text(
        &review_dir.join("VERDICT_SCHEMA.md"),
        &staleness_verdict_schema(input),
    )?;
    write_json(
        &review_dir.join("verdict.json"),
        &json!({
            "schema_version": "gepa_staleness_review_v1",
            "verdict": "accept",
            "reason": "",
            "patched_payload": null,
        }),
    )?;
    write_json(
        &state_dir.join("staleness_review_request.json"),
        &json!({
            "schema_version": "gepa_staleness_review_request_v1",
            "run_id": input.config.run.run_id,
            "target_modules": input.config.candidate.target_modules,
            "item": input.item,
            "stale_candidates": input.stale_candidates,
            "current_best": input.current_best,
            "pool_summary": input.pool_summary,
            "program": input.program,
        }),
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(())
}

fn staleness_review_readme(input: &CodexStalenessReviewerInput<'_>) -> String {
    format!(
        r#"# GEPA Staleness Reviewer Workspace

You are reviewing stale FlashEvolve work before it is folded back into the current artifact pool.

Read `state/staleness_review_request.json` and `review/VERDICT_SCHEMA.md`.
Choose exactly one verdict:

- `accept`: the stale candidate is still compatible with the current pool.
- `discard`: the stale candidate is obsolete or conflicts with newer accepted pool changes.
- `patch`: provide a full replacement `patched_payload` for the target modules.

If patching, preserve every target module key and write a complete payload object. Target modules: {}.
Write strict JSON to `review/verdict.json`.
"#,
        input.config.candidate.target_modules.join(", ")
    )
}

fn staleness_verdict_schema(input: &CodexStalenessReviewerInput<'_>) -> String {
    let payload_hint = if input.config.candidate.target_modules.len() > 1 {
        "When verdict is patch, patched_payload must include every target module key."
    } else {
        "When verdict is patch, patched_payload must include the single target module key."
    };
    format!(
        r#"# GEPA Staleness Review Verdict Schema

Write `review/verdict.json` as strict JSON:

```json
{{
  "schema_version": "gepa_staleness_review_v1",
  "verdict": "accept | discard | patch",
  "reason": "Short explanation grounded in state/staleness_review_request.json.",
  "patched_payload": {{
    "<target_module>": "<full replacement instruction>"
  }}
}}
```

Rules:

- Use `accept` when the stale candidate's prompt remains compatible with the current pool.
- Use `discard` when the candidate duplicates, conflicts with, or is dominated by current pool changes.
- Use `patch` when a small language-space repair can rebase the stale candidate onto the current pool.
- {payload_hint}
- If verdict is not `patch`, set `patched_payload` to null.
"#
    )
}

fn materialize_workspace(input: &CodexProposerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let proposal_dir = input.workspace_dir.join("proposal");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&proposal_dir)
        .map_err(|source| OptimizerError::io(&proposal_dir, source))?;

    write_text(
        &input.workspace_dir.join("README.md"),
        &workspace_readme(input),
    )?;
    let prompting_best_practices = resolved_prompting_best_practices(input)?;
    write_text(
        &input.workspace_dir.join("prompting_best_practices.md"),
        &prompting_best_practices,
    )?;
    write_text(
        &proposal_dir.join("PROPOSAL_SCHEMA.md"),
        &proposal_schema(input),
    )?;
    write_json(
        &proposal_dir.join("manifest.json"),
        &json!({
            "schema_version": GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
            "critique": "",
            "evidence": {
                "reviewed_files": [],
                "candidate_comparison": "",
                "failure_patterns": [],
                "winning_patterns": [],
                "example_ids_used": [],
            },
            "rationale": "",
            "proposals": [],
        }),
    )?;
    let parent_payload = json!(&input.parent.payload);
    let proposal_request = proposal_request(input, &prompting_best_practices);
    let rollouts = rollouts_read_model(input);
    // Annotate after rollouts are available so jesterky_* land in state/ before
    // the presence-gated read-model path and the live proposer turn.
    let _jesterky_receipt = crate::jesterky_workflow::prepare_jesterky_workflow_for_generation(
        input.config,
        &rollouts,
        &input.workspace_dir,
        input.generation,
    )?;
    let proposer_examples = proposer_examples_read_model(input);
    let proposer_failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    let proposer_repair_hints = proposer_repair_hints_read_model(input, &proposer_examples);
    let proposer_metadata = sanitize_proposer_workspace_value(proposer_metadata_read_model(
        input,
        &rollouts,
        &proposer_examples,
        &proposer_failure_summary,
    ));
    let proposer_readme = proposer_readme_read_model();
    let reflective_frames = reflective_frames_read_model(input);
    let proposer_task_info =
        sanitize_proposer_workspace_value(task_info_value(input).cloned().unwrap_or(Value::Null));
    let proposer_program = sanitize_proposer_workspace_value(serde_json::to_value(input.program)?);
    let jesterky_read_model = jesterky_workspace_read_model_for_proposer(input)?;
    write_json(
        &state_dir.join("run_context.json"),
        &json!({
            "run_id": input.config.run.run_id,
            "generation": input.generation,
            "task": "GEPA prompt proposal",
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "proposals_per_generation": input.config.gepa.proposals_per_generation,
            "proposals_per_round": input.config.gepa.proposals_per_generation,
            "parent_candidate_id": input.parent.candidate_id,
            "jesterky_workflow_enabled": input.config.jesterky_workflow.enabled,
        }),
    )?;
    write_json(&state_dir.join("task_info.json"), &proposer_task_info)?;
    write_json(
        &state_dir.join("program_contract.json"),
        &json!({
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "mutable_fields": input.program.mutable_field_ids(),
            "program": proposer_program.clone(),
        }),
    )?;
    write_json(&state_dir.join("program.json"), &proposer_program)?;
    write_json(
        &state_dir.join("parent_candidate.json"),
        &parent_candidate_read_model(input),
    )?;
    write_json(&state_dir.join("parent_payload.json"), &parent_payload)?;
    write_json(
        &state_dir.join("proposer_examples.json"),
        &proposer_examples,
    )?;
    write_json(
        &state_dir.join("proposer_failure_summary.json"),
        &proposer_failure_summary,
    )?;
    write_json(
        &state_dir.join("proposer_repair_hints.json"),
        &proposer_repair_hints,
    )?;
    write_json(
        &state_dir.join("proposer_metadata.json"),
        &proposer_metadata,
    )?;
    write_json(&state_dir.join("proposer_readme.json"), &proposer_readme)?;
    write_json(
        &state_dir.join("reflective_frames.json"),
        &reflective_frames,
    )?;
    write_json(&state_dir.join("proposal_request.json"), &proposal_request)?;
    write_json(
        &state_dir.join("reflector_input.json"),
        &reflector_input_read_model(input, &prompting_best_practices),
    )?;
    if let Some(read_model) = jesterky_read_model {
        write_jesterky_workspace_read_model(&state_dir, read_model)?;
    }
    write_workspace_pack_manifest(&input.workspace_dir)?;
    assert_proposer_workspace_no_leaks(&input.workspace_dir)?;
    Ok(())
}

fn write_jesterky_workspace_read_model(state_dir: &Path, read_model: Value) -> Result<()> {
    let read_model = sanitize_proposer_workspace_value(read_model);
    write_json(&state_dir.join("jesterky_read_model.json"), &read_model)?;
    for (name, path) in [
        ("summary", "jesterky_manifest_summary.json"),
        ("trace_rows", "jesterky_trace_rows.json"),
        ("optimizer_triples", "jesterky_optimizer_triples.json"),
        ("evidence_refs", "jesterky_evidence_refs.json"),
    ] {
        if let Some(value) = read_model.get(name) {
            write_json(&state_dir.join(path), value)?;
        }
    }
    Ok(())
}

fn jesterky_workspace_read_model_for_proposer(
    input: &CodexProposerInput<'_>,
) -> Result<Option<Value>> {
    let Some(manifest) = find_jesterky_manifest_for_proposer(input)? else {
        return Ok(None);
    };
    jesterky_workspace_read_model(&manifest).map(Some)
}

fn find_jesterky_manifest_for_proposer(input: &CodexProposerInput<'_>) -> Result<Option<Value>> {
    // Prefer the annotate manifest written by jesterky_workflow this generation.
    let annotate_path = input
        .workspace_dir
        .join(crate::jesterky_workflow::JESTERKY_ANNOTATE_MANIFEST_FILE);
    if annotate_path.is_file() {
        return read_jesterky_manifest(&annotate_path).map(Some);
    }
    let program_metadata = Value::Object(input.program.metadata.clone());
    if let Some(manifest) = find_jesterky_manifest_in_value(&program_metadata, &input.workspace_dir)?
    {
        return Ok(Some(manifest));
    }
    for candidate in input.candidates {
        let acceptance_metadata = Value::Object(candidate.acceptance_metadata.clone());
        if let Some(manifest) =
            find_jesterky_manifest_in_value(&acceptance_metadata, &input.workspace_dir)?
        {
            return Ok(Some(manifest));
        }
        for frame in &candidate.sensor_frames {
            let frame_metadata = Value::Object(frame.metadata.clone());
            if let Some(manifest) =
                find_jesterky_manifest_in_value(&frame_metadata, &input.workspace_dir)?
            {
                return Ok(Some(manifest));
            }
            if let Some(side_info) = frame.actionable_side_info.as_ref() {
                if let Some(manifest) =
                    find_jesterky_manifest_in_value(side_info, &input.workspace_dir)?
                {
                    return Ok(Some(manifest));
                }
            }
        }
    }
    Ok(None)
}

fn find_jesterky_manifest_in_value(value: &Value, workspace_dir: &Path) -> Result<Option<Value>> {
    if looks_like_jesterky_manifest(value) {
        return Ok(Some(value.clone()));
    }
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let lower = key.to_ascii_lowercase();
                if lower.contains("jesterky")
                    && lower.contains("manifest")
                    && child.as_str().is_some()
                {
                    return read_jesterky_manifest_from_string(child, workspace_dir).map(Some);
                }
                if lower.contains("jesterky") && lower.contains("manifest") {
                    if let Some(manifest) = find_jesterky_manifest_in_value(child, workspace_dir)? {
                        return Ok(Some(manifest));
                    }
                } else if let Some(manifest) = find_jesterky_manifest_in_value(child, workspace_dir)?
                {
                    return Ok(Some(manifest));
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                if let Some(manifest) = find_jesterky_manifest_in_value(child, workspace_dir)? {
                    return Ok(Some(manifest));
                }
            }
        }
        _ => {}
    }
    Ok(None)
}

fn read_jesterky_manifest_from_string(value: &Value, workspace_dir: &Path) -> Result<Value> {
    let path = value.as_str().unwrap_or_default().trim();
    if path.is_empty() {
        return Err(OptimizerError::Config(
            "jesterky manifest path must not be empty".to_string(),
        ));
    }
    let path = resolve_jesterky_manifest_path(path, workspace_dir);
    read_jesterky_manifest(&path)
}

fn resolve_jesterky_manifest_path(path: &str, workspace_dir: &Path) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if let Some(run_dir) = workspace_dir.parent().and_then(Path::parent) {
        let run_relative = run_dir.join(&candidate);
        if run_relative.exists() {
            return run_relative;
        }
    }
    let workspace_relative = workspace_dir.join(&candidate);
    if workspace_relative.exists() {
        return workspace_relative;
    }
    candidate
}

fn resolved_prompting_best_practices(input: &CodexProposerInput<'_>) -> Result<String> {
    let prompt = &input.config.proposer.prompt;
    if let Some(text) = prompt.best_practices.as_deref() {
        return Ok(text.to_string());
    }
    if let Some(path) = &prompt.best_practices_path {
        return fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source));
    }
    Ok(crate::default_proposer_best_practices().to_string())
}

fn write_workspace_pack_manifest(workspace_dir: &Path) -> Result<()> {
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    let mut files = Vec::new();
    collect_workspace_files(workspace_dir, workspace_dir, &mut files)?;
    files.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .cmp(&right.get("path").and_then(Value::as_str))
    });
    write_json(
        &state_dir.join("workspace_pack_manifest.json"),
        &json!({
            "schema_version": "gepa_workspace_pack_manifest.v1",
            "file_count": files.len(),
            "files": files,
        }),
    )
}

fn collect_workspace_files(root: &Path, current: &Path, files: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| OptimizerError::io(current, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(current, source))?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip_workspace_manifest_path(relative) {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.is_dir() {
            collect_workspace_files(root, &path, files)?;
        } else if metadata.is_file() {
            files.push(json!({
                "path": relative.to_string_lossy(),
                "bytes": metadata.len(),
            }));
        }
    }
    Ok(())
}

fn should_skip_workspace_manifest_path(path: &Path) -> bool {
    path.components().any(|component| {
        let text = component.as_os_str().to_string_lossy();
        matches!(text.as_ref(), ".codex_home" | ".codex_api_key_home")
    })
}

fn proposer_workspace_blocked_terms() -> &'static [&'static str] {
    &[
        "heldout",
        "heldout_",
        "val_score",
        "pareto_eval",
        "pareto_front",
        "win_counts",
        "candidate_selector",
        "algorithm_read_model",
        "gepa_sidecar",
        "frontier_cells",
    ]
}

fn sanitize_proposer_workspace_value(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sanitized = map
                .into_iter()
                .filter_map(|(key, value)| {
                    let lower_key = key.to_ascii_lowercase();
                    if proposer_workspace_blocked_terms()
                        .iter()
                        .any(|term| lower_key.contains(term))
                    {
                        return None;
                    }
                    Some((key, sanitize_proposer_workspace_value(value)))
                })
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(sanitize_proposer_workspace_value)
                .filter(|value| !value.is_null())
                .collect(),
        ),
        Value::String(text) => {
            let lower = text.to_ascii_lowercase();
            if proposer_workspace_blocked_terms()
                .iter()
                .any(|term| lower.contains(term))
            {
                Value::Null
            } else {
                Value::String(text)
            }
        }
        other => other,
    }
}

fn assert_proposer_workspace_no_leaks(workspace_dir: &Path) -> Result<()> {
    let blocked_files = [
        "state/scores.json",
        "state/evidence_frames.json",
        "state/links.json",
        "state/task_pools.json",
        "state/algorithm_read_model.json",
        "state/pareto_front.json",
        "state/gepa_sidecar.json",
        "state/gepa_summary.json",
    ];
    for relative in blocked_files {
        let path = workspace_dir.join(relative);
        if path.exists() {
            return Err(OptimizerError::Invariant(format!(
                "proposer workspace contains blocked file {}",
                relative
            )));
        }
    }

    let mut files = Vec::new();
    collect_workspace_files(workspace_dir, workspace_dir, &mut files)?;
    for file in files {
        let Some(relative) = file.get("path").and_then(Value::as_str) else {
            continue;
        };
        // Annotate sidecars are wall-safe theme summaries; skip term scan so
        // theme text cannot trip the heldout/frontier leak detector.
        if relative.contains("jesterky_") {
            continue;
        }
        let path = workspace_dir.join(relative);
        let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
        let lower = text.to_ascii_lowercase();
        for term in proposer_workspace_blocked_terms() {
            if lower.contains(term) {
                return Err(OptimizerError::Invariant(format!(
                    "proposer workspace file {} contains blocked term {:?}",
                    relative, term
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_agent_artifacts(
    input: &CodexProposerInput<'_>,
    model: &str,
    thread_id: &str,
    turn_id: &str,
    thread_response: &Value,
    final_turn: &Value,
    response: &Value,
    outcome: &AgentTurnOutcome,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("opencode_session.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_session.v1",
            "backend": "codex_app_server",
            "model": model,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "workspace": input.workspace_dir,
            "sandbox_mode": input.config.proposer.sandbox_mode,
            "approval_policy": input.config.proposer.approval_policy,
            "auth_mode": input.config.proposer.auth_mode,
            "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
            "supervisor_receipt": &outcome.supervisor_receipt,
            "thread_response": thread_response,
            "final_turn": final_turn,
        }),
    )?;
    write_json(
        &artifact_dir.join("opencode_messages.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_messages.v1",
            "sent": &outcome.sent_messages,
            "received": &outcome.received_messages,
        }),
    )?;
    write_json(&artifact_dir.join("opencode_response.json"), response)?;
    let mut events = String::new();
    for message in &outcome.received_messages {
        events.push_str(&serde_json::to_string(message)?);
        events.push('\n');
    }
    write_text(&artifact_dir.join("opencode_sse_events.jsonl"), &events)?;
    Ok(())
}

fn workspace_readme(input: &CodexProposerInput<'_>) -> String {
    let proposal_policy = proposer_policy_text(input);
    let jesterky_rule = crate::jesterky_workflow::jesterky_workspace_rule(
        input.config.jesterky_workflow.enabled,
    )
    .unwrap_or("");
    let jesterky_section = if input.config.jesterky_workflow.enabled {
        "13. REQUIRED: `state/jesterky_proposer_context.md`, `state/jesterky_theme_registry.json`, and `state/jesterky_trace_annotations.jsonl` for jesterky annotate themes.\n14. If present, `state/jesterky_manifest_summary.json`, `state/jesterky_trace_rows.json`, `state/jesterky_optimizer_triples.json`, and `state/jesterky_evidence_refs.json` for jesterky process-tree evidence."
    } else {
        "13. If present, `state/jesterky_manifest_summary.json`, `state/jesterky_trace_rows.json`, `state/jesterky_optimizer_triples.json`, and `state/jesterky_evidence_refs.json` for jesterky process-tree evidence."
    };
    format!(
        r#"# GEPA Proposer Workspace

You are proposing the next GEPA prompt candidate.

Read:

1. `prompting_best_practices.md` for the shared premise/context/task_priority/heuristics/constraints/rules typology.
2. `proposal/PROPOSAL_SCHEMA.md` for the exact manifest schema.
3. `state/proposer_metadata.json` for run/generation metadata, model names, target levers, counts, and top failures.
4. `state/proposer_readme.json` for a machine-readable file index.
5. `state/proposer_failure_summary.json` first for flat losses, wins, label confusions, text, expected labels, predictions, rewards, and prompt payloads.
6. `state/proposer_repair_hints.json` for generalized reflection hints, label-confusion clusters, and guard wins.
7. `state/proposer_examples.json` for flat reflection evidence rows.
8. `state/run_context.json` for the optimizer run context and target modules.
9. `state/task_info.json` for the container-declared task, output space, metrics, and proposer hints.
10. `state/program_contract.json` for the program and mutable fields.
11. `state/parent_candidate.json` and `state/parent_payload.json` for the current prompt to mutate.
12. `state/reflective_frames.json` and `state/reflector_input.json` for nested reflection evidence and sampled wins/losses.
{jesterky_section}

Before writing the manifest, inspect those files with shell, Python, or JQ and form a short evidence summary. Use `state/task_info.json`, reflection traces, rationales, and expected/predicted outputs to infer what kind of task this is before deciding what style of prompt edit is valid.
Use a real review workflow: summarize the parent prompt, inspect reflection wins/losses, inspect the parent payload, then write `proposal/manifest.json`.

Reflect over the evidence like GEPA's Python workspace proposer. You have wide latitude over the prompt content: rewrite structure, add role priming, include numbered sections, restate the task contract, and add examples when the task policy allows them.

{jesterky_rule}

{proposal_policy}

Write exactly {proposal_count} distinct candidate proposals to `proposal/manifest.json`.
"#,
        jesterky_section = jesterky_section,
        jesterky_rule = jesterky_rule,
        proposal_policy = proposal_policy,
        proposal_count = input.config.gepa.proposals_per_generation
    )
}

fn proposal_schema(input: &CodexProposerInput<'_>) -> String {
    let payload_rule = if input.config.candidate.target_modules.len() > 1 {
        "- Multiple target modules are active. Each `proposed_payload` must be the full candidate payload, with every target module present and non-empty.\n"
    } else {
        "- Keep the change targeted to the target module named in `state/run_context.json`.\n"
    };
    let proposal_policy = proposer_policy_text(input);
    format!(
        r#"# GEPA Workspace Proposer Schema

Write `proposal/manifest.json` as strict JSON using this schema:

```json
{{
  "schema_version": "gepa_workspace_proposal_v3",
  "critique": "What the parent candidate is missing, grounded in state/ evidence.",
  "evidence": {{
    "reviewed_files": [
      "prompting_best_practices.md",
      "state/proposer_metadata.json",
      "state/proposer_readme.json",
      "state/run_context.json",
      "state/task_info.json",
      "state/program_contract.json",
      "state/proposer_failure_summary.json",
      "state/proposer_repair_hints.json",
      "state/proposer_examples.json",
      "state/reflective_frames.json",
      "state/parent_payload.json",
      "state/reflector_input.json",
      "state/proposal_request.json",
      "state/jesterky_manifest_summary.json",
      "state/jesterky_trace_rows.json",
      "state/jesterky_optimizer_triples.json",
      "state/jesterky_evidence_refs.json"
    ],
    "candidate_comparison": "Short summary of the parent prompt and the reflection evidence it should address.",
    "failure_patterns": ["Observed failure pattern grounded in losing rollout examples."],
    "winning_patterns": ["Observed winning pattern grounded in successful rollout examples."],
    "example_ids_used": ["<example_id_1>", "<example_id_2>"]
  }},
  "rationale": "Why the proposed prompts should improve the target module.",
  "proposals": [
    {{
      "proposal_type": "parent_variation",
      "parent_candidate_ids": ["<parent_candidate_id>"],
      "rationale": "Why this variation should help.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }},
    {{
      "proposal_type": "parent_variation",
      "parent_candidate_ids": ["<parent_candidate_id>"],
      "rationale": "Which distinct failure cluster this variation attempts to fix.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }}
  ]
}}
```

Rules:

- Read `prompting_best_practices.md`, `state/proposer_metadata.json`, `state/proposer_readme.json`, `state/run_context.json`, `state/task_info.json`, `state/program_contract.json`, `state/proposer_failure_summary.json`, `state/proposer_repair_hints.json`, `state/proposer_examples.json`, `state/reflective_frames.json`, `state/parent_payload.json`, `state/reflector_input.json`, and `state/proposal_request.json`. If present, also read the `state/jesterky_*` files before proposing from jesterky process evidence.
- Preserve the exact top-level and evidence field names from the JSON schema. In particular, use `evidence.reviewed_files` and `evidence.example_ids_used`; do not rename them to `files_reviewed`, `example_ids`, or any other alias.
- Use shell/Python/JQ inspection to summarize the workspace before writing the manifest. Do not jump straight to editing `proposal/manifest.json`.
- Minimum review workflow: inspect `state/proposer_metadata.json`, inspect `state/task_info.json`, inspect reflection wins/losses and trace refs, inspect parent payload, then write the manifest.
- Use `state/proposer_failure_summary.json`, `state/proposer_repair_hints.json`, and `state/proposer_examples.json` as the primary source for rewards, failures, wins, expected outputs, predictions, and example text. Use nested evidence frames when task semantics or trace-level behavior are unclear.
- Use `prompting_best_practices.md` to classify each proposed change as a premise, context, task_priority, core_task_description, heuristic, constraint, rule, input_description, or output_description.
- Fill `evidence` with concrete files reviewed, candidate comparison, failure patterns, winning patterns, and example ids from `state/proposer_failure_summary.json`.
- Do not leave required evidence fields empty. `evidence.candidate_comparison` must be a non-empty summary of the parent prompt against observed reflection evidence, and `evidence.example_ids_used` must include at least one concrete example id inspected from the state files.
- Proposals should aim to generalize. Add structural sections (role, task, output rules, examples) and domain-specific rules only when they are task-valid.
- {proposal_policy}
- `proposed_payload` keys must be exactly the mutable target module ids from `state/run_context.json`. Do not use chat-message keys such as `content`, `role`, or `modules` unless one of those strings is literally listed as a target module id.
- For a single target module, write the target module id directly as the only `proposed_payload` key. For example, if `state/run_context.json.target_modules == ["stage2_system"]`, write `"proposed_payload": {{"stage2_system": "<full replacement instruction>"}}`, not `"content"` and not a `modules` array.
- At most one proposal may be conservative. The remaining proposals must be very ambitious, high-variance, task-specific updates that could plausibly produce substantially better task performance than the parent, and each rationale must name the failure clusters it attacks.
- Shoot for large wins. Mild parent clarifications are wasted candidate budget unless they are the single conservative control.
- Do not waste candidates on generic output-contract polish, canonical-label reminders, or baseline paraphrases unless the dominant failures are actually output-format failures.
- Use whatever combination works: label-disambiguation rules, output-format constraints, structural rewrites, few-shot examples, role priming, edge-case enumeration. Distinct proposals should explore distinct strategies, not paraphrase each other.
- Create exactly `state/proposal_request.json.proposals_per_round` distinct proposals.
- Use `proposal_type="parent_variation"` for each proposal.
- Do not propose a duplicate of `state/parent_payload.json`.
- Preserve all parent payload keys unless a key is intentionally changed.
- Each `proposed_payload` must be the full payload object to register as a GEPA candidate.
- For each proposal, at least one targeted module must change from the selected parent payload.
{payload_rule}"#
    )
}

fn proposal_request(input: &CodexProposerInput<'_>, prompting_best_practices: &str) -> Value {
    let proposal_count = input.config.gepa.proposals_per_generation;
    json!({
        "proposal_count": proposal_count,
        "proposals_per_round": proposal_count,
        "proposal_type": "parent_variation",
        "variation_parent_candidate_ids": [input.parent.candidate_id.clone()],
        "target_modules": input.config.candidate.target_modules,
        "parent_candidate_id": input.parent.candidate_id,
        "literal_example_policy": proposer_literal_policy_json(input),
        "prompting_best_practices": prompting_best_practices,
        "ambition_contract": [
            "At most one proposal may be conservative.",
            "Every other proposal must be a very ambitious, task-specific prompt update that names the top failure cluster it is meant to fix and could plausibly produce substantially better task performance than the parent.",
            "Shoot for large wins. Small prompt polish, extra canonical-output reminders, or mild clarifications are not acceptable except for the single conservative control.",
            "Generic output-contract reminders, canonical-label reminders, or paraphrases of the parent are wasted proposals unless paired with concrete task heuristics.",
            "Make at least half the proposals structurally different from the parent, not just longer."
        ],
        "instructions": format!("Create exactly proposals_per_round distinct parent_variation candidates. {} At most one proposal may be conservative; the rest must be very ambitious, task-specific changes aimed at named top failure clusters and designed to substantially outperform the parent. Make distinct candidates explore genuinely different strategies (structural rewrites, boundary taxonomies, conflict precedence, answer-routing procedures, few-shot examples when allowed, role priming, etc.) rather than paraphrasing one another.", proposer_policy_text(input)),
    })
}

fn task_pool_counts(input: &CodexProposerInput<'_>) -> Value {
    let mut counts = Map::new();
    if let Some(pools) = input.task_pool_rows.as_object() {
        for (name, pool) in pools {
            if name == "schema_version" || name == "heldout" || name == "pareto" {
                continue;
            }
            let row_count = pool
                .get("row_count")
                .and_then(Value::as_u64)
                .or_else(|| {
                    pool.get("rows")
                        .and_then(Value::as_array)
                        .map(|rows| rows.len() as u64)
                })
                .unwrap_or(0);
            counts.insert(name.clone(), json!(row_count));
        }
    }
    Value::Object(counts)
}

fn parent_candidate_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "candidate_id": input.parent.candidate_id.clone(),
        "parent_id": input.parent.parent_id.clone(),
        "source": input.parent.source.clone(),
        "status": input.parent.status.clone(),
        "payload": input.parent.payload.clone(),
    })
}

fn proposer_visible_task_ids(input: &CodexProposerInput<'_>) -> BTreeSet<String> {
    input
        .task_pool_rows
        .get("reflection")
        .and_then(|pool| pool.get("task_ids"))
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default()
}

fn proposer_visible_frame(
    input: &CodexProposerInput<'_>,
    frame: &synth_optimizer_platform::SensorFrame,
) -> bool {
    if frame.evaluation_stage == "heldout" || frame.split == input.config.taskset.heldout_split {
        return false;
    }
    if !matches!(
        frame.evaluation_stage.as_str(),
        "seed_full_train" | "candidate_minibatch" | "parent_minibatch_reference" | "reflection"
    ) {
        return false;
    }
    let visible_task_ids = proposer_visible_task_ids(input);
    visible_task_ids.is_empty() || visible_task_ids.contains(&frame.task_id)
}

fn frame_rollout_trace(
    input: &CodexProposerInput<'_>,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let inline = frame
        .metadata
        .get("rollout_trace")
        .cloned()
        .unwrap_or(Value::Null);
    if rollout_trace_has_proposer_evidence(&inline) {
        return inline;
    }
    read_frame_rollout_trace_artifact(input, frame).unwrap_or(inline)
}

fn rollout_trace_has_proposer_evidence(trace: &Value) -> bool {
    json_path(trace, &["task_payload", "example"])
        .and_then(Value::as_object)
        .is_some()
        || string_path(trace, &["summary", "expected"]).is_some()
        || string_path(trace, &["summary", "prediction"]).is_some()
}

fn read_frame_rollout_trace_artifact(
    input: &CodexProposerInput<'_>,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Option<Value> {
    let sensor_frame_id = frame.sensor_frame_id.trim();
    if sensor_frame_id.is_empty() {
        return None;
    }
    let run_dir = input.workspace_dir.parent()?.parent()?;
    let path = run_dir
        .join("rollout_traces")
        .join(format!("{sensor_frame_id}.json"));
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn rollouts_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut rows = Vec::new();
    for candidate in input.candidates {
        let visible_frame_count = candidate
            .sensor_frames
            .iter()
            .filter(|frame| proposer_visible_frame(input, frame))
            .count();
        for frame in &candidate.sensor_frames {
            if !proposer_visible_frame(input, frame) {
                continue;
            }
            let rollout_trace = frame_rollout_trace(input, frame);
            let summary = json_path(&rollout_trace, &["summary"])
                .cloned()
                .unwrap_or(Value::Null);
            let outcome = json_path(&rollout_trace, &["outcome"])
                .cloned()
                .unwrap_or_else(|| {
                    json!({
                        "status": frame.status,
                        "success_status": frame.success_status,
                        "reward": frame.reward,
                    })
                });
            let example = json_path(&rollout_trace, &["task_payload", "example"])
                .cloned()
                .unwrap_or_else(|| {
                    json!({
                        "example_id": frame.example_id,
                        "task_id": frame.task_id,
                    })
                });
            let reward_details = json_path(&outcome, &["reward_info", "details"])
                .or_else(|| frame.metadata.get("reward_details"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            let expected = string_path(&summary, &["expected"])
                .or_else(|| string_path(&reward_details, &["expected"]))
                .or_else(|| string_path(&example, &["label"]))
                .unwrap_or_default();
            let prediction = string_path(&summary, &["prediction"])
                .or_else(|| string_path(&reward_details, &["prediction"]))
                .unwrap_or_default();
            let text = string_path(&example, &["text"]).unwrap_or_default();
            let policy_model = string_path(&reward_details, &["policy_model"])
                .or_else(|| string_path(&frame.usage, &["model"]))
                .unwrap_or_default();
            let rationale_text = frame
                .objective_scores
                .iter()
                .filter_map(|score| score.rationale.as_deref())
                .find(|value| !value.trim().is_empty())
                .unwrap_or_default()
                .to_string();
            let artifact_refs = frame
                .artifact_refs
                .iter()
                .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
                .filter(|value| !value.is_null())
                .collect::<Vec<_>>();
            let trace_refs = frame
                .trace_digest
                .as_ref()
                .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
                .unwrap_or_default();
            rows.push(json!({
                "candidate_id": frame.candidate_id,
                "evaluation_stage": frame.evaluation_stage,
                "example_id": frame.example_id,
                "task_id": frame.task_id,
                "reward": frame.reward,
                "status": frame.status,
                "success_status": frame.success_status,
                "failure": frame.failure,
                "summary": summary,
                "outcome": outcome,
                "rationale_text": rationale_text,
                "expected": expected,
                "prediction": prediction,
                "text": text,
                "policy_model": policy_model,
                "usage": frame.usage,
                "artifact_refs": artifact_refs,
                "trace_refs": trace_refs,
                "candidate_status": candidate.status,
                "actionable_side_info": frame.actionable_side_info,
            }));
        }
        if visible_frame_count == 0 {
            rows.extend(proposer_score_only_rows(input, candidate));
        }
    }
    Value::Array(rows)
}

fn proposer_examples_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut rows = Vec::new();
    for candidate in input.candidates {
        let visible_frame_count = candidate
            .sensor_frames
            .iter()
            .filter(|frame| proposer_visible_frame(input, frame))
            .count();
        for frame in &candidate.sensor_frames {
            if proposer_visible_frame(input, frame) {
                rows.push(proposer_example_row(input, candidate, frame));
            }
        }
        if visible_frame_count == 0 {
            rows.extend(proposer_score_only_rows(input, candidate));
        }
    }
    rows.sort_by(|left, right| {
        proposer_example_sort_key(left).cmp(&proposer_example_sort_key(right))
    });
    Value::Array(rows)
}

fn proposer_score_only_rows(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
) -> Vec<Value> {
    let visible_task_ids = proposer_visible_task_ids(input);
    let scores = if !candidate.train_scores.is_empty() {
        (&candidate.train_scores, "train_score")
    } else {
        (&candidate.minibatch_scores, "minibatch_score")
    };
    scores
        .0
        .iter()
        .filter(|score| {
            visible_task_ids.is_empty()
                || visible_task_ids.contains(&score.task_id)
                || visible_task_ids.contains(&score.example_id)
        })
        .map(|score| proposer_score_only_row(input, candidate, score, scores.1))
        .collect()
}

fn proposer_score_only_row(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
    score: &RolloutScore,
    evaluation_stage: &str,
) -> Value {
    json!({
        "schema_version": "gepa_proposer_example.v1",
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "candidate_status": candidate.status,
        "is_parent": candidate.candidate_id == input.parent.candidate_id,
        "evaluation_stage": evaluation_stage,
        "example_id": score.example_id,
        "task_id": score.task_id,
        "reward": score.reward,
        "status": "score_only",
        "success_status": if score.reward >= 1.0 { "success" } else { "failure" },
        "expected": "",
        "prediction": "",
        "text": "",
        "policy_model": input.config.policy.model,
        "objective_rationale": "Score-only fallback row: rollout score was present but no sensor frame was materialized for this example.",
        "failure": Value::Null,
        "usage": Value::Null,
        "artifact_refs": [],
        "trace_refs": [],
    })
}

fn proposer_example_row(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let rollout_trace = frame_rollout_trace(input, frame);
    let summary = json_path(&rollout_trace, &["summary"])
        .cloned()
        .unwrap_or(Value::Null);
    let outcome = json_path(&rollout_trace, &["outcome"])
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "status": frame.status,
                "success_status": frame.success_status,
                "reward": frame.reward,
            })
        });
    let example = json_path(&rollout_trace, &["task_payload", "example"])
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "example_id": frame.example_id,
                "task_id": frame.task_id,
            })
        });
    let reward_details = json_path(&outcome, &["reward_info", "details"])
        .or_else(|| frame.metadata.get("reward_details"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let expected = string_path(&summary, &["expected"])
        .or_else(|| string_path(&reward_details, &["expected"]))
        .or_else(|| string_path(&example, &["label"]))
        .unwrap_or_default();
    let prediction = string_path(&summary, &["prediction"])
        .or_else(|| string_path(&reward_details, &["prediction"]))
        .unwrap_or_default();
    let text = string_path(&example, &["text"]).unwrap_or_default();
    let policy_model = string_path(&reward_details, &["policy_model"])
        .or_else(|| string_path(&frame.usage, &["model"]))
        .unwrap_or_default();
    let objective_rationale = frame
        .objective_scores
        .iter()
        .filter_map(|score| score.rationale.as_deref())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
        .to_string();
    let artifact_refs = frame
        .artifact_refs
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    let trace_refs = frame
        .trace_digest
        .as_ref()
        .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
        .unwrap_or_default();
    json!({
        "schema_version": "gepa_proposer_example.v1",
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "candidate_status": candidate.status,
        "is_parent": candidate.candidate_id == input.parent.candidate_id,
        "evaluation_stage": frame.evaluation_stage,
        "example_id": frame.example_id,
        "task_id": frame.task_id,
        "reward": frame.reward,
        "status": frame.status,
        "success_status": frame.success_status,
        "expected": expected,
        "prediction": prediction,
        "text": text,
        "policy_model": policy_model,
        "objective_rationale": objective_rationale,
        "failure": frame.failure,
        "usage": frame.usage,
        "artifact_refs": artifact_refs,
        "trace_refs": trace_refs,
    })
}

fn proposer_failure_summary_read_model(
    input: &CodexProposerInput<'_>,
    proposer_examples: &Value,
) -> Value {
    let rows = proposer_examples.as_array().cloned().unwrap_or_default();
    let abstract_training_targets = literal_training_target_policy(input) == "forbid";
    let mut losses = Vec::new();
    let mut wins = Vec::new();
    let mut loss_labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut win_labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut confusion_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut candidate_stats: BTreeMap<String, ProposerCandidateEvidenceStats> = BTreeMap::new();

    for row in &rows {
        let candidate_id = string_path(row, &["candidate_id"]).unwrap_or_default();
        let expected = string_path(row, &["expected"]).unwrap_or_else(|| "unknown".to_string());
        let prediction = string_path(row, &["prediction"]).unwrap_or_else(|| "unknown".to_string());
        let expected_key = if abstract_training_targets {
            answer_value_bucket(&expected)
        } else {
            expected.clone()
        };
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        let stats = candidate_stats.entry(candidate_id).or_default();
        stats.rollouts += 1;
        stats.reward_sum += reward;
        if reward >= 1.0 {
            stats.wins += 1;
            *win_labels.entry(expected_key).or_default() += 1;
            wins.push(proposer_example_compact(row));
        } else {
            stats.losses += 1;
            *loss_labels.entry(expected_key).or_default() += 1;
            *confusion_counts
                .entry(if abstract_training_targets {
                    abstract_target_confusion_key(&expected, &prediction)
                } else {
                    format!("{expected} -> {prediction}")
                })
                .or_default() += 1;
            losses.push(proposer_example_compact(row));
        }
    }

    let total_loss_count = losses.len();
    let total_win_count = wins.len();

    losses.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    wins.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    losses.truncate(80);
    wins.truncate(80);

    let candidate_outcomes = candidate_stats
        .into_iter()
        .map(|(candidate_id, stats)| {
            json!({
                "candidate_id": candidate_id,
                "rollouts": stats.rollouts,
                "wins": stats.wins,
                "losses": stats.losses,
                "average_reward": if stats.rollouts == 0 {
                    0.0
                } else {
                    stats.reward_sum / stats.rollouts as f64
                },
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema_version": "gepa_proposer_failure_summary.v1",
        "instructions": if abstract_training_targets {
            "Use this file first. The container task policy forbids literal training-target mappings: label_confusions are bucketed by output shape. Do not convert exact expected outputs or predictions from rollout evidence into prompt mappings."
        } else {
            "Use this file first. It is a flat, jq-friendly view of reflection evidence with text, expected output, prediction, and reward for wins/losses."
        },
        "literal_example_policy": proposer_literal_policy_json(input),
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "loss_count": total_loss_count,
        "win_count": total_win_count,
        "losses": losses,
        "wins": wins,
        "loss_labels": top_count_pairs(loss_labels, 40),
        "win_labels": top_count_pairs(win_labels, 40),
        "label_confusions": top_count_pairs(confusion_counts, 40),
        "candidate_outcomes": candidate_outcomes,
    })
}

fn proposer_repair_hints_read_model(
    input: &CodexProposerInput<'_>,
    proposer_examples: &Value,
) -> Value {
    let mut parent_loss_examples = Vec::new();
    let mut guard_win_examples = Vec::new();
    let mut confusion_clusters: BTreeMap<String, ProposerConfusionCluster> = BTreeMap::new();
    let rows = proposer_examples.as_array().cloned().unwrap_or_default();
    let abstract_training_targets = literal_training_target_policy(input) == "forbid";
    for row in &rows {
        let is_parent = row
            .get("is_parent")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !is_parent {
            continue;
        }
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        if reward >= 1.0 {
            guard_win_examples.push(proposer_reflection_example(row));
        } else {
            let expected = string_path(row, &["expected"]).unwrap_or_else(|| "unknown".to_string());
            let prediction =
                string_path(row, &["prediction"]).unwrap_or_else(|| "unknown".to_string());
            let key = if abstract_training_targets {
                abstract_target_confusion_key(&expected, &prediction)
            } else {
                format!("{expected} -> {prediction}")
            };
            let cluster =
                confusion_clusters
                    .entry(key)
                    .or_insert_with(|| ProposerConfusionCluster {
                        expected_label: if abstract_training_targets {
                            answer_value_bucket(&expected)
                        } else {
                            expected
                        },
                        observed_prediction: if abstract_training_targets {
                            answer_value_bucket(&prediction)
                        } else {
                            prediction
                        },
                        count: 0,
                        example_ids: Vec::new(),
                        candidate_ids: BTreeSet::new(),
                    });
            cluster.count += 1;
            if let Some(example_id) = string_path(row, &["example_id"]) {
                if cluster.example_ids.len() < 8 {
                    cluster.example_ids.push(example_id);
                }
            }
            if let Some(candidate_id) = string_path(row, &["candidate_id"]) {
                cluster.candidate_ids.insert(candidate_id);
            }
            parent_loss_examples.push(proposer_reflection_example(row));
        }
    }
    parent_loss_examples.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    guard_win_examples.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    parent_loss_examples.truncate(40);
    guard_win_examples.truncate(24);
    let mut clusters = confusion_clusters.into_values().collect::<Vec<_>>();
    clusters.sort_by(|left, right| {
        right.count.cmp(&left.count).then_with(|| {
            left.expected_label
                .cmp(&right.expected_label)
                .then_with(|| left.observed_prediction.cmp(&right.observed_prediction))
        })
    });
    clusters.truncate(40);
    json!({
        "schema_version": "gepa_proposer_repair_hints.v1",
        "instructions": if abstract_training_targets {
            "Use this after proposer_failure_summary. The task policy forbids literal training-target mappings: use rollout evidence to infer general rules, but do not lift observed exact targets into prompt tables."
        } else {
            "Use this after proposer_failure_summary. It is a reflection guide that highlights common confusions and guard wins. Use concrete examples only when the task_info/program policy says they are valid for this output space."
        },
        "literal_example_policy": proposer_literal_policy_json(input),
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "label_confusion_clusters": clusters.into_iter().map(ProposerConfusionCluster::into_json).collect::<Vec<_>>(),
        "parent_loss_examples": parent_loss_examples,
        "guard_win_examples": guard_win_examples,
        "proposal_guidance": [
            "Preserve the program's hard output contract (e.g. exact label format), but feel free to change everything else.",
            if abstract_training_targets { "Turn repeated confusions into general task procedures. Do not add literal train target mappings." } else { "Turn repeated confusions into explicit fixes: rules, output-disambiguation tables, or input→output few-shot pairs — whichever is most direct and task-valid." },
            if abstract_training_targets { "If examples are useful, make them abstract or synthetic examples rather than copied train input-target pairs." } else { "You may quote, paraphrase, or summarize training inputs as illustrative examples when that is the clearest way to teach a distinction." },
            "At most one proposal may be conservative. The remaining proposals should be very ambitious, high-variance, task-specific changes that could plausibly produce substantially better task performance than the parent.",
            "Shoot for large wins; mild polish and safe clarifications are wasted unless used as the single conservative control.",
            "Do not spend proposals on output-contract polish unless the dominant failures are output-format failures. Target the top confusion clusters directly.",
            "Make proposals meaningfully distinct from each other and from the parent — paraphrases of the seed are wasted budget."
        ],
    })
}

struct ProposerConfusionCluster {
    expected_label: String,
    observed_prediction: String,
    count: usize,
    example_ids: Vec<String>,
    candidate_ids: BTreeSet<String>,
}

impl ProposerConfusionCluster {
    fn into_json(self) -> Value {
        json!({
            "expected_label": self.expected_label,
            "observed_prediction": self.observed_prediction,
            "count": self.count,
            "example_ids": self.example_ids,
            "candidate_ids": self.candidate_ids.into_iter().collect::<Vec<_>>(),
            "reflection_prompt": "Diagnose and fix the confusion. First infer the task/output space from task_info and traces. Use concrete tables only for tasks where literal mappings are valid; otherwise write general procedures instead of target memorization.",
        })
    }
}

fn proposer_reflection_example(row: &Value) -> Value {
    json!({
        "candidate_id": row.get("candidate_id").cloned().unwrap_or(Value::Null),
        "candidate_status": row.get("candidate_status").cloned().unwrap_or(Value::Null),
        "is_parent": row.get("is_parent").cloned().unwrap_or(Value::Bool(false)),
        "evaluation_stage": row.get("evaluation_stage").cloned().unwrap_or(Value::Null),
        "example_id": row.get("example_id").cloned().unwrap_or(Value::Null),
        "seed": row.get("seed").cloned().unwrap_or(Value::Null),
        "reward": row.get("reward").cloned().unwrap_or(Value::Null),
        "expected": row.get("expected").cloned().unwrap_or(Value::Null),
        "prediction": row.get("prediction").cloned().unwrap_or(Value::Null),
        "policy_model": row.get("policy_model").cloned().unwrap_or(Value::Null),
        "objective_rationale": row.get("objective_rationale").cloned().unwrap_or(Value::Null),
        "artifact_refs": row.get("artifact_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "trace_refs": row.get("trace_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "text_policy": "Full text is in proposer_examples.json. Follow state/proposal_request.json.literal_example_policy before quoting or mapping examples inside candidate prompts.",
    })
}

fn proposer_metadata_read_model(
    input: &CodexProposerInput<'_>,
    rollouts: &Value,
    proposer_examples: &Value,
    proposer_failure_summary: &Value,
) -> Value {
    let proposer_model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let workspace_root = input
        .workspace_dir
        .parent()
        .and_then(Path::parent)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| input.workspace_dir.display().to_string());
    json!({
        "schema_version": "gepa_proposer_metadata.v1",
        "run_id": input.config.run.run_id,
        "generation": input.generation,
        "workspace_dir": input.workspace_dir,
        "workspace_root": workspace_root.clone(),
        "run_artifact_dir": workspace_root,
        "parent_candidate_id": input.parent.candidate_id,
        "candidate_count": input.candidates.len(),
        "rollout_row_count": count_json_array(rollouts),
        "proposer_example_count": count_json_array(proposer_examples),
        "loss_count": proposer_failure_summary
            .get("loss_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "win_count": proposer_failure_summary
            .get("win_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "top_failures": proposer_failure_summary
            .get("label_confusions")
            .and_then(Value::as_array)
            .map(|items| Value::Array(items.iter().take(5).cloned().collect()))
            .unwrap_or_else(|| Value::Array(Vec::new())),
        "task_info": task_info_value(input).cloned().unwrap_or(Value::Null),
        "task_output_space_kind": task_output_space_kind(input),
        "literal_training_target_policy": literal_training_target_policy(input),
        "literal_example_policy": proposer_literal_policy_json(input),
        "target_modules": input.config.candidate.target_modules,
        "mutable_levers": input.program.mutable_field_ids(),
        "proposal_count": input.config.gepa.proposals_per_generation,
        "proposals_per_generation": input.config.gepa.proposals_per_generation,
        "minibatch_size": input.config.gepa.minibatch_size,
        "policy": {
            "provider": input.config.policy.provider,
            "model": input.config.policy.model,
            "api_family": input.config.policy.api_family,
            "base_url": input.config.policy.base_url,
            "inference_url": input.config.policy.inference_url,
            "max_tokens": input.config.policy.max_tokens,
            "disable_reasoning": input.config.policy.disable_reasoning,
            "tool_call_style": input.config.policy.tool_call_style,
            "proxy_mode": input.config.policy.proxy_mode,
            "credential_mode": input.config.policy.credential_mode,
            "config": input.config.policy.config,
        },
        "proposer": {
            "backend": input.config.proposer.backend,
            "model": proposer_model,
            "timeout_seconds": input.config.proposer.timeout_seconds,
            "sandbox_mode": input.config.proposer.sandbox_mode,
            "approval_policy": input.config.proposer.approval_policy,
        },
        "budgets": {
            "max_total_rollouts": input.config.gepa.effective_max_total_rollouts(),
            "max_train_rollouts": input.config.gepa.train_rollout_limit(),
            "max_cost_usd": input.config.gepa.max_cost_usd,
        },
        "task_pool_counts": task_pool_counts(input),
        "read_first": [
            "state/proposer_metadata.json",
            "state/task_info.json",
            "state/proposer_failure_summary.json",
            "state/proposer_repair_hints.json",
            "state/proposer_examples.json",
            "state/parent_payload.json",
            "state/reflective_frames.json",
            "state/reflector_input.json"
        ],
    })
}

fn proposer_readme_read_model() -> Value {
    json!({
        "schema_version": "gepa_proposer_readme.v1",
        "purpose": "Machine-readable index for the GEPA proposer workspace.",
        "read_order": [
            {
                "path": "prompting_best_practices.md",
                "use": "Shared GEPA proposer/reflector guidance: instruction typology, evidence-first loop, diagnosis rules, good changes, and bad changes."
            },
            {
                "path": "state/proposer_metadata.json",
                "use": "Small run/generation context, policy/proposer model names, counts, target modules, and top failure labels."
            },
            {
                "path": "state/task_info.json",
                "use": "Container-declared task description, output space, metrics, and proposer hints. Read this before deciding whether literal examples, tables, or abstract rules are valid."
            },
            {
                "path": "state/proposer_failure_summary.json",
                "use": "Primary flat evidence file. Start here for losses, wins, label confusions, expected/predicted labels, text, rewards, and target payloads."
            },
            {
                "path": "state/proposer_repair_hints.json",
                "use": "Reflection hints derived from parent losses and guard wins. Use it to pick which confusions to fix. Follow state/proposal_request.json.literal_example_policy before quoting or mapping examples inside candidate prompts."
            },
            {
                "path": "state/proposer_examples.json",
                "use": "All flat reflection evidence rows with text, expected, prediction, reward, trace refs, and usage."
            },
            {
                "path": "state/jesterky_manifest_summary.json",
                "use": "Optional. Jesterky RunManifest identity, status, stop_reason, budgets, goals, invariants, and process-tree counts. Read stop_reason as typed data."
            },
            {
                "path": "state/jesterky_optimizer_triples.json",
                "use": "Optional. Jesterky optimizer-facing process-tree rows: inputs, outputs, copied score, signal, artifacts, addr, and label per leaf plus scored interior nodes."
            },
            {
                "path": "state/jesterky_trace_rows.json",
                "use": "Optional. Full Addr-sorted jesterky process tree rows for context and alignment."
            },
            {
                "path": "state/jesterky_evidence_refs.json",
                "use": "Optional. Jesterky process addrs and artifact refs that can be cited in proposal evidence."
            },
            {
                "path": "state/parent_payload.json",
                "use": "The parent prompt payload to mutate."
            },
            {
                "path": "state/reflective_frames.json",
                "use": "Nested reflective evidence under .frames[] for deeper trace-level detail."
            },
            {
                "path": "state/reflector_input.json",
                "use": "Sampled winning and losing reflection traces plus proposal guidance."
            }
        ],
        "manifest_evidence_contract": [
            "List the files actually reviewed.",
            "Summarize parent prompt and reflection evidence.",
            "Name failure patterns grounded in losing examples.",
            "Name winning patterns grounded in successful examples.",
            "Cite concrete example_id values used."
        ],
        "candidate_prompt_contract": [
            "Read prompting_best_practices.md and classify each change as a premise, context, task_priority, core_task_description, heuristic, constraint, rule, input_description, or output_description.",
            "Ground the prompt in the rollout evidence; pull in any task-specific factual knowledge you can extract from wins and losses.",
            "Follow state/proposal_request.json.literal_example_policy. The valid use of concrete examples depends on the container-declared task/output space and the rollout traces.",
            "Mix strategies across proposals: structural rewrites, examples where task-valid, role priming, output-disambiguation rules, terse contracts — don't paraphrase the seed.",
            "Preserve any hard output contract (e.g. exact label format) declared in the program contract."
        ]
    })
}

fn proposer_example_compact(row: &Value) -> Value {
    json!({
        "candidate_id": row.get("candidate_id").cloned().unwrap_or(Value::Null),
        "candidate_status": row.get("candidate_status").cloned().unwrap_or(Value::Null),
        "is_parent": row.get("is_parent").cloned().unwrap_or(Value::Bool(false)),
        "evaluation_stage": row.get("evaluation_stage").cloned().unwrap_or(Value::Null),
        "example_id": row.get("example_id").cloned().unwrap_or(Value::Null),
        "seed": row.get("seed").cloned().unwrap_or(Value::Null),
        "reward": row.get("reward").cloned().unwrap_or(Value::Null),
        "expected": row.get("expected").cloned().unwrap_or(Value::Null),
        "prediction": row.get("prediction").cloned().unwrap_or(Value::Null),
        "text": row.get("text").cloned().unwrap_or(Value::Null),
        "policy_model": row.get("policy_model").cloned().unwrap_or(Value::Null),
        "artifact_refs": row.get("artifact_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "trace_refs": row.get("trace_refs").cloned().unwrap_or(Value::Array(Vec::new())),
    })
}

fn count_json_array(value: &Value) -> usize {
    value.as_array().map(Vec::len).unwrap_or(0)
}

fn proposer_example_sort_key(row: &Value) -> (String, String, i64, String) {
    (
        string_path(row, &["candidate_id"]).unwrap_or_default(),
        string_path(row, &["evaluation_stage"]).unwrap_or_default(),
        row.get("seed").and_then(Value::as_i64).unwrap_or(0),
        string_path(row, &["example_id"]).unwrap_or_default(),
    )
}

fn proposer_summary_sort_key(row: &Value) -> (String, i64, String) {
    (
        string_path(row, &["evaluation_stage"]).unwrap_or_default(),
        row.get("seed").and_then(Value::as_i64).unwrap_or(0),
        string_path(row, &["example_id"]).unwrap_or_default(),
    )
}

fn top_count_pairs(counts: BTreeMap<String, usize>, limit: usize) -> Value {
    let mut items = counts.into_iter().collect::<Vec<_>>();
    items.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    Value::Array(
        items
            .into_iter()
            .take(limit)
            .map(|(key, count)| json!({"key": key, "count": count}))
            .collect(),
    )
}

fn json_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

fn string_path(value: &Value, path: &[&str]) -> Option<String> {
    json_path(value, path)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn task_info_value<'a>(input: &'a CodexProposerInput<'_>) -> Option<&'a Value> {
    input.program.metadata.get("task_info")
}

fn metadata_string_at(input: &CodexProposerInput<'_>, paths: &[&[&str]]) -> Option<String> {
    let metadata = Value::Object(input.program.metadata.clone());
    for path in paths {
        if let Some(value) = json_path(&metadata, path).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn task_output_space_kind(input: &CodexProposerInput<'_>) -> Option<String> {
    metadata_string_at(
        input,
        &[
            &["task_info", "output_space", "kind"],
            &["task_info", "task", "output_space", "kind"],
            &["task_info", "metadata", "output_space", "kind"],
            &["task_info", "metadata", "task_output_space"],
            &["proposer_hints", "task_output_space"],
            &["proposer_constraints", "task_output_space"],
        ],
    )
}

fn literal_training_target_policy(input: &CodexProposerInput<'_>) -> &'static str {
    if let Some(policy) = metadata_string_at(
        input,
        &[
            &["task_info", "proposer_hints", "literal_training_targets"],
            &[
                "task_info",
                "metadata",
                "proposer_hints",
                "literal_training_targets",
            ],
            &["proposer_hints", "literal_training_targets"],
            &["proposer_constraints", "literal_training_targets"],
        ],
    ) {
        let policy = policy.to_ascii_lowercase();
        if policy.contains("forbid") || policy.contains("disallow") || policy.contains("avoid") {
            return "forbid";
        }
        if policy.contains("allow") {
            return "allow";
        }
    }
    if let Some(kind) = task_output_space_kind(input) {
        let kind = kind.to_ascii_lowercase();
        if kind.contains("finite")
            || kind.contains("label")
            || kind.contains("closed")
            || kind.contains("intent")
            || kind.contains("class")
        {
            return "allow";
        }
        if kind.contains("open")
            || kind.contains("free")
            || kind.contains("answer")
            || kind.contains("extract")
            || kind.contains("span")
            || kind.contains("generat")
        {
            return "forbid";
        }
    }
    if input.program.metadata.get("labels").is_some()
        || input.program.metadata.get("label_guidance").is_some()
    {
        return "allow";
    }
    let primary_metric = metadata_string_at(
        input,
        &[
            &["task_info", "metadata", "primary_metric"],
            &["primary_metric"],
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    let answer_contract = metadata_string_at(
        input,
        &[
            &["task_info", "metadata", "answer_contract"],
            &["answer_contract"],
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    if primary_metric.contains("f1")
        || primary_metric.contains("rouge")
        || answer_contract.contains("answer")
        || answer_contract.contains("span")
    {
        return "forbid";
    }
    "infer"
}

fn proposer_literal_policy_json(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "source": "container /task_info, /program metadata, and rollout trace evidence",
        "task_output_space_kind": task_output_space_kind(input),
        "literal_training_targets": literal_training_target_policy(input),
        "policy": [
            "First infer the task and output space from state/program_contract.json.metadata.task_info, state/proposer_metadata.json, rollout traces, expected outputs, predictions, and objective rationales.",
            "If the task has a finite closed output space, concrete boundary examples or compact output mappings can be valid when they teach the output contract.",
            "If the task is not a finite closed-output mapping, convert trace evidence into general procedures and do not copy observed training targets into the prompt.",
            "When task_info.proposer_hints is present, treat it as authoritative unless trace evidence contradicts it."
        ],
    })
}

fn proposer_policy_text(input: &CodexProposerInput<'_>) -> String {
    let output_kind = task_output_space_kind(input).unwrap_or_else(|| "unspecified".to_string());
    let literal_policy = literal_training_target_policy(input);
    format!(
        "Task policy: infer the task from container task_info, program metadata, rollout traces, expected outputs, predictions, and objective rationales before proposing. Output-space kind from the container is {output_kind:?}; literal-training-target policy is {literal_policy:?}. Finite closed-output tasks may use concrete boundary examples or output tables when they generalize. Non-closed-output tasks should turn traces into general procedures and avoid copying observed training targets into candidate prompts. Follow any task_info.proposer_hints fields."
    )
}

fn answer_value_bucket(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "empty answer".to_string();
    }
    if trimmed.eq_ignore_ascii_case("yes") || trimmed.eq_ignore_ascii_case("no") {
        return "yes/no answer".to_string();
    }
    if trimmed.chars().any(|ch| ch.is_ascii_digit()) {
        return "date/number answer".to_string();
    }
    let word_count = trimmed.split_whitespace().count();
    if word_count >= 8 {
        return "long clause answer".to_string();
    }
    if trimmed
        .chars()
        .next()
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
    {
        return "proper-name/title answer".to_string();
    }
    if word_count <= 3 {
        "short phrase/category answer".to_string()
    } else {
        "clause answer".to_string()
    }
}

fn abstract_target_confusion_key(expected: &str, prediction: &str) -> String {
    format!(
        "{} expected -> {} prediction",
        answer_value_bucket(expected),
        answer_value_bucket(prediction)
    )
}

fn literal_target_is_specific(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.len() < 4 || trimmed.len() > 96 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    !matches!(
        lower.as_str(),
        "yes" | "no" | "unknown" | "none" | "true" | "false" | "n/a"
    )
}

fn reflective_frames_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut frames = input
        .candidates
        .iter()
        .flat_map(|candidate| {
            candidate
                .sensor_frames
                .iter()
                .filter(|frame| proposer_visible_frame(input, frame))
                .map(move |frame| reflective_frame_value(input, candidate, frame))
        })
        .collect::<Vec<_>>();
    frames.sort_by(|left, right| {
        let left_key = left
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let right_key = right
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        left_key.cmp(right_key)
    });
    if frames.len() > 80 {
        frames.truncate(80);
    }
    json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "adapter": reflective_adapter_spec(),
        "frame_count": frames.len(),
        "frames": frames,
    })
}

fn reflective_frame_value(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let component_id = input
        .config
        .candidate
        .target_modules
        .first()
        .cloned()
        .unwrap_or_else(|| "candidate".to_string());
    let rollout_id = frame.rollout_id.clone().unwrap_or_default();
    let trace_refs = frame
        .trace_digest
        .as_ref()
        .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
        .unwrap_or_default();
    let confidence = reflective_confidence(frame);
    let artifact_refs = frame
        .artifact_refs
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    let failure_class = frame
        .failure
        .as_ref()
        .map(|failure| failure.failure_class().to_string())
        .unwrap_or_default();
    let verifier_rationale = frame
        .objective_scores
        .iter()
        .filter_map(|score| score.rationale.as_deref())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
        .to_string();
    let rollout_trace = frame_rollout_trace(input, frame);
    let rollout_trace = rollout_trace.as_object();
    let trace_summary = rollout_trace
        .and_then(|trace| trace.get("summary"))
        .cloned()
        .or_else(|| frame.metadata.get("summary").cloned())
        .unwrap_or(Value::Null);
    let trace_outcome = rollout_trace
        .and_then(|trace| trace.get("outcome"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "status": frame.status,
                "success_status": frame.success_status,
                "reward": frame.reward,
            })
        });
    let task_example = rollout_trace
        .and_then(|trace| trace.get("task_payload"))
        .and_then(|task_payload| task_payload.get("example"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "example_id": frame.example_id,
                "task_id": frame.task_id,
            })
        });
    let request = json!({
        "evaluation_stage": frame.evaluation_stage,
        "target_modules": input.config.candidate.target_modules.clone(),
        "example_id": frame.example_id,
        "task_id": frame.task_id,
    });
    let tool_calls = rollout_trace
        .and_then(|trace| trace.get("tool_calls"))
        .cloned()
        .unwrap_or_else(|| json!([]));
    let substitution_stats = rollout_trace
        .and_then(|trace| trace.get("substitution_stats"))
        .cloned()
        .unwrap_or_else(|| json!({"attempted": 0, "applied": 0, "warnings": []}));
    let evidence = json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "source": "sensor_frame_adapter",
        "adapter": reflective_adapter_spec(),
        "subject": {
            "algorithm_id": GEPA_ALGORITHM_ID,
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_id,
            "component_id": component_id,
            "rollout_id": rollout_id,
            "example_id": frame.example_id,
        },
        "adapter_source": GEPA_ADAPTER_SOURCE,
        "rollout_id": rollout_id,
        "example_id": frame.example_id,
        "inputs": {
            "example": task_example,
            "request": request,
        },
        "generated_outputs": {
            "summary": trace_summary,
            "outcome": trace_outcome,
        },
        "feedback": {
            "reward": frame.reward,
            "objective_scores": frame.objective_scores,
            "verifier_rationale": verifier_rationale,
        },
        "actionable_side_info": frame.actionable_side_info.clone().unwrap_or_else(|| json!({})),
        "sensors": {
            "confidence": confidence,
            "trace_digest": frame.trace_digest,
        },
        "refs": {
            "trace_refs": trace_refs,
            "rollout_id": rollout_id,
            "sensor_frame_id": frame.sensor_frame_id,
            "artifact_refs": artifact_refs,
        },
        "trace_refs": trace_refs,
        "tool_calls": tool_calls,
        "substitution_stats": substitution_stats,
        "failure_class": failure_class,
        "usage": frame.usage,
        "confidence": confidence,
        "component_id": component_id,
    });
    json!({
        "frame_id": format!("reflect:{}:{}:{}", GEPA_ALGORITHM_ID, candidate.candidate_id, frame.sensor_frame_id),
        "algorithm_id": GEPA_ALGORITHM_ID,
        "component_id": component_id,
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "rollout_id": frame.rollout_id,
        "artifact_refs": artifact_refs,
        "metadata": {
            "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
            "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
            "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
            "sensor_frame_id": frame.sensor_frame_id,
        },
        "evidence": evidence,
    })
}

fn reflective_adapter_spec() -> Value {
    json!({
        "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
        "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
        "source": GEPA_ADAPTER_SOURCE,
        "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "required_evidence_keys": [
            "schema_version",
            "source",
            "adapter",
            "subject",
            "inputs",
            "generated_outputs",
            "feedback",
            "actionable_side_info",
            "sensors",
            "refs",
        ],
    })
}

fn reflective_confidence(frame: &synth_optimizer_platform::SensorFrame) -> f64 {
    let support_count = frame
        .trace_digest
        .as_ref()
        .map(|digest| digest.llm_request_count + digest.tool_call_count)
        .unwrap_or(0);
    if frame.failure.is_some() {
        0.55
    } else if support_count > 0 {
        0.85
    } else if frame.actionable_side_info.is_some() {
        0.7
    } else {
        0.35
    }
}

fn reflector_input_read_model(
    input: &CodexProposerInput<'_>,
    prompting_best_practices: &str,
) -> Value {
    let mut wins = Vec::new();
    let mut losses = Vec::new();
    let proposer_examples = proposer_examples_read_model(input);
    let proposal_policy = proposer_policy_text(input);
    for row in proposer_examples.as_array().into_iter().flatten() {
        let sample = proposer_example_compact(row);
        if row.get("reward").and_then(Value::as_f64).unwrap_or(0.0) >= 1.0 {
            wins.push(sample);
        } else {
            losses.push(sample);
        }
    }
    wins.truncate(20);
    losses.truncate(20);
    json!({
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "sample_winning_traces": wins,
        "sample_losing_traces": losses,
        "wins": wins,
        "losses": losses,
        "prompting_best_practices": prompting_best_practices,
        "instructions": [
            "Read prompting_best_practices.md before diagnosing prompt changes.",
            "Classify likely edits using the shared typology: premise, context, task_priority, core_task_description, heuristics, constraints, rules, input_description, output_description.",
            "Use sampled wins and losses to drive substantive prompt changes — not paraphrases of the seed.",
            "At most one proposal may be conservative. The others must be very ambitious task-specific changes that attack named top failure clusters and could substantially outperform the parent.",
            proposal_policy,
            "Across proposals, explore different strategies (structural rewrite, few-shot examples, role priming, label-disambiguation table, terse contract). Distinct proposals should be genuinely distinct."
        ],
    })
}

fn thread_start_params(input: &CodexProposerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the GEPA workspace proposer. Work only inside this workspace.".to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn staleness_thread_start_params(input: &CodexStalenessReviewerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the GEPA FlashEvolve staleness reviewer. Work only inside this workspace."
                .to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn turn_start_params(input: &CodexProposerInput<'_>, model: &str) -> Result<Value> {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": proposer_instructions(input)?,
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(service_tier) = non_empty(input.config.proposer.service_tier.as_deref()) {
        params.insert(
            "serviceTier".to_string(),
            Value::String(service_tier.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Ok(Value::Object(params))
}

fn staleness_turn_start_params(input: &CodexStalenessReviewerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": staleness_reviewer_instructions(input),
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(service_tier) = non_empty(input.config.proposer.service_tier.as_deref()) {
        params.insert(
            "serviceTier".to_string(),
            Value::String(service_tier.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Value::Object(params)
}

fn proposer_instructions(input: &CodexProposerInput<'_>) -> Result<String> {
    let context = proposer_prompt_context(input);
    let proposal_policy = proposer_policy_text(input);
    let best_practices = resolved_prompting_best_practices(input)?;
    let best_practices = best_practices.trim();
    Ok(format!(
        "{context}\n\n\
         Prompting best practices:\n\
         {best_practices}\n\n\
         Read README.md, prompting_best_practices.md, proposal/PROPOSAL_SCHEMA.md, and the allow-listed state files named there.\n\
         Start with state/proposer_metadata.json, state/proposer_failure_summary.json, state/proposer_repair_hints.json, and state/proposer_examples.json.\n\
         Use shell/Python/JQ tools to inspect reflection wins/losses and the parent payload before editing proposal/manifest.json.\n\
         Propose exactly {} prompt candidates for generation {}.\n\
         Use only these target modules: {}.\n\
         Follow the Python GEPA workspace proposer style: diagnose the missing instruction type, reflect over wins and losses, then propose substantive prompt changes.\n\
         {proposal_policy}\n\
         At most one proposal may be conservative. The others must be very ambitious, task-specific updates that target named top failure clusters and are intended to substantially outperform the parent.\n\
         Shoot for large task-performance wins, not mild prompt polish. A safe clarification is only acceptable as the single conservative control.\n\
         Do not spend candidates on generic output-contract polish or parent paraphrases unless the dominant failures are output-format failures.\n\
         Across the requested proposals, explore genuinely different strategies (structural rewrite, few-shot examples, terse contract, label-table, role priming) rather than paraphrasing the seed or each other.\n\
         Write strict JSON to proposal/manifest.json using schema_version gepa_workspace_proposal_v3.\n\
         Include the required evidence block with reviewed files, parent/evidence summary, failure patterns, winning patterns, and example ids.\n\
         Do not print pseudo-tool calls. Use real file inspection and file editing.",
        input.config.gepa.proposals_per_generation,
        input.generation,
        input.config.candidate.target_modules.join(", ")
    ))
}

fn staleness_reviewer_instructions(input: &CodexStalenessReviewerInput<'_>) -> String {
    format!(
        "Read README.md, review/VERDICT_SCHEMA.md, and state/staleness_review_request.json. \
         Decide whether the stale FlashEvolve work should be accepted, discarded, or patched. \
         Use only these target modules when patching: {}. \
         Write the verdict as strict JSON to review/verdict.json. Do not print pseudo-tool calls; inspect files and edit the verdict file.",
        input.config.candidate.target_modules.join(", ")
    )
}

fn proposer_prompt_context(input: &CodexProposerInput<'_>) -> String {
    let rollouts = rollouts_read_model(input);
    let proposer_examples = proposer_examples_read_model(input);
    let proposer_failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    let metadata = proposer_metadata_read_model(
        input,
        &rollouts,
        &proposer_examples,
        &proposer_failure_summary,
    );
    let proposer_model = metadata
        .pointer("/proposer/model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let policy_model = metadata
        .pointer("/policy/model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_count = metadata
        .get("candidate_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let rollout_count = metadata
        .get("rollout_row_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let loss_count = metadata
        .get("loss_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let win_count = metadata
        .get("win_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let top_failures = prompt_top_failures(&metadata);
    format!(
        "GEPA proposer context\n\
         run_id: {}\n\
         generation: {}\n\
         parent: {}\n\
         policy_model: {}\n\
         proposer_model: {}\n\
         target_levers: {}\n\
         mutable_levers: {}\n\
         proposals_requested: {}\n\
         minibatch_size: {}\n\
         candidates_seen: {}\n\
         rollout_rows: {}\n\
         evidence: losses={} wins={}\n\
         workspace: {}\n\
         top_failures:\n\
         {}\n\
         read_first:\n\
         1. prompting_best_practices.md\n\
         2. state/proposer_metadata.json\n\
         3. state/proposer_failure_summary.json\n\
         4. state/proposer_repair_hints.json\n\
         5. state/proposer_examples.json\n\
         6. state/parent_payload.json\n\
         7. state/reflective_frames.json\n\
         8. state/reflector_input.json",
        input.config.run.run_id,
        input.generation,
        input.parent.candidate_id,
        policy_model,
        proposer_model,
        input.config.candidate.target_modules.join(", "),
        input.program.mutable_field_ids().join(", "),
        input.config.gepa.proposals_per_generation,
        input.config.gepa.minibatch_size,
        candidate_count,
        rollout_count,
        loss_count,
        win_count,
        input.workspace_dir.display(),
        top_failures
    )
}

fn prompt_top_failures(metadata: &Value) -> String {
    let items = metadata
        .get("top_failures")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if items.is_empty() {
        return "         - none observed yet".to_string();
    }
    items
        .into_iter()
        .take(3)
        .map(|item| {
            format!(
                "         - {}: {}",
                item.get("key").and_then(Value::as_str).unwrap_or("unknown"),
                item.get("count").and_then(Value::as_u64).unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn sandbox_policy_for_mode(mode: &str) -> Value {
    match mode {
        "danger-full-access" => json!({"type": "dangerFullAccess"}),
        "read-only" => {
            json!({"type": "readOnly", "access": {"type": "fullAccess"}, "networkAccess": true})
        }
        "workspace-write" => {
            json!({"type": "workspaceWrite", "readOnlyAccess": {"type": "fullAccess"}, "networkAccess": true})
        }
        _ => Value::String(mode.to_string()),
    }
}

fn read_manifest(workspace_dir: &Path) -> Result<Value> {
    let path = workspace_dir.join("proposal").join("manifest.json");
    let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
    if text.trim().is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer wrote an empty manifest: {}",
            path.display()
        )));
    }
    match serde_json::from_str(&text) {
        Ok(value) => normalize_manifest_contract(value, &path),
        Err(original_error) => {
            let repaired = join_adjacent_json_strings(&text);
            if repaired != text {
                let value = serde_json::from_str(&repaired).map_err(|_| original_error)?;
                write_text(&path, &repaired)?;
                return normalize_manifest_contract(value, &path);
            }
            if let Some(value) = last_json_value_from_stream(&text) {
                write_json(&path, &value)?;
                return normalize_manifest_contract(value, &path);
            }
            Err(original_error.into())
        }
    }
}

fn read_json_value(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    Ok(serde_json::from_str(&text)?)
}

fn normalize_manifest_contract(mut manifest: Value, path: &Path) -> Result<Value> {
    let Some(object) = manifest.as_object_mut() else {
        return Ok(manifest);
    };
    let schema_version = object
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let has_proposals = object
        .get("proposals")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    if schema_version.is_empty() && has_proposals {
        object.insert(
            "schema_version".to_string(),
            Value::String(GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION.to_string()),
        );
        write_json(path, &manifest)?;
    }
    Ok(manifest)
}

fn join_adjacent_json_strings(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < chars.len() {
        let ch = chars[index];
        out.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
                let mut next = index + 1;
                while next < chars.len() && chars[next].is_whitespace() {
                    next += 1;
                }
                if next < chars.len() && chars[next] == '"' {
                    out.pop();
                    index = next;
                }
            }
        } else if ch == '"' {
            in_string = true;
        }
        index += 1;
    }
    out
}

fn last_json_value_from_stream(input: &str) -> Option<Value> {
    let stream = serde_json::Deserializer::from_str(input).into_iter::<Value>();
    let mut count = 0usize;
    let mut last = None;
    for value in stream {
        let value = value.ok()?;
        count += 1;
        last = Some(value);
    }
    if count > 1 {
        last
    } else {
        None
    }
}

fn proposals_from_manifest(manifest: &Value) -> Result<Value> {
    validate_manifest_contract(manifest)?;
    let proposals = manifest
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if proposals.is_empty() {
        return Err(OptimizerError::Proposer(
            "codex app-server proposer manifest contained no proposals".to_string(),
        ));
    }
    Ok(Value::Array(proposals))
}

fn validate_manifest_contract(manifest: &Value) -> Result<()> {
    let schema_version = manifest
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema_version != GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer manifest schema_version={schema_version:?}; expected {GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION}"
        )));
    }
    let evidence = manifest
        .get("evidence")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "codex app-server proposer manifest omitted required evidence object".to_string(),
            )
        })?;
    for field in [
        "candidate_comparison",
        "failure_patterns",
        "winning_patterns",
        "example_ids_used",
    ] {
        let Some(value) = evidence.get(field) else {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence missing {field}"
            )));
        };
        // Pattern fields may legitimately be empty when the observed evidence is
        // one-sided: all losses have no winning pattern, and all wins have no
        // failure pattern. Require the fields to be present, but do not force
        // content that the rollouts cannot ground.
        if matches!(field, "failure_patterns" | "winning_patterns") {
            continue;
        }
        let has_content = match value {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => items.iter().any(|item| {
                item.as_str()
                    .map(|text| !text.trim().is_empty())
                    .unwrap_or(false)
            }),
            _ => false,
        };
        if !has_content {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence field {field} is empty"
            )));
        }
    }
    Ok(())
}

fn manifest_evidence_warnings(
    input: &CodexProposerInput<'_>,
    manifest: &Value,
    proposals: &Value,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let evidence = manifest.get("evidence").and_then(Value::as_object);
    let reviewed = evidence
        .and_then(|evidence| evidence.get("reviewed_files"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    for required_file in [
        "state/proposer_metadata.json",
        "state/proposer_failure_summary.json",
        "state/proposer_repair_hints.json",
        "state/proposer_examples.json",
        "state/parent_payload.json",
        "state/reflective_frames.json",
    ] {
        if !reviewed.contains(required_file) {
            warnings.push(format!(
                "evidence reviewed_files did not include {required_file}"
            ));
        }
    }

    let cited_examples = evidence
        .and_then(|evidence| evidence.get("example_ids_used"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    let proposer_examples = proposer_examples_read_model(input);
    let mut losing_examples = BTreeSet::new();
    let mut winning_examples = BTreeSet::new();
    for row in proposer_examples.as_array().into_iter().flatten() {
        let Some(example_id) = string_path(row, &["example_id"]) else {
            continue;
        };
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        if reward >= 1.0 {
            winning_examples.insert(example_id);
        } else {
            losing_examples.insert(example_id);
        }
    }
    if !losing_examples.is_empty() && cited_examples.is_disjoint(&losing_examples) {
        warnings
            .push("evidence example_ids_used did not cite any losing rollout examples".to_string());
    }
    if !winning_examples.is_empty() && cited_examples.is_disjoint(&winning_examples) {
        warnings.push(
            "evidence example_ids_used did not cite any winning rollout examples".to_string(),
        );
    }

    let failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    if let Some(dominant_failure_label) = dominant_failure_label(&failure_summary) {
        let evidence_text = evidence
            .map(|evidence| Value::Object(evidence.clone()).to_string())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let proposal_text = proposals.to_string().to_ascii_lowercase();
        let needle = dominant_failure_label.to_ascii_lowercase();
        if !evidence_text.contains(&needle) && !proposal_text.contains(&needle) {
            warnings.push(format!(
                "proposal did not mention dominant failure label {dominant_failure_label:?}"
            ));
        }
    }
    let proposal_text = proposals.to_string().to_ascii_lowercase();
    for row in proposer_examples.as_array().into_iter().flatten() {
        let Some(example_text) = string_path(row, &["text"]) else {
            continue;
        };
        let normalized = example_text.trim().to_ascii_lowercase();
        if normalized.len() >= 32 && proposal_text.contains(&normalized) {
            let example_id = string_path(row, &["example_id"]).unwrap_or_default();
            warnings.push(format!(
                "proposal appears to quote training example text from {example_id}; GEPA candidates should generalize rather than memorize exact queries"
            ));
            break;
        }
    }
    if literal_training_target_policy(input) == "forbid" {
        for row in proposer_examples.as_array().into_iter().flatten() {
            for field in ["expected", "prediction"] {
                let Some(literal) = string_path(row, &[field]) else {
                    continue;
                };
                let normalized = literal.trim().to_ascii_lowercase();
                if literal_target_is_specific(&normalized) && proposal_text.contains(&normalized) {
                    let example_id = string_path(row, &["example_id"]).unwrap_or_default();
                    warnings.push(format!(
                        "proposal appears to quote {field} target literal from {example_id} despite task policy forbidding literal training-target mappings"
                    ));
                    return warnings;
                }
            }
        }
    }
    warnings
}

fn dominant_failure_label(failure_summary: &Value) -> Option<String> {
    let key = failure_summary
        .get("label_confusions")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("key"))
        .and_then(Value::as_str)?;
    let label = key
        .split(" -> ")
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    if label.is_empty() || label == "unknown" {
        None
    } else {
        Some(label)
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    write_text(path, &format!("{text}\n"))
}

fn read_staleness_verdict_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => return Ok(value),
        Err(source) if !source.to_string().contains("trailing characters") => {
            return Err(OptimizerError::from(source));
        }
        Err(_) => {}
    }

    let mut values = Vec::new();
    for value in serde_json::Deserializer::from_str(&text).into_iter::<Value>() {
        values.push(value.map_err(OptimizerError::from)?);
    }
    values.into_iter().last().ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "staleness reviewer verdict file {} did not contain a JSON object",
            path.display()
        ))
    })
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).map_err(|source| OptimizerError::io(path, source))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
