use super::*;

#[test]
fn openai_chat_completions_use_the_current_token_limit_field() {
    assert_eq!(
        chat_completions_token_limit_field("openai"),
        "max_completion_tokens"
    );
    assert_eq!(
        chat_completions_token_limit_field("openrouter"),
        "max_tokens"
    );
    assert_eq!(chat_completions_token_limit_field("deepseek"), "max_tokens");
}

#[test]
fn chatgpt_proposer_emits_explicit_zero_incremental_api_cost() {
    let mut config = SynthOptimizerConfig::default();
    config.proposer.auth_mode = "chatgpt".to_string();
    let usage = normalize_proposer_usage(
        &config,
        "gpt-5.6-luna",
        json!({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}),
    );
    assert_eq!(usage.get("cost_usd"), Some(&json!(0.0)));
    assert_eq!(
        usage.get("cost_source"),
        Some(&json!("chatgpt_subscription_no_incremental_api_charge"))
    );
    assert_eq!(usage.get("provider"), Some(&json!("chatgpt_subscription")));
}

#[test]
fn chatgpt_proposer_preserves_an_explicit_cost_receipt() {
    let mut config = SynthOptimizerConfig::default();
    config.proposer.auth_mode = "chatgpt".to_string();
    let usage = normalize_proposer_usage(
        &config,
        "gpt-5.6-luna",
        json!({"cost_usd": 0.25, "cost_source": "provider"}),
    );
    assert_eq!(usage.get("cost_usd"), Some(&json!(0.25)));
    assert_eq!(usage.get("cost_source"), Some(&json!("provider")));
}
