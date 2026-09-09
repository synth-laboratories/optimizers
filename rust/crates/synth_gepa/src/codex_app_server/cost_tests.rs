use super::*;

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
