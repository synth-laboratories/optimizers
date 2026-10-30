//! Unpaid regression replay of the SYN-5275 r12 proposer settlement.
//!
//! Retained checkpoint `gepa_syn5230_installed_20261029_r12`
//! (object `02b707e593f61d96…`, `checkpoint_workspace_db`) holds budget commit
//! `budget_commit_b80d5a777b0c6cd3` for the chat_completions proposer effect
//! `effect_715ce0f3e4f5c836` with prompt/completion/total tokens 22149/2840/24989
//! and `cost_usd = NULL`. OpenRouter billed that call as `usage.cost`; the old
//! reader only looked at `cost_usd`. These tests replay that exact usage shape
//! through normalization, the runtime reader and settlement with no provider call.

use super::*;
use crate::runtime::proposer_reported_cost_usd;
use crate::wire_cost::{COST_UNKNOWN_REASON_FIELD, PROVIDER_COST_INVALID};

const R12_MODEL: &str = "anthropic/claude-sonnet-4.6";
const R12_PROMPT_TOKENS: u64 = 22_149;
const R12_COMPLETION_TOKENS: u64 = 2_840;
const R12_TOTAL_TOKENS: u64 = 24_989;
const R12_RESERVED_CEILING_USD: f64 = 0.25;

fn openrouter_config() -> SynthOptimizerConfig {
    let mut config = SynthOptimizerConfig::default();
    config.proposer.provider = "openrouter".to_string();
    config.proposer.backend = "chat_completions".to_string();
    config
}

fn r12_wire_usage(cost: Value) -> Value {
    let mut usage = json!({
        "prompt_tokens": R12_PROMPT_TOKENS,
        "completion_tokens": R12_COMPLETION_TOKENS,
        "total_tokens": R12_TOTAL_TOKENS,
    });
    if !cost.is_null() {
        usage["cost"] = cost;
    }
    usage
}

fn replay_settlement(cost: Value) -> (Value, Option<f64>, Option<f64>) {
    let usage = codex_app_server::normalize_proposer_usage(
        &openrouter_config(),
        R12_MODEL,
        r12_wire_usage(cost),
    );
    let reported = proposer_reported_cost_usd(&usage);
    let settled = settled_effect_cost("completed", reported, Some(R12_RESERVED_CEILING_USD));
    (usage, reported, settled)
}

#[test]
fn r12_numeric_openrouter_cost_settles_as_known_provider_cost() {
    let (usage, reported, settled) = replay_settlement(json!(0.105_183));
    assert_eq!(usage.get("cost_usd"), Some(&json!(0.105_183)));
    assert_eq!(reported, Some(0.105_183));
    assert_eq!(
        settled,
        Some(0.105_183),
        "settlement must use the billed amount"
    );
    assert_eq!(usage.get("provider"), Some(&json!("openrouter")));
}

#[test]
fn r12_string_openrouter_cost_settles_as_known_provider_cost() {
    let (usage, reported, _) = replay_settlement(json!("0.105183"));
    assert_eq!(usage.get("cost_usd"), Some(&json!(0.105_183)));
    assert_eq!(reported, Some(0.105_183));
}

#[test]
fn r12_real_zero_cost_is_known_not_unknown() {
    let (usage, reported, settled) = replay_settlement(json!(0));
    assert_eq!(usage.get("cost_usd"), Some(&json!(0.0)));
    assert_eq!(reported, Some(0.0));
    assert_eq!(
        settled,
        Some(0.0),
        "a billed zero must not fall back to the ceiling"
    );
}

#[test]
fn r12_absent_cost_stays_unknown_and_hard_budget_fails_closed() {
    let (usage, reported, settled) = replay_settlement(Value::Null);
    assert!(usage.get("cost_usd").is_none());
    assert!(usage.get(COST_UNKNOWN_REASON_FIELD).is_none());
    assert_eq!(reported, None);
    assert_eq!(settled, Some(R12_RESERVED_CEILING_USD));
    let ledger = [UsageLedgerRecord::from_input(UsageLedgerInput {
        boundary: "proposer.codex",
        source_type: "proposer_generation",
        source_id: "effect_715ce0f3e4f5c836",
        candidate_id: None,
        evaluation_stage: None,
        model: Some(R12_MODEL),
        provider: Some("openrouter"),
        call_count: 1,
        usage,
        cost_usd: reported,
        metadata: Map::new(),
    })];
    assert!(reported_cost_budget_blocked(2.0, 0.0, &ledger));
}

#[test]
fn r12_invalid_cost_stays_unknown_with_typed_reason() {
    for invalid in [
        json!(-0.01),
        json!("NaN"),
        json!("inf"),
        json!("n/a"),
        json!(true),
    ] {
        let (usage, reported, _) = replay_settlement(invalid.clone());
        assert!(usage.get("cost_usd").is_none(), "{invalid}");
        assert_eq!(
            usage.get(COST_UNKNOWN_REASON_FIELD),
            Some(&json!(PROVIDER_COST_INVALID)),
            "{invalid}"
        );
        assert_eq!(reported, None, "{invalid}");
    }
}
