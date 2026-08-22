use super::*;

fn row(id: &str, cost_usd: Option<f64>) -> UsageLedgerRecord {
    UsageLedgerRecord::from_input(UsageLedgerInput {
        boundary: "provider", source_type: "call", source_id: id, candidate_id: None,
        evaluation_stage: None, model: None, provider: None, call_count: 1,
        usage: json!({}), cost_usd, metadata: Map::new(),
    })
}

#[test]
fn aggregate_is_null_if_any_provider_cost_is_unknown() {
    assert_eq!(reported_cost_from_usage_ledger(&[]), None);
    assert_eq!(reported_cost_from_usage_ledger(&[row("a", None)]), None);
    assert_eq!(reported_cost_from_usage_ledger(&[row("a", Some(0.12)), row("b", None)]), None);
    assert_eq!(reported_cost_from_usage_ledger(&[row("a", Some(0.0)), row("b", Some(0.12))]), Some(0.12));
}

#[test]
fn configured_cost_budget_stops_on_unknown_receipt() {
    assert!(!reported_cost_budget_blocked(1.0, 0.0, &[]));
    assert!(reported_cost_budget_blocked(1.0, 0.0, &[row("a", None)]));
    assert!(!reported_cost_budget_blocked(1.0, 0.12, &[row("a", Some(0.12))]));
    assert!(reported_cost_budget_blocked(1.0, 1.0, &[row("a", Some(1.0))]));
    assert!(!reported_cost_budget_blocked(0.0, 0.0, &[row("a", None)]));
}

#[test]
fn completed_effect_uses_reserved_ceiling_when_provider_omits_cost() {
    assert_eq!(settled_effect_cost("completed", None, Some(0.05)), Some(0.05));
    assert_eq!(settled_effect_cost("completed", Some(0.02), Some(0.05)), Some(0.02));
    assert_eq!(settled_effect_cost("failed", None, Some(0.05)), None);
}
