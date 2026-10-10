use super::*;

fn row(id: &str, cost_usd: Option<f64>) -> UsageLedgerRecord {
    UsageLedgerRecord::from_input(UsageLedgerInput {
        boundary: "provider",
        source_type: "call",
        source_id: id,
        candidate_id: None,
        evaluation_stage: None,
        model: None,
        provider: None,
        call_count: 1,
        usage: json!({}),
        cost_usd,
        metadata: Map::new(),
    })
}

#[test]
fn aggregate_is_null_if_any_provider_cost_is_unknown() {
    assert_eq!(reported_cost_from_usage_ledger(&[]), None);
    assert_eq!(reported_cost_from_usage_ledger(&[row("a", None)]), None);
    assert_eq!(
        reported_cost_from_usage_ledger(&[row("a", Some(0.12)), row("b", None)]),
        None
    );
    assert_eq!(
        reported_cost_from_usage_ledger(&[row("a", Some(0.0)), row("b", Some(0.12))]),
        Some(0.12)
    );
}

#[test]
fn reconciliation_counts_rollouts_but_prices_all_boundaries() {
    let mut rollout = row("rollout", Some(0.02));
    rollout.boundary = "container.rollout".into();
    rollout.source_type = "sensor_frame".into();
    let mut proposer = row("proposal", Some(0.1));
    proposer.boundary = "proposer.codex".into();
    proposer.source_type = "proposer_generation".into();
    let (rollouts, cost) = ledger_rollout_and_cost(&[rollout.clone(), proposer.clone()]);
    assert_eq!(rollouts, Some(1));
    assert!((cost.unwrap() - 0.12).abs() < 1e-9);
    assert_eq!(
        identities::reconciliation_status(1, rollouts, Some(0.12), cost).status,
        "aligned"
    );
    proposer.cost_usd = None;
    assert_eq!(
        ledger_rollout_and_cost(&[rollout, proposer]),
        (Some(1), None)
    );
    assert_eq!(ledger_rollout_and_cost(&[]), (None, None));
}

#[test]
fn repeated_receipts_are_priced_once_and_conflicts_stay_unknown() {
    let receipt = row("charge-a", Some(0.12));
    assert_eq!(
        reported_cost_from_usage_ledger(&[receipt.clone(), receipt.clone()]),
        Some(0.12)
    );
    assert_eq!(
        reported_cost_from_usage_ledger(&[receipt.clone(), row("charge-b", Some(0.12))]),
        Some(0.24)
    );
    for cost in [
        None,
        Some(0.13),
        Some(-1.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        let mut changed = receipt.clone();
        changed.cost_usd = cost;
        assert_eq!(
            reported_cost_from_usage_ledger(&[receipt.clone(), changed]),
            None
        );
    }
}

#[test]
fn actual_banking77_cursors_match_durable_and_provider_costs() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/banking77-replayed-usage.json"
    ))
    .unwrap();
    for arm in fixture["arms"].as_array().unwrap() {
        let observations: Vec<UsageLedgerRecord> =
            serde_json::from_value(arm["observations"].clone()).unwrap();
        let durable: Vec<UsageLedgerRecord> =
            serde_json::from_value(arm["durable"].clone()).unwrap();
        let provider = arm["provider_cost_usd"].as_f64().unwrap();
        assert_eq!(observations.len(), 69);
        assert_eq!(durable.len(), 67);
        assert!((reported_cost_from_usage_ledger(&observations).unwrap() - provider).abs() < 1e-9);
        assert!((reported_cost_from_usage_ledger(&durable).unwrap() - provider).abs() < 1e-9);
        // The previous plain fold reproduces the actual manifest overcount.
        let previous =
            fold_reported_cost(observations.iter().map(|record| record.cost_usd)).unwrap();
        assert!((previous - arm["previous_manifest_cost_usd"].as_f64().unwrap()).abs() < 1e-9);
        assert!(previous > provider);
    }
}

#[test]
fn configured_cost_budget_stops_on_unknown_receipt() {
    assert!(!reported_cost_budget_blocked(1.0, 0.0, &[]));
    assert!(reported_cost_budget_blocked(1.0, 0.0, &[row("a", None)]));
    assert!(!reported_cost_budget_blocked(
        1.0,
        0.12,
        &[row("a", Some(0.12))]
    ));
    assert!(reported_cost_budget_blocked(
        1.0,
        1.0,
        &[row("a", Some(1.0))]
    ));
    assert!(!reported_cost_budget_blocked(0.0, 0.0, &[row("a", None)]));
}

#[test]
fn completed_effect_uses_reserved_ceiling_when_provider_omits_cost() {
    assert_eq!(
        settled_effect_cost("completed", None, Some(0.05)),
        Some(0.05)
    );
    assert_eq!(
        settled_effect_cost("completed", Some(0.02), Some(0.05)),
        Some(0.02)
    );
    assert_eq!(settled_effect_cost("failed", None, Some(0.05)), None);
}
