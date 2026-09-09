use super::*;

#[test]
fn every_terminal_path_records_storage_before_sealing() {
    // Guard all four entry points, including the legacy monolithic runner.
    // An enrichment lane on the raw feed cannot change canonical event order.
    let source = include_str!("lib.rs");
    for (function, terminal) in [
        (
            "terminalize_gepa_run_state(",
            "terminal_event_type,\n        message,",
        ),
        ("finalize_completed_gepa_run(", "\"gepa.run.finished\","),
        (
            "execute_gepa_monolithic_with_options(",
            "\"gepa.run.finished\",",
        ),
        (
            "fail_gepa_run_and_return<T>(",
            "terminal_event_type,\n        input.message,",
        ),
    ] {
        let body = source.split_once(&format!("fn {function}")).unwrap().1;
        let body = body.split("\nfn ").next().unwrap();
        let snapshot = body.find("record_terminal_storage_snapshot(").unwrap();
        let seal = body.find(terminal).unwrap();
        assert!(
            snapshot < seal,
            "{function} appends storage after the terminal event"
        );
        assert_eq!(body.matches("record_terminal_storage_snapshot(").count(), 1);
    }
}

#[test]
fn budget_exhaustion_is_a_typed_terminal_run_loop_error() {
    let error = OptimizerError::BudgetExceeded {
        run_id: "gepa_budget_test".to_string(),
        limit: "max_cost_usd".to_string(),
        requested: "0.05".to_string(),
        available: "0.04".to_string(),
    };
    assert_eq!(
        terminal_message_for_run_loop_error(&error),
        Some("GEPA budget exhausted")
    );
    assert_eq!(error.error_code(), "synth_optimizer_budget_exceeded");
}

#[test]
fn unrelated_orchestration_errors_are_not_reclassified_as_budget_terminal() {
    assert_eq!(
        terminal_message_for_run_loop_error(&OptimizerError::Container(
            "provider unavailable".to_string()
        )),
        None
    );
}

#[test]
fn proposer_runtime_jobs_fail_closed_without_replay() {
    let policy = runtime_effect_retry_policy(&OptimizerJobKind::Proposer);
    assert_eq!(policy.max_attempts, 1);
    assert_eq!(policy.backoff_seconds, 0);
    assert!(policy.retryable_failure_types.is_empty());
}

#[test]
fn unrelated_runtime_jobs_keep_the_fail_closed_default() {
    let policy = runtime_effect_retry_policy(&OptimizerJobKind::Annotation);
    assert_eq!(policy.max_attempts, 1);
    assert!(policy.retryable_failure_types.is_empty());
}
