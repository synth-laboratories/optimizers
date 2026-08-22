use super::*;

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
