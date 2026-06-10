use std::time::{Duration, Instant};

use crate::OptimizerError;

/// Separates overall turn budget from per-read stall detection on JSON-RPC transports.
pub(crate) struct JsonRpcReadWindow {
    overall_deadline: Instant,
    message_stall_timeout: Duration,
}

impl JsonRpcReadWindow {
    pub(crate) fn new(overall_timeout: Duration, message_stall_timeout: Duration) -> Self {
        Self {
            overall_deadline: Instant::now() + overall_timeout,
            message_stall_timeout,
        }
    }

    pub(crate) fn per_read_deadline(&self) -> Instant {
        Instant::now()
            .checked_add(self.message_stall_timeout)
            .unwrap_or(self.overall_deadline)
            .min(self.overall_deadline)
    }

    pub(crate) fn overall_expired(&self) -> bool {
        Instant::now() >= self.overall_deadline
    }

    pub(crate) fn stall_expired_before_overall(&self, read_deadline: Instant) -> bool {
        !self.overall_expired() && Instant::now() >= read_deadline
    }

    pub(crate) fn overall_timeout_error(prefix: &str, stderr_tail: &str) -> OptimizerError {
        OptimizerError::Proposer(format!(
            "{prefix} timed out waiting for response{stderr_tail}"
        ))
    }

    pub(crate) fn stall_timeout_error(
        prefix: &str,
        waiting_for: &str,
        message_stall_timeout: Duration,
        stderr_tail: &str,
    ) -> OptimizerError {
        OptimizerError::Proposer(format!(
            "{prefix} stalled: no JSON-RPC progress for {}s while waiting for {waiting_for}{stderr_tail}",
            message_stall_timeout.as_secs(),
        ))
    }

    pub(crate) fn map_read_error(
        &self,
        prefix: &str,
        waiting_for: &str,
        read_deadline: Instant,
        stderr_tail: &str,
        source: OptimizerError,
    ) -> OptimizerError {
        if self.stall_expired_before_overall(read_deadline) {
            return Self::stall_timeout_error(
                prefix,
                waiting_for,
                self.message_stall_timeout,
                stderr_tail,
            );
        }
        if self.overall_expired() {
            return Self::overall_timeout_error(prefix, stderr_tail);
        }
        source
    }
}
