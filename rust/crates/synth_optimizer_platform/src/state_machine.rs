use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;

use crate::error::{OptimizerError, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerRunState {
    Created,
    Initializing,
    Ready,
    Proposing,
    TrialQueueing,
    RolloutQueueing,
    RolloutRunning,
    Annotating,
    Verifying,
    Evaluating,
    WaitingHumanInput,
    Paused,
    Checkpointing,
    Restoring,
    Completed,
    Failed,
    Cancelled,
}

impl OptimizerRunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Initializing => "initializing",
            Self::Ready => "ready",
            Self::Proposing => "proposing",
            Self::TrialQueueing => "trial_queueing",
            Self::RolloutQueueing => "rollout_queueing",
            Self::RolloutRunning => "rollout_running",
            Self::Annotating => "annotating",
            Self::Verifying => "verifying",
            Self::Evaluating => "evaluating",
            Self::WaitingHumanInput => "waiting_human_input",
            Self::Paused => "paused",
            Self::Checkpointing => "checkpointing",
            Self::Restoring => "restoring",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerTransitionTrigger {
    RunStarted,
    ContainerReady,
    ProposerStarted,
    ProposerFinished,
    RolloutsQueued,
    RolloutsStarted,
    RolloutsFinished,
    AnnotationStarted,
    AnnotationFinished,
    VerificationStarted,
    VerificationFinished,
    EvaluationFinished,
    CheckpointStarted,
    CheckpointFinished,
    RestoreStarted,
    RestoreFinished,
    PauseRequested,
    ResumeRequested,
    CancelRequested,
    FailureRaised,
    RunCompleted,
}

impl OptimizerTransitionTrigger {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RunStarted => "run_started",
            Self::ContainerReady => "container_ready",
            Self::ProposerStarted => "proposer_started",
            Self::ProposerFinished => "proposer_finished",
            Self::RolloutsQueued => "rollouts_queued",
            Self::RolloutsStarted => "rollouts_started",
            Self::RolloutsFinished => "rollouts_finished",
            Self::AnnotationStarted => "annotation_started",
            Self::AnnotationFinished => "annotation_finished",
            Self::VerificationStarted => "verification_started",
            Self::VerificationFinished => "verification_finished",
            Self::EvaluationFinished => "evaluation_finished",
            Self::CheckpointStarted => "checkpoint_started",
            Self::CheckpointFinished => "checkpoint_finished",
            Self::RestoreStarted => "restore_started",
            Self::RestoreFinished => "restore_finished",
            Self::PauseRequested => "pause_requested",
            Self::ResumeRequested => "resume_requested",
            Self::CancelRequested => "cancel_requested",
            Self::FailureRaised => "failure_raised",
            Self::RunCompleted => "run_completed",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerTransition {
    pub run_id: String,
    pub from: OptimizerRunState,
    pub to: OptimizerRunState,
    pub trigger: OptimizerTransitionTrigger,
    pub message: String,
    pub at: String,
    #[serde(default)]
    pub details: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerStateMachine {
    pub run_id: String,
    pub state: OptimizerRunState,
    #[serde(default)]
    pub history: Vec<OptimizerTransition>,
}

impl OptimizerStateMachine {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            state: OptimizerRunState::Created,
            history: Vec::new(),
        }
    }

    pub fn state(&self) -> OptimizerRunState {
        self.state
    }

    pub fn transition(
        &mut self,
        to: OptimizerRunState,
        trigger: OptimizerTransitionTrigger,
        message: impl Into<String>,
        details: Map<String, Value>,
    ) -> Result<OptimizerTransition> {
        let from = self.state;
        if !is_valid_transition(from, to, trigger) {
            return Err(OptimizerError::StateTransition {
                from: from.as_str().to_string(),
                to: to.as_str().to_string(),
                trigger: trigger.as_str().to_string(),
            });
        }
        let transition = OptimizerTransition {
            run_id: self.run_id.clone(),
            from,
            to,
            trigger,
            message: message.into(),
            at: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
            details,
        };
        self.state = to;
        self.history.push(transition.clone());
        Ok(transition)
    }
}

fn is_valid_transition(
    from: OptimizerRunState,
    to: OptimizerRunState,
    trigger: OptimizerTransitionTrigger,
) -> bool {
    if from.is_terminal() {
        return false;
    }
    if matches!(to, OptimizerRunState::Failed) {
        return matches!(trigger, OptimizerTransitionTrigger::FailureRaised);
    }
    if matches!(to, OptimizerRunState::Cancelled) {
        return matches!(trigger, OptimizerTransitionTrigger::CancelRequested);
    }
    matches!(
        (from, to),
        (OptimizerRunState::Created, OptimizerRunState::Initializing)
            | (OptimizerRunState::Created, OptimizerRunState::Restoring)
            | (OptimizerRunState::Initializing, OptimizerRunState::Ready)
            | (OptimizerRunState::Restoring, OptimizerRunState::Ready)
            | (OptimizerRunState::Ready, OptimizerRunState::Proposing)
            | (OptimizerRunState::Ready, OptimizerRunState::RolloutQueueing)
            | (OptimizerRunState::Ready, OptimizerRunState::RolloutRunning)
            | (OptimizerRunState::Ready, OptimizerRunState::Evaluating)
            | (OptimizerRunState::Ready, OptimizerRunState::Checkpointing)
            | (OptimizerRunState::Ready, OptimizerRunState::Completed)
            | (
                OptimizerRunState::Proposing,
                OptimizerRunState::TrialQueueing
            )
            | (
                OptimizerRunState::Proposing,
                OptimizerRunState::RolloutQueueing
            )
            | (OptimizerRunState::Proposing, OptimizerRunState::Evaluating)
            | (OptimizerRunState::Proposing, OptimizerRunState::Ready)
            | (
                OptimizerRunState::TrialQueueing,
                OptimizerRunState::RolloutQueueing
            )
            | (
                OptimizerRunState::TrialQueueing,
                OptimizerRunState::Proposing
            )
            | (
                OptimizerRunState::RolloutQueueing,
                OptimizerRunState::RolloutRunning
            )
            | (
                OptimizerRunState::RolloutQueueing,
                OptimizerRunState::Evaluating
            )
            | (OptimizerRunState::RolloutQueueing, OptimizerRunState::Ready)
            | (
                OptimizerRunState::RolloutRunning,
                OptimizerRunState::RolloutQueueing
            )
            | (
                OptimizerRunState::RolloutRunning,
                OptimizerRunState::Annotating
            )
            | (
                OptimizerRunState::RolloutRunning,
                OptimizerRunState::Verifying
            )
            | (
                OptimizerRunState::RolloutRunning,
                OptimizerRunState::Evaluating
            )
            | (OptimizerRunState::Annotating, OptimizerRunState::Verifying)
            | (OptimizerRunState::Annotating, OptimizerRunState::Evaluating)
            | (
                OptimizerRunState::Annotating,
                OptimizerRunState::RolloutQueueing
            )
            | (OptimizerRunState::Verifying, OptimizerRunState::Evaluating)
            | (
                OptimizerRunState::Verifying,
                OptimizerRunState::RolloutQueueing
            )
            | (OptimizerRunState::Evaluating, OptimizerRunState::Ready)
            | (OptimizerRunState::Evaluating, OptimizerRunState::Proposing)
            | (
                OptimizerRunState::Evaluating,
                OptimizerRunState::RolloutQueueing
            )
            | (
                OptimizerRunState::Evaluating,
                OptimizerRunState::Checkpointing
            )
            | (OptimizerRunState::Evaluating, OptimizerRunState::Completed)
            | (OptimizerRunState::Checkpointing, OptimizerRunState::Ready)
            | (
                OptimizerRunState::Checkpointing,
                OptimizerRunState::Proposing
            )
            | (
                OptimizerRunState::Checkpointing,
                OptimizerRunState::Completed
            )
            | (
                OptimizerRunState::WaitingHumanInput,
                OptimizerRunState::Ready
            )
            | (
                OptimizerRunState::WaitingHumanInput,
                OptimizerRunState::Paused
            )
            | (
                OptimizerRunState::WaitingHumanInput,
                OptimizerRunState::Proposing
            )
            | (OptimizerRunState::Paused, OptimizerRunState::Ready)
    )
}
