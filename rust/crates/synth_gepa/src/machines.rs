use synth_optimizer_platform::StateMachineEntity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateState {
    Registered,
    MinibatchEvaluating,
    MinibatchEvaluated,
    AcceptedMinibatch,
    RejectedMinibatch,
    FullTrainEvaluating,
    FullTrainEvaluated,
    Accepted,
    RejectedFullTrain,
    DeferredBudget,
    HeldoutEvaluating,
    HeldoutScored,
    Archived,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateTrigger {
    Registered,
    EvaluationStarted,
    EvaluationFinished,
    MinibatchAccepted,
    MinibatchRejected,
    FullTrainAccepted,
    FullTrainRejected,
    DeferredBudget,
    HeldoutStarted,
    HeldoutFinished,
    Archived,
}

pub struct CandidateEntity;

impl StateMachineEntity for CandidateEntity {
    type State = CandidateState;
    type Trigger = CandidateTrigger;

    const ENTITY_TYPE: &'static str = "candidate";

    fn initial_state() -> Self::State {
        CandidateState::Registered
    }

    fn state_name(state: Self::State) -> &'static str {
        match state {
            CandidateState::Registered => "registered",
            CandidateState::MinibatchEvaluating => "minibatch_evaluating",
            CandidateState::MinibatchEvaluated => "minibatch_evaluated",
            CandidateState::AcceptedMinibatch => "accepted_minibatch",
            CandidateState::RejectedMinibatch => "rejected_minibatch",
            CandidateState::FullTrainEvaluating => "full_train_evaluating",
            CandidateState::FullTrainEvaluated => "full_train_evaluated",
            CandidateState::Accepted => "accepted",
            CandidateState::RejectedFullTrain => "rejected_full_train",
            CandidateState::DeferredBudget => "deferred_budget",
            CandidateState::HeldoutEvaluating => "heldout_evaluating",
            CandidateState::HeldoutScored => "heldout_scored",
            CandidateState::Archived => "archived",
        }
    }

    fn state_from_name(name: &str) -> Option<Self::State> {
        Some(match name {
            "registered" => CandidateState::Registered,
            "minibatch_evaluating" => CandidateState::MinibatchEvaluating,
            "minibatch_evaluated" => CandidateState::MinibatchEvaluated,
            "accepted_minibatch" => CandidateState::AcceptedMinibatch,
            "rejected_minibatch" => CandidateState::RejectedMinibatch,
            "full_train_evaluating" => CandidateState::FullTrainEvaluating,
            "full_train_evaluated" => CandidateState::FullTrainEvaluated,
            "accepted" => CandidateState::Accepted,
            "rejected_full_train" => CandidateState::RejectedFullTrain,
            "deferred_budget" => CandidateState::DeferredBudget,
            "heldout_evaluating" => CandidateState::HeldoutEvaluating,
            "heldout_scored" => CandidateState::HeldoutScored,
            "archived" => CandidateState::Archived,
            _ => return None,
        })
    }

    fn trigger_name(trigger: Self::Trigger) -> &'static str {
        match trigger {
            CandidateTrigger::Registered => "registered",
            CandidateTrigger::EvaluationStarted => "evaluation_started",
            CandidateTrigger::EvaluationFinished => "evaluation_finished",
            CandidateTrigger::MinibatchAccepted => "minibatch_accepted",
            CandidateTrigger::MinibatchRejected => "minibatch_rejected",
            CandidateTrigger::FullTrainAccepted => "full_train_accepted",
            CandidateTrigger::FullTrainRejected => "full_train_rejected",
            CandidateTrigger::DeferredBudget => "deferred_budget",
            CandidateTrigger::HeldoutStarted => "heldout_started",
            CandidateTrigger::HeldoutFinished => "heldout_finished",
            CandidateTrigger::Archived => "archived",
        }
    }

    fn transition_allowed(from: Self::State, to: Self::State, trigger: Self::Trigger) -> bool {
        use CandidateState as State;
        use CandidateTrigger as Trigger;

        if matches!(from, State::Archived) {
            return false;
        }
        if from == to {
            return true;
        }
        matches!(
            (from, to, trigger),
            (
                State::Registered,
                State::MinibatchEvaluating,
                Trigger::EvaluationStarted,
            ) | (
                State::Registered,
                State::FullTrainEvaluating,
                Trigger::EvaluationStarted,
            ) | (
                State::Registered,
                State::DeferredBudget,
                Trigger::DeferredBudget,
            ) | (State::Registered, State::Archived, Trigger::Archived)
                | (
                    State::MinibatchEvaluating,
                    State::MinibatchEvaluated,
                    Trigger::EvaluationFinished,
                )
                | (
                    State::MinibatchEvaluated,
                    State::AcceptedMinibatch,
                    Trigger::MinibatchAccepted,
                )
                | (
                    State::MinibatchEvaluated,
                    State::RejectedMinibatch,
                    Trigger::MinibatchRejected,
                )
                | (
                    State::MinibatchEvaluated,
                    State::DeferredBudget,
                    Trigger::DeferredBudget,
                )
                | (
                    State::AcceptedMinibatch,
                    State::FullTrainEvaluating,
                    Trigger::EvaluationStarted,
                )
                | (
                    State::AcceptedMinibatch,
                    State::DeferredBudget,
                    Trigger::DeferredBudget,
                )
                | (
                    State::FullTrainEvaluating,
                    State::FullTrainEvaluated,
                    Trigger::EvaluationFinished,
                )
                | (
                    State::FullTrainEvaluated,
                    State::Accepted,
                    Trigger::FullTrainAccepted,
                )
                | (
                    State::FullTrainEvaluated,
                    State::RejectedFullTrain,
                    Trigger::FullTrainRejected,
                )
                | (
                    State::Accepted,
                    State::HeldoutEvaluating,
                    Trigger::HeldoutStarted,
                )
                | (
                    State::HeldoutEvaluating,
                    State::HeldoutScored,
                    Trigger::HeldoutFinished,
                )
                | (
                    State::HeldoutScored,
                    State::Accepted,
                    Trigger::FullTrainAccepted,
                )
                | (State::DeferredBudget, State::Archived, Trigger::Archived,)
                | (State::RejectedMinibatch, State::Archived, Trigger::Archived,)
                | (State::RejectedFullTrain, State::Archived, Trigger::Archived,)
                | (State::Accepted, State::Archived, Trigger::Archived)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutState {
    Queued,
    Running,
    Completed,
    Failed,
    Cached,
    Cancelled,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutTrigger {
    Scheduled,
    Started,
    Succeeded,
    Failed,
    CacheHit,
    Cancelled,
}

pub struct RolloutEntity;

impl StateMachineEntity for RolloutEntity {
    type State = RolloutState;
    type Trigger = RolloutTrigger;

    const ENTITY_TYPE: &'static str = "rollout";

    fn initial_state() -> Self::State {
        RolloutState::Queued
    }

    fn state_name(state: Self::State) -> &'static str {
        match state {
            RolloutState::Queued => "queued",
            RolloutState::Running => "running",
            RolloutState::Completed => "completed",
            RolloutState::Failed => "failed",
            RolloutState::Cached => "cached",
            RolloutState::Cancelled => "cancelled",
        }
    }

    fn state_from_name(name: &str) -> Option<Self::State> {
        Some(match name {
            "queued" => RolloutState::Queued,
            "running" => RolloutState::Running,
            "completed" => RolloutState::Completed,
            "failed" => RolloutState::Failed,
            "cached" => RolloutState::Cached,
            "cancelled" => RolloutState::Cancelled,
            _ => return None,
        })
    }

    fn trigger_name(trigger: Self::Trigger) -> &'static str {
        match trigger {
            RolloutTrigger::Scheduled => "scheduled",
            RolloutTrigger::Started => "started",
            RolloutTrigger::Succeeded => "succeeded",
            RolloutTrigger::Failed => "failed",
            RolloutTrigger::CacheHit => "cache_hit",
            RolloutTrigger::Cancelled => "cancelled",
        }
    }

    fn transition_allowed(from: Self::State, to: Self::State, trigger: Self::Trigger) -> bool {
        use RolloutState as State;
        use RolloutTrigger as Trigger;

        if matches!(
            from,
            State::Completed | State::Failed | State::Cached | State::Cancelled
        ) {
            return false;
        }
        if from == to {
            return true;
        }
        matches!(
            (from, to, trigger),
            (State::Queued, State::Running, Trigger::Started)
                | (State::Queued, State::Cached, Trigger::CacheHit)
                | (State::Queued, State::Cancelled, Trigger::Cancelled)
                | (State::Running, State::Completed, Trigger::Succeeded)
                | (State::Running, State::Failed, Trigger::Failed)
                | (State::Running, State::Cached, Trigger::CacheHit)
                | (State::Running, State::Cancelled, Trigger::Cancelled)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposerRoundState {
    Requested,
    Dispatched,
    Generating,
    Returned,
    ParsedOk,
    ParseFailed,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposerRoundTrigger {
    Requested,
    Dispatched,
    GenerationStarted,
    GenerationReturned,
    Parsed,
    ParseFailed,
    Closed,
}

pub struct ProposerRoundEntity;

impl StateMachineEntity for ProposerRoundEntity {
    type State = ProposerRoundState;
    type Trigger = ProposerRoundTrigger;

    const ENTITY_TYPE: &'static str = "proposer_round";

    fn initial_state() -> Self::State {
        ProposerRoundState::Requested
    }

    fn state_name(state: Self::State) -> &'static str {
        match state {
            ProposerRoundState::Requested => "requested",
            ProposerRoundState::Dispatched => "dispatched",
            ProposerRoundState::Generating => "generating",
            ProposerRoundState::Returned => "returned",
            ProposerRoundState::ParsedOk => "parsed_ok",
            ProposerRoundState::ParseFailed => "parse_failed",
            ProposerRoundState::Closed => "closed",
        }
    }

    fn state_from_name(name: &str) -> Option<Self::State> {
        Some(match name {
            "requested" => ProposerRoundState::Requested,
            "dispatched" => ProposerRoundState::Dispatched,
            "generating" => ProposerRoundState::Generating,
            "returned" => ProposerRoundState::Returned,
            "parsed_ok" => ProposerRoundState::ParsedOk,
            "parse_failed" => ProposerRoundState::ParseFailed,
            "closed" => ProposerRoundState::Closed,
            _ => return None,
        })
    }

    fn trigger_name(trigger: Self::Trigger) -> &'static str {
        match trigger {
            ProposerRoundTrigger::Requested => "requested",
            ProposerRoundTrigger::Dispatched => "dispatched",
            ProposerRoundTrigger::GenerationStarted => "generation_started",
            ProposerRoundTrigger::GenerationReturned => "generation_returned",
            ProposerRoundTrigger::Parsed => "parsed",
            ProposerRoundTrigger::ParseFailed => "parse_failed",
            ProposerRoundTrigger::Closed => "closed",
        }
    }

    fn transition_allowed(from: Self::State, to: Self::State, trigger: Self::Trigger) -> bool {
        use ProposerRoundState as State;
        use ProposerRoundTrigger as Trigger;

        if matches!(from, State::Closed) {
            return false;
        }
        if from == to {
            return true;
        }
        matches!(
            (from, to, trigger),
            (State::Requested, State::Dispatched, Trigger::Dispatched)
                | (
                    State::Dispatched,
                    State::Generating,
                    Trigger::GenerationStarted
                )
                | (
                    State::Generating,
                    State::Returned,
                    Trigger::GenerationReturned
                )
                | (State::Returned, State::ParsedOk, Trigger::Parsed)
                | (State::Returned, State::ParseFailed, Trigger::ParseFailed)
                | (State::ParsedOk, State::Closed, Trigger::Closed)
                | (State::ParseFailed, State::Closed, Trigger::Closed)
        )
    }
}
