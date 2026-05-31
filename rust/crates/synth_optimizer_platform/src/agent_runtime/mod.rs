//! Platform-owned agent runtime boundaries.
//!
//! Algorithm crates prepare domain workspaces and prompts. This module owns the
//! generic Codex launch, auth-home preparation, JSON-RPC transport, and turn
//! usage normalization used by those algorithms.

pub mod app_server;
pub mod codex_home;
pub mod docker;
pub mod limits;
pub mod local;
pub mod session;
pub mod substrate;
pub mod supervisor;
pub mod usage;

pub use app_server::{
    ensure_turn_completed, extract_thread_id, CodexAppServerClient, CodexAppServerLaunch,
    CodexAppServerProcessLaunch,
};
pub use codex_home::{prepare_proposer_codex_launch, ProposerCodexLaunch};
pub use session::{run_turn, AgentRuntimeSubstrate, AgentTurnOutcome, CodexTurnRequest};
pub use substrate::{validate_execution_mode_compat, ExecutionSubstrate};
pub use supervisor::SupervisorReceipt;
pub use usage::{usage_from_message, usage_from_messages};
