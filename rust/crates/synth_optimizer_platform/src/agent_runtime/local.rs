use crate::Result;

use super::app_server::{CodexAppServerClient, CodexAppServerLaunch};
use super::session::{
    run_codex_jsonrpc_command_exec, run_codex_jsonrpc_turn, AgentCommandExecOutcome,
    AgentRuntimeSubstrate, AgentTurnOutcome, CodexCommandExecRequest, CodexTurnRequest,
};

pub struct LocalCodexSubstrate;

impl AgentRuntimeSubstrate for LocalCodexSubstrate {
    fn run_codex_turn(&self, request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
        let client = CodexAppServerClient::start(CodexAppServerLaunch {
            proposer: request.proposer,
            workspace_dir: request.workspace_dir,
            model: request.model,
        })?;
        run_codex_jsonrpc_turn(client, request, None)
    }

    fn run_codex_command_exec(
        &self,
        request: CodexCommandExecRequest<'_>,
    ) -> Result<AgentCommandExecOutcome> {
        let client = CodexAppServerClient::start(CodexAppServerLaunch {
            proposer: request.proposer,
            workspace_dir: request.workspace_dir,
            model: request.model,
        })?;
        run_codex_jsonrpc_command_exec(client, request, None)
    }
}
