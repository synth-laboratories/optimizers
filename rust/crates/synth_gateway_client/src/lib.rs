//! Inference-gateway client for optimizer proposers (SYN-4192).
//!
//! The optimizer holds no provider key. A proposer configured with
//! `backend = "gateway_responses"` names a session *handle* in `base_url`:
//!
//! * `gateway-session://run/<run_id>?org=<org_id>&route_set=<route_set>` — a hosted
//!   run. The session is issued by the Synth backend
//!   (`POST /api/v1/optimizers/internal/runs/{run_id}/gateway-session`) with this
//!   service's enrolled optimizer identity, scoped to the run's org and funded by its
//!   wallet. The handle carries no secret, so a persisted config stays safe and a
//!   restarted service re-issues the same scope.
//! * `gateway-session://env` — a local research run with a session minted out of
//!   band: `SYNTH_GATEWAY_SESSION_TOKEN` and `SYNTH_RESPONSES_GATEWAY_URL`.
//!
//! Every call speaks the Responses wire to the gateway's `/v1/responses`; the gateway
//! enforces the session ceiling per call and emits one usage receipt per call, which
//! is the money record. Usage returned to the optimizer is marked
//! `billing_authority = "inference_gateway"` so the optimizer never bills it again.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

pub const BACKEND: &str = "gateway_responses";
pub const HANDLE_SCHEME: &str = "gateway-session://";
pub const BILLING_AUTHORITY: &str = "inference_gateway";
pub const SESSION_TOKEN_ENV: &str = "SYNTH_GATEWAY_SESSION_TOKEN";
pub const GATEWAY_URL_ENV: &str = "SYNTH_RESPONSES_GATEWAY_URL";
/// Renew a cached session this long before it expires.
const RENEW_MARGIN: Duration = Duration::from_secs(120);
const ISSUE_TIMEOUT: Duration = Duration::from_secs(15);
/// Hosted session bounds requested from the backend (it may refuse larger).
pub const RUN_SESSION_CEILING_MICROS: u64 = 5_000_000;
pub const RUN_SESSION_LIFETIME_SECONDS: u64 = 6 * 3600;

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("invalid gateway session handle {0:?}")]
    InvalidHandle(String),
    #[error("gateway session issuer is not installed in this process")]
    IssuerMissing,
    #[error("no enrolled optimizer identity for org {0}")]
    IdentityMissing(String),
    #[error("gateway session refused: {code} (http {status})")]
    SessionRefused { code: String, status: u16 },
    #[error("gateway session unavailable: {0}")]
    SessionUnavailable(String),
    #[error("model {model:?} is not admitted by the gateway session (admitted: {admitted:?})")]
    ModelNotAdmitted {
        model: String,
        admitted: Vec<String>,
    },
    #[error("gateway call refused: status {status} code {code}")]
    CallRefused { status: u16, code: String },
    #[error("gateway call failed after {attempts} attempts: {last}")]
    CallFailed { attempts: usize, last: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SessionHandle {
    Run {
        run_id: String,
        org_id: String,
        route_set: String,
    },
    Env,
}

impl SessionHandle {
    pub fn parse(base_url: &str) -> Result<Self, GatewayError> {
        let invalid = || GatewayError::InvalidHandle(base_url.to_string());
        let rest = base_url
            .trim()
            .strip_prefix(HANDLE_SCHEME)
            .ok_or_else(invalid)?;
        if rest == "env" {
            return Ok(Self::Env);
        }
        let rest = rest.strip_prefix("run/").ok_or_else(invalid)?;
        let (run_id, query) = rest.split_once('?').ok_or_else(invalid)?;
        let mut org_id = None;
        let mut route_set = None;
        for pair in query.split('&') {
            match pair.split_once('=') {
                Some(("org", value)) => org_id = Some(value.to_string()),
                Some(("route_set", value)) => route_set = Some(value.to_string()),
                _ => return Err(invalid()),
            }
        }
        let valid = |value: &str| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
        };
        match (org_id, route_set) {
            (Some(org_id), Some(route_set))
                if valid(run_id) && valid(&org_id) && valid(&route_set) =>
            {
                Ok(Self::Run {
                    run_id: run_id.to_string(),
                    org_id,
                    route_set,
                })
            }
            _ => Err(invalid()),
        }
    }

    pub fn for_run(run_id: &str, org_id: &str, route_set: &str) -> String {
        format!("{HANDLE_SCHEME}run/{run_id}?org={org_id}&route_set={route_set}")
    }
}

/// Authority names cannot be supplied as arbitrary sandbox overrides.
pub fn is_sandbox_authority_environment(name: &str) -> bool {
    name.starts_with("SYNTH_GATEWAY_")
        || matches!(name, "OPENAI_API_KEY" | "OPENAI_BASE_URL" | "OPENROUTER_API_KEY" | "OPENROUTER_BASE_URL" | "ANTHROPIC_API_KEY" | "ANTHROPIC_AUTH_TOKEN" | "ANTHROPIC_BASE_URL" | "NVIDIA_API_KEY" | "DEEPSEEK_API_KEY" | "GEMINI_API_KEY" | "GOOGLE_API_KEY" | "SYNTH_API_KEY")
}

/// Where hosted sessions come from: the backend origin and this service's enrolled
/// optimizer identity per org. Installed once at service start.
pub type CredentialForOrg = dyn Fn(&str) -> Option<String> + Send + Sync;

pub struct Issuer {
    pub backend_url: String,
    pub credential_for_org: Box<CredentialForOrg>,
}

static ISSUER: OnceLock<Issuer> = OnceLock::new();
static SESSIONS: OnceLock<Mutex<HashMap<SessionHandle, Session>>> = OnceLock::new();

pub fn install_issuer(issuer: Issuer) -> bool {
    ISSUER.set(issuer).is_ok()
}

#[derive(Clone)]
pub struct Session {
    token: String,
    pub gateway_url: String,
    pub models: Vec<String>,
    renew_at: Option<Instant>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("token", &"<redacted>")
            .field("gateway_url", &self.gateway_url)
            .field("models", &self.models)
            .finish()
    }
}

impl Session {
    pub fn responses_url(&self) -> String {
        format!("{}/v1/responses", self.gateway_url.trim_end_matches('/'))
    }

    /// Runtime-only sandbox environment. Values never enter persisted config.
    pub fn sandbox_environment(
        &self,
        model: &str,
    ) -> Result<std::collections::BTreeMap<String, String>, GatewayError> {
        if !self.admits(model) {
            return Err(GatewayError::ModelNotAdmitted {
                model: model.into(),
                admitted: self.models.clone(),
            });
        }
        let url = reqwest::Url::parse(&self.gateway_url)
            .map_err(|_| GatewayError::SessionUnavailable("gateway origin invalid".into()))?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(GatewayError::SessionUnavailable(
                "gateway origin invalid".into(),
            ));
        }
        Ok(std::collections::BTreeMap::from([
            (SESSION_TOKEN_ENV.into(), self.token.clone()),
            (
                "SYNTH_GATEWAY_SANDBOX_BASE_URL".into(),
                format!("{}/v1", self.gateway_url.trim_end_matches('/')),
            ),
            ("SYNTH_GATEWAY_SANDBOX_MODEL".into(), model.into()),
        ]))
    }

    pub fn admits(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|m| m == model)
    }
}

pub fn session(handle: &SessionHandle) -> Result<Session, GatewayError> {
    let cache = SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache.lock().expect("session cache").get(handle) {
        if cached.renew_at.is_none_or(|at| Instant::now() < at) {
            return Ok(cached.clone());
        }
    }
    let issued = match handle {
        SessionHandle::Env => env_session()?,
        SessionHandle::Run {
            run_id,
            org_id,
            route_set,
        } => {
            let issuer = ISSUER.get().ok_or(GatewayError::IssuerMissing)?;
            issue_run_session(issuer, run_id, org_id, route_set)?
        }
    };
    cache
        .lock()
        .expect("session cache")
        .insert(handle.clone(), issued.clone());
    Ok(issued)
}

fn env_session() -> Result<Session, GatewayError> {
    let token = std::env::var(SESSION_TOKEN_ENV).unwrap_or_default();
    let gateway_url = std::env::var(GATEWAY_URL_ENV).unwrap_or_default();
    if !token.starts_with("gw_") || gateway_url.trim().is_empty() {
        return Err(GatewayError::SessionUnavailable(format!(
            "{SESSION_TOKEN_ENV} (gw_…) and {GATEWAY_URL_ENV} are required for gateway-session://env"
        )));
    }
    Ok(Session {
        token,
        gateway_url,
        models: Vec::new(),
        renew_at: None,
    })
}

fn issue_run_session(
    issuer: &Issuer,
    run_id: &str,
    org_id: &str,
    route_set: &str,
) -> Result<Session, GatewayError> {
    let credential = (issuer.credential_for_org)(org_id)
        .ok_or_else(|| GatewayError::IdentityMissing(org_id.to_string()))?;
    let url = format!(
        "{}/api/v1/optimizers/internal/runs/{run_id}/gateway-session",
        issuer.backend_url.trim_end_matches('/')
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(ISSUE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| GatewayError::SessionUnavailable(e.to_string()))?;
    let response = client
        .post(&url)
        .bearer_auth(credential)
        .json(&json!({
            "org_id": org_id,
            "route_set": route_set,
            "ceiling_micros": RUN_SESSION_CEILING_MICROS,
            "lifetime_seconds": RUN_SESSION_LIFETIME_SECONDS,
        }))
        .send()
        .map_err(|e| GatewayError::SessionUnavailable(format!("backend unreachable: {e}")))?;
    let status = response.status().as_u16();
    let body: Value = response.json().unwrap_or(Value::Null);
    if status != 200 {
        let code = body
            .pointer("/detail/error_code")
            .or_else(|| body.get("error_code"))
            .and_then(Value::as_str)
            .unwrap_or("session_refused")
            .to_string();
        return Err(GatewayError::SessionRefused { code, status });
    }
    session_from_issue_response(&body, Instant::now())
}

/// Parse the backend's `synth.gateway-caller-session.v1` answer.
pub fn session_from_issue_response(body: &Value, now: Instant) -> Result<Session, GatewayError> {
    let malformed = || GatewayError::SessionUnavailable("malformed session response".into());
    if body.get("schema_version").and_then(Value::as_str) != Some("synth.gateway-caller-session.v1")
    {
        return Err(malformed());
    }
    let token = body
        .get("token")
        .and_then(Value::as_str)
        .filter(|t| t.starts_with("gw_"));
    let gateway_url = body
        .get("gateway_url")
        .and_then(Value::as_str)
        .filter(|u| !u.is_empty());
    let expires_at = body.get("expires_at").and_then(Value::as_str);
    let (Some(token), Some(gateway_url), Some(expires_at)) = (token, gateway_url, expires_at)
    else {
        return Err(malformed());
    };
    let expires =
        time::OffsetDateTime::parse(expires_at, &time::format_description::well_known::Rfc3339)
            .map_err(|_| malformed())?;
    let remaining = (expires - time::OffsetDateTime::now_utc())
        .whole_seconds()
        .max(0) as u64;
    let renew_in = Duration::from_secs(remaining).saturating_sub(RENEW_MARGIN);
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Ok(Session {
        token: token.to_string(),
        gateway_url: gateway_url.trim_end_matches('/').to_string(),
        models,
        renew_at: Some(now + renew_in),
    })
}

/// A Responses request for a system + user turn. `json_object` asks for a JSON
/// object (Responses `text.format`).
pub fn responses_request(
    model: &str,
    instructions: &str,
    user: &str,
    max_output_tokens: u64,
    json_object: bool,
    temperature: Option<f64>,
) -> Value {
    let mut body = json!({
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
        "max_output_tokens": max_output_tokens,
        "store": false,
        "stream": false,
    });
    if json_object {
        body["text"] = json!({"format": {"type": "json_object"}});
    }
    if let Some(temperature) = temperature {
        body["temperature"] = json!(temperature);
    }
    body
}

/// Concatenated `output_text` of every assistant message item; `None` when empty.
pub fn output_text(response: &Value) -> Option<String> {
    let text = response
        .get("output")
        .and_then(Value::as_array)?
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    (!text.trim().is_empty()).then_some(text)
}

/// `true` when the response stopped on the output-token bound (Chat's
/// `finish_reason = "length"`).
pub fn truncated(response: &Value) -> bool {
    response.get("status").and_then(Value::as_str) == Some("incomplete")
        && response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            == Some("max_output_tokens")
}

/// Usage for the optimizer's own records: token counts kept, billing handed to
/// the gateway receipt.
pub fn optimizer_usage(response: &Value, receipt_id: Option<&str>) -> Value {
    let mut usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    if let Value::Object(map) = &mut usage {
        map.insert("billing_authority".into(), json!(BILLING_AUTHORITY));
        // The optimizer's own cost for this call is zero: the receipt already billed it.
        map.insert("cost_usd".into(), json!(0.0));
        map.insert("cost_source".into(), json!("inference_gateway_receipt"));
        if let Some(receipt_id) = receipt_id {
            map.insert("gateway_receipt_id".into(), json!(receipt_id));
        }
    }
    usage
}

#[derive(Debug)]
pub struct GatewayResponse {
    pub body: Value,
    pub receipt_id: Option<String>,
}

/// POST one Responses request. Only refusals before provider work (429, 5xx from
/// the gateway with no receipt dispatch) are retried; a typed refusal such as
/// `ceiling_exhausted` (402) is returned as is.
pub fn post_responses(
    handle: &SessionHandle,
    model: &str,
    body: &Value,
    timeout: Duration,
    max_attempts: usize,
) -> Result<GatewayResponse, GatewayError> {
    let session = session(handle)?;
    if !session.admits(model) {
        return Err(GatewayError::ModelNotAdmitted {
            model: model.to_string(),
            admitted: session.models.clone(),
        });
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| GatewayError::CallFailed {
            attempts: 0,
            last: e.to_string(),
        })?;
    let mut last = String::new();
    let attempts = max_attempts.max(1);
    for attempt in 1..=attempts {
        let response = match client
            .post(session.responses_url())
            .bearer_auth(&session.token)
            .json(body)
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                // A send error after dispatch is in doubt; the gateway receipt decides
                // cost, so a retry is a new call with its own receipt.
                last = format!("send error: {error}");
                std::thread::sleep(backoff(attempt));
                continue;
            }
        };
        let status = response.status().as_u16();
        let receipt_id = response
            .headers()
            .get("x-gateway-receipt-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let text = response.text().unwrap_or_default();
        if (200..300).contains(&status) {
            let body = serde_json::from_str(&text).map_err(|e| GatewayError::CallFailed {
                attempts: attempt,
                last: e.to_string(),
            })?;
            return Ok(GatewayResponse { body, receipt_id });
        }
        let code = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("error_code")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "gateway_error".to_string());
        if attempt < attempts && matches!(status, 429 | 500 | 502 | 503 | 504) {
            last = format!("status {status} {code}");
            std::thread::sleep(backoff(attempt));
            continue;
        }
        return Err(GatewayError::CallRefused { status, code });
    }
    Err(GatewayError::CallFailed { attempts, last })
}

fn backoff(attempt: usize) -> Duration {
    Duration::from_secs(match attempt {
        1 => 2,
        2 => 5,
        _ => 10,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_binding_pins_model_origin_and_hides_token() {
        let session = Session { token: "gw_fixture".into(), gateway_url: "http://127.0.0.1:8123".into(), models: vec!["admitted-model".into()], renew_at: None };
        let env = session.sandbox_environment("admitted-model").unwrap();
        assert_eq!(env["SYNTH_GATEWAY_SANDBOX_BASE_URL"], "http://127.0.0.1:8123/v1");
        assert_eq!(env[SESSION_TOKEN_ENV], "gw_fixture");
        assert!(session.sandbox_environment("other-model").is_err());
        assert!(!format!("{session:?}").contains("gw_fixture"));
        let invalid = Session { gateway_url: "https://user:secret@localhost".into(), ..session };
        assert!(invalid.sandbox_environment("admitted-model").is_err());
    }

    #[test]
    fn run_handle_round_trips_and_carries_no_secret() {
        let handle = SessionHandle::for_run(
            "run_1",
            "6f6c1f2e-1d1a-4d55-9a6f-3a3b9f0c0001",
            "optimizer_openrouter_v1",
        );
        assert!(!handle.contains("gw_") && !handle.contains("sk_"));
        assert_eq!(
            SessionHandle::parse(&handle).unwrap(),
            SessionHandle::Run {
                run_id: "run_1".into(),
                org_id: "6f6c1f2e-1d1a-4d55-9a6f-3a3b9f0c0001".into(),
                route_set: "optimizer_openrouter_v1".into(),
            }
        );
        assert_eq!(
            SessionHandle::parse("gateway-session://env").unwrap(),
            SessionHandle::Env
        );
    }

    #[test]
    fn provider_urls_and_malformed_handles_are_refused() {
        for bad in [
            "https://openrouter.ai/api/v1",
            "gateway-session://run/r?org=o",
            "gateway-session://run/r?org=o&route_set=x&extra=1",
            "gateway-session://run/r/../x?org=o&route_set=x",
        ] {
            assert!(SessionHandle::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn output_text_and_truncation_follow_the_responses_wire() {
        let response = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "{\"a\":"},
                    {"type": "output_text", "text": "1}"}
                ]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4}
        });
        assert_eq!(output_text(&response).as_deref(), Some("{\"a\":1}"));
        assert!(!truncated(&response));
        let cut = json!({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []});
        assert!(truncated(&cut) && output_text(&cut).is_none());
    }

    #[test]
    fn optimizer_usage_hands_billing_to_the_gateway_receipt() {
        let usage = optimizer_usage(&json!({"usage": {"input_tokens": 3}}), Some("r-1"));
        assert_eq!(usage["billing_authority"], BILLING_AUTHORITY);
        assert_eq!(usage["gateway_receipt_id"], "r-1");
        assert_eq!(usage["input_tokens"], 3);
        assert_eq!(usage["cost_usd"], 0.0);
    }

    #[test]
    fn request_is_responses_shaped_with_json_object_format() {
        let body = responses_request("m", "sys", "user", 64, true, Some(0.2));
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(body["text"]["format"]["type"], "json_object");
        assert_eq!(body["store"], false);
        assert!(body.get("messages").is_none());
    }

    #[test]
    fn issue_response_parses_and_redacts() {
        let expires = (time::OffsetDateTime::now_utc() + time::Duration::hours(1))
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap();
        let session = session_from_issue_response(
            &json!({"schema_version": "synth.gateway-caller-session.v1", "token": "gw_abc",
                    "gateway_url": "https://gw.test/", "expires_at": expires,
                    "models": ["deepseek/deepseek-v4-flash"]}),
            Instant::now(),
        )
        .unwrap();
        assert_eq!(session.responses_url(), "https://gw.test/v1/responses");
        assert!(session.admits("deepseek/deepseek-v4-flash") && !session.admits("other"));
        assert!(!format!("{session:?}").contains("gw_abc"));
    }
}
