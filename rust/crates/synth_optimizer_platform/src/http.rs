use std::{collections::BTreeMap, env, time::Duration};

use reqwest::{
    blocking::Client,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::container_contract::{
    ContainerMetadataResponse, HealthResponse, RolloutResponse, TasksetResponse,
    TasksetTasksRequest, TasksetTasksResponse,
};
use crate::error::{OptimizerError, Result};
use crate::prompt_program::PromptProgram;

#[derive(Clone)]
pub struct ContainerClient {
    base_url: String,
    client: Client,
    headers: HeaderMap,
    bearer_env: Option<String>,
}

impl ContainerClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Self::with_headers(base_url, BTreeMap::new())
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        headers: BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::with_headers_and_bearer_env(base_url, headers, None)
    }

    pub fn with_headers_and_timeout(
        base_url: impl Into<String>,
        headers: BTreeMap<String, String>,
        timeout_seconds: Option<f64>,
    ) -> Result<Self> {
        Self::with_headers_bearer_env_and_timeout(base_url, headers, None, timeout_seconds)
    }

    pub fn with_headers_and_bearer_env(
        base_url: impl Into<String>,
        headers: BTreeMap<String, String>,
        bearer_env: Option<&str>,
    ) -> Result<Self> {
        Self::with_headers_bearer_env_and_timeout(base_url, headers, bearer_env, None)
    }

    pub fn with_headers_bearer_env_and_timeout(
        base_url: impl Into<String>,
        headers: BTreeMap<String, String>,
        bearer_env: Option<&str>,
        timeout_seconds: Option<f64>,
    ) -> Result<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(OptimizerError::Config(
                "container url is required".to_string(),
            ));
        }
        let timeout_seconds = container_http_timeout_seconds(
            timeout_seconds.unwrap_or_else(default_container_timeout_seconds),
        )?;
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(timeout_seconds))
            .build()?;
        let headers = parse_headers(headers)?;
        if !headers.contains_key("authorization") {
            let mut probe = headers.clone();
            apply_bearer_env(&mut probe, bearer_env)?;
        }
        Ok(Self {
            base_url,
            client,
            headers,
            bearer_env: bearer_env
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
        })
    }

    pub fn health(&self) -> Result<Value> {
        self.get("/health")
    }

    pub fn health_typed(&self) -> Result<HealthResponse> {
        self.get_typed("/health")
    }

    pub fn metadata(&self) -> Result<Value> {
        self.get("/metadata")
    }

    pub fn metadata_typed(&self) -> Result<ContainerMetadataResponse> {
        self.get_typed("/metadata")
    }

    pub fn program(&self) -> Result<Value> {
        self.get("/program")
    }

    pub fn program_typed(&self) -> Result<PromptProgram> {
        self.get_typed("/program")
    }

    pub fn task_info(&self) -> Result<Value> {
        self.get("/task_info")
    }

    pub fn taskset(&self) -> Result<Value> {
        self.get("/taskset")
    }

    pub fn taskset_typed(&self) -> Result<TasksetResponse> {
        self.get_typed("/taskset")
    }

    pub fn taskset_tasks(&self, request: &Value) -> Result<Value> {
        self.post("/taskset/tasks", request)
    }

    pub fn taskset_tasks_typed(
        &self,
        request: &TasksetTasksRequest,
    ) -> Result<TasksetTasksResponse> {
        let response: TasksetTasksResponse = self.post_typed("/taskset/tasks", request)?;
        response.validate_for_request(request)?;
        Ok(response)
    }

    pub fn rollout(&self, request: &Value) -> Result<Value> {
        self.post("/rollout", request)
    }

    pub fn register_candidate(&self, request: &Value) -> Result<Value> {
        self.register_candidate_at("/candidates", request)
    }

    pub fn register_candidate_at(&self, route: &str, request: &Value) -> Result<Value> {
        let route = route.trim();
        if !route.starts_with('/') {
            return Err(OptimizerError::Container(format!(
                "candidates route must be absolute, got {route:?}"
            )));
        }
        self.post(route, request)
    }

    pub fn resume_rollout(
        &self,
        parent_rollout_id: &str,
        checkpoint_id: &str,
        request: &Value,
    ) -> Result<Value> {
        let target_rollout_id = request
            .get("rollout_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.starts_with('$'))
            .or_else(|| {
                request
                    .get("trace_correlation_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty() && !value.starts_with('$'))
            })
            .ok_or_else(|| {
                OptimizerError::Container(
                    "resume_rollout requires concrete request.rollout_id or trace_correlation_id for target_rollout_id".to_string(),
                )
        })?;
        self.post(
            &format!("/rollouts/{parent_rollout_id}/resume_async"),
            &json!({
                "checkpoint_id": checkpoint_id,
                "target_rollout_id": target_rollout_id,
                "overrides": request,
            }),
        )
    }

    pub fn create_rollout_checkpoint(
        &self,
        rollout_id: &str,
        checkpoint_id: &str,
        request: &Value,
    ) -> Result<Value> {
        self.post(
            &format!("/rollouts/{rollout_id}/checkpoints"),
            &json!({
                "checkpoint_id": checkpoint_id,
                "metadata": request,
            }),
        )
    }

    pub fn rollout_typed(&self, request: &Value) -> Result<RolloutResponse> {
        let response: RolloutResponse = self.post_typed("/rollout", request)?;
        response.validate_for_gepa()?;
        Ok(response)
    }

    pub fn rollout_state(&self, rollout_id: &str) -> Result<Value> {
        self.get(&format!("/rollouts/{rollout_id}/state"))
    }

    pub fn rollout_record(&self, rollout_id: &str) -> Result<Value> {
        self.get(&format!("/rollouts/{rollout_id}"))
    }

    pub fn rollout_terminate(&self, rollout_id: &str, reason: &str) -> Result<Value> {
        self.post(
            &format!("/rollouts/{rollout_id}/terminate"),
            &json!({ "reason": reason }),
        )
    }

    pub fn verify_gepa_contract(&self) -> Result<ContainerMetadataResponse> {
        let metadata = self.metadata_typed()?;
        metadata.validate_gepa_contract()?;
        Ok(metadata)
    }

    fn get(&self, path: &str) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        let headers = self.headers()?;
        let (status, body) = send_with_transient_retry(|| {
            let response = self.client.get(&url).headers(headers.clone()).send()?;
            // Read the body INSIDE the retry: a connection reset mid-transfer
            // surfaces here (reqwest "error decoding response body"), not at send().
            let status = response.status();
            let body = response.text()?;
            Ok((status, body))
        })?;
        Self::json_from_status_body(path, status, body)
    }

    fn post(&self, path: &str, request: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        let headers = self.headers()?;
        let (status, body) = send_with_transient_retry(|| {
            let response = self
                .client
                .post(&url)
                .headers(headers.clone())
                .json(request)
                .send()?;
            let status = response.status();
            let body = response.text()?;
            Ok((status, body))
        })?;
        Self::json_from_status_body(path, status, body)
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = self.headers.clone();
        apply_bearer_env(&mut headers, self.bearer_env.as_deref())?;
        Ok(headers)
    }

    fn get_typed<T>(&self, path: &str) -> Result<T>
    where
        T: DeserializeOwned,
    {
        Ok(serde_json::from_value(self.get(path)?)?)
    }

    fn post_typed<T, R>(&self, path: &str, request: &R) -> Result<T>
    where
        T: DeserializeOwned,
        R: serde::Serialize,
    {
        let request = serde_json::to_value(request)?;
        Ok(serde_json::from_value(self.post(path, &request)?)?)
    }

    fn json_from_status_body(
        path: &str,
        status: reqwest::StatusCode,
        text: String,
    ) -> Result<Value> {
        if !status.is_success() {
            return Err(OptimizerError::ContainerHttpStatus {
                path: path.to_string(),
                status_code: status.as_u16(),
                body: text.chars().take(1000).collect::<String>(),
            });
        }
        if text.trim().is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        Ok(serde_json::from_str(&text)?)
    }
}

/// Retry a single HTTP send on TRANSIENT transport failures only — a connection
/// reset / connect failure / timeout under heavy concurrent container load (the
/// 20-wide full-rollout lane momentarily saturating the crafter app-server). This
/// is NOT a data fallback: it re-issues the identical request and never degrades or
/// fabricates a response. An HTTP error STATUS (4xx/5xx) is a real server reply and
/// is returned as-is (handled by `json_response`), never retried here. Without this,
/// one momentary `error sending request` kills an entire multi-round optimization.
fn send_with_transient_retry<F, T>(mut attempt: F) -> reqwest::Result<T>
where
    F: FnMut() -> reqwest::Result<T>,
{
    const MAX_ATTEMPTS: usize = 4;
    let mut backoff = Duration::from_millis(250);
    let mut last_error: Option<reqwest::Error> = None;
    for attempt_index in 0..MAX_ATTEMPTS {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(error) => {
                // Transient transport failures under heavy concurrent container
                // load: connect failure, timeout, request-build, or a connection
                // reset mid-body (is_body/is_decode → reqwest "error decoding
                // response body"). An HTTP error STATUS is NOT a reqwest::Error here
                // (it is surfaced later by json_from_status_body), so 4xx/5xx are
                // never retried by this wrapper.
                let transient = error.is_connect()
                    || error.is_timeout()
                    || error.is_request()
                    || error.is_body()
                    || error.is_decode();
                if !transient || attempt_index + 1 == MAX_ATTEMPTS {
                    return Err(error);
                }
                last_error = Some(error);
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(2));
            }
        }
    }
    Err(last_error.expect("retry loop recorded no error"))
}

fn default_container_timeout_seconds() -> f64 {
    env::var("SYNTH_OPTIMIZERS_CONTAINER_HTTP_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(120.0)
}

fn container_http_timeout_seconds(value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(OptimizerError::Config(
            "container HTTP timeout must be a finite positive number".to_string(),
        ))
    }
}

fn parse_headers(headers: BTreeMap<String, String>) -> Result<HeaderMap> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|source| {
            OptimizerError::Config(format!(
                "container.headers contains invalid header name {name:?}: {source}"
            ))
        })?;
        let header_value = HeaderValue::from_str(&value).map_err(|source| {
            OptimizerError::Config(format!(
                "container.headers contains invalid value for header {name:?}: {source}"
            ))
        })?;
        map.insert(header_name, header_value);
    }
    Ok(map)
}

fn apply_bearer_env(headers: &mut HeaderMap, bearer_env: Option<&str>) -> Result<()> {
    let Some(env_name) = bearer_env.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(());
    };
    if headers.contains_key("authorization") {
        return Ok(());
    }
    let token = env::var(env_name)
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(format!(
                "container.auth_bearer_env references missing environment variable {env_name:?}"
            ))
        })?;
    let header_value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|source| {
        OptimizerError::Config(format!(
            "container.auth_bearer_env {env_name:?} produced invalid bearer header: {source}"
        ))
    })?;
    headers.insert(HeaderName::from_static("authorization"), header_value);
    Ok(())
}
