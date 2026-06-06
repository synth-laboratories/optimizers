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
}

impl ContainerClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Self::with_headers(base_url, BTreeMap::new())
    }

    pub fn with_headers(
        base_url: impl Into<String>,
        headers: BTreeMap<String, String>,
    ) -> Result<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(OptimizerError::Config(
                "container url is required".to_string(),
            ));
        }
        let timeout_seconds = env::var("SYNTH_OPTIMIZERS_CONTAINER_HTTP_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(120);
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()?;
        Ok(Self {
            base_url,
            client,
            headers: parse_headers(headers)?,
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

    pub fn resume_rollout(
        &self,
        parent_rollout_id: &str,
        checkpoint_id: &str,
        request: &Value,
    ) -> Result<Value> {
        self.post(
            &format!("/rollouts/{parent_rollout_id}/resume"),
            &json!({
                "checkpoint_id": checkpoint_id,
                "target_rollout_id": request
                    .get("trace_correlation_id")
                    .and_then(Value::as_str),
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
        let response = self.client.get(url).headers(self.headers.clone()).send()?;
        Self::json_response(path, response)
    }

    fn post(&self, path: &str, request: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .client
            .post(url)
            .headers(self.headers.clone())
            .json(request)
            .send()?;
        Self::json_response(path, response)
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

    fn json_response(path: &str, response: reqwest::blocking::Response) -> Result<Value> {
        let status = response.status();
        let text = response.text()?;
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
