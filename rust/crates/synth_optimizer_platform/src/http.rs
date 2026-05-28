use std::{env, time::Duration};

use reqwest::blocking::Client;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::container_contract::{
    ContainerMetadataResponse, DatasetResponse, DatasetRowsRequest, DatasetRowsResponse,
    HealthResponse, RolloutResponse,
};
use crate::error::{OptimizerError, Result};
use crate::prompt_program::PromptProgram;

#[derive(Clone)]
pub struct ContainerClient {
    base_url: String,
    client: Client,
}

impl ContainerClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
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
        Ok(Self { base_url, client })
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

    pub fn dataset(&self) -> Result<Value> {
        self.get("/dataset")
    }

    pub fn dataset_typed(&self) -> Result<DatasetResponse> {
        self.get_typed("/dataset")
    }

    pub fn dataset_rows(&self, request: &Value) -> Result<Value> {
        self.post("/dataset/rows", request)
    }

    pub fn dataset_rows_typed(&self, request: &DatasetRowsRequest) -> Result<DatasetRowsResponse> {
        let response: DatasetRowsResponse = self.post_typed("/dataset/rows", request)?;
        response.validate_for_request(request)?;
        Ok(response)
    }

    pub fn rollout(&self, request: &Value) -> Result<Value> {
        self.post("/rollout", request)
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
        let response = self.client.get(url).send()?;
        Self::json_response(path, response)
    }

    fn post(&self, path: &str, request: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client.post(url).json(request).send()?;
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
            return Err(OptimizerError::Container(format!(
                "{} failed with status {}: {}",
                path,
                status,
                text.chars().take(1000).collect::<String>()
            )));
        }
        if text.trim().is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        Ok(serde_json::from_str(&text)?)
    }
}
