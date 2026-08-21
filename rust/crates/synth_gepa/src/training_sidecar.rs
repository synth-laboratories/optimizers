use std::collections::BTreeMap;
use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use reqwest::blocking::{Client, Response};
use reqwest::Method;
use serde_json::{json, Value};

#[derive(Clone)]
pub(crate) struct TrainingSidecar {
    state: Arc<Mutex<SidecarState>>,
    client: Client,
    root: PathBuf,
}

struct SidecarState {
    backend_url: Option<String>,
    child: Option<Child>,
    model_path: Option<PathBuf>,
    jobs: BTreeMap<String, TrainingPlacement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainingPlacement {
    SftLocal,
    SftHosted,
    CisPoLocal,
}

pub(crate) struct SidecarResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl TrainingSidecar {
    pub(crate) fn new(service_db_path: &Path) -> Self {
        let root = service_db_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("training-sidecar");
        Self {
            state: Arc::new(Mutex::new(SidecarState {
                backend_url: None,
                child: None,
                model_path: None,
                jobs: BTreeMap::new(),
            })),
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("training sidecar HTTP client"),
            root,
        }
    }

    pub(crate) fn cispo_supported(&self) -> bool {
        // Capability discovery is part of the optimizer service handshake and
        // must stay side-effect free. Calling `mlx_capabilities()` here used to
        // run `ensure_backend()`, which could start a model service and block
        // longer than Workshop's handshake deadline. Live backend capability
        // validation still happens when a CISPO job is created.
        ["SYNTH_MLX_RL_URL", "SYNTH_MLX_RL_BIN", "SYNTH_MLX_RL_ROOT"]
            .into_iter()
            .any(|name| env::var(name).is_ok_and(|value| !value.trim().is_empty()))
    }

    pub(crate) fn route(
        &self,
        method: &str,
        segments: &[&str],
        query: &str,
        body: &[u8],
    ) -> Option<SidecarResponse> {
        match (method, segments) {
            ("POST", ["v1", "training", "jobs"]) => Some(self.create_job(body)),
            ("GET", ["v1", "training", "jobs", job_id]) => {
                Some(self.job_request(job_id, "GET", "", query, body))
            }
            ("GET", ["v1", "training", "jobs", job_id, "events"]) => {
                Some(self.job_request(job_id, "GET", "/events", query, body))
            }
            ("GET", ["v1", "training", "jobs", job_id, "handoff"]) => {
                Some(self.job_request(job_id, "GET", "/handoff", query, body))
            }
            ("POST", ["v1", "training", "jobs", job_id, "cancel"]) => {
                Some(self.job_request(job_id, "POST", "/cancel", query, body))
            }
            ("POST", ["v1", "training", "jobs", job_id, "resume"]) => {
                Some(self.job_request(job_id, "POST", "/resume", query, body))
            }
            ("POST", ["v1", "training", "jobs", job_id, "chat"]) => Some(self.chat(job_id, body)),
            ("POST", ["v1", "inference", "chat", "completions"]) => {
                Some(self.infer_openai("chat_completions", body))
            }
            ("POST", ["v1", "inference", "responses"]) => {
                Some(self.infer_openai("responses", body))
            }
            _ => None,
        }
    }

    fn create_job(&self, body: &[u8]) -> SidecarResponse {
        let request: Value = match serde_json::from_slice(body) {
            Ok(Value::Object(value)) => Value::Object(value),
            Ok(_) => return error(400, "training request must be a JSON object"),
            Err(source) => return error(400, &format!("invalid training request: {source}")),
        };
        let placement_name = match request.get("placement").and_then(Value::as_str) {
            Some(value) => value,
            None => return error(422, "placement is required"),
        };
        let placement = match placement_name {
            "training.sft.local" => TrainingPlacement::SftLocal,
            "training.sft.hosted" => TrainingPlacement::SftHosted,
            "training.cispo.local" => TrainingPlacement::CisPoLocal,
            "training.cispo.hosted" => return error(422, "training.cispo.hosted is not admitted"),
            _ => return error(422, "unknown training placement"),
        };
        let recipe_id = match request.get("recipe_id").and_then(Value::as_str) {
            Some(value) if !value.trim().is_empty() => value,
            _ => return error(422, "recipe_id is required"),
        };
        let mut config = match request.get("config").and_then(Value::as_object) {
            Some(value) => value.clone(),
            None => return error(422, "config must be an object"),
        };
        let job_id = request
            .get("job_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(new_job_id);

        if placement == TrainingPlacement::SftHosted {
            let response = self.hosted_request(
                "POST",
                "/api/v1/optimizers/runs",
                Some(&json!({
                    "algorithm": "sft",
                    "config_json": config,
                    "idempotency_key": job_id,
                })),
            );
            if let Ok(value) = response_json(&response) {
                if let Some(remote_id) = value.get("run_id").and_then(Value::as_str) {
                    self.remember_job(remote_id, placement);
                }
            }
            return response;
        }

        if placement == TrainingPlacement::CisPoLocal && !self.cispo_supported() {
            return error(422, "local CISPO is not supported by synth-mlx-rl");
        }

        config.insert(
            "backend".to_string(),
            Value::String(
                if placement == TrainingPlacement::CisPoLocal {
                    "cispo"
                } else {
                    "qwen_lora"
                }
                .to_string(),
            ),
        );
        let model_path = take_model_path(&mut config);
        config.entry("output_dir".to_string()).or_insert_with(|| {
            Value::String(
                self.root
                    .join("jobs")
                    .join(&job_id)
                    .join("output")
                    .display()
                    .to_string(),
            )
        });
        let base = match self.ensure_backend(model_path.as_deref()) {
            Ok(value) => value,
            Err(message) => return error(503, &message),
        };
        let configured = self.forward(
            "POST",
            &format!("{base}/v1/jobs"),
            Some(&json!({"job_id": job_id, "config": config})),
        );
        if !(200..300).contains(&configured.status) {
            return configured;
        }
        let launched = self.forward(
            "POST",
            &format!("{base}/v1/jobs/{job_id}/launch"),
            Some(&json!({})),
        );
        if (200..300).contains(&launched.status) {
            self.remember_job(&job_id, placement);
            if let Ok(mut value) = response_json(&launched) {
                if let Some(object) = value.as_object_mut() {
                    object.insert("placement".to_string(), json!(placement_name));
                    object.insert("recipe_id".to_string(), json!(recipe_id));
                }
                return json_value(launched.status, &value);
            }
        }
        launched
    }

    fn job_request(
        &self,
        job_id: &str,
        method: &str,
        suffix: &str,
        query: &str,
        body: &[u8],
    ) -> SidecarResponse {
        let placement = match self.job_placement(job_id) {
            Some(value) => value,
            None => return error(404, "training job not found"),
        };
        if placement == TrainingPlacement::SftHosted {
            let hosted_suffix = match suffix {
                "" => "",
                "/cancel" => "/cancel",
                "/events" => "/algorithm-events",
                "/resume" => "/resume",
                "/handoff" => "",
                _ => return error(404, "training route not found"),
            };
            let path = format!(
                "/api/v1/optimizers/runs/{job_id}{hosted_suffix}{}",
                query_suffix(query)
            );
            let payload = parse_optional_json(body);
            let response = self.hosted_request(method, &path, payload.as_ref());
            if suffix == "/handoff" {
                return match response_json(&response) {
                    Ok(run) => json_value(
                        response.status,
                        &json!({"job_id": job_id, "result": run.get("result")}),
                    ),
                    Err(_) => response,
                };
            }
            return response;
        }
        let base = match self.ensure_backend(None) {
            Ok(value) => value,
            Err(message) => return error(503, &message),
        };
        self.forward(
            method,
            &format!("{base}/v1/jobs/{job_id}{suffix}{}", query_suffix(query)),
            parse_optional_json(body).as_ref(),
        )
    }

    fn chat(&self, job_id: &str, body: &[u8]) -> SidecarResponse {
        match self.job_placement(job_id) {
            Some(TrainingPlacement::SftLocal | TrainingPlacement::CisPoLocal) => {}
            Some(TrainingPlacement::SftHosted) => {
                return error(409, "chat is owned by local synth-mlx-rl jobs")
            }
            None => return error(404, "training job not found"),
        }
        let base = match self.ensure_backend(None) {
            Ok(value) => value,
            Err(message) => return error(503, &message),
        };
        self.forward(
            "POST",
            &format!("{base}/v1/chat/completions"),
            parse_optional_json(body).as_ref(),
        )
    }

    fn infer_openai(&self, family: &str, body: &[u8]) -> SidecarResponse {
        let envelope: Value = match serde_json::from_slice(body) {
            Ok(value) => value,
            Err(source) => return error(400, &format!("invalid inference request: {source}")),
        };
        let placement = envelope
            .get("placement")
            .and_then(Value::as_str)
            .unwrap_or("this_mac");
        if placement == "hosted" {
            return error(409, "hosted inference is owned by the public SFT service");
        }
        let stream = envelope.get("stream").and_then(Value::as_bool) == Some(true)
            || envelope
                .pointer("/body/stream")
                .and_then(Value::as_bool)
                == Some(true);
        let mut payload = envelope
            .get("body")
            .cloned()
            .unwrap_or_else(|| envelope.clone());
        let pin = envelope
            .get("policy_snapshot_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                payload
                    .get("policy_snapshot_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        if let Some(object) = payload.as_object_mut() {
            if let Some(pin) = pin {
                object.insert("policy_snapshot_id".into(), json!(pin));
            }
            if stream {
                object.insert("stream".into(), json!(true));
            }
        }
        let base = match self.ensure_backend(None) {
            Ok(value) => value,
            Err(message) => return error(503, &message),
        };
        let path = match family {
            "responses" => format!("{base}/v1/responses"),
            _ => format!("{base}/v1/chat/completions"),
        };
        let mut response = self.forward("POST", &path, Some(&payload));
        if matches!(response.status, 404 | 410) {
            if let Some(adapter) = envelope.get("adapter_path").and_then(Value::as_str) {
                if let Some(name) = Path::new(adapter).file_name().and_then(|value| value.to_str())
                {
                    let _ = self.forward(
                        "POST",
                        &format!("{base}/v1/checkpoints/load"),
                        Some(&json!({ "name": name })),
                    );
                    response = self.forward("POST", &path, Some(&payload));
                }
            }
        }
        response
    }

    fn remember_job(&self, job_id: &str, placement: TrainingPlacement) {
        if let Ok(mut state) = self.state.lock() {
            state.jobs.insert(job_id.to_string(), placement);
        }
    }

    fn job_placement(&self, job_id: &str) -> Option<TrainingPlacement> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.jobs.get(job_id).copied())
    }

    fn ensure_backend(&self, model_path: Option<&Path>) -> Result<String, String> {
        if let Ok(configured) = env::var("SYNTH_MLX_RL_URL") {
            let base = configured.trim_end_matches('/').to_string();
            self.health_check(&base)?;
            return Ok(base);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| "synth-mlx-rl process lock is poisoned".to_string())?;
        let requested_model = model_path.map(Path::to_path_buf);
        let child_alive = state
            .child
            .as_mut()
            .map(|child| child.try_wait().ok().flatten().is_none())
            .unwrap_or(false);
        if child_alive && (requested_model.is_none() || requested_model == state.model_path) {
            return state
                .backend_url
                .clone()
                .ok_or_else(|| "synth-mlx-rl URL is unavailable".to_string());
        }
        if child_alive {
            if !state.jobs.is_empty() {
                return Err(
                    "cannot change the managed model path while local training jobs exist"
                        .to_string(),
                );
            }
            if let Some(child) = state.child.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        fs_create_dir_all(&self.root)?;
        let port = reserve_port()?;
        let base = format!("http://127.0.0.1:{port}");
        let mut command = mlx_command();
        command
            .arg("serve")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--root")
            .arg(self.root.join("mlx-rl"));
        if let Some(path) = requested_model.as_ref() {
            command.env("SYNTH_MLX_RL_MODEL_PATH", path);
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command
            .spawn()
            .map_err(|source| format!("failed to spawn synth-mlx-rl serve: {source}"))?;
        state.backend_url = Some(base.clone());
        state.child = Some(child);
        state.model_path = requested_model;
        drop(state);
        self.health_check(&base)?;
        Ok(base)
    }

    fn health_check(&self, base: &str) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(response) = self.client.get(format!("{base}/healthz")).send() {
                if response.status().is_success() {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(format!("synth-mlx-rl did not become healthy at {base}"));
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn forward(&self, method: &str, url: &str, body: Option<&Value>) -> SidecarResponse {
        match self.send(method, url, body) {
            Ok(response) => response_from_reqwest(response),
            Err(message) => error(502, &message),
        }
    }

    fn hosted_request(&self, method: &str, path: &str, body: Option<&Value>) -> SidecarResponse {
        let api_key = match env::var("SYNTH_API_KEY") {
            Ok(value) if !value.trim().is_empty() => value,
            _ => return error(401, "SYNTH_API_KEY is required for hosted SFT"),
        };
        let base =
            env::var("SYNTH_BACKEND_URL").unwrap_or_else(|_| "https://api.usesynth.ai".to_string());
        let url = format!("{}{}", base.trim_end_matches('/'), path);
        let method = match Method::from_bytes(method.as_bytes()) {
            Ok(value) => value,
            Err(_) => return error(500, "invalid hosted HTTP method"),
        };
        let mut request = self.client.request(method, url).bearer_auth(api_key);
        if let Some(value) = body {
            request = request.json(value);
        }
        match request.send() {
            Ok(response) => response_from_reqwest(response),
            Err(source) => error(502, &format!("hosted training request failed: {source}")),
        }
    }

    fn send(&self, method: &str, url: &str, body: Option<&Value>) -> Result<Response, String> {
        let method = Method::from_bytes(method.as_bytes())
            .map_err(|source| format!("invalid synth-mlx-rl HTTP method: {source}"))?;
        let mut request = self.client.request(method, url);
        if let Some(value) = body {
            request = request.json(value);
        }
        request
            .send()
            .map_err(|source| format!("synth-mlx-rl request failed: {source}"))
    }
}

impl Drop for SidecarState {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn mlx_command() -> Command {
    if let Ok(binary) = env::var("SYNTH_MLX_RL_BIN") {
        return Command::new(binary);
    }
    if let Ok(root) = env::var("SYNTH_MLX_RL_ROOT") {
        let mut command = Command::new("uv");
        command
            .arg("run")
            .arg("--directory")
            .arg(root)
            .arg("synth-mlx-rl");
        return command;
    }
    Command::new("synth-mlx-rl")
}

fn reserve_port() -> Result<u16, String> {
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .map_err(|source| format!("failed to reserve synth-mlx-rl port: {source}"))
}

fn take_model_path(config: &mut serde_json::Map<String, Value>) -> Option<PathBuf> {
    for key in ["managed_model_path", "model_path"] {
        if let Some(path) = config
            .remove(key)
            .and_then(|value| value.as_str().map(PathBuf::from))
        {
            return Some(path);
        }
    }
    None
}

fn new_job_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("training-{}-{nanos}", std::process::id())
}

fn query_suffix(query: &str) -> String {
    if query.is_empty() {
        String::new()
    } else {
        format!("?{query}")
    }
}

fn parse_optional_json(body: &[u8]) -> Option<Value> {
    if body.is_empty() {
        None
    } else {
        serde_json::from_slice(body).ok()
    }
}

fn response_from_reqwest(response: Response) -> SidecarResponse {
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    match response.bytes() {
        Ok(body) => SidecarResponse {
            status,
            content_type,
            body: body.to_vec(),
        },
        Err(source) => error(502, &format!("training response read failed: {source}")),
    }
}

fn response_json(response: &SidecarResponse) -> Result<Value, serde_json::Error> {
    serde_json::from_slice(&response.body)
}

fn json_value(status: u16, value: &Value) -> SidecarResponse {
    SidecarResponse {
        status,
        content_type: "application/json".to_string(),
        body: serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec()),
    }
}

fn error(status: u16, message: &str) -> SidecarResponse {
    json_value(status, &json!({"error": message}))
}

fn fs_create_dir_all(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path)
        .map_err(|source| format!("failed to create training sidecar root: {source}"))
}
