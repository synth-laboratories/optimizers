#![allow(clippy::useless_conversion, unexpected_cfgs)]

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};
use synth_optimizer_platform::{
    compare_normalized_event_feeds, replay_event_feed, GepaRunResult as RustGepaRunResult,
    OptimizerError, WorkspaceStore,
};

create_exception!(synth_optimizers_py, SynthOptimizerError, PyRuntimeError);
create_exception!(synth_optimizers_py, ConfigError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    ContainerContractError,
    SynthOptimizerError
);
create_exception!(synth_optimizers_py, ProposerError, SynthOptimizerError);
create_exception!(synth_optimizers_py, CacheMissError, SynthOptimizerError);
create_exception!(synth_optimizers_py, CacheFullError, SynthOptimizerError);
create_exception!(synth_optimizers_py, CacheCorruptError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    BudgetExceededError,
    SynthOptimizerError
);
create_exception!(synth_optimizers_py, CancelledError, SynthOptimizerError);
create_exception!(synth_optimizers_py, EventCompareError, SynthOptimizerError);
create_exception!(synth_optimizers_py, RunFailedError, SynthOptimizerError);
create_exception!(synth_optimizers_py, InvariantError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    StateTransitionError,
    SynthOptimizerError
);
create_exception!(synth_optimizers_py, OptimizerIoError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    OptimizerDiskBudgetError,
    SynthOptimizerError
);
create_exception!(synth_optimizers_py, OptimizerJsonError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    OptimizerTomlDecodeError,
    SynthOptimizerError
);
create_exception!(synth_optimizers_py, OptimizerHttpError, SynthOptimizerError);
create_exception!(
    synth_optimizers_py,
    OptimizerSqliteError,
    SynthOptimizerError
);

#[pyclass]
pub struct GepaRun {
    config_path: String,
}

#[pymethods]
impl GepaRun {
    #[staticmethod]
    pub fn from_toml(path: &str) -> Self {
        Self {
            config_path: path.to_string(),
        }
    }

    pub fn execute(&self) -> PyResult<GepaRunResult> {
        synth_gepa::execute_gepa_from_toml(&self.config_path)
            .map(GepaRunResult::from)
            .map_err(py_error)
    }

    #[getter]
    pub fn config_path(&self) -> String {
        self.config_path.clone()
    }
}

#[pyclass]
pub struct GepaRunResult {
    inner: RustGepaRunResult,
}

#[pymethods]
impl GepaRunResult {
    #[getter]
    pub fn best_candidate(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.inner.best_candidate)
    }

    #[getter]
    pub fn manifest_path(&self) -> String {
        self.inner.manifest_path.clone()
    }

    #[getter]
    pub fn event_feed_path(&self) -> String {
        self.inner.event_feed_path.clone()
    }

    #[getter]
    pub fn normalized_event_feed_path(&self) -> String {
        self.inner.normalized_event_feed_path.clone()
    }

    #[getter]
    pub fn cache_profile_path(&self) -> String {
        self.inner.cache_profile_path.clone()
    }

    #[getter]
    pub fn candidate_registry_path(&self) -> String {
        self.inner.candidate_registry_path.clone()
    }

    #[getter]
    pub fn frontier_path(&self) -> String {
        self.inner.frontier_path.clone()
    }

    #[getter]
    pub fn score_chart_path(&self) -> String {
        self.inner.score_chart_path.clone()
    }

    #[getter]
    pub fn run_registry_path(&self) -> String {
        self.inner.run_registry_path.clone()
    }

    #[getter]
    pub fn workspace_db_path(&self) -> String {
        self.inner.workspace_db_path.clone()
    }

    #[getter]
    pub fn artifact_refs(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner.artifact_refs)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        value_to_py(py, &value)
    }

    #[getter]
    pub fn cost_usd(&self) -> f64 {
        self.inner.cost_usd
    }

    #[getter]
    pub fn usage(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.inner.usage)
    }

    #[getter]
    pub fn state_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.inner.state_history)
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "best_candidate",
            value_to_py(py, &self.inner.best_candidate)?,
        )?;
        dict.set_item("manifest_path", self.inner.manifest_path.clone())?;
        dict.set_item("event_feed_path", self.inner.event_feed_path.clone())?;
        dict.set_item(
            "normalized_event_feed_path",
            self.inner.normalized_event_feed_path.clone(),
        )?;
        dict.set_item("cache_profile_path", self.inner.cache_profile_path.clone())?;
        dict.set_item(
            "candidate_registry_path",
            self.inner.candidate_registry_path.clone(),
        )?;
        dict.set_item("frontier_path", self.inner.frontier_path.clone())?;
        dict.set_item("score_chart_path", self.inner.score_chart_path.clone())?;
        dict.set_item("run_registry_path", self.inner.run_registry_path.clone())?;
        dict.set_item("workspace_db_path", self.inner.workspace_db_path.clone())?;
        dict.set_item("artifact_refs", self.artifact_refs(py)?)?;
        dict.set_item("cost_usd", self.inner.cost_usd)?;
        dict.set_item("usage", value_to_py(py, &self.inner.usage)?)?;
        dict.set_item("state_history", value_to_py(py, &self.inner.state_history)?)?;
        Ok(dict.into())
    }
}

impl From<RustGepaRunResult> for GepaRunResult {
    fn from(inner: RustGepaRunResult) -> Self {
        Self { inner }
    }
}

#[pyfunction]
pub fn events_replay(path: &str) -> PyResult<String> {
    replay_event_feed(path).map_err(py_error)
}

#[pyfunction]
pub fn events_compare(left: &str, right: &str) -> PyResult<bool> {
    compare_normalized_event_feeds(left, right)
        .map(|_| true)
        .map_err(py_error)
}

#[pyfunction]
pub fn workspace_status(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let status = WorkspaceStore::open_existing(path)
        .and_then(|store| store.status())
        .map_err(py_error)?;
    let value =
        serde_json::to_value(status).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, config_path, priority=0))]
pub fn workspace_submit_run_request(
    py: Python<'_>,
    db_path: &str,
    config_path: &str,
    priority: i64,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open(db_path)
        .and_then(|store| store.submit_run_request(config_path, priority))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, lease_id, worker_id=None, lease_seconds=3600))]
pub fn workspace_claim_next_run_request(
    py: Python<'_>,
    db_path: &str,
    lease_id: &str,
    worker_id: Option<&str>,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.claim_next_run_request(lease_id, worker_id, lease_seconds))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, request_id, lease_id, lease_seconds=3600))]
pub fn workspace_heartbeat_run_request(
    py: Python<'_>,
    db_path: &str,
    request_id: &str,
    lease_id: &str,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.heartbeat_run_request(request_id, lease_id, lease_seconds))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction]
pub fn workspace_start_run_request(
    py: Python<'_>,
    db_path: &str,
    request_id: &str,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.mark_run_request_started(request_id))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction]
pub fn workspace_complete_run_request(
    py: Python<'_>,
    db_path: &str,
    request_id: &str,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.mark_run_request_completed(request_id))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, request_id, error_message, reason_code=None))]
pub fn workspace_fail_run_request(
    py: Python<'_>,
    db_path: &str,
    request_id: &str,
    error_message: &str,
    reason_code: Option<&str>,
) -> PyResult<PyObject> {
    let error = json!({
        "message": error_message,
        "reason_code": reason_code.unwrap_or("run_request_failed"),
    });
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.mark_run_request_failed(request_id, &error))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, request_id, reason="cancelled"))]
pub fn workspace_cancel_run_request(
    py: Python<'_>,
    db_path: &str,
    request_id: &str,
    reason: &str,
) -> PyResult<PyObject> {
    let request = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.mark_run_request_cancelled(request_id, reason))
        .map_err(py_error)?;
    let value = serde_json::to_value(request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction]
pub fn workspace_recover_expired_run_requests(py: Python<'_>, db_path: &str) -> PyResult<PyObject> {
    let requests = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.recover_expired_run_requests())
        .map_err(py_error)?;
    let value = serde_json::to_value(requests)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, run_id, lease_id, worker_id=None, lease_seconds=300))]
pub fn workspace_claim_next_optimizer_job(
    py: Python<'_>,
    db_path: &str,
    run_id: &str,
    lease_id: &str,
    worker_id: Option<&str>,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let job = WorkspaceStore::open_existing(db_path)
        .and_then(|store| {
            store.claim_next_optimizer_job(run_id, lease_id, worker_id, lease_seconds)
        })
        .map_err(py_error)?;
    let value =
        serde_json::to_value(job).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, run_id, job_id, lease_id, worker_id=None, lease_seconds=300))]
pub fn workspace_claim_optimizer_job(
    py: Python<'_>,
    db_path: &str,
    run_id: &str,
    job_id: &str,
    lease_id: &str,
    worker_id: Option<&str>,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let job = WorkspaceStore::open_existing(db_path)
        .and_then(|store| {
            store.claim_optimizer_job(run_id, job_id, lease_id, worker_id, lease_seconds)
        })
        .map_err(py_error)?;
    let value =
        serde_json::to_value(job).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, run_id, job_id, lease_id, lease_seconds=300))]
pub fn workspace_mark_optimizer_job_running(
    py: Python<'_>,
    db_path: &str,
    run_id: &str,
    job_id: &str,
    lease_id: &str,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let job = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.mark_optimizer_job_running(run_id, job_id, lease_id, lease_seconds))
        .map_err(py_error)?;
    let value =
        serde_json::to_value(job).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, run_id, job_id, lease_id, lease_seconds=300))]
pub fn workspace_heartbeat_optimizer_job(
    py: Python<'_>,
    db_path: &str,
    run_id: &str,
    job_id: &str,
    lease_id: &str,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let job = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.heartbeat_optimizer_job(run_id, job_id, lease_id, lease_seconds))
        .map_err(py_error)?;
    let value =
        serde_json::to_value(job).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction]
pub fn workspace_recover_expired_optimizer_jobs(
    py: Python<'_>,
    db_path: &str,
    run_id: &str,
) -> PyResult<PyObject> {
    let jobs = WorkspaceStore::open_existing(db_path)
        .and_then(|store| store.recover_expired_optimizer_jobs(run_id))
        .map_err(py_error)?;
    let value =
        serde_json::to_value(jobs).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, bind_addr="127.0.0.1:8879", worker_id=None, lease_seconds=3600))]
pub fn gepa_serve(
    db_path: &str,
    bind_addr: &str,
    worker_id: Option<&str>,
    lease_seconds: u64,
) -> PyResult<()> {
    let mut config = synth_gepa::service::GepaServiceConfig::new(db_path, bind_addr);
    if let Some(worker_id) = worker_id {
        config.worker_id = worker_id.to_string();
    }
    config.lease_seconds = lease_seconds;
    synth_gepa::service::run_gepa_service(config).map_err(py_error)
}

#[pyfunction(signature = (db_path, worker_id="synth-gepa-worker", lease_seconds=3600))]
pub fn gepa_service_run_next(
    py: Python<'_>,
    db_path: &str,
    worker_id: &str,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let outcome = synth_gepa::service::run_next_queued_request(db_path, worker_id, lease_seconds)
        .map_err(py_error)?;
    let value = serde_json::to_value(outcome)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(signature = (db_path, worker_id="synth-gepa-worker", lease_seconds=3600))]
pub fn gepa_service_tick(
    py: Python<'_>,
    db_path: &str,
    worker_id: &str,
    lease_seconds: u64,
) -> PyResult<PyObject> {
    let outcome =
        synth_gepa::service::tick_next_unit(db_path, worker_id, lease_seconds).map_err(py_error)?;
    let value = serde_json::to_value(outcome)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction]
pub fn gepa_service_recover(py: Python<'_>, db_path: &str) -> PyResult<PyObject> {
    let outcome = synth_gepa::service::recover_service_state(db_path).map_err(py_error)?;
    let value = serde_json::to_value(outcome)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    value_to_py(py, &value)
}

#[pymodule]
fn _synth_optimizers(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GepaRun>()?;
    module.add_class::<GepaRunResult>()?;
    add_exception_class::<SynthOptimizerError>(
        py,
        module,
        "SynthOptimizerError",
        "synth_optimizer_error",
    )?;
    add_exception_class::<ConfigError>(py, module, "ConfigError", "synth_optimizer_config_error")?;
    add_exception_class::<ContainerContractError>(
        py,
        module,
        "ContainerContractError",
        "synth_optimizer_container_error",
    )?;
    add_exception_class::<ProposerError>(
        py,
        module,
        "ProposerError",
        "synth_optimizer_proposer_error",
    )?;
    add_exception_class::<CacheMissError>(
        py,
        module,
        "CacheMissError",
        "synth_optimizer_cache_miss",
    )?;
    add_exception_class::<CacheFullError>(
        py,
        module,
        "CacheFullError",
        "synth_optimizer_cache_full",
    )?;
    add_exception_class::<CacheCorruptError>(
        py,
        module,
        "CacheCorruptError",
        "synth_optimizer_cache_corrupt",
    )?;
    add_exception_class::<BudgetExceededError>(
        py,
        module,
        "BudgetExceededError",
        "synth_optimizer_budget_exceeded",
    )?;
    add_exception_class::<CancelledError>(
        py,
        module,
        "CancelledError",
        "synth_optimizer_cancelled",
    )?;
    add_exception_class::<EventCompareError>(
        py,
        module,
        "EventCompareError",
        "synth_optimizer_event_compare_failed",
    )?;
    add_exception_class::<RunFailedError>(py, module, "RunFailedError", "synth_optimizer_failed")?;
    add_exception_class::<InvariantError>(
        py,
        module,
        "InvariantError",
        "synth_optimizer_invariant_error",
    )?;
    add_exception_class::<StateTransitionError>(
        py,
        module,
        "StateTransitionError",
        "synth_optimizer_state_transition_error",
    )?;
    add_exception_class::<OptimizerIoError>(
        py,
        module,
        "OptimizerIoError",
        "synth_optimizer_io_error",
    )?;
    add_exception_class::<OptimizerDiskBudgetError>(
        py,
        module,
        "OptimizerDiskBudgetError",
        "synth_optimizer_disk_budget_exceeded",
    )?;
    add_exception_class::<OptimizerJsonError>(
        py,
        module,
        "OptimizerJsonError",
        "synth_optimizer_json_error",
    )?;
    add_exception_class::<OptimizerTomlDecodeError>(
        py,
        module,
        "OptimizerTomlDecodeError",
        "synth_optimizer_toml_decode_error",
    )?;
    add_exception_class::<OptimizerHttpError>(
        py,
        module,
        "OptimizerHttpError",
        "synth_optimizer_http_error",
    )?;
    add_exception_class::<OptimizerSqliteError>(
        py,
        module,
        "OptimizerSqliteError",
        "synth_optimizer_sqlite_error",
    )?;
    module.add_function(wrap_pyfunction!(events_replay, module)?)?;
    module.add_function(wrap_pyfunction!(events_compare, module)?)?;
    module.add_function(wrap_pyfunction!(gepa_serve, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

fn add_exception_class<T>(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    name: &str,
    error_code: &str,
) -> PyResult<()>
where
    T: pyo3::type_object::PyTypeInfo,
{
    let error_type = py.get_type_bound::<T>();
    error_type.setattr("error_code", error_code)?;
    module.add(name, error_type)
}

fn py_error(error: OptimizerError) -> PyErr {
    let message = format!("{}: {}", error.error_code(), error);
    match error {
        OptimizerError::Config(_) => ConfigError::new_err(message),
        OptimizerError::Container(_) => ContainerContractError::new_err(message),
        OptimizerError::Proposer(_) => ProposerError::new_err(message),
        OptimizerError::CacheMiss { .. } => CacheMissError::new_err(message),
        OptimizerError::CacheFull { .. } => CacheFullError::new_err(message),
        OptimizerError::CacheCorrupt { .. } => CacheCorruptError::new_err(message),
        OptimizerError::BudgetExceeded { .. } => BudgetExceededError::new_err(message),
        OptimizerError::Cancelled { .. } => CancelledError::new_err(message),
        OptimizerError::EventCompare(_) => EventCompareError::new_err(message),
        OptimizerError::Failed(_) => RunFailedError::new_err(message),
        OptimizerError::Invariant(_) => InvariantError::new_err(message),
        OptimizerError::StateTransition { .. } => StateTransitionError::new_err(message),
        OptimizerError::Io { .. } => OptimizerIoError::new_err(message),
        OptimizerError::DiskBudgetExceeded { .. } => OptimizerDiskBudgetError::new_err(message),
        OptimizerError::Json(_) => OptimizerJsonError::new_err(message),
        OptimizerError::TomlDecode(_) => OptimizerTomlDecodeError::new_err(message),
        OptimizerError::Http(_) => OptimizerHttpError::new_err(message),
        OptimizerError::Sqlite(_) => OptimizerSqliteError::new_err(message),
    }
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(item) => Ok(item.into_py(py)),
        Value::Number(number) => {
            if let Some(item) = number.as_i64() {
                Ok(item.into_py(py))
            } else if let Some(item) = number.as_u64() {
                Ok(item.into_py(py))
            } else {
                Ok(number.as_f64().unwrap_or(0.0).into_py(py))
            }
        }
        Value::String(item) => Ok(item.into_py(py)),
        Value::Array(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in map {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            Ok(dict.into())
        }
    }
}
