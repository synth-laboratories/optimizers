//! Background execution for async pipeline lane jobs.
//!
//! The GEPA driver is a single-threaded tick loop. Before this module existed
//! it also *executed* every leased runtime job inline on that tick, which meant
//! a propose lease and a rollout lease could be open at the same time while
//! never occupying wall clock at the same time. Lane leases modelled a
//! concurrency the executor never delivered, and that — not admission order —
//! is why FlashEvolve measured ~0.33s of propose/rollout overlap on the
//! 2026-06-02 Banking77 matrix.
//!
//! `LaneExecutorPool` moves job execution onto worker threads. The workspace
//! sqlite database is the shared state: a worker claims the job, runs it, and
//! writes the terminal job status back, so the driver's existing
//! `consume_async_lane_work` fold sees the result through the same path it
//! always did. Each worker owns a private `WorkspaceStore` and `RequestCache`
//! handle on the same files (both are WAL with a busy timeout), because
//! rusqlite `Connection` is `Send` but not `Sync`.
//!
//! The driver still does all state-machine work — folding outcomes, adaptive
//! concurrency, staleness, budgets — on the main thread. Workers only run the
//! job and report back.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use synth_optimizer_platform::{
    CacheAccessRecord, CacheCounters, CacheMode, ContainerClient, OptimizerError, RequestCache,
    Result, SynthOptimizerConfig, WorkspaceStore,
};

use crate::runtime::{self, RuntimeEffectExecutorConfig, RuntimeEffectOutcome};

/// A job the driver has leased and wants executed off the tick.
#[derive(Clone, Debug)]
pub(crate) struct LaneJobRequest {
    pub run_id: String,
    pub job_id: String,
    pub lane: String,
    pub lease_key: String,
    pub worker_id: String,
    pub lease_seconds: u64,
}

/// Everything the driver needs to resume main-thread bookkeeping for a job a
/// worker finished.
pub(crate) struct LaneJobCompletion {
    pub request: LaneJobRequest,
    pub outcome: Result<RuntimeEffectOutcome>,
    pub wall_seconds: f64,
    pub cache_counters: CacheCounters,
    pub cache_access_log: Vec<CacheAccessRecord>,
}

/// What a worker thread needs to build its private handles. Everything here is
/// `Send`; the handles themselves are constructed inside the worker.
#[derive(Clone)]
pub(crate) struct LaneWorkerResources {
    pub workspace_path: PathBuf,
    pub cache_path: PathBuf,
    pub cache_mode: CacheMode,
    pub cache_max_bytes: u64,
    pub config: SynthOptimizerConfig,
    pub client: ContainerClient,
}

/// Runs one lane job. Production uses [`RuntimeLaneJobRunner`]; tests inject a
/// synthetic runner so pool concurrency can be asserted without a container.
pub(crate) trait LaneJobRunner: Send {
    fn run(&mut self, request: &LaneJobRequest) -> Result<RuntimeEffectOutcome>;

    /// Hand the driver this worker's cache activity so the run manifest still
    /// reports a single cache boundary.
    fn drain_cache_activity(&mut self) -> (CacheCounters, Vec<CacheAccessRecord>) {
        (CacheCounters::default(), Vec::new())
    }
}

/// Builds a runner inside the worker thread it will run on.
pub(crate) type LaneJobRunnerFactory =
    Arc<dyn Fn(usize) -> Result<Box<dyn LaneJobRunner>> + Send + Sync>;

struct RuntimeLaneJobRunner {
    workspace: WorkspaceStore,
    cache: RequestCache,
    config: SynthOptimizerConfig,
    client: ContainerClient,
}

impl LaneJobRunner for RuntimeLaneJobRunner {
    fn run(&mut self, request: &LaneJobRequest) -> Result<RuntimeEffectOutcome> {
        runtime::execute_one_pending_optimizer_job_from_run_workspace(
            &self.workspace,
            &mut self.cache,
            &self.config,
            &self.client,
            &request.run_id,
            &request.job_id,
            RuntimeEffectExecutorConfig {
                worker_id: request.worker_id.clone(),
                lease_seconds: request.lease_seconds,
            },
        )
    }

    fn drain_cache_activity(&mut self) -> (CacheCounters, Vec<CacheAccessRecord>) {
        (self.cache.counters(), self.cache.take_access_log())
    }
}

pub(crate) fn runtime_lane_job_runner_factory(
    resources: LaneWorkerResources,
) -> LaneJobRunnerFactory {
    Arc::new(move |_worker_index| {
        let workspace = WorkspaceStore::open_existing(&resources.workspace_path)?;
        let cache = RequestCache::open_with_max_bytes(
            &resources.cache_path,
            resources.cache_mode,
            resources.cache_max_bytes,
        )?;
        Ok(Box::new(RuntimeLaneJobRunner {
            workspace,
            cache,
            config: resources.config.clone(),
            client: resources.client.clone(),
        }) as Box<dyn LaneJobRunner>)
    })
}

/// A dispatched job the driver is still waiting on.
#[derive(Clone, Debug)]
pub(crate) struct LaneDispatchRecord {
    pub lane: String,
    pub lease_key: String,
    pub dispatched_at: Instant,
}

pub(crate) struct LaneExecutorPool {
    submit: Option<Sender<LaneJobRequest>>,
    completions: Receiver<LaneJobCompletion>,
    handles: Vec<JoinHandle<()>>,
    in_flight: BTreeMap<String, LaneDispatchRecord>,
    /// Completions received off the channel but not yet folded by the driver.
    /// The job stays in `in_flight` until the driver takes it, because
    /// `consume_async_lane_work` uses that set to know a terminal job is not
    /// foldable yet.
    ready: VecDeque<LaneJobCompletion>,
    capacity: usize,
}

impl LaneExecutorPool {
    pub fn new(workers: usize, factory: LaneJobRunnerFactory) -> Self {
        let workers = workers.max(1);
        let (submit, requests) = mpsc::channel::<LaneJobRequest>();
        let (done, completions) = mpsc::channel::<LaneJobCompletion>();
        // One shared receiver, many workers: whichever worker is idle takes the
        // next request, which is exactly the lane-capacity semantics the driver
        // already enforces upstream via lane lease counts.
        let requests = Arc::new(std::sync::Mutex::new(requests));
        let mut handles = Vec::with_capacity(workers);
        for worker_index in 0..workers {
            let requests = Arc::clone(&requests);
            let done = done.clone();
            let factory = Arc::clone(&factory);
            handles.push(thread::spawn(move || {
                let mut runner = match factory(worker_index) {
                    Ok(runner) => runner,
                    Err(error) => {
                        // The worker cannot open its handles. Fail every request
                        // it would have taken rather than silently stalling the
                        // driver, which would otherwise wait forever.
                        loop {
                            let request = {
                                let Ok(guard) = requests.lock() else {
                                    return;
                                };
                                match guard.recv() {
                                    Ok(request) => request,
                                    Err(_) => return,
                                }
                            };
                            let message = format!(
                                "lane worker {worker_index} could not open its runtime handles: {error}"
                            );
                            if done
                                .send(LaneJobCompletion {
                                    request,
                                    outcome: Err(OptimizerError::Invariant(message)),
                                    wall_seconds: 0.0,
                                    cache_counters: CacheCounters::default(),
                                    cache_access_log: Vec::new(),
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                };
                loop {
                    let request = {
                        let Ok(guard) = requests.lock() else {
                            return;
                        };
                        match guard.recv() {
                            Ok(request) => request,
                            Err(_) => return,
                        }
                    };
                    let started = Instant::now();
                    let outcome = runner.run(&request);
                    let (cache_counters, cache_access_log) = runner.drain_cache_activity();
                    if done
                        .send(LaneJobCompletion {
                            request,
                            outcome,
                            wall_seconds: started.elapsed().as_secs_f64(),
                            cache_counters,
                            cache_access_log,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }));
        }
        Self {
            submit: Some(submit),
            completions,
            handles,
            in_flight: BTreeMap::new(),
            ready: VecDeque::new(),
            capacity: workers,
        }
    }

    pub fn production(workers: usize, resources: LaneWorkerResources) -> Self {
        Self::new(workers, runtime_lane_job_runner_factory(resources))
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    pub fn has_capacity(&self) -> bool {
        self.in_flight.len() < self.capacity
    }

    pub fn is_dispatched(&self, job_id: &str) -> bool {
        self.in_flight.contains_key(job_id)
    }

    pub fn dispatched_job_ids(&self) -> std::collections::BTreeSet<String> {
        self.in_flight.keys().cloned().collect()
    }

    /// Per-lane count of jobs currently executing on workers, plus how long the
    /// oldest has been out. Surfaced on the cursor so a stalled run shows which
    /// lane is holding it.
    pub fn in_flight_summary(&self) -> serde_json::Value {
        let mut by_lane: BTreeMap<String, usize> = BTreeMap::new();
        let mut oldest_seconds = 0.0f64;
        for record in self.in_flight.values() {
            *by_lane.entry(record.lane.clone()).or_insert(0) += 1;
            oldest_seconds = oldest_seconds.max(record.dispatched_at.elapsed().as_secs_f64());
        }
        serde_json::json!({
            "count": self.in_flight.len(),
            "capacity": self.capacity,
            "by_lane": by_lane,
            "oldest_dispatch_age_seconds": oldest_seconds,
            "lease_keys": self
                .in_flight
                .values()
                .map(|record| record.lease_key.clone())
                .collect::<Vec<_>>(),
        })
    }

    /// Hand a job to a worker. Returns `false` when the pool is saturated or
    /// the job is already dispatched, so the driver can fall through to other
    /// scheduling work instead of blocking.
    pub fn dispatch(&mut self, request: LaneJobRequest) -> Result<bool> {
        if !self.has_capacity() || self.in_flight.contains_key(&request.job_id) {
            return Ok(false);
        }
        let record = LaneDispatchRecord {
            lane: request.lane.clone(),
            lease_key: request.lease_key.clone(),
            dispatched_at: Instant::now(),
        };
        let job_id = request.job_id.clone();
        let Some(submit) = self.submit.as_ref() else {
            return Err(OptimizerError::Invariant(
                "lane executor pool is shut down".to_string(),
            ));
        };
        submit.send(request).map_err(|_| {
            OptimizerError::Invariant("lane executor pool workers are gone".to_string())
        })?;
        self.in_flight.insert(job_id, record);
        Ok(true)
    }

    /// Non-blocking poll for one finished job.
    pub fn try_take_completion(&mut self) -> Option<LaneJobCompletion> {
        self.drain_channel();
        let completion = self.ready.pop_front()?;
        self.in_flight.remove(&completion.request.job_id);
        Some(completion)
    }

    fn drain_channel(&mut self) {
        loop {
            match self.completions.try_recv() {
                Ok(completion) => self.ready.push_back(completion),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return,
            }
        }
    }

    /// Block up to `timeout` for a worker to finish something. Returns true
    /// when there is work for the driver to fold, so the run loop can react as
    /// soon as a lane frees up instead of sleeping a fixed interval.
    pub fn await_completion(&mut self, timeout: Duration) -> bool {
        self.drain_channel();
        if !self.ready.is_empty() {
            return true;
        }
        if self.in_flight.is_empty() {
            return false;
        }
        match self.completions.recv_timeout(timeout) {
            Ok(completion) => {
                self.ready.push_back(completion);
                true
            }
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => false,
        }
    }

    /// Stop accepting work and, when nothing is still executing, join the
    /// workers. If jobs are still in flight — the cancellation path — detach
    /// instead: joining there would block teardown for as long as a provider
    /// call takes, and the run is already terminalized. A late worker write
    /// lands in a WAL-mode sqlite file whose job rows nothing reads any more.
    pub fn shutdown(&mut self) {
        self.submit = None;
        if !self.in_flight.is_empty() {
            self.handles.clear();
            return;
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for LaneExecutorPool {
    fn drop(&mut self) {
        self.submit = None;
        // Do not join here: a worker mid-rollout would block teardown for as
        // long as the provider takes. Dropping the sender is enough to stop
        // idle workers, and the process owns the run.
        self.handles.clear();
    }
}

/// Wall-clock accounting for lane concurrency.
///
/// This is the number that decides whether FlashEvolve earns its name.
/// `overlap_seconds` is the time during which at least one propose lane job and
/// at least one rollout lane job were both executing — not merely both leased.
#[derive(Debug)]
pub(crate) struct LaneOverlapTracker {
    started_at: Instant,
    last_transition: Instant,
    active: BTreeMap<String, usize>,
    propose_busy_seconds: f64,
    rollout_busy_seconds: f64,
    evaluate_busy_seconds: f64,
    any_busy_seconds: f64,
    overlap_seconds: f64,
    max_concurrent_lanes: usize,
    dispatched_jobs: usize,
    stale_gap_total: u64,
    stale_gap_samples: u64,
}

impl Default for LaneOverlapTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LaneOverlapTracker {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            started_at: now,
            last_transition: now,
            active: BTreeMap::new(),
            propose_busy_seconds: 0.0,
            rollout_busy_seconds: 0.0,
            evaluate_busy_seconds: 0.0,
            any_busy_seconds: 0.0,
            overlap_seconds: 0.0,
            max_concurrent_lanes: 0,
            dispatched_jobs: 0,
            stale_gap_total: 0,
            stale_gap_samples: 0,
        }
    }

    fn accrue(&mut self, now: Instant) {
        let delta = now
            .saturating_duration_since(self.last_transition)
            .as_secs_f64();
        self.last_transition = now;
        if delta <= 0.0 {
            return;
        }
        let propose = self.active.get("propose").copied().unwrap_or(0);
        let rollout = self.active.get("rollout").copied().unwrap_or(0);
        let evaluate = self.active.get("evaluate").copied().unwrap_or(0);
        if propose > 0 {
            self.propose_busy_seconds += delta;
        }
        if rollout > 0 {
            self.rollout_busy_seconds += delta;
        }
        if evaluate > 0 {
            self.evaluate_busy_seconds += delta;
        }
        if propose + rollout + evaluate > 0 {
            self.any_busy_seconds += delta;
        }
        if propose > 0 && rollout > 0 {
            self.overlap_seconds += delta;
        }
    }

    pub fn lane_started(&mut self, lane: &str) {
        self.started_at_now();
        let entry = self.active.entry(lane.to_string()).or_insert(0);
        *entry += 1;
        self.dispatched_jobs += 1;
        let concurrent: usize = self.active.values().sum();
        self.max_concurrent_lanes = self.max_concurrent_lanes.max(concurrent);
    }

    pub fn lane_finished(&mut self, lane: &str) {
        self.started_at_now();
        if let Some(entry) = self.active.get_mut(lane) {
            *entry = entry.saturating_sub(1);
        }
    }

    fn started_at_now(&mut self) {
        let now = Instant::now();
        self.accrue(now);
    }

    pub fn record_stale_gap(&mut self, stale_gap: u64) {
        self.stale_gap_total = self.stale_gap_total.saturating_add(stale_gap);
        self.stale_gap_samples = self.stale_gap_samples.saturating_add(1);
    }

    pub fn snapshot(&mut self) -> LaneOverlapSnapshot {
        self.started_at_now();
        let wall_seconds = self
            .last_transition
            .saturating_duration_since(self.started_at)
            .as_secs_f64();
        LaneOverlapSnapshot {
            wall_seconds,
            propose_busy_seconds: self.propose_busy_seconds,
            rollout_busy_seconds: self.rollout_busy_seconds,
            evaluate_busy_seconds: self.evaluate_busy_seconds,
            lane_busy_seconds: self.any_busy_seconds,
            overlap_seconds: self.overlap_seconds,
            overlap_ratio: if self.any_busy_seconds > 0.0 {
                self.overlap_seconds / self.any_busy_seconds
            } else {
                0.0
            },
            max_concurrent_lane_jobs: self.max_concurrent_lanes,
            dispatched_lane_jobs: self.dispatched_jobs,
            mean_stale_gap: if self.stale_gap_samples > 0 {
                self.stale_gap_total as f64 / self.stale_gap_samples as f64
            } else {
                0.0
            },
            stale_gap_samples: self.stale_gap_samples,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LaneOverlapSnapshot {
    pub wall_seconds: f64,
    pub propose_busy_seconds: f64,
    pub rollout_busy_seconds: f64,
    pub evaluate_busy_seconds: f64,
    pub lane_busy_seconds: f64,
    pub overlap_seconds: f64,
    pub overlap_ratio: f64,
    pub max_concurrent_lane_jobs: usize,
    pub dispatched_lane_jobs: usize,
    pub mean_stale_gap: f64,
    pub stale_gap_samples: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SleepRunner {
        sleep: Duration,
        concurrent: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    impl LaneJobRunner for SleepRunner {
        fn run(&mut self, _request: &LaneJobRequest) -> Result<RuntimeEffectOutcome> {
            let now = self.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(self.sleep);
            self.concurrent.fetch_sub(1, Ordering::SeqCst);
            Ok(RuntimeEffectOutcome::Proposer(Box::new(
                runtime::RuntimeProposerOutcome {
                    response: json!({}),
                    proposals: Vec::new(),
                    usage: Default::default(),
                    cost_usd: 0.0,
                    backend: "test".to_string(),
                    runtime_substrate: "test".to_string(),
                    workspace: None,
                    evidence_warnings: Vec::new(),
                    cache_key: String::new(),
                    cache_hit: false,
                },
            )))
        }
    }

    fn request(job_id: &str, lane: &str) -> LaneJobRequest {
        LaneJobRequest {
            run_id: "run".to_string(),
            job_id: job_id.to_string(),
            lane: lane.to_string(),
            lease_key: format!("{lane}:{job_id}"),
            worker_id: "test".to_string(),
            lease_seconds: 60,
        }
    }

    #[test]
    fn pool_runs_propose_and_rollout_jobs_concurrently() {
        let concurrent = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let factory: LaneJobRunnerFactory = {
            let concurrent = Arc::clone(&concurrent);
            let peak = Arc::clone(&peak);
            Arc::new(move |_index| {
                Ok(Box::new(SleepRunner {
                    sleep: Duration::from_millis(300),
                    concurrent: Arc::clone(&concurrent),
                    peak: Arc::clone(&peak),
                }) as Box<dyn LaneJobRunner>)
            })
        };
        let mut pool = LaneExecutorPool::new(4, factory);
        let started = Instant::now();
        assert!(pool.dispatch(request("job-rollout", "rollout")).unwrap());
        assert!(pool.dispatch(request("job-propose", "propose")).unwrap());
        assert_eq!(pool.in_flight_count(), 2);

        let mut completed = 0;
        while completed < 2 {
            if pool.await_completion(Duration::from_secs(5)) && pool.try_take_completion().is_some()
            {
                completed += 1;
            } else {
                panic!("lane pool did not report completions");
            }
        }
        let elapsed = started.elapsed();
        assert_eq!(peak.load(Ordering::SeqCst), 2, "lanes did not run at once");
        assert!(
            elapsed < Duration::from_millis(550),
            "two 300ms lane jobs took {elapsed:?}; they ran serially"
        );
        pool.shutdown();
    }

    #[test]
    fn pool_refuses_dispatch_beyond_capacity() {
        let concurrent = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let factory: LaneJobRunnerFactory = {
            let concurrent = Arc::clone(&concurrent);
            let peak = Arc::clone(&peak);
            Arc::new(move |_index| {
                Ok(Box::new(SleepRunner {
                    sleep: Duration::from_millis(50),
                    concurrent: Arc::clone(&concurrent),
                    peak: Arc::clone(&peak),
                }) as Box<dyn LaneJobRunner>)
            })
        };
        let mut pool = LaneExecutorPool::new(1, factory);
        assert!(pool.dispatch(request("a", "rollout")).unwrap());
        assert!(!pool.dispatch(request("b", "rollout")).unwrap());
        assert!(!pool.dispatch(request("a", "rollout")).unwrap());
        assert!(pool.await_completion(Duration::from_secs(5)));
        assert!(pool.try_take_completion().is_some());
        assert!(pool.dispatch(request("b", "rollout")).unwrap());
        pool.shutdown();
    }

    #[test]
    fn overlap_tracker_counts_only_simultaneous_propose_and_rollout() {
        let mut tracker = LaneOverlapTracker::new();
        // Serial: rollout, then propose. No overlap.
        tracker.lane_started("rollout");
        thread::sleep(Duration::from_millis(60));
        tracker.lane_finished("rollout");
        tracker.lane_started("propose");
        thread::sleep(Duration::from_millis(60));
        tracker.lane_finished("propose");
        let serial = tracker.snapshot();
        assert!(
            serial.overlap_seconds < 0.01,
            "serial lanes reported {}s overlap",
            serial.overlap_seconds
        );

        // Overlapped: both lanes live at once.
        let mut tracker = LaneOverlapTracker::new();
        tracker.lane_started("rollout");
        tracker.lane_started("propose");
        thread::sleep(Duration::from_millis(80));
        tracker.lane_finished("propose");
        tracker.lane_finished("rollout");
        let overlapped = tracker.snapshot();
        assert!(
            overlapped.overlap_seconds > 0.05,
            "overlapping lanes reported only {}s overlap",
            overlapped.overlap_seconds
        );
        assert_eq!(overlapped.max_concurrent_lane_jobs, 2);
        assert!(overlapped.overlap_ratio > 0.8);
    }
}
