use super::{
    checkpoint::{checkpoint_path, write_checkpoint},
    paths::{
        atomic_write, ensure_safe_directory, ensure_safe_file_path, ensure_safe_read_file_path,
        exclusive_write, task_path,
    },
    simulation::{run_task, TaskRuntime},
    validate_safe_component, Job, JobAssignment, JobResult, MonteCarlo, Task, TaskResult,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    error::Error,
    fmt, fs,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunOptions {
    pub assignment: Option<JobAssignment>,
    pub checkpoint_dir: Option<PathBuf>,
    pub resume: bool,
    pub checkpoint_interval: usize,
    pub output_path: Option<PathBuf>,
    pub sweep_limit: Option<usize>,
    /// Wall-clock limit for this run. `Some` overrides the `Runner` default.
    pub deadline: Option<Duration>,
    /// Wall-clock checkpoint interval for this run. `Some` overrides the `Runner` default.
    pub checkpoint_interval_time: Option<Duration>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult<P> {
    pub result: JobResult<P>,
    pub output_path: Option<PathBuf>,
    pub checkpoint_paths: Vec<PathBuf>,
    pub stopped_early: bool,
}
#[derive(Debug)]
#[non_exhaustive]
pub enum GenericJobError {
    InvalidTask {
        task: String,
        reason: &'static str,
    },
    InvalidAssignment {
        rank: usize,
        world_size: usize,
    },
    InvalidMeasurement {
        observable: String,
        reason: &'static str,
    },
    Model {
        task: String,
        source: Box<dyn Error + Send + Sync>,
    },
    Io {
        operation: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: Option<PathBuf>,
        source: serde_json::Error,
    },
    Hdf5 {
        path: PathBuf,
        reason: String,
    },
    Schema {
        path: PathBuf,
        reason: &'static str,
    },
    CheckpointMismatch {
        path: PathBuf,
        task: String,
    },
    InsufficientSamples {
        task: String,
        observable: String,
    },
    UnsafePath(String),
    Merge(&'static str),
}
impl GenericJobError {
    pub(crate) fn io(op: &'static str, p: &Path, e: std::io::Error) -> Self {
        Self::Io {
            operation: op,
            path: p.into(),
            source: e,
        }
    }
    pub(crate) fn json(p: Option<&Path>, e: serde_json::Error) -> Self {
        Self::Json {
            path: p.map(Into::into),
            source: e,
        }
    }
}
impl fmt::Display for GenericJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask { task, reason } => write!(f, "invalid task {task:?}: {reason}"),
            Self::InvalidAssignment { rank, world_size } => {
                write!(f, "invalid assignment {rank}/{world_size}")
            }
            Self::InvalidMeasurement { observable, reason } => {
                write!(f, "invalid measurement {observable:?}: {reason}")
            }
            Self::Model { task, source } => write!(f, "model error in {task:?}: {source}"),
            Self::Io {
                operation,
                path,
                source,
            } => write!(f, "failed to {operation} {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "JSON error at {path:?}: {source}"),
            Self::Hdf5 { path, reason } => write!(f, "HDF5 error at {}: {reason}", path.display()),
            Self::Schema { path, reason } => {
                write!(f, "invalid schema at {}: {reason}", path.display())
            }
            Self::CheckpointMismatch { path, task } => {
                write!(f, "checkpoint {} mismatches {task:?}", path.display())
            }
            Self::InsufficientSamples { task, observable } => {
                write!(f, "no complete bin for {task:?}/{observable:?}")
            }
            Self::UnsafePath(p) => write!(f, "unsafe path component {p:?}"),
            Self::Merge(r) => write!(f, "invalid result merge: {r}"),
        }
    }
}
impl Error for GenericJobError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Model { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduling {
    Static,
    Dynamic,
}
const DONE_SCHEMA_VERSION: u32 = 1;
const CLAIM_SCHEMA_VERSION: u32 = 1;
static CLAIM_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DoneMarker<P> {
    schema_version: u32,
    job_name: String,
    task_index: usize,
    task: Task<P>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ClaimLease {
    schema_version: u32,
    token: String,
    pid: u32,
    renewed_unix_nanos: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ClaimHeartbeat {
    schema_version: u32,
    token: String,
    renewed_unix_nanos: u128,
}

#[derive(Debug)]
struct OwnedClaim {
    task_index: usize,
    token: String,
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunnerDurationConfig {
    pub deadline: Option<Duration>,
    pub checkpoint_interval: Option<Duration>,
}

#[derive(Debug)]
pub struct Runner<M: MonteCarlo> {
    stale_claim_after: Duration,
    scheduling: Scheduling,
    duration_config: RunnerDurationConfig,
    model: PhantomData<fn() -> M>,
}
impl<M: MonteCarlo> Default for Runner<M> {
    fn default() -> Self {
        Self::new()
    }
}
impl<M: MonteCarlo> Runner<M> {
    pub fn new() -> Self {
        Self {
            stale_claim_after: Duration::from_secs(3600),
            scheduling: Scheduling::Static,
            duration_config: RunnerDurationConfig::default(),
            model: PhantomData,
        }
    }
    pub fn stale_claim_after(mut self, d: Duration) -> Self {
        self.stale_claim_after = d;
        self
    }
    pub fn set_stale_claim_after(&mut self, d: Duration) -> &mut Self {
        self.stale_claim_after = d;
        self
    }
    pub fn dynamic(mut self) -> Self {
        self.scheduling = Scheduling::Dynamic;
        self
    }
    pub fn scheduling(mut self, s: Scheduling) -> Self {
        self.scheduling = s;
        self
    }
    pub fn duration_config(mut self, config: RunnerDurationConfig) -> Self {
        self.duration_config = config;
        self
    }
    pub fn deadline(mut self, deadline: Duration) -> Self {
        self.duration_config.deadline = Some(deadline);
        self
    }
    pub fn checkpoint_every(mut self, interval: Duration) -> Self {
        self.duration_config.checkpoint_interval = Some(interval);
        self
    }
    pub fn run(
        &self,
        job: &Job<M>,
        o: &RunOptions,
    ) -> Result<RunResult<M::Parameters>, GenericJobError> {
        validate(job, o)?;
        let a = o.assignment.unwrap_or_else(JobAssignment::single);
        let started = Instant::now();
        let deadline = o
            .deadline
            .or(self.duration_config.deadline)
            .and_then(|duration| started.checked_add(duration));
        let checkpoint_interval_time = o
            .checkpoint_interval_time
            .or(self.duration_config.checkpoint_interval);
        let mut out = vec![];
        let mut cps = vec![];
        let mut budget = o.sweep_limit.unwrap_or(usize::MAX);
        let mut stopped = false;
        let mut static_indices = (0..job.tasks.len()).filter(|i| i % a.world_size == a.rank);
        loop {
            if budget == 0 || deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                stopped = true;
                break;
            }
            let (i, claim) = match self.scheduling {
                Scheduling::Static => match static_indices.next() {
                    Some(i) => (i, None),
                    None => break,
                },
                Scheduling::Dynamic => match self.claim_one(job, o, a)? {
                    Some(claim) => (claim.task_index, Some(claim)),
                    None => break,
                },
            };
            let cp = o.checkpoint_dir.as_ref().map(|d| checkpoint_path(d, i));
            let mut last_checkpoint = Instant::now();
            let run = run_task::<M, _>(
                &job.tasks[i],
                TaskRuntime {
                    job_name: &job.name,
                    task_index: i,
                    assignment: a,
                    checkpoint_path: cp.as_deref(),
                    restore_checkpoint: o.resume || self.scheduling == Scheduling::Dynamic,
                    allow_different_rank: self.scheduling == Scheduling::Dynamic,
                    sweep_budget: budget,
                    deadline,
                },
                |checkpoint, used, final_checkpoint| {
                    if let Some(claim) = claim.as_ref() {
                        self.renew_claim(claim)?;
                    }
                    let duration_due = checkpoint_interval_time
                        .is_some_and(|interval| last_checkpoint.elapsed() >= interval);
                    let sweep_due =
                        o.checkpoint_interval > 0 && used > 0 && used % o.checkpoint_interval == 0;
                    if (final_checkpoint || sweep_due || duration_due) && cp.is_some() {
                        write_checkpoint(cp.as_deref().unwrap(), checkpoint)?;
                        last_checkpoint = Instant::now();
                    }
                    Ok(())
                },
            );
            let (r, used) = match run {
                Ok(result) => result,
                Err(error) => {
                    if let Some(claim) = claim.as_ref() {
                        self.release_claim(claim)?;
                    }
                    return Err(error);
                }
            };
            budget = budget.saturating_sub(used);
            stopped |= !r.completed;
            if let Some(p) = cp {
                cps.push(p)
            }
            if let Some(claim) = claim.as_ref() {
                self.finish_claim(job, o, claim, &r)?
            }
            out.push(r);
            if stopped {
                break;
            }
        }
        let result = JobResult {
            job_name: job.name.clone(),
            rank: a.rank,
            world_size: a.world_size,
            tasks: out,
        };
        if let Some(p) = &o.output_path {
            if p.extension().and_then(|x| x.to_str()) == Some("h5")
                || p.file_name().and_then(|x| x.to_str()) == Some("result")
            {
                result.write_hdf5(p)?
            } else {
                result.write_json(p)?
            }
        }
        Ok(RunResult {
            result,
            output_path: o.output_path.clone(),
            checkpoint_paths: cps,
            stopped_early: stopped,
        })
    }
    fn claim_one(
        &self,
        job: &Job<M>,
        o: &RunOptions,
        assignment: JobAssignment,
    ) -> Result<Option<OwnedClaim>, GenericJobError> {
        let d = o.checkpoint_dir.as_ref().ok_or(GenericJobError::Merge(
            "dynamic scheduling requires checkpoint_dir",
        ))?;
        ensure_safe_directory(d)?;
        for i in dynamic_task_order(job.tasks.len(), assignment) {
            let task = &job.tasks[i];
            let done = done_path(d, i, 0);
            match read_optional_safe(&done, "read done marker")? {
                Some(payload) => {
                    let marker: DoneMarker<M::Parameters> = serde_json::from_slice(&payload)
                        .map_err(|_| GenericJobError::Schema {
                            path: done.clone(),
                            reason: "invalid done marker",
                        })?;
                    if marker.schema_version != DONE_SCHEMA_VERSION
                        || marker.job_name != job.name
                        || marker.task_index != i
                        || marker.task != *task
                    {
                        return Err(GenericJobError::Schema {
                            path: done,
                            reason: "done marker does not match job or task",
                        });
                    }
                    continue;
                }
                None => {}
            }

            let path = claim_path(d, i, 0);
            if let Some(observed) = read_claim(&path)? {
                if !claim_is_stale(&path, &observed, self.stale_claim_after)? {
                    continue;
                }
                if !quarantine_claim_if_stale(&path, &observed, self.stale_claim_after)? {
                    continue;
                }
            }
            let token = format!(
                "{}-{}-{}",
                std::process::id(),
                unix_nanos(),
                CLAIM_ID.fetch_add(1, Ordering::Relaxed)
            );
            let lease = ClaimLease {
                schema_version: CLAIM_SCHEMA_VERSION,
                token: token.clone(),
                pid: std::process::id(),
                renewed_unix_nanos: unix_nanos(),
            };
            let payload =
                serde_json::to_vec(&lease).map_err(|e| GenericJobError::json(Some(&path), e))?;
            if exclusive_write(&path, &payload)? {
                let claim = OwnedClaim {
                    task_index: i,
                    token,
                    path,
                };
                self.renew_claim(&claim)?;
                return Ok(Some(claim));
            }
        }
        Ok(None)
    }

    fn release_claim(&self, claim: &OwnedClaim) -> Result<(), GenericJobError> {
        if remove_claim_if_token(&claim.path, &claim.token)? {
            let heartbeat = heartbeat_path(&claim.path);
            if read_heartbeat(&heartbeat)?.is_some_and(|value| value.token == claim.token) {
                super::paths::durable_remove_if_exists(&heartbeat)?;
            }
        }
        Ok(())
    }

    fn renew_claim(&self, claim: &OwnedClaim) -> Result<(), GenericJobError> {
        let current = read_claim(&claim.path)?
            .ok_or(GenericJobError::Merge("dynamic claim ownership changed"))?;
        if current.token != claim.token {
            return Err(GenericJobError::Merge("dynamic claim ownership changed"));
        }
        let heartbeat = ClaimHeartbeat {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: claim.token.clone(),
            renewed_unix_nanos: unix_nanos(),
        };
        let heartbeat_path = heartbeat_path(&claim.path);
        let payload = serde_json::to_vec(&heartbeat)
            .map_err(|error| GenericJobError::json(Some(&heartbeat_path), error))?;
        atomic_write(&heartbeat_path, &payload)?;
        if read_claim(&claim.path)?.is_some_and(|lease| lease.token == claim.token) {
            Ok(())
        } else {
            Err(GenericJobError::Merge("dynamic claim ownership changed"))
        }
    }

    fn finish_claim(
        &self,
        job: &Job<M>,
        o: &RunOptions,
        claim: &OwnedClaim,
        r: &TaskResult<M::Parameters>,
    ) -> Result<(), GenericJobError> {
        self.renew_claim(claim)?;
        let d = o.checkpoint_dir.as_ref().unwrap();
        let completion = if r.completed {
            let marker = DoneMarker {
                schema_version: DONE_SCHEMA_VERSION,
                job_name: job.name.clone(),
                task_index: claim.task_index,
                task: r.task.clone(),
            };
            let done = done_path(d, claim.task_index, 0);
            serde_json::to_vec(&marker)
                .map_err(|e| GenericJobError::json(Some(&done), e))
                .and_then(|payload| atomic_write(&done, &payload))
        } else {
            Ok(())
        };
        let release = self.release_claim(claim);
        completion.and(release)
    }
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos()
}

fn read_optional_safe(
    path: &Path,
    operation: &'static str,
) -> Result<Option<Vec<u8>>, GenericJobError> {
    ensure_safe_read_file_path(path)?;
    match fs::read(path) {
        Ok(payload) => Ok(Some(payload)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(GenericJobError::io(operation, path, error)),
    }
}

fn read_claim(path: &Path) -> Result<Option<ClaimLease>, GenericJobError> {
    let Some(payload) = read_optional_safe(path, "read dynamic claim")? else {
        return Ok(None);
    };
    let claim: ClaimLease =
        serde_json::from_slice(&payload).map_err(|_| GenericJobError::Schema {
            path: path.into(),
            reason: "invalid dynamic claim",
        })?;
    if claim.schema_version != CLAIM_SCHEMA_VERSION || claim.token.is_empty() {
        return Err(GenericJobError::Schema {
            path: path.into(),
            reason: "invalid dynamic claim",
        });
    }
    Ok(Some(claim))
}

fn run_state_path(root: &Path, task_index: usize, run_index: usize, state: &str) -> PathBuf {
    task_path(root, task_index).join(format!("run{:04}.{state}", run_index + 1))
}

fn claim_path(root: &Path, task_index: usize, run_index: usize) -> PathBuf {
    run_state_path(root, task_index, run_index, "claim")
}

fn heartbeat_path(claim_path: &Path) -> PathBuf {
    claim_path.with_extension("heartbeat")
}

fn done_path(root: &Path, task_index: usize, run_index: usize) -> PathBuf {
    run_state_path(root, task_index, run_index, "done")
}

fn read_heartbeat(path: &Path) -> Result<Option<ClaimHeartbeat>, GenericJobError> {
    let Some(payload) = read_optional_safe(path, "read dynamic heartbeat")? else {
        return Ok(None);
    };
    let heartbeat: ClaimHeartbeat =
        serde_json::from_slice(&payload).map_err(|_| GenericJobError::Schema {
            path: path.into(),
            reason: "invalid dynamic heartbeat",
        })?;
    if heartbeat.schema_version != CLAIM_SCHEMA_VERSION || heartbeat.token.is_empty() {
        return Err(GenericJobError::Schema {
            path: path.into(),
            reason: "invalid dynamic heartbeat",
        });
    }
    Ok(Some(heartbeat))
}

fn timestamp_age(timestamp: u128) -> Duration {
    let elapsed = unix_nanos().saturating_sub(timestamp);
    Duration::from_nanos(elapsed.min(u64::MAX as u128) as u64)
}

fn claim_is_stale(
    path: &Path,
    claim: &ClaimLease,
    stale_after: Duration,
) -> Result<bool, GenericJobError> {
    if let Some(heartbeat) = read_heartbeat(&heartbeat_path(path))? {
        if heartbeat.token == claim.token {
            return Ok(timestamp_age(heartbeat.renewed_unix_nanos) > stale_after);
        }
    }
    Ok(timestamp_age(claim.renewed_unix_nanos) > stale_after)
}

fn quarantine_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("claim");
    path.with_file_name(format!(
        "{name}.stale.{}.{}",
        std::process::id(),
        CLAIM_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn restore_quarantined_claim(quarantine: &Path, path: &Path) -> Result<bool, GenericJobError> {
    match fs::hard_link(quarantine, path) {
        Ok(()) => {
            super::paths::durable_remove_if_exists(quarantine)?;
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(GenericJobError::io("restore dynamic claim", path, error)),
    }
}

fn quarantine_claim_if(
    path: &Path,
    expected_token: &str,
    stale_after: Option<Duration>,
) -> Result<bool, GenericJobError> {
    let quarantine = quarantine_path(path);
    match fs::rename(path, &quarantine) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(GenericJobError::io("quarantine dynamic claim", path, error)),
    }
    let moved = read_claim(&quarantine)?;
    let matches = moved
        .as_ref()
        .is_some_and(|claim| claim.token == expected_token);
    let removable = if matches {
        match (stale_after, moved.as_ref()) {
            (Some(stale_after), Some(claim)) => {
                let heartbeat = read_heartbeat(&heartbeat_path(path))?;
                let renewed = heartbeat
                    .filter(|heartbeat| heartbeat.token == claim.token)
                    .map_or(claim.renewed_unix_nanos, |heartbeat| {
                        heartbeat.renewed_unix_nanos
                    });
                timestamp_age(renewed) > stale_after
            }
            (None, Some(_)) => true,
            _ => false,
        }
    } else {
        false
    };
    if removable {
        super::paths::durable_remove_if_exists(&quarantine)?;
        Ok(true)
    } else {
        restore_quarantined_claim(&quarantine, path)?;
        Ok(false)
    }
}

fn quarantine_claim_if_stale(
    path: &Path,
    observed: &ClaimLease,
    stale_after: Duration,
) -> Result<bool, GenericJobError> {
    quarantine_claim_if(path, &observed.token, Some(stale_after))
}

fn remove_claim_if_token(path: &Path, token: &str) -> Result<bool, GenericJobError> {
    quarantine_claim_if(path, token, None)
}

fn dynamic_task_order(task_count: usize, assignment: JobAssignment) -> impl Iterator<Item = usize> {
    let offset = assignment.rank * task_count / assignment.world_size;
    (0..task_count).map(move |step| (offset + step) % task_count)
}

fn validate<M: MonteCarlo>(j: &Job<M>, o: &RunOptions) -> Result<(), GenericJobError> {
    let a = o.assignment.unwrap_or_else(JobAssignment::single);
    if let Some(directory) = &o.checkpoint_dir {
        ensure_safe_directory(directory)?;
    }
    if let Some(path) = &o.output_path {
        ensure_safe_file_path(path)?;
    }
    if a.world_size == 0 || a.rank >= a.world_size {
        return Err(GenericJobError::InvalidAssignment {
            rank: a.rank,
            world_size: a.world_size,
        });
    }
    validate_safe_component(&j.name)?;
    let mut names = BTreeSet::new();
    for t in &j.tasks {
        let reason = if validate_safe_component(&t.name).is_err() {
            Some("unsafe name")
        } else if !names.insert(&t.name) {
            Some("duplicate name")
        } else if t.sweeps == 0 || t.binsize == 0 || t.sweeps < t.binsize {
            Some("invalid sweeps/bin size")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(GenericJobError::InvalidTask {
                task: t.name.clone(),
                reason,
            });
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::super::checkpoint::{
        encode_rng_position, read_checkpoint, Checkpoint, CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
    };
    use super::super::paths::{absolute_temp_dir, temp_dir};
    use super::*;
    use crate::Context;
    use serde::{Deserialize, Serialize};
    use std::{
        collections::{BTreeMap, BTreeSet},
        convert::Infallible,
        sync::atomic::{AtomicBool, Ordering as AtomicOrdering},
    };
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct P;
    #[derive(Serialize, Deserialize)]
    struct M {
        v: f64,
    }
    impl MonteCarlo for M {
        type Parameters = P;
        type Error = Infallible;
        fn new(_: &P) -> Result<Self, Self::Error> {
            Ok(Self { v: 0. })
        }
        fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
        fn sweep(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            self.v += 1.;
            Ok(())
        }
        fn measure(&mut self, c: &mut Context) -> Result<(), Self::Error> {
            c.measure("v", self.v).unwrap();
            Ok(())
        }
    }
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct CheckpointProbe {
        path: PathBuf,
    }
    #[derive(Serialize, Deserialize)]
    struct CheckpointProbeModel {
        sweeps: usize,
        path: PathBuf,
    }
    static SAW_DURATION_CHECKPOINT: AtomicBool = AtomicBool::new(false);
    impl MonteCarlo for CheckpointProbeModel {
        type Parameters = CheckpointProbe;
        type Error = Infallible;
        fn new(parameters: &CheckpointProbe) -> Result<Self, Self::Error> {
            Ok(Self {
                sweeps: 0,
                path: parameters.path.clone(),
            })
        }
        fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
        fn sweep(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            if self.sweeps == 1 && self.path.exists() {
                SAW_DURATION_CHECKPOINT.store(true, AtomicOrdering::SeqCst);
            }
            self.sweeps += 1;
            Ok(())
        }
        fn measure(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    #[test]
    fn run_options_keep_old_fields_and_add_duration_overrides() {
        let old_style = RunOptions {
            assignment: Some(JobAssignment::single()),
            checkpoint_dir: Some(PathBuf::from("checkpoints")),
            resume: true,
            checkpoint_interval: 7,
            output_path: Some(PathBuf::from("result.json")),
            sweep_limit: Some(11),
            ..RunOptions::default()
        };
        assert_eq!(old_style.checkpoint_interval, 7);
        assert_eq!(old_style.deadline, None);
        assert_eq!(old_style.checkpoint_interval_time, None);

        let RunOptions {
            assignment: _,
            checkpoint_dir: _,
            resume: _,
            checkpoint_interval: _,
            output_path: _,
            sweep_limit: _,
            deadline: _,
            checkpoint_interval_time: _,
        } = RunOptions::default();
    }
    #[test]
    fn deterministic() {
        let j = Job::<M>::new("j", vec![Task::new("t", P).sweeps(4).binsize(2)]);
        let a = Runner::new().run(&j, &RunOptions::default()).unwrap();
        let b = Runner::new().run(&j, &RunOptions::default()).unwrap();
        assert_eq!(a.result, b.result)
    }
    #[test]
    fn static_runner_assigns_tasks_by_rank() {
        let job = Job::<M>::new(
            "j",
            (0..7)
                .map(|index| Task::new(format!("task-{index}"), P))
                .collect(),
        );
        let result = Runner::new()
            .run(
                &job,
                &RunOptions {
                    assignment: Some(JobAssignment::new(1, 3).unwrap()),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert_eq!(
            result
                .result
                .tasks
                .iter()
                .map(|task| task.task_index)
                .collect::<Vec<_>>(),
            vec![1, 4]
        );
    }

    #[test]
    fn dynamic_task_order_starts_at_rank_offset_and_wraps() {
        let assignment = JobAssignment::new(2, 3).unwrap();
        assert_eq!(
            dynamic_task_order(8, assignment).collect::<Vec<_>>(),
            vec![5, 6, 7, 0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn dynamic_multi_rank_execution_completes_each_task_once() {
        let directory = temp_dir("dynamic-multi-rank-order");
        let job = Job::<M>::new(
            "j",
            (0..8)
                .map(|index| Task::new(format!("task-{index}"), P))
                .collect(),
        );
        let mut executed = BTreeSet::new();
        for _ in 0..2 {
            for rank in 0..4 {
                let run = Runner::new()
                    .dynamic()
                    .run(
                        &job,
                        &RunOptions {
                            assignment: Some(JobAssignment::new(rank, 4).unwrap()),
                            checkpoint_dir: Some(directory.clone()),
                            sweep_limit: Some(1),
                            ..RunOptions::default()
                        },
                    )
                    .unwrap();
                assert_eq!(run.result.tasks.len(), 1);
                assert!(run.result.tasks[0].completed);
                assert!(executed.insert(run.result.tasks[0].task_index));
            }
        }
        assert_eq!(executed, BTreeSet::from_iter(0..8));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn duration_builder_configures_defaults() {
        let r = Runner::<M>::new()
            .stale_claim_after(Duration::from_secs(1))
            .duration_config(RunnerDurationConfig {
                deadline: Some(Duration::from_secs(2)),
                checkpoint_interval: Some(Duration::from_secs(3)),
            });
        assert_eq!(r.stale_claim_after, Duration::from_secs(1));
        assert_eq!(r.duration_config.deadline, Some(Duration::from_secs(2)));
        assert_eq!(
            r.duration_config.checkpoint_interval,
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn duration_deadline_stops_run_stably() {
        let j = Job::<M>::new("j", vec![Task::new("t", P).sweeps(4).binsize(2)]);
        let result = Runner::new()
            .deadline(Duration::from_secs(60))
            .run(
                &j,
                &RunOptions {
                    deadline: Some(Duration::ZERO),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert!(result.stopped_early);
        assert!(result.result.tasks.is_empty());
    }

    #[test]
    fn duration_checkpoint_runs_during_task_and_coexists_with_sweep_interval() {
        let d = temp_dir("duration-checkpoint");
        let path = checkpoint_path(&d, 0);
        SAW_DURATION_CHECKPOINT.store(false, AtomicOrdering::SeqCst);
        let job = Job::<CheckpointProbeModel>::new(
            "j",
            vec![Task::new("t", CheckpointProbe { path: path.clone() }).sweeps(2)],
        );
        Runner::new()
            .checkpoint_every(Duration::from_secs(60))
            .run(
                &job,
                &RunOptions {
                    checkpoint_dir: Some(d.clone()),
                    checkpoint_interval: usize::MAX,
                    checkpoint_interval_time: Some(Duration::ZERO),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert!(SAW_DURATION_CHECKPOINT.load(AtomicOrdering::SeqCst));
        let checkpoint: Checkpoint<CheckpointProbe> = read_checkpoint(&path).unwrap();
        assert_eq!(checkpoint.measurement_sweeps, 2);
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn absolute_checkpoint_and_result_paths_roundtrip() {
        let directory = absolute_temp_dir("absolute-run-roundtrip");
        let output = directory.join("result.json");
        let checkpoint = checkpoint_path(&directory, 0);
        let job = Job::<M>::new("j", vec![Task::new("t", P).sweeps(4).binsize(2)]);

        let run = Runner::new()
            .run(
                &job,
                &RunOptions {
                    checkpoint_dir: Some(directory.clone()),
                    checkpoint_interval: 1,
                    output_path: Some(output.clone()),
                    ..RunOptions::default()
                },
            )
            .unwrap();

        let restored_checkpoint: Checkpoint<P> = read_checkpoint(&checkpoint).unwrap();
        assert_eq!(restored_checkpoint.measurement_sweeps, 4);
        assert_eq!(JobResult::read_json(&output).unwrap(), run.result);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn canonical_result_path_is_written_as_hdf5() {
        let d = temp_dir("canonical-hdf5-result");
        let output = super::super::result_path(&d);
        let job = Job::<M>::new("j", vec![Task::new("t", P).sweeps(2)]);
        let run = Runner::new()
            .run(
                &job,
                &RunOptions {
                    output_path: Some(output.clone()),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run.output_path.as_deref(), Some(output.as_path()));
        assert_eq!(JobResult::read_hdf5(&output).unwrap(), run.result);
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_run() {
        let d = temp_dir("resume");
        let j = Job::<M>::new(
            "j",
            vec![Task::new("t", P).thermalization(2).sweeps(4).binsize(2)],
        );
        let partial = RunOptions {
            checkpoint_dir: Some(d.clone()),
            checkpoint_interval: 1,
            sweep_limit: Some(3),
            ..RunOptions::default()
        };
        assert!(Runner::new().run(&j, &partial).unwrap().stopped_early);
        let checkpoint: Checkpoint<P> = read_checkpoint(&checkpoint_path(&d, 0)).unwrap();
        let pending = &checkpoint.observables["v"];
        assert_eq!(checkpoint.thermalization_sweeps, 2);
        assert_eq!(checkpoint.measurement_sweeps, 1);
        assert_eq!(pending.pending_count, 1);
        assert_eq!(pending.pending_sum, 3.0);
        assert_eq!(pending.total_count, 1);
        assert!(pending.internal_bins.is_empty());
        let resumed = Runner::new()
            .run(
                &j,
                &RunOptions {
                    checkpoint_dir: Some(d.clone()),
                    resume: true,
                    ..RunOptions::default()
                },
            )
            .unwrap();
        let full = Runner::new().run(&j, &RunOptions::default()).unwrap();
        assert_eq!(resumed.result, full.result);
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn dynamic_other_rank_resumes_claimed_checkpoint() {
        let d = temp_dir("dynamic-rank-takeover");
        let job = Job::<M>::new(
            "j",
            vec![Task::new("t", P).thermalization(1).sweeps(4).binsize(2)],
        );
        let partial = Runner::new()
            .dynamic()
            .run(
                &job,
                &RunOptions {
                    assignment: Some(JobAssignment::new(0, 2).unwrap()),
                    checkpoint_dir: Some(d.clone()),
                    checkpoint_interval: 1,
                    sweep_limit: Some(2),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert!(partial.stopped_early);

        let resumed = Runner::new()
            .dynamic()
            .run(
                &job,
                &RunOptions {
                    assignment: Some(JobAssignment::new(1, 2).unwrap()),
                    checkpoint_dir: Some(d.clone()),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        let uninterrupted = Runner::new().run(&job, &RunOptions::default()).unwrap();
        assert_eq!(resumed.result.tasks, uninterrupted.result.tasks);
        assert!(done_path(&d, 0, 0).exists());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn dynamic_claim_done_and_stale_reclaim() {
        let d = temp_dir("dynamic");
        fs::create_dir_all(&d).unwrap();
        let j = Job::<M>::new("j", vec![Task::new("t", P).sweeps(2).binsize(1)]);
        let options = RunOptions {
            checkpoint_dir: Some(d.clone()),
            ..RunOptions::default()
        };
        let runner = Runner::new().dynamic();
        let first = runner.run(&j, &options).unwrap();
        assert_eq!(first.result.tasks.len(), 1);
        assert!(done_path(&d, 0, 0).exists());
        assert!(checkpoint_path(&d, 0).exists());
        assert_ne!(done_path(&d, 0, 0), checkpoint_path(&d, 0));
        assert_eq!(
            heartbeat_path(&claim_path(&d, 0, 0)),
            d.join("task0001/run0001.heartbeat")
        );
        assert!(!claim_path(&d, 0, 0).exists());
        assert!(runner.run(&j, &options).unwrap().result.tasks.is_empty());

        super::super::paths::durable_remove_if_exists(&done_path(&d, 0, 0)).unwrap();
        let stale = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "stale-owner".into(),
            pid: 1,
            renewed_unix_nanos: 0,
        };
        atomic_write(&claim_path(&d, 0, 0), &serde_json::to_vec(&stale).unwrap()).unwrap();
        let reclaimed = Runner::new()
            .dynamic()
            .stale_claim_after(Duration::ZERO)
            .run(&j, &options)
            .unwrap();
        assert_eq!(reclaimed.result.tasks.len(), 1);
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn dynamic_partial_claims_only_executed_task_and_resumes_checkpoint() {
        let d = temp_dir("dynamic-partial");
        let j = Job::<M>::new(
            "j",
            vec![
                Task::new("a", P).thermalization(1).sweeps(2),
                Task::new("b", P).sweeps(1),
            ],
        );
        let partial = Runner::new()
            .dynamic()
            .run(
                &j,
                &RunOptions {
                    checkpoint_dir: Some(d.clone()),
                    checkpoint_interval: 1,
                    sweep_limit: Some(1),
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert!(partial.stopped_early);
        assert_eq!(partial.result.tasks[0].task_index, 0);
        assert_eq!(partial.result.tasks[0].thermalization_sweeps, 1);
        assert!(!claim_path(&d, 0, 0).exists());
        assert!(!claim_path(&d, 1, 0).exists());
        assert!(!done_path(&d, 0, 0).exists());
        assert!(!checkpoint_path(&d, 1).exists());

        let resumed = Runner::new()
            .dynamic()
            .run(
                &j,
                &RunOptions {
                    checkpoint_dir: Some(d.clone()),
                    resume: true,
                    ..RunOptions::default()
                },
            )
            .unwrap();
        assert_eq!(resumed.result.tasks.len(), 2);
        assert!(resumed.result.tasks.iter().all(|task| task.completed));
        assert_eq!(resumed.result.tasks[0].thermalization_sweeps, 1);
        assert!(done_path(&d, 0, 0).exists());
        assert!(done_path(&d, 1, 0).exists());
        assert!(!claim_path(&d, 0, 0).exists());
        assert!(!claim_path(&d, 1, 0).exists());
        assert!(Runner::new()
            .dynamic()
            .run(
                &j,
                &RunOptions {
                    checkpoint_dir: Some(d.clone()),
                    ..RunOptions::default()
                }
            )
            .unwrap()
            .result
            .tasks
            .is_empty());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn dynamic_rejects_corrupt_or_mismatched_done_markers() {
        let d = temp_dir("done-validation");
        fs::create_dir_all(&d).unwrap();
        let j = Job::<M>::new("j", vec![Task::new("t", P)]);
        let options = RunOptions {
            checkpoint_dir: Some(d.clone()),
            ..RunOptions::default()
        };
        let done = done_path(&d, 0, 0);
        for payload in [
            b"not json".to_vec(),
            br#"{"schema_version":1,"job_name":"other","task_index":0,"task":{"name":"t","parameters":null,"sweeps":1,"thermalization":0,"binsize":1,"seed":0}}"#.to_vec(),
            br#"{"schema_version":1,"job_name":"j","task_index":1,"task":{"name":"t","parameters":null,"sweeps":1,"thermalization":0,"binsize":1,"seed":0}}"#.to_vec(),
            br#"{"schema_version":1,"job_name":"j","task_index":0,"task":{"name":"different","parameters":null,"sweeps":1,"thermalization":0,"binsize":1,"seed":0}}"#.to_vec(),
            br#"{"schema_version":1,"job_name":"j","task_index":0,"task":{"name":"t","parameters":null,"sweeps":1,"thermalization":0,"binsize":1,"seed":0},"extra":true}"#.to_vec(),
        ] {
            atomic_write(&done, &payload).unwrap();
            assert!(Runner::new().dynamic().run(&j, &options).is_err());
            assert!(!claim_path(&d, 0, 0).exists());
        }
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn checkpoint_resume_rejects_job_task_index_and_corrupt_state() {
        let d = temp_dir("checkpoint-validation");
        fs::create_dir_all(&d).unwrap();
        let task = Task::new("t", P).thermalization(2).sweeps(4).binsize(2);
        let job = Job::<M>::new("j", vec![task.clone()]);
        let path = checkpoint_path(&d, 0);
        let base = Checkpoint {
            schema_version: CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
            rank: 0,
            world_size: 1,
            job_name: job.name.clone(),
            task: task.clone(),
            task_index: 0,
            model: serde_json::to_value(M { v: 0. }).unwrap(),
            rng_position_words: encode_rng_position(0),
            thermalization_sweeps: 2,
            measurement_sweeps: 1,
            observables: BTreeMap::from([(
                "v".into(),
                crate::CompactAccumulator {
                    internal_bins: vec![],
                    pending_sum: 1.,
                    pending_count: 1,
                    total_count: 1,
                    binsize: 2,
                },
            )]),
        };
        let options = RunOptions {
            checkpoint_dir: Some(d.clone()),
            resume: true,
            ..RunOptions::default()
        };

        let mut cases = Vec::new();
        let mut wrong_version = base.clone();
        wrong_version.schema_version += 1;
        cases.push(wrong_version);
        let mut wrong_rank = base.clone();
        wrong_rank.rank = 1;
        wrong_rank.world_size = 2;
        cases.push(wrong_rank);
        let mut wrong_world = base.clone();
        wrong_world.world_size = 2;
        cases.push(wrong_world);
        let mut wrong_job = base.clone();
        wrong_job.job_name = "other".into();
        cases.push(wrong_job);
        let mut wrong_task = base.clone();
        wrong_task.task.name = "other".into();
        cases.push(wrong_task);
        let mut wrong_index = base.clone();
        wrong_index.task_index = 1;
        cases.push(wrong_index);
        let mut thermalization_over = base.clone();
        thermalization_over.thermalization_sweeps = 3;
        cases.push(thermalization_over);
        let mut measurement_over = base.clone();
        measurement_over.measurement_sweeps = 5;
        cases.push(measurement_over);
        let mut measured_early = base.clone();
        measured_early.thermalization_sweeps = 1;
        cases.push(measured_early);
        let mut zero_binsize = base.clone();
        zero_binsize.observables.get_mut("v").unwrap().binsize = 0;
        cases.push(zero_binsize);
        let mut full_pending = base.clone();
        full_pending.observables.get_mut("v").unwrap().pending_count = 2;
        full_pending.observables.get_mut("v").unwrap().total_count = 2;
        cases.push(full_pending);
        let mut bad_total = base.clone();
        bad_total.observables.get_mut("v").unwrap().total_count = 2;
        cases.push(bad_total);
        let mut residual_sum = base.clone();
        residual_sum.observables.get_mut("v").unwrap().pending_count = 0;
        residual_sum.observables.get_mut("v").unwrap().total_count = 0;
        cases.push(residual_sum);
        let mut nonfinite_sum = base.clone();
        nonfinite_sum.observables.get_mut("v").unwrap().pending_sum = f64::NAN;
        cases.push(nonfinite_sum);
        let mut nonfinite_bin = base.clone();
        nonfinite_bin
            .observables
            .get_mut("v")
            .unwrap()
            .internal_bins = vec![f64::INFINITY];
        nonfinite_bin.observables.get_mut("v").unwrap().total_count = 3;
        cases.push(nonfinite_bin);
        let mut empty_observable = base;
        let accumulator = empty_observable.observables.remove("v").unwrap();
        empty_observable
            .observables
            .insert(String::new(), accumulator);
        cases.push(empty_observable);

        for checkpoint in cases {
            write_checkpoint(&path, &checkpoint).unwrap();
            assert!(Runner::new().run(&job, &options).is_err());
        }
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn claim_release_and_stale_reclaim_preserve_replacement_owner() {
        let d = temp_dir("claim-ownership");
        fs::create_dir_all(&d).unwrap();
        let path = claim_path(&d, 0, 0);
        let observed = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "observed".into(),
            pid: 1,
            renewed_unix_nanos: 1,
        };
        let replacement = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "replacement".into(),
            pid: 2,
            renewed_unix_nanos: 2,
        };
        atomic_write(&path, &serde_json::to_vec(&replacement).unwrap()).unwrap();
        assert!(
            !quarantine_claim_if_stale(&path, &observed, Duration::from_secs(u64::MAX)).unwrap()
        );
        assert!(!remove_claim_if_token(&path, &observed.token).unwrap());
        assert_eq!(read_claim(&path).unwrap(), Some(replacement));
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn fresh_heartbeat_overrides_stale_claim_and_rename_recovery_restores_it() {
        let d = temp_dir("heartbeat-priority");
        fs::create_dir_all(&d).unwrap();
        let path = claim_path(&d, 0, 0);
        let claim = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "active-owner".into(),
            pid: 1,
            renewed_unix_nanos: 0,
        };
        atomic_write(&path, &serde_json::to_vec(&claim).unwrap()).unwrap();
        let heartbeat = ClaimHeartbeat {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: claim.token.clone(),
            renewed_unix_nanos: unix_nanos(),
        };
        atomic_write(
            &heartbeat_path(&path),
            &serde_json::to_vec(&heartbeat).unwrap(),
        )
        .unwrap();

        assert!(!claim_is_stale(&path, &claim, Duration::from_secs(60)).unwrap());
        assert!(!quarantine_claim_if_stale(&path, &claim, Duration::from_secs(60)).unwrap());
        assert_eq!(read_claim(&path).unwrap(), Some(claim));
        assert!(heartbeat_path(&path).exists());
        assert!(!fs::read_dir(path.parent().unwrap())
            .unwrap()
            .any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".stale.")));
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn claim_renewal_updates_only_its_owner() {
        let d = temp_dir("claim-renewal");
        fs::create_dir_all(&d).unwrap();
        let path = claim_path(&d, 0, 0);
        let lease = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "mine".into(),
            pid: 1,
            renewed_unix_nanos: 1,
        };
        atomic_write(&path, &serde_json::to_vec(&lease).unwrap()).unwrap();
        let owned = OwnedClaim {
            task_index: 0,
            token: lease.token.clone(),
            path: path.clone(),
        };
        Runner::<M>::new().renew_claim(&owned).unwrap();
        let heartbeat = read_heartbeat(&heartbeat_path(&path)).unwrap().unwrap();
        assert_eq!(heartbeat.token, lease.token);
        assert!(heartbeat.renewed_unix_nanos > lease.renewed_unix_nanos);

        let replacement = ClaimLease {
            schema_version: CLAIM_SCHEMA_VERSION,
            token: "theirs".into(),
            pid: 2,
            renewed_unix_nanos: unix_nanos(),
        };
        atomic_write(&path, &serde_json::to_vec(&replacement).unwrap()).unwrap();
        assert!(Runner::<M>::new().renew_claim(&owned).is_err());
        assert_eq!(read_claim(&path).unwrap(), Some(replacement));
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn dynamic_releases_claim_when_checkpoint_read_fails() {
        let d = temp_dir("claim-error");
        fs::create_dir_all(&d).unwrap();
        let j = Job::<M>::new("j", vec![Task::new("t", P)]);
        atomic_write(&checkpoint_path(&d, 0), b"invalid hdf5").unwrap();
        let result = Runner::new().dynamic().run(
            &j,
            &RunOptions {
                checkpoint_dir: Some(d.clone()),
                resume: true,
                ..RunOptions::default()
            },
        );
        assert!(result.is_err());
        assert!(!claim_path(&d, 0, 0).exists());
        fs::remove_dir_all(d).unwrap();
    }
}
