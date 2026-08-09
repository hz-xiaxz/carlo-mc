use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use rand::{SeedableRng, TryRng};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// A deterministic SplitMix64 random number generator with resumable position.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    seed: u64,
    state: u64,
    draws: u128,
}

impl DeterministicRng {
    #[inline]
    fn splitmix64_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(SPLITMIX64_GAMMA);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Returns the number of generated words.
    #[inline]
    pub fn position(&self) -> u128 {
        self.draws
    }

    /// Restores the generator to a previous word position.
    pub fn set_position(&mut self, position: u128) {
        self.state = self
            .seed
            .wrapping_add(SPLITMIX64_GAMMA.wrapping_mul(position as u64));
        self.draws = position;
    }
}

impl SeedableRng for DeterministicRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        Self::seed_from_u64(u64::from_le_bytes(seed))
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self {
            seed,
            state: seed,
            draws: 0,
        }
    }
}

impl TryRng for DeterministicRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let word = Self::splitmix64_next(&mut self.state);
        self.draws += 1;
        Ok(word)
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        let (chunks, remainder) = destination.as_chunks_mut::<8>();
        for chunk in chunks {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes());
        }
        if !remainder.is_empty() {
            let word = self.try_next_u64()?.to_le_bytes();
            remainder.copy_from_slice(&word[..remainder.len()]);
        }
        Ok(())
    }
}

/// A rank's static share of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobAssignment {
    /// Zero-based rank index.
    pub rank: usize,
    /// Total number of ranks.
    pub world_size: usize,
}

/// Error returned when a rank assignment is outside its valid range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssignmentError {
    /// Requested zero-based rank.
    pub rank: usize,
    /// Requested total number of ranks.
    pub world_size: usize,
}

impl fmt::Display for AssignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.world_size == 0 {
            write!(formatter, "world_size must be positive")
        } else {
            write!(formatter, "rank must be smaller than world_size")
        }
    }
}

impl StdError for AssignmentError {}

impl JobAssignment {
    /// Creates and validates a rank assignment.
    pub fn new(rank: usize, world_size: usize) -> Result<Self, AssignmentError> {
        if world_size == 0 || rank >= world_size {
            return Err(AssignmentError { rank, world_size });
        }
        Ok(Self { rank, world_size })
    }

    /// Returns rank zero in a single-rank job.
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    /// Returns an assignment from common MPI or Slurm environment variables,
    /// defaulting to a single rank when none are present.
    pub fn from_env() -> Result<Self, AssignmentError> {
        Self::new(
            env_usize_any(&[
                "XY_RANK",
                "SLURM_PROCID",
                "OMPI_COMM_WORLD_RANK",
                "PMI_RANK",
                "PMIX_RANK",
                "MV2_COMM_WORLD_RANK",
            ])
            .unwrap_or(0),
            env_usize_any(&[
                "XY_WORLD_SIZE",
                "SLURM_NTASKS",
                "OMPI_COMM_WORLD_SIZE",
                "PMI_SIZE",
                "PMIX_SIZE",
                "MV2_COMM_WORLD_SIZE",
            ])
            .unwrap_or(1),
        )
    }
}

fn env_usize_any(keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CompactObservableAccumulator {
    internal_bins: Vec<f64>,
    pending_sum: f64,
    pending_count: usize,
    total_count: usize,
    internal_bin_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ScalarAccumulator {
    internal_bins: Vec<f64>,
    pending_sum: f64,
    pending_count: usize,
    total_count: usize,
    internal_bin_length: usize,
}

impl ScalarAccumulator {
    fn new(internal_bin_length: usize) -> Self {
        Self {
            internal_bins: Vec::new(),
            pending_sum: 0.0,
            pending_count: 0,
            total_count: 0,
            internal_bin_length,
        }
    }

    fn from_compact(compact: CompactObservableAccumulator, fallback_bin_length: usize) -> Self {
        let internal_bin_length = if compact.internal_bin_length == 0 {
            fallback_bin_length
        } else {
            compact.internal_bin_length
        };
        Self {
            internal_bins: compact.internal_bins,
            pending_sum: compact.pending_sum,
            pending_count: compact.pending_count,
            total_count: compact.total_count,
            internal_bin_length,
        }
    }

    fn push(&mut self, value: f64) {
        self.pending_sum += value;
        self.pending_count += 1;
        self.total_count += 1;
        if self.pending_count == self.internal_bin_length {
            self.internal_bins
                .push(self.pending_sum / self.internal_bin_length as f64);
            self.pending_sum = 0.0;
            self.pending_count = 0;
        }
    }

    fn compact(&self) -> CompactObservableAccumulator {
        CompactObservableAccumulator {
            internal_bins: self.internal_bins.clone(),
            pending_sum: self.pending_sum,
            pending_count: self.pending_count,
            total_count: self.total_count,
            internal_bin_length: self.internal_bin_length,
        }
    }

    fn estimate(&self) -> Option<AccumulatorEstimate> {
        if self.internal_bins.is_empty() {
            return None;
        }
        let rebin_length = carlo_rebin_length(self.internal_bins.len());
        let usable = self.internal_bins.len() - self.internal_bins.len() % rebin_length;
        let bins = self.internal_bins[..usable]
            .chunks_exact(rebin_length)
            .map(mean)
            .collect::<Vec<_>>();
        Some(AccumulatorEstimate {
            mean: mean(&bins),
            stderr: standard_error(&bins),
            internal_bins: self.internal_bins.len(),
            rebin_length,
            rebin_count: bins.len(),
            internal_bin_length: self.internal_bin_length,
        })
    }
}

struct AccumulatorEstimate {
    mean: f64,
    stderr: f64,
    internal_bins: usize,
    rebin_length: usize,
    rebin_count: usize,
    internal_bin_length: usize,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn carlo_rebin_count(sample_count: usize) -> usize {
    if sample_count <= 10 {
        sample_count
    } else {
        10 + ((sample_count - 10) as f64).cbrt().round() as usize
    }
    .max(1)
}

fn carlo_rebin_length(sample_count: usize) -> usize {
    (sample_count / carlo_rebin_count(sample_count)).max(1)
}

fn standard_error(bins: &[f64]) -> f64 {
    if bins.len() <= 1 {
        return f64::NAN;
    }
    let average = mean(bins);
    let variance = bins
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (bins.len() - 1) as f64;
    variance.sqrt() / (bins.len() as f64).sqrt()
}

/// A strongly typed Monte Carlo task.
///
/// `parameters` contains only model-specific input. Scheduling parameters are
/// explicit fields so the runner can validate and execute every model in the
/// same way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task<P> {
    /// Human-readable task name, unique within a job.
    pub name: String,
    /// Model-specific parameters captured for this task.
    pub parameters: P,
    /// Number of measurement sweeps.
    pub sweeps: usize,
    /// Number of sweeps discarded for thermalization.
    pub thermalization: usize,
    /// Number of scalar samples combined into one internal bin.
    pub binsize: usize,
    /// Seed used by the task's deterministic random number generator.
    pub seed: u64,
}

impl<P> Task<P> {
    /// Creates a task with one measurement sweep, no thermalization, bin size
    /// one, and seed zero.
    pub fn new(name: impl Into<String>, parameters: P) -> Self {
        Self {
            name: name.into(),
            parameters,
            sweeps: 1,
            thermalization: 0,
            binsize: 1,
            seed: 0,
        }
    }

    /// Sets the number of measurement sweeps.
    pub fn sweeps(mut self, sweeps: usize) -> Self {
        self.sweeps = sweeps;
        self
    }

    /// Sets the number of thermalization sweeps.
    pub fn thermalization(mut self, thermalization: usize) -> Self {
        self.thermalization = thermalization;
        self
    }

    /// Sets the default internal observable bin size.
    pub fn binsize(mut self, binsize: usize) -> Self {
        self.binsize = binsize;
        self
    }

    /// Sets the deterministic random-number seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Carlo-style helper that snapshots shared strongly typed parameters into
/// tasks.
#[derive(Debug, Clone)]
pub struct TaskMaker<P> {
    shared: P,
    defaults: TaskDefaults,
    tasks: Vec<Task<P>>,
}

#[derive(Debug, Clone, Copy)]
struct TaskDefaults {
    sweeps: usize,
    thermalization: usize,
    binsize: usize,
    seed: u64,
}

impl<P: Clone> TaskMaker<P> {
    /// Creates a task maker with the shared model parameters.
    pub fn new(shared: P) -> Self {
        Self {
            shared,
            defaults: TaskDefaults {
                sweeps: 1,
                thermalization: 0,
                binsize: 1,
                seed: 0,
            },
            tasks: Vec::new(),
        }
    }

    /// Replaces the shared parameters used by future snapshots.
    pub fn set_shared(&mut self, shared: P) -> &mut Self {
        self.shared = shared;
        self
    }

    /// Returns the shared parameters for in-place updates before adding a task.
    pub fn shared_mut(&mut self) -> &mut P {
        &mut self.shared
    }

    /// Sets the default measurement sweep count for future tasks.
    pub fn set_sweeps(&mut self, sweeps: usize) -> &mut Self {
        self.defaults.sweeps = sweeps;
        self
    }

    /// Sets the default thermalization count for future tasks.
    pub fn set_thermalization(&mut self, thermalization: usize) -> &mut Self {
        self.defaults.thermalization = thermalization;
        self
    }

    /// Sets the default internal bin size for future tasks.
    pub fn set_binsize(&mut self, binsize: usize) -> &mut Self {
        self.defaults.binsize = binsize;
        self
    }

    /// Sets the default seed for future tasks.
    pub fn set_seed(&mut self, seed: u64) -> &mut Self {
        self.defaults.seed = seed;
        self
    }

    /// Adds a task by cloning the current shared parameters.
    pub fn add_task(&mut self, name: impl Into<String>) -> &mut Self {
        let defaults = self.defaults;
        self.tasks.push(
            Task::new(name, self.shared.clone())
                .sweeps(defaults.sweeps)
                .thermalization(defaults.thermalization)
                .binsize(defaults.binsize)
                .seed(defaults.seed),
        );
        self
    }

    /// Adds a task snapshot after applying a task-local parameter override.
    /// The shared parameters are not changed.
    pub fn add_task_with<F>(&mut self, name: impl Into<String>, update: F) -> &mut Self
    where
        F: FnOnce(&mut P),
    {
        let mut parameters = self.shared.clone();
        update(&mut parameters);
        let defaults = self.defaults;
        self.tasks.push(
            Task::new(name, parameters)
                .sweeps(defaults.sweeps)
                .thermalization(defaults.thermalization)
                .binsize(defaults.binsize)
                .seed(defaults.seed),
        );
        self
    }

    /// Returns snapshots accumulated so far without consuming the maker.
    pub fn tasks(&self) -> &[Task<P>] {
        &self.tasks
    }

    /// Consumes the maker and returns all task snapshots.
    pub fn make_tasks(self) -> Vec<Task<P>> {
        self.tasks
    }
}

/// A model-independent collection of Monte Carlo tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job<M: MonteCarlo> {
    /// Job name used in JSON output and checkpoint directory names.
    pub name: String,
    /// Strongly typed tasks in stable scheduling order.
    pub tasks: Vec<Task<M::Parameters>>,
    #[serde(skip)]
    model: PhantomData<fn() -> M>,
}

impl<M: MonteCarlo> Job<M> {
    /// Creates a job for model `M`.
    pub fn new(name: impl Into<String>, tasks: Vec<Task<M::Parameters>>) -> Self {
        Self {
            name: name.into(),
            tasks,
            model: PhantomData,
        }
    }

    /// Iterates over tasks statically assigned to a rank by task index modulo
    /// world size.
    pub fn selected_tasks(
        &self,
        assignment: JobAssignment,
    ) -> impl Iterator<Item = (usize, &Task<M::Parameters>)> {
        self.tasks
            .iter()
            .enumerate()
            .filter(move |(index, _)| index % assignment.world_size == assignment.rank)
    }
}

/// Interface implemented by a serializable Monte Carlo model.
///
/// The complete model value is serialized in checkpoints. Consequently all
/// configuration state needed to continue a run must be part of `Self`.
///
/// # Example
///
/// ```
/// use serde::{Deserialize, Serialize};
/// use carlo_mc::{Context, Job, MonteCarlo, RunOptions, Runner, TaskMaker};
/// use std::convert::Infallible;
///
/// #[derive(Clone, PartialEq, Serialize, Deserialize)]
/// struct Parameters {
///     initial_value: f64,
/// }
///
/// #[derive(Serialize, Deserialize)]
/// struct RandomWalk {
///     value: f64,
/// }
///
/// impl MonteCarlo for RandomWalk {
///     type Parameters = Parameters;
///     type Error = Infallible;
///
///     fn new(parameters: &Parameters) -> Result<Self, Self::Error> {
///         Ok(Self { value: parameters.initial_value })
///     }
///
///     fn init(&mut self, _context: &mut Context) -> Result<(), Self::Error> {
///         Ok(())
///     }
///
///     fn sweep(&mut self, _context: &mut Context) -> Result<(), Self::Error> {
///         self.value += 1.0;
///         Ok(())
///     }
///
///     fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error> {
///         // Infallible is suitable here because this fixed measurement is valid.
///         context.measure("Position", self.value).expect("valid measurement");
///         Ok(())
///     }
/// }
///
/// let mut tasks = TaskMaker::new(Parameters { initial_value: 0.0 });
/// tasks
///     .set_thermalization(10)
///     .set_sweeps(100)
///     .set_binsize(10)
///     .set_seed(42)
///     .add_task("walk");
/// let job = Job::<RandomWalk>::new("example", tasks.make_tasks());
/// let run = Runner::<RandomWalk>::new().run(&job, &RunOptions::default())?;
/// assert!(run.result.tasks[0].completed);
/// # Ok::<(), carlo_mc::GenericJobError>(())
/// ```
pub trait MonteCarlo: Sized + Serialize + DeserializeOwned {
    /// Strongly typed task parameter type.
    type Parameters: Clone + PartialEq + Serialize + DeserializeOwned;
    /// Model-specific structured error type.
    type Error: StdError + Send + Sync + 'static;

    /// Constructs model state for a fresh task.
    fn new(parameters: &Self::Parameters) -> Result<Self, Self::Error>;

    /// Initializes a freshly constructed model. This is not called when a
    /// checkpoint is restored.
    fn init(&mut self, context: &mut Context) -> Result<(), Self::Error>;

    /// Performs one Monte Carlo update or sweep.
    fn sweep(&mut self, context: &mut Context) -> Result<(), Self::Error>;

    /// Records one measurement through [`Context::measure`].
    fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error>;
}

/// Runner-owned context exposed to Monte Carlo models.
#[derive(Debug)]
pub struct Context {
    /// Deterministic random number generator for this task.
    pub rng: DeterministicRng,
    thermalized: bool,
    binsize: usize,
    observables: BTreeMap<String, ScalarAccumulator>,
}

impl Context {
    fn fresh(seed: u64, binsize: usize) -> Self {
        Self {
            rng: DeterministicRng::seed_from_u64(seed),
            thermalized: false,
            binsize,
            observables: BTreeMap::new(),
        }
    }

    fn restored(
        seed: u64,
        rng_position: u128,
        binsize: usize,
        observables: BTreeMap<String, CompactObservableAccumulator>,
        thermalized: bool,
    ) -> Self {
        let mut rng = DeterministicRng::seed_from_u64(seed);
        rng.set_position(rng_position);
        Self {
            rng,
            thermalized,
            binsize,
            observables: observables
                .into_iter()
                .map(|(name, accumulator)| {
                    (name, ScalarAccumulator::from_compact(accumulator, binsize))
                })
                .collect(),
        }
    }

    /// Returns whether thermalization has completed.
    pub fn is_thermalized(&self) -> bool {
        self.thermalized
    }

    /// Records a finite scalar sample using the task's default bin size.
    pub fn measure(&mut self, name: impl Into<String>, value: f64) -> Result<(), GenericJobError> {
        self.measure_with_binsize(name, value, self.binsize)
    }

    /// Records a finite scalar sample with an observable-specific bin size.
    pub fn measure_with_binsize(
        &mut self,
        name: impl Into<String>,
        value: f64,
        binsize: usize,
    ) -> Result<(), GenericJobError> {
        let name = name.into();
        if name.is_empty() {
            return Err(GenericJobError::InvalidMeasurement {
                observable: name,
                reason: "observable name must not be empty",
            });
        }
        if !value.is_finite() {
            return Err(GenericJobError::InvalidMeasurement {
                observable: name,
                reason: "observable value must be finite",
            });
        }
        if binsize == 0 {
            return Err(GenericJobError::InvalidMeasurement {
                observable: name,
                reason: "observable bin size must be positive",
            });
        }
        let accumulator = self
            .observables
            .entry(name.clone())
            .or_insert_with(|| ScalarAccumulator::new(binsize));
        if accumulator.compact().internal_bin_length != binsize {
            return Err(GenericJobError::InvalidMeasurement {
                observable: name,
                reason: "observable bin size changed during the run",
            });
        }
        accumulator.push(value);
        Ok(())
    }

    fn compact(&self) -> BTreeMap<String, CompactObservableAccumulator> {
        self.observables
            .iter()
            .map(|(name, accumulator)| (name.clone(), accumulator.compact()))
            .collect()
    }
}

/// Options controlling generic job execution and persistence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunOptions {
    /// Static rank assignment; defaults to rank zero of one.
    pub assignment: Option<JobAssignment>,
    /// Directory containing one JSON checkpoint per task.
    pub checkpoint_dir: Option<PathBuf>,
    /// Resume assigned tasks from existing checkpoints.
    pub resume: bool,
    /// Write a checkpoint after this many completed sweeps. Zero disables
    /// periodic writes; an interrupted run still writes its final state.
    pub checkpoint_interval: usize,
    /// Optional JSON result path.
    pub output_path: Option<PathBuf>,
    /// Optional total sweep budget for this invocation, useful for
    /// wall-clock-controlled resumable runs.
    pub sweep_limit: Option<usize>,
}

/// Final Carlo-style estimate for a scalar observable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarEstimate {
    /// Mean of the automatically rebinned samples.
    pub mean: f64,
    /// Standard error of the rebinned mean.
    pub stderr: f64,
    /// Number of complete internal bins retained.
    pub internal_bins: usize,
    /// Number of internal bins merged into each error-analysis bin.
    pub rebin_length: usize,
    /// Number of error-analysis bins.
    pub rebin_count: usize,
    /// Samples per internal bin.
    pub bin_length: usize,
}

/// Result of one generic Monte Carlo task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult<P> {
    /// Stable index in the job.
    pub task_index: usize,
    /// Original strongly typed task definition.
    pub task: Task<P>,
    /// Scalar observable estimates, ordered by name.
    pub observables: BTreeMap<String, ScalarEstimate>,
    /// Completed thermalization sweeps.
    pub thermalization_sweeps: usize,
    /// Completed measurement sweeps.
    pub measurement_sweeps: usize,
    /// Whether this task reached both requested sweep counts.
    pub completed: bool,
}

/// Serializable result for one rank of a generic job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobResult<P> {
    /// Job name.
    pub job_name: String,
    /// Rank that produced this result.
    pub rank: usize,
    /// Number of ranks participating in static assignment.
    pub world_size: usize,
    /// Results for tasks assigned to this rank.
    pub tasks: Vec<TaskResult<P>>,
}

/// Outcome of a generic runner invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult<P> {
    /// Rank-local job result.
    pub result: JobResult<P>,
    /// JSON path written by the runner, when configured.
    pub output_path: Option<PathBuf>,
    /// Checkpoints written or updated during this invocation.
    pub checkpoint_paths: Vec<PathBuf>,
    /// True when the sweep budget stopped execution before all assigned tasks
    /// completed.
    pub stopped_early: bool,
}

/// Structured errors produced by the generic job API.
#[derive(Debug)]
#[non_exhaustive]
pub enum GenericJobError {
    /// A task definition is invalid.
    InvalidTask {
        /// Task name.
        task: String,
        /// Validation failure.
        reason: &'static str,
    },
    /// Rank assignment is invalid.
    InvalidAssignment {
        /// Requested rank.
        rank: usize,
        /// Requested world size.
        world_size: usize,
    },
    /// A model attempted an invalid scalar measurement.
    InvalidMeasurement {
        /// Observable name, possibly empty.
        observable: String,
        /// Validation failure.
        reason: &'static str,
    },
    /// The model returned its own structured error.
    Model {
        /// Task being executed.
        task: String,
        /// Model-specific source error.
        source: Box<dyn StdError + Send + Sync>,
    },
    /// A filesystem operation failed.
    Io {
        /// Operation being performed.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// JSON serialization or deserialization failed.
    Json {
        /// Affected path, if the JSON was file-backed.
        path: Option<PathBuf>,
        /// Underlying serde error.
        source: serde_json::Error,
    },
    /// A checkpoint belongs to a different task definition.
    CheckpointMismatch {
        /// Checkpoint path.
        path: PathBuf,
        /// Expected task name.
        task: String,
    },
    /// An observable has no complete internal bin at result time.
    InsufficientSamples {
        /// Task name.
        task: String,
        /// Observable name.
        observable: String,
    },
}

impl fmt::Display for GenericJobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask { task, reason } => {
                write!(formatter, "invalid task {task:?}: {reason}")
            }
            Self::InvalidAssignment { rank, world_size } => write!(
                formatter,
                "invalid rank assignment: rank {rank}, world size {world_size}"
            ),
            Self::InvalidMeasurement { observable, reason } => {
                write!(formatter, "invalid measurement {observable:?}: {reason}")
            }
            Self::Model { task, source } => {
                write!(formatter, "model error in task {task:?}: {source}")
            }
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} {}: {source}",
                path.display()
            ),
            Self::Json { path, source } => match path {
                Some(path) => write!(formatter, "invalid JSON at {}: {source}", path.display()),
                None => write!(formatter, "JSON error: {source}"),
            },
            Self::CheckpointMismatch { path, task } => write!(
                formatter,
                "checkpoint {} does not match task {task:?}",
                path.display()
            ),
            Self::InsufficientSamples { task, observable } => write!(
                formatter,
                "task {task:?} has no complete bin for observable {observable:?}"
            ),
        }
    }
}

impl StdError for GenericJobError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Model { source, .. } => Some(source.as_ref()),
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GenericCheckpoint<P> {
    task: Task<P>,
    task_index: usize,
    model: serde_json::Value,
    rng_position: u128,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
    observables: BTreeMap<String, CompactObservableAccumulator>,
}

/// Executes [`Job`] values for model `M`.
#[derive(Debug, Default)]
pub struct Runner<M: MonteCarlo> {
    model: PhantomData<fn() -> M>,
}

impl<M: MonteCarlo> Runner<M> {
    /// Creates a model runner.
    pub fn new() -> Self {
        Self { model: PhantomData }
    }

    /// Runs statically assigned tasks, optionally checkpointing, resuming, and
    /// writing rank-local JSON results.
    pub fn run(
        &self,
        job: &Job<M>,
        options: &RunOptions,
    ) -> Result<RunResult<M::Parameters>, GenericJobError> {
        let assignment = options.assignment.unwrap_or_else(JobAssignment::single);
        validate_assignment(assignment)?;
        validate_job(job)?;

        let mut tasks = Vec::new();
        let mut checkpoint_paths = Vec::new();
        let mut remaining_sweeps = options.sweep_limit.unwrap_or(usize::MAX);
        let mut stopped_early = false;

        for (task_index, task) in job.selected_tasks(assignment) {
            if remaining_sweeps == 0 {
                stopped_early = true;
                break;
            }
            let checkpoint_path = options
                .checkpoint_dir
                .as_ref()
                .map(|directory| generic_checkpoint_path(directory, task_index));
            let (task_result, used_sweeps) = self.run_task(
                task_index,
                task,
                options,
                checkpoint_path.as_deref(),
                remaining_sweeps,
            )?;
            remaining_sweeps = remaining_sweeps.saturating_sub(used_sweeps);
            if let Some(path) = checkpoint_path {
                checkpoint_paths.push(path);
            }
            stopped_early |= !task_result.completed;
            tasks.push(task_result);
            if stopped_early {
                break;
            }
        }

        let result = JobResult {
            job_name: job.name.clone(),
            rank: assignment.rank,
            world_size: assignment.world_size,
            tasks,
        };
        let output_path = if let Some(path) = &options.output_path {
            write_json_atomic(path, &result)?;
            Some(path.clone())
        } else {
            None
        };
        Ok(RunResult {
            result,
            output_path,
            checkpoint_paths,
            stopped_early,
        })
    }

    fn run_task(
        &self,
        task_index: usize,
        task: &Task<M::Parameters>,
        options: &RunOptions,
        checkpoint_path: Option<&Path>,
        sweep_budget: usize,
    ) -> Result<(TaskResult<M::Parameters>, usize), GenericJobError> {
        let restored = checkpoint_path
            .filter(|path| options.resume && path.exists())
            .map(read_json::<GenericCheckpoint<M::Parameters>>)
            .transpose()?;
        let (mut model, mut context, mut thermalization_sweeps, mut measurement_sweeps) =
            if let Some(checkpoint) = restored {
                if checkpoint.task != *task || checkpoint.task_index != task_index {
                    return Err(GenericJobError::CheckpointMismatch {
                        path: checkpoint_path
                            .expect("restored checkpoint has a path")
                            .to_path_buf(),
                        task: task.name.clone(),
                    });
                }
                let model = serde_json::from_value(checkpoint.model).map_err(|source| {
                    GenericJobError::Json {
                        path: checkpoint_path.map(Path::to_path_buf),
                        source,
                    }
                })?;
                (
                    model,
                    Context::restored(
                        task.seed,
                        checkpoint.rng_position,
                        task.binsize,
                        checkpoint.observables,
                        checkpoint.thermalization_sweeps >= task.thermalization,
                    ),
                    checkpoint.thermalization_sweeps.min(task.thermalization),
                    checkpoint.measurement_sweeps.min(task.sweeps),
                )
            } else {
                let mut model =
                    M::new(&task.parameters).map_err(|source| GenericJobError::Model {
                        task: task.name.clone(),
                        source: Box::new(source),
                    })?;
                let mut context = Context::fresh(task.seed, task.binsize);
                model
                    .init(&mut context)
                    .map_err(|source| GenericJobError::Model {
                        task: task.name.clone(),
                        source: Box::new(source),
                    })?;
                (model, context, 0, 0)
            };

        let mut used_sweeps = 0usize;
        while thermalization_sweeps < task.thermalization && used_sweeps < sweep_budget {
            model
                .sweep(&mut context)
                .map_err(|source| GenericJobError::Model {
                    task: task.name.clone(),
                    source: Box::new(source),
                })?;
            thermalization_sweeps += 1;
            used_sweeps += 1;
            maybe_write_generic_checkpoint(
                checkpoint_path,
                options.checkpoint_interval,
                used_sweeps,
                task,
                task_index,
                &model,
                &context,
                thermalization_sweeps,
                measurement_sweeps,
            )?;
        }
        context.thermalized = thermalization_sweeps >= task.thermalization;
        while measurement_sweeps < task.sweeps && used_sweeps < sweep_budget {
            model
                .sweep(&mut context)
                .map_err(|source| GenericJobError::Model {
                    task: task.name.clone(),
                    source: Box::new(source),
                })?;
            model
                .measure(&mut context)
                .map_err(|source| GenericJobError::Model {
                    task: task.name.clone(),
                    source: Box::new(source),
                })?;
            measurement_sweeps += 1;
            used_sweeps += 1;
            maybe_write_generic_checkpoint(
                checkpoint_path,
                options.checkpoint_interval,
                used_sweeps,
                task,
                task_index,
                &model,
                &context,
                thermalization_sweeps,
                measurement_sweeps,
            )?;
        }

        let completed =
            thermalization_sweeps == task.thermalization && measurement_sweeps == task.sweeps;
        if let Some(path) = checkpoint_path {
            write_generic_checkpoint(
                path,
                task,
                task_index,
                &model,
                &context,
                thermalization_sweeps,
                measurement_sweeps,
            )?;
        }
        let observables = if completed {
            context
                .observables
                .iter()
                .map(|(name, accumulator)| {
                    let estimate = accumulator.estimate().ok_or_else(|| {
                        GenericJobError::InsufficientSamples {
                            task: task.name.clone(),
                            observable: name.clone(),
                        }
                    })?;
                    Ok((
                        name.clone(),
                        ScalarEstimate {
                            mean: estimate.mean,
                            stderr: estimate.stderr,
                            internal_bins: estimate.internal_bins,
                            rebin_length: estimate.rebin_length,
                            rebin_count: estimate.rebin_count,
                            bin_length: estimate.internal_bin_length,
                        },
                    ))
                })
                .collect::<Result<_, GenericJobError>>()?
        } else {
            BTreeMap::new()
        };
        Ok((
            TaskResult {
                task_index,
                task: task.clone(),
                observables,
                thermalization_sweeps,
                measurement_sweeps,
                completed,
            },
            used_sweeps,
        ))
    }
}

fn validate_assignment(assignment: JobAssignment) -> Result<(), GenericJobError> {
    if assignment.world_size == 0 || assignment.rank >= assignment.world_size {
        return Err(GenericJobError::InvalidAssignment {
            rank: assignment.rank,
            world_size: assignment.world_size,
        });
    }
    Ok(())
}

fn validate_job<M: MonteCarlo>(job: &Job<M>) -> Result<(), GenericJobError> {
    let mut names = BTreeSet::new();
    for task in &job.tasks {
        let reason = if task.name.is_empty() {
            Some("task name must not be empty")
        } else if !names.insert(task.name.as_str()) {
            Some("task names must be unique")
        } else if task.sweeps == 0 {
            Some("sweeps must be positive")
        } else if task.binsize == 0 {
            Some("binsize must be positive")
        } else if task.sweeps < task.binsize {
            Some("sweeps must be at least binsize")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(GenericJobError::InvalidTask {
                task: task.name.clone(),
                reason,
            });
        }
    }
    Ok(())
}

fn generic_checkpoint_path(directory: &Path, task_index: usize) -> PathBuf {
    directory.join(format!("task{:04}.checkpoint.json", task_index + 1))
}

#[allow(clippy::too_many_arguments)]
fn maybe_write_generic_checkpoint<M: MonteCarlo>(
    path: Option<&Path>,
    interval: usize,
    used_sweeps: usize,
    task: &Task<M::Parameters>,
    task_index: usize,
    model: &M,
    context: &Context,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
) -> Result<(), GenericJobError> {
    if interval > 0 && used_sweeps.is_multiple_of(interval) {
        if let Some(path) = path {
            write_generic_checkpoint(
                path,
                task,
                task_index,
                model,
                context,
                thermalization_sweeps,
                measurement_sweeps,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_generic_checkpoint<M: MonteCarlo>(
    path: &Path,
    task: &Task<M::Parameters>,
    task_index: usize,
    model: &M,
    context: &Context,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
) -> Result<(), GenericJobError> {
    write_json_atomic(
        path,
        &GenericCheckpoint {
            task: task.clone(),
            task_index,
            model: serde_json::to_value(model).map_err(|source| GenericJobError::Json {
                path: Some(path.to_path_buf()),
                source,
            })?,
            rng_position: context.rng.position(),
            thermalization_sweeps,
            measurement_sweeps,
            observables: context.compact(),
        },
    )
}

fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<(), GenericJobError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| GenericJobError::Io {
            operation: "create directory",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temporary = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let file = File::create(&temporary).map_err(|source| GenericJobError::Io {
        operation: "create temporary JSON file",
        path: temporary.clone(),
        source,
    })?;
    serde_json::to_writer_pretty(BufWriter::new(file), value).map_err(|source| {
        GenericJobError::Json {
            path: Some(temporary.clone()),
            source,
        }
    })?;
    fs::rename(&temporary, path).map_err(|source| GenericJobError::Io {
        operation: "replace JSON file",
        path: path.to_path_buf(),
        source,
    })
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, GenericJobError> {
    let file = File::open(path).map_err(|source| GenericJobError::Io {
        operation: "open JSON file",
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(BufReader::new(file)).map_err(|source| GenericJobError::Json {
        path: Some(path.to_path_buf()),
        source,
    })
}

#[cfg(test)]
mod generic_job_tests {
    use super::*;
    use rand::RngExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestParameters {
        offset: f64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct TestModel {
        value: f64,
    }

    #[derive(Debug)]
    struct TestError(GenericJobError);

    impl fmt::Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(formatter)
        }
    }

    impl StdError for TestError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            Some(&self.0)
        }
    }

    impl MonteCarlo for TestModel {
        type Parameters = TestParameters;
        type Error = TestError;

        fn new(parameters: &Self::Parameters) -> Result<Self, Self::Error> {
            Ok(Self {
                value: parameters.offset,
            })
        }

        fn init(&mut self, context: &mut Context) -> Result<(), Self::Error> {
            assert!(!context.is_thermalized());
            Ok(())
        }

        fn sweep(&mut self, context: &mut Context) -> Result<(), Self::Error> {
            self.value += context.rng.random::<f64>();
            Ok(())
        }

        fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error> {
            assert!(context.is_thermalized());
            context.measure("Value", self.value).map_err(TestError)
        }
    }

    fn task(name: &str, offset: f64, seed: u64) -> Task<TestParameters> {
        Task::new(name, TestParameters { offset })
            .thermalization(3)
            .sweeps(12)
            .binsize(2)
            .seed(seed)
    }

    fn temporary_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "carlo-mc-job-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time after epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn task_maker_captures_parameter_snapshots() {
        let mut maker = TaskMaker::new(TestParameters { offset: 1.0 });
        maker
            .set_sweeps(12)
            .set_thermalization(3)
            .set_binsize(2)
            .set_seed(7)
            .add_task("first");
        maker.shared_mut().offset = 2.0;
        maker.add_task_with("override", |parameters| parameters.offset = 3.0);
        maker.add_task("second");
        let tasks = maker.make_tasks();

        assert_eq!(tasks[0].parameters.offset, 1.0);
        assert_eq!(tasks[1].parameters.offset, 3.0);
        assert_eq!(tasks[2].parameters.offset, 2.0);
        assert_eq!(tasks[0].sweeps, 12);
        assert_eq!(tasks[0].thermalization, 3);
        assert_eq!(tasks[0].binsize, 2);
        assert_eq!(tasks[0].seed, 7);
    }

    #[test]
    fn runner_is_deterministic_for_fixed_seed() {
        let job = Job::<TestModel>::new("deterministic", vec![task("only", 1.0, 42)]);
        let first = Runner::<TestModel>::new()
            .run(&job, &RunOptions::default())
            .expect("first run");
        let second = Runner::<TestModel>::new()
            .run(&job, &RunOptions::default())
            .expect("second run");

        assert_eq!(first.result, second.result);
        assert!(first.result.tasks[0].completed);
        assert_eq!(first.result.tasks[0].observables["Value"].internal_bins, 6);
    }

    #[test]
    fn runner_assigns_tasks_by_rank() {
        let tasks = (0..7)
            .map(|index| task(&format!("task-{index}"), index as f64, index as u64))
            .collect();
        let job = Job::<TestModel>::new("ranks", tasks);
        let options = RunOptions {
            assignment: Some(JobAssignment {
                rank: 1,
                world_size: 3,
            }),
            ..RunOptions::default()
        };
        let result = Runner::<TestModel>::new()
            .run(&job, &options)
            .expect("rank run");
        let indices = result
            .result
            .tasks
            .iter()
            .map(|task| task.task_index)
            .collect::<Vec<_>>();

        assert_eq!(indices, vec![1, 4]);
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_run() {
        let directory = temporary_directory("resume");
        let job = Job::<TestModel>::new("resume", vec![task("only", 2.0, 99)]);
        let uninterrupted = Runner::<TestModel>::new()
            .run(&job, &RunOptions::default())
            .expect("uninterrupted run");
        let partial_options = RunOptions {
            checkpoint_dir: Some(directory.clone()),
            checkpoint_interval: 1,
            sweep_limit: Some(8),
            ..RunOptions::default()
        };
        let partial = Runner::<TestModel>::new()
            .run(&job, &partial_options)
            .expect("partial run");
        assert!(partial.stopped_early);
        assert_eq!(partial.result.tasks[0].thermalization_sweeps, 3);
        assert_eq!(partial.result.tasks[0].measurement_sweeps, 5);

        let resumed = Runner::<TestModel>::new()
            .run(
                &job,
                &RunOptions {
                    checkpoint_dir: Some(directory.clone()),
                    resume: true,
                    checkpoint_interval: 1,
                    ..RunOptions::default()
                },
            )
            .expect("resumed run");

        assert_eq!(resumed.result, uninterrupted.result);
        fs::remove_dir_all(directory).expect("remove test checkpoint directory");
    }
}
