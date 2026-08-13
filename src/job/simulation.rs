use super::{
    checkpoint::{CheckpointState, RestoredState},
    GenericJobError, JobAssignment, MonteCarlo, Task, TaskResult,
};
use rand::{SeedableRng, TryRng};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, convert::Infallible, path::Path, time::Instant};
const GAMMA: u64 = 0x9E3779B97F4A7C15;
/// A deterministic, seekable RNG used for reproducible runs.
///
/// It implements `rand::SeedableRng` and `rand::TryRng` (with an infallible error type), so
/// it automatically gains `rand::Rng` and `rand::RngExt` methods such as
/// `next_u64`, `random::<f64>()`, and `random_bool`. The internal counter can be inspected
/// and rewound with [`DeterministicRng::position`] and [`DeterministicRng::set_position`],
/// which is what makes checkpoint resume exact.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    seed: u64,
    draws: u128,
}
impl DeterministicRng {
    /// Returns the number of u64 draws consumed so far.
    pub fn position(&self) -> u128 {
        self.draws
    }
    /// Rewinds or fast-forwards the draw counter to an exact position.
    pub fn set_position(&mut self, position: u128) {
        self.draws = position;
    }
    fn mix(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }
    fn at(&self, position: u128) -> u64 {
        let low = position as u64;
        let high = (position >> 64) as u64;
        let counter = self
            .seed
            .wrapping_add(GAMMA.wrapping_mul(low.wrapping_add(1)));
        if high == 0 {
            Self::mix(counter)
        } else {
            Self::mix(counter ^ Self::mix(high))
        }
    }
}
impl SeedableRng for DeterministicRng {
    type Seed = [u8; 8];
    fn from_seed(s: Self::Seed) -> Self {
        Self::seed_from_u64(u64::from_le_bytes(s))
    }
    fn seed_from_u64(seed: u64) -> Self {
        Self { seed, draws: 0 }
    }
}
impl TryRng for DeterministicRng {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let value = self.at(self.draws);
        self.draws = self.draws.wrapping_add(1);
        Ok(value)
    }
    fn try_fill_bytes(&mut self, d: &mut [u8]) -> Result<(), Self::Error> {
        for c in d.chunks_mut(8) {
            let w = self.try_next_u64()?.to_le_bytes();
            c.copy_from_slice(&w[..c.len()])
        }
        Ok(())
    }
}
/// Accumulates raw samples for one observable until they form completed internal bins.
///
/// `internal_bins` holds the completed bin averages, `pending_sum`/`pending_count` track the
/// partially filled bin, `total_count` is the number of samples accepted so far, and
/// `binsize` is the fixed number of samples per internal bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactAccumulator {
    pub internal_bins: Vec<f64>,
    pub pending_sum: f64,
    pub pending_count: usize,
    pub total_count: usize,
    pub binsize: usize,
}
/// The execution context passed to every [`MonteCarlo`](crate::MonteCarlo) callback.
///
/// It owns the task's deterministic RNG (`context.rng`) and the observable accumulators.
/// Models should draw random numbers from `context.rng` (never from a separate RNG) and
/// record observables with [`Context::measure`] or [`Context::measure_with_binsize`].
#[derive(Debug)]
pub struct Context {
    /// Deterministic RNG for this task.
    pub rng: DeterministicRng,
    pub(crate) thermalized: bool,
    binsize: usize,
    pub(crate) observables: BTreeMap<String, CompactAccumulator>,
}
impl Context {
    pub(crate) fn fresh(seed: u64, binsize: usize) -> Self {
        Self {
            rng: DeterministicRng::seed_from_u64(seed),
            thermalized: false,
            binsize,
            observables: BTreeMap::new(),
        }
    }
    pub(crate) fn restored(
        seed: u64,
        pos: u128,
        binsize: usize,
        o: BTreeMap<String, CompactAccumulator>,
        t: bool,
    ) -> Self {
        let mut x = Self::fresh(seed, binsize);
        x.rng.set_position(pos);
        x.observables = o;
        x.thermalized = t;
        x
    }
    /// Whether the configured number of thermalization sweeps has completed.
    pub fn is_thermalized(&self) -> bool {
        self.thermalized
    }
    /// Records one scalar sample for `n` using the task's default bin size.
    ///
    /// Samples are accumulated into internal bins; a completed internal bin contributes
    /// one averaged value to the final estimate. The observable name must be non-empty and
    /// the value finite.
    pub fn measure(&mut self, n: impl Into<String>, v: f64) -> Result<(), GenericJobError> {
        self.measure_with_binsize(n, v, self.binsize)
    }
    /// Records one scalar sample for `n` using an explicit bin size.
    ///
    /// The bin size is fixed on first use for a given observable name; calling this again
    /// with a different bin size for the same name is an error.
    pub fn measure_with_binsize(
        &mut self,
        n: impl Into<String>,
        v: f64,
        b: usize,
    ) -> Result<(), GenericJobError> {
        let n = n.into();
        if n.is_empty() || !v.is_finite() || b == 0 {
            return Err(GenericJobError::InvalidMeasurement {
                observable: n,
                reason: "name/value/bin size invalid",
            });
        }
        let a = self
            .observables
            .entry(n.clone())
            .or_insert(CompactAccumulator {
                internal_bins: vec![],
                pending_sum: 0.,
                pending_count: 0,
                total_count: 0,
                binsize: b,
            });
        if a.binsize != b {
            return Err(GenericJobError::InvalidMeasurement {
                observable: n,
                reason: "bin size changed",
            });
        }
        let sum = a.pending_sum + v;
        if !sum.is_finite() {
            return Err(GenericJobError::InvalidMeasurement {
                observable: n,
                reason: "observable accumulation overflowed",
            });
        }
        a.pending_sum = sum;
        a.pending_count += 1;
        a.total_count += 1;
        if a.pending_count == b {
            a.internal_bins.push(a.pending_sum / b as f64);
            a.pending_sum = 0.;
            a.pending_count = 0
        }
        Ok(())
    }

    /// Returns the completed internal bins recorded for `name`, if any.
    ///
    /// These are the raw bin averages before rebinning; they are useful for
    /// models that want to export or post-process samples themselves (for
    /// example, to compute jackknife errors over nonlinear observables).
    pub fn raw_bins(&self, name: &str) -> Option<&[f64]> {
        self.observables
            .get(name)
            .map(|a| a.internal_bins.as_slice())
    }

    /// Returns the fixed bin size (samples per internal bin) for `name`, if any.
    pub fn bin_length(&self, name: &str) -> Option<usize> {
        self.observables.get(name).map(|a| a.binsize)
    }

    /// Iterates over all measured observables and their accumulators.
    ///
    /// This is the raw view used by [`MonteCarlo::finalize_estimates`](crate::MonteCarlo::finalize_estimates)
    /// when a model wants to post-process completed bins itself (for example to derive
    /// additional observables through jackknife).
    pub fn observables(&self) -> impl Iterator<Item = (&str, &CompactAccumulator)> {
        self.observables
            .iter()
            .map(|(name, accumulator)| (name.as_str(), accumulator))
    }
}

pub(crate) struct TaskRuntime<'a> {
    pub job_name: &'a str,
    pub task_index: usize,
    pub assignment: JobAssignment,
    pub checkpoint_path: Option<&'a Path>,
    pub restore_checkpoint: bool,
    pub allow_different_rank: bool,
    pub sweep_budget: usize,
    pub deadline: Option<Instant>,
}

/// Return type of [`run_task`]: the task's result and the number of sweeps it used.
type RunTaskResult<M> = Result<
    (
        TaskResult<<M as MonteCarlo>::Parameters, <M as MonteCarlo>::Estimate>,
        usize,
    ),
    GenericJobError,
>;
pub(crate) fn run_task<M, F>(
    task: &Task<M::Parameters>,
    runtime: TaskRuntime<'_>,
    mut maintain: F,
) -> RunTaskResult<M>
where
    M: MonteCarlo,
    F: FnMut(&CheckpointState<M::Parameters>, &M, usize, bool) -> Result<(), GenericJobError>,
{
    let restored: Option<RestoredState<M>> = runtime
        .checkpoint_path
        .filter(|path| runtime.restore_checkpoint && path.exists())
        .map(M::read_checkpoint)
        .transpose()?;
    let (mut model, mut context, mut thermalization_sweeps, mut measurement_sweeps) =
        if let Some(restored) = restored {
            if (!runtime.allow_different_rank && restored.rank != runtime.assignment.rank)
                || restored.world_size != runtime.assignment.world_size
                || restored.world_size == 0
                || restored.rank >= restored.world_size
                || restored.job_name != runtime.job_name
                || restored.task != *task
                || restored.task_index != runtime.task_index
                || validate_restored_state(task, &restored).is_err()
            {
                return Err(GenericJobError::CheckpointMismatch {
                    path: runtime.checkpoint_path.unwrap().into(),
                    task: task.name.clone(),
                });
            }
            (
                restored.model,
                Context::restored(
                    task.seed,
                    restored.rng_position,
                    task.binsize,
                    restored.observables,
                    restored.thermalization_sweeps >= task.thermalization,
                ),
                restored.thermalization_sweeps,
                restored.measurement_sweeps,
            )
        } else {
            let mut model = M::new(&task.parameters).map_err(|error| GenericJobError::Model {
                task: task.name.clone(),
                source: Box::new(error),
            })?;
            let mut context = Context::fresh(task.seed, task.binsize);
            model
                .init(&mut context)
                .map_err(|error| GenericJobError::Model {
                    task: task.name.clone(),
                    source: Box::new(error),
                })?;
            (model, context, 0, 0)
        };

    let mut used = 0;
    while thermalization_sweeps < task.thermalization
        && used < runtime.sweep_budget
        && runtime
            .deadline
            .is_none_or(|deadline| Instant::now() < deadline)
    {
        model
            .sweep(&mut context)
            .map_err(|error| GenericJobError::Model {
                task: task.name.clone(),
                source: Box::new(error),
            })?;
        thermalization_sweeps += 1;
        used += 1;
        let state = make_checkpoint_state(
            &runtime,
            task,
            &context,
            thermalization_sweeps,
            measurement_sweeps,
        );
        maintain(&state, &model, used, false)?;
    }
    context.thermalized = thermalization_sweeps >= task.thermalization;
    while measurement_sweeps < task.sweeps
        && used < runtime.sweep_budget
        && runtime
            .deadline
            .is_none_or(|deadline| Instant::now() < deadline)
    {
        model
            .sweep(&mut context)
            .map_err(|error| GenericJobError::Model {
                task: task.name.clone(),
                source: Box::new(error),
            })?;
        model
            .measure(&mut context)
            .map_err(|error| GenericJobError::Model {
                task: task.name.clone(),
                source: Box::new(error),
            })?;
        measurement_sweeps += 1;
        used += 1;
        let state = make_checkpoint_state(
            &runtime,
            task,
            &context,
            thermalization_sweeps,
            measurement_sweeps,
        );
        maintain(&state, &model, used, false)?;
    }

    let state = make_checkpoint_state(
        &runtime,
        task,
        &context,
        thermalization_sweeps,
        measurement_sweeps,
    );
    maintain(&state, &model, used, true)?;
    let completed =
        thermalization_sweeps == task.thermalization && measurement_sweeps == task.sweeps;
    let (observables, measurement_bins) = if completed {
        for (name, accumulator) in &context.observables {
            if accumulator.internal_bins.is_empty() {
                return Err(GenericJobError::InsufficientSamples {
                    task: task.name.clone(),
                    observable: name.clone(),
                });
            }
        }
        let raw_bins = context
            .observables
            .iter()
            .map(|(name, accumulator)| (name.clone(), accumulator.internal_bins.clone()))
            .collect::<BTreeMap<_, _>>();
        let bin_lengths = context
            .observables
            .iter()
            .map(|(name, accumulator)| (name.clone(), accumulator.binsize))
            .collect::<BTreeMap<_, _>>();
        let estimates = model.finalize_estimates(&task.parameters, &raw_bins, &bin_lengths)?;
        (estimates, raw_bins)
    } else {
        (BTreeMap::new(), BTreeMap::new())
    };
    Ok((
        TaskResult {
            task_index: runtime.task_index,
            task: task.clone(),
            observables,
            thermalization_sweeps,
            measurement_sweeps,
            completed,
            metadata: model.task_metadata(),
            measurement_bins,
        },
        used,
    ))
}

fn make_checkpoint_state<P: Clone>(
    runtime: &TaskRuntime<'_>,
    task: &Task<P>,
    context: &Context,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
) -> CheckpointState<P> {
    CheckpointState {
        rank: runtime.assignment.rank,
        world_size: runtime.assignment.world_size,
        job_name: runtime.job_name.into(),
        task: task.clone(),
        task_index: runtime.task_index,
        rng_position: context.rng.position(),
        thermalization_sweeps,
        measurement_sweeps,
        observables: context.observables.clone(),
    }
}

fn validate_restored_state<M: MonteCarlo>(
    task: &Task<M::Parameters>,
    restored: &RestoredState<M>,
) -> Result<(), &'static str> {
    if restored.thermalization_sweeps > task.thermalization
        || restored.measurement_sweeps > task.sweeps
    {
        return Err("checkpoint sweep count exceeds task");
    }
    if restored.measurement_sweeps > 0 && restored.thermalization_sweeps != task.thermalization {
        return Err("checkpoint measured before thermalization completed");
    }
    for (name, accumulator) in &restored.observables {
        if name.is_empty() {
            return Err("checkpoint has an empty observable name");
        }
        if accumulator.binsize == 0
            || accumulator.pending_count >= accumulator.binsize
            || accumulator.total_count > restored.measurement_sweeps
        {
            return Err("checkpoint accumulator has invalid bin state");
        }
        if accumulator.pending_count == 0 && accumulator.pending_sum != 0.0 {
            return Err("checkpoint accumulator has a residual completed-bin sum");
        }
        let complete_samples = accumulator
            .internal_bins
            .len()
            .checked_mul(accumulator.binsize)
            .and_then(|samples| samples.checked_add(accumulator.pending_count))
            .ok_or("checkpoint accumulator total overflow")?;
        if accumulator.total_count != complete_samples {
            return Err("checkpoint accumulator total is inconsistent");
        }
        if !accumulator.pending_sum.is_finite()
            || accumulator
                .internal_bins
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("checkpoint accumulator contains a non-finite value");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rng_resume() {
        let mut rng = DeterministicRng::seed_from_u64(3);
        let _ = rng.try_next_u64();
        let position = rng.position();
        let expected = rng.try_next_u64().unwrap();
        rng.set_position(position);
        assert_eq!(rng.try_next_u64().unwrap(), expected);
    }

    #[test]
    fn rng_position_uses_all_u128_bits() {
        let mut rng = DeterministicRng::seed_from_u64(3);
        rng.set_position(7);
        let low = rng.try_next_u64().unwrap();
        rng.set_position((1_u128 << 64) | 7);
        let high = rng.try_next_u64().unwrap();
        assert_ne!(low, high);
        assert_eq!(rng.position(), (1_u128 << 64) | 8);
    }
    #[test]
    fn measurement_validation() {
        let mut c = Context::fresh(0, 2);
        assert!(c.measure("", 1.).is_err());
        assert!(c.measure("x", f64::NAN).is_err());
        c.measure("x", 1.).unwrap();
        c.measure("x", 3.).unwrap();
        assert_eq!(c.observables["x"].internal_bins, vec![2.]);
        assert!(c.measure("overflow", f64::MAX).is_ok());
        assert!(c.measure("overflow", f64::MAX).is_err());
        assert!(c.observables["overflow"].pending_sum.is_finite());
        assert_eq!(c.observables["overflow"].total_count, 1);
    }
}
