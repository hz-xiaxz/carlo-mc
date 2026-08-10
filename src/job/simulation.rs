use super::{
    checkpoint::{
        decode_rng_position, encode_rng_position, read_checkpoint, Checkpoint,
        CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
    },
    GenericJobError, JobAssignment, MonteCarlo, ScalarEstimate, Task, TaskResult,
};
use rand::{SeedableRng, TryRng};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, convert::Infallible, path::Path, time::Instant};
const GAMMA: u64 = 0x9E3779B97F4A7C15;
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    seed: u64,
    draws: u128,
}
impl DeterministicRng {
    pub fn position(&self) -> u128 {
        self.draws
    }
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompactAccumulator {
    pub internal_bins: Vec<f64>,
    pub pending_sum: f64,
    pub pending_count: usize,
    pub total_count: usize,
    pub binsize: usize,
}
#[derive(Debug)]
pub struct Context {
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
    pub fn is_thermalized(&self) -> bool {
        self.thermalized
    }
    pub fn measure(&mut self, n: impl Into<String>, v: f64) -> Result<(), GenericJobError> {
        self.measure_with_binsize(n, v, self.binsize)
    }
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

pub(crate) fn run_task<M, F>(
    task: &Task<M::Parameters>,
    runtime: TaskRuntime<'_>,
    mut maintain: F,
) -> Result<(TaskResult<M::Parameters>, usize), GenericJobError>
where
    M: MonteCarlo,
    F: FnMut(&Checkpoint<M::Parameters>, usize, bool) -> Result<(), GenericJobError>,
{
    let restored: Option<Checkpoint<M::Parameters>> = runtime
        .checkpoint_path
        .filter(|path| runtime.restore_checkpoint && path.exists())
        .map(read_checkpoint)
        .transpose()?;
    let (mut model, mut context, mut thermalization_sweeps, mut measurement_sweeps) =
        if let Some(checkpoint) = restored {
            if checkpoint.schema_version != CHECKPOINT_PAYLOAD_SCHEMA_VERSION
                || (!runtime.allow_different_rank && checkpoint.rank != runtime.assignment.rank)
                || checkpoint.world_size != runtime.assignment.world_size
                || checkpoint.world_size == 0
                || checkpoint.rank >= checkpoint.world_size
                || checkpoint.job_name != runtime.job_name
                || checkpoint.task != *task
                || checkpoint.task_index != runtime.task_index
                || validate_checkpoint_state(task, &checkpoint).is_err()
            {
                return Err(GenericJobError::CheckpointMismatch {
                    path: runtime.checkpoint_path.unwrap().into(),
                    task: task.name.clone(),
                });
            }
            let model: M = serde_json::from_value(checkpoint.model.clone())
                .map_err(|error| GenericJobError::json(runtime.checkpoint_path, error))?;
            let restored_model = serde_json::to_value(&model)
                .map_err(|error| GenericJobError::json(runtime.checkpoint_path, error))?;
            if restored_model != checkpoint.model {
                return Err(GenericJobError::CheckpointMismatch {
                    path: runtime.checkpoint_path.unwrap().into(),
                    task: task.name.clone(),
                });
            }
            (
                model,
                Context::restored(
                    task.seed,
                    decode_rng_position(checkpoint.rng_position_words),
                    task.binsize,
                    checkpoint.observables,
                    checkpoint.thermalization_sweeps >= task.thermalization,
                ),
                checkpoint.thermalization_sweeps,
                checkpoint.measurement_sweeps,
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
        let checkpoint = make_checkpoint(
            &runtime,
            task,
            &model,
            &context,
            thermalization_sweeps,
            measurement_sweeps,
        )?;
        maintain(&checkpoint, used, false)?;
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
        let checkpoint = make_checkpoint(
            &runtime,
            task,
            &model,
            &context,
            thermalization_sweeps,
            measurement_sweeps,
        )?;
        maintain(&checkpoint, used, false)?;
    }

    let checkpoint = make_checkpoint(
        &runtime,
        task,
        &model,
        &context,
        thermalization_sweeps,
        measurement_sweeps,
    )?;
    maintain(&checkpoint, used, true)?;
    let completed =
        thermalization_sweeps == task.thermalization && measurement_sweeps == task.sweeps;
    let observables = if completed {
        estimates(task, &context)?
    } else {
        BTreeMap::new()
    };
    Ok((
        TaskResult {
            task_index: runtime.task_index,
            task: task.clone(),
            observables,
            thermalization_sweeps,
            measurement_sweeps,
            completed,
        },
        used,
    ))
}

fn make_checkpoint<M: MonteCarlo>(
    runtime: &TaskRuntime<'_>,
    task: &Task<M::Parameters>,
    model: &M,
    context: &Context,
    thermalization_sweeps: usize,
    measurement_sweeps: usize,
) -> Result<Checkpoint<M::Parameters>, GenericJobError> {
    Ok(Checkpoint {
        schema_version: CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
        rank: runtime.assignment.rank,
        world_size: runtime.assignment.world_size,
        job_name: runtime.job_name.into(),
        task: task.clone(),
        task_index: runtime.task_index,
        model: serde_json::to_value(model).map_err(|error| GenericJobError::json(None, error))?,
        rng_position_words: encode_rng_position(context.rng.position()),
        thermalization_sweeps,
        measurement_sweeps,
        observables: context.observables.clone(),
    })
}

fn validate_checkpoint_state<P>(
    task: &Task<P>,
    checkpoint: &Checkpoint<P>,
) -> Result<(), &'static str> {
    if checkpoint.thermalization_sweeps > task.thermalization
        || checkpoint.measurement_sweeps > task.sweeps
    {
        return Err("checkpoint sweep count exceeds task");
    }
    if checkpoint.measurement_sweeps > 0 && checkpoint.thermalization_sweeps != task.thermalization
    {
        return Err("checkpoint measured before thermalization completed");
    }
    for (name, accumulator) in &checkpoint.observables {
        if name.is_empty() {
            return Err("checkpoint has an empty observable name");
        }
        if accumulator.binsize == 0
            || accumulator.pending_count >= accumulator.binsize
            || accumulator.total_count > checkpoint.measurement_sweeps
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

fn estimates<P>(
    task: &Task<P>,
    context: &Context,
) -> Result<BTreeMap<String, ScalarEstimate>, GenericJobError> {
    context
        .observables
        .iter()
        .map(|(name, accumulator)| {
            if accumulator.internal_bins.is_empty() {
                return Err(GenericJobError::InsufficientSamples {
                    task: task.name.clone(),
                    observable: name.clone(),
                });
            }
            let rebin_length = super::results::rebin_length(accumulator.internal_bins.len());
            let bins = accumulator
                .internal_bins
                .chunks_exact(rebin_length)
                .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                .collect::<Vec<_>>();
            let mean = bins.iter().sum::<f64>() / bins.len() as f64;
            let stderr = if bins.len() > 1 {
                (bins.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                    / (bins.len() - 1) as f64)
                    .sqrt()
                    / (bins.len() as f64).sqrt()
            } else {
                f64::NAN
            };
            Ok((
                name.clone(),
                ScalarEstimate {
                    mean,
                    stderr,
                    internal_bins: accumulator.internal_bins.len(),
                    rebin_length,
                    rebin_count: bins.len(),
                    bin_length: accumulator.binsize,
                },
            ))
        })
        .collect()
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
