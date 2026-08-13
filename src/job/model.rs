use super::checkpoint::{
    decode_rng_position, encode_rng_position, read_checkpoint as read_default_checkpoint,
    write_checkpoint as write_default_checkpoint, Checkpoint, CheckpointState, RestoredState,
    CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
};
use crate::{default_estimates, Context, GenericJobError, ResultEstimate};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::BTreeMap, error::Error, path::Path};

/// One independent Monte Carlo run: a parameter point plus its execution settings.
///
/// A task is identified by `name` within a [`Job`](crate::Job). `sweeps` is the number of
/// measurement sweeps, `thermalization` is the number of warm-up sweeps discarded before
/// measuring, `binsize` is the number of raw samples averaged into one internal bin, and
/// `seed` initializes the deterministic RNG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task<P> {
    pub name: String,
    pub parameters: P,
    pub sweeps: usize,
    pub thermalization: usize,
    pub binsize: usize,
    pub seed: u64,
}
impl<P> Task<P> {
    /// Creates a task with the defaults `sweeps = 1`, `thermalization = 0`,
    /// `binsize = 1`, and `seed = 0`. Use the builder methods to override them.
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
    #[must_use]
    pub fn sweeps(mut self, v: usize) -> Self {
        self.sweeps = v;
        self
    }
    /// Sets the number of thermalization sweeps discarded before measurement.
    #[must_use]
    pub fn thermalization(mut self, v: usize) -> Self {
        self.thermalization = v;
        self
    }
    /// Sets the number of raw samples averaged into one internal bin.
    #[must_use]
    pub fn binsize(mut self, v: usize) -> Self {
        self.binsize = v;
        self
    }
    /// Sets the deterministic RNG seed.
    #[must_use]
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }
}
/// A fluent builder for creating many tasks from a shared parameter template.
///
/// Each task inherits the current defaults (sweeps, thermalization, binsize, seed).
/// Use [`TaskMaker::add_task`] to add the shared parameters, or
/// [`TaskMaker::add_task_with`] to derive a mutated parameter set for one task.
#[derive(Debug, Clone)]
pub struct TaskMaker<P> {
    shared: P,
    defaults: (usize, usize, usize, u64),
    tasks: Vec<Task<P>>,
}
impl<P: Clone> TaskMaker<P> {
    pub fn new(shared: P) -> Self {
        Self {
            shared,
            defaults: (1, 0, 1, 0),
            tasks: vec![],
        }
    }
    pub fn set_shared(&mut self, v: P) -> &mut Self {
        self.shared = v;
        self
    }
    pub fn shared_mut(&mut self) -> &mut P {
        &mut self.shared
    }
    /// Sets the default number of measurement sweeps for subsequently added tasks.
    pub fn set_sweeps(&mut self, v: usize) -> &mut Self {
        self.defaults.0 = v;
        self
    }
    /// Sets the default number of thermalization sweeps for subsequently added tasks.
    pub fn set_thermalization(&mut self, v: usize) -> &mut Self {
        self.defaults.1 = v;
        self
    }
    /// Sets the default bin size for subsequently added tasks.
    pub fn set_binsize(&mut self, v: usize) -> &mut Self {
        self.defaults.2 = v;
        self
    }
    /// Sets the default RNG seed for subsequently added tasks.
    pub fn set_seed(&mut self, v: u64) -> &mut Self {
        self.defaults.3 = v;
        self
    }
    fn push(&mut self, name: impl Into<String>, p: P) {
        let d = self.defaults;
        self.tasks.push(
            Task::new(name, p)
                .sweeps(d.0)
                .thermalization(d.1)
                .binsize(d.2)
                .seed(d.3),
        );
    }
    /// Adds a task using the shared parameter template.
    pub fn add_task(&mut self, name: impl Into<String>) -> &mut Self {
        self.push(name, self.shared.clone());
        self
    }
    /// Adds a task whose parameters are the shared template after applying `f`.
    pub fn add_task_with<F: FnOnce(&mut P)>(&mut self, name: impl Into<String>, f: F) -> &mut Self {
        let mut p = self.shared.clone();
        f(&mut p);
        self.push(name, p);
        self
    }
    /// Returns the tasks accumulated so far.
    pub fn tasks(&self) -> &[Task<P>] {
        &self.tasks
    }
    /// Consumes the builder and returns the accumulated tasks.
    pub fn make_tasks(self) -> Vec<Task<P>> {
        self.tasks
    }
}
/// The model interface that users implement.
///
/// The model type must be `Serialize` and `DeserializeOwned` so its state can be written to
/// and restored from HDF5 checkpoints. [`MonteCarlo::Parameters`] is the associated
/// per-task configuration type; it must be clonable, comparable, and serializable.
///
/// The lifecycle is:
/// 1. [`MonteCarlo::new`] constructs the model from a task's parameters.
/// 2. [`MonteCarlo::init`] performs one-time setup, optionally drawing from `context.rng`.
/// 3. [`MonteCarlo::sweep`] advances the simulation by one step.
/// 4. [`MonteCarlo::measure`] records observables into the [`Context`].
pub trait MonteCarlo: Sized + Serialize + DeserializeOwned {
    type Parameters: Clone + PartialEq + Serialize + DeserializeOwned;
    type Error: Error + Send + Sync + 'static;
    /// The per-observable estimate type produced by [`MonteCarlo::finalize_estimates`].
    ///
    /// The default [`crate::Estimate`] is a fully featured binned estimate with error,
    /// covariance, and autocorrelation time. A model can substitute its own serializable
    /// estimate type, as long as it is constructible from the default [`crate::Estimate`]
    /// (which keeps the framework's default finalization path usable).
    type Estimate: ResultEstimate + From<crate::Estimate>;
    /// Constructs the model for the given task parameters.
    fn new(parameters: &Self::Parameters) -> Result<Self, Self::Error>;
    /// Performs one-time initialization before the first sweep.
    fn init(&mut self, context: &mut Context) -> Result<(), Self::Error>;
    /// Advances the simulation by one Monte Carlo sweep.
    fn sweep(&mut self, context: &mut Context) -> Result<(), Self::Error>;
    /// Records observables, typically via [`Context::measure`].
    fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error>;

    /// Expands a configuration table into the model's task grid.
    ///
    /// This is the generic config-to-task hook. Models with a parameter-grid
    /// convention override it to describe how their [`Self::Parameters`] are
    /// derived from a [`Params`](crate::Params) dictionary (for example,
    /// sweeping over lattice sizes, temperatures, and disorder samples).
    /// The returned tasks can be passed directly to [`Job::new`](crate::Job::new).
    ///
    /// The default implementation returns an empty grid; models without a
    /// config-driven grid can rely on [`TaskMaker`] instead.
    fn build_tasks(_config: &crate::Params) -> Result<Vec<Task<Self::Parameters>>, Self::Error> {
        Ok(Vec::new())
    }

    /// Converts measured raw bins into final per-observable estimates.
    ///
    /// `raw_bins` maps each observable name to its completed internal-bin averages, and
    /// `bin_lengths` maps the same name to the number of raw samples per internal bin. The
    /// runner collects these from every [`Context::measure`] call and invokes this hook once
    /// per completed task, after [`MonteCarlo::measure`] has finished.
    ///
    /// The default implementation rebins each series with Carlo-style rebinning and wraps the
    /// result in [`Self::Estimate`]. Models that compute *derived* observables (for
    /// example ratios or nonlinear functions of measured ones, via
    /// [`Evaluator`](crate::Evaluator) jackknife) should override this method, using
    /// [`default_estimates`] as a starting point for the directly measured observables.
    fn finalize_estimates(
        &self,
        _parameters: &Self::Parameters,
        raw_bins: &BTreeMap<String, Vec<f64>>,
        bin_lengths: &BTreeMap<String, usize>,
    ) -> Result<BTreeMap<String, Self::Estimate>, GenericJobError> {
        Ok(default_estimates(raw_bins, bin_lengths)
            .into_iter()
            .map(|(name, estimate)| (name, Self::Estimate::from(estimate)))
            .collect())
    }

    /// Writes a checkpoint for this model at `path`.
    ///
    /// `state` carries everything the runner owns (task, assignment, RNG position, sweep
    /// progress, and observable bins). The model is responsible for persisting its own state
    /// (via `&self`) together with `state`, in whatever on-disk format it needs. This is what
    /// lets a model keep a fully custom checkpoint layout that the framework does not dictate.
    ///
    /// The default implementation uses the framework's JSON-model HDF5 codec: the model is
    /// serialized to JSON and stored under `metadata/model`, with `state` spread across the
    /// standard `assignment`/`progress`/`state`/`observables` groups.
    fn write_checkpoint(
        &self,
        state: &CheckpointState<Self::Parameters>,
        path: &Path,
    ) -> Result<(), GenericJobError> {
        let model =
            serde_json::to_value(self).map_err(|error| GenericJobError::json(Some(path), error))?;
        let checkpoint = Checkpoint {
            schema_version: CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
            rank: state.rank,
            world_size: state.world_size,
            job_name: state.job_name.clone(),
            task: state.task.clone(),
            task_index: state.task_index,
            model,
            rng_position_words: encode_rng_position(state.rng_position),
            thermalization_sweeps: state.thermalization_sweeps,
            measurement_sweeps: state.measurement_sweeps,
            observables: state.observables.clone(),
        };
        write_default_checkpoint(path, &checkpoint)
    }

    /// Restores a model and the runner state from a checkpoint previously written by
    /// [`MonteCarlo::write_checkpoint`].
    ///
    /// This is an associated function (not a method) because it reconstructs the model. The
    /// returned [`RestoredState`] carries both the reconstructed model and the runner-owned
    /// fields; the runner validates those fields against the run being resumed before
    /// continuing the sweep loop.
    ///
    /// The default implementation reads the framework's JSON-model HDF5 codec. A model that
    /// overrides [`MonteCarlo::write_checkpoint`] with a custom format must override this
    /// method symmetrically.
    fn read_checkpoint(path: &Path) -> Result<RestoredState<Self>, GenericJobError> {
        let checkpoint: Checkpoint<Self::Parameters> = read_default_checkpoint(path)?;
        let model: Self = serde_json::from_value(checkpoint.model.clone())
            .map_err(|error| GenericJobError::json(Some(path), error))?;
        let restored_model = serde_json::to_value(&model)
            .map_err(|error| GenericJobError::json(Some(path), error))?;
        if restored_model != checkpoint.model {
            return Err(GenericJobError::CheckpointMismatch {
                path: path.into(),
                task: checkpoint.task.name.clone(),
            });
        }
        Ok(RestoredState {
            model,
            task: checkpoint.task,
            rank: checkpoint.rank,
            world_size: checkpoint.world_size,
            job_name: checkpoint.job_name,
            task_index: checkpoint.task_index,
            rng_position: decode_rng_position(checkpoint.rng_position_words),
            thermalization_sweeps: checkpoint.thermalization_sweeps,
            measurement_sweeps: checkpoint.measurement_sweeps,
            observables: checkpoint.observables,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builders() {
        let mut m = TaskMaker::new(1);
        m.set_sweeps(4).set_binsize(2).add_task("a");
        *m.shared_mut() = 2;
        m.add_task_with("b", |p| *p = 3);
        assert_eq!(m.tasks()[0].parameters, 1);
        assert_eq!(m.tasks()[1].parameters, 3);
    }
    #[test]
    fn task_defaults() {
        let t = Task::new("x", ());
        assert_eq!(
            (t.sweeps, t.thermalization, t.binsize, t.seed),
            (1, 0, 1, 0)
        );
    }
}
