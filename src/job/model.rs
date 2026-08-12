use crate::Context;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::error::Error;

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
    /// Constructs the model for the given task parameters.
    fn new(parameters: &Self::Parameters) -> Result<Self, Self::Error>;
    /// Performs one-time initialization before the first sweep.
    fn init(&mut self, context: &mut Context) -> Result<(), Self::Error>;
    /// Advances the simulation by one Monte Carlo sweep.
    fn sweep(&mut self, context: &mut Context) -> Result<(), Self::Error>;
    /// Records observables, typically via [`Context::measure`].
    fn measure(&mut self, context: &mut Context) -> Result<(), Self::Error>;
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
