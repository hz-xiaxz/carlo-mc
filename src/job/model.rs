use crate::Context;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::error::Error;

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
    pub fn sweeps(mut self, v: usize) -> Self {
        self.sweeps = v;
        self
    }
    pub fn thermalization(mut self, v: usize) -> Self {
        self.thermalization = v;
        self
    }
    pub fn binsize(mut self, v: usize) -> Self {
        self.binsize = v;
        self
    }
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }
}
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
    pub fn set_sweeps(&mut self, v: usize) -> &mut Self {
        self.defaults.0 = v;
        self
    }
    pub fn set_thermalization(&mut self, v: usize) -> &mut Self {
        self.defaults.1 = v;
        self
    }
    pub fn set_binsize(&mut self, v: usize) -> &mut Self {
        self.defaults.2 = v;
        self
    }
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
    pub fn add_task(&mut self, name: impl Into<String>) -> &mut Self {
        self.push(name, self.shared.clone());
        self
    }
    pub fn add_task_with<F: FnOnce(&mut P)>(&mut self, name: impl Into<String>, f: F) -> &mut Self {
        let mut p = self.shared.clone();
        f(&mut p);
        self.push(name, p);
        self
    }
    pub fn tasks(&self) -> &[Task<P>] {
        &self.tasks
    }
    pub fn make_tasks(self) -> Vec<Task<P>> {
        self.tasks
    }
}
pub trait MonteCarlo: Sized + Serialize + DeserializeOwned {
    type Parameters: Clone + PartialEq + Serialize + DeserializeOwned;
    type Error: Error + Send + Sync + 'static;
    fn new(parameters: &Self::Parameters) -> Result<Self, Self::Error>;
    fn init(&mut self, context: &mut Context) -> Result<(), Self::Error>;
    fn sweep(&mut self, context: &mut Context) -> Result<(), Self::Error>;
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
