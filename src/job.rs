mod checkpoint;
pub mod model;
mod paths;
mod results;
mod runner;
mod simulation;

use serde::{Deserialize, Serialize};
use std::{error::Error, fmt, marker::PhantomData};

pub use checkpoint::*;
pub use model::{MonteCarlo, Task, TaskMaker};
pub use paths::*;
pub use results::*;
pub use runner::*;
pub use simulation::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobAssignment {
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssignmentError {
    pub rank: usize,
    pub world_size: usize,
}
impl fmt::Display for AssignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.world_size == 0 {
            write!(f, "world_size must be positive")
        } else {
            write!(f, "rank must be smaller than world_size")
        }
    }
}
impl Error for AssignmentError {}
impl JobAssignment {
    pub fn new(rank: usize, world_size: usize) -> Result<Self, AssignmentError> {
        if world_size == 0 || rank >= world_size {
            Err(AssignmentError { rank, world_size })
        } else {
            Ok(Self { rank, world_size })
        }
    }
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
    pub fn from_env() -> Result<Self, AssignmentError> {
        Self::new(
            env_any(&[
                "XY_RANK",
                "SLURM_PROCID",
                "OMPI_COMM_WORLD_RANK",
                "PMI_RANK",
                "PMIX_RANK",
                "MV2_COMM_WORLD_RANK",
            ])
            .unwrap_or(0),
            env_any(&[
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
fn env_any(keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|k| std::env::var(k).ok()?.parse().ok())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job<M: MonteCarlo> {
    pub name: String,
    pub tasks: Vec<Task<M::Parameters>>,
    #[serde(skip)]
    model: PhantomData<fn() -> M>,
}
impl<M: MonteCarlo> Job<M> {
    pub fn new(name: impl Into<String>, tasks: Vec<Task<M::Parameters>>) -> Self {
        Self {
            name: name.into(),
            tasks,
            model: PhantomData,
        }
    }
    pub fn selected_tasks(
        &self,
        a: JobAssignment,
    ) -> impl Iterator<Item = (usize, &Task<M::Parameters>)> {
        self.tasks
            .iter()
            .enumerate()
            .filter(move |(i, _)| i % a.world_size == a.rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    use serde::{Deserialize, Serialize};
    use std::convert::Infallible;
    #[derive(Serialize, Deserialize)]
    struct M;
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct P;
    impl MonteCarlo for M {
        type Parameters = P;
        type Error = Infallible;
        fn new(_: &P) -> Result<Self, Self::Error> {
            Ok(M)
        }
        fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
        fn sweep(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
        fn measure(&mut self, _: &mut Context) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    #[test]
    fn assignment_validation() {
        assert!(JobAssignment::new(1, 1).is_err());
        assert!(JobAssignment::new(0, 0).is_err());
        assert_eq!(JobAssignment::single().world_size, 1);
    }
    #[test]
    fn static_selection() {
        let j = Job::<M>::new("j", (0..5).map(|i| Task::new(i.to_string(), P)).collect());
        let a = JobAssignment::new(1, 2).unwrap();
        assert_eq!(
            j.selected_tasks(a).map(|x| x.0).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }
}
