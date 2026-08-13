//! # carlo-mc
//!
//! A model-independent framework for **deterministic, resumable Monte Carlo jobs**.
//!
//! `carlo-mc` separates the physical model (the [`MonteCarlo`] implementation) from the
//! infrastructure that runs it: sweep loops, thermalization, binning, checkpointing, rank
//! scheduling, and result merging. A job is a collection of [`Task`]s, where each task is an
//! independent parameter point together with execution settings.
//!
//! ## Quick start
//!
//! ```no_run
//! use carlo_mc::prelude::*;
//! use rand::RngExt;
//! use serde::{Deserialize, Serialize};
//! use std::convert::Infallible;
//!
//! // 1. Define the model state. It must be serializable so it can be checkpointed.
//! #[derive(Serialize, Deserialize)]
//! struct Pi {
//!     inside: u64,
//!     total: u64,
//! }
//!
//! #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
//! struct Params;
//!
//! impl MonteCarlo for Pi {
//!     type Parameters = Params;
//!     type Error = Infallible;
//!     type Estimate = carlo_mc::Estimate;
//!
//!     fn new(_: &Params) -> Result<Self, Self::Error> {
//!         Ok(Pi { inside: 0, total: 0 })
//!     }
//!     fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
//!         Ok(())
//!     }
//!     fn sweep(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
//!         let x: f64 = ctx.rng.random();
//!         let y: f64 = ctx.rng.random();
//!         self.total += 1;
//!         if x * x + y * y <= 1.0 {
//!             self.inside += 1;
//!         }
//!         Ok(())
//!     }
//!     fn measure(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
//!         ctx.measure("pi", 4.0 * self.inside as f64 / self.total as f64)
//!             .expect("measure");
//!         Ok(())
//!     }
//! }
//!
//! // 2. Build a job and run it.
//! let job = Job::<Pi>::new(
//!     "pi",
//!     vec![Task::new("sample", Params).seed(42).sweeps(100_000).binsize(100)],
//! );
//! let run = Runner::new().run(&job, &RunOptions::default())?;
//! println!("{:#?}", run.result);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Concepts
//!
//! - [`Job`] bundles a name and a list of [`Task`]s.
//! - [`Task`] holds parameters plus `sweeps`, `thermalization`, `binsize`, and `seed`.
//! - [`MonteCarlo`] is the trait you implement for a physical model.
//! - [`Context`] owns the deterministic RNG and observable accumulators.
//! - [`Runner`] executes tasks, writes HDF5 checkpoints, and can resume them.
//! - [`JobResult`] can be serialized (JSON or HDF5) and merged across ranks.
//! - [`Params`] is a generic TOML config table, consumed by
//!   [`MonteCarlo::build_tasks`] to expand a model's parameter grid.
//!
//! ## Scheduling
//!
//! Static scheduling (`Runner::default`) assigns task `i` to rank `i % world_size`.
//! Dynamic scheduling ([`Runner::dynamic`]) uses file-based leases so that any rank can claim
//! the next unfinished task, which is useful for heterogeneous workers.
//!
//! See the `examples/` directory for complete, runnable programs.

mod job;

pub use job::*;

/// Commonly used items, intended for `use carlo_mc::prelude::*;`.
pub mod prelude {
    pub use crate::{
        checkpoint_path, dump_path, measurement_path, merge_dynamic_results,
        merge_dynamic_results_to_hdf5, merge_dynamic_results_to_json, merge_results,
        merge_results_to_hdf5, merge_results_to_json, merge_results_with_scheduling, result_path,
        task_path, BinnedEstimate, CheckpointState, CompactAccumulator, Context, DeterministicRng,
        Estimate, Evaluator, GenericJobError, Job, JobAssignment, MonteCarlo, Params, ParamsError,
        RestoredState, ResultEstimate, RunOptions, RunResult, Runner, RunnerDurationConfig,
        ScalarEstimate, Scheduling, Task, TaskMaker, TaskResult,
    };
}
