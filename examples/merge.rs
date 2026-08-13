//! Run a job across two ranks and merge the results.
//!
//! Run with:
//! ```sh
//! cargo run --example merge
//! ```
//!
//! In real MPI/SLURM deployments each process calls `Runner::run` with its own
//! `JobAssignment` (see `JobAssignment::from_env`). Here both ranks run in one process and
//! their results are merged into a single `JobResult`.

use carlo_mc::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, path::Path};

#[derive(Serialize, Deserialize)]
struct Pi {
    inside: u64,
    total: u64,
}

impl MonteCarlo for Pi {
    type Parameters = ();
    type Error = Infallible;
    type Estimate = carlo_mc::Estimate;

    fn new(_: &()) -> Result<Self, Self::Error> {
        Ok(Pi {
            inside: 0,
            total: 0,
        })
    }

    fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
        Ok(())
    }

    fn sweep(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        let x: f64 = ctx.rng.random();
        let y: f64 = ctx.rng.random();
        self.total += 1;
        if x * x + y * y <= 1.0 {
            self.inside += 1;
        }
        Ok(())
    }

    fn measure(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        ctx.measure("pi", 4.0 * self.inside as f64 / self.total as f64)
            .expect("finite value");
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let job = Job::<Pi>::new(
        "pi",
        vec![
            Task::new("a", ()).seed(1).sweeps(200_000).binsize(200),
            Task::new("b", ()).seed(2).sweeps(200_000).binsize(200),
        ],
    );

    // Two ranks, static scheduling: rank 0 owns task 0, rank 1 owns task 1.
    let rank0 = Runner::new().run(
        &job,
        &RunOptions {
            assignment: Some(JobAssignment::new(0, 2)?),
            ..RunOptions::default()
        },
    )?;
    let rank1 = Runner::new().run(
        &job,
        &RunOptions {
            assignment: Some(JobAssignment::new(1, 2)?),
            ..RunOptions::default()
        },
    )?;

    let parts = vec![rank0.result, rank1.result];
    let merged = merge_results(&job, parts.clone())?;
    for task in &merged.tasks {
        println!(
            "{:>4}: pi = {:.6} +/- {:.6}",
            task.task.name, task.observables["pi"].mean, task.observables["pi"].stderr
        );
    }

    merge_results_to_json(&job, parts.clone(), Path::new("merged.json"))?;
    merge_results_to_hdf5(&job, parts, Path::new("merged.h5"))?;
    println!("wrote merged.json and merged.h5");
    Ok(())
}
