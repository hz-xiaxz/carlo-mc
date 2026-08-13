//! Estimate π with a deterministic Monte Carlo job.
//!
//! Run with:
//! ```sh
//! cargo run --example pi
//! ```
//!
//! This is the smallest complete `carlo-mc` program: define a model, build a job, run it,
//! and read back the binned estimates.

use carlo_mc::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, path::Path};

/// The simulation state. It must be serializable so it can be checkpointed.
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
        // The estimate converges to π as `sweeps` grows.
        ctx.measure("pi", 4.0 * self.inside as f64 / self.total as f64)
            .expect("finite value");
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One task per independent parameter point. Different seeds give different streams.
    let tasks = (0..4)
        .map(|i| {
            Task::new(format!("estimate-{i}"), ())
                .seed(i)
                .thermalization(1_000)
                .sweeps(200_000)
                .binsize(200)
        })
        .collect();

    let job = Job::<Pi>::new("pi", tasks);

    // Static scheduling with a single rank runs every task in order.
    let run = Runner::new().run(&job, &RunOptions::default())?;

    for task in &run.result.tasks {
        let estimate = &task.observables["pi"];
        println!(
            "{:>12}: pi = {:.6} +/- {:.6}",
            task.task.name, estimate.mean, estimate.stderr
        );
    }

    // Results can be persisted as JSON or HDF5.
    run.result.write_json(Path::new("pi.json"))?;
    println!("wrote pi.json");
    Ok(())
}
