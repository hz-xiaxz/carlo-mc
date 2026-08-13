//! Demonstrate checkpointing and exact resume.
//!
//! Run with:
//! ```sh
//! cargo run --example checkpoint_resume
//! ```
//!
//! The first run is interrupted part-way through (via `sweep_limit`). The second run resumes
//! from the checkpoint directory. A third, uninterrupted run proves that resuming produces
//! bit-for-bit identical results to running the whole task in one shot.

use carlo_mc::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, fs, path::Path};

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
    let directory = Path::new("carlo-mc-example-checkpoint");
    let _ = fs::remove_dir_all(directory);

    let job = Job::<Pi>::new(
        "pi",
        vec![Task::new("sample", ()).seed(7).sweeps(500_000).binsize(500)],
    );

    // First chunk: stop after 200_000 sweeps and write a checkpoint every 100_000.
    let first = Runner::new().run(
        &job,
        &RunOptions {
            checkpoint_dir: Some(directory.to_path_buf()),
            checkpoint_interval: 100_000,
            sweep_limit: Some(200_000),
            ..RunOptions::default()
        },
    )?;
    println!("first run stopped early: {}", first.stopped_early);

    // Resume: restore model state, RNG position, and accumulated observables.
    let resumed = Runner::new().run(
        &job,
        &RunOptions {
            checkpoint_dir: Some(directory.to_path_buf()),
            resume: true,
            ..RunOptions::default()
        },
    )?;
    println!("resumed run stopped early: {}", resumed.stopped_early);

    // Reference: run the entire task without interruption.
    let fresh = Runner::new().run(&job, &RunOptions::default())?;

    // Determinism means the resumed and fresh results must match exactly.
    println!(
        "resume reproduces fresh run exactly: {}",
        resumed.result == fresh.result
    );
    println!(
        "pi = {:.6} +/- {:.6}",
        fresh.result.tasks[0].observables["pi"].mean,
        fresh.result.tasks[0].observables["pi"].stderr
    );

    let _ = fs::remove_dir_all(directory);
    Ok(())
}
