//! Dynamic (lease-based) task scheduling across two concurrent ranks.
//!
//! Run with:
//! ```sh
//! cargo run --example dynamic
//! ```
//!
//! Unlike static scheduling, dynamic scheduling does not assign tasks by `i % world_size`.
//! Each rank claims the next unfinished task via filesystem leases, which is robust for
//! heterogeneous or fault-prone workers. This example runs both ranks as threads sharing a
//! checkpoint directory, then merges their results.

use carlo_mc::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, fs, path::PathBuf, sync::Arc, thread};

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

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let directory = PathBuf::from("carlo-mc-example-dynamic");
    let _ = fs::remove_dir_all(&directory);

    let job = Arc::new(Job::<Pi>::new(
        "pi",
        vec![
            Task::new("a", ()).seed(1).sweeps(2_000).binsize(20),
            Task::new("b", ()).seed(2).sweeps(2_000).binsize(20),
        ],
    ));
    let runner = Arc::new(Runner::new().dynamic());

    // Spawn two cooperating ranks. Each needs the same checkpoint directory so leases are
    // visible to both.
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let job = Arc::clone(&job);
            let runner = Arc::clone(&runner);
            let directory = directory.clone();
            thread::spawn(
                move || -> Result<RunResult<()>, Box<dyn std::error::Error + Send + Sync>> {
                    runner
                        .run(
                            &job,
                            &RunOptions {
                                assignment: Some(JobAssignment::new(rank, 2)?),
                                checkpoint_dir: Some(directory),
                                ..RunOptions::default()
                            },
                        )
                        .map_err(Into::into)
                },
            )
        })
        .collect();

    let mut parts = Vec::new();
    for handle in handles {
        let run = handle.join().expect("rank thread panicked")?;
        for task in &run.result.tasks {
            println!("rank {} ran task {}", run.result.rank, task.task.name);
        }
        parts.push(run.result);
    }

    let merged = merge_dynamic_results(&job, parts)?;
    for task in &merged.tasks {
        println!(
            "{:>4}: pi = {:.6} +/- {:.6}",
            task.task.name, task.observables["pi"].mean, task.observables["pi"].stderr
        );
    }

    let _ = fs::remove_dir_all(&directory);
    Ok(())
}
