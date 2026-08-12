# carlo-mc

A model-independent Rust framework for **deterministic, resumable Monte Carlo jobs**.

`carlo-mc` separates the *physical model* (the part you write) from the *infrastructure* that
runs it: sweep loops, thermalization, binning, checkpointing, rank scheduling, and result
merging. You implement one trait, then get reproducible runs, HDF5 checkpoints, and strict
merging across ranks for free.

## Features

- **Model-independent** — implement `MonteCarlo` for any serializable simulation state.
- **Deterministic** — each task has its own seekable RNG; the same seed always gives the
  same stream, and the RNG position is saved in checkpoints.
- **Resumable** — interrupted runs restore model state, RNG position, and accumulated
  observables exactly, continuing without losing samples.
- **Checkpointed** — HDF5 checkpoints with a strict, versioned schema and atomic writes.
- **Multi-rank** — static (`i % world_size`) or dynamic (filesystem-lease) task scheduling.
- **Statistically sound** — internal binning plus automatic rebinning to a target number of
  bins for error estimation.
- **Strict result merging** — validates ranks, tasks, and counts before merging to JSON/HDF5.

## Quick start

Add the dependency:

```toml
[dependencies]
carlo-mc = "0.1"
rand = "0.10"
serde = { version = "1", features = ["derive"] }
```

Estimate π:

```rust
use carlo_mc::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

#[derive(Serialize, Deserialize)]
struct Pi { inside: u64, total: u64 }

impl MonteCarlo for Pi {
    type Parameters = ();
    type Error = Infallible;

    fn new(_: &()) -> Result<Self, Self::Error> {
        Ok(Pi { inside: 0, total: 0 })
    }
    fn init(&mut self, _: &mut Context) -> Result<(), Self::Error> {
        Ok(())
    }
    fn sweep(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        let x: f64 = ctx.rng.random();
        let y: f64 = ctx.rng.random();
        self.total += 1;
        if x * x + y * y <= 1.0 { self.inside += 1; }
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
        vec![Task::new("sample", ()).seed(42).sweeps(100_000).binsize(100)],
    );
    let run = Runner::new().run(&job, &RunOptions::default())?;
    let estimate = &run.result.tasks[0].observables["pi"];
    println!("pi = {:.6} +/- {:.6}", estimate.mean, estimate.stderr);
    Ok(())
}
```

> Note: in a model whose `Error` type is not `Infallible`, `ctx.measure(...)?` works
> directly; with `Infallible` use `.expect(...)` instead.

## Concepts

| Type | Purpose |
| --- | --- |
| `MonteCarlo` | The trait you implement for your simulation state. |
| `Task` | One parameter point plus `sweeps`, `thermalization`, `binsize`, and `seed`. |
| `TaskMaker` | Fluent builder for many tasks from a shared parameter template. |
| `Job` | A named collection of `Task`s. |
| `Context` | Owns the deterministic RNG (`ctx.rng`) and observable accumulators. |
| `Runner` | Executes tasks, writes checkpoints, and resumes them. |
| `RunOptions` | Per-run configuration (checkpoint dir, resume, sweep limit, deadline, …). |
| `JobResult` | A rank's results; serialize to JSON/HDF5 and merge across ranks. |

All the commonly used items are re-exported through `carlo_mc::prelude`:

```rust
use carlo_mc::prelude::*;
```

## The `MonteCarlo` lifecycle

1. `MonteCarlo::new(&parameters)` — construct the model for a task.
2. `MonteCarlo::init(&mut ctx)` — one-time setup; may draw from `ctx.rng`.
3. `MonteCarlo::sweep(&mut ctx)` — advance the simulation by one step.
4. `MonteCarlo::measure(&mut ctx)` — record observables with `ctx.measure`.

`Parameters` must be `Clone + PartialEq + Serialize + DeserializeOwned`; the model state must
be `Serialize + DeserializeOwned` so it can be checkpointed. `Error` must implement
`std::error::Error`.

## Scheduling

- **Static** (default) assigns task `i` to rank `i % world_size`.
- **Dynamic** (`Runner::dynamic()`) lets any rank claim any unfinished task through
  filesystem leases, useful for heterogeneous or fault-prone workers. Dynamic scheduling
  requires `RunOptions::checkpoint_dir`.

Detect your rank from the environment with `JobAssignment::from_env()`, which understands
common SLURM/MPI variables (`SLURM_PROCID`, `OMPI_COMM_WORLD_RANK`, …).

## Checkpointing & resume

```rust
let run = Runner::new().run(
    &job,
    &RunOptions {
        checkpoint_dir: Some("ckpt".into()),
        checkpoint_interval: 100_000,   // every N sweeps
        sweep_limit: Some(200_000),     // stop early, e.g. wall-clock slicing
        ..RunOptions::default()
    },
)?;

// Later, finish the job exactly:
let resumed = Runner::new().run(
    &job,
    &RunOptions {
        checkpoint_dir: Some("ckpt".into()),
        resume: true,
        ..RunOptions::default()
    },
)?;
```

Resume restores the model, the RNG position, and the accumulated observables, so the resumed
result matches an uninterrupted run bit-for-bit.

## Merging results

Each rank produces a `JobResult`. Merge them with:

- `merge_results(&job, parts)` — static scheduling.
- `merge_dynamic_results(&job, parts)` — dynamic scheduling.
- `merge_results_to_json` / `merge_results_to_hdf5` — merge and write.
- `merge_dynamic_results_to_json` / `merge_dynamic_results_to_hdf5` — merge and write.

Merging is strict: every rank must be present exactly once, and every task must be complete
and consistent with the shared `Job` definition.

## Examples

Run them with `cargo run --example <name>`:

| Example | What it shows |
| --- | --- |
| `pi` | Minimal end-to-end model, run, and result output. |
| `checkpoint_resume` | Interrupting a run and resuming it exactly. |
| `merge` | Two static-scheduling ranks merged into one result. |
| `dynamic` | Two concurrent dynamic-scheduling ranks sharing leases. |

## Paths

`carlo-mc` writes files under a user-supplied root:

- `task0001/run0001.dump.h5` — a task checkpoint (via `dump_path`).
- `task0001/run0001.meas.h5` — measurement path (via `measurement_path`).
- `result` — canonical HDF5 result path (via `result_path`).

Path components are validated against traversal, symlink escapes, and reserved names; writes
are atomic and durable.

## License

MIT
