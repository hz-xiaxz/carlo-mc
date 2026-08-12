use super::{
    checkpoint::{
        add_bytes, add_f64, add_u64, add_u8, add_utf8, exact_group, exact_root_group, finish_hdf5,
        group, numbered_groups, open_hdf5, read_f64_scalar, read_json, read_u64_scalar,
        read_u8_scalar, read_usize, read_utf8, schema, set_root_attrs, to_u64,
    },
    paths::{atomic_write, ensure_safe_read_file_path},
    GenericJobError, Job, MonteCarlo, Scheduling, Task,
};
use hdf5_pure::FileBuilder;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
const KIND: &str = "carlo-mc-result";
/// A binned statistical estimate for one observable.
///
/// `mean` is the average of rebinned internal bins and `stderr` is the rebinned
/// standard error. When only one rebinned bin exists, `stderr` is `NaN`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarEstimate {
    pub mean: f64,
    pub stderr: f64,
    /// Number of raw internal bins before rebinning.
    pub internal_bins: usize,
    /// Number of internal bins combined into one rebinned bin.
    pub rebin_length: usize,
    /// Number of rebinned bins used for the estimate.
    pub rebin_count: usize,
    /// Raw samples per internal bin.
    pub bin_length: usize,
}
/// The measured result of a single [`Task`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskResult<P> {
    pub task_index: usize,
    pub task: Task<P>,
    pub observables: BTreeMap<String, ScalarEstimate>,
    pub thermalization_sweeps: usize,
    pub measurement_sweeps: usize,
    /// Whether the task ran to completion (full thermalization and measurement).
    pub completed: bool,
}
/// One rank's contribution to a job, serializable to JSON or HDF5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobResult<P> {
    pub job_name: String,
    pub rank: usize,
    pub world_size: usize,
    pub tasks: Vec<TaskResult<P>>,
}
impl<P: Serialize + DeserializeOwned> JobResult<P> {
    /// Atomically writes this result as pretty-printed JSON.
    pub fn write_json(&self, path: &Path) -> Result<(), GenericJobError> {
        validate_result_schema(path, self)?;
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| GenericJobError::json(Some(path), e))?;
        atomic_write(path, &bytes)
    }

    /// Reads and validates a JSON result written by [`JobResult::write_json`].
    pub fn read_json(path: &Path) -> Result<Self, GenericJobError> {
        ensure_safe_read_file_path(path)?;
        let result: Self = serde_json::from_slice(
            &fs::read(path).map_err(|e| GenericJobError::io("read JSON", path, e))?,
        )
        .map_err(|e| GenericJobError::json(Some(path), e))?;
        validate_result_schema(path, &result)?;
        Ok(result)
    }

    /// Atomically writes this result as HDF5.
    pub fn write_hdf5(&self, path: &Path) -> Result<(), GenericJobError> {
        validate_result_schema(path, self)?;
        if self.tasks.len() > 10_000 {
            return Err(schema(path, "too many result tasks for four-digit names"));
        }
        let mut builder = FileBuilder::new();
        set_root_attrs(&mut builder, KIND);

        let mut metadata = builder.create_group("metadata");
        add_u64(&mut metadata, "schema", 1);
        add_utf8(path, &mut metadata, "job", &self.job_name)?;
        builder.add_group(metadata.finish());

        let mut assignment = builder.create_group("assignment");
        add_u64(&mut assignment, "rank", to_u64(path, self.rank)?);
        add_u64(
            &mut assignment,
            "world_size",
            to_u64(path, self.world_size)?,
        );
        builder.add_group(assignment.finish());

        let mut tasks = builder.create_group("tasks");
        for (task_number, result) in self.tasks.iter().enumerate() {
            if result.observables.len() > 10_000 {
                return Err(schema(
                    path,
                    "too many result observables for four-digit names",
                ));
            }
            let mut result_group = tasks.create_group(&format!("task{task_number:04}"));
            let task = serde_json::to_vec(&result.task)
                .map_err(|error| GenericJobError::json(Some(path), error))?;
            add_bytes(path, &mut result_group, "task", &task)?;
            add_u64(&mut result_group, "index", to_u64(path, result.task_index)?);
            add_u8(&mut result_group, "completed", u8::from(result.completed));

            let mut progress = result_group.create_group("progress");
            add_u64(
                &mut progress,
                "thermalization_sweeps",
                to_u64(path, result.thermalization_sweeps)?,
            );
            add_u64(
                &mut progress,
                "measurement_sweeps",
                to_u64(path, result.measurement_sweeps)?,
            );
            result_group.add_group(progress.finish());

            let mut observables = result_group.create_group("observables");
            for (index, (name, estimate)) in result.observables.iter().enumerate() {
                let mut observable = observables.create_group(&format!("observable{index:04}"));
                add_utf8(path, &mut observable, "name", name)?;
                add_f64(&mut observable, "mean", estimate.mean);
                add_f64(&mut observable, "stderr", estimate.stderr);
                add_u64(
                    &mut observable,
                    "internal_bins",
                    to_u64(path, estimate.internal_bins)?,
                );
                add_u64(
                    &mut observable,
                    "rebin_length",
                    to_u64(path, estimate.rebin_length)?,
                );
                add_u64(
                    &mut observable,
                    "rebin_count",
                    to_u64(path, estimate.rebin_count)?,
                );
                add_u64(
                    &mut observable,
                    "bin_length",
                    to_u64(path, estimate.bin_length)?,
                );
                observables.add_group(observable.finish());
            }
            result_group.add_group(observables.finish());
            tasks.add_group(result_group.finish());
        }
        builder.add_group(tasks.finish());
        finish_hdf5(path, builder)
    }

    /// Reads and validates an HDF5 result written by [`JobResult::write_hdf5`].
    pub fn read_hdf5(path: &Path) -> Result<Self, GenericJobError> {
        let file = open_hdf5(path, KIND)?;
        exact_root_group(path, file.root(), &[], &["assignment", "metadata", "tasks"])?;
        exact_group(
            path,
            group(path, &file, "metadata")?,
            &["job", "schema"],
            &[],
        )?;
        exact_group(
            path,
            group(path, &file, "assignment")?,
            &["rank", "world_size"],
            &[],
        )?;
        if read_u64_scalar(path, &file, "metadata/schema")? != 1 {
            return Err(schema(path, "unsupported result schema"));
        }
        let tasks_group = group(path, &file, "tasks")?;
        let task_groups = tasks_group
            .groups()
            .map_err(|error| GenericJobError::Hdf5 {
                path: path.into(),
                reason: error.to_string(),
            })?;
        exact_group(
            path,
            tasks_group,
            &[],
            &task_groups.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;

        let mut tasks = Vec::new();
        for (task_number, group_name) in task_groups.into_iter().enumerate() {
            if parse_indexed_group(path, &group_name, "task")? != task_number {
                return Err(schema(path, "invalid four-digit result task sequence"));
            }
            let base = format!("tasks/{group_name}");
            exact_group(
                path,
                group(path, &file, &base)?,
                &["completed", "index", "task"],
                &["observables", "progress"],
            )?;
            let task_index = read_usize(path, &file, &format!("{base}/index"))?;
            let task: Task<P> = read_json(path, &file, &format!("{base}/task"))?;
            exact_group(
                path,
                group(path, &file, &format!("{base}/progress"))?,
                &["measurement_sweeps", "thermalization_sweeps"],
                &[],
            )?;

            let observables_base = format!("{base}/observables");
            let observables_group = group(path, &file, &observables_base)?;
            let observable_groups = numbered_groups(path, &observables_group, "observable")?;
            exact_group(
                path,
                observables_group,
                &[],
                &observable_groups
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )?;
            let mut observables = BTreeMap::new();
            for observable_group in observable_groups {
                let observable_base = format!("{observables_base}/{observable_group}");
                exact_group(
                    path,
                    group(path, &file, &observable_base)?,
                    &[
                        "bin_length",
                        "internal_bins",
                        "mean",
                        "name",
                        "rebin_count",
                        "rebin_length",
                        "stderr",
                    ],
                    &[],
                )?;
                let name = read_utf8(path, &file, &format!("{observable_base}/name"))?;
                let estimate = ScalarEstimate {
                    mean: read_f64_scalar(path, &file, &format!("{observable_base}/mean"))?,
                    stderr: read_f64_scalar(path, &file, &format!("{observable_base}/stderr"))?,
                    internal_bins: read_usize(
                        path,
                        &file,
                        &format!("{observable_base}/internal_bins"),
                    )?,
                    rebin_length: read_usize(
                        path,
                        &file,
                        &format!("{observable_base}/rebin_length"),
                    )?,
                    rebin_count: read_usize(
                        path,
                        &file,
                        &format!("{observable_base}/rebin_count"),
                    )?,
                    bin_length: read_usize(path, &file, &format!("{observable_base}/bin_length"))?,
                };
                if observables.insert(name, estimate).is_some() {
                    return Err(schema(path, "duplicate result observable name"));
                }
            }
            let completed = match read_u8_scalar(path, &file, &format!("{base}/completed"))? {
                0 => false,
                1 => true,
                _ => return Err(schema(path, "invalid result completion flag")),
            };
            tasks.push(TaskResult {
                task_index,
                task,
                observables,
                thermalization_sweeps: read_usize(
                    path,
                    &file,
                    &format!("{base}/progress/thermalization_sweeps"),
                )?,
                measurement_sweeps: read_usize(
                    path,
                    &file,
                    &format!("{base}/progress/measurement_sweeps"),
                )?,
                completed,
            });
        }
        let result = Self {
            job_name: read_utf8(path, &file, "metadata/job")?,
            rank: read_usize(path, &file, "assignment/rank")?,
            world_size: read_usize(path, &file, "assignment/world_size")?,
            tasks,
        };
        validate_result_schema(path, &result)?;
        Ok(result)
    }
}

pub(crate) fn rebin_length(internal_bins: usize) -> usize {
    let target_count = if internal_bins <= 10 {
        internal_bins
    } else {
        10 + ((internal_bins - 10) as f64).cbrt().round() as usize
    };
    (internal_bins / target_count.max(1)).max(1)
}

fn parse_indexed_group(path: &Path, name: &str, prefix: &str) -> Result<usize, GenericJobError> {
    let digits = name
        .strip_prefix(prefix)
        .filter(|digits| digits.len() == 4 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| schema(path, "invalid indexed result group name"))?;
    digits
        .parse::<usize>()
        .map_err(|_| schema(path, "invalid four-digit result group index"))
}

fn valid_estimate(estimate: &ScalarEstimate) -> bool {
    if !estimate.mean.is_finite()
        || estimate.internal_bins == 0
        || estimate.rebin_length == 0
        || estimate.rebin_count == 0
        || estimate.bin_length == 0
        || estimate.rebin_length != rebin_length(estimate.internal_bins)
        || estimate.rebin_count != estimate.internal_bins / estimate.rebin_length
    {
        return false;
    }
    if estimate.rebin_count == 1 {
        estimate.stderr.is_nan()
    } else {
        estimate.stderr.is_finite() && estimate.stderr >= 0.0
    }
}

fn validate_result_schema<P>(path: &Path, result: &JobResult<P>) -> Result<(), GenericJobError> {
    if result.world_size == 0 || result.rank >= result.world_size {
        return Err(GenericJobError::Schema {
            path: path.into(),
            reason: "invalid result assignment",
        });
    }
    for task in &result.tasks {
        if task.completed
            != (task.thermalization_sweeps == task.task.thermalization
                && task.measurement_sweeps == task.task.sweeps)
            || task.thermalization_sweeps > task.task.thermalization
            || task.measurement_sweeps > task.task.sweeps
            || task
                .observables
                .values()
                .any(|estimate| !valid_estimate(estimate))
        {
            return Err(GenericJobError::Schema {
                path: path.into(),
                reason: "invalid task result",
            });
        }
    }
    Ok(())
}
/// Merges per-rank results from static scheduling into a single result.
pub fn merge_results<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_with_scheduling(job, parts, Scheduling::Static)
}

/// Merges per-rank results from dynamic scheduling into a single result.
pub fn merge_dynamic_results<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_with_scheduling(job, parts, Scheduling::Dynamic)
}

/// Merges static-scheduling results and atomically writes the output as JSON.
pub fn merge_results_to_json<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    output: &Path,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_to(
        job,
        parts,
        Scheduling::Static,
        output,
        JobResult::write_json,
    )
}

/// Merges static-scheduling results and atomically writes the output as HDF5.
pub fn merge_results_to_hdf5<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    output: &Path,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_to(
        job,
        parts,
        Scheduling::Static,
        output,
        JobResult::write_hdf5,
    )
}

/// Merges dynamic-scheduling results and atomically writes the output as JSON.
pub fn merge_dynamic_results_to_json<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    output: &Path,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_to(
        job,
        parts,
        Scheduling::Dynamic,
        output,
        JobResult::write_json,
    )
}

/// Merges dynamic-scheduling results and atomically writes the output as HDF5.
pub fn merge_dynamic_results_to_hdf5<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    output: &Path,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    merge_results_to(
        job,
        parts,
        Scheduling::Dynamic,
        output,
        JobResult::write_hdf5,
    )
}

fn merge_results_to<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    scheduling: Scheduling,
    output: &Path,
    write: fn(&JobResult<M::Parameters>, &Path) -> Result<(), GenericJobError>,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    let result = merge_results_with_scheduling(job, parts, scheduling)?;
    write(&result, output)?;
    Ok(result)
}

/// Merges per-rank results under the given scheduling strategy.
///
/// Validates every part strictly: ranks must be unique and complete, and each task must
/// be completed, correctly owned, and consistent with the shared `job` definition.
pub fn merge_results_with_scheduling<M: MonteCarlo>(
    job: &Job<M>,
    parts: Vec<JobResult<M::Parameters>>,
    scheduling: Scheduling,
) -> Result<JobResult<M::Parameters>, GenericJobError> {
    if parts.is_empty() {
        return Err(GenericJobError::Merge("no rank results"));
    }
    let world = parts[0].world_size;
    if world == 0 || parts.len() != world {
        return Err(GenericJobError::Merge("rank set incomplete"));
    }
    let mut ranks = BTreeSet::new();
    let mut tasks = BTreeMap::new();
    for p in parts {
        validate_result_schema(Path::new("<in-memory-result>"), &p)?;
        if p.job_name != job.name
            || p.world_size != world
            || p.rank >= world
            || !ranks.insert(p.rank)
        {
            return Err(GenericJobError::Merge("job/world/rank mismatch"));
        }
        for t in p.tasks {
            if !t.completed
                || t.task_index >= job.tasks.len()
                || (scheduling == Scheduling::Static && t.task_index % world != p.rank)
                || t.task != job.tasks[t.task_index]
                || t.thermalization_sweeps != t.task.thermalization
                || t.measurement_sweeps != t.task.sweeps
                || tasks.insert(t.task_index, t).is_some()
            {
                return Err(GenericJobError::Merge(
                    "task mismatch, wrong rank, wrong counts, duplicate, or incomplete",
                ));
            }
        }
    }
    if ranks.len() != world
        || (0..world).any(|r| !ranks.contains(&r))
        || tasks.len() != job.tasks.len()
        || (0..job.tasks.len()).any(|i| !tasks.contains_key(&i))
    {
        return Err(GenericJobError::Merge("rank or task set incomplete"));
    }
    Ok(JobResult {
        job_name: job.name.clone(),
        rank: 0,
        world_size: world,
        tasks: tasks.into_values().collect(),
    })
}
#[cfg(test)]
mod tests {
    use super::super::{paths::temp_dir, Context, Job};
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::convert::Infallible;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct P(u8);
    #[derive(Serialize, Deserialize)]
    struct M;
    impl MonteCarlo for M {
        type Parameters = P;
        type Error = Infallible;
        fn new(_: &P) -> Result<Self, Self::Error> {
            Ok(Self)
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
    fn task_result(index: usize, task: Task<P>) -> TaskResult<P> {
        let thermalization_sweeps = task.thermalization;
        let measurement_sweeps = task.sweeps;
        TaskResult {
            task_index: index,
            task,
            observables: BTreeMap::new(),
            thermalization_sweeps,
            measurement_sweeps,
            completed: true,
        }
    }

    fn estimate() -> ScalarEstimate {
        ScalarEstimate {
            mean: 1.0,
            stderr: 0.25,
            internal_bins: 2,
            rebin_length: 1,
            rebin_count: 2,
            bin_length: 1,
        }
    }

    fn static_parts(job: &Job<M>) -> (JobResult<P>, JobResult<P>) {
        (
            JobResult {
                job_name: job.name.clone(),
                rank: 0,
                world_size: 2,
                tasks: vec![task_result(0, job.tasks[0].clone())],
            },
            JobResult {
                job_name: job.name.clone(),
                rank: 1,
                world_size: 2,
                tasks: vec![task_result(1, job.tasks[1].clone())],
            },
        )
    }
    #[test]
    fn roundtrips() {
        let d = temp_dir("results");
        let r = JobResult::<()> {
            job_name: "j".into(),
            rank: 0,
            world_size: 1,
            tasks: vec![],
        };
        let j = d.join("r.json");
        let h = d.join("r.h5");
        r.write_json(&j).unwrap();
        r.write_hdf5(&h).unwrap();
        assert_eq!(JobResult::read_json(&j).unwrap(), r);
        assert_eq!(JobResult::read_hdf5(&h).unwrap(), r);
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn result_hdf5_has_strict_structure_and_roundtrips() {
        let directory = temp_dir("results-hdf5-schema");
        let path = directory.join("result.h5");
        let result = JobResult {
            job_name: "job".into(),
            rank: 0,
            world_size: 1,
            tasks: vec![TaskResult {
                task_index: 7,
                task: Task::new("task", P(3))
                    .thermalization(2)
                    .sweeps(4)
                    .binsize(2),
                observables: BTreeMap::from([("energy/value".into(), estimate())]),
                thermalization_sweeps: 2,
                measurement_sweeps: 4,
                completed: true,
            }],
        };
        result.write_hdf5(&path).unwrap();

        let file = hdf5_pure::File::open(&path).unwrap();
        assert_eq!(
            file.root().attrs().unwrap(),
            std::collections::HashMap::from([
                (
                    "carlo_kind".into(),
                    hdf5_pure::AttrValue::String(KIND.into()),
                ),
                ("schema_version".into(), hdf5_pure::AttrValue::U64(1),),
            ])
        );
        assert_eq!(
            file.root()
                .groups()
                .unwrap()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            ["assignment", "metadata", "tasks"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
        assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
        assert_eq!(
            file.group("metadata").unwrap().datasets().unwrap(),
            ["schema", "job"]
        );
        assert_eq!(
            file.group("assignment").unwrap().datasets().unwrap(),
            ["rank", "world_size"]
        );
        assert!(!file
            .as_bytes()
            .windows(b"payload_json".len())
            .any(|window| window == b"payload_json"));
        assert_eq!(
            file.dataset("assignment/rank").unwrap().dtype().unwrap(),
            hdf5_pure::DType::U64
        );
        assert!(file
            .dataset("assignment/rank")
            .unwrap()
            .shape()
            .unwrap()
            .is_empty());
        assert_eq!(
            file.group("tasks/task0000").unwrap().groups().unwrap(),
            ["progress", "observables"]
        );
        assert_eq!(
            file.group("tasks/task0000").unwrap().datasets().unwrap(),
            ["task", "index", "completed"]
        );
        assert_eq!(
            file.group("tasks/task0000/progress")
                .unwrap()
                .datasets()
                .unwrap(),
            ["thermalization_sweeps", "measurement_sweeps"]
        );
        assert_eq!(
            file.group("tasks/task0000/observables/observable0000")
                .unwrap()
                .datasets()
                .unwrap(),
            [
                "name",
                "mean",
                "stderr",
                "internal_bins",
                "rebin_length",
                "rebin_count",
                "bin_length"
            ]
        );
        assert_eq!(
            file.dataset("tasks/task0000/completed")
                .unwrap()
                .dtype()
                .unwrap(),
            hdf5_pure::DType::U8
        );
        assert_eq!(
            file.dataset("tasks/task0000/task")
                .unwrap()
                .dtype()
                .unwrap(),
            hdf5_pure::DType::U8
        );
        assert!(file
            .dataset("tasks/task0000/observables/observable0000/mean")
            .unwrap()
            .shape()
            .unwrap()
            .is_empty());
        assert_eq!(JobResult::read_hdf5(&path).unwrap(), result);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn merge_is_strict_and_orders_tasks() {
        let job = Job::<M>::new("j", vec![Task::new("a", P(1)), Task::new("b", P(2))]);
        let (rank_zero, rank_one) = static_parts(&job);
        let merged = merge_results(&job, vec![rank_one.clone(), rank_zero.clone()]).unwrap();
        assert_eq!(
            merged
                .tasks
                .iter()
                .map(|task| task.task_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut cases = Vec::new();
        let mut bad_name = rank_zero.clone();
        bad_name.job_name = "other".into();
        cases.push(vec![bad_name, rank_one.clone()]);
        cases.push(vec![rank_zero.clone(), rank_zero.clone()]);
        cases.push(vec![rank_zero.clone()]);
        let mut duplicate_task = rank_one.clone();
        duplicate_task.tasks[0] = task_result(0, job.tasks[0].clone());
        cases.push(vec![rank_zero.clone(), duplicate_task]);
        let mut wrong_definition = rank_one.clone();
        wrong_definition.tasks[0].task.parameters = P(9);
        cases.push(vec![rank_zero.clone(), wrong_definition]);
        let mut incomplete = rank_one.clone();
        incomplete.tasks[0].completed = false;
        cases.push(vec![rank_zero.clone(), incomplete]);
        let mut forged_counts = rank_one.clone();
        forged_counts.tasks[0].measurement_sweeps = 0;
        cases.push(vec![rank_zero.clone(), forged_counts]);
        let mut wrong_rank_task = rank_one;
        wrong_rank_task.tasks = vec![task_result(0, job.tasks[0].clone())];
        cases.push(vec![rank_zero, wrong_rank_task]);
        for parts in cases {
            assert!(merge_results(&job, parts).is_err());
        }
    }

    #[test]
    fn merge_rejects_malicious_in_memory_result_schema() {
        let job = Job::<M>::new("j", vec![Task::new("a", P(1)), Task::new("b", P(2))]);
        let (rank_zero, rank_one) = static_parts(&job);
        let mut cases = Vec::new();

        let mut invalid_assignment = rank_zero.clone();
        invalid_assignment.rank = invalid_assignment.world_size;
        cases.push(vec![invalid_assignment, rank_one.clone()]);

        let mut invalid_completion = rank_zero.clone();
        invalid_completion.tasks[0].completed = false;
        cases.push(vec![invalid_completion, rank_one.clone()]);

        let mut excessive_count = rank_zero.clone();
        excessive_count.tasks[0].measurement_sweeps += 1;
        cases.push(vec![excessive_count, rank_one.clone()]);

        for mutate in [
            |estimate: &mut ScalarEstimate| estimate.mean = f64::INFINITY,
            |estimate: &mut ScalarEstimate| estimate.stderr = f64::NAN,
            |estimate: &mut ScalarEstimate| estimate.stderr = -0.1,
            |estimate: &mut ScalarEstimate| estimate.internal_bins = 3,
            |estimate: &mut ScalarEstimate| estimate.rebin_length = 2,
            |estimate: &mut ScalarEstimate| estimate.rebin_count = 1,
            |estimate: &mut ScalarEstimate| estimate.bin_length = 0,
        ] {
            let mut malicious = rank_zero.clone();
            malicious.tasks[0]
                .observables
                .insert("value".into(), estimate());
            mutate(malicious.tasks[0].observables.get_mut("value").unwrap());
            cases.push(vec![malicious, rank_one.clone()]);
        }

        for parts in cases {
            assert!(matches!(
                merge_results(&job, parts),
                Err(GenericJobError::Schema { .. })
            ));
        }

        let mut one_bin = rank_zero;
        one_bin.tasks[0].observables.insert(
            "value".into(),
            ScalarEstimate {
                mean: 1.0,
                stderr: f64::NAN,
                internal_bins: 1,
                rebin_length: 1,
                rebin_count: 1,
                bin_length: 1,
            },
        );
        assert!(merge_results(&job, vec![one_bin, rank_one]).is_ok());
    }

    #[test]
    fn static_and_dynamic_merge_outputs_are_atomic_json_and_hdf5() {
        let directory = temp_dir("merge-output");
        let job = Job::<M>::new("j", vec![Task::new("a", P(1)), Task::new("b", P(2))]);
        let (rank_zero, rank_one) = static_parts(&job);
        let static_json = directory.join("static.json");
        let static_hdf5 = directory.join("static.h5");
        let dynamic_json = directory.join("dynamic.json");
        let dynamic_hdf5 = directory.join("dynamic.h5");
        for path in [&static_json, &static_hdf5, &dynamic_json, &dynamic_hdf5] {
            atomic_write(path, b"stale").unwrap();
        }

        let static_result = merge_results_to_json(
            &job,
            vec![rank_one.clone(), rank_zero.clone()],
            &static_json,
        )
        .unwrap();
        assert_eq!(JobResult::read_json(&static_json).unwrap(), static_result);
        let static_hdf5_result = merge_results_to_hdf5(
            &job,
            vec![rank_one.clone(), rank_zero.clone()],
            &static_hdf5,
        )
        .unwrap();
        assert_eq!(
            JobResult::read_hdf5(&static_hdf5).unwrap(),
            static_hdf5_result
        );

        let dynamic_result = merge_dynamic_results_to_json(
            &job,
            vec![rank_zero.clone(), rank_one.clone()],
            &dynamic_json,
        )
        .unwrap();
        assert_eq!(JobResult::read_json(&dynamic_json).unwrap(), dynamic_result);
        let dynamic_hdf5_result =
            merge_dynamic_results_to_hdf5(&job, vec![rank_zero, rank_one], &dynamic_hdf5).unwrap();
        assert_eq!(
            JobResult::read_hdf5(&dynamic_hdf5).unwrap(),
            dynamic_hdf5_result
        );
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 4);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn result_io_rejects_unsafe_and_symlink_paths() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("result-safe-path");
        let outside = temp_dir("result-safe-path-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let result = JobResult::<()> {
            job_name: "j".into(),
            rank: 0,
            world_size: 1,
            tasks: vec![],
        };

        for unsafe_path in [
            Path::new("../escape.json"),
            Path::new("/tmp/../escape.json"),
        ] {
            assert!(result.write_json(unsafe_path).is_err());
            assert!(result.write_hdf5(unsafe_path).is_err());
            assert!(JobResult::<()>::read_json(unsafe_path).is_err());
            assert!(JobResult::<()>::read_hdf5(unsafe_path).is_err());
        }

        let outside_json = outside.join("result.json");
        let outside_hdf5 = outside.join("result.h5");
        result.write_json(&outside_json).unwrap();
        result.write_hdf5(&outside_hdf5).unwrap();
        symlink(&outside_json, root.join("result.json")).unwrap();
        symlink(&outside_hdf5, root.join("result.h5")).unwrap();
        assert!(JobResult::<()>::read_json(&root.join("result.json")).is_err());
        assert!(JobResult::<()>::read_hdf5(&root.join("result.h5")).is_err());
        assert!(result.write_json(&root.join("result.json")).is_err());
        assert!(result.write_hdf5(&root.join("result.h5")).is_err());

        symlink(&outside, root.join("escape")).unwrap();
        assert!(JobResult::<()>::read_json(&root.join("escape/result.json")).is_err());
        assert!(JobResult::<()>::read_hdf5(&root.join("escape/result.h5")).is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn dynamic_merge_is_strict_without_modulo_ownership() {
        let job = Job::<M>::new("j", vec![Task::new("a", P(1)), Task::new("b", P(2))]);
        let rank_zero = JobResult {
            job_name: "j".into(),
            rank: 0,
            world_size: 2,
            tasks: vec![task_result(1, job.tasks[1].clone())],
        };
        let rank_one = JobResult {
            job_name: "j".into(),
            rank: 1,
            world_size: 2,
            tasks: vec![task_result(0, job.tasks[0].clone())],
        };
        let merged =
            merge_dynamic_results(&job, vec![rank_zero.clone(), rank_one.clone()]).unwrap();
        assert_eq!(
            merged
                .tasks
                .iter()
                .map(|task| task.task_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(merge_results(&job, vec![rank_zero.clone(), rank_one.clone()]).is_err());

        let mut duplicate = rank_one.clone();
        duplicate.tasks[0] = task_result(1, job.tasks[1].clone());
        assert!(merge_dynamic_results(&job, vec![rank_zero.clone(), duplicate]).is_err());

        let missing = JobResult {
            tasks: vec![],
            ..rank_one.clone()
        };
        assert!(merge_dynamic_results(&job, vec![rank_zero.clone(), missing]).is_err());

        let mut conflict = rank_one;
        conflict.tasks[0].task.parameters = P(9);
        assert!(merge_dynamic_results(&job, vec![rank_zero, conflict]).is_err());
    }
}
