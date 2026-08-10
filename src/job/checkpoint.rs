use super::{
    paths::{atomic_write, dump_path, ensure_safe_read_file_path},
    CompactAccumulator, GenericJobError, Task,
};
use hdf5_pure::{AttrValue, DType, File, FileBuilder, Group};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const KIND: &str = "carlo-mc-checkpoint";
const VERSION: u64 = 1;
pub(crate) const CHECKPOINT_PAYLOAD_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Checkpoint<P> {
    pub schema_version: u32,
    pub rank: usize,
    pub world_size: usize,
    pub job_name: String,
    pub task: Task<P>,
    pub task_index: usize,
    pub model: serde_json::Value,
    pub rng_position_words: [u64; 2],
    pub thermalization_sweeps: usize,
    pub measurement_sweeps: usize,
    pub observables: BTreeMap<String, CompactAccumulator>,
}

/// Returns the first-run dump path for a zero-based task index.
///
/// This compatibility helper is equivalent to `dump_path(dir, index, 0)`.
pub fn checkpoint_path(dir: &Path, index: usize) -> PathBuf {
    dump_path(dir, index, 0)
}

pub(crate) fn write_checkpoint<P: Serialize>(
    path: &Path,
    checkpoint: &Checkpoint<P>,
) -> Result<(), GenericJobError> {
    if checkpoint.observables.len() > 10_000 {
        return Err(schema(
            path,
            "too many checkpoint observables for four-digit names",
        ));
    }
    let task = serde_json::to_vec(&checkpoint.task)
        .map_err(|error| GenericJobError::json(Some(path), error))?;
    let parameters = serde_json::to_vec(&checkpoint.task.parameters)
        .map_err(|error| GenericJobError::json(Some(path), error))?;
    let model = serde_json::to_vec(&checkpoint.model)
        .map_err(|error| GenericJobError::json(Some(path), error))?;

    let mut builder = FileBuilder::new();
    set_root_attrs(&mut builder, KIND);

    let mut metadata = builder.create_group("metadata");
    add_u64(&mut metadata, "schema", checkpoint.schema_version.into());
    add_utf8(path, &mut metadata, "job", &checkpoint.job_name)?;
    add_bytes(path, &mut metadata, "task", &task)?;
    add_bytes(path, &mut metadata, "model", &model)?;
    add_bytes(path, &mut metadata, "parameters", &parameters)?;
    builder.add_group(metadata.finish());

    let mut assignment = builder.create_group("assignment");
    add_u64(&mut assignment, "rank", to_u64(path, checkpoint.rank)?);
    add_u64(
        &mut assignment,
        "world_size",
        to_u64(path, checkpoint.world_size)?,
    );
    builder.add_group(assignment.finish());

    let mut progress = builder.create_group("progress");
    add_u64(
        &mut progress,
        "task_index",
        to_u64(path, checkpoint.task_index)?,
    );
    add_u64(
        &mut progress,
        "thermalization_sweeps",
        to_u64(path, checkpoint.thermalization_sweeps)?,
    );
    add_u64(
        &mut progress,
        "measurement_sweeps",
        to_u64(path, checkpoint.measurement_sweeps)?,
    );
    builder.add_group(progress.finish());

    let mut state = builder.create_group("state");
    state
        .create_dataset("rng_position")
        .with_u64_data(&checkpoint.rng_position_words)
        .with_shape(&[2]);
    builder.add_group(state.finish());

    let mut observables = builder.create_group("observables");
    for (index, (name, accumulator)) in checkpoint.observables.iter().enumerate() {
        let mut observable = observables.create_group(&format!("observable{index:04}"));
        add_utf8(path, &mut observable, "name", name)?;
        observable
            .create_dataset("internal_bins")
            .with_f64_data(&accumulator.internal_bins)
            .with_shape(&[to_u64(path, accumulator.internal_bins.len())?]);
        add_f64(&mut observable, "pending_sum", accumulator.pending_sum);
        add_u64(
            &mut observable,
            "pending_count",
            to_u64(path, accumulator.pending_count)?,
        );
        add_u64(
            &mut observable,
            "total_count",
            to_u64(path, accumulator.total_count)?,
        );
        add_u64(
            &mut observable,
            "bin_length",
            to_u64(path, accumulator.binsize)?,
        );
        observables.add_group(observable.finish());
    }
    builder.add_group(observables.finish());

    finish_hdf5(path, builder)
}

pub(crate) fn read_checkpoint<P: DeserializeOwned>(
    path: &Path,
) -> Result<Checkpoint<P>, GenericJobError> {
    let file = open_hdf5(path, KIND)?;
    exact_root_group(
        path,
        file.root(),
        &[],
        &["assignment", "metadata", "observables", "progress", "state"],
    )?;
    exact_group(
        path,
        group(path, &file, "metadata")?,
        &["job", "model", "parameters", "schema", "task"],
        &[],
    )?;
    exact_group(
        path,
        group(path, &file, "assignment")?,
        &["rank", "world_size"],
        &[],
    )?;
    exact_group(
        path,
        group(path, &file, "progress")?,
        &["measurement_sweeps", "task_index", "thermalization_sweeps"],
        &[],
    )?;
    exact_group(path, group(path, &file, "state")?, &["rng_position"], &[])?;

    let schema_version = u32::try_from(read_u64_scalar(path, &file, "metadata/schema")?)
        .map_err(|_| schema(path, "checkpoint schema version exceeds u32"))?;
    let rank = read_usize(path, &file, "assignment/rank")?;
    let world_size = read_usize(path, &file, "assignment/world_size")?;
    let job_name = read_utf8(path, &file, "metadata/job")?;
    let task_index = read_usize(path, &file, "progress/task_index")?;
    let task_value: serde_json::Value = read_json(path, &file, "metadata/task")?;
    let parameters_value: serde_json::Value = read_json(path, &file, "metadata/parameters")?;
    if task_value.get("parameters") != Some(&parameters_value) {
        return Err(schema(path, "checkpoint parameters do not match task"));
    }
    let task: Task<P> = serde_json::from_value(task_value)
        .map_err(|_| schema(path, "invalid checkpoint task JSON"))?;
    let model = read_json(path, &file, "metadata/model")?;
    let rng = read_u64_array(path, &file, "state/rng_position", 2)?;
    let thermalization_sweeps = read_usize(path, &file, "progress/thermalization_sweeps")?;
    let measurement_sweeps = read_usize(path, &file, "progress/measurement_sweeps")?;

    let observables_group = group(path, &file, "observables")?;
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
        let base = format!("observables/{observable_group}");
        exact_group(
            path,
            group(path, &file, &base)?,
            &[
                "bin_length",
                "internal_bins",
                "name",
                "pending_count",
                "pending_sum",
                "total_count",
            ],
            &[],
        )?;
        let name = read_utf8(path, &file, &format!("{base}/name"))?;
        let accumulator = CompactAccumulator {
            internal_bins: read_f64_array(path, &file, &format!("{base}/internal_bins"))?,
            pending_sum: read_f64_scalar(path, &file, &format!("{base}/pending_sum"))?,
            pending_count: read_usize(path, &file, &format!("{base}/pending_count"))?,
            total_count: read_usize(path, &file, &format!("{base}/total_count"))?,
            binsize: read_usize(path, &file, &format!("{base}/bin_length"))?,
        };
        if observables.insert(name, accumulator).is_some() {
            return Err(schema(path, "duplicate checkpoint observable name"));
        }
    }

    let checkpoint = Checkpoint {
        schema_version,
        rank,
        world_size,
        job_name,
        task,
        task_index,
        model,
        rng_position_words: [rng[0], rng[1]],
        thermalization_sweeps,
        measurement_sweeps,
        observables,
    };
    validate_checkpoint(path, &checkpoint)?;
    Ok(checkpoint)
}

pub(crate) fn encode_rng_position(position: u128) -> [u64; 2] {
    [position as u64, (position >> 64) as u64]
}

pub(crate) fn decode_rng_position(words: [u64; 2]) -> u128 {
    u128::from(words[0]) | (u128::from(words[1]) << 64)
}

pub(crate) fn set_root_attrs(builder: &mut FileBuilder, kind: &str) {
    builder.set_attr("carlo_kind", AttrValue::String(kind.into()));
    builder.set_attr("schema_version", AttrValue::U64(VERSION));
}

pub(crate) fn finish_hdf5(path: &Path, builder: FileBuilder) -> Result<(), GenericJobError> {
    let bytes = builder.finish().map_err(|error| hdf5(path, error))?;
    atomic_write(path, &bytes)
}

pub(crate) fn open_hdf5(path: &Path, kind: &str) -> Result<File, GenericJobError> {
    ensure_safe_read_file_path(path)?;
    let bytes =
        fs::read(path).map_err(|error| GenericJobError::io("read HDF5 file", path, error))?;
    let file = File::from_bytes(bytes).map_err(|error| hdf5(path, error))?;
    let root = file.root();
    let attrs = root.attrs().map_err(|error| hdf5(path, error))?;
    let attr_datatypes = root.attr_datatypes().map_err(|error| hdf5(path, error))?;
    if attrs.len() != 2
        || attr_datatypes.len() != 2
        || !attr_datatypes.contains_key("carlo_kind")
        || !attr_datatypes.contains_key("schema_version")
        || attrs.get("carlo_kind") != Some(&AttrValue::String(kind.into()))
        || attrs.get("schema_version") != Some(&AttrValue::U64(VERSION))
    {
        return Err(schema(path, "invalid root attributes"));
    }
    if !root
        .named_datatypes()
        .map_err(|error| hdf5(path, error))?
        .is_empty()
    {
        return Err(schema(path, "unexpected named datatype"));
    }
    Ok(file)
}

pub(crate) fn exact_root_group(
    path: &Path,
    group: Group,
    datasets: &[&str],
    groups: &[&str],
) -> Result<(), GenericJobError> {
    exact_group_contents(path, &group, datasets, groups)?;
    if !group
        .named_datatypes()
        .map_err(|error| hdf5(path, error))?
        .is_empty()
    {
        return Err(schema(path, "unexpected HDF5 root metadata"));
    }
    Ok(())
}

pub(crate) fn exact_group(
    path: &Path,
    group: Group,
    datasets: &[&str],
    groups: &[&str],
) -> Result<(), GenericJobError> {
    exact_group_contents(path, &group, datasets, groups)?;
    if !group
        .attr_datatypes()
        .map_err(|error| hdf5(path, error))?
        .is_empty()
        || !group
            .named_datatypes()
            .map_err(|error| hdf5(path, error))?
            .is_empty()
    {
        return Err(schema(path, "unexpected HDF5 group metadata"));
    }
    Ok(())
}

fn exact_group_contents(
    path: &Path,
    group: &Group,
    datasets: &[&str],
    groups: &[&str],
) -> Result<(), GenericJobError> {
    let mut actual_datasets = group.datasets().map_err(|error| hdf5(path, error))?;
    let mut actual_groups = group.groups().map_err(|error| hdf5(path, error))?;
    actual_datasets.sort();
    actual_groups.sort();
    let mut expected_datasets = datasets.to_vec();
    let mut expected_groups = groups.to_vec();
    expected_datasets.sort();
    expected_groups.sort();
    if actual_datasets != expected_datasets || actual_groups != expected_groups {
        return Err(schema(path, "unexpected HDF5 group contents"));
    }
    Ok(())
}
pub(crate) fn group(path: &Path, file: &File, name: &str) -> Result<Group, GenericJobError> {
    file.group(name).map_err(|error| hdf5(path, error))
}

pub(crate) fn read_utf8(path: &Path, file: &File, name: &str) -> Result<String, GenericJobError> {
    String::from_utf8(read_u8_array(path, file, name)?)
        .map_err(|_| schema(path, "string dataset is not valid UTF-8"))
}

pub(crate) fn read_json<T: DeserializeOwned>(
    path: &Path,
    file: &File,
    name: &str,
) -> Result<T, GenericJobError> {
    let bytes = read_u8_array(path, file, name)?;
    std::str::from_utf8(&bytes).map_err(|_| schema(path, "JSON dataset is not valid UTF-8"))?;
    serde_json::from_slice(&bytes).map_err(|_| schema(path, "invalid JSON dataset"))
}

pub(crate) fn read_u64_scalar(
    path: &Path,
    file: &File,
    name: &str,
) -> Result<u64, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::U64
        || !dataset
            .shape()
            .map_err(|error| hdf5(path, error))?
            .is_empty()
    {
        return Err(schema(path, "invalid u64 scalar dataset type or shape"));
    }
    let values = dataset.read_u64().map_err(|error| hdf5(path, error))?;
    if values.len() != 1 {
        return Err(schema(path, "invalid u64 scalar dataset length"));
    }
    Ok(values[0])
}

pub(crate) fn read_usize(path: &Path, file: &File, name: &str) -> Result<usize, GenericJobError> {
    usize::try_from(read_u64_scalar(path, file, name)?)
        .map_err(|_| schema(path, "u64 value exceeds usize"))
}

pub(crate) fn read_u8_scalar(path: &Path, file: &File, name: &str) -> Result<u8, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::U8
        || !dataset
            .shape()
            .map_err(|error| hdf5(path, error))?
            .is_empty()
    {
        return Err(schema(path, "invalid u8 scalar dataset type or shape"));
    }
    let values = dataset.read_u8().map_err(|error| hdf5(path, error))?;
    if values.len() != 1 {
        return Err(schema(path, "invalid u8 scalar dataset length"));
    }
    Ok(values[0])
}

pub(crate) fn read_u8_array(
    path: &Path,
    file: &File,
    name: &str,
) -> Result<Vec<u8>, GenericJobError> {
    read_typed_u8(path, file, name, None)
}

fn read_typed_u8(
    path: &Path,
    file: &File,
    name: &str,
    exact_len: Option<usize>,
) -> Result<Vec<u8>, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    let shape = dataset.shape().map_err(|error| hdf5(path, error))?;
    let exact_len = exact_len.map(|length| to_u64(path, length)).transpose()?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::U8
        || shape.len() != 1
        || exact_len.is_some_and(|length| shape[0] != length)
    {
        return Err(schema(path, "invalid u8 dataset type or shape"));
    }
    let values = dataset.read_u8().map_err(|error| hdf5(path, error))?;
    validate_length(path, shape[0], values.len())?;
    Ok(values)
}

pub(crate) fn read_u64_array(
    path: &Path,
    file: &File,
    name: &str,
    length: usize,
) -> Result<Vec<u64>, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    let shape = dataset.shape().map_err(|error| hdf5(path, error))?;
    let length = to_u64(path, length)?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::U64 || shape != [length] {
        return Err(schema(path, "invalid u64 dataset type or shape"));
    }
    let values = dataset.read_u64().map_err(|error| hdf5(path, error))?;
    validate_length(path, shape[0], values.len())?;
    Ok(values)
}

pub(crate) fn read_f64_scalar(
    path: &Path,
    file: &File,
    name: &str,
) -> Result<f64, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::F64
        || !dataset
            .shape()
            .map_err(|error| hdf5(path, error))?
            .is_empty()
    {
        return Err(schema(path, "invalid f64 scalar dataset type or shape"));
    }
    let values = dataset.read_f64().map_err(|error| hdf5(path, error))?;
    if values.len() != 1 {
        return Err(schema(path, "invalid f64 scalar dataset length"));
    }
    Ok(values[0])
}

pub(crate) fn read_f64_array(
    path: &Path,
    file: &File,
    name: &str,
) -> Result<Vec<f64>, GenericJobError> {
    read_f64(path, file, name, None)
}

fn read_f64(
    path: &Path,
    file: &File,
    name: &str,
    exact_len: Option<usize>,
) -> Result<Vec<f64>, GenericJobError> {
    let dataset = file.dataset(name).map_err(|error| hdf5(path, error))?;
    exact_dataset(path, &dataset)?;
    let shape = dataset.shape().map_err(|error| hdf5(path, error))?;
    let exact_len = exact_len.map(|length| to_u64(path, length)).transpose()?;
    if dataset.dtype().map_err(|error| hdf5(path, error))? != DType::F64
        || shape.len() != 1
        || exact_len.is_some_and(|length| shape[0] != length)
    {
        return Err(schema(path, "invalid f64 dataset type or shape"));
    }
    let values = dataset.read_f64().map_err(|error| hdf5(path, error))?;
    validate_length(path, shape[0], values.len())?;
    Ok(values)
}

pub(crate) fn numbered_groups(
    path: &Path,
    group: &Group,
    prefix: &str,
) -> Result<Vec<String>, GenericJobError> {
    let groups = group.groups().map_err(|error| hdf5(path, error))?;
    for (index, name) in groups.iter().enumerate() {
        if index > 9_999 || name != &format!("{prefix}{index:04}") {
            return Err(schema(path, "invalid four-digit HDF5 group sequence"));
        }
    }
    Ok(groups)
}

pub(crate) fn add_utf8(
    path: &Path,
    group: &mut hdf5_pure::GroupBuilder,
    name: &str,
    value: &str,
) -> Result<(), GenericJobError> {
    add_bytes(path, group, name, value.as_bytes())
}

pub(crate) fn add_bytes(
    path: &Path,
    group: &mut hdf5_pure::GroupBuilder,
    name: &str,
    value: &[u8],
) -> Result<(), GenericJobError> {
    group
        .create_dataset(name)
        .with_u8_data(value)
        .with_shape(&[to_u64(path, value.len())?]);
    Ok(())
}

pub(crate) fn add_u64(group: &mut hdf5_pure::GroupBuilder, name: &str, value: u64) {
    group
        .create_dataset(name)
        .with_u64_data(&[value])
        .with_shape(&[]);
}

pub(crate) fn add_u8(group: &mut hdf5_pure::GroupBuilder, name: &str, value: u8) {
    group
        .create_dataset(name)
        .with_u8_data(&[value])
        .with_shape(&[]);
}

pub(crate) fn add_f64(group: &mut hdf5_pure::GroupBuilder, name: &str, value: f64) {
    group
        .create_dataset(name)
        .with_f64_data(&[value])
        .with_shape(&[]);
}

pub(crate) fn to_u64(path: &Path, value: usize) -> Result<u64, GenericJobError> {
    u64::try_from(value).map_err(|_| schema(path, "usize value exceeds u64"))
}

pub(crate) fn schema(path: &Path, reason: &'static str) -> GenericJobError {
    GenericJobError::Schema {
        path: path.into(),
        reason,
    }
}

fn hdf5(path: &Path, error: impl std::fmt::Display) -> GenericJobError {
    GenericJobError::Hdf5 {
        path: path.into(),
        reason: error.to_string(),
    }
}

fn exact_dataset(path: &Path, dataset: &hdf5_pure::Dataset) -> Result<(), GenericJobError> {
    if !dataset
        .attr_datatypes()
        .map_err(|error| hdf5(path, error))?
        .is_empty()
    {
        return Err(schema(path, "unexpected HDF5 dataset attributes"));
    }
    Ok(())
}

fn validate_length(path: &Path, declared: u64, actual: usize) -> Result<(), GenericJobError> {
    if usize::try_from(declared).map_err(|_| schema(path, "dataset length exceeds usize"))?
        != actual
    {
        return Err(schema(path, "dataset shape does not match data length"));
    }
    Ok(())
}

fn validate_checkpoint<P>(path: &Path, checkpoint: &Checkpoint<P>) -> Result<(), GenericJobError> {
    if checkpoint.schema_version != CHECKPOINT_PAYLOAD_SCHEMA_VERSION
        || checkpoint.world_size == 0
        || checkpoint.rank >= checkpoint.world_size
        || checkpoint.task.name.is_empty()
        || checkpoint.task.sweeps == 0
        || checkpoint.task.binsize == 0
        || checkpoint.task.sweeps < checkpoint.task.binsize
        || checkpoint.thermalization_sweeps > checkpoint.task.thermalization
        || checkpoint.measurement_sweeps > checkpoint.task.sweeps
        || (checkpoint.measurement_sweeps > 0
            && checkpoint.thermalization_sweeps != checkpoint.task.thermalization)
    {
        return Err(schema(path, "invalid checkpoint assignment or sweep state"));
    }
    for (name, accumulator) in &checkpoint.observables {
        if name.is_empty()
            || accumulator.binsize == 0
            || accumulator.pending_count >= accumulator.binsize
            || accumulator.total_count > checkpoint.measurement_sweeps
            || (accumulator.pending_count == 0 && accumulator.pending_sum != 0.0)
            || !accumulator.pending_sum.is_finite()
            || accumulator
                .internal_bins
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(schema(path, "invalid checkpoint accumulator"));
        }
        let samples = accumulator
            .internal_bins
            .len()
            .checked_mul(accumulator.binsize)
            .and_then(|count| count.checked_add(accumulator.pending_count))
            .ok_or_else(|| schema(path, "checkpoint accumulator total overflow"))?;
        if samples != accumulator.total_count {
            return Err(schema(path, "inconsistent checkpoint accumulator total"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::paths::temp_dir;
    use super::*;

    fn checkpoint() -> Checkpoint<()> {
        Checkpoint {
            schema_version: CHECKPOINT_PAYLOAD_SCHEMA_VERSION,
            rank: 0,
            world_size: 1,
            job_name: "job".into(),
            task: Task::new("task", ()).thermalization(1).sweeps(5).binsize(2),
            task_index: 3,
            model: serde_json::json!({"value": 4.0}),
            rng_position_words: [7, 9],
            thermalization_sweeps: 1,
            measurement_sweeps: 3,
            observables: BTreeMap::from([(
                "energy/value".into(),
                CompactAccumulator {
                    internal_bins: vec![1.5],
                    pending_sum: 2.5,
                    pending_count: 1,
                    total_count: 3,
                    binsize: 2,
                },
            )]),
        }
    }

    #[test]
    fn checkpoint_hdf5_has_strict_structure_and_roundtrips_partial_bin() {
        let directory = temp_dir("checkpoint-hdf5-schema");
        let path = directory.join("checkpoint.h5");
        let expected = checkpoint();
        write_checkpoint(&path, &expected).unwrap();

        let file = File::open(&path).unwrap();
        assert_eq!(
            file.root().attrs().unwrap(),
            std::collections::HashMap::from([
                ("carlo_kind".into(), AttrValue::String(KIND.into())),
                ("schema_version".into(), AttrValue::U64(VERSION)),
            ])
        );
        assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
        assert_eq!(
            file.root()
                .groups()
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            ["assignment", "metadata", "observables", "progress", "state"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
        assert_eq!(
            file.group("metadata").unwrap().datasets().unwrap(),
            ["schema", "job", "task", "model", "parameters"]
        );
        assert_eq!(
            file.group("assignment").unwrap().datasets().unwrap(),
            ["rank", "world_size"]
        );
        assert_eq!(
            file.group("progress").unwrap().datasets().unwrap(),
            ["task_index", "thermalization_sweeps", "measurement_sweeps"]
        );
        assert_eq!(
            file.group("state").unwrap().datasets().unwrap(),
            ["rng_position"]
        );
        assert_eq!(
            file.group("observables/observable0000")
                .unwrap()
                .datasets()
                .unwrap(),
            [
                "name",
                "internal_bins",
                "pending_sum",
                "pending_count",
                "total_count",
                "bin_length"
            ]
        );
        assert!(!file
            .as_bytes()
            .windows(b"payload_json".len())
            .any(|window| window == b"payload_json"));
        assert_eq!(
            file.dataset("assignment/rank").unwrap().dtype().unwrap(),
            DType::U64
        );
        assert!(file
            .dataset("assignment/rank")
            .unwrap()
            .shape()
            .unwrap()
            .is_empty());
        assert_eq!(
            file.dataset("metadata/task").unwrap().dtype().unwrap(),
            DType::U8
        );
        assert_eq!(
            file.dataset("state/rng_position").unwrap().shape().unwrap(),
            [2]
        );
        assert_eq!(
            file.dataset("observables/observable0000/internal_bins")
                .unwrap()
                .shape()
                .unwrap(),
            [1]
        );
        assert!(file
            .dataset("observables/observable0000/pending_sum")
            .unwrap()
            .shape()
            .unwrap()
            .is_empty());
        assert_eq!(
            file.dataset("observables/observable0000/bin_length")
                .unwrap()
                .dtype()
                .unwrap(),
            DType::U64
        );
        assert!(file.dataset("observables/observable0000/binsize").is_err());

        let restored: Checkpoint<()> = read_checkpoint(&path).unwrap();
        assert_eq!(restored, expected);
        assert_eq!(restored.observables["energy/value"].pending_count, 1);
        assert_eq!(restored.observables["energy/value"].pending_sum, 2.5);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn checkpoint_rejects_observable_index_above_four_digits() {
        let directory = temp_dir("checkpoint-observable-limit");
        let path = directory.join("checkpoint.h5");
        let mut checkpoint = checkpoint();
        checkpoint.measurement_sweeps = 0;
        checkpoint.observables = (0..10_001)
            .map(|index| {
                (
                    format!("observable-{index}"),
                    CompactAccumulator {
                        internal_bins: vec![],
                        pending_sum: 0.0,
                        pending_count: 0,
                        total_count: 0,
                        binsize: 1,
                    },
                )
            })
            .collect();
        assert!(matches!(
            write_checkpoint(&path, &checkpoint),
            Err(GenericJobError::Schema { .. })
        ));
        assert!(!path.exists());
        if directory.exists() {
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn checkpoint_name_and_rng_position_are_preserved() {
        assert_eq!(
            checkpoint_path(Path::new("x"), 0),
            Path::new("x/task0001/run0001.dump.h5")
        );
        let position = (u128::from(u64::MAX) << 64) | 17;
        assert_eq!(decode_rng_position(encode_rng_position(position)), position);
    }
}
