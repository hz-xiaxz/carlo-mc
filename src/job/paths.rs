use crate::GenericJobError;
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);
const TEMP_ATTEMPTS: usize = 128;

/// Returns the directory for a zero-based task index.
pub fn task_path(root: &Path, task_index: usize) -> PathBuf {
    root.join(format!("task{:04}", task_index + 1))
}

/// Returns the HDF5 dump path for zero-based task and run indices.
pub fn dump_path(root: &Path, task_index: usize, run_index: usize) -> PathBuf {
    task_path(root, task_index).join(format!("run{:04}.dump.h5", run_index + 1))
}

/// Returns the HDF5 measurement path for zero-based task and run indices.
pub fn measurement_path(root: &Path, task_index: usize, run_index: usize) -> PathBuf {
    task_path(root, task_index).join(format!("run{:04}.meas.h5", run_index + 1))
}

/// Returns the canonical HDF5 result path.
pub fn result_path(root: &Path) -> PathBuf {
    root.join("result")
}

fn create_unique_temporary(
    parent: &Path,
    prefix: &str,
) -> Result<(std::path::PathBuf, File), GenericJobError> {
    create_unique_temporary_with(parent, prefix, || TEMP_ID.fetch_add(1, Ordering::Relaxed))
}

fn create_unique_temporary_with(
    parent: &Path,
    prefix: &str,
    mut next_id: impl FnMut() -> u64,
) -> Result<(std::path::PathBuf, File), GenericJobError> {
    for _ in 0..TEMP_ATTEMPTS {
        let temporary = parent.join(format!(
            ".{prefix}.tmp.{}.{}",
            std::process::id(),
            next_id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(GenericJobError::io(
                    "create temporary file",
                    &temporary,
                    error,
                ));
            }
        }
    }
    Err(GenericJobError::io(
        "create unique temporary file",
        parent,
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "temporary name retry limit exhausted",
        ),
    ))
}

pub fn validate_safe_component(value: &str) -> Result<(), GenericJobError> {
    let stem = value
        .split('.')
        .next()
        .unwrap_or(value)
        .to_ascii_uppercase();
    let windows_reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'));
    let bad = value.is_empty()
        || value.trim() != value
        || value.ends_with('.')
        || value.contains(['/', '\\', ':', '\0'])
        || value.chars().any(char::is_control)
        || matches!(value, "." | "..")
        || windows_reserved
        || Path::new(value).is_absolute()
        || Path::new(value)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)));
    if bad {
        Err(GenericJobError::UnsafePath(value.into()))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_safe_path(path: &Path) -> Result<(), GenericJobError> {
    let unsafe_path = || GenericJobError::UnsafePath(path.display().to_string());
    let rendered = path.to_str().ok_or_else(unsafe_path)?;
    let component_text = if path.is_absolute() {
        rendered.strip_prefix(std::path::MAIN_SEPARATOR)
    } else {
        Some(rendered)
    }
    .ok_or_else(unsafe_path)?;
    if component_text.is_empty()
        || component_text
            .split(std::path::MAIN_SEPARATOR)
            .any(|component| component.is_empty() || validate_safe_component(component).is_err())
        || rendered.contains('\\')
    {
        return Err(unsafe_path());
    }

    let mut saw_root = false;
    for component in path.components() {
        match component {
            Component::RootDir if path.is_absolute() && !saw_root => saw_root = true,
            Component::Normal(value) => {
                validate_safe_component(value.to_str().ok_or_else(unsafe_path)?)?
            }
            _ => return Err(unsafe_path()),
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_safe_relative_path(path: &Path) -> Result<(), GenericJobError> {
    if path.is_absolute() {
        return Err(GenericJobError::UnsafePath(path.display().to_string()));
    }
    validate_safe_path(path)
}

fn ensure_existing_component_is_safe(path: &Path, directory: bool) -> Result<(), GenericJobError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(GenericJobError::UnsafePath(path.display().to_string()))
        }
        Ok(metadata) if directory && !metadata.is_dir() => Err(GenericJobError::io(
            "use path component as directory",
            path,
            std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                "component is not a directory",
            ),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(GenericJobError::io("inspect path component", path, error)),
    }
}

pub(crate) fn ensure_safe_directory(path: &Path) -> Result<(), GenericJobError> {
    validate_safe_path(path)?;
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => {
                current.push(component.as_os_str());
                ensure_existing_component_is_safe(&current, true)?;
            }
            Component::Normal(component) => {
                current.push(component);
                ensure_existing_component_is_safe(&current, true)?;
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => {
                        return Err(GenericJobError::io("create directory", &current, error))
                    }
                }
                ensure_existing_component_is_safe(&current, true)?;
            }
            _ => unreachable!("validated path has only a root and normal components"),
        }
    }
    Ok(())
}

pub(crate) fn ensure_safe_file_path(path: &Path) -> Result<(), GenericJobError> {
    validate_safe_path(path)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        ensure_safe_directory(parent)?;
    }
    ensure_existing_component_is_safe(path, false)
}

pub(crate) fn ensure_safe_read_file_path(path: &Path) -> Result<(), GenericJobError> {
    validate_safe_path(path)?;
    let mut current = PathBuf::new();
    let component_count = path.components().count();
    for (index, component) in path.components().enumerate() {
        match component {
            Component::RootDir => current.push(component.as_os_str()),
            Component::Normal(component) => current.push(component),
            _ => unreachable!("validated path has only a root and normal components"),
        }
        ensure_existing_component_is_safe(&current, index + 1 < component_count)?;
    }
    Ok(())
}

fn safe_parent(path: &Path) -> Result<&Path, GenericJobError> {
    ensure_safe_file_path(path)?;
    Ok(path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new(".")))
}
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), GenericJobError> {
    let parent = safe_parent(path)?;
    let name = path
        .file_name()
        .ok_or_else(|| GenericJobError::UnsafePath(path.display().to_string()))?;
    let prefix = name.to_string_lossy();
    let (tmp, mut f) = create_unique_temporary(parent, &prefix)?;
    let result = (|| {
        f.write_all(bytes)
            .map_err(|e| GenericJobError::io("write temporary file", &tmp, e))?;
        f.flush()
            .map_err(|e| GenericJobError::io("flush temporary file", &tmp, e))?;
        f.sync_all()
            .map_err(|e| GenericJobError::io("sync temporary file", &tmp, e))?;
        drop(f);
        fs::rename(&tmp, path)
            .map_err(|e| GenericJobError::io("rename temporary file", path, e))?;
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| GenericJobError::io("sync parent directory", parent, e))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}
pub(crate) fn durable_remove_if_exists(path: &Path) -> Result<bool, GenericJobError> {
    let parent = safe_parent(path)?;
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(GenericJobError::io("remove file", path, e)),
    }
    File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|e| GenericJobError::io("sync parent directory", parent, e))?;
    Ok(true)
}

pub(crate) fn exclusive_write(path: &Path, bytes: &[u8]) -> Result<bool, GenericJobError> {
    let parent = safe_parent(path)?;
    let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(error) => return Err(GenericJobError::io("create task claim", path, error)),
    };
    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| GenericJobError::io("write task claim", path, error))?;
        file.flush()
            .and_then(|_| file.sync_all())
            .map_err(|error| GenericJobError::io("sync task claim", path, error))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| GenericJobError::io("sync parent directory", parent, error))
    })();
    if result.is_err() {
        drop(file);
        let _ = fs::remove_file(path);
    }
    result.map(|()| true)
}
#[cfg(test)]
pub(crate) fn temp_dir(label: &str) -> PathBuf {
    PathBuf::from(format!(
        "carlo-mc-test-{label}-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(test)]
pub(crate) fn absolute_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "carlo-mc-test-{label}-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_layout_uses_one_based_task_and_run_names() {
        let root = Path::new("output");
        assert_eq!(task_path(root, 0), Path::new("output/task0001"));
        assert_eq!(task_path(root, 11), Path::new("output/task0012"));
        assert_eq!(
            dump_path(root, 0, 0),
            Path::new("output/task0001/run0001.dump.h5")
        );
        assert_eq!(
            dump_path(root, 1, 2),
            Path::new("output/task0002/run0003.dump.h5")
        );
        assert_eq!(
            measurement_path(root, 0, 0),
            Path::new("output/task0001/run0001.meas.h5")
        );
        assert_eq!(result_path(root), Path::new("output/result"));
    }

    #[test]
    fn rejects_dangerous() {
        for value in [
            "", ".", "..", "a/b", "a\\b", "/tmp", "../x", "x/..", " name", "name ", "name.", "a:b",
            "a\0b", "a\nb", "CON", "con.txt", "PRN", "AUX.json", "NUL", "COM1", "com9.log", "LPT1",
            "lpt9.txt",
        ] {
            assert!(
                validate_safe_component(value).is_err(),
                "accepted {value:?}"
            );
        }
        for value in ["safe-_1", ".hidden", "COM0", "LPT10", "auxiliary"] {
            assert!(validate_safe_component(value).is_ok(), "rejected {value:?}");
        }
        for path in ["/tmp/x", "../x", "x/../y", "x\\y", "x//y"] {
            assert!(validate_safe_relative_path(Path::new(path)).is_err());
        }
        assert!(validate_safe_relative_path(Path::new("safe/path.json")).is_ok());
        assert!(validate_safe_path(Path::new("/tmp/safe/path.json")).is_ok());
        for path in ["/", "/tmp//x", "/tmp/./x", "/tmp/../x", "/tmp/x/"] {
            assert!(
                validate_safe_path(Path::new(path)).is_err(),
                "accepted {path:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_existing_symlink_escape_for_read_and_write_paths() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("symlink-root");
        let outside = temp_dir("symlink-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("escape")).unwrap();
        let escaped = root.join("escape/payload");
        assert!(ensure_safe_file_path(&escaped).is_err());
        assert!(atomic_write(&escaped, b"no").is_err());
        assert!(!outside.join("payload").exists());

        fs::write(outside.join("target"), b"preserve").unwrap();
        symlink(outside.join("target"), root.join("target")).unwrap();
        assert!(ensure_safe_file_path(&root.join("target")).is_err());
        assert!(atomic_write(&root.join("target"), b"no").is_err());
        assert_eq!(fs::read(outside.join("target")).unwrap(), b"preserve");
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }
    #[test]
    fn atomic_is_durable_and_cleans() {
        let d = temp_dir("atomic");
        let p = d.join("x");
        atomic_write(&p, b"ok").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"ok");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1);
        assert!(durable_remove_if_exists(&p).unwrap());
        assert_eq!(fs::read_dir(&d).unwrap().count(), 0);
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn missing_concurrent_remove_is_not_an_error() {
        let d = temp_dir("remove-race");
        fs::create_dir_all(&d).unwrap();
        assert!(!durable_remove_if_exists(&d.join("missing")).unwrap());
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn temporary_creation_retries_collisions_without_overwrite() {
        let d = temp_dir("temporary-collision");
        fs::create_dir_all(&d).unwrap();
        let collision = d.join(format!(".x.tmp.{}.7", std::process::id()));
        fs::write(&collision, b"preserve").unwrap();
        let mut ids = [7, 8].into_iter();
        let (temporary, mut file) =
            create_unique_temporary_with(&d, "x", || ids.next().unwrap()).unwrap();
        file.write_all(b"result").unwrap();
        file.sync_all().unwrap();
        drop(file);
        assert_eq!(
            temporary,
            d.join(format!(".x.tmp.{}.8", std::process::id()))
        );
        assert_eq!(fs::read(&collision).unwrap(), b"preserve");
        assert_eq!(fs::read(&temporary).unwrap(), b"result");
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn exclusive_creation_has_one_winner_and_no_temporary_files() {
        let d = temp_dir("exclusive");
        let p = d.join("claim");
        assert!(exclusive_write(&p, b"first").unwrap());
        assert!(!exclusive_write(&p, b"second").unwrap());
        assert_eq!(fs::read(&p).unwrap(), b"first");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1);
        fs::remove_dir_all(d).unwrap();
    }
}
