use std::{error::Error, fmt};

/// A parsed, read-only configuration table.
///
/// `Params` is the generic configuration layer shared by models that want a
/// common way to turn a TOML document (or any nested table of values) into a
/// list of [`Task`](crate::Task)s. A model implements
/// [`MonteCarlo::build_tasks`](crate::MonteCarlo::build_tasks) to read the
/// values it needs from a `Params` and produce its parameter grid.
///
/// Values are stored as `toml::Value`s, so a `Params` can represent any valid
/// TOML structure (tables, arrays, scalars). Typed accessors such as
/// [`Params::get_f64`] and [`Params::get_usize`] validate the value shape and
/// return [`ParamsError`] on a type mismatch, so configuration errors surface
/// early with a clear key name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params {
    table: toml::Table,
}

impl Params {
    /// Creates an empty parameter table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses a TOML document into a `Params` table.
    ///
    /// The document must have a table (or table-array) at its root.
    pub fn parse(text: &str) -> Result<Self, ParamsError> {
        text.parse()
    }

    /// Wraps an existing TOML table.
    pub fn from_table(table: toml::Table) -> Self {
        Self { table }
    }

    /// Returns the underlying TOML table.
    pub fn as_table(&self) -> &toml::Table {
        &self.table
    }

    /// Consumes `self` and returns the underlying TOML table.
    pub fn into_table(self) -> toml::Table {
        self.table
    }

    /// Returns whether `key` exists in this table.
    pub fn contains(&self, key: &str) -> bool {
        self.table.contains_key(key)
    }

    /// Returns the raw value stored under `key`, if present.
    pub fn get(&self, key: &str) -> Option<&toml::Value> {
        self.table.get(key)
    }

    /// Returns the nested table stored under `key`, if present.
    pub fn get_table(&self, key: &str) -> Result<Option<Params>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Table(table)) => Ok(Some(Params::from_table(table.clone()))),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "table",
            }),
        }
    }

    /// Returns the string stored under `key`, if present.
    pub fn get_str(&self, key: &str) -> Result<Option<String>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "string",
            }),
        }
    }

    /// Returns the boolean stored under `key`, if present.
    pub fn get_bool(&self, key: &str) -> Result<Option<bool>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Boolean(value)) => Ok(Some(*value)),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "boolean",
            }),
        }
    }

    /// Returns the signed integer stored under `key`, if present.
    pub fn get_i64(&self, key: &str) -> Result<Option<i64>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Integer(value)) => Ok(Some(*value)),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "integer",
            }),
        }
    }

    /// Returns the non-negative integer stored under `key`, if present.
    pub fn get_usize(&self, key: &str) -> Result<Option<usize>, ParamsError> {
        match self.get_i64(key)? {
            None => Ok(None),
            Some(value) => usize::try_from(value)
                .map(Some)
                .map_err(|_| ParamsError::Type {
                    key: key.to_string(),
                    expected: "non-negative integer",
                }),
        }
    }

    /// Returns the floating-point value stored under `key`, if present.
    ///
    /// Integer values are accepted and widened to `f64` for convenience.
    pub fn get_f64(&self, key: &str) -> Result<Option<f64>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Float(value)) => Ok(Some(*value)),
            Some(toml::Value::Integer(value)) => Ok(Some(*value as f64)),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "float",
            }),
        }
    }

    /// Returns the array of strings stored under `key`, if present.
    pub fn get_array_str(&self, key: &str) -> Result<Option<Vec<String>>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Array(values)) => values
                .iter()
                .map(|value| match value {
                    toml::Value::String(value) => Ok(value.clone()),
                    _ => Err(ParamsError::Type {
                        key: key.to_string(),
                        expected: "array of strings",
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "array of strings",
            }),
        }
    }

    /// Returns the array of floating-point values stored under `key`, if present.
    ///
    /// Integer elements are accepted and widened to `f64`.
    pub fn get_array_f64(&self, key: &str) -> Result<Option<Vec<f64>>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Array(values)) => values
                .iter()
                .map(|value| match value {
                    toml::Value::Float(value) => Ok(*value),
                    toml::Value::Integer(value) => Ok(*value as f64),
                    _ => Err(ParamsError::Type {
                        key: key.to_string(),
                        expected: "array of floats",
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "array of floats",
            }),
        }
    }

    /// Returns the array of signed integers stored under `key`, if present.
    pub fn get_array_i64(&self, key: &str) -> Result<Option<Vec<i64>>, ParamsError> {
        match self.table.get(key) {
            None => Ok(None),
            Some(toml::Value::Array(values)) => values
                .iter()
                .map(|value| match value {
                    toml::Value::Integer(value) => Ok(*value),
                    _ => Err(ParamsError::Type {
                        key: key.to_string(),
                        expected: "array of integers",
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
            Some(_) => Err(ParamsError::Type {
                key: key.to_string(),
                expected: "array of integers",
            }),
        }
    }

    /// Returns the array of non-negative integers stored under `key`, if present.
    pub fn get_array_usize(&self, key: &str) -> Result<Option<Vec<usize>>, ParamsError> {
        match self.get_array_i64(key)? {
            None => Ok(None),
            Some(values) => values
                .into_iter()
                .map(|value| {
                    usize::try_from(value).map_err(|_| ParamsError::Type {
                        key: key.to_string(),
                        expected: "array of non-negative integers",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some),
        }
    }
}

impl std::str::FromStr for Params {
    type Err = ParamsError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let table = toml::from_str::<toml::Table>(text).map_err(|error| ParamsError::Parse {
            message: error.to_string(),
        })?;
        Ok(Self { table })
    }
}

/// Error returned by [`Params`] accessors and parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamsError {
    /// The document could not be parsed as a TOML table.
    Parse { message: String },
    /// A value under `key` had a different type than `expected`.
    Type { key: String, expected: &'static str },
}

impl fmt::Display for ParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamsError::Parse { message } => write!(f, "failed to parse config: {message}"),
            ParamsError::Type { key, expected } => {
                write!(f, "config key `{key}` must be a {expected}")
            }
        }
    }
}

impl Error for ParamsError {}
