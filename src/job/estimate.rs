//! Statistical estimation for Monte Carlo observables.
//!
//! This module provides the building blocks for turning raw measurement samples into
//! error estimates:
//!
//! - [`BinnedEstimate`] rebins raw/internal bins and computes a mean and standard error.
//! - [`Estimate`] is the serializable, per-observable result carried by
//!   [`TaskResult`](crate::TaskResult). It extends [`BinnedEstimate`] with fields used by
//!   downstream tooling (jackknife error, covariance, autocorrelation time).
//! - [`Evaluator`] computes *derived* observables from measured ones using jackknife
//!   resampling, mirroring the `register_evaluables` pattern of Carlo.jl.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::BTreeMap;

use super::ScalarEstimate;

/// The contract every per-observable estimate type stored in a
/// [`TaskResult`](crate::TaskResult) must satisfy.
///
/// A model can use the default [`Estimate`] (a fully featured estimate with binning
/// metadata, error, covariance, and autocorrelation time) or supply its own type, as
/// long as it is serializable, comparable, and can self-validate with [`ResultEstimate::valid`].
pub trait ResultEstimate:
    Clone + PartialEq + Serialize + DeserializeOwned + Send + Sync + 'static
{
    /// Returns whether the estimate is internally consistent and carries valid numbers.
    fn valid(&self) -> bool;
}

/// The arithmetic mean of a non-empty slice.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The number of rebinned bins Carlo-style rebinning targets for a sample count.
///
/// For at most 10 samples every sample is its own bin; beyond that the target count
/// grows as the cube root of the excess, so the number of bins stays small while each
/// bin becomes large enough for a stable variance estimate.
pub fn carlo_rebin_count(sample_count: usize) -> usize {
    if sample_count <= 10 {
        sample_count
    } else {
        10 + ((sample_count - 10) as f64).cbrt().round() as usize
    }
    .max(1)
}

/// The number of internal bins combined into one rebinned bin.
pub fn carlo_rebin_length(total_sample_count: usize) -> usize {
    if total_sample_count == 0 {
        1
    } else {
        (total_sample_count / carlo_rebin_count(total_sample_count)).max(1)
    }
}

/// The standard error of the mean of already-rebinned bins.
pub fn carlo_std_of_mean(bins: &[f64]) -> f64 {
    if bins.len() <= 1 {
        return f64::NAN;
    }
    let average = mean(bins);
    let variance = bins
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (bins.len() - 1) as f64;
    variance.sqrt() / (bins.len() as f64).sqrt()
}

/// A binned estimate before conversion into the final [`Estimate`].
///
/// `internal_bins` are the raw bin averages, and `bins` are the averages after
/// Carlo-style rebinning. `internal_bin_length` is the number of samples per internal
/// bin and `rebin_length` is the number of internal bins per rebinned bin.
#[derive(Debug, Clone, PartialEq)]
pub struct BinnedEstimate {
    pub mean: f64,
    pub stderr: f64,
    pub bins: Vec<f64>,
    pub internal_bins: Vec<f64>,
    pub internal_bin_length: usize,
    pub rebin_length: usize,
}

impl BinnedEstimate {
    /// Builds an estimate from raw samples using `internal_bin_length` samples per bin.
    pub fn from_samples(samples: &[f64], internal_bin_length: usize) -> Result<Self, String> {
        if internal_bin_length == 0 {
            return Err("binsize must be positive".to_string());
        }
        let usable = samples.len() - samples.len() % internal_bin_length;
        if usable == 0 {
            return Err("binsize is larger than the sample series".to_string());
        }
        let internal_bins = samples[..usable]
            .chunks_exact(internal_bin_length)
            .map(mean)
            .collect::<Vec<_>>();
        Self::from_internal_bins(internal_bins, internal_bin_length)
    }

    /// Builds an estimate from completed internal bins.
    pub fn from_internal_bins(
        internal_bins: Vec<f64>,
        internal_bin_length: usize,
    ) -> Result<Self, String> {
        if internal_bin_length == 0 {
            return Err("binsize must be positive".to_string());
        }
        if internal_bins.is_empty() {
            return Err("binsize is larger than the sample series".to_string());
        }
        let rebin_length = carlo_rebin_length(internal_bins.len());
        let rebin_usable = internal_bins.len() - internal_bins.len() % rebin_length;
        if rebin_usable == 0 {
            return Err("rebin length is larger than the internal bin series".to_string());
        }
        let bins = internal_bins[..rebin_usable]
            .chunks_exact(rebin_length)
            .map(mean)
            .collect::<Vec<_>>();
        Ok(Self {
            mean: mean(&bins),
            stderr: carlo_std_of_mean(&bins),
            bins,
            internal_bins,
            internal_bin_length,
            rebin_length,
        })
    }
}

/// The final per-observable estimate stored in a [`TaskResult`](crate::TaskResult).
///
/// The field set intentionally matches common downstream conventions (for example
/// Carlo.jl and the `SLSF-rs` theta model):
///
/// - `mean`/`stderr`: rebinned mean and its standard error.
/// - `error`: the reported error; equal to `stderr` for directly measured observables.
/// - `covariance`: reserved for array-valued observables; `None` for scalars.
/// - `autocorr_time`: reserved for integrated autocorrelation time; `0.0` when unknown.
/// - `bins`/`bin_length`/`rebin_len`/`rebin_count`/`internal_bin_len`: binning metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
    pub error: f64,
    pub covariance: Option<f64>,
    pub autocorr_time: f64,
    pub bins: usize,
    pub bin_length: usize,
    pub rebin_len: usize,
    pub rebin_count: usize,
    pub internal_bin_len: usize,
}

impl Estimate {
    /// Converts a [`BinnedEstimate`] into the final serializable estimate.
    pub fn from_binned(estimate: &BinnedEstimate, bin_length: usize) -> Self {
        Self {
            mean: estimate.mean,
            stderr: estimate.stderr,
            error: estimate.stderr,
            covariance: None,
            autocorr_time: 0.0,
            bins: estimate.bins.len(),
            bin_length,
            rebin_len: estimate.rebin_length,
            rebin_count: estimate.bins.len(),
            internal_bin_len: estimate.internal_bin_length,
        }
    }
}

impl From<&BinnedEstimate> for Estimate {
    fn from(estimate: &BinnedEstimate) -> Self {
        Self::from_binned(estimate, estimate.internal_bin_length)
    }
}

impl From<ScalarEstimate> for Estimate {
    /// Widens a plain rebinned [`ScalarEstimate`] into the rich estimate form.
    fn from(estimate: ScalarEstimate) -> Self {
        Self {
            mean: estimate.mean,
            stderr: estimate.stderr,
            error: estimate.stderr,
            covariance: None,
            autocorr_time: 0.0,
            bins: estimate.rebin_count,
            bin_length: estimate.bin_length,
            rebin_len: estimate.rebin_length,
            rebin_count: estimate.rebin_count,
            internal_bin_len: estimate.bin_length,
        }
    }
}

impl ResultEstimate for Estimate {
    fn valid(&self) -> bool {
        self.bins > 0
            && self.bins == self.rebin_count
            && self.bin_length > 0
            && self.bin_length == self.internal_bin_len
            && self.rebin_len > 0
            && self.rebin_count > 0
            && self.internal_bin_len > 0
            && !self.mean.is_infinite()
            && !self.stderr.is_infinite()
            && !self.error.is_infinite()
            && if self.rebin_count == 1 {
                self.stderr.is_nan()
            } else {
                self.stderr >= 0.0 && self.error >= 0.0
            }
    }
}

/// Builds the default per-observable [`Estimate`]s from measured raw bins.
///
/// `raw_bins` maps an observable name to its completed internal-bin averages, and
/// `bin_lengths` maps the same name to the number of raw samples per internal bin.
///
/// This is the default estimate strategy used by
/// [`MonteCarlo::finalize_estimates`](crate::MonteCarlo::finalize_estimates). Every
/// entry in `raw_bins` is expected to have at least one completed internal bin; the
/// runner guarantees this before finalization, so the panic branch indicates a framework
/// invariant violation rather than a user error.
pub fn default_estimates(
    raw_bins: &BTreeMap<String, Vec<f64>>,
    bin_lengths: &BTreeMap<String, usize>,
) -> BTreeMap<String, Estimate> {
    raw_bins
        .iter()
        .map(|(name, bins)| {
            let bin_length = *bin_lengths.get(name).unwrap_or(&1);
            let binned = BinnedEstimate::from_internal_bins(bins.clone(), bin_length)
                .expect("finalize received an empty internal-bin series");
            (name.clone(), Estimate::from_binned(&binned, bin_length))
        })
        .collect()
}

/// Computes derived observables from measured ones through jackknife resampling.
///
/// A model's [`MonteCarlo::finalize_estimates`](crate::MonteCarlo::finalize_estimates)
/// implementation constructs one of these from the measured bin averages and then
/// registers nonlinear observables (for example
/// `Magnetization = sqrt(MagnetizationSquared)`) in terms of measured ingredients.
/// The evaluator applies each registration to produce a jackknifed estimate of type `E`
/// (defaulting to [`Estimate`]).
pub struct Evaluator<'a, E = Estimate> {
    binned: BTreeMap<String, BinnedEstimate>,
    estimates: &'a mut BTreeMap<String, E>,
}

impl<'a, E> Evaluator<'a, E>
where
    E: ResultEstimate + From<Estimate>,
{
    /// Creates an evaluator from measured [`BinnedEstimate`]s and the estimate map
    /// that will receive the derived observables.
    pub fn new(
        binned: BTreeMap<String, BinnedEstimate>,
        estimates: &'a mut BTreeMap<String, E>,
    ) -> Self {
        Self { binned, estimates }
    }

    /// Registers the derived observable `name` as `evaluation(ingredients)`.
    ///
    /// If any ingredient is missing or has no rebinned bins, this is a no-op, so a
    /// model can register all of its derived observables unconditionally.
    pub fn evaluate<const N: usize, F>(&mut self, name: &str, ingredients: [&str; N], evaluation: F)
    where
        F: Fn([f64; N]) -> f64,
    {
        if let Some(estimate) = evaluate(&self.binned, ingredients, evaluation) {
            self.estimates.insert(name.to_string(), E::from(estimate));
        }
    }
}

fn evaluate<const N: usize, F>(
    binned: &BTreeMap<String, BinnedEstimate>,
    ingredients: [&str; N],
    evaluation: F,
) -> Option<Estimate>
where
    F: Fn([f64; N]) -> f64,
{
    let used = ingredients
        .iter()
        .map(|name| binned.get(*name))
        .collect::<Option<Vec<_>>>()?;
    let internal_bin_length = used
        .iter()
        .map(|estimate| estimate.internal_bin_length)
        .min()
        .unwrap_or(1);
    let rebin_length = used
        .iter()
        .map(|estimate| estimate.rebin_length)
        .min()
        .unwrap_or(1);
    let rebin_count = used
        .iter()
        .map(|estimate| estimate.bins.len())
        .min()
        .unwrap_or(0);
    if rebin_count == 0 {
        return None;
    }
    let jackknifed = jackknife_evaluate(&used, rebin_count, evaluation);
    let estimate = BinnedEstimate {
        mean: jackknifed.mean,
        stderr: jackknifed.stderr,
        bins: jackknifed.jacked_evals,
        internal_bins: Vec::new(),
        internal_bin_length,
        rebin_length,
    };
    Some(Estimate::from_binned(&estimate, internal_bin_length))
}

struct JackknifeEstimate {
    mean: f64,
    stderr: f64,
    jacked_evals: Vec<f64>,
}

fn jackknife_evaluate<const N: usize, F>(
    sample_set: &[&BinnedEstimate],
    sample_count: usize,
    evaluation: F,
) -> JackknifeEstimate
where
    F: Fn([f64; N]) -> f64,
{
    let sums = std::array::from_fn::<_, N, _>(|index| {
        sample_set[index].bins[..sample_count].iter().sum::<f64>()
    });
    let complete_eval = evaluation(std::array::from_fn(|index| {
        sums[index] / sample_count as f64
    }));
    if sample_count <= 1 {
        return JackknifeEstimate {
            mean: complete_eval,
            stderr: f64::NAN,
            jacked_evals: vec![complete_eval],
        };
    }

    let jacked_evals = (0..sample_count)
        .map(|sample_index| {
            evaluation(std::array::from_fn(|index| {
                (sums[index] - sample_set[index].bins[sample_index]) / (sample_count - 1) as f64
            }))
        })
        .collect::<Vec<_>>();
    let jacked_mean = mean(&jacked_evals);
    let bias_corrected_mean =
        sample_count as f64 * complete_eval - (sample_count - 1) as f64 * jacked_mean;
    let error = jacked_evals
        .iter()
        .map(|value| (value - jacked_mean).powi(2))
        .sum::<f64>();
    let stderr = (((sample_count - 1) as f64) * error / sample_count as f64).sqrt();
    JackknifeEstimate {
        mean: bias_corrected_mean,
        stderr,
        jacked_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebin_matches_carlo_convention() {
        assert_eq!(carlo_rebin_count(0), 1);
        assert_eq!(carlo_rebin_count(10), 10);
        assert_eq!(carlo_rebin_length(64), carlo_rebin_length(64));
        assert_eq!(
            BinnedEstimate::from_internal_bins(vec![1.0], 1)
                .unwrap()
                .rebin_length,
            1
        );
    }

    #[test]
    fn derived_observables_use_jackknife() {
        let energy = BinnedEstimate::from_samples(&[1.0, 3.0, 5.0, 7.0], 2).unwrap();
        let energy2 = BinnedEstimate::from_samples(&[1.0, 9.0, 25.0, 49.0], 2).unwrap();
        let mag2 = BinnedEstimate::from_samples(&[0.25, 0.49, 0.81, 1.0], 2).unwrap();
        let binned = BTreeMap::from([
            ("Energy".to_string(), energy),
            ("EnergySquared".to_string(), energy2),
            ("MagnetizationSquared".to_string(), mag2),
        ]);
        let mut estimates = binned
            .iter()
            .map(|(name, estimate)| {
                (
                    name.clone(),
                    Estimate::from_binned(estimate, estimate.internal_bin_length),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut evaluator = Evaluator::new(binned, &mut estimates);
        evaluator.evaluate("Magnetization", ["MagnetizationSquared"], |[mag2]| {
            mag2.sqrt()
        });
        evaluator.evaluate("Chi", ["MagnetizationSquared"], |[mag2]| 2.0 * 8.0 * mag2);
        evaluator.evaluate(
            "SpecificHeat",
            ["EnergySquared", "Energy"],
            |[energy2, energy]| 2.0 * 2.0 * 8.0 * (energy2 - energy * energy),
        );
        assert!((estimates["Magnetization"].mean - 0.817076375991209).abs() < 1e-12);
        assert!((estimates["Chi"].mean - 10.2).abs() < 1e-12);
        assert!((estimates["SpecificHeat"].mean - 288.0).abs() < 1e-12);
    }
}
