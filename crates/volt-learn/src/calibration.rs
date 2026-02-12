//! Certainty calibration metric for RLVF.
//!
//! Measures whether the model's certainty (gamma) is well-calibrated:
//! "When Volt says gamma=0.9, is it correct 90% of the time?"
//!
//! ## Expected Calibration Error (ECE)
//!
//! Predictions are binned by gamma value into 10 equal-width bins.
//! For each bin, the actual accuracy is compared to the mean predicted
//! gamma. A perfectly calibrated model has ECE = 0.0.
//!
//! ECE = Σ (|accuracy_i - mean_gamma_i| × n_i / n_total)

use crate::reward::RewardOutcome;

/// Number of calibration bins.
const NUM_BINS: usize = 10;

/// A single calibration bin grouping predictions by gamma range.
///
/// # Example
///
/// ```
/// use volt_learn::calibration::CalibrationBin;
///
/// let bin = CalibrationBin {
///     bin_start: 0.0,
///     bin_end: 0.1,
///     count: 10,
///     mean_gamma: 0.05,
///     accuracy: 0.03,
/// };
/// assert_eq!(bin.count, 10);
/// ```
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    /// Lower bound of this bin (inclusive).
    pub bin_start: f32,
    /// Upper bound of this bin (exclusive, except for the last bin).
    pub bin_end: f32,
    /// Number of samples in this bin.
    pub count: usize,
    /// Mean gamma of all samples in this bin.
    pub mean_gamma: f32,
    /// Fraction of correct predictions in this bin.
    pub accuracy: f32,
}

/// Result of a calibration computation.
///
/// # Example
///
/// ```
/// use volt_learn::calibration::CalibrationResult;
///
/// let result = CalibrationResult {
///     bins: vec![],
///     ece: 0.0,
///     total_samples: 0,
/// };
/// assert_eq!(result.ece, 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// The 10 calibration bins.
    pub bins: Vec<CalibrationBin>,
    /// Expected Calibration Error (lower is better, 0.0 = perfect).
    pub ece: f32,
    /// Total number of samples used.
    pub total_samples: usize,
}

/// Computes the certainty calibration from a set of reward outcomes.
///
/// Bins all outcomes by their gamma value into 10 equal-width bins
/// (0.0–0.1, 0.1–0.2, ..., 0.9–1.0). For each bin, computes the
/// mean gamma and actual accuracy (fraction correct). The Expected
/// Calibration Error (ECE) is the weighted average of
/// |accuracy - mean_gamma| across all bins.
///
/// # Example
///
/// ```
/// use volt_learn::calibration::compute_calibration;
/// use volt_learn::reward::RewardOutcome;
///
/// let outcomes = vec![
///     RewardOutcome { reward: 1.0, is_correct: true, correctness: 0.9, gamma: 0.9 },
///     RewardOutcome { reward: -2.0, is_correct: false, correctness: 0.1, gamma: 0.9 },
/// ];
/// let result = compute_calibration(&outcomes);
/// assert_eq!(result.total_samples, 2);
/// assert!(result.ece >= 0.0);
/// ```
pub fn compute_calibration(outcomes: &[RewardOutcome]) -> CalibrationResult {
    if outcomes.is_empty() {
        let bins = (0..NUM_BINS)
            .map(|i| {
                let start = i as f32 * 0.1;
                let end = (i + 1) as f32 * 0.1;
                CalibrationBin {
                    bin_start: start,
                    bin_end: end,
                    count: 0,
                    mean_gamma: 0.0,
                    accuracy: 0.0,
                }
            })
            .collect();
        return CalibrationResult {
            bins,
            ece: 0.0,
            total_samples: 0,
        };
    }

    // Accumulate per-bin counts and sums
    let mut bin_gamma_sum = [0.0f32; NUM_BINS];
    let mut bin_correct_count = [0usize; NUM_BINS];
    let mut bin_total_count = [0usize; NUM_BINS];

    for outcome in outcomes {
        let bin_idx = gamma_to_bin(outcome.gamma);
        bin_gamma_sum[bin_idx] += outcome.gamma;
        if outcome.is_correct {
            bin_correct_count[bin_idx] += 1;
        }
        bin_total_count[bin_idx] += 1;
    }

    let total = outcomes.len();
    let mut ece = 0.0f32;
    let mut bins = Vec::with_capacity(NUM_BINS);

    for i in 0..NUM_BINS {
        let start = i as f32 * 0.1;
        let end = (i + 1) as f32 * 0.1;
        let count = bin_total_count[i];

        let (mean_gamma, accuracy) = if count > 0 {
            let mg = bin_gamma_sum[i] / count as f32;
            let acc = bin_correct_count[i] as f32 / count as f32;
            ece += (acc - mg).abs() * count as f32 / total as f32;
            (mg, acc)
        } else {
            (0.0, 0.0)
        };

        bins.push(CalibrationBin {
            bin_start: start,
            bin_end: end,
            count,
            mean_gamma,
            accuracy,
        });
    }

    CalibrationResult {
        bins,
        ece,
        total_samples: total,
    }
}

/// Maps a gamma value to a bin index (0–9).
fn gamma_to_bin(gamma: f32) -> usize {
    let clamped = gamma.clamp(0.0, 1.0);
    let idx = (clamped * NUM_BINS as f32) as usize;
    // Last bin includes 1.0
    idx.min(NUM_BINS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reward::RewardOutcome;

    fn make_outcome(correct: bool, gamma: f32) -> RewardOutcome {
        RewardOutcome {
            reward: if correct { 1.0 } else { -1.0 },
            is_correct: correct,
            correctness: if correct { 0.9 } else { 0.1 },
            gamma,
        }
    }

    #[test]
    fn empty_outcomes_zero_ece() {
        let result = compute_calibration(&[]);
        assert_eq!(result.total_samples, 0);
        assert!((result.ece - 0.0).abs() < f32::EPSILON);
        assert_eq!(result.bins.len(), NUM_BINS);
    }

    #[test]
    fn perfect_calibration_low_ece() {
        // All correct predictions at gamma=0.95 -> bin 9 should have
        // accuracy=1.0, mean_gamma≈0.95, difference = 0.05
        let outcomes: Vec<_> = (0..100)
            .map(|_| make_outcome(true, 0.95))
            .collect();
        let result = compute_calibration(&outcomes);
        assert_eq!(result.total_samples, 100);
        // ECE should be small (0.05 since all samples are correct at 0.95)
        assert!(result.ece < 0.1, "ECE should be small: {}", result.ece);
    }

    #[test]
    fn overconfident_model_high_ece() {
        // All wrong predictions at gamma=0.9 -> accuracy=0.0, gamma=0.9
        // ECE should be 0.9 (|0.0 - 0.9|)
        let outcomes: Vec<_> = (0..100)
            .map(|_| make_outcome(false, 0.9))
            .collect();
        let result = compute_calibration(&outcomes);
        assert!(result.ece > 0.5, "ECE should be high: {}", result.ece);
    }

    #[test]
    fn bins_cover_full_range() {
        let result = compute_calibration(&[]);
        assert_eq!(result.bins.len(), 10);
        assert!((result.bins[0].bin_start - 0.0).abs() < f32::EPSILON);
        assert!((result.bins[9].bin_end - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn samples_assigned_to_correct_bins() {
        let outcomes = vec![
            make_outcome(true, 0.05),  // bin 0
            make_outcome(true, 0.55),  // bin 5
            make_outcome(false, 0.95), // bin 9
        ];
        let result = compute_calibration(&outcomes);
        assert_eq!(result.bins[0].count, 1);
        assert_eq!(result.bins[5].count, 1);
        assert_eq!(result.bins[9].count, 1);
        // All other bins should be empty
        for i in [1, 2, 3, 4, 6, 7, 8] {
            assert_eq!(result.bins[i].count, 0);
        }
    }

    #[test]
    fn gamma_at_boundary() {
        // gamma=1.0 should go to bin 9
        assert_eq!(gamma_to_bin(1.0), 9);
        // gamma=0.0 should go to bin 0
        assert_eq!(gamma_to_bin(0.0), 0);
        // gamma=0.1 should go to bin 1
        assert_eq!(gamma_to_bin(0.1), 1);
    }

    #[test]
    fn gamma_out_of_range_clamped() {
        assert_eq!(gamma_to_bin(-0.5), 0);
        assert_eq!(gamma_to_bin(1.5), 9);
    }

    #[test]
    fn accuracy_per_bin() {
        // 3 correct, 2 wrong in same bin (gamma 0.75-0.79)
        let outcomes = vec![
            make_outcome(true, 0.75),
            make_outcome(true, 0.76),
            make_outcome(true, 0.77),
            make_outcome(false, 0.78),
            make_outcome(false, 0.79),
        ];
        let result = compute_calibration(&outcomes);
        let bin7 = &result.bins[7];
        assert_eq!(bin7.count, 5);
        assert!((bin7.accuracy - 0.6).abs() < 1e-5);
    }

    #[test]
    fn ece_is_non_negative() {
        let outcomes: Vec<_> = (0..50)
            .map(|i| make_outcome(i % 3 == 0, i as f32 / 50.0))
            .collect();
        let result = compute_calibration(&outcomes);
        assert!(result.ece >= 0.0);
    }
}
