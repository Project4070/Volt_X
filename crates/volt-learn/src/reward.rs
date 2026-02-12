//! Reward computation for RLVF joint alignment.
//!
//! Computes rewards based on the correctness of VFN output compared to
//! a verified reference answer, combined with the calibration quality
//! of the model's certainty (gamma).
//!
//! ## Reward Shaping
//!
//! | Correctness | Gamma    | Reward | Rationale                    |
//! |-------------|----------|--------|------------------------------|
//! | Correct     | High     | +1.0   | Calibrated confidence        |
//! | Correct     | Low      | +0.5   | Correct but under-confident  |
//! | Wrong       | Low      | +0.2   | Honest uncertainty           |
//! | Wrong       | High     | -2.0   | Overconfident error           |
//! | Wrong       | Mid      | -0.5   | Moderate error               |

use volt_core::{TensorFrame, SLOT_DIM};

/// Configuration for reward computation.
///
/// # Example
///
/// ```
/// use volt_learn::reward::RewardConfig;
///
/// let config = RewardConfig::default();
/// assert!(config.correctness_threshold > 0.0);
/// assert!(config.overconfident_gamma > config.uncertain_gamma);
/// ```
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Cosine similarity threshold above which an output is "correct".
    /// Default: 0.5.
    pub correctness_threshold: f32,
    /// Gamma above which the model is considered "confident". Default: 0.7.
    pub overconfident_gamma: f32,
    /// Gamma below which the model is considered "uncertain". Default: 0.3.
    pub uncertain_gamma: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            correctness_threshold: 0.5,
            overconfident_gamma: 0.7,
            uncertain_gamma: 0.3,
        }
    }
}

/// The outcome of reward computation for a single evaluation sample.
///
/// # Example
///
/// ```
/// use volt_learn::reward::RewardOutcome;
///
/// let outcome = RewardOutcome {
///     reward: 1.0,
///     is_correct: true,
///     correctness: 0.85,
///     gamma: 0.9,
/// };
/// assert!(outcome.is_correct);
/// assert!(outcome.reward > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RewardOutcome {
    /// The computed reward value.
    pub reward: f32,
    /// Whether the output was considered correct.
    pub is_correct: bool,
    /// The cosine similarity between output and reference (0.0 to 1.0).
    pub correctness: f32,
    /// The gamma (certainty) value for this output.
    pub gamma: f32,
}

/// Computes a shaped reward from correctness and gamma values.
///
/// The reward function penalizes overconfident errors heavily (-2.0)
/// while mildly rewarding honest uncertainty (+0.2). Correct outputs
/// receive positive rewards scaled by confidence calibration.
///
/// # Example
///
/// ```
/// use volt_learn::reward::{compute_reward, RewardConfig};
///
/// let config = RewardConfig::default();
///
/// // Correct + confident -> high reward
/// let outcome = compute_reward(0.8, 0.9, &config);
/// assert!(outcome.reward > 0.0);
/// assert!(outcome.is_correct);
///
/// // Wrong + overconfident -> strong negative reward
/// let outcome = compute_reward(0.1, 0.9, &config);
/// assert!(outcome.reward < -1.0);
/// assert!(!outcome.is_correct);
/// ```
pub fn compute_reward(correctness: f32, gamma: f32, config: &RewardConfig) -> RewardOutcome {
    let is_correct = correctness >= config.correctness_threshold;

    let reward = if is_correct {
        if gamma >= config.overconfident_gamma {
            // Correct + confident = calibrated confidence
            1.0
        } else {
            // Correct + uncertain = under-confident
            0.5
        }
    } else if gamma >= config.overconfident_gamma {
        // Wrong + confident = overconfident error (strong penalty)
        -2.0
    } else if gamma <= config.uncertain_gamma {
        // Wrong + uncertain = honest uncertainty (mild positive)
        0.2
    } else {
        // Wrong + moderate confidence = moderate error
        -0.5
    };

    RewardOutcome {
        reward,
        is_correct,
        correctness,
        gamma,
    }
}

/// Computes average cosine similarity between matching active R₀ slots
/// of two frames.
///
/// Only slots that are active in both frames contribute to the score.
/// Returns 0.0 if no slots overlap.
///
/// # Example
///
/// ```
/// use volt_learn::reward::slot_cosine_similarity;
/// use volt_core::TensorFrame;
///
/// let f1 = TensorFrame::new();
/// let f2 = TensorFrame::new();
/// let sim = slot_cosine_similarity(&f1, &f2);
/// assert!(sim >= 0.0);
/// ```
pub fn slot_cosine_similarity(output: &TensorFrame, reference: &TensorFrame) -> f32 {
    let mut total_sim = 0.0f32;
    let mut count = 0usize;

    for i in 0..output.slots.len() {
        if let Some(out_slot) = &output.slots[i]
            && let Some(ref_slot) = &reference.slots[i]
            && let Some(out_r0) = &out_slot.resolutions[0]
            && let Some(ref_r0) = &ref_slot.resolutions[0]
        {
            let sim = cosine_similarity_256(out_r0, ref_r0);
            total_sim += sim;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    total_sim / count as f32
}

/// Cosine similarity between two 256-dim vectors.
fn cosine_similarity_256(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::slot::{SlotData, SlotRole};

    #[test]
    fn default_config_sensible() {
        let config = RewardConfig::default();
        assert!((config.correctness_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.overconfident_gamma - 0.7).abs() < f32::EPSILON);
        assert!((config.uncertain_gamma - 0.3).abs() < f32::EPSILON);
        assert!(config.overconfident_gamma > config.uncertain_gamma);
    }

    #[test]
    fn correct_confident_positive_reward() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.8, 0.9, &config);
        assert!(outcome.is_correct);
        assert!((outcome.reward - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn correct_uncertain_mild_positive() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.8, 0.2, &config);
        assert!(outcome.is_correct);
        assert!((outcome.reward - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn wrong_overconfident_strong_negative() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.1, 0.9, &config);
        assert!(!outcome.is_correct);
        assert!((outcome.reward - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn wrong_uncertain_honest_mild_positive() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.1, 0.2, &config);
        assert!(!outcome.is_correct);
        assert!((outcome.reward - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn wrong_moderate_confidence_negative() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.1, 0.5, &config);
        assert!(!outcome.is_correct);
        assert!((outcome.reward - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn outcome_preserves_inputs() {
        let config = RewardConfig::default();
        let outcome = compute_reward(0.75, 0.85, &config);
        assert!((outcome.correctness - 0.75).abs() < f32::EPSILON);
        assert!((outcome.gamma - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn slot_similarity_empty_frames() {
        let f1 = TensorFrame::new();
        let f2 = TensorFrame::new();
        let sim = slot_cosine_similarity(&f1, &f2);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn slot_similarity_identical_frames() {
        let mut f1 = TensorFrame::new();
        let mut f2 = TensorFrame::new();

        let mut r0 = [0.0f32; SLOT_DIM];
        r0[0] = 1.0;
        r0[1] = 0.5;

        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, r0);
        f1.slots[0] = Some(slot.clone());
        f2.slots[0] = Some(slot);

        let sim = slot_cosine_similarity(&f1, &f2);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn slot_similarity_orthogonal_frames() {
        let mut f1 = TensorFrame::new();
        let mut f2 = TensorFrame::new();

        let mut r0_a = [0.0f32; SLOT_DIM];
        r0_a[0] = 1.0;
        let mut r0_b = [0.0f32; SLOT_DIM];
        r0_b[1] = 1.0;

        let mut slot_a = SlotData::new(SlotRole::Agent);
        slot_a.write_resolution(0, r0_a);
        let mut slot_b = SlotData::new(SlotRole::Agent);
        slot_b.write_resolution(0, r0_b);
        f1.slots[0] = Some(slot_a);
        f2.slots[0] = Some(slot_b);

        let sim = slot_cosine_similarity(&f1, &f2);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = [0.0f32; SLOT_DIM];
        let mut b = [0.0f32; SLOT_DIM];
        b[0] = 1.0;
        let sim = cosine_similarity_256(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn boundary_correctness_threshold() {
        let config = RewardConfig::default();
        // Exactly at threshold
        let outcome = compute_reward(0.5, 0.9, &config);
        assert!(outcome.is_correct);
        // Just below threshold
        let outcome = compute_reward(0.499, 0.9, &config);
        assert!(!outcome.is_correct);
    }
}
