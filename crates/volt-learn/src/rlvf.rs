//! RLVF (Reinforcement Learning from Verified Feedback) training loop.
//!
//! Implements REINFORCE with baseline for joint alignment of the VFN.
//! The loop evaluates the VFN on (question, verified_answer) pairs,
//! computes rewards based on correctness and gamma calibration, then
//! applies policy gradient updates layer-by-layer.
//!
//! ## Algorithm
//!
//! 1. Encode question → TensorFrame via translator
//! 2. Run VFN forward on each active R₀ slot → drift-modified output
//! 3. Encode answer → reference TensorFrame
//! 4. Compute cosine similarity → correctness
//! 5. Compute reward from correctness + gamma calibration
//! 6. REINFORCE: advantage = reward - baseline
//! 7. Layer-local weight updates (no backprop)
//!
//! ## Relation to Forward-Forward
//!
//! RLVF reuses the Forward-Forward gradient machinery but replaces
//! the goodness-threshold signal with a reward-based advantage signal.
//! Positive advantage → treat as positive sample (push goodness up).
//! Negative advantage → treat as negative sample (push goodness down).

use volt_core::{VoltError, SLOT_DIM};
use volt_soft::vfn::Vfn;
use volt_translate::StubTranslator;
use volt_translate::Translator;

use crate::calibration::{self, CalibrationResult};
use crate::eval_dataset::EvalPair;
use crate::reward::{self, RewardConfig, RewardOutcome};
use crate::self_play;

/// Configuration for RLVF training.
///
/// # Example
///
/// ```
/// use volt_learn::rlvf::RlvfConfig;
///
/// let config = RlvfConfig::default();
/// assert_eq!(config.num_epochs, 3);
/// assert!(config.learning_rate > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RlvfConfig {
    /// Learning rate for weight updates. Default: 0.0005.
    pub learning_rate: f32,
    /// Number of training epochs over the eval dataset. Default: 3.
    pub num_epochs: usize,
    /// Exponential moving average decay for baseline. Default: 0.9.
    pub baseline_decay: f32,
    /// Reward computation configuration.
    pub reward_config: RewardConfig,
    /// Random seed for reproducibility. Default: 42.
    pub seed: u64,
    /// Number of self-play puzzles to evaluate. Default: 100.
    pub puzzle_count: usize,
    /// Similarity threshold for grading puzzles. Default: 0.3.
    pub puzzle_threshold: f32,
}

impl Default for RlvfConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.0005,
            num_epochs: 3,
            baseline_decay: 0.9,
            reward_config: RewardConfig::default(),
            seed: 42,
            puzzle_count: 100,
            puzzle_threshold: 0.3,
        }
    }
}

/// Result of an RLVF training run.
///
/// # Example
///
/// ```
/// use volt_learn::rlvf::RlvfResult;
/// use volt_learn::calibration::CalibrationResult;
///
/// let result = RlvfResult {
///     epochs_completed: 3,
///     mean_reward_before: -0.5,
///     mean_reward_after: 0.2,
///     calibration_before: CalibrationResult { bins: vec![], ece: 0.5, total_samples: 0 },
///     calibration_after: CalibrationResult { bins: vec![], ece: 0.3, total_samples: 0 },
///     puzzles_correct_before: 10,
///     puzzles_correct_after: 15,
///     total_puzzles: 100,
/// };
/// assert_eq!(result.epochs_completed, 3);
/// ```
#[derive(Debug, Clone)]
pub struct RlvfResult {
    /// Number of training epochs completed.
    pub epochs_completed: usize,
    /// Mean reward across eval pairs before training.
    pub mean_reward_before: f32,
    /// Mean reward across eval pairs after training.
    pub mean_reward_after: f32,
    /// Certainty calibration before training.
    pub calibration_before: CalibrationResult,
    /// Certainty calibration after training.
    pub calibration_after: CalibrationResult,
    /// Number of logic puzzles solved before training.
    pub puzzles_correct_before: usize,
    /// Number of logic puzzles solved after training.
    pub puzzles_correct_after: usize,
    /// Total number of logic puzzles evaluated.
    pub total_puzzles: usize,
}

/// An intermediate sample produced from one eval pair.
struct RlvfSample {
    /// The R₀ slot embedding from the question frame.
    embedding: [f32; SLOT_DIM],
    /// The reward-based advantage (reward - baseline).
    advantage: f32,
}

/// Trains the VFN using RLVF (REINFORCE with baseline).
///
/// Evaluates the VFN on the provided evaluation pairs, computes
/// rewards from correctness + gamma calibration, and updates VFN
/// weights layer-by-layer using the advantage signal.
///
/// Also evaluates self-play logic puzzles before and after training
/// to measure reasoning improvement.
///
/// # Errors
///
/// Returns [`VoltError::LearnError`] if no valid samples can be
/// extracted from the evaluation pairs.
///
/// # Example
///
/// ```no_run
/// use volt_learn::rlvf::{train_rlvf, RlvfConfig};
/// use volt_learn::eval_dataset::generate_eval_dataset;
/// use volt_soft::vfn::Vfn;
/// use volt_translate::StubTranslator;
///
/// let mut vfn = Vfn::new_random(42);
/// let translator = StubTranslator::new();
/// let dataset = generate_eval_dataset();
/// let config = RlvfConfig { num_epochs: 1, ..RlvfConfig::default() };
/// let result = train_rlvf(&mut vfn, &dataset[..10], &translator, &config).unwrap();
/// assert_eq!(result.epochs_completed, 1);
/// ```
pub fn train_rlvf(
    vfn: &mut Vfn,
    eval_pairs: &[EvalPair],
    translator: &StubTranslator,
    config: &RlvfConfig,
) -> Result<RlvfResult, VoltError> {
    if eval_pairs.is_empty() {
        return Err(VoltError::LearnError {
            message: "train_rlvf: no evaluation pairs provided".to_string(),
        });
    }

    // Measure before-training metrics
    let outcomes_before = evaluate_all(vfn, eval_pairs, translator, &config.reward_config)?;
    let calibration_before = calibration::compute_calibration(&outcomes_before);
    let mean_reward_before = mean_reward(&outcomes_before);

    let puzzles = self_play::generate_puzzles(config.puzzle_count, config.seed);
    let puzzles_correct_before = count_puzzle_correct(
        vfn,
        translator,
        &puzzles,
        config.puzzle_threshold,
    );

    // RLVF training loop
    let mut baseline = mean_reward_before;

    for _epoch in 0..config.num_epochs {
        let outcomes = evaluate_all(vfn, eval_pairs, translator, &config.reward_config)?;

        // Collect samples with advantages
        let samples = collect_rlvf_samples(
            eval_pairs,
            &outcomes,
            translator,
            baseline,
        )?;

        if samples.is_empty() {
            continue;
        }

        // Update baseline with EMA
        let epoch_mean = mean_reward(&outcomes);
        baseline = config.baseline_decay * baseline
            + (1.0 - config.baseline_decay) * epoch_mean;

        // Apply REINFORCE updates layer-by-layer
        apply_reinforce_updates(vfn, &samples, config.learning_rate)?;
    }

    // Measure after-training metrics
    let outcomes_after = evaluate_all(vfn, eval_pairs, translator, &config.reward_config)?;
    let calibration_after = calibration::compute_calibration(&outcomes_after);
    let mean_reward_after = mean_reward(&outcomes_after);

    let puzzles_correct_after = count_puzzle_correct(
        vfn,
        translator,
        &puzzles,
        config.puzzle_threshold,
    );

    Ok(RlvfResult {
        epochs_completed: config.num_epochs,
        mean_reward_before,
        mean_reward_after,
        calibration_before,
        calibration_after,
        puzzles_correct_before,
        puzzles_correct_after,
        total_puzzles: config.puzzle_count,
    })
}

/// Evaluates the VFN on all eval pairs and returns reward outcomes.
fn evaluate_all(
    vfn: &Vfn,
    eval_pairs: &[EvalPair],
    translator: &StubTranslator,
    reward_config: &RewardConfig,
) -> Result<Vec<RewardOutcome>, VoltError> {
    let mut outcomes = Vec::with_capacity(eval_pairs.len());

    for pair in eval_pairs {
        let question_frame = translator.encode(&pair.question)?;
        let answer_frame = translator.encode(&pair.answer)?;

        // Run VFN forward on each active R₀ slot of the question
        let mut output_frame = question_frame.frame.clone();
        for slot_idx in 0..output_frame.slots.len() {
            if let Some(slot) = &mut output_frame.slots[slot_idx]
                && let Some(r0) = &slot.resolutions[0]
            {
                let drift = vfn.forward(r0)?;
                // Apply drift: output = input + drift (bounded)
                let mut updated = [0.0f32; SLOT_DIM];
                for k in 0..SLOT_DIM {
                    updated[k] = r0[k] + drift[k] * 0.1; // Damped drift
                }
                // L2-normalize
                let norm: f32 = updated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for v in &mut updated {
                        *v /= norm;
                    }
                }
                slot.resolutions[0] = Some(updated);
            }
        }

        // Compute correctness as cosine similarity between output and answer
        let correctness = reward::slot_cosine_similarity(
            &output_frame,
            &answer_frame.frame,
        );

        // Use average gamma from the question frame's meta
        let active_gammas: Vec<f32> = question_frame
            .frame
            .meta
            .iter()
            .map(|m| m.certainty)
            .filter(|&g| g > 0.0)
            .collect();
        let gamma = if active_gammas.is_empty() {
            0.5 // Default gamma for frames without certainty
        } else {
            active_gammas.iter().sum::<f32>() / active_gammas.len() as f32
        };

        let outcome = reward::compute_reward(correctness, gamma, reward_config);
        outcomes.push(outcome);
    }

    Ok(outcomes)
}

/// Collects RLVF samples with advantage signals from eval pairs.
fn collect_rlvf_samples(
    eval_pairs: &[EvalPair],
    outcomes: &[RewardOutcome],
    translator: &StubTranslator,
    baseline: f32,
) -> Result<Vec<RlvfSample>, VoltError> {
    let mut samples = Vec::new();

    for (pair, outcome) in eval_pairs.iter().zip(outcomes.iter()) {
        let advantage = outcome.reward - baseline;
        if advantage.abs() < 1e-8 {
            continue; // No useful gradient signal
        }

        let question_output = translator.encode(&pair.question)?;

        // Extract R₀ embeddings from active slots
        for slot_opt in &question_output.frame.slots {
            if let Some(slot) = slot_opt
                && let Some(r0) = &slot.resolutions[0]
            {
                samples.push(RlvfSample {
                    embedding: *r0,
                    advantage,
                });
            }
        }
    }

    Ok(samples)
}

/// Applies REINFORCE gradient updates to the VFN layer-by-layer.
///
/// Uses the same gradient computation as Forward-Forward but scales
/// by the advantage signal instead of a binary positive/negative label.
fn apply_reinforce_updates(
    vfn: &mut Vfn,
    samples: &[RlvfSample],
    learning_rate: f32,
) -> Result<(), VoltError> {
    let n_layers = vfn.layer_count();

    for layer_idx in 0..n_layers {
        let prev_layers: Vec<usize> = (0..layer_idx).collect();
        let (in_dim, out_dim) = vfn.layer_shape(layer_idx)?;

        for sample in samples {
            // Forward through previous layers (detached)
            let mut input: Vec<f32> = sample.embedding.to_vec();
            for &prev in &prev_layers {
                input = vfn.forward_layer(prev, &input)?;
            }

            // Forward through current layer
            let activations = vfn.forward_layer(layer_idx, &input)?;

            // Compute gradients of goodness w.r.t. weights
            // goodness = Σ a_i²
            // d(goodness)/d(w_ij) = 2 * a_i * x_j * relu_mask_i
            let mut weight_deltas = vec![0.0f32; in_dim * out_dim];
            let mut bias_deltas = vec![0.0f32; out_dim];

            // Sign: positive advantage → increase goodness (like positive FF sample)
            //        negative advantage → decrease goodness (like negative FF sample)
            let sign = sample.advantage.signum();
            let magnitude = sample.advantage.abs().min(2.0); // Clamp magnitude

            for i in 0..out_dim {
                // ReLU mask: hidden layers require activation > 0
                let relu_active = layer_idx >= 2 || activations[i] > 0.0;
                if !relu_active {
                    continue;
                }

                let grad_factor = sign * magnitude * 2.0 * activations[i];

                let row_start = i * in_dim;
                for j in 0..in_dim {
                    weight_deltas[row_start + j] = grad_factor * input[j];
                }
                bias_deltas[i] = grad_factor;
            }

            vfn.update_layer(
                layer_idx,
                &weight_deltas,
                &bias_deltas,
                learning_rate,
            )?;
        }
    }

    Ok(())
}

/// Computes mean reward from a slice of outcomes.
fn mean_reward(outcomes: &[RewardOutcome]) -> f32 {
    if outcomes.is_empty() {
        return 0.0;
    }
    outcomes.iter().map(|o| o.reward).sum::<f32>() / outcomes.len() as f32
}

/// Counts how many logic puzzles the VFN solves correctly.
fn count_puzzle_correct(
    vfn: &Vfn,
    translator: &StubTranslator,
    puzzles: &[self_play::LogicPuzzle],
    threshold: f32,
) -> usize {
    let mut correct = 0;

    for puzzle in puzzles {
        let premise_frame = match translator.encode(&puzzle.premises) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let conclusion_frame = match translator.encode(&puzzle.conclusion) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Apply VFN to premise frame
        let mut output = premise_frame.frame.clone();
        for slot_idx in 0..output.slots.len() {
            if let Some(slot) = &mut output.slots[slot_idx]
                && let Some(r0) = &slot.resolutions[0]
                && let Ok(drift) = vfn.forward(r0)
            {
                let mut updated = [0.0f32; SLOT_DIM];
                for k in 0..SLOT_DIM {
                    updated[k] = r0[k] + drift[k] * 0.1;
                }
                let norm: f32 = updated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for v in &mut updated {
                        *v /= norm;
                    }
                }
                slot.resolutions[0] = Some(updated);
            }
        }

        if self_play::grade_puzzle(&output, &conclusion_frame.frame, threshold) {
            correct += 1;
        }
    }

    correct
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_dataset::generate_eval_dataset;

    #[test]
    fn default_config_sensible() {
        let config = RlvfConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.num_epochs > 0);
        assert!(config.baseline_decay > 0.0);
        assert!(config.baseline_decay < 1.0);
        assert!(config.puzzle_count > 0);
    }

    #[test]
    fn train_rlvf_empty_pairs_errors() {
        let mut vfn = Vfn::new_random(42);
        let translator = StubTranslator::new();
        let config = RlvfConfig::default();
        assert!(train_rlvf(&mut vfn, &[], &translator, &config).is_err());
    }

    #[test]
    fn train_rlvf_single_epoch() {
        let mut vfn = Vfn::new_random(42);
        let translator = StubTranslator::new();
        let dataset = generate_eval_dataset();
        let config = RlvfConfig {
            num_epochs: 1,
            puzzle_count: 10,
            ..RlvfConfig::default()
        };
        let result = train_rlvf(&mut vfn, &dataset[..20], &translator, &config).unwrap();
        assert_eq!(result.epochs_completed, 1);
        assert_eq!(result.total_puzzles, 10);
    }

    #[test]
    fn train_rlvf_produces_calibration() {
        let mut vfn = Vfn::new_random(42);
        let translator = StubTranslator::new();
        let dataset = generate_eval_dataset();
        let config = RlvfConfig {
            num_epochs: 1,
            puzzle_count: 5,
            ..RlvfConfig::default()
        };
        let result = train_rlvf(&mut vfn, &dataset[..10], &translator, &config).unwrap();
        assert!(result.calibration_before.ece >= 0.0);
        assert!(result.calibration_after.ece >= 0.0);
        assert_eq!(result.calibration_before.total_samples, 10);
        assert_eq!(result.calibration_after.total_samples, 10);
    }

    #[test]
    fn evaluate_all_returns_outcomes() {
        let vfn = Vfn::new_random(42);
        let translator = StubTranslator::new();
        let dataset = generate_eval_dataset();
        let config = RewardConfig::default();
        let outcomes = evaluate_all(&vfn, &dataset[..5], &translator, &config).unwrap();
        assert_eq!(outcomes.len(), 5);
    }

    #[test]
    fn mean_reward_empty() {
        assert!((mean_reward(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mean_reward_computes_correctly() {
        let outcomes = vec![
            RewardOutcome { reward: 1.0, is_correct: true, correctness: 0.9, gamma: 0.9 },
            RewardOutcome { reward: -1.0, is_correct: false, correctness: 0.1, gamma: 0.5 },
        ];
        assert!((mean_reward(&outcomes) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_reinforce_updates_runs() {
        let mut vfn = Vfn::new_random(42);
        let samples = vec![
            RlvfSample {
                embedding: [0.1; SLOT_DIM],
                advantage: 1.0,
            },
            RlvfSample {
                embedding: [0.2; SLOT_DIM],
                advantage: -0.5,
            },
        ];
        apply_reinforce_updates(&mut vfn, &samples, 0.001).unwrap();
    }

    #[test]
    fn count_puzzle_correct_with_fresh_vfn() {
        let vfn = Vfn::new_random(42);
        let translator = StubTranslator::new();
        let puzzles = self_play::generate_puzzles(10, 42);
        let correct = count_puzzle_correct(&vfn, &translator, &puzzles, 0.3);
        // With a random VFN, some puzzles may pass by chance with low threshold
        assert!(correct <= 10);
    }
}
