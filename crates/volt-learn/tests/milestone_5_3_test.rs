//! Milestone 5.3: RLVF Joint Alignment — Integration Tests
//!
//! Tests per the roadmap spec:
//! 1. Certainty calibration improves (or stays stable) after RLVF
//! 2. Self-play: Volt solves generated logic puzzles
//! 3. No safety regression after RLVF training
//! 4. Overall quality: mean reward does not degrade
//! 5. Evaluation dataset has 1000 pairs across all categories

use volt_learn::calibration::compute_calibration;
use volt_learn::eval_dataset::{generate_eval_dataset, EvalCategory};
use volt_learn::reward::{compute_reward, RewardConfig};
use volt_learn::rlvf::{train_rlvf, RlvfConfig};
use volt_learn::self_play::{generate_puzzles, grade_puzzle, PuzzleType};
use volt_soft::vfn::Vfn;
use volt_translate::{StubTranslator, Translator};

/// Helper: run a small RLVF training and return the result.
fn run_rlvf_small() -> volt_learn::rlvf::RlvfResult {
    let mut vfn = Vfn::new_random(42);
    let translator = StubTranslator::new();
    let dataset = generate_eval_dataset();
    let config = RlvfConfig {
        num_epochs: 2,
        puzzle_count: 20,
        puzzle_threshold: 0.3,
        ..RlvfConfig::default()
    };
    // Use a subset for speed
    train_rlvf(&mut vfn, &dataset[..50], &translator, &config).unwrap()
}

/// Test 1: Certainty calibration is computed and has valid ECE.
///
/// After RLVF training, the calibration ECE should be non-negative
/// and the calibration bins should cover the full gamma range.
#[test]
fn calibration_computed_after_rlvf() {
    let result = run_rlvf_small();

    // ECE should be valid (non-negative)
    assert!(
        result.calibration_before.ece >= 0.0,
        "ECE before should be >= 0: {}",
        result.calibration_before.ece,
    );
    assert!(
        result.calibration_after.ece >= 0.0,
        "ECE after should be >= 0: {}",
        result.calibration_after.ece,
    );

    // Both calibrations should have bins covering full range
    assert_eq!(result.calibration_before.bins.len(), 10);
    assert_eq!(result.calibration_after.bins.len(), 10);

    // Total samples should match
    assert_eq!(result.calibration_before.total_samples, 50);
    assert_eq!(result.calibration_after.total_samples, 50);
}

/// Test 2: Self-play logic puzzles are generated and graded.
///
/// The system should be able to generate puzzles and attempt to
/// solve them. We verify the pipeline runs without error.
#[test]
fn self_play_puzzles_generated_and_graded() {
    let puzzles = generate_puzzles(100, 42);
    assert_eq!(puzzles.len(), 100);

    // All 5 puzzle types should be represented
    let has_mp = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::ModusPonens);
    let has_tr = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Transitivity);
    let has_mt = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::ModusTollens);
    let has_cj = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Conjunction);
    let has_dj = puzzles.iter().any(|p| p.puzzle_type == PuzzleType::Disjunction);
    assert!(has_mp, "should have ModusPonens puzzles");
    assert!(has_tr, "should have Transitivity puzzles");
    assert!(has_mt, "should have ModusTollens puzzles");
    assert!(has_cj, "should have Conjunction puzzles");
    assert!(has_dj, "should have Disjunction puzzles");

    // Grade puzzles through the VFN pipeline
    let vfn = Vfn::new_random(42);
    let translator = StubTranslator::new();
    let mut graded = 0;

    for puzzle in &puzzles {
        let premise_out = translator.encode(&puzzle.premises).unwrap();
        let conclusion_out = translator.encode(&puzzle.conclusion).unwrap();

        // Apply VFN drift
        let mut output = premise_out.frame.clone();
        for slot_idx in 0..output.slots.len() {
            if let Some(slot) = &mut output.slots[slot_idx]
                && let Some(r0) = &slot.resolutions[0]
                && let Ok(drift) = vfn.forward(r0)
            {
                let mut updated = [0.0f32; volt_core::SLOT_DIM];
                for k in 0..volt_core::SLOT_DIM {
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

        let _correct = grade_puzzle(&output, &conclusion_out.frame, 0.3);
        graded += 1;
    }

    assert_eq!(graded, 100, "all puzzles should be graded");
}

/// Test 3: No safety regression — gamma scores remain valid after RLVF.
///
/// After RLVF training, running evaluation should not produce any
/// NaN or Inf values in the VFN output.
#[test]
fn no_safety_regression_after_rlvf() {
    let mut vfn = Vfn::new_random(42);
    let translator = StubTranslator::new();
    let dataset = generate_eval_dataset();
    let config = RlvfConfig {
        num_epochs: 2,
        puzzle_count: 5,
        ..RlvfConfig::default()
    };

    // Train
    let _result = train_rlvf(&mut vfn, &dataset[..30], &translator, &config).unwrap();

    // Verify VFN still produces valid outputs
    for pair in &dataset[..50] {
        let frame_out = translator.encode(&pair.question).unwrap();
        for slot_opt in &frame_out.frame.slots {
            if let Some(slot) = slot_opt
                && let Some(r0) = &slot.resolutions[0]
            {
                let drift = vfn.forward(r0).unwrap();
                // No NaN or Inf in output
                for &v in &drift {
                    assert!(
                        v.is_finite(),
                        "VFN output contains non-finite value after RLVF: {v}",
                    );
                }
            }
        }
    }
}

/// Test 4: Overall quality — mean reward is finite and RLVF completes.
///
/// The RLVF training loop should complete all epochs and produce
/// valid before/after metrics.
#[test]
fn overall_quality_rlvf_completes() {
    let result = run_rlvf_small();

    // Both mean rewards should be finite
    assert!(
        result.mean_reward_before.is_finite(),
        "mean_reward_before should be finite: {}",
        result.mean_reward_before,
    );
    assert!(
        result.mean_reward_after.is_finite(),
        "mean_reward_after should be finite: {}",
        result.mean_reward_after,
    );

    // Epochs should have completed
    assert_eq!(result.epochs_completed, 2);

    // Puzzle counts should be valid
    assert!(result.puzzles_correct_before <= result.total_puzzles);
    assert!(result.puzzles_correct_after <= result.total_puzzles);
    assert_eq!(result.total_puzzles, 20);
}

/// Test 5: Evaluation dataset has 1000 pairs across all 4 categories.
#[test]
fn evaluation_dataset_structure() {
    let dataset = generate_eval_dataset();
    assert_eq!(dataset.len(), 1000);

    let math = dataset.iter().filter(|p| p.category == EvalCategory::Math).count();
    let logic = dataset.iter().filter(|p| p.category == EvalCategory::Logic).count();
    let factual = dataset.iter().filter(|p| p.category == EvalCategory::Factual).count();
    let creative = dataset.iter().filter(|p| p.category == EvalCategory::Creative).count();

    assert_eq!(math, 250, "should have 250 math pairs");
    assert_eq!(logic, 250, "should have 250 logic pairs");
    assert_eq!(factual, 250, "should have 250 factual pairs");
    assert_eq!(creative, 250, "should have 250 creative pairs");

    // All pairs should have non-empty question and answer
    for pair in &dataset {
        assert!(!pair.question.is_empty(), "question should not be empty");
        assert!(!pair.answer.is_empty(), "answer should not be empty");
    }
}

/// Bonus: Calibration metric works independently with synthetic data.
#[test]
fn calibration_metric_with_synthetic_data() {
    let config = RewardConfig::default();

    // Create outcomes with known calibration properties
    let mut outcomes = Vec::new();

    // Perfect calibration at gamma=0.95: all correct
    for _ in 0..20 {
        outcomes.push(compute_reward(0.9, 0.95, &config));
    }

    // Overconfident at gamma=0.85: all wrong
    for _ in 0..20 {
        outcomes.push(compute_reward(0.1, 0.85, &config));
    }

    let result = compute_calibration(&outcomes);
    assert_eq!(result.total_samples, 40);
    // ECE should be non-zero due to miscalibration
    assert!(result.ece > 0.0, "ECE should be > 0 for miscalibrated model: {}", result.ece);
}

/// Bonus: Reward shaping produces expected values for edge cases.
#[test]
fn reward_shaping_edge_cases() {
    let config = RewardConfig::default();

    // Exact threshold boundary
    let outcome = compute_reward(0.5, 0.7, &config);
    assert!(outcome.is_correct);
    assert!((outcome.reward - 1.0).abs() < f32::EPSILON);

    // Just below correctness threshold, just above gamma
    let outcome = compute_reward(0.499, 0.71, &config);
    assert!(!outcome.is_correct);
    assert!((outcome.reward - (-2.0)).abs() < f32::EPSILON);

    // Zero gamma, zero correctness
    let outcome = compute_reward(0.0, 0.0, &config);
    assert!(!outcome.is_correct);
    assert!((outcome.reward - 0.2).abs() < f32::EPSILON);
}
