//! Phase 0.2: Code Dataset Pipeline — Integration Test
//!
//! Tests the complete code dataset pipeline:
//! - Load unified JSONL datasets
//! - Stream (query, solution) TensorFrame pairs
//! - Verify deterministic encoding
//! - Process 1,000+ pairs successfully

use std::thread;
use volt_learn::code_dataset::CodeDataset;
use volt_translate::stub::StubTranslator;

/// Phase 0.2 Integration Test: Load 1,000+ code pairs and verify
/// deterministic TensorFrame encoding.
///
/// This test validates:
/// 1. CodeDataset loads unified JSONL format
/// 2. Streaming iterator yields (TensorFrame, TensorFrame) pairs
/// 3. Encoding is deterministic (same input → same frame ID)
/// 4. Can process 1,000+ pairs without errors
#[test]
fn phase_0_2_code_dataset_pipeline() {
    // Use larger stack on Windows to avoid stack overflow with TensorFrame
    let result = thread::Builder::new()
        .stack_size(8 * 1024 * 1024) // 8 MB stack
        .spawn(|| {
            // Load combined dataset (HumanEval + MBPP = 421 problems)
            let dataset_path = "D:/VoltData/phase0/code_training_combined.jsonl";
            let dataset = CodeDataset::from_file(dataset_path)
                .expect("Failed to load code dataset");

            println!("Loaded {} problems from combined dataset", dataset.len());
            assert!(
                dataset.len() >= 400,
                "Expected at least 400 problems, got {}",
                dataset.len()
            );

            // Test 1: Dataset iteration works
            let mut count = 0;
            for problem in dataset.iter() {
                assert!(!problem.id.is_empty(), "Problem ID should not be empty");
                assert!(!problem.query.is_empty(), "Query should not be empty");
                assert!(!problem.solution.is_empty(), "Solution should not be empty");
                count += 1;
            }
            assert_eq!(count, dataset.len());
            println!("✓ Test 1 passed: Iterated over {} problems", count);

            // Test 2: TensorFrame encoding works
            let translator = StubTranslator::new();
            let sample_problem = dataset.get(0).expect("Dataset should have at least 1 problem");

            let (query_frame, solution_frame) = sample_problem
                .to_frame_pair(&translator)
                .expect("Failed to encode first problem");

            println!(
                "✓ Test 2 passed: Encoded first problem (query={}, solution={})",
                query_frame.frame_meta.frame_id, solution_frame.frame_meta.frame_id
            );

            // Test 3: Deterministic encoding
            let (query_frame2, solution_frame2) = sample_problem
                .to_frame_pair(&translator)
                .expect("Failed to encode first problem (2nd time)");

            assert_eq!(
                query_frame.frame_meta.frame_id, query_frame2.frame_meta.frame_id,
                "Query frame IDs should be identical"
            );
            assert_eq!(
                solution_frame.frame_meta.frame_id,
                solution_frame2.frame_meta.frame_id,
                "Solution frame IDs should be identical"
            );
            println!("✓ Test 3 passed: Encoding is deterministic");

            // Test 4: Stream 1,000+ pairs
            // We have 421 problems, so iterate ~3 times to get 1,000+
            let target_pairs = 1000;
            let mut encoded_count = 0;
            let mut errors = 0;

            // Repeat dataset iteration to get 1,000+ pairs
            for round in 0..3 {
                for result in dataset.iter_frames(&translator) {
                    match result {
                        Ok((_query, _solution)) => {
                            // Frame created successfully
                            encoded_count += 1;

                            if encoded_count >= target_pairs {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Encoding error in round {}: {}", round, e);
                            errors += 1;
                        }
                    }
                }
                if encoded_count >= target_pairs {
                    break;
                }
            }

            assert!(
                encoded_count >= target_pairs,
                "Expected to encode at least {target_pairs} pairs, got {encoded_count}"
            );
            assert_eq!(errors, 0, "Should have no encoding errors, got {errors}");
            println!(
                "✓ Test 4 passed: Successfully encoded {} pairs (target: {})",
                encoded_count, target_pairs
            );

            // Test 5: Batch conversion
            let batch_size = 50;
            let batch_problems: Vec<_> = dataset.iter().take(batch_size).collect();
            let mut batch_pairs = Vec::new();

            for problem in batch_problems {
                match problem.to_frame_pair(&translator) {
                    Ok(pair) => batch_pairs.push(pair),
                    Err(e) => panic!("Batch encoding failed: {}", e),
                }
            }

            assert_eq!(
                batch_pairs.len(),
                batch_size,
                "Should encode all batch problems"
            );
            println!("✓ Test 5 passed: Batch encoded {} pairs", batch_pairs.len());

            println!("\n✅ Phase 0.2 complete: Code dataset pipeline validated");
            println!("   - Loaded {} problems", dataset.len());
            println!("   - Verified deterministic encoding");
            println!("   - Processed 1,000+ TensorFrame pairs");
        })
        .expect("Failed to spawn test thread")
        .join();

    assert!(result.is_ok(), "Test thread panicked: {:?}", result.err());
}

/// Test loading individual datasets
#[test]
fn load_humaneval_dataset() {
    let dataset_path = "D:/VoltData/phase0/humaneval_unified.jsonl";
    let dataset = CodeDataset::from_file(dataset_path).expect("Failed to load HumanEval");

    assert_eq!(dataset.len(), 164, "HumanEval should have 164 problems");

    let first = dataset.get(0).expect("Should have first problem");
    assert!(first.id.starts_with("HumanEval/"));
    assert_eq!(
        first.language,
        Some("python".to_string()),
        "Language should be python"
    );
    println!("✓ HumanEval dataset loaded: {} problems", dataset.len());
}

/// Test loading MBPP dataset
#[test]
fn load_mbpp_dataset() {
    let dataset_path = "D:/VoltData/phase0/mbpp_unified.jsonl";
    let dataset = CodeDataset::from_file(dataset_path).expect("Failed to load MBPP");

    assert_eq!(dataset.len(), 257, "MBPP should have 257 problems");

    let first = dataset.get(0).expect("Should have first problem");
    assert!(first.id.starts_with("MBPP/"));
    assert_eq!(
        first.language,
        Some("python".to_string()),
        "Language should be python"
    );
    println!("✓ MBPP dataset loaded: {} problems", dataset.len());
}
