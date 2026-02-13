//! Integration tests for Phase 2 code frame pair generation.

use volt_learn::code_dataset::{CodeDataset, CodeProblem};
use volt_learn::code_frame_pairs::{
    classify_problem, generate_code_frame_pairs, CodeFramePairConfig, CurriculumStageFilter,
};

fn make_test_dataset() -> CodeDataset {
    use std::io::Write;
    let temp_dir = std::env::temp_dir();
    let tid = std::thread::current().id();
    let path = temp_dir.join(format!("phase2_integ_{tid:?}.jsonl"));

    let mut file = std::fs::File::create(&path).unwrap();

    // Simple function
    writeln!(
        file,
        r#"{{"id":"simple/1","query":"Add two numbers","solution":"def add(a, b): return a + b","tests":["assert add(1,2)==3"],"difficulty":"easy"}}"#
    ).unwrap();

    // With loop (escape newlines for valid JSON)
    writeln!(
        file,
        r#"{{"id":"loop/1","query":"Sum a list","solution":"def sum_list(lst):\\n    total = 0\\n    for x in lst:\\n        total += x\\n    return total","tests":[],"difficulty":"medium"}}"#
    ).unwrap();

    // Multi-function
    writeln!(
        file,
        r#"{{"id":"multi/1","query":"Helper functions","solution":"def helper():\\n    pass\\ndef main():\\n    helper()","tests":[]}}"#
    ).unwrap();

    // Hard algorithmic
    writeln!(
        file,
        r#"{{"id":"algo/1","query":"Sort an array","solution":"def merge_sort(arr):\\n    if len(arr) <= 1: return arr\\n    mid = len(arr) // 2\\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))","tests":[],"difficulty":"hard"}}"#
    ).unwrap();

    // More simple functions for variety
    for i in 2..8 {
        writeln!(
            file,
            r#"{{"id":"simple/{i}","query":"Function {i}","solution":"def f{i}(x): return x + {i}","tests":[]}}"#
        ).unwrap();
    }

    let dataset = CodeDataset::from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    dataset
}

#[test]
fn generate_pairs_from_dataset() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig::default();
    let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
    assert_eq!(pairs.len(), dataset.len());
}

#[test]
fn pairs_have_4_active_slots() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig::default();
    let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();

    for (i, pair) in pairs.iter().enumerate() {
        assert_eq!(
            pair.question.active_slot_count(),
            4,
            "pair {i} question should have 4 active slots"
        );
        assert_eq!(
            pair.answer.active_slot_count(),
            4,
            "pair {i} answer should have 4 active slots"
        );
    }
}

#[test]
fn pairs_are_normalized() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig::default();
    let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();

    for pair in &pairs {
        for slot_idx in 0..4 {
            let q_slot = pair.question.read_slot(slot_idx).unwrap();
            let q_vec = q_slot.resolutions[0].as_ref().unwrap();
            let norm: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "slot {slot_idx} not normalized: norm={norm}"
            );
        }
    }
}

#[test]
fn query_and_answer_frames_differ() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig::default();
    let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();

    for pair in &pairs {
        let q_vec = pair.question.read_slot(0).unwrap().resolutions[0].unwrap();
        let a_vec = pair.answer.read_slot(0).unwrap().resolutions[0].unwrap();

        // Query and answer should differ (different text)
        assert_ne!(q_vec, a_vec, "query and answer should produce different embeddings");
    }
}

#[test]
fn pairs_are_deterministic() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig::default();

    let pairs1 = generate_code_frame_pairs(&dataset, &config).unwrap();
    let pairs2 = generate_code_frame_pairs(&dataset, &config).unwrap();

    assert_eq!(pairs1.len(), pairs2.len());

    for (i, (a, b)) in pairs1.iter().zip(pairs2.iter()).enumerate() {
        let q1 = a.question.read_slot(0).unwrap().resolutions[0].unwrap();
        let q2 = b.question.read_slot(0).unwrap().resolutions[0].unwrap();
        assert_eq!(q1, q2, "pair {i} not deterministic");
    }
}

#[test]
fn curriculum_classification_correct() {
    let simple = CodeProblem {
        id: "1".into(),
        query: "Add".into(),
        solution: "def add(a, b): return a + b".into(),
        tests: vec![],
        language: None,
        difficulty: Some("easy".into()),
    };
    assert_eq!(
        classify_problem(&simple),
        CurriculumStageFilter::SimpleFunctions
    );

    let loopy = CodeProblem {
        id: "2".into(),
        query: "Sum".into(),
        solution: "def f(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total".into(),
        tests: vec![],
        language: None,
        difficulty: None,
    };
    assert_eq!(
        classify_problem(&loopy),
        CurriculumStageFilter::LoopsAndConditionals
    );

    let hard = CodeProblem {
        id: "3".into(),
        query: "Sort".into(),
        solution: "def sort(x): pass".into(),
        tests: vec![],
        language: None,
        difficulty: Some("hard".into()),
    };
    assert_eq!(
        classify_problem(&hard),
        CurriculumStageFilter::AlgorithmicReasoning
    );
}

#[test]
fn curriculum_filter_works() {
    let dataset = make_test_dataset();

    // Only simple functions
    let config = CodeFramePairConfig {
        stage_filter: Some(CurriculumStageFilter::SimpleFunctions),
        ..CodeFramePairConfig::default()
    };
    let simple_pairs = generate_code_frame_pairs(&dataset, &config).unwrap();

    // All pairs
    let all_config = CodeFramePairConfig::default();
    let all_pairs = generate_code_frame_pairs(&dataset, &all_config).unwrap();

    assert!(simple_pairs.len() < all_pairs.len());
    assert!(!simple_pairs.is_empty());
}

#[test]
fn max_pairs_limits_output() {
    let dataset = make_test_dataset();
    let config = CodeFramePairConfig {
        max_pairs: 3,
        ..CodeFramePairConfig::default()
    };
    let pairs = generate_code_frame_pairs(&dataset, &config).unwrap();
    assert_eq!(pairs.len(), 3);
}
