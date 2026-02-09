//! Integration tests for Milestone 1.3 translator.
//!
//! Tests the stub translator end-to-end: encode, decode, roundtrip,
//! error handling, and edge cases.

use volt_core::{SlotRole, TensorFrame, MAX_SLOTS, SLOT_DIM};
use volt_translate::{StubTranslator, Translator};

#[test]
fn milestone_encode_cat_sat_mat() {
    // PHASE-1.md: Send "The cat sat on the mat" -> 3+ filled slots
    let t = StubTranslator::new();
    let output = t.encode("The cat sat on the mat").unwrap();

    assert!(
        output.slots_filled >= 3,
        "Expected >= 3 filled slots, got {}",
        output.slots_filled
    );
    assert_eq!(output.token_count, 6);

    // Verify slot roles
    let slot0 = output.frame.read_slot(0).unwrap();
    assert_eq!(slot0.role, SlotRole::Agent);

    let slot1 = output.frame.read_slot(1).unwrap();
    assert_eq!(slot1.role, SlotRole::Predicate);

    let slot2 = output.frame.read_slot(2).unwrap();
    assert_eq!(slot2.role, SlotRole::Patient);
}

#[test]
fn milestone_roundtrip_readable() {
    // PHASE-1.md: Round-trip encode -> decode -> readable text (even if clumsy)
    let t = StubTranslator::new();
    let output = t.encode("The cat sat on the mat").unwrap();
    let decoded = t.decode(&output.frame).unwrap();

    assert!(!decoded.is_empty());
    let lower = decoded.to_lowercase();
    // The stub should recover at least some of the original words
    assert!(
        lower.contains("cat") || lower.contains("sat") || lower.contains("mat"),
        "Decoded text '{}' should contain recognizable words",
        decoded
    );
}

#[test]
fn encode_deterministic_across_instances() {
    // Same input should produce identical vectors in different translators
    let t1 = StubTranslator::new();
    let t2 = StubTranslator::new();

    let out1 = t1.encode("hello world").unwrap();
    let out2 = t2.encode("hello world").unwrap();

    let v1 = out1.frame.read_slot(0).unwrap().resolutions[1].unwrap();
    let v2 = out2.frame.read_slot(0).unwrap().resolutions[1].unwrap();

    for i in 0..SLOT_DIM {
        assert_eq!(v1[i], v2[i], "Vectors differ at dim {}", i);
    }
}

#[test]
fn encode_empty_input_errors() {
    let t = StubTranslator::new();
    assert!(t.encode("").is_err());
}

#[test]
fn encode_whitespace_only_errors() {
    let t = StubTranslator::new();
    assert!(t.encode("   \t\n  ").is_err());
}

#[test]
fn encode_huge_input_errors() {
    let t = StubTranslator::new();
    let huge = "word ".repeat(10_000); // ~50KB
    assert!(t.encode(&huge).is_err());
}

#[test]
fn encode_caps_at_max_slots() {
    let t = StubTranslator::new();
    let many_words: String = (0..20)
        .map(|i| format!("word{}", i))
        .collect::<Vec<_>>()
        .join(" ");
    let output = t.encode(&many_words).unwrap();

    assert_eq!(output.slots_filled, MAX_SLOTS);
    assert_eq!(output.token_count, 20);
}

#[test]
fn decode_empty_frame() {
    let t = StubTranslator::new();
    let frame = TensorFrame::new();
    let decoded = t.decode(&frame).unwrap();
    assert_eq!(decoded, "[empty frame]");
}

#[test]
fn encode_unicode_no_panic() {
    let t = StubTranslator::new();
    let result = t.encode("caf\u{00e9} na\u{00ef}ve \u{1f600}");
    assert!(result.is_ok());
}

#[test]
fn encode_single_word() {
    let t = StubTranslator::new();
    let output = t.encode("hello").unwrap();
    assert_eq!(output.slots_filled, 1);
    assert_eq!(output.token_count, 1);

    let slot = output.frame.read_slot(0).unwrap();
    assert_eq!(slot.role, SlotRole::Agent);
}

#[test]
fn roundtrip_single_word() {
    let t = StubTranslator::new();
    let output = t.encode("hello").unwrap();
    let decoded = t.decode(&output.frame).unwrap();
    assert!(
        decoded.to_lowercase().contains("hello"),
        "decoded: {}",
        decoded
    );
}

#[test]
fn roundtrip_preserves_all_words() {
    let t = StubTranslator::new();
    let output = t.encode("cat sat mat").unwrap();
    let decoded = t.decode(&output.frame).unwrap();
    let lower = decoded.to_lowercase();
    assert!(lower.contains("cat"), "missing 'cat' in: {}", decoded);
    assert!(lower.contains("sat"), "missing 'sat' in: {}", decoded);
    assert!(lower.contains("mat"), "missing 'mat' in: {}", decoded);
}

#[test]
fn different_words_produce_different_vectors() {
    let t = StubTranslator::new();
    let out = t.encode("cat dog").unwrap();

    let v_cat = out.frame.read_slot(0).unwrap().resolutions[1].unwrap();
    let v_dog = out.frame.read_slot(1).unwrap().resolutions[1].unwrap();

    // Cosine similarity should be low (near 0)
    let sim: f32 = v_cat.iter().zip(v_dog.iter()).map(|(a, b)| a * b).sum();
    assert!(sim.abs() < 0.3, "sim = {} (should be near 0)", sim);
}
