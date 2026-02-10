//! Integration tests for the LLM-backed translator (Milestone 2.2).
//!
//! Feature-gated: `cargo test -p volt-translate --features llm`
//!
//! ## Test Tiers
//!
//! - **Tier 1**: Pure logic — role mapping, projection shapes, aggregation.
//!   No model files needed, no external deps.
//! - **Tier 2**: Mock backbone — full pipeline with synthetic weights.
//!   No model files needed, tests structural correctness.
//! - **Tier 3**: Real model (`#[ignore]`) — requires downloaded Qwen3-0.6B
//!   and trained projection weights.

#![cfg(feature = "llm")]

use candle_core::{DType, Device, Tensor};
use volt_core::{MAX_SLOTS, SLOT_DIM};
use volt_translate::llm::backbone::LlmBackbone;
use volt_translate::llm::projection::{
    aggregate_to_slots, FrameProjectionHead, ProjectionConfig,
};
use volt_translate::llm::roles::{
    parse_propbank_label, propbank_to_slot, role_to_class_index, slot_to_propbank, slot_to_role,
    PropBankRole, NUM_ROLE_CLASSES,
};
use volt_translate::llm::translator::{LlmTranslator, LlmTranslatorConfig};
use volt_translate::Translator;

// ═══════════════════════════════════════════════════════════════════
// TIER 1: Pure logic tests (no model files)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn tier1_all_propbank_labels_parse() {
    let labels = [
        "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "V", "ARGM-LOC", "ARGM-TMP", "ARGM-MNR",
        "ARGM-CAU", "ARGM-DIR", "ARGM-ADV", "ARGM-PNC", "ARGM-DIS", "ARGM-NEG", "ARGM-MOD",
        "O",
    ];
    for label in &labels {
        assert!(
            parse_propbank_label(label).is_ok(),
            "failed to parse {label}"
        );
    }
}

#[test]
fn tier1_role_mapping_roundtrip() {
    for role in [
        PropBankRole::Arg0,
        PropBankRole::V,
        PropBankRole::Arg1,
        PropBankRole::ArgmLoc,
        PropBankRole::ArgmTmp,
        PropBankRole::ArgmMnr,
        PropBankRole::Arg2,
        PropBankRole::ArgmCau,
        PropBankRole::ArgmDir,
        PropBankRole::ArgmAdv,
        PropBankRole::ArgmPnc,
        PropBankRole::ArgmDis,
        PropBankRole::ArgmNeg,
        PropBankRole::ArgmMod,
        PropBankRole::NoRole,
    ] {
        let (idx, _slot_role) = propbank_to_slot(role);
        let back = slot_to_propbank(idx);
        assert_eq!(
            propbank_to_slot(back).0,
            idx,
            "round-trip failed for {role:?}"
        );
    }
}

#[test]
fn tier1_all_16_slots_covered() {
    for i in 0..NUM_ROLE_CLASSES {
        let role = slot_to_propbank(i);
        let (idx, _) = propbank_to_slot(role);
        assert_eq!(idx, i, "slot {i} not properly covered");
    }
}

#[test]
fn tier1_slot_to_role_matches_propbank_mapping() {
    for i in 0..NUM_ROLE_CLASSES {
        let via_propbank = propbank_to_slot(slot_to_propbank(i)).1;
        let via_direct = slot_to_role(i);
        assert_eq!(
            via_propbank, via_direct,
            "slot_to_role disagrees with propbank roundtrip at slot {i}"
        );
    }
}

#[test]
fn tier1_class_index_is_slot_index() {
    for role in [
        PropBankRole::Arg0,
        PropBankRole::V,
        PropBankRole::Arg1,
        PropBankRole::ArgmLoc,
        PropBankRole::Arg2,
        PropBankRole::NoRole,
    ] {
        assert_eq!(
            role_to_class_index(role),
            propbank_to_slot(role).0,
            "class index != slot index for {role:?}"
        );
    }
}

#[test]
fn tier1_num_role_classes_matches_max_slots() {
    assert_eq!(NUM_ROLE_CLASSES, MAX_SLOTS);
}

#[test]
fn tier1_projection_forward_correct_shapes() {
    let config = ProjectionConfig {
        hidden_dim: 64,
        mlp_dim: 128,
    };
    let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
    let input = Tensor::randn(0f32, 1.0, (7, 64), &Device::Cpu).unwrap();

    let (role_probs, token_embeds) = head.forward(&input).unwrap();

    assert_eq!(role_probs.dims(), &[7, NUM_ROLE_CLASSES]);
    assert_eq!(token_embeds.dims(), &[7, SLOT_DIM]);
}

#[test]
fn tier1_projection_softmax_valid() {
    let config = ProjectionConfig {
        hidden_dim: 64,
        mlp_dim: 128,
    };
    let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
    let input = Tensor::randn(0f32, 1.0, (5, 64), &Device::Cpu).unwrap();

    let (role_probs, _) = head.forward(&input).unwrap();
    let probs = role_probs.to_vec2::<f32>().unwrap();

    for (i, row) in probs.iter().enumerate() {
        // All probabilities non-negative
        for &p in row {
            assert!(p >= 0.0, "negative probability at token {i}");
        }
        // Sum to 1.0
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax row {i} sums to {sum}, expected 1.0"
        );
    }
}

#[test]
fn tier1_aggregate_produces_valid_slot_vectors() {
    // Synthetic: token 0 → role 0 (Agent), token 1 → role 1 (Predicate)
    let probs = Tensor::new(
        &[
            [
                0.9_f32, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
            [
                0.05, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
        ],
        &Device::Cpu,
    )
    .unwrap();
    let embeds = Tensor::randn(0f32, 1.0, (2, SLOT_DIM), &Device::Cpu).unwrap();

    let slots = aggregate_to_slots(&probs, &embeds, 0.5).unwrap();

    // Should get Agent and Predicate
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[0].0, 0); // Agent slot
    assert_eq!(slots[1].0, 1); // Predicate slot
}

#[test]
fn tier1_aggregate_slot_vectors_l2_normalized() {
    let probs = Tensor::new(
        &[
            [
                0.8_f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
            [
                0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
        ],
        &Device::Cpu,
    )
    .unwrap();
    let embeds = Tensor::randn(0f32, 1.0, (2, SLOT_DIM), &Device::Cpu).unwrap();

    let slots = aggregate_to_slots(&probs, &embeds, 0.1).unwrap();

    for (idx, vec, _) in &slots {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "slot {idx} vector norm = {norm}, expected 1.0"
        );
    }
}

#[test]
fn tier1_aggregate_empty_on_high_threshold() {
    // Uniform distribution: each role gets 1/16 ≈ 0.0625
    let uniform = [1.0 / 16.0_f32; NUM_ROLE_CLASSES];
    let probs = Tensor::new(&[uniform], &Device::Cpu).unwrap();
    let embeds = Tensor::ones((1, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

    let slots = aggregate_to_slots(&probs, &embeds, 1.0).unwrap();
    assert!(
        slots.is_empty(),
        "expected empty with threshold=1.0, got {} slots",
        slots.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// TIER 2: Mock backbone tests (full pipeline, no model files)
// ═══════════════════════════════════════════════════════════════════

fn mock_translator() -> LlmTranslator {
    LlmTranslator::mock(64, &Device::Cpu).unwrap()
}

#[test]
fn tier2_encode_produces_nonempty_frame() {
    let t = mock_translator();
    let output = t.encode("the cat sat on the mat").unwrap();

    assert!(!output.frame.is_empty());
    assert!(output.slots_filled > 0);
    assert!(output.slots_filled <= MAX_SLOTS);
    assert_eq!(output.token_count, 6);
}

#[test]
fn tier2_encode_respects_max_16_slots() {
    let t = mock_translator();
    // Many words → should still cap at 16 slots
    let output = t
        .encode("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen")
        .unwrap();
    assert!(output.slots_filled <= MAX_SLOTS);
}

#[test]
fn tier2_encode_empty_input_errors() {
    let t = mock_translator();
    assert!(t.encode("").is_err());
    assert!(t.encode("   ").is_err());
    assert!(t.encode("\n\t").is_err());
}

#[test]
fn tier2_different_inputs_both_produce_valid_frames() {
    let t = mock_translator();
    let out1 = t.encode("the cat sat").unwrap();
    let out2 = t.encode("dogs run fast").unwrap();

    // Both should produce non-empty frames with correct token counts
    assert!(!out1.frame.is_empty());
    assert!(!out2.frame.is_empty());
    assert_eq!(out1.token_count, 3);
    assert_eq!(out2.token_count, 3);
    assert!(out1.slots_filled > 0);
    assert!(out2.slots_filled > 0);
}

#[test]
fn tier2_encode_sets_frame_metadata() {
    let t = mock_translator();
    let output = t.encode("hello world").unwrap();

    assert!(output.frame.frame_meta.created_at > 0);
    assert!(output.frame.frame_meta.global_certainty >= 0.0);
}

#[test]
fn tier2_encode_sets_slot_metadata() {
    let t = mock_translator();
    let output = t.encode("the cat sat").unwrap();

    for i in 0..MAX_SLOTS {
        if output.frame.slots[i].is_some() {
            let meta = &output.frame.meta[i];
            assert_eq!(
                meta.source,
                volt_core::slot::SlotSource::Translator,
                "slot {i} source should be Translator"
            );
            assert!(meta.needs_verify, "slot {i} should need verification");
            assert!(
                meta.certainty > 0.0,
                "slot {i} certainty should be positive"
            );
            assert!(meta.updated_at > 0, "slot {i} should have a timestamp");
        }
    }
}

#[test]
fn tier2_encode_discourse_classification() {
    let t = mock_translator();

    let q = t.encode("what is life?").unwrap();
    assert_eq!(
        q.frame.frame_meta.discourse_type,
        volt_core::meta::DiscourseType::Query
    );

    let cmd = t.encode("stop now!").unwrap();
    assert_eq!(
        cmd.frame.frame_meta.discourse_type,
        volt_core::meta::DiscourseType::Command
    );

    let stmt = t.encode("the cat sat on the mat").unwrap();
    assert_eq!(
        stmt.frame.frame_meta.discourse_type,
        volt_core::meta::DiscourseType::Statement
    );
}

#[test]
fn tier2_no_codebook_means_no_codebook_id() {
    let t = mock_translator();
    let output = t.encode("test input here").unwrap();

    for i in 0..MAX_SLOTS {
        if let Some(slot) = &output.frame.slots[i] {
            assert!(
                slot.codebook_id.is_none(),
                "mock translator should not set codebook_id on slot {i}"
            );
        }
    }
}

#[test]
fn tier2_slot_vectors_stored_at_r0() {
    let t = mock_translator();
    let output = t.encode("hello world").unwrap();

    for i in 0..MAX_SLOTS {
        if let Some(slot) = &output.frame.slots[i] {
            assert!(
                slot.resolutions[0].is_some(),
                "slot {i} should have data at R0 (discourse level)"
            );
        }
    }
}

#[test]
fn tier2_slot_roles_match_role_mapping() {
    let t = mock_translator();
    let output = t.encode("the cat sat on the mat").unwrap();

    for i in 0..MAX_SLOTS {
        if let Some(slot) = &output.frame.slots[i] {
            let expected_role = slot_to_role(i);
            assert_eq!(
                slot.role, expected_role,
                "slot {i} role {:?} != expected {:?}",
                slot.role, expected_role
            );
        }
    }
}

#[test]
fn tier2_decode_produces_output() {
    let t = mock_translator();
    let output = t.encode("hello world").unwrap();
    let decoded = t.decode(&output.frame).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn tier2_decode_slots_count_matches() {
    let t = mock_translator();
    let output = t.encode("the cat sat").unwrap();
    let slots = t.decode_slots(&output.frame).unwrap();
    assert_eq!(
        slots.len(),
        output.slots_filled,
        "decode_slots count should match slots_filled"
    );
}

#[test]
fn tier2_decode_empty_frame() {
    let t = mock_translator();
    let frame = volt_core::TensorFrame::new();
    let decoded = t.decode(&frame).unwrap();
    assert!(decoded.is_empty());
    let slots = t.decode_slots(&frame).unwrap();
    assert!(slots.is_empty());
}

#[test]
fn tier2_mock_backbone_deterministic() {
    let mut backbone = LlmBackbone::mock(64, &Device::Cpu);
    let tokens = backbone.tokenize("test input").unwrap();
    let h1 = backbone.extract_hidden_states(&tokens).unwrap();
    let h2 = backbone.extract_hidden_states(&tokens).unwrap();

    let diff = (&h1 - &h2)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    assert!(
        diff < 1e-6,
        "mock backbone should be deterministic, got diff={diff}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// TIER 3: Real model tests (require downloaded model)
// ═══════════════════════════════════════════════════════════════════

/// Helper: build an [`LlmTranslatorConfig`] from environment variables.
///
/// Required env vars:
/// - `VOLT_MODEL_DIR` — path to the Qwen3-0.6B directory
/// - `VOLT_PROJECTION_WEIGHTS` — path to projection.safetensors
/// - `VOLT_CODEBOOK_PATH` — path to codebook.bin
fn tier3_config() -> LlmTranslatorConfig {
    let model_dir = std::env::var("VOLT_MODEL_DIR")
        .expect("set VOLT_MODEL_DIR to the Qwen3-0.6B directory");
    let projection_weights = std::env::var("VOLT_PROJECTION_WEIGHTS")
        .expect("set VOLT_PROJECTION_WEIGHTS to the projection.safetensors file");
    let codebook_path = std::env::var("VOLT_CODEBOOK_PATH")
        .expect("set VOLT_CODEBOOK_PATH to the codebook.bin file");

    LlmTranslatorConfig {
        model_dir: model_dir.into(),
        projection_weights: projection_weights.into(),
        codebook_path: codebook_path.into(),
        ..Default::default()
    }
}

#[test]
#[ignore = "requires downloaded Qwen3-0.6B model + trained projection weights"]
fn tier3_real_model_agent_predicate_location() {
    use volt_core::SlotRole;

    let config = tier3_config();
    let device = Device::Cpu;
    let translator =
        LlmTranslator::new(&config, &device).expect("failed to load LLM translator");

    let output = translator
        .encode("The cat sat on the mat")
        .expect("failed to encode");

    let slots = translator
        .decode_slots(&output.frame)
        .expect("failed to decode slots");

    // Build a lookup: slot_index -> role
    let role_map: std::collections::HashMap<usize, SlotRole> =
        slots.iter().map(|(idx, role, _)| (*idx, *role)).collect();

    // S0 = Agent ("cat")
    assert_eq!(
        role_map.get(&0),
        Some(&SlotRole::Agent),
        "slot 0 should be Agent, got {:?}",
        role_map.get(&0),
    );

    // S1 = Predicate ("sat")
    assert_eq!(
        role_map.get(&1),
        Some(&SlotRole::Predicate),
        "slot 1 should be Predicate, got {:?}",
        role_map.get(&1),
    );

    // S3 = Location ("mat")
    assert_eq!(
        role_map.get(&3),
        Some(&SlotRole::Location),
        "slot 3 should be Location, got {:?}",
        role_map.get(&3),
    );

    assert!(
        output.slots_filled >= 3,
        "expected at least 3 slots, got {}",
        output.slots_filled,
    );
}

#[test]
#[ignore = "requires downloaded Qwen3-0.6B model + trained projection weights"]
fn tier3_real_model_accuracy_above_80() {
    use volt_core::SlotRole;

    let config = tier3_config();
    let device = Device::Cpu;
    let translator =
        LlmTranslator::new(&config, &device).expect("failed to load LLM translator");

    // Test sentences with known dominant semantic roles.
    // Each entry: (sentence, expected [(slot_idx, SlotRole)])
    let test_cases: &[(&str, &[(usize, SlotRole)])] = &[
        (
            "The cat sat on the mat",
            &[
                (0, SlotRole::Agent),
                (1, SlotRole::Predicate),
                (3, SlotRole::Location),
            ],
        ),
        (
            "She quickly wrote a letter",
            &[
                (0, SlotRole::Agent),
                (1, SlotRole::Predicate),
                (2, SlotRole::Patient),
            ],
        ),
        (
            "The engineer built a bridge in the city",
            &[
                (0, SlotRole::Agent),
                (1, SlotRole::Predicate),
                (2, SlotRole::Patient),
                (3, SlotRole::Location),
            ],
        ),
        (
            "Yesterday the doctor examined the patient",
            &[
                (0, SlotRole::Agent),
                (1, SlotRole::Predicate),
                (2, SlotRole::Patient),
                (4, SlotRole::Time),
            ],
        ),
        (
            "He repaired the engine with a wrench",
            &[
                (0, SlotRole::Agent),
                (1, SlotRole::Predicate),
                (2, SlotRole::Patient),
                (6, SlotRole::Instrument),
            ],
        ),
    ];

    let mut total_checks = 0;
    let mut correct = 0;

    for (sentence, expected_roles) in test_cases {
        let output = translator
            .encode(sentence)
            .unwrap_or_else(|e| panic!("failed to encode '{sentence}': {e}"));

        let slots = translator
            .decode_slots(&output.frame)
            .unwrap_or_else(|e| panic!("failed to decode slots for '{sentence}': {e}"));

        let role_map: std::collections::HashMap<usize, SlotRole> =
            slots.iter().map(|(idx, role, _)| (*idx, *role)).collect();

        for &(slot_idx, ref expected_role) in *expected_roles {
            total_checks += 1;
            if role_map.get(&slot_idx) == Some(expected_role) {
                correct += 1;
            }
        }
    }

    let accuracy = correct as f64 / total_checks as f64;
    assert!(
        accuracy > 0.80,
        "role accuracy {:.1}% ({}/{}) is below 80% threshold",
        accuracy * 100.0,
        correct,
        total_checks,
    );
}
