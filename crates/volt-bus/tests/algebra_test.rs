//! Integration tests for LLL Algebra (Milestone 1.2).
//!
//! These tests validate the milestone requirements from PHASE-1.md lines 62-66:
//! 1. unbind(bind(a, b), a) ~ b with cosine similarity > 0.85
//! 2. sim(superpose([a, b]), a) > 0 (constituent detectable)
//! 3. sim(bind(a, b), a) ~ 0 (bound pair dissimilar to inputs)
//! 4. sim(permute(a, 1), permute(a, 2)) ~ 0 (different shifts orthogonal)

use volt_bus::{bind, unbind, superpose, permute, similarity};
use volt_bus::{bind_frames, unbind_frames, similarity_frames};
use volt_core::{TensorFrame, SlotRole, SLOT_DIM};

/// Create a normalized test vector from seed (deterministic pseudo-random using hash)
fn normalized_test_vec(seed: u8) -> [f32; SLOT_DIM] {
    let mut v = [0.0; SLOT_DIM];
    // Use hash-like mixing for true pseudo-randomness
    for i in 0..SLOT_DIM {
        // Mix seed and index using prime multipliers and XOR
        let mut h = (seed as u64).wrapping_mul(0xd2b74407b1ce6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        // Map to [-1, 1]
        v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    // Normalize to unit length
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn milestone_requirement_unbind_accuracy() {
    // From PHASE-1.md line 62: unbind(bind(a,b), a) ~ b with sim > 0.85
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);

    let bound = bind(&a, &b).unwrap();
    let recovered = unbind(&bound, &a).unwrap();

    let sim = similarity(&recovered, &b);
    assert!(
        sim > 0.85,
        "Milestone requirement: unbind(bind(a,b), a) ~ b with sim > 0.85, got {}",
        sim
    );
}

#[test]
fn milestone_requirement_superpose_constituency() {
    // From PHASE-1.md line 63: sim(superpose([a, b]), a) > 0 (constituents detectable)
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);

    let composite = superpose(&[&a, &b]).unwrap();

    let sim_a = similarity(&composite, &a);
    let sim_b = similarity(&composite, &b);

    assert!(
        sim_a > 0.0,
        "Milestone requirement: constituent a should be detectable, got {}",
        sim_a
    );
    assert!(
        sim_b > 0.0,
        "Milestone requirement: constituent b should be detectable, got {}",
        sim_b
    );
}

#[test]
fn milestone_requirement_bind_dissimilarity() {
    // From PHASE-1.md line 64: sim(bind(a, b), a) ~ 0 (bound pair dissimilar to inputs)
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);

    let bound = bind(&a, &b).unwrap();

    let sim_a = similarity(&bound, &a);
    let sim_b = similarity(&bound, &b);

    assert!(
        sim_a.abs() < 0.1,
        "Milestone requirement: bound should be dissimilar to input a, got {}",
        sim_a
    );
    assert!(
        sim_b.abs() < 0.1,
        "Milestone requirement: bound should be dissimilar to input b, got {}",
        sim_b
    );
}

#[test]
fn milestone_requirement_permute_orthogonality() {
    // From PHASE-1.md line 65: sim(permute(a, 1), permute(a, 2)) ~ 0 (different shifts orthogonal)
    let a = normalized_test_vec(1);

    let shift1 = permute(&a, 1);
    let shift2 = permute(&a, 2);

    let sim = similarity(&shift1, &shift2);

    assert!(
        sim.abs() < 0.1,
        "Milestone requirement: different shifts should be nearly orthogonal, got {}",
        sim
    );
}

// Additional integration tests for comprehensive coverage

#[test]
fn bind_unbind_multiple_iterations() {
    // Test that bind/unbind works through multiple iterations
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);
    let c = normalized_test_vec(3);

    // Bind a and b
    let ab = bind(&a, &b).unwrap();

    // Bind result with c
    let abc = bind(&ab, &c).unwrap();

    // Unbind c to get ab back
    let ab_recovered = unbind(&abc, &c).unwrap();

    // Unbind a to get b back
    let b_recovered = unbind(&ab_recovered, &a).unwrap();

    let sim = similarity(&b_recovered, &b);
    assert!(sim > 0.7, "Multi-level unbind should work, got similarity {}", sim);
}

#[test]
fn superpose_many_vectors() {
    // Test superposition with many vectors
    let vectors: Vec<[f32; SLOT_DIM]> = (0..10).map(|i| normalized_test_vec(i)).collect();
    let refs: Vec<&[f32; SLOT_DIM]> = vectors.iter().collect();

    let composite = superpose(&refs).unwrap();

    // All constituents should be somewhat detectable
    for (i, vec) in vectors.iter().enumerate() {
        let sim = similarity(&composite, vec);
        assert!(
            sim > 0.0,
            "Vector {} should be detectable in superposition, got {}",
            i,
            sim
        );
    }
}

#[test]
fn frame_bind_unbind_workflow() {
    // Test full workflow with TensorFrames
    let mut frame_a = TensorFrame::new();
    frame_a.write_at(0, 0, SlotRole::Agent, normalized_test_vec(1)).unwrap();
    frame_a.write_at(1, 0, SlotRole::Predicate, normalized_test_vec(2)).unwrap();

    let mut frame_b = TensorFrame::new();
    frame_b.write_at(0, 0, SlotRole::Agent, normalized_test_vec(3)).unwrap();
    frame_b.write_at(1, 0, SlotRole::Predicate, normalized_test_vec(4)).unwrap();

    // Bind frames
    let bound = bind_frames(&frame_a, &frame_b).unwrap();
    assert_eq!(bound.active_slot_count(), 2);

    // Unbind to recover frame_b
    let recovered = unbind_frames(&bound, &frame_a).unwrap();
    assert_eq!(recovered.active_slot_count(), 2);

    // Check similarity at each slot
    let similarities = similarity_frames(&frame_b, &recovered);
    assert!(
        similarities[0].unwrap() > 0.85,
        "Slot 0 recovery similarity should be > 0.85"
    );
    assert!(
        similarities[1].unwrap() > 0.85,
        "Slot 1 recovery similarity should be > 0.85"
    );
}

#[test]
fn frame_similarity_computation() {
    // Test frame similarity computation
    let mut frame_a = TensorFrame::new();
    frame_a.write_at(0, 0, SlotRole::Agent, normalized_test_vec(1)).unwrap();
    frame_a.write_at(2, 0, SlotRole::Patient, normalized_test_vec(2)).unwrap();

    let mut frame_b = TensorFrame::new();
    frame_b.write_at(0, 0, SlotRole::Agent, normalized_test_vec(1)).unwrap(); // Same as frame_a
    frame_b.write_at(1, 0, SlotRole::Predicate, normalized_test_vec(3)).unwrap(); // Different slot

    let similarities = similarity_frames(&frame_a, &frame_b);

    // Slot 0: both have same vector, similarity = 1.0
    assert!((similarities[0].unwrap() - 1.0).abs() < 1e-6);

    // Slot 1: only frame_b has data, similarity = None
    assert!(similarities[1].is_none());

    // Slot 2: only frame_a has data, similarity = None
    assert!(similarities[2].is_none());
}

// Note: Permute does not distribute over circular convolution (bind)
// because they operate in different domains (spatial vs frequency).
// This is expected behavior for FFT-based HDC operations.

#[test]
fn superpose_associative() {
    // Test that superposition is associative
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);
    let c = normalized_test_vec(3);

    // (a + b) + c
    let ab = superpose(&[&a, &b]).unwrap();
    let ab_c = superpose(&[&ab, &c]).unwrap();

    // a + (b + c)
    let bc = superpose(&[&b, &c]).unwrap();
    let a_bc = superpose(&[&a, &bc]).unwrap();

    // Results should be similar
    let sim = similarity(&ab_c, &a_bc);
    assert!(sim > 0.95, "Superpose should be associative, got {}", sim);
}

#[test]
fn bind_with_permuted_vectors() {
    // Test binding with permuted vectors
    let a = normalized_test_vec(1);
    let b = normalized_test_vec(2);

    let perm_b = permute(&b, 3);

    // Bind a with permuted b
    let bound = bind(&a, &perm_b).unwrap();

    // Unbind with original a
    let recovered = unbind(&bound, &a).unwrap();

    // Should recover permuted b
    let sim = similarity(&recovered, &perm_b);
    assert!(sim > 0.85, "Should recover permuted vector, got {}", sim);
}
