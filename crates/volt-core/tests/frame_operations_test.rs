//! Integration tests for TensorFrame merge and normalization operations.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};

#[test]
fn merge_both_empty_produces_empty() {
    let frame1 = TensorFrame::new();
    let frame2 = TensorFrame::new();

    let merged = frame1.merge(frame2);
    assert!(merged.is_empty());
}

#[test]
fn merge_one_empty_returns_other() {
    let mut frame1 = TensorFrame::new();
    frame1
        .write_slot(0, SlotData::new(SlotRole::Agent))
        .unwrap();
    let frame2 = TensorFrame::new();

    let merged = frame1.merge(frame2);
    assert_eq!(merged.active_slot_count(), 1);
}

#[test]
fn merge_no_overlap_combines_all_slots() {
    let mut frame1 = TensorFrame::new();
    frame1
        .write_slot(0, SlotData::new(SlotRole::Agent))
        .unwrap();

    let mut frame2 = TensorFrame::new();
    frame2
        .write_slot(1, SlotData::new(SlotRole::Predicate))
        .unwrap();

    let merged = frame1.merge(frame2);
    assert_eq!(merged.active_slot_count(), 2);
}

#[test]
fn merge_conflict_resolved_by_certainty() {
    let mut frame1 = TensorFrame::new();
    frame1
        .write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM])
        .unwrap();
    frame1.meta[0].certainty = 0.9;

    let mut frame2 = TensorFrame::new();
    frame2
        .write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM])
        .unwrap();
    frame2.meta[0].certainty = 0.7;

    let merged = frame1.merge(frame2);

    // frame1 had higher certainty, so its data should be kept
    let slot = merged.read_slot(0).unwrap();
    assert_eq!(slot.resolutions[0].unwrap()[0], 1.0);
    assert_eq!(merged.meta[0].certainty, 0.9);
}

#[test]
fn merge_equal_certainty_prefers_left() {
    let mut frame1 = TensorFrame::new();
    frame1
        .write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM])
        .unwrap();
    frame1.meta[0].certainty = 0.8;

    let mut frame2 = TensorFrame::new();
    frame2
        .write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM])
        .unwrap();
    frame2.meta[0].certainty = 0.8;

    let merged = frame1.merge(frame2);

    // Equal certainty → prefer left (frame1)
    let slot = merged.read_slot(0).unwrap();
    assert_eq!(slot.resolutions[0].unwrap()[0], 1.0);
}

#[test]
fn merge_complex_scenario_multiple_conflicts() {
    let mut frame1 = TensorFrame::new();
    frame1
        .write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM])
        .unwrap();
    frame1.meta[0].certainty = 0.9;
    frame1
        .write_at(1, 0, SlotRole::Predicate, [2.0; SLOT_DIM])
        .unwrap();
    frame1.meta[1].certainty = 0.7;
    frame1
        .write_at(3, 0, SlotRole::Location, [3.0; SLOT_DIM])
        .unwrap();
    frame1.meta[3].certainty = 0.85;

    let mut frame2 = TensorFrame::new();
    frame2
        .write_at(0, 0, SlotRole::Agent, [10.0; SLOT_DIM])
        .unwrap();
    frame2.meta[0].certainty = 0.8; // Lower than frame1
    frame2
        .write_at(1, 0, SlotRole::Predicate, [20.0; SLOT_DIM])
        .unwrap();
    frame2.meta[1].certainty = 0.95; // Higher than frame1
    frame2
        .write_at(4, 0, SlotRole::Time, [40.0; SLOT_DIM])
        .unwrap();
    frame2.meta[4].certainty = 0.9;

    let merged = frame1.merge(frame2);

    // Should have 4 slots total (0, 1, 3, 4)
    assert_eq!(merged.active_slot_count(), 4);

    // Slot 0: frame1 wins (0.9 > 0.8)
    assert_eq!(merged.read_slot(0).unwrap().resolutions[0].unwrap()[0], 1.0);

    // Slot 1: frame2 wins (0.95 > 0.7)
    assert_eq!(
        merged.read_slot(1).unwrap().resolutions[0].unwrap()[0],
        20.0
    );

    // Slot 3: only frame1 had it
    assert_eq!(merged.read_slot(3).unwrap().resolutions[0].unwrap()[0], 3.0);

    // Slot 4: only frame2 had it
    assert_eq!(
        merged.read_slot(4).unwrap().resolutions[0].unwrap()[0],
        40.0
    );
}

#[test]
fn normalize_slot_produces_unit_vector() {
    let mut frame = TensorFrame::new();
    frame
        .write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM])
        .unwrap();

    frame.normalize_slot(0, 0).unwrap();

    let slot = frame.read_slot(0).unwrap();
    let vec = slot.resolutions[0].as_ref().unwrap();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (norm - 1.0).abs() < 1e-6,
        "Expected norm ≈ 1.0, got {}",
        norm
    );
}

#[test]
fn normalize_all_normalizes_all_active_slots() {
    let mut frame = TensorFrame::new();

    // Add two slots with two resolutions each
    let mut slot1 = SlotData::new(SlotRole::Agent);
    slot1.write_resolution(0, [1.0; SLOT_DIM]);
    slot1.write_resolution(1, [2.0; SLOT_DIM]);
    frame.write_slot(0, slot1).unwrap();

    let mut slot2 = SlotData::new(SlotRole::Predicate);
    slot2.write_resolution(0, [3.0; SLOT_DIM]);
    frame.write_slot(1, slot2).unwrap();

    frame.normalize_all().unwrap();

    // Check all resolutions are normalized
    for slot_idx in 0..2 {
        let slot = frame.read_slot(slot_idx).unwrap();
        for res in &slot.resolutions {
            if let Some(vec) = res {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn normalize_zero_vector_returns_error() {
    let mut frame = TensorFrame::new();
    frame
        .write_at(0, 0, SlotRole::Agent, [0.0; SLOT_DIM])
        .unwrap();

    let result = frame.normalize_slot(0, 0);
    assert!(result.is_err());
}

#[test]
fn normalize_with_mixed_values() {
    let mut frame = TensorFrame::new();

    // Create a vector with mixed values
    let mut data = [0.0_f32; SLOT_DIM];
    data[0] = 3.0;
    data[1] = 4.0;
    // Remaining values are 0.0

    frame.write_at(0, 0, SlotRole::Agent, data).unwrap();
    frame.normalize_slot(0, 0).unwrap();

    let slot = frame.read_slot(0).unwrap();
    let vec = slot.resolutions[0].as_ref().unwrap();

    // Original norm was sqrt(3^2 + 4^2) = 5
    // Normalized: [3/5, 4/5, 0, ..., 0] = [0.6, 0.8, 0, ..., 0]
    assert!((vec[0] - 0.6).abs() < 1e-6);
    assert!((vec[1] - 0.8).abs() < 1e-6);
    for &val in &vec[2..] {
        assert!(val.abs() < 1e-6);
    }

    // Verify unit norm
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn normalize_and_merge_workflow() {
    // Test a realistic workflow: normalize frames before merging
    let mut frame1 = TensorFrame::new();
    frame1
        .write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM])
        .unwrap();
    frame1.meta[0].certainty = 0.9;

    let mut frame2 = TensorFrame::new();
    frame2
        .write_at(1, 0, SlotRole::Predicate, [3.0; SLOT_DIM])
        .unwrap();
    frame2.meta[1].certainty = 0.85;

    // Normalize both frames
    frame1.normalize_all().unwrap();
    frame2.normalize_all().unwrap();

    // Merge
    let merged = frame1.merge(frame2);

    // Verify merged frame has normalized vectors
    assert_eq!(merged.active_slot_count(), 2);

    for slot_idx in 0..2 {
        let slot = merged.read_slot(slot_idx).unwrap();
        for res in &slot.resolutions {
            if let Some(vec) = res {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-6);
            }
        }
    }
}
