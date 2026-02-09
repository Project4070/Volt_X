//! Integration test: TensorFrame round-trip through bus operations.
//!
//! Tests that a frame can be created, written, read, and manipulated
//! without data corruption.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};

#[test]
fn frame_create_write_read_roundtrip() {
    let mut frame = TensorFrame::new();

    // Write agent slot at R₀
    let embedding = [0.42_f32; SLOT_DIM];
    frame
        .write_at(0, 0, SlotRole::Agent, embedding)
        .expect("write_at should succeed");

    // Write predicate slot at R₀ and R₁
    let mut pred = SlotData::new(SlotRole::Predicate);
    pred.write_resolution(0, [0.7; SLOT_DIM]);
    pred.write_resolution(1, [0.3; SLOT_DIM]);
    frame.write_slot(1, pred).expect("write_slot should succeed");

    // Verify
    assert_eq!(frame.active_slot_count(), 2);
    assert_eq!(frame.read_slot(0).unwrap().role, SlotRole::Agent);
    assert_eq!(frame.read_slot(1).unwrap().role, SlotRole::Predicate);
    assert_eq!(frame.read_slot(1).unwrap().active_resolution_count(), 2);

    // Verify data integrity
    let agent_data = frame.read_slot(0).unwrap().resolutions[0].unwrap();
    assert!((agent_data[0] - 0.42).abs() < f32::EPSILON);
}

#[test]
fn frame_data_size_tracks_actual_content() {
    let mut frame = TensorFrame::new();

    // Empty frame = 0 bytes
    assert_eq!(frame.data_size_bytes(), 0);

    // One slot, one resolution = 256 × 4 = 1024 bytes
    frame
        .write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM])
        .unwrap();
    assert_eq!(frame.data_size_bytes(), 1024);

    // Add another resolution to same slot = 2048 bytes
    frame
        .write_at(0, 1, SlotRole::Agent, [1.0; SLOT_DIM])
        .unwrap();
    assert_eq!(frame.data_size_bytes(), 2048);

    // Add a second slot = 3072 bytes
    frame
        .write_at(5, 0, SlotRole::Manner, [1.0; SLOT_DIM])
        .unwrap();
    assert_eq!(frame.data_size_bytes(), 3072);
}

#[test]
fn frame_min_certainty_across_slots() {
    let mut frame = TensorFrame::new();

    frame
        .write_slot(0, SlotData::new(SlotRole::Agent))
        .unwrap();
    frame
        .write_slot(1, SlotData::new(SlotRole::Predicate))
        .unwrap();
    frame
        .write_slot(2, SlotData::new(SlotRole::Patient))
        .unwrap();

    frame.meta[0].certainty = 0.95;
    frame.meta[1].certainty = 0.78;
    frame.meta[2].certainty = 0.82;

    // γ_final = min(all slot γ)
    assert_eq!(frame.min_certainty(), Some(0.78));
}
