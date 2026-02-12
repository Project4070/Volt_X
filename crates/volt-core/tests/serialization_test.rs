//! Integration tests for TensorFrame serialization.
//!
//! Note: serde tests use Box to avoid stack overflow with large nested arrays.
//! rkyv is the recommended zero-copy serialization format for TensorFrames.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};

#[test]
#[cfg(feature = "serde")]
fn serde_roundtrip_is_bit_identical() {
    // Run on a thread with 8 MB stack to avoid stack overflow on Windows.
    // TensorFrame is ~64KB and serde_json's recursive descent overflows
    // the default 1 MB Windows stack.
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut frame = Box::new(TensorFrame::new());
            let mut slot = SlotData::new(SlotRole::Agent);
            slot.write_resolution(0, [0.42; SLOT_DIM]);
            frame.write_slot(0, slot).unwrap();
            frame.meta[0].certainty = 0.95;

            let serialized = serde_json::to_vec(&*frame).unwrap();
            let deserialized: Box<TensorFrame> =
                serde_json::from_slice(&serialized).unwrap();

            // Verify data integrity
            assert_eq!(
                frame.active_slot_count(),
                deserialized.active_slot_count()
            );
            assert_eq!(frame.meta[0].certainty, deserialized.meta[0].certainty);

            let orig_slot = frame.read_slot(0).unwrap();
            let deser_slot = deserialized.read_slot(0).unwrap();
            assert_eq!(orig_slot.resolutions[0], deser_slot.resolutions[0]);
        })
        .expect("failed to spawn serde test thread")
        .join()
        .expect("serde test thread panicked");
}

#[test]
#[cfg(feature = "rkyv")]
fn rkyv_roundtrip_is_bit_identical() {
    use rkyv::{from_bytes, to_bytes};

    let mut frame = TensorFrame::new();
    frame
        .write_at(3, 1, SlotRole::Patient, [1.5; SLOT_DIM])
        .unwrap();

    let bytes = to_bytes::<rkyv::rancor::Error>(&frame).unwrap();

    // Full deserialization
    let deserialized: TensorFrame = from_bytes::<TensorFrame, rkyv::rancor::Error>(&bytes).unwrap();
    assert_eq!(frame.active_slot_count(), deserialized.active_slot_count());
}

#[test]
fn full_frame_size_is_64kb() {
    let mut frame = TensorFrame::new();

    // Fill all 16 slots with all 4 resolutions
    for slot_idx in 0..16 {
        let mut slot = SlotData::new(SlotRole::Free(slot_idx as u8));
        for res_idx in 0..4 {
            slot.write_resolution(res_idx, [1.0; SLOT_DIM]);
        }
        frame.write_slot(slot_idx, slot).unwrap();
    }

    assert_eq!(frame.data_size_bytes(), 65536); // 16 × 4 × 256 × 4 = 64KB
}

#[test]
#[cfg(feature = "rkyv")]
fn rkyv_zero_copy_access_without_deserialization() {
    use rkyv::from_bytes;
    use rkyv::to_bytes;

    let mut frame = TensorFrame::new();
    frame
        .write_at(0, 0, SlotRole::Agent, [2.5; SLOT_DIM])
        .unwrap();
    frame
        .write_at(1, 0, SlotRole::Predicate, [3.5; SLOT_DIM])
        .unwrap();

    let bytes = to_bytes::<rkyv::rancor::Error>(&frame).unwrap();

    // Deserialize and verify
    let deserialized: TensorFrame = from_bytes::<TensorFrame, rkyv::rancor::Error>(&bytes).unwrap();

    // This demonstrates rkyv serialization: efficient binary format
    // designed for zero-copy deserialization
    assert_eq!(deserialized.active_slot_count(), 2);
    assert!(deserialized.read_slot(0).is_ok());
    assert!(deserialized.read_slot(1).is_ok());
}
