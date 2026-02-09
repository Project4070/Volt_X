//! Smoke test: verify all crates compile and basic types are accessible.

#[test]
fn volt_core_types_accessible() {
    let _frame = volt_core::TensorFrame::new();
    let _slot = volt_core::SlotData::new(volt_core::SlotRole::Agent);
    let _meta = volt_core::FrameMeta::default();
    let _err = volt_core::VoltError::Internal {
        message: "test".to_string(),
    };
}

#[test]
fn constants_match_architecture_spec() {
    // Architecture spec: S=16, R=4, D=256
    assert_eq!(volt_core::MAX_SLOTS, 16);
    assert_eq!(volt_core::NUM_RESOLUTIONS, 4);
    assert_eq!(volt_core::SLOT_DIM, 256);

    // Max frame size: 16 × 4 × 256 × 4 bytes = 64KB
    let max_bytes =
        volt_core::MAX_SLOTS * volt_core::NUM_RESOLUTIONS * volt_core::SLOT_DIM * 4;
    assert_eq!(max_bytes, 65536);
}
