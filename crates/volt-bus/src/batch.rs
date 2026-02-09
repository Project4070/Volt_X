//! Batch operations for applying HDC algebra to entire TensorFrames.
//!
//! These operations apply single-vector operations (bind, unbind, similarity)
//! to all corresponding slots/resolutions in TensorFrames.

use volt_core::slot::SlotSource;
use volt_core::{SlotData, TensorFrame, VoltError, MAX_SLOTS, NUM_RESOLUTIONS};

/// Apply bind operation to all corresponding slots/resolutions in two frames.
///
/// For each slot/resolution pair:
/// - If both frames have data at that position, bind them
/// - If only one has data, copy it unchanged
/// - If neither has data, leave empty
///
/// # Metadata Handling
///
/// - **Certainty**: Uses min-rule: `γ_result = min(γ_a, γ_b)`
/// - **Source**: Set to `SlotSource::HardCore` (from algebra operation)
///
/// # Example
///
/// ```
/// use volt_bus::bind_frames;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame_a = TensorFrame::new();
/// frame_a.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let mut frame_b = TensorFrame::new();
/// frame_b.write_at(0, 0, SlotRole::Agent, [0.5; SLOT_DIM]).unwrap();
///
/// let bound = bind_frames(&frame_a, &frame_b).unwrap();
/// assert_eq!(bound.active_slot_count(), 1);
/// ```
pub fn bind_frames(frame_a: &TensorFrame, frame_b: &TensorFrame) -> Result<TensorFrame, VoltError> {
    let mut result = TensorFrame::new();

    for slot_idx in 0..MAX_SLOTS {
        let slot_a = &frame_a.slots[slot_idx];
        let slot_b = &frame_b.slots[slot_idx];

        match (slot_a, slot_b) {
            (Some(data_a), Some(data_b)) => {
                // Both slots exist: bind corresponding resolutions
                let mut result_slot_data = SlotData::new(data_a.role);

                for res_idx in 0..NUM_RESOLUTIONS {
                    if let (Some(vec_a), Some(vec_b)) =
                        (&data_a.resolutions[res_idx], &data_b.resolutions[res_idx])
                    {
                        let bound = crate::ops::bind(vec_a, vec_b)?;
                        result_slot_data.write_resolution(res_idx, bound);
                    }
                }

                result.write_slot(slot_idx, result_slot_data)?;

                // Metadata: min certainty (conservative propagation)
                result.meta[slot_idx].certainty = frame_a.meta[slot_idx]
                    .certainty
                    .min(frame_b.meta[slot_idx].certainty);
                result.meta[slot_idx].source = SlotSource::HardCore;
            }
            (Some(data_a), None) => {
                // Only frame_a has data: copy unchanged
                result.write_slot(slot_idx, data_a.clone())?;
                result.meta[slot_idx] = frame_a.meta[slot_idx].clone();
            }
            (None, Some(data_b)) => {
                // Only frame_b has data: copy unchanged
                result.write_slot(slot_idx, data_b.clone())?;
                result.meta[slot_idx] = frame_b.meta[slot_idx].clone();
            }
            (None, None) => {
                // Both empty: leave empty
            }
        }
    }

    Ok(result)
}

/// Apply unbind operation to all corresponding slots/resolutions in two frames.
///
/// For each slot/resolution pair:
/// - If both frames have data at that position, unbind them
/// - If only frame_c has data, copy it unchanged
/// - If neither has data, leave empty
///
/// # Metadata Handling
///
/// - **Certainty**: Uses min-rule: `γ_result = min(γ_c, γ_a)`
/// - **Source**: Set to `SlotSource::HardCore` (from algebra operation)
///
/// # Example
///
/// ```
/// use volt_bus::{bind_frames, unbind_frames};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame_a = TensorFrame::new();
/// frame_a.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let mut frame_b = TensorFrame::new();
/// frame_b.write_at(0, 0, SlotRole::Agent, [0.5; SLOT_DIM]).unwrap();
///
/// let bound = bind_frames(&frame_a, &frame_b).unwrap();
/// let recovered = unbind_frames(&bound, &frame_a).unwrap();
///
/// assert_eq!(recovered.active_slot_count(), 1);
/// ```
pub fn unbind_frames(
    frame_c: &TensorFrame,
    frame_a: &TensorFrame,
) -> Result<TensorFrame, VoltError> {
    let mut result = TensorFrame::new();

    for slot_idx in 0..MAX_SLOTS {
        let slot_c = &frame_c.slots[slot_idx];
        let slot_a = &frame_a.slots[slot_idx];

        match (slot_c, slot_a) {
            (Some(data_c), Some(data_a)) => {
                // Both slots exist: unbind corresponding resolutions
                let mut result_slot_data = SlotData::new(data_c.role);

                for res_idx in 0..NUM_RESOLUTIONS {
                    if let (Some(vec_c), Some(vec_a)) =
                        (&data_c.resolutions[res_idx], &data_a.resolutions[res_idx])
                    {
                        let unbound = crate::ops::unbind(vec_c, vec_a)?;
                        result_slot_data.write_resolution(res_idx, unbound);
                    }
                }

                result.write_slot(slot_idx, result_slot_data)?;

                // Metadata: min certainty
                result.meta[slot_idx].certainty = frame_c.meta[slot_idx]
                    .certainty
                    .min(frame_a.meta[slot_idx].certainty);
                result.meta[slot_idx].source = SlotSource::HardCore;
            }
            (Some(data_c), None) => {
                // Only frame_c has data: copy unchanged
                result.write_slot(slot_idx, data_c.clone())?;
                result.meta[slot_idx] = frame_c.meta[slot_idx].clone();
            }
            (None, Some(_)) => {
                // Only frame_a has data: cannot unbind, leave empty
            }
            (None, None) => {
                // Both empty: leave empty
            }
        }
    }

    Ok(result)
}

/// Compute per-slot similarity between two frames at R0 (discourse level).
///
/// Returns a vector of length `MAX_SLOTS` containing:
/// - `Some(similarity)` if both frames have data at that slot's R0
/// - `None` if either frame is missing data at that slot's R0
///
/// # Example
///
/// ```
/// use volt_bus::similarity_frames;
/// use volt_core::{TensorFrame, SlotRole, SLOT_DIM, MAX_SLOTS};
///
/// let mut frame_a = TensorFrame::new();
/// frame_a.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let mut frame_b = TensorFrame::new();
/// frame_b.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let similarities = similarity_frames(&frame_a, &frame_b);
/// assert_eq!(similarities.len(), MAX_SLOTS);
/// assert!(similarities[0].unwrap() > 0.99); // Identical at slot 0
/// assert!(similarities[1].is_none()); // Empty at slot 1
/// ```
pub fn similarity_frames(frame_a: &TensorFrame, frame_b: &TensorFrame) -> Vec<Option<f32>> {
    let mut results = Vec::with_capacity(MAX_SLOTS);

    for slot_idx in 0..MAX_SLOTS {
        // Compare at R0 (discourse level) for coarse similarity
        let sim = match (&frame_a.slots[slot_idx], &frame_b.slots[slot_idx]) {
            (Some(data_a), Some(data_b)) => {
                match (&data_a.resolutions[0], &data_b.resolutions[0]) {
                    (Some(vec_a), Some(vec_b)) => Some(crate::ops::similarity(vec_a, vec_b)),
                    _ => None,
                }
            }
            _ => None,
        };
        results.push(sim);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotRole, SLOT_DIM};

    fn test_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut v = [0.0; SLOT_DIM];
        // Use hash-like mixing for true pseudo-randomness
        for i in 0..SLOT_DIM {
            let mut h = seed.wrapping_mul(0xd2b74407b1ce6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn bind_frames_both_active() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();
        frame_a.meta[0].certainty = 0.9;

        let mut frame_b = TensorFrame::new();
        frame_b
            .write_at(0, 0, SlotRole::Agent, test_vector(99))
            .unwrap();
        frame_b.meta[0].certainty = 0.8;

        let bound = bind_frames(&frame_a, &frame_b).unwrap();

        assert_eq!(bound.active_slot_count(), 1);
        assert_eq!(bound.meta[0].certainty, 0.8); // Min of 0.9 and 0.8
        assert_eq!(bound.meta[0].source, SlotSource::HardCore);
    }

    #[test]
    fn bind_frames_one_empty() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();

        let frame_b = TensorFrame::new(); // Empty

        let bound = bind_frames(&frame_a, &frame_b).unwrap();

        assert_eq!(bound.active_slot_count(), 1);
        // Should just copy frame_a's data
    }

    #[test]
    fn bind_frames_no_overlap() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();

        let mut frame_b = TensorFrame::new();
        frame_b
            .write_at(1, 0, SlotRole::Predicate, test_vector(99))
            .unwrap();

        let bound = bind_frames(&frame_a, &frame_b).unwrap();

        assert_eq!(bound.active_slot_count(), 2);
    }

    #[test]
    fn unbind_frames_recovers_original() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();

        let mut frame_b = TensorFrame::new();
        frame_b
            .write_at(0, 0, SlotRole::Agent, test_vector(99))
            .unwrap();

        let bound = bind_frames(&frame_a, &frame_b).unwrap();
        let recovered = unbind_frames(&bound, &frame_a).unwrap();

        // Check that slot 0, R0 was recovered with high similarity
        let orig = frame_b.read_slot(0).unwrap();
        let recov = recovered.read_slot(0).unwrap();

        let sim = crate::ops::similarity(
            orig.resolutions[0].as_ref().unwrap(),
            recov.resolutions[0].as_ref().unwrap(),
        );

        assert!(sim > 0.85, "Recovery similarity should be > 0.85, got {}", sim);
    }

    #[test]
    fn similarity_frames_identical() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();
        frame_a
            .write_at(1, 0, SlotRole::Predicate, test_vector(99))
            .unwrap();

        let similarities = similarity_frames(&frame_a, &frame_a);

        assert_eq!(similarities.len(), MAX_SLOTS);
        assert!((similarities[0].unwrap() - 1.0).abs() < 1e-6);
        assert!((similarities[1].unwrap() - 1.0).abs() < 1e-6);
        assert!(similarities[2].is_none());
    }

    #[test]
    fn similarity_frames_partial_overlap() {
        let mut frame_a = TensorFrame::new();
        frame_a
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();

        let mut frame_b = TensorFrame::new();
        frame_b
            .write_at(0, 0, SlotRole::Agent, test_vector(42))
            .unwrap();
        frame_b
            .write_at(1, 0, SlotRole::Predicate, test_vector(99))
            .unwrap();

        let similarities = similarity_frames(&frame_a, &frame_b);

        assert!(similarities[0].is_some()); // Both have slot 0
        assert!(similarities[1].is_none()); // Only frame_b has slot 1
    }

    #[test]
    fn batch_operations_multi_resolution() {
        let mut frame_a = TensorFrame::new();
        let mut slot_a = SlotData::new(SlotRole::Agent);
        slot_a.write_resolution(0, test_vector(42));
        slot_a.write_resolution(1, test_vector(43));
        frame_a.write_slot(0, slot_a).unwrap();

        let mut frame_b = TensorFrame::new();
        let mut slot_b = SlotData::new(SlotRole::Agent);
        slot_b.write_resolution(0, test_vector(99));
        slot_b.write_resolution(1, test_vector(100));
        frame_b.write_slot(0, slot_b).unwrap();

        let bound = bind_frames(&frame_a, &frame_b).unwrap();

        // Both resolutions should be bound
        let bound_slot = bound.read_slot(0).unwrap();
        assert!(bound_slot.resolutions[0].is_some());
        assert!(bound_slot.resolutions[1].is_some());
    }
}
