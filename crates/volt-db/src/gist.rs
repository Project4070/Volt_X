//! R₀ gist extraction from TensorFrames.
//!
//! A frame gist is a single 256-dim unit vector summarizing a frame's
//! discourse-level content. It is computed by superposing all active
//! slot R₀ (resolution 0) embeddings via [`volt_bus::superpose`].
//!
//! Gists are the primary indexing key for HNSW semantic search and
//! the content stored in the Ghost Bleed Buffer.

use volt_core::{TensorFrame, VoltError, SLOT_DIM};

/// A frame gist: a single 256-dim unit vector summarizing the frame's R₀ content.
///
/// Produced by [`extract_gist`]. Contains the gist vector plus enough
/// metadata to correlate back to the source frame.
///
/// # Example
///
/// ```
/// use volt_db::gist::{extract_gist, FrameGist};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
/// frame.frame_meta.frame_id = 42;
/// frame.frame_meta.strand_id = 1;
///
/// let gist = extract_gist(&frame).unwrap().unwrap();
/// assert_eq!(gist.frame_id, 42);
/// assert_eq!(gist.strand_id, 1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FrameGist {
    /// The R₀ gist vector (256 dims, L2-normalized).
    pub vector: [f32; SLOT_DIM],
    /// The source frame's unique ID.
    pub frame_id: u64,
    /// The strand this frame belongs to.
    pub strand_id: u64,
    /// Creation timestamp in microseconds.
    pub created_at: u64,
}

/// Extracts the R₀ gist from a TensorFrame.
///
/// Collects all active slots that have R₀ (resolution 0) data, then:
/// - If one slot has R₀: returns that vector, L2-normalized.
/// - If multiple slots have R₀: superposing via [`volt_bus::superpose`]
///   (element-wise sum + L2 normalize).
/// - If no slots have R₀: returns `Ok(None)`.
///
/// # Errors
///
/// Returns [`VoltError::BusError`] if superposition fails (e.g. zero vectors).
///
/// # Example
///
/// ```
/// use volt_db::gist::extract_gist;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
///
/// let gist = extract_gist(&frame).unwrap();
/// assert!(gist.is_some());
/// ```
pub fn extract_gist(frame: &TensorFrame) -> Result<Option<FrameGist>, VoltError> {
    // Collect all active R₀ vectors
    let r0_vecs: Vec<&[f32; SLOT_DIM]> = frame
        .slots
        .iter()
        .filter_map(|s| s.as_ref())
        .filter_map(|slot| slot.resolutions[0].as_ref())
        .collect();

    if r0_vecs.is_empty() {
        return Ok(None);
    }

    let vector = if r0_vecs.len() == 1 {
        // Single vector: L2 normalize
        let mut v = *r0_vecs[0];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return Ok(None);
        }
        for x in &mut v {
            *x /= norm;
        }
        v
    } else {
        // Multiple vectors: superpose (sum + L2 normalize)
        volt_bus::superpose(&r0_vecs)?
    };

    Ok(Some(FrameGist {
        vector,
        frame_id: frame.frame_meta.frame_id,
        strand_id: frame.frame_meta.strand_id,
        created_at: frame.frame_meta.created_at,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole};

    #[test]
    fn empty_frame_returns_none() {
        let frame = TensorFrame::new();
        let gist = extract_gist(&frame).unwrap();
        assert!(gist.is_none());
    }

    #[test]
    fn single_r0_slot_returns_normalized() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.frame_meta.frame_id = 7;
        frame.frame_meta.strand_id = 3;
        frame.frame_meta.created_at = 12345;

        let gist = extract_gist(&frame).unwrap().unwrap();
        assert_eq!(gist.frame_id, 7);
        assert_eq!(gist.strand_id, 3);
        assert_eq!(gist.created_at, 12345);

        // Check unit norm
        let norm: f32 = gist.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "gist should be unit normalized, got {norm}");
    }

    #[test]
    fn multiple_r0_slots_returns_superposed() {
        let mut frame = TensorFrame::new();

        // Slot 0: all 0.1
        let mut slot0 = SlotData::new(SlotRole::Agent);
        slot0.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot0).unwrap();

        // Slot 1: all 0.2
        let mut slot1 = SlotData::new(SlotRole::Predicate);
        slot1.write_resolution(0, [0.2; SLOT_DIM]);
        frame.write_slot(1, slot1).unwrap();

        let gist = extract_gist(&frame).unwrap().unwrap();

        // Superposed vector should be unit normalized
        let norm: f32 = gist.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "superposed gist should be unit normalized");
    }

    #[test]
    fn no_r0_only_r1_returns_none() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        // Only write R₁, not R₀
        slot.write_resolution(1, [0.5; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();

        let gist = extract_gist(&frame).unwrap();
        assert!(gist.is_none());
    }

    #[test]
    fn zero_r0_vector_returns_none() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.0; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();

        let gist = extract_gist(&frame).unwrap();
        assert!(gist.is_none(), "zero vector should yield None");
    }

    #[test]
    fn mixed_slots_some_with_r0_some_without() {
        let mut frame = TensorFrame::new();

        // Slot 0: has R₀
        let mut slot0 = SlotData::new(SlotRole::Agent);
        slot0.write_resolution(0, [0.3; SLOT_DIM]);
        frame.write_slot(0, slot0).unwrap();

        // Slot 1: only R₁
        let mut slot1 = SlotData::new(SlotRole::Predicate);
        slot1.write_resolution(1, [0.5; SLOT_DIM]);
        frame.write_slot(1, slot1).unwrap();

        // Slot 2: has R₀
        let mut slot2 = SlotData::new(SlotRole::Patient);
        slot2.write_resolution(0, [0.7; SLOT_DIM]);
        frame.write_slot(2, slot2).unwrap();

        let gist = extract_gist(&frame).unwrap().unwrap();
        // Should superpose slot 0 and slot 2 R₀ vectors (not slot 1)
        let norm: f32 = gist.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
