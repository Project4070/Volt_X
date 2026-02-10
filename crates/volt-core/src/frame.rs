//! The TensorFrame — the fundamental unit of thought in Volt X.
//!
//! A TensorFrame is a structured 3D tensor: `[S=16 slots × R=4 resolutions × D=256 dims]`.
//! Unlike flat vectors (0D) or token streams (1D), TensorFrames provide
//! inspectable, composable, multi-resolution representations of thoughts.

use crate::error::VoltError;
use crate::meta::FrameMeta;
use crate::slot::{SlotData, SlotMeta};
use crate::{MAX_SLOTS, NUM_RESOLUTIONS, SLOT_DIM};

/// The fundamental unit of thought in Volt X.
///
/// A structured 3D tensor: `[S=16 slots × R=4 resolutions × D=256 dims]`.
/// Most slots are sparse (empty). A simple thought uses ~4 slots × 2 resolutions = 8KB.
/// Maximum size when fully populated: 64KB.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::default();
/// assert!(frame.is_empty());
///
/// // Write an agent slot
/// let mut agent = SlotData::new(SlotRole::Agent);
/// agent.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, agent).unwrap();
///
/// assert!(!frame.is_empty());
/// assert_eq!(frame.active_slot_count(), 1);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct TensorFrame {
    /// Structured thought: `[slots × resolutions × dims]`.
    /// Sparse: most slots are `None` (empty).
    pub slots: [Option<SlotData>; MAX_SLOTS],

    /// Per-slot metadata (certainty, source, timestamp).
    pub meta: [SlotMeta; MAX_SLOTS],

    /// Frame-level metadata: strand, discourse type, global γ.
    pub frame_meta: FrameMeta,
}

impl Default for TensorFrame {
    fn default() -> Self {
        Self {
            slots: [const { None }; MAX_SLOTS],
            meta: std::array::from_fn(|_| SlotMeta::default()),
            frame_meta: FrameMeta::default(),
        }
    }
}

impl TensorFrame {
    /// Creates a new empty TensorFrame.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::TensorFrame;
    ///
    /// let frame = TensorFrame::new();
    /// assert!(frame.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if all slots are empty.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::TensorFrame;
    ///
    /// let frame = TensorFrame::new();
    /// assert!(frame.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }

    /// Returns the number of populated (non-empty) slots.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// assert_eq!(frame.active_slot_count(), 0);
    ///
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.5; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    /// assert_eq!(frame.active_slot_count(), 1);
    /// ```
    pub fn active_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Writes slot data at the given index.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::SlotOutOfRange`] if `index >= MAX_SLOTS`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, VoltError, MAX_SLOTS, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// let slot = SlotData::new(SlotRole::Predicate);
    /// assert!(frame.write_slot(1, slot).is_ok());
    ///
    /// let slot2 = SlotData::new(SlotRole::Agent);
    /// assert!(frame.write_slot(MAX_SLOTS, slot2).is_err());
    /// ```
    pub fn write_slot(&mut self, index: usize, data: SlotData) -> Result<(), VoltError> {
        if index >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index,
                max: MAX_SLOTS,
            });
        }
        self.slots[index] = Some(data);
        Ok(())
    }

    /// Reads slot data at the given index.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::SlotOutOfRange`] if `index >= MAX_SLOTS`.
    /// Returns [`VoltError::EmptySlot`] if the slot is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.5; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    ///
    /// let read = frame.read_slot(0).unwrap();
    /// assert_eq!(read.role, SlotRole::Agent);
    /// ```
    pub fn read_slot(&self, index: usize) -> Result<&SlotData, VoltError> {
        if index >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index,
                max: MAX_SLOTS,
            });
        }
        self.slots[index].as_ref().ok_or(VoltError::EmptySlot { index })
    }

    /// Clears a slot at the given index.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::SlotOutOfRange`] if `index >= MAX_SLOTS`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole};
    ///
    /// let mut frame = TensorFrame::new();
    /// frame.write_slot(0, SlotData::new(SlotRole::Agent)).unwrap();
    /// assert_eq!(frame.active_slot_count(), 1);
    ///
    /// frame.clear_slot(0).unwrap();
    /// assert_eq!(frame.active_slot_count(), 0);
    /// ```
    pub fn clear_slot(&mut self, index: usize) -> Result<(), VoltError> {
        if index >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index,
                max: MAX_SLOTS,
            });
        }
        self.slots[index] = None;
        self.meta[index] = SlotMeta::default();
        Ok(())
    }

    /// Returns the minimum certainty (γ) across all active slots.
    ///
    /// Returns `None` if no slots are active.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::TensorFrame;
    ///
    /// let frame = TensorFrame::new();
    /// assert!(frame.min_certainty().is_none());
    /// ```
    pub fn min_certainty(&self) -> Option<f32> {
        let active_gammas: Vec<f32> = self
            .slots
            .iter()
            .zip(self.meta.iter())
            .filter(|(slot, _)| slot.is_some())
            .map(|(_, meta)| meta.certainty)
            .collect();

        if active_gammas.is_empty() {
            None
        } else {
            active_gammas
                .into_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        }
    }

    /// Returns the approximate memory size in bytes of the populated data.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [1.0; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    ///
    /// // One slot with one resolution = 256 * 4 bytes = 1024 bytes
    /// assert_eq!(frame.data_size_bytes(), 1024);
    /// ```
    pub fn data_size_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|slot| {
                slot.resolutions
                    .iter()
                    .filter(|r| r.is_some())
                    .count()
                    * SLOT_DIM
                    * std::mem::size_of::<f32>()
            })
            .sum()
    }

    /// Writes a raw embedding at a specific slot and resolution.
    ///
    /// Creates the slot with the given role if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::SlotOutOfRange`] if `slot_index >= MAX_SLOTS`.
    /// Returns [`VoltError::ResolutionOutOfRange`] if `resolution >= NUM_RESOLUTIONS`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// let embedding = [0.42_f32; SLOT_DIM];
    /// frame.write_at(0, 1, SlotRole::Agent, embedding).unwrap();
    ///
    /// let slot = frame.read_slot(0).unwrap();
    /// assert!(slot.resolutions[1].is_some());
    /// ```
    pub fn write_at(
        &mut self,
        slot_index: usize,
        resolution: usize,
        role: crate::SlotRole,
        data: [f32; SLOT_DIM],
    ) -> Result<(), VoltError> {
        if slot_index >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index: slot_index,
                max: MAX_SLOTS,
            });
        }
        if resolution >= NUM_RESOLUTIONS {
            return Err(VoltError::ResolutionOutOfRange {
                index: resolution,
                max: NUM_RESOLUTIONS,
            });
        }

        let slot = self.slots[slot_index].get_or_insert_with(|| SlotData::new(role));
        slot.resolutions[resolution] = Some(data);
        Ok(())
    }

    /// Merges two TensorFrames, resolving slot conflicts by certainty (gamma).
    ///
    /// When both frames have data at the same slot index, the slot with higher
    /// certainty is kept. If certainties are equal, the slot from `self` is kept.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole};
    ///
    /// let mut frame1 = TensorFrame::new();
    /// frame1.write_slot(0, SlotData::new(SlotRole::Agent)).unwrap();
    /// frame1.meta[0].certainty = 0.9;
    ///
    /// let mut frame2 = TensorFrame::new();
    /// frame2.write_slot(1, SlotData::new(SlotRole::Predicate)).unwrap();
    ///
    /// let merged = frame1.merge(frame2);
    /// assert_eq!(merged.active_slot_count(), 2);
    /// ```
    pub fn merge(self, other: TensorFrame) -> TensorFrame {
        let mut merged = TensorFrame::new();

        // Merge each slot
        for i in 0..MAX_SLOTS {
            match (&self.slots[i], &other.slots[i]) {
                (Some(_), Some(_)) => {
                    // Conflict: resolve by certainty
                    if self.meta[i].certainty >= other.meta[i].certainty {
                        merged.slots[i] = self.slots[i].clone();
                        merged.meta[i] = self.meta[i].clone();
                    } else {
                        merged.slots[i] = other.slots[i].clone();
                        merged.meta[i] = other.meta[i].clone();
                    }
                }
                (Some(_), None) => {
                    merged.slots[i] = self.slots[i].clone();
                    merged.meta[i] = self.meta[i].clone();
                }
                (None, Some(_)) => {
                    merged.slots[i] = other.slots[i].clone();
                    merged.meta[i] = other.meta[i].clone();
                }
                (None, None) => {} // Both empty
            }
        }

        // Merge frame metadata
        merged.frame_meta = Self::merge_frame_meta(&self.frame_meta, &other.frame_meta);
        merged.frame_meta.global_certainty = merged.min_certainty().unwrap_or(0.0);

        merged
    }

    /// Merges frame metadata from two frames.
    fn merge_frame_meta(left: &FrameMeta, right: &FrameMeta) -> FrameMeta {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        FrameMeta {
            frame_id: 0, // Generate new ID (TODO: use proper ID generation in future)
            strand_id: left.strand_id, // Prefer left
            global_certainty: 0.0,      // Recalculated after merge
            discourse_type: left.discourse_type, // Prefer left
            created_at: now,
            rar_iterations: left.rar_iterations + right.rar_iterations,
            verified: false, // Merged frames need re-verification
            proof_length: left.proof_length.max(right.proof_length),
        }
    }

    /// Normalizes a specific slot's embedding at a given resolution to unit length.
    ///
    /// L2 normalization: v' = v / ||v||₂ where ||v||₂ = sqrt(Σ v_i²)
    ///
    /// # Errors
    ///
    /// Returns `VoltError::SlotOutOfRange` if slot_index >= MAX_SLOTS.
    /// Returns `VoltError::ResolutionOutOfRange` if resolution >= NUM_RESOLUTIONS.
    /// Returns `VoltError::EmptySlot` if the slot is empty.
    /// Returns `VoltError::FrameError` if the vector is zero or near-zero (norm < 1e-10).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// frame.write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM]).unwrap();
    /// frame.normalize_slot(0, 0).unwrap();
    ///
    /// let slot = frame.read_slot(0).unwrap();
    /// let vec = slot.resolutions[0].as_ref().unwrap();
    /// let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    /// assert!((norm - 1.0).abs() < 1e-6);
    /// ```
    pub fn normalize_slot(
        &mut self,
        slot_index: usize,
        resolution: usize,
    ) -> Result<(), VoltError> {
        // Validate indices
        if slot_index >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index: slot_index,
                max: MAX_SLOTS,
            });
        }
        if resolution >= NUM_RESOLUTIONS {
            return Err(VoltError::ResolutionOutOfRange {
                index: resolution,
                max: NUM_RESOLUTIONS,
            });
        }

        // Get mutable reference to slot
        let slot = self.slots[slot_index]
            .as_mut()
            .ok_or(VoltError::EmptySlot { index: slot_index })?;

        // Get mutable reference to resolution
        let vec = slot.resolutions[resolution].as_mut().ok_or(
            VoltError::FrameError {
                message: format!(
                    "resolution {} is empty in slot {}",
                    resolution, slot_index
                ),
            },
        )?;

        // Calculate L2 norm
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Check for zero or near-zero vector
        if norm < 1e-10 {
            return Err(VoltError::FrameError {
                message: format!(
                    "cannot normalize zero vector at slot {}, resolution {}",
                    slot_index, resolution
                ),
            });
        }

        // Normalize in-place
        for x in vec.iter_mut() {
            *x /= norm;
        }

        Ok(())
    }

    /// Normalizes all populated resolutions in all active slots.
    ///
    /// Skips empty slots and empty resolutions. If any normalization fails,
    /// returns early with the error (partial normalization may occur).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut frame = TensorFrame::new();
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [1.0; SLOT_DIM]);
    /// slot.write_resolution(1, [2.0; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    ///
    /// frame.normalize_all().unwrap();
    /// // All populated resolutions now have unit norm
    /// ```
    pub fn normalize_all(&mut self) -> Result<(), VoltError> {
        for slot_idx in 0..MAX_SLOTS {
            if self.slots[slot_idx].is_some() {
                for res_idx in 0..NUM_RESOLUTIONS {
                    let has_resolution = self.slots[slot_idx]
                        .as_ref()
                        .is_some_and(|slot| slot.resolutions[res_idx].is_some());
                    if has_resolution {
                        self.normalize_slot(slot_idx, res_idx)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::SlotRole;

    #[test]
    fn new_frame_is_empty() {
        let frame = TensorFrame::new();
        assert!(frame.is_empty());
        assert_eq!(frame.active_slot_count(), 0);
    }

    #[test]
    fn write_and_read_slot() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.5; SLOT_DIM]);

        frame.write_slot(0, slot).unwrap();
        assert_eq!(frame.active_slot_count(), 1);

        let read = frame.read_slot(0).unwrap();
        assert_eq!(read.role, SlotRole::Agent);
        assert!(read.resolutions[0].is_some());
    }

    #[test]
    fn slot_out_of_range() {
        let mut frame = TensorFrame::new();
        let slot = SlotData::new(SlotRole::Agent);
        let result = frame.write_slot(MAX_SLOTS, slot);
        assert!(result.is_err());
    }

    #[test]
    fn read_empty_slot_errors() {
        let frame = TensorFrame::new();
        let result = frame.read_slot(0);
        assert!(result.is_err());
    }

    #[test]
    fn clear_slot_works() {
        let mut frame = TensorFrame::new();
        frame
            .write_slot(0, SlotData::new(SlotRole::Agent))
            .unwrap();
        assert_eq!(frame.active_slot_count(), 1);

        frame.clear_slot(0).unwrap();
        assert_eq!(frame.active_slot_count(), 0);
    }

    #[test]
    fn min_certainty_empty_is_none() {
        let frame = TensorFrame::new();
        assert!(frame.min_certainty().is_none());
    }

    #[test]
    fn min_certainty_returns_smallest() {
        let mut frame = TensorFrame::new();
        frame
            .write_slot(0, SlotData::new(SlotRole::Agent))
            .unwrap();
        frame
            .write_slot(1, SlotData::new(SlotRole::Predicate))
            .unwrap();
        frame.meta[0].certainty = 0.95;
        frame.meta[1].certainty = 0.78;

        assert_eq!(frame.min_certainty(), Some(0.78));
    }

    #[test]
    fn data_size_bytes_calculation() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [1.0; SLOT_DIM]);
        slot.write_resolution(1, [1.0; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();

        // 2 resolutions × 256 dims × 4 bytes = 2048
        assert_eq!(frame.data_size_bytes(), 2048);
    }

    #[test]
    fn write_at_creates_slot_if_missing() {
        let mut frame = TensorFrame::new();
        frame
            .write_at(3, 1, SlotRole::Patient, [0.42; SLOT_DIM])
            .unwrap();

        let slot = frame.read_slot(3).unwrap();
        assert_eq!(slot.role, SlotRole::Patient);
        assert!(slot.resolutions[1].is_some());
        assert!(slot.resolutions[0].is_none());
    }

    #[test]
    fn merge_both_empty_produces_empty() {
        let frame1 = TensorFrame::new();
        let frame2 = TensorFrame::new();

        let merged = frame1.merge(frame2);
        assert!(merged.is_empty());
        assert_eq!(merged.active_slot_count(), 0);
    }

    #[test]
    fn merge_one_empty_returns_other() {
        let mut frame1 = TensorFrame::new();
        frame1.write_slot(0, SlotData::new(SlotRole::Agent)).unwrap();
        frame1.meta[0].certainty = 0.8;
        let frame2 = TensorFrame::new();

        let merged = frame1.merge(frame2);
        assert_eq!(merged.active_slot_count(), 1);
        assert_eq!(merged.meta[0].certainty, 0.8);
    }

    #[test]
    fn merge_no_overlap_combines_all_slots() {
        let mut frame1 = TensorFrame::new();
        frame1.write_slot(0, SlotData::new(SlotRole::Agent)).unwrap();

        let mut frame2 = TensorFrame::new();
        frame2.write_slot(1, SlotData::new(SlotRole::Predicate)).unwrap();

        let merged = frame1.merge(frame2);
        assert_eq!(merged.active_slot_count(), 2);
        assert!(merged.read_slot(0).is_ok());
        assert!(merged.read_slot(1).is_ok());
    }

    #[test]
    fn merge_conflict_resolved_by_higher_certainty() {
        let mut frame1 = TensorFrame::new();
        frame1.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
        frame1.meta[0].certainty = 0.9;

        let mut frame2 = TensorFrame::new();
        frame2.write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM]).unwrap();
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
        frame1.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
        frame1.meta[0].certainty = 0.8;

        let mut frame2 = TensorFrame::new();
        frame2.write_at(0, 0, SlotRole::Agent, [2.0; SLOT_DIM]).unwrap();
        frame2.meta[0].certainty = 0.8;

        let merged = frame1.merge(frame2);

        // Equal certainty → prefer left (frame1)
        let slot = merged.read_slot(0).unwrap();
        assert_eq!(slot.resolutions[0].unwrap()[0], 1.0);
    }

    #[test]
    fn merge_recalculates_global_certainty() {
        let mut frame1 = TensorFrame::new();
        frame1.write_slot(0, SlotData::new(SlotRole::Agent)).unwrap();
        frame1.meta[0].certainty = 0.95;

        let mut frame2 = TensorFrame::new();
        frame2.write_slot(1, SlotData::new(SlotRole::Predicate)).unwrap();
        frame2.meta[1].certainty = 0.78;

        let merged = frame1.merge(frame2);

        // Global certainty should be min of all active slots
        assert_eq!(merged.frame_meta.global_certainty, 0.78);
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
    fn normalize_zero_vector_returns_error() {
        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, [0.0; SLOT_DIM])
            .unwrap();

        let result = frame.normalize_slot(0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn normalize_slot_out_of_range_errors() {
        let mut frame = TensorFrame::new();
        let result = frame.normalize_slot(MAX_SLOTS, 0);
        assert!(result.is_err());

        let result2 = frame.normalize_slot(0, NUM_RESOLUTIONS);
        assert!(result2.is_err());
    }

    #[test]
    fn normalize_empty_slot_errors() {
        let mut frame = TensorFrame::new();
        let result = frame.normalize_slot(0, 0);
        assert!(result.is_err());
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
    fn normalize_all_skips_empty_slots() {
        let mut frame = TensorFrame::new();

        // Add only one slot
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [2.0; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();

        // normalize_all should not error on empty slots
        frame.normalize_all().unwrap();

        let slot = frame.read_slot(0).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_already_normalized_vector_unchanged() {
        let mut frame = TensorFrame::new();

        // Create a vector that's already normalized
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = 1.0; // Unit vector along first axis

        frame.write_at(0, 0, SlotRole::Agent, data).unwrap();
        frame.normalize_slot(0, 0).unwrap();

        let slot = frame.read_slot(0).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();

        // Should still be approximately [1, 0, 0, ..., 0]
        assert!((vec[0] - 1.0).abs() < 1e-6);
        for &val in &vec[1..] {
            assert!(val.abs() < 1e-6);
        }
    }
}
