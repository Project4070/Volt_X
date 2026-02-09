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
}
