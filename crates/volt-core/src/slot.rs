//! Slot-level data structures for TensorFrames.
//!
//! Each slot in a TensorFrame holds multi-resolution embeddings
//! for a single semantic role (Agent, Predicate, Patient, etc.).

use crate::{NUM_RESOLUTIONS, SLOT_DIM};

#[cfg(feature = "serde")]
mod serde_helpers {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        arr: &[Option<[f32; SLOT_DIM]>; NUM_RESOLUTIONS],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert to Vec for serialization
        let vec: Vec<Option<Vec<f32>>> = arr
            .iter()
            .map(|opt| opt.as_ref().map(|a| a.to_vec()))
            .collect();
        vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<[Option<[f32; SLOT_DIM]>; NUM_RESOLUTIONS], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<Option<Vec<f32>>> = Vec::deserialize(deserializer)?;

        let mut arr = [const { None }; NUM_RESOLUTIONS];
        for (i, opt_vec) in vec.into_iter().enumerate().take(NUM_RESOLUTIONS) {
            arr[i] = opt_vec.and_then(|v| {
                if v.len() == SLOT_DIM {
                    let mut fixed = [0.0_f32; SLOT_DIM];
                    fixed.copy_from_slice(&v);
                    Some(fixed)
                } else {
                    None
                }
            });
        }

        Ok(arr)
    }
}


/// Multi-resolution embedding data for a single slot.
///
/// Each slot can hold embeddings at up to 4 resolution levels:
/// - R₀: Discourse (coarsest — topic gist)
/// - R₁: Proposition (sentence-level)
/// - R₂: Phrase (detail-level)
/// - R₃: Token (finest — BPE subwords for output)
///
/// Slots are sparse: most resolutions are `None` (empty) in practice.
///
/// # Example
///
/// ```
/// use volt_core::slot::{SlotData, SlotRole};
/// use volt_core::SLOT_DIM;
///
/// let mut slot = SlotData::new(SlotRole::Agent);
/// assert!(slot.resolutions[0].is_none());
///
/// // Write a discourse-level embedding
/// let embedding = [0.1_f32; SLOT_DIM];
/// slot.write_resolution(0, embedding);
/// assert!(slot.resolutions[0].is_some());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct SlotData {
    /// Multi-resolution embeddings for this slot.
    /// R0=discourse, R1=proposition, R2=phrase, R3=token.
    #[cfg_attr(feature = "serde", serde(with = "serde_helpers"))]
    pub resolutions: [Option<[f32; SLOT_DIM]>; NUM_RESOLUTIONS],

    /// The semantic role assigned to this slot.
    pub role: SlotRole,

    /// Codebook address (if quantized via VQ-VAE).
    pub codebook_id: Option<u16>,
}

impl SlotData {
    /// Creates a new empty `SlotData` with the given role.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::slot::{SlotData, SlotRole};
    ///
    /// let slot = SlotData::new(SlotRole::Predicate);
    /// assert_eq!(slot.role, SlotRole::Predicate);
    /// assert!(slot.codebook_id.is_none());
    /// ```
    pub fn new(role: SlotRole) -> Self {
        Self {
            resolutions: [const { None }; NUM_RESOLUTIONS],
            role,
            codebook_id: None,
        }
    }

    /// Writes an embedding at the given resolution level.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::slot::{SlotData, SlotRole};
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [1.0; SLOT_DIM]);
    /// assert!(slot.resolutions[0].is_some());
    /// ```
    pub fn write_resolution(&mut self, resolution: usize, data: [f32; SLOT_DIM]) {
        if resolution < NUM_RESOLUTIONS {
            self.resolutions[resolution] = Some(data);
        }
    }

    /// Returns the number of populated resolution levels.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::slot::{SlotData, SlotRole};
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// assert_eq!(slot.active_resolution_count(), 0);
    /// slot.write_resolution(0, [1.0; SLOT_DIM]);
    /// assert_eq!(slot.active_resolution_count(), 1);
    /// ```
    pub fn active_resolution_count(&self) -> usize {
        self.resolutions.iter().filter(|r| r.is_some()).count()
    }
}

/// Semantic role assignment for a TensorFrame slot.
///
/// The first 9 roles are fixed semantic roles from linguistic theory.
/// `Free(u8)` allows 7 additional domain-specific extensions (indices 9-15).
///
/// # Example
///
/// ```
/// use volt_core::slot::SlotRole;
///
/// let role = SlotRole::Agent;
/// assert_ne!(role, SlotRole::Predicate);
///
/// let custom = SlotRole::Free(0);
/// assert_eq!(custom, SlotRole::Free(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub enum SlotRole {
    /// The entity performing the action (e.g., "user").
    Agent,
    /// The action or state (e.g., "has_bug").
    Predicate,
    /// The entity affected by the action (e.g., "lifetime_code").
    Patient,
    /// Where the action takes place.
    Location,
    /// When the action takes place.
    Time,
    /// How the action is performed (e.g., "borrow_check").
    Manner,
    /// The tool or means used.
    Instrument,
    /// Why the action happened.
    Cause,
    /// The outcome of the action.
    Result,
    /// Domain-specific extension slot (indices 9-15).
    Free(u8),
}

/// Per-slot metadata tracking certainty, source, and timestamps.
///
/// # Example
///
/// ```
/// use volt_core::slot::SlotMeta;
///
/// let meta = SlotMeta::default();
/// assert_eq!(meta.certainty, 0.0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct SlotMeta {
    /// Per-slot certainty score (gamma, γ). Range: 0.0 to 1.0.
    /// 0.0 = completely uncertain, 1.0 = fully certain.
    pub certainty: f32,

    /// Source identifier for this slot's data.
    pub source: SlotSource,

    /// Timestamp (microseconds since epoch) when this slot was last updated.
    pub updated_at: u64,

    /// Whether this slot needs verification by the Hard Core.
    pub needs_verify: bool,
}

impl Default for SlotMeta {
    fn default() -> Self {
        Self {
            certainty: 0.0,
            source: SlotSource::Empty,
            updated_at: 0,
            needs_verify: false,
        }
    }
}

/// Where a slot's data originated from.
///
/// # Example
///
/// ```
/// use volt_core::slot::SlotSource;
///
/// let source = SlotSource::Translator;
/// assert_ne!(source, SlotSource::SoftCore);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub enum SlotSource {
    /// Slot is empty / unused.
    Empty,
    /// Data came from an input translator.
    Translator,
    /// Data was generated by the GPU Soft Core.
    SoftCore,
    /// Data was produced by the CPU Hard Core.
    HardCore,
    /// Data was recalled from VoltDB memory.
    Memory,
    /// Data came from a personal/user strand.
    Personal,
}
