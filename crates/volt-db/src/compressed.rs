//! Compressed frame types for Tier 2 storage and the GC decay pipeline.
//!
//! TensorFrames decay through four levels:
//! - **Full**: Complete frame with all 4 resolutions (~64KB)
//! - **Compressed**: R₀ + R₁ only (~8KB typical)
//! - **Gist**: R₀ only (~1KB typical)
//! - **Tombstoned**: Metadata only (32 bytes)
//!
//! This module defines the compressed representations and conversions.

use serde::{Deserialize, Serialize};
use volt_core::meta::DiscourseType;
use volt_core::slot::SlotRole;
use volt_core::{TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

/// The decay level of a frame in the GC pipeline.
///
/// Levels are ordered: Full > Compressed > Gist > Tombstoned.
/// GC only demotes frames (never promotes).
///
/// # Example
///
/// ```
/// use volt_db::compressed::DecayLevel;
///
/// assert!(DecayLevel::Full > DecayLevel::Compressed);
/// assert!(DecayLevel::Compressed > DecayLevel::Gist);
/// assert!(DecayLevel::Gist > DecayLevel::Tombstoned);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DecayLevel {
    /// Deleted, only metadata remains.
    Tombstoned = 0,
    /// R₀ gist only (~1KB).
    Gist = 1,
    /// R₀ + R₁ (~8KB typical).
    Compressed = 2,
    /// Full TensorFrame with all resolutions (~64KB).
    Full = 3,
}

impl DecayLevel {
    /// Converts a `u8` tag to a `DecayLevel`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::compressed::DecayLevel;
    ///
    /// assert_eq!(DecayLevel::from_tag(3), Some(DecayLevel::Full));
    /// assert_eq!(DecayLevel::from_tag(99), None);
    /// ```
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Tombstoned),
            1 => Some(Self::Gist),
            2 => Some(Self::Compressed),
            3 => Some(Self::Full),
            _ => None,
        }
    }

    /// Returns the `u8` tag for this decay level.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::compressed::DecayLevel;
    ///
    /// assert_eq!(DecayLevel::Full.tag(), 3);
    /// ```
    pub fn tag(self) -> u8 {
        self as u8
    }
}

/// A single slot compressed to R₀ + R₁ only.
///
/// # Example
///
/// ```
/// use volt_db::compressed::CompressedSlot;
/// use volt_core::slot::SlotRole;
/// use volt_core::SLOT_DIM;
///
/// let slot = CompressedSlot {
///     role: SlotRole::Agent,
///     certainty: 0.9,
///     r0: Some([0.1; SLOT_DIM]),
///     r1: None,
///     codebook_id: None,
/// };
/// assert!(slot.r0.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct CompressedSlot {
    /// Semantic role of this slot.
    pub role: SlotRole,
    /// Per-slot certainty (gamma).
    pub certainty: f32,
    /// R₀ (discourse-level) embedding, if present.
    pub r0: Option<[f32; SLOT_DIM]>,
    /// R₁ (proposition-level) embedding, if present.
    pub r1: Option<[f32; SLOT_DIM]>,
    /// VQ-VAE codebook index.
    pub codebook_id: Option<u16>,
}

/// A compressed frame retaining only R₀ and R₁ resolutions.
///
/// Created by [`compress`] from a full `TensorFrame`. Stored in T2.
///
/// # Example
///
/// ```
/// use volt_db::compressed::compress;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
/// frame.frame_meta.frame_id = 1;
///
/// let compressed = compress(&frame);
/// assert_eq!(compressed.frame_id, 1);
/// assert!(compressed.slots[0].is_some());
/// ```
#[derive(Debug, Clone)]
pub struct CompressedFrame {
    /// Unique frame identifier.
    pub frame_id: u64,
    /// The strand this frame belongs to.
    pub strand_id: u64,
    /// Timestamp (microseconds since epoch).
    pub created_at: u64,
    /// Global certainty score (gamma).
    pub global_certainty: f32,
    /// Discourse type classification.
    pub discourse_type: DiscourseType,
    /// Whether verified by Hard Core.
    pub verified: bool,
    /// Compressed slots (R₀ + R₁ only). Most are None.
    pub slots: [Option<CompressedSlot>; MAX_SLOTS],
}

/// A gist-level frame retaining only R₀ per slot.
///
/// The decay step below [`CompressedFrame`].
///
/// # Example
///
/// ```
/// use volt_db::compressed::{compress, to_gist_frame};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
///
/// let compressed = compress(&frame);
/// let gist = to_gist_frame(&compressed, [0.1; SLOT_DIM]);
/// assert!(gist.slot_gists[0].is_some());
/// ```
#[derive(Debug, Clone)]
pub struct GistFrame {
    /// Unique frame identifier.
    pub frame_id: u64,
    /// The strand this frame belongs to.
    pub strand_id: u64,
    /// Timestamp (microseconds since epoch).
    pub created_at: u64,
    /// Global certainty score (gamma).
    pub global_certainty: f32,
    /// Per-slot R₀ vectors. Sparse — most are None.
    pub slot_gists: [Option<[f32; SLOT_DIM]>; MAX_SLOTS],
    /// The superposed R₀ gist vector (same as FrameGist.vector).
    pub gist_vector: [f32; SLOT_DIM],
}

/// A tombstone marking a frame as deleted by GC.
///
/// Only metadata remains — all embedding data is gone.
///
/// # Example
///
/// ```
/// use volt_db::compressed::Tombstone;
///
/// let ts = Tombstone {
///     frame_id: 42,
///     strand_id: 1,
///     tombstoned_at: 1000000,
///     superseded_by: Some(99),
/// };
/// assert_eq!(ts.frame_id, 42);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tombstone {
    /// Original frame identifier.
    pub frame_id: u64,
    /// The strand this frame belonged to.
    pub strand_id: u64,
    /// Timestamp when this frame was tombstoned (microseconds).
    pub tombstoned_at: u64,
    /// If superseded by a consolidation wisdom frame.
    pub superseded_by: Option<u64>,
}

/// A frame at any decay level.
///
/// Used by T2 storage and GC to handle frames uniformly.
///
/// # Example
///
/// ```
/// use volt_db::compressed::{FrameEntry, DecayLevel};
/// use volt_core::TensorFrame;
///
/// let entry = FrameEntry::Full(Box::new(TensorFrame::new()));
/// assert_eq!(entry.decay_level(), DecayLevel::Full);
/// assert_eq!(entry.frame_id(), 0);
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum FrameEntry {
    /// Full TensorFrame (T0/T1).
    Full(Box<TensorFrame>),
    /// Compressed frame (T2, R₀+R₁).
    Compressed(CompressedFrame),
    /// Gist frame (T2, R₀ only).
    Gist(GistFrame),
    /// Tombstone (T2, metadata only).
    Tombstone(Tombstone),
}

impl FrameEntry {
    /// Returns the decay level of this entry.
    pub fn decay_level(&self) -> DecayLevel {
        match self {
            Self::Full(_) => DecayLevel::Full,
            Self::Compressed(_) => DecayLevel::Compressed,
            Self::Gist(_) => DecayLevel::Gist,
            Self::Tombstone(_) => DecayLevel::Tombstoned,
        }
    }

    /// Returns the frame ID of this entry.
    pub fn frame_id(&self) -> u64 {
        match self {
            Self::Full(f) => f.frame_meta.frame_id,
            Self::Compressed(c) => c.frame_id,
            Self::Gist(g) => g.frame_id,
            Self::Tombstone(t) => t.frame_id,
        }
    }

    /// Returns the strand ID of this entry.
    pub fn strand_id(&self) -> u64 {
        match self {
            Self::Full(f) => f.frame_meta.strand_id,
            Self::Compressed(c) => c.strand_id,
            Self::Gist(g) => g.strand_id,
            Self::Tombstone(t) => t.strand_id,
        }
    }

    /// Returns the `created_at` timestamp (microseconds).
    pub fn created_at(&self) -> u64 {
        match self {
            Self::Full(f) => f.frame_meta.created_at,
            Self::Compressed(c) => c.created_at,
            Self::Gist(g) => g.created_at,
            Self::Tombstone(t) => t.tombstoned_at,
        }
    }

    /// Returns the global certainty (gamma) if available.
    pub fn global_certainty(&self) -> f32 {
        match self {
            Self::Full(f) => f.frame_meta.global_certainty,
            Self::Compressed(c) => c.global_certainty,
            Self::Gist(g) => g.global_certainty,
            Self::Tombstone(_) => 0.0,
        }
    }

    /// Serializes this entry to bytes.
    ///
    /// Format: `[decay_level: u8][payload_bytes]`
    ///
    /// - Full frames use serde_json (TensorFrame has Serialize).
    /// - Compressed/Gist frames use custom binary (efficient for `[f32; 256]` arrays).
    /// - Tombstones use serde_json (small, no arrays).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VoltError> {
        let mut buf = Vec::new();
        buf.push(self.decay_level().tag());
        match self {
            Self::Full(f) => {
                let json = serde_json::to_vec(f.as_ref()).map_err(|e| {
                    VoltError::StorageError {
                        message: format!("failed to serialize full frame: {e}"),
                    }
                })?;
                buf.extend_from_slice(&json);
            }
            Self::Compressed(c) => {
                compressed_frame_to_binary(c, &mut buf);
            }
            Self::Gist(g) => {
                gist_frame_to_binary(g, &mut buf);
            }
            Self::Tombstone(t) => {
                let json = serde_json::to_vec(t).map_err(|e| VoltError::StorageError {
                    message: format!("failed to serialize tombstone: {e}"),
                })?;
                buf.extend_from_slice(&json);
            }
        }
        Ok(buf)
    }

    /// Deserializes a `FrameEntry` from bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the format is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VoltError> {
        if bytes.is_empty() {
            return Err(VoltError::StorageError {
                message: "empty frame entry bytes".to_string(),
            });
        }
        let tag = bytes[0];
        let payload = &bytes[1..];
        let level = DecayLevel::from_tag(tag).ok_or_else(|| VoltError::StorageError {
            message: format!("invalid decay level tag: {tag}"),
        })?;
        match level {
            DecayLevel::Full => {
                let frame: TensorFrame =
                    serde_json::from_slice(payload).map_err(|e| VoltError::StorageError {
                        message: format!("failed to deserialize full frame: {e}"),
                    })?;
                Ok(Self::Full(Box::new(frame)))
            }
            DecayLevel::Compressed => {
                let c = compressed_frame_from_binary(payload)?;
                Ok(Self::Compressed(c))
            }
            DecayLevel::Gist => {
                let g = gist_frame_from_binary(payload)?;
                Ok(Self::Gist(g))
            }
            DecayLevel::Tombstoned => {
                let t: Tombstone =
                    serde_json::from_slice(payload).map_err(|e| VoltError::StorageError {
                        message: format!("failed to deserialize tombstone: {e}"),
                    })?;
                Ok(Self::Tombstone(t))
            }
        }
    }
}

/// Compresses a full `TensorFrame` to R₀ + R₁ only.
///
/// R₂ and R₃ resolution data is discarded. Metadata is preserved.
///
/// # Example
///
/// ```
/// use volt_db::compressed::compress;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]); // R₀
/// slot.write_resolution(1, [0.2; SLOT_DIM]); // R₁
/// slot.write_resolution(2, [0.3; SLOT_DIM]); // R₂ (will be dropped)
/// frame.write_slot(0, slot).unwrap();
/// frame.frame_meta.frame_id = 42;
///
/// let compressed = compress(&frame);
/// assert_eq!(compressed.frame_id, 42);
/// let cslot = compressed.slots[0].as_ref().unwrap();
/// assert!(cslot.r0.is_some());
/// assert!(cslot.r1.is_some());
/// ```
pub fn compress(frame: &TensorFrame) -> CompressedFrame {
    let mut slots: [Option<CompressedSlot>; MAX_SLOTS] = [const { None }; MAX_SLOTS];

    for (i, slot_opt) in frame.slots.iter().enumerate() {
        if let Some(slot) = slot_opt {
            slots[i] = Some(CompressedSlot {
                role: slot.role,
                certainty: frame.meta[i].certainty,
                r0: slot.resolutions[0],
                r1: slot.resolutions[1],
                codebook_id: slot.codebook_id,
            });
        }
    }

    CompressedFrame {
        frame_id: frame.frame_meta.frame_id,
        strand_id: frame.frame_meta.strand_id,
        created_at: frame.frame_meta.created_at,
        global_certainty: frame.frame_meta.global_certainty,
        discourse_type: frame.frame_meta.discourse_type,
        verified: frame.frame_meta.verified,
        slots,
    }
}

/// Converts a [`CompressedFrame`] to a [`GistFrame`] (R₀ only).
///
/// R₁ data is discarded. The caller provides the superposed gist vector.
///
/// # Example
///
/// ```
/// use volt_db::compressed::{compress, to_gist_frame};
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// slot.write_resolution(1, [0.2; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
///
/// let compressed = compress(&frame);
/// let gist = to_gist_frame(&compressed, [0.1; SLOT_DIM]);
/// assert!(gist.slot_gists[0].is_some());
/// ```
pub fn to_gist_frame(
    compressed: &CompressedFrame,
    gist_vector: [f32; SLOT_DIM],
) -> GistFrame {
    let mut slot_gists: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];

    for (i, slot_opt) in compressed.slots.iter().enumerate() {
        if let Some(slot) = slot_opt {
            slot_gists[i] = slot.r0;
        }
    }

    GistFrame {
        frame_id: compressed.frame_id,
        strand_id: compressed.strand_id,
        created_at: compressed.created_at,
        global_certainty: compressed.global_certainty,
        slot_gists,
        gist_vector,
    }
}

/// Creates a [`Tombstone`] for a deleted frame.
///
/// # Example
///
/// ```
/// use volt_db::compressed::to_tombstone;
///
/// let ts = to_tombstone(42, 1, 1000, Some(99));
/// assert_eq!(ts.frame_id, 42);
/// assert_eq!(ts.superseded_by, Some(99));
/// ```
pub fn to_tombstone(
    frame_id: u64,
    strand_id: u64,
    tombstoned_at: u64,
    superseded_by: Option<u64>,
) -> Tombstone {
    Tombstone {
        frame_id,
        strand_id,
        tombstoned_at,
        superseded_by,
    }
}

// ---------------------------------------------------------------------------
// Binary codec helpers for CompressedFrame / GistFrame
//
// serde doesn't support `[f32; 256]` arrays (max 32), so we use a
// compact custom binary format for these types.
// ---------------------------------------------------------------------------

/// Encodes a `SlotRole` as a `(tag, data)` byte pair.
fn slot_role_to_bytes(role: SlotRole) -> [u8; 2] {
    match role {
        SlotRole::Agent => [0, 0],
        SlotRole::Predicate => [1, 0],
        SlotRole::Patient => [2, 0],
        SlotRole::Location => [3, 0],
        SlotRole::Time => [4, 0],
        SlotRole::Manner => [5, 0],
        SlotRole::Instrument => [6, 0],
        SlotRole::Cause => [7, 0],
        SlotRole::Result => [8, 0],
        SlotRole::Free(n) => [9, n],
    }
}

/// Decodes a `SlotRole` from a `(tag, data)` byte pair.
fn slot_role_from_bytes(tag: u8, data: u8) -> Option<SlotRole> {
    match tag {
        0 => Some(SlotRole::Agent),
        1 => Some(SlotRole::Predicate),
        2 => Some(SlotRole::Patient),
        3 => Some(SlotRole::Location),
        4 => Some(SlotRole::Time),
        5 => Some(SlotRole::Manner),
        6 => Some(SlotRole::Instrument),
        7 => Some(SlotRole::Cause),
        8 => Some(SlotRole::Result),
        9 => Some(SlotRole::Free(data)),
        _ => None,
    }
}

/// Encodes a `DiscourseType` as a single byte.
fn discourse_type_to_byte(dt: DiscourseType) -> u8 {
    match dt {
        DiscourseType::Query => 0,
        DiscourseType::Statement => 1,
        DiscourseType::Command => 2,
        DiscourseType::Response => 3,
        DiscourseType::Creative => 4,
        DiscourseType::Unknown => 5,
    }
}

/// Decodes a `DiscourseType` from a single byte.
fn discourse_type_from_byte(b: u8) -> Option<DiscourseType> {
    match b {
        0 => Some(DiscourseType::Query),
        1 => Some(DiscourseType::Statement),
        2 => Some(DiscourseType::Command),
        3 => Some(DiscourseType::Response),
        4 => Some(DiscourseType::Creative),
        5 => Some(DiscourseType::Unknown),
        _ => None,
    }
}

/// Writes an `[f32; SLOT_DIM]` to the buffer as little-endian bytes.
fn write_f32_array(buf: &mut Vec<u8>, arr: &[f32; SLOT_DIM]) {
    for &v in arr {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Reads an `[f32; SLOT_DIM]` from a byte slice at the given offset.
///
/// Returns the array and the number of bytes consumed.
fn read_f32_array(data: &[u8], offset: usize) -> Option<([f32; SLOT_DIM], usize)> {
    let needed = SLOT_DIM * 4;
    if data.len() < offset + needed {
        return None;
    }
    let mut arr = [0.0f32; SLOT_DIM];
    for (i, v) in arr.iter_mut().enumerate() {
        let pos = offset + i * 4;
        *v = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
    }
    Some((arr, needed))
}

/// Serializes a [`CompressedFrame`] into a binary buffer.
///
/// Format:
/// ```text
/// frame_id:u64 | strand_id:u64 | created_at:u64 | global_certainty:f32
/// discourse_type:u8 | verified:u8 | slot_presence:u16
/// For each present slot:
///   role_tag:u8 | role_data:u8 | certainty:f32
///   r0_present:u8 [| r0:[f32;256]]
///   r1_present:u8 [| r1:[f32;256]]
///   has_codebook:u8 [| codebook_id:u16]
/// ```
fn compressed_frame_to_binary(frame: &CompressedFrame, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&frame.frame_id.to_le_bytes());
    buf.extend_from_slice(&frame.strand_id.to_le_bytes());
    buf.extend_from_slice(&frame.created_at.to_le_bytes());
    buf.extend_from_slice(&frame.global_certainty.to_le_bytes());
    buf.push(discourse_type_to_byte(frame.discourse_type));
    buf.push(frame.verified as u8);

    // Slot presence bitmask
    let mut presence: u16 = 0;
    for (i, slot) in frame.slots.iter().enumerate() {
        if slot.is_some() {
            presence |= 1 << i;
        }
    }
    buf.extend_from_slice(&presence.to_le_bytes());

    // Slot data
    for s in frame.slots.iter().flatten() {
        let rb = slot_role_to_bytes(s.role);
        buf.push(rb[0]);
        buf.push(rb[1]);
        buf.extend_from_slice(&s.certainty.to_le_bytes());

        // R0
        if let Some(ref r0) = s.r0 {
            buf.push(1);
            write_f32_array(buf, r0);
        } else {
            buf.push(0);
        }

        // R1
        if let Some(ref r1) = s.r1 {
            buf.push(1);
            write_f32_array(buf, r1);
        } else {
            buf.push(0);
        }

        // Codebook
        if let Some(cb) = s.codebook_id {
            buf.push(1);
            buf.extend_from_slice(&cb.to_le_bytes());
        } else {
            buf.push(0);
        }
    }
}

/// Deserializes a [`CompressedFrame`] from binary bytes.
fn compressed_frame_from_binary(data: &[u8]) -> Result<CompressedFrame, VoltError> {
    let err = |msg: &str| VoltError::StorageError {
        message: format!("compressed frame decode: {msg}"),
    };
    // Header: 8+8+8+4+1+1+2 = 32 bytes
    if data.len() < 32 {
        return Err(err("header too short"));
    }
    let mut pos = 0;

    let frame_id = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("frame_id"))?);
    pos += 8;
    let strand_id = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("strand_id"))?);
    pos += 8;
    let created_at = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("created_at"))?);
    pos += 8;
    let global_certainty = f32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| err("gamma"))?);
    pos += 4;
    let discourse_type = discourse_type_from_byte(data[pos]).ok_or_else(|| err("discourse_type"))?;
    pos += 1;
    let verified = data[pos] != 0;
    pos += 1;
    let presence = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| err("presence"))?);
    pos += 2;

    let mut slots: [Option<CompressedSlot>; MAX_SLOTS] = [const { None }; MAX_SLOTS];

    for (i, slot) in slots.iter_mut().enumerate() {
        if presence & (1 << i) == 0 {
            continue;
        }
        if data.len() < pos + 8 {
            return Err(err("slot header truncated"));
        }
        let role = slot_role_from_bytes(data[pos], data[pos + 1])
            .ok_or_else(|| err("slot role"))?;
        pos += 2;
        let certainty = f32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| err("slot certainty"))?,
        );
        pos += 4;

        // R0
        if pos >= data.len() {
            return Err(err("r0 flag missing"));
        }
        let r0 = if data[pos] != 0 {
            pos += 1;
            let (arr, consumed) = read_f32_array(data, pos).ok_or_else(|| err("r0 data"))?;
            pos += consumed;
            Some(arr)
        } else {
            pos += 1;
            None
        };

        // R1
        if pos >= data.len() {
            return Err(err("r1 flag missing"));
        }
        let r1 = if data[pos] != 0 {
            pos += 1;
            let (arr, consumed) = read_f32_array(data, pos).ok_or_else(|| err("r1 data"))?;
            pos += consumed;
            Some(arr)
        } else {
            pos += 1;
            None
        };

        // Codebook
        if pos >= data.len() {
            return Err(err("codebook flag missing"));
        }
        let codebook_id = if data[pos] != 0 {
            pos += 1;
            if data.len() < pos + 2 {
                return Err(err("codebook_id truncated"));
            }
            let cb = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| err("codebook_id"))?,
            );
            pos += 2;
            Some(cb)
        } else {
            pos += 1;
            None
        };

        *slot = Some(CompressedSlot {
            role,
            certainty,
            r0,
            r1,
            codebook_id,
        });
    }

    Ok(CompressedFrame {
        frame_id,
        strand_id,
        created_at,
        global_certainty,
        discourse_type,
        verified,
        slots,
    })
}

/// Serializes a [`GistFrame`] into a binary buffer.
///
/// Format:
/// ```text
/// frame_id:u64 | strand_id:u64 | created_at:u64 | global_certainty:f32
/// slot_presence:u16
/// For each present slot_gist:
///   vector:[f32;256]
/// gist_vector:[f32;256]
/// ```
fn gist_frame_to_binary(gist: &GistFrame, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&gist.frame_id.to_le_bytes());
    buf.extend_from_slice(&gist.strand_id.to_le_bytes());
    buf.extend_from_slice(&gist.created_at.to_le_bytes());
    buf.extend_from_slice(&gist.global_certainty.to_le_bytes());

    // Slot gist presence bitmask
    let mut presence: u16 = 0;
    for (i, sg) in gist.slot_gists.iter().enumerate() {
        if sg.is_some() {
            presence |= 1 << i;
        }
    }
    buf.extend_from_slice(&presence.to_le_bytes());

    // Slot gist vectors
    for v in gist.slot_gists.iter().flatten() {
        write_f32_array(buf, v);
    }

    // Global gist vector
    write_f32_array(buf, &gist.gist_vector);
}

/// Deserializes a [`GistFrame`] from binary bytes.
fn gist_frame_from_binary(data: &[u8]) -> Result<GistFrame, VoltError> {
    let err = |msg: &str| VoltError::StorageError {
        message: format!("gist frame decode: {msg}"),
    };
    // Header: 8+8+8+4+2 = 30 bytes
    if data.len() < 30 {
        return Err(err("header too short"));
    }
    let mut pos = 0;

    let frame_id = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("frame_id"))?);
    pos += 8;
    let strand_id = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("strand_id"))?);
    pos += 8;
    let created_at = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| err("created_at"))?);
    pos += 8;
    let global_certainty = f32::from_le_bytes(data[pos..pos + 4].try_into().map_err(|_| err("gamma"))?);
    pos += 4;
    let presence = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| err("presence"))?);
    pos += 2;

    let mut slot_gists: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];

    for (i, slot_gist) in slot_gists.iter_mut().enumerate() {
        if presence & (1 << i) != 0 {
            let (arr, consumed) =
                read_f32_array(data, pos).ok_or_else(|| err("slot gist data"))?;
            pos += consumed;
            *slot_gist = Some(arr);
        }
    }

    // Global gist vector
    let (gist_vector, _consumed) =
        read_f32_array(data, pos).ok_or_else(|| err("gist_vector data"))?;

    Ok(GistFrame {
        frame_id,
        strand_id,
        created_at,
        global_certainty,
        slot_gists,
        gist_vector,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::slot::{SlotData, SlotRole};

    fn make_full_frame() -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]); // R0
        slot.write_resolution(1, [0.2; SLOT_DIM]); // R1
        slot.write_resolution(2, [0.3; SLOT_DIM]); // R2
        slot.write_resolution(3, [0.4; SLOT_DIM]); // R3
        frame.write_slot(0, slot).unwrap();

        let mut slot2 = SlotData::new(SlotRole::Predicate);
        slot2.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(1, slot2).unwrap();

        frame.frame_meta.frame_id = 42;
        frame.frame_meta.strand_id = 1;
        frame.frame_meta.created_at = 1_000_000;
        frame.frame_meta.global_certainty = 0.85;
        frame.frame_meta.discourse_type = DiscourseType::Statement;
        frame.frame_meta.verified = true;
        frame.meta[0].certainty = 0.9;
        frame.meta[1].certainty = 0.8;
        frame
    }

    #[test]
    fn compress_preserves_r0_r1() {
        let frame = make_full_frame();
        let compressed = compress(&frame);

        assert_eq!(compressed.frame_id, 42);
        assert_eq!(compressed.strand_id, 1);
        assert_eq!(compressed.created_at, 1_000_000);
        assert_eq!(compressed.global_certainty, 0.85);
        assert_eq!(compressed.discourse_type, DiscourseType::Statement);
        assert!(compressed.verified);

        let slot0 = compressed.slots[0].as_ref().unwrap();
        assert_eq!(slot0.role, SlotRole::Agent);
        assert_eq!(slot0.certainty, 0.9);
        assert!(slot0.r0.is_some());
        assert!(slot0.r1.is_some());
        assert_eq!(slot0.r0.unwrap()[0], 0.1);
        assert_eq!(slot0.r1.unwrap()[0], 0.2);

        let slot1 = compressed.slots[1].as_ref().unwrap();
        assert_eq!(slot1.role, SlotRole::Predicate);
        assert!(slot1.r0.is_some());
        assert!(slot1.r1.is_none()); // only R0 was set
    }

    #[test]
    fn compress_drops_r2_r3() {
        let frame = make_full_frame();
        let compressed = compress(&frame);

        // CompressedSlot only has r0 and r1 fields — R2/R3 are structurally absent
        let slot0 = compressed.slots[0].as_ref().unwrap();
        assert!(slot0.r0.is_some());
        assert!(slot0.r1.is_some());
        // No r2/r3 fields exist — compression is structural
    }

    #[test]
    fn compress_empty_frame() {
        let frame = TensorFrame::new();
        let compressed = compress(&frame);

        for slot in &compressed.slots {
            assert!(slot.is_none());
        }
    }

    #[test]
    fn to_gist_preserves_r0() {
        let frame = make_full_frame();
        let compressed = compress(&frame);
        let gist_vec = [0.42; SLOT_DIM];
        let gist = to_gist_frame(&compressed, gist_vec);

        assert_eq!(gist.frame_id, 42);
        assert_eq!(gist.strand_id, 1);
        assert_eq!(gist.created_at, 1_000_000);
        assert!(gist.slot_gists[0].is_some());
        assert_eq!(gist.slot_gists[0].unwrap()[0], 0.1);
        assert!(gist.slot_gists[1].is_some());
        assert_eq!(gist.slot_gists[1].unwrap()[0], 0.5);
        assert!(gist.slot_gists[2].is_none());
        assert_eq!(gist.gist_vector[0], 0.42);
    }

    #[test]
    fn tombstone_fields() {
        let ts = to_tombstone(42, 1, 2_000_000, Some(99));
        assert_eq!(ts.frame_id, 42);
        assert_eq!(ts.strand_id, 1);
        assert_eq!(ts.tombstoned_at, 2_000_000);
        assert_eq!(ts.superseded_by, Some(99));
    }

    #[test]
    fn tombstone_no_supersede() {
        let ts = to_tombstone(10, 0, 500, None);
        assert_eq!(ts.superseded_by, None);
    }

    #[test]
    fn decay_level_ordering() {
        assert!(DecayLevel::Full > DecayLevel::Compressed);
        assert!(DecayLevel::Compressed > DecayLevel::Gist);
        assert!(DecayLevel::Gist > DecayLevel::Tombstoned);
    }

    #[test]
    fn decay_level_tag_roundtrip() {
        for level in [
            DecayLevel::Full,
            DecayLevel::Compressed,
            DecayLevel::Gist,
            DecayLevel::Tombstoned,
        ] {
            assert_eq!(DecayLevel::from_tag(level.tag()), Some(level));
        }
        assert_eq!(DecayLevel::from_tag(99), None);
    }

    #[test]
    fn frame_entry_decay_level() {
        let full = FrameEntry::Full(Box::new(TensorFrame::new()));
        assert_eq!(full.decay_level(), DecayLevel::Full);

        let frame = make_full_frame();
        let compressed = FrameEntry::Compressed(compress(&frame));
        assert_eq!(compressed.decay_level(), DecayLevel::Compressed);

        let c = compress(&frame);
        let gist = FrameEntry::Gist(to_gist_frame(&c, [0.0; SLOT_DIM]));
        assert_eq!(gist.decay_level(), DecayLevel::Gist);

        let ts = FrameEntry::Tombstone(to_tombstone(1, 0, 0, None));
        assert_eq!(ts.decay_level(), DecayLevel::Tombstoned);
    }

    #[test]
    fn frame_entry_accessors() {
        let mut frame = TensorFrame::new();
        frame.frame_meta.frame_id = 10;
        frame.frame_meta.strand_id = 2;
        frame.frame_meta.created_at = 5000;
        frame.frame_meta.global_certainty = 0.7;

        let entry = FrameEntry::Full(Box::new(frame));
        assert_eq!(entry.frame_id(), 10);
        assert_eq!(entry.strand_id(), 2);
        assert_eq!(entry.created_at(), 5000);
        assert_eq!(entry.global_certainty(), 0.7);
    }

    #[test]
    fn frame_entry_bytes_roundtrip_compressed() {
        let frame = make_full_frame();
        let entry = FrameEntry::Compressed(compress(&frame));
        let bytes = entry.to_bytes().unwrap();
        let restored = FrameEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.decay_level(), DecayLevel::Compressed);
        assert_eq!(restored.frame_id(), 42);
        assert_eq!(restored.strand_id(), 1);
    }

    #[test]
    fn frame_entry_bytes_roundtrip_gist() {
        let frame = make_full_frame();
        let c = compress(&frame);
        let g = to_gist_frame(&c, [0.5; SLOT_DIM]);
        let entry = FrameEntry::Gist(g);
        let bytes = entry.to_bytes().unwrap();
        let restored = FrameEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.decay_level(), DecayLevel::Gist);
        assert_eq!(restored.frame_id(), 42);
    }

    #[test]
    fn frame_entry_bytes_roundtrip_tombstone() {
        let ts = to_tombstone(42, 1, 2_000_000, Some(99));
        let entry = FrameEntry::Tombstone(ts);
        let bytes = entry.to_bytes().unwrap();
        let restored = FrameEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.decay_level(), DecayLevel::Tombstoned);
        assert_eq!(restored.frame_id(), 42);
    }

    #[test]
    fn frame_entry_from_bytes_empty_fails() {
        let result = FrameEntry::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn frame_entry_from_bytes_invalid_tag() {
        let result = FrameEntry::from_bytes(&[99]);
        assert!(result.is_err());
    }
}
