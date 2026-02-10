//! HDCAlgebra Hard Strand — compositional reasoning via HDC operations.
//!
//! Exposes [`volt_bus`] operations (bind, unbind, superpose, permute,
//! similarity) as a callable [`HardStrand`] for compositional reasoning.
//! The Intent Router activates this strand when a frame encodes an HDC
//! algebra request in the Instrument slot.
//!
//! ## Slot Convention
//!
//! | Slot | Resolution | Meaning |
//! |------|-----------|---------|
//! | S6 (Instrument) R0 | dim[0] | Operation code (11-15) |
//! | S6 (Instrument) R0 | dim[1] | Source slot A index |
//! | S6 (Instrument) R0 | dim[2] | Source slot B index (binary ops) |
//! | S6 (Instrument) R0 | dim[3] | Permute offset k (permute only) |
//! | S8 (Result) R0 | - | Result vector (or sim score in dim[0]) |
//!
//! ## Operation Codes
//!
//! | Code | Operation | Description |
//! |------|-----------|-------------|
//! | 11.0 | bind | Circular convolution (⊗) |
//! | 12.0 | unbind | Circular correlation (⊙) |
//! | 13.0 | superpose | Element-wise sum + normalize |
//! | 14.0 | permute | Cyclic rotation by k positions |
//! | 15.0 | similarity | Cosine similarity score |
//!
//! # Example
//!
//! ```
//! use volt_hard::hdc_algebra::HDCAlgebra;
//! use volt_hard::strand::HardStrand;
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//!
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let algebra = HDCAlgebra::new();
//!     let mut frame = TensorFrame::new();
//!
//!     // Source vectors in S0 and S2
//!     let mut s0 = SlotData::new(SlotRole::Agent);
//!     let mut v0 = [0.0_f32; SLOT_DIM];
//!     v0[0] = 1.0;
//!     s0.write_resolution(0, v0);
//!     frame.write_slot(0, s0).unwrap();
//!     frame.meta[0].certainty = 0.9;
//!
//!     let mut s2 = SlotData::new(SlotRole::Patient);
//!     let mut v2 = [0.0_f32; SLOT_DIM];
//!     v2[1] = 1.0;
//!     s2.write_resolution(0, v2);
//!     frame.write_slot(2, s2).unwrap();
//!     frame.meta[2].certainty = 0.85;
//!
//!     // Request: similarity(S0, S2)
//!     let mut inst = SlotData::new(SlotRole::Instrument);
//!     let mut data = [0.0_f32; SLOT_DIM];
//!     data[0] = 15.0; // OP_SIMILARITY
//!     data[1] = 0.0;  // slot A = S0
//!     data[2] = 2.0;  // slot B = S2
//!     inst.write_resolution(0, data);
//!     frame.write_slot(6, inst).unwrap();
//!     frame.meta[6].certainty = 1.0;
//!
//!     let result = algebra.process(&frame).unwrap();
//!     assert!(result.activated);
//! }).unwrap().join().unwrap();
//! ```

use volt_bus::{bind, permute, similarity, superpose, unbind};
use volt_core::{
    slot::SlotSource, SlotData, SlotMeta, SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM,
};

use crate::strand::{HardStrand, StrandResult};

/// Operation codes for HDC algebra operations.
const OP_HDC_BIND: f32 = 11.0;
const OP_HDC_UNBIND: f32 = 12.0;
const OP_HDC_SUPERPOSE: f32 = 13.0;
const OP_HDC_PERMUTE: f32 = 14.0;
const OP_HDC_SIMILARITY: f32 = 15.0;

/// Slot index for operation input (Instrument = S6).
const INSTRUMENT_SLOT: usize = 6;
/// Slot index for result output (Result = S8).
const RESULT_SLOT: usize = 8;

/// The HDCAlgebra Hard Strand — compositional reasoning via HDC operations.
///
/// Delegates to [`volt_bus`] for bind, unbind, superpose, permute, and
/// similarity operations. Operand vectors are read from frame slots
/// specified in the Instrument slot.
///
/// # Example
///
/// ```
/// use volt_hard::hdc_algebra::HDCAlgebra;
/// use volt_hard::strand::HardStrand;
///
/// let algebra = HDCAlgebra::new();
/// assert_eq!(algebra.name(), "hdc_algebra");
/// ```
pub struct HDCAlgebra {
    capability: [f32; SLOT_DIM],
}

impl HDCAlgebra {
    /// Creates a new HDCAlgebra strand with a deterministic capability vector.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::hdc_algebra::HDCAlgebra;
    /// use volt_hard::strand::HardStrand;
    ///
    /// let algebra = HDCAlgebra::new();
    /// assert_eq!(algebra.name(), "hdc_algebra");
    /// ```
    pub fn new() -> Self {
        Self {
            capability: Self::build_capability_vector(),
        }
    }

    /// Build the deterministic capability vector for HDC algebra operations.
    fn build_capability_vector() -> [f32; SLOT_DIM] {
        const HDC_SEED: u64 = 0x4844_4341_4C47_4231; // "HDCALGB1"
        let mut v = [0.0_f32; SLOT_DIM];
        for (i, val) in v.iter_mut().enumerate() {
            let mut h = HDC_SEED.wrapping_mul(0xd2b7_4407_b1ce_6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^= h >> 33;
            *val = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Read a 256-dim vector from the given slot at R0.
    fn read_slot_vector(
        frame: &TensorFrame,
        slot_idx: usize,
    ) -> Result<[f32; SLOT_DIM], VoltError> {
        if slot_idx >= MAX_SLOTS {
            return Err(VoltError::SlotOutOfRange {
                index: slot_idx,
                max: MAX_SLOTS,
            });
        }
        let slot = frame.slots[slot_idx]
            .as_ref()
            .ok_or(VoltError::EmptySlot { index: slot_idx })?;
        slot.resolutions[0].ok_or_else(|| VoltError::StrandError {
            strand_id: 0,
            message: format!("hdc_algebra: slot {slot_idx} has no R0 data"),
        })
    }

    /// Recompute global certainty as min of all active slot gammas.
    fn recompute_global_certainty(frame: &mut TensorFrame) {
        let mut min_gamma = f32::MAX;
        for i in 0..MAX_SLOTS {
            if frame.slots[i].is_some() {
                let g = frame.meta[i].certainty;
                if g < min_gamma {
                    min_gamma = g;
                }
            }
        }
        if min_gamma < f32::MAX {
            frame.frame_meta.global_certainty = min_gamma;
        }
    }
}

impl Default for HDCAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

impl HardStrand for HDCAlgebra {
    fn name(&self) -> &str {
        "hdc_algebra"
    }

    fn capability_vector(&self) -> &[f32; SLOT_DIM] {
        &self.capability
    }

    fn threshold(&self) -> f32 {
        0.3
    }

    fn process(&self, frame: &TensorFrame) -> Result<StrandResult, VoltError> {
        // Check if Instrument slot (S6) has data at R0
        let instrument = match frame.slots[INSTRUMENT_SLOT].as_ref() {
            Some(slot) => slot,
            None => {
                return Ok(StrandResult {
                    frame: frame.clone(),
                    activated: false,
                    description: "hdc_algebra: no instrument slot data".to_string(),
                });
            }
        };

        let r0_data = match instrument.resolutions[0] {
            Some(ref d) => d,
            None => {
                return Ok(StrandResult {
                    frame: frame.clone(),
                    activated: false,
                    description: "hdc_algebra: no R0 data in instrument slot".to_string(),
                });
            }
        };

        let op_code = r0_data[0];
        let slot_a_idx = r0_data[1] as usize;
        let slot_b_idx = r0_data[2] as usize;
        let permute_k = r0_data[3] as isize;

        let (result_vec, description) = if (op_code - OP_HDC_BIND).abs() < 0.5 {
            let vec_a = Self::read_slot_vector(frame, slot_a_idx)?;
            let vec_b = Self::read_slot_vector(frame, slot_b_idx)?;
            let bound = bind(&vec_a, &vec_b)?;
            (bound, format!("bind(S{slot_a_idx}, S{slot_b_idx})"))
        } else if (op_code - OP_HDC_UNBIND).abs() < 0.5 {
            let vec_a = Self::read_slot_vector(frame, slot_a_idx)?;
            let vec_b = Self::read_slot_vector(frame, slot_b_idx)?;
            let unbound = unbind(&vec_a, &vec_b)?;
            (unbound, format!("unbind(S{slot_a_idx}, S{slot_b_idx})"))
        } else if (op_code - OP_HDC_SUPERPOSE).abs() < 0.5 {
            let vec_a = Self::read_slot_vector(frame, slot_a_idx)?;
            let vec_b = Self::read_slot_vector(frame, slot_b_idx)?;
            let superposed = superpose(&[&vec_a, &vec_b])?;
            (
                superposed,
                format!("superpose(S{slot_a_idx}, S{slot_b_idx})"),
            )
        } else if (op_code - OP_HDC_PERMUTE).abs() < 0.5 {
            let vec_a = Self::read_slot_vector(frame, slot_a_idx)?;
            let permuted = permute(&vec_a, permute_k);
            (permuted, format!("permute(S{slot_a_idx}, k={permute_k})"))
        } else if (op_code - OP_HDC_SIMILARITY).abs() < 0.5 {
            let vec_a = Self::read_slot_vector(frame, slot_a_idx)?;
            let vec_b = Self::read_slot_vector(frame, slot_b_idx)?;
            let sim = similarity(&vec_a, &vec_b);
            let mut result = [0.0_f32; SLOT_DIM];
            result[0] = sim;
            result[1] = 1.0; // valid flag
            (
                result,
                format!("similarity(S{slot_a_idx}, S{slot_b_idx}) = {sim:.4}"),
            )
        } else {
            return Ok(StrandResult {
                frame: frame.clone(),
                activated: false,
                description: format!("hdc_algebra: unknown op code {op_code}"),
            });
        };

        // Write result to S8
        let mut result_frame = frame.clone();
        let mut result_slot = SlotData::new(SlotRole::Result);
        result_slot.write_resolution(0, result_vec);
        result_frame.write_slot(RESULT_SLOT, result_slot)?;

        result_frame.meta[RESULT_SLOT] = SlotMeta {
            certainty: 1.0,
            source: SlotSource::HardCore,
            updated_at: 0,
            needs_verify: false,
        };

        result_frame.frame_meta.verified = true;
        result_frame.frame_meta.proof_length += 1;

        Self::recompute_global_certainty(&mut result_frame);

        Ok(StrandResult {
            frame: result_frame,
            activated: true,
            description: format!("hdc_algebra: {description}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a frame with two source vectors and an HDC algebra request.
    fn make_hdc_frame(
        op: f32,
        slot_a: usize,
        slot_b: usize,
        vec_a: [f32; SLOT_DIM],
        vec_b: [f32; SLOT_DIM],
    ) -> TensorFrame {
        let mut frame = TensorFrame::new();

        let mut sa = SlotData::new(SlotRole::Agent);
        sa.write_resolution(0, vec_a);
        frame.write_slot(slot_a, sa).unwrap();
        frame.meta[slot_a].certainty = 0.9;

        let mut sb = SlotData::new(SlotRole::Patient);
        sb.write_resolution(0, vec_b);
        frame.write_slot(slot_b, sb).unwrap();
        frame.meta[slot_b].certainty = 0.85;

        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = op;
        data[1] = slot_a as f32;
        data[2] = slot_b as f32;
        instrument.write_resolution(0, data);
        frame.write_slot(INSTRUMENT_SLOT, instrument).unwrap();
        frame.meta[INSTRUMENT_SLOT].certainty = 1.0;

        frame
    }

    /// Build a normalized pseudo-random vector from a seed.
    fn seeded_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut v = [0.0_f32; SLOT_DIM];
        for (i, val) in v.iter_mut().enumerate() {
            let mut h = seed.wrapping_mul(0xd2b7_4407_b1ce_6e93);
            h = h.wrapping_add(i as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            *val = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn hdc_algebra_name() {
        let algebra = HDCAlgebra::new();
        assert_eq!(algebra.name(), "hdc_algebra");
    }

    #[test]
    fn hdc_algebra_capability_vector_is_normalized() {
        let algebra = HDCAlgebra::new();
        let v = algebra.capability_vector();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "capability vector should be unit norm, got {norm}"
        );
    }

    #[test]
    fn hdc_algebra_bind() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0xAAAA);
        let vec_b = seeded_vector(0xBBBB);

        let frame = make_hdc_frame(OP_HDC_BIND, 0, 2, vec_a, vec_b);
        let result = algebra.process(&frame).unwrap();

        assert!(result.activated);

        // Compare with direct volt_bus::bind
        let expected = bind(&vec_a, &vec_b).unwrap();
        let actual = result.frame.read_slot(RESULT_SLOT).unwrap();
        let actual_vec = actual.resolutions[0].unwrap();

        let sim = similarity(&expected, &actual_vec);
        assert!(
            sim > 0.99,
            "HDCAlgebra bind should match volt_bus::bind, similarity = {sim}"
        );
    }

    #[test]
    fn hdc_algebra_unbind() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0xCCCC);
        let vec_b = seeded_vector(0xDDDD);

        let frame = make_hdc_frame(OP_HDC_UNBIND, 0, 2, vec_a, vec_b);
        let result = algebra.process(&frame).unwrap();

        assert!(result.activated);

        let expected = unbind(&vec_a, &vec_b).unwrap();
        let actual = result.frame.read_slot(RESULT_SLOT).unwrap();
        let actual_vec = actual.resolutions[0].unwrap();

        let sim = similarity(&expected, &actual_vec);
        assert!(
            sim > 0.99,
            "HDCAlgebra unbind should match volt_bus::unbind, similarity = {sim}"
        );
    }

    #[test]
    fn hdc_algebra_superpose() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0xEEEE);
        let vec_b = seeded_vector(0xFFFF);

        let frame = make_hdc_frame(OP_HDC_SUPERPOSE, 0, 2, vec_a, vec_b);
        let result = algebra.process(&frame).unwrap();

        assert!(result.activated);

        let expected = superpose(&[&vec_a, &vec_b]).unwrap();
        let actual = result.frame.read_slot(RESULT_SLOT).unwrap();
        let actual_vec = actual.resolutions[0].unwrap();

        let sim = similarity(&expected, &actual_vec);
        assert!(
            sim > 0.99,
            "HDCAlgebra superpose should match volt_bus::superpose, similarity = {sim}"
        );
    }

    #[test]
    fn hdc_algebra_permute() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0x1111);
        let vec_b = seeded_vector(0x2222); // unused for permute

        // Set permute offset k=5 in dim[3]
        let mut frame = make_hdc_frame(OP_HDC_PERMUTE, 0, 2, vec_a, vec_b);
        let inst = frame.slots[INSTRUMENT_SLOT].as_mut().unwrap();
        let mut data = inst.resolutions[0].unwrap();
        data[3] = 5.0;
        inst.write_resolution(0, data);

        let result = algebra.process(&frame).unwrap();
        assert!(result.activated);

        let expected = permute(&vec_a, 5);
        let actual = result.frame.read_slot(RESULT_SLOT).unwrap();
        let actual_vec = actual.resolutions[0].unwrap();

        // permute is exact (no FFT), so values should match exactly
        for i in 0..SLOT_DIM {
            assert!(
                (expected[i] - actual_vec[i]).abs() < 1e-6,
                "permute mismatch at dim {i}: expected {}, got {}",
                expected[i],
                actual_vec[i]
            );
        }
    }

    #[test]
    fn hdc_algebra_similarity() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0x3333);

        // Similarity of a vector with itself should be ~1.0
        let frame = make_hdc_frame(OP_HDC_SIMILARITY, 0, 2, vec_a, vec_a);
        let result = algebra.process(&frame).unwrap();

        assert!(result.activated);

        let actual = result.frame.read_slot(RESULT_SLOT).unwrap();
        let vals = actual.resolutions[0].unwrap();
        assert!(
            (vals[0] - 1.0).abs() < 0.01,
            "self-similarity should be ~1.0, got {}",
            vals[0]
        );
        assert!((vals[1] - 1.0).abs() < 0.01, "valid flag should be 1.0");
    }

    #[test]
    fn hdc_algebra_unknown_op_passthrough() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0x4444);
        let vec_b = seeded_vector(0x5555);

        let frame = make_hdc_frame(99.0, 0, 2, vec_a, vec_b);
        let result = algebra.process(&frame).unwrap();

        assert!(!result.activated);
    }

    #[test]
    fn hdc_algebra_missing_source_slot_errors() {
        let algebra = HDCAlgebra::new();

        // Build frame with only instrument slot, no source slots
        let mut frame = TensorFrame::new();
        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = OP_HDC_BIND;
        data[1] = 0.0; // S0 — empty
        data[2] = 2.0; // S2 — empty
        instrument.write_resolution(0, data);
        frame.write_slot(INSTRUMENT_SLOT, instrument).unwrap();

        let result = algebra.process(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn hdc_algebra_no_instrument_slot_passthrough() {
        let algebra = HDCAlgebra::new();
        let frame = TensorFrame::new();
        let result = algebra.process(&frame).unwrap();
        assert!(!result.activated);
    }

    #[test]
    fn hdc_algebra_sets_gamma_and_source() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0x6666);
        let vec_b = seeded_vector(0x7777);

        let frame = make_hdc_frame(OP_HDC_BIND, 0, 2, vec_a, vec_b);
        let result = algebra.process(&frame).unwrap();

        assert_eq!(result.frame.meta[RESULT_SLOT].certainty, 1.0);
        assert_eq!(
            result.frame.meta[RESULT_SLOT].source,
            SlotSource::HardCore
        );
        assert!(result.frame.frame_meta.verified);
        assert!(result.frame.frame_meta.proof_length >= 1);
    }

    #[test]
    fn hdc_algebra_global_certainty_is_min() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0x8888);
        let vec_b = seeded_vector(0x9999);

        let frame = make_hdc_frame(OP_HDC_BIND, 0, 2, vec_a, vec_b);
        // Slots have certainty: S0=0.9, S2=0.85, S6=1.0
        // After processing, S8=1.0
        // Min should be 0.85

        let result = algebra.process(&frame).unwrap();

        assert!(
            (result.frame.frame_meta.global_certainty - 0.85).abs() < 0.01,
            "global certainty should be min = 0.85, got {}",
            result.frame.frame_meta.global_certainty
        );
    }

    #[test]
    fn hdc_algebra_returns_in_under_1ms() {
        let algebra = HDCAlgebra::new();
        let vec_a = seeded_vector(0xAA00);
        let vec_b = seeded_vector(0xBB00);

        let frame = make_hdc_frame(OP_HDC_BIND, 0, 2, vec_a, vec_b);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = algebra.process(&frame).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 100;
        assert!(
            per_call.as_micros() < 1000,
            "HDCAlgebra should return in < 1ms, got {:?}",
            per_call
        );
    }

    #[test]
    fn hdc_algebra_default_trait() {
        let algebra = HDCAlgebra::default();
        assert_eq!(algebra.name(), "hdc_algebra");
    }
}
