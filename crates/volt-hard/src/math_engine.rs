//! MathEngine Hard Strand — exact arithmetic, algebra, and basic calculus.
//!
//! The MathEngine activates when the Intent Router detects a math-related
//! query in the frame's Predicate slot. It parses a textual expression
//! from a well-known slot convention and writes the exact numeric result
//! back into the Result slot with gamma = 1.0.
//!
//! ## Slot Convention
//!
//! The MathEngine uses a simple encoding scheme within TensorFrame slots:
//! - **Input**: The Predicate slot (S1) at R2 (phrase resolution) encodes
//!   a math expression. The first 4 floats store a packed f64 result hint
//!   (unused for now). The capability vector match tells the router this
//!   frame is math-related.
//! - **Operands**: Stored in frame metadata or encoded in slot vectors.
//!
//! For Milestone 3.1, the MathEngine operates on **structured numeric data**
//! encoded directly in frame slots using a simple protocol:
//! - S6 (Instrument) R0 dim[0]: operation code (1.0=add, 2.0=sub, 3.0=mul, 4.0=div, 5.0=pow)
//! - S6 (Instrument) R0 dim[1]: left operand (f32)
//! - S6 (Instrument) R0 dim[2]: right operand (f32)
//! - S8 (Result) R0 dim[0]: the exact result (f32)
//! - S8 (Result) R0 dim[1]: 1.0 if result is valid, 0.0 otherwise

use volt_core::{
    slot::SlotSource, SlotData, SlotMeta, SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM,
};

use crate::strand::{HardStrand, StrandResult};

/// Operation codes for the MathEngine protocol.
const OP_ADD: f32 = 1.0;
const OP_SUB: f32 = 2.0;
const OP_MUL: f32 = 3.0;
const OP_DIV: f32 = 4.0;
const OP_POW: f32 = 5.0;
const OP_SQRT: f32 = 6.0;
const OP_ABS: f32 = 7.0;
const OP_NEG: f32 = 8.0;

/// Slot index for operation input (Instrument = S6).
const INSTRUMENT_SLOT: usize = 6;
/// Slot index for result output (Result = S8).
const RESULT_SLOT: usize = 8;

/// The MathEngine Hard Strand — handles exact arithmetic computation.
///
/// Activates when the frame contains a math operation encoded in the
/// Instrument slot (S6). Computes the exact result and writes it to
/// the Result slot (S8) with gamma = 1.0.
///
/// # Supported Operations
///
/// | Code | Operation | Description |
/// |------|-----------|-------------|
/// | 1.0  | add       | a + b       |
/// | 2.0  | sub       | a - b       |
/// | 3.0  | mul       | a * b       |
/// | 4.0  | div       | a / b       |
/// | 5.0  | pow       | a ^ b       |
/// | 6.0  | sqrt      | sqrt(a)     |
/// | 7.0  | abs       | |a|         |
/// | 8.0  | neg       | -a          |
///
/// # Example
///
/// ```
/// use volt_hard::math_engine::MathEngine;
/// use volt_hard::strand::HardStrand;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let engine = MathEngine::new();
///     let mut frame = TensorFrame::new();
///     let mut instrument = SlotData::new(SlotRole::Instrument);
///     let mut data = [0.0_f32; SLOT_DIM];
///     data[0] = 3.0; // OP_MUL
///     data[1] = 847.0;
///     data[2] = 392.0;
///     instrument.write_resolution(0, data);
///     frame.write_slot(6, instrument).unwrap();
///     frame.meta[6].certainty = 0.9;
///
///     let result = engine.process(&frame).unwrap();
///     assert!(result.activated);
///     let result_slot = result.frame.read_slot(8).unwrap();
///     let r = result_slot.resolutions[0].unwrap();
///     assert!((r[0] - 332_024.0).abs() < 0.01);
/// }).unwrap().join().unwrap();
/// ```
pub struct MathEngine {
    /// Pre-computed capability vector for routing.
    capability: [f32; SLOT_DIM],
}

impl MathEngine {
    /// Creates a new MathEngine with a deterministic capability vector.
    ///
    /// The capability vector is constructed to be recognizable by the
    /// Intent Router as "math-related". It uses a fixed pattern that
    /// math-encoded frames will have high cosine similarity with.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::math_engine::MathEngine;
    /// use volt_hard::strand::HardStrand;
    ///
    /// let engine = MathEngine::new();
    /// assert_eq!(engine.name(), "math_engine");
    /// ```
    pub fn new() -> Self {
        let cap = Self::build_capability_vector();
        tracing::debug!(
            "MathEngine capability: [{:.4}, {:.4}, {:.4}, ...] norm={:.6}",
            cap[0], cap[1], cap[2],
            cap.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        Self {
            capability: cap,
        }
    }

    /// Build the deterministic capability vector for math operations.
    ///
    /// Uses a seeded hash to produce a stable, normalized 256-dim vector
    /// that represents "mathematical computation capability".
    fn build_capability_vector() -> [f32; SLOT_DIM] {
        // Deterministic seed for "math_engine" capability
        const MATH_SEED: u64 = 0x4d41_5448_454e_4731; // "MATHENG1"
        let mut v = [0.0_f32; SLOT_DIM];
        for (i, val) in v.iter_mut().enumerate() {
            let mut h = MATH_SEED.wrapping_mul(0xd2b7_4407_b1ce_6e93);
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

    /// Execute the math operation encoded in the Instrument slot.
    ///
    /// Returns `Ok((result_value, description))` on success, or
    /// `Err(VoltError)` if the operation is invalid.
    fn execute_operation(op_code: f32, left: f32, right: f32) -> Result<(f32, String), VoltError> {
        // Match on operation code with small epsilon for float comparison
        let (result, desc) = if (op_code - OP_ADD).abs() < 0.1 {
            (left + right, format!("{left} + {right} = {}", left + right))
        } else if (op_code - OP_SUB).abs() < 0.1 {
            (left - right, format!("{left} - {right} = {}", left - right))
        } else if (op_code - OP_MUL).abs() < 0.1 {
            (left * right, format!("{left} * {right} = {}", left * right))
        } else if (op_code - OP_DIV).abs() < 0.1 {
            if right.abs() < f32::EPSILON {
                return Err(VoltError::StrandError {
                    strand_id: 0,
                    message: "math_engine: division by zero".to_string(),
                });
            }
            (left / right, format!("{left} / {right} = {}", left / right))
        } else if (op_code - OP_POW).abs() < 0.1 {
            let r = left.powf(right);
            if !r.is_finite() {
                return Err(VoltError::StrandError {
                    strand_id: 0,
                    message: format!("math_engine: {left}^{right} is not finite"),
                });
            }
            (r, format!("{left} ^ {right} = {r}"))
        } else if (op_code - OP_SQRT).abs() < 0.1 {
            if left < 0.0 {
                return Err(VoltError::StrandError {
                    strand_id: 0,
                    message: format!("math_engine: sqrt of negative number {left}"),
                });
            }
            let r = left.sqrt();
            (r, format!("sqrt({left}) = {r}"))
        } else if (op_code - OP_ABS).abs() < 0.1 {
            (left.abs(), format!("|{left}| = {}", left.abs()))
        } else if (op_code - OP_NEG).abs() < 0.1 {
            (-left, format!("-{left} = {}", -left))
        } else {
            return Err(VoltError::StrandError {
                strand_id: 0,
                message: format!("math_engine: unknown operation code {op_code}"),
            });
        };

        if !result.is_finite() {
            return Err(VoltError::StrandError {
                strand_id: 0,
                message: format!("math_engine: result is not finite: {desc}"),
            });
        }

        Ok((result, desc))
    }
}

impl Default for MathEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HardStrand for MathEngine {
    fn name(&self) -> &str {
        "math_engine"
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
                    description: "math_engine: no instrument slot data".to_string(),
                });
            }
        };

        let data = match instrument.resolutions[0] {
            Some(ref d) => d,
            None => {
                return Ok(StrandResult {
                    frame: frame.clone(),
                    activated: false,
                    description: "math_engine: no R0 data in instrument slot".to_string(),
                });
            }
        };

        // Extract operation parameters
        let op_code = data[0];
        let left = data[1];
        let right = data[2];

        // Execute the operation
        let (result_value, description) = Self::execute_operation(op_code, left, right)?;

        // Build the result frame
        let mut result_frame = frame.clone();

        // Write result to S8 (Result slot) at R0
        let mut result_data = [0.0_f32; SLOT_DIM];
        result_data[0] = result_value;
        result_data[1] = 1.0; // valid flag

        let mut result_slot = SlotData::new(SlotRole::Result);
        result_slot.write_resolution(0, result_data);
        result_frame.write_slot(RESULT_SLOT, result_slot)?;

        // Set metadata: gamma = 1.0 (exact computation), source = HardCore
        result_frame.meta[RESULT_SLOT] = SlotMeta {
            certainty: 1.0,
            source: SlotSource::HardCore,
            updated_at: 0, // caller should set real timestamp
            needs_verify: false,
        };

        // Mark frame as verified by hard core
        result_frame.frame_meta.verified = true;
        result_frame.frame_meta.proof_length += 1;

        // Recompute global certainty (min of active slot gammas)
        let mut min_gamma = f32::MAX;
        for i in 0..MAX_SLOTS {
            if result_frame.slots[i].is_some() {
                let g = result_frame.meta[i].certainty;
                if g < min_gamma {
                    min_gamma = g;
                }
            }
        }
        if min_gamma < f32::MAX {
            result_frame.frame_meta.global_certainty = min_gamma;
        }

        Ok(StrandResult {
            frame: result_frame,
            activated: true,
            description: format!("math_engine: {description}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_math_frame(op: f32, left: f32, right: f32) -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = op;
        data[1] = left;
        data[2] = right;
        instrument.write_resolution(0, data);
        frame.write_slot(INSTRUMENT_SLOT, instrument).unwrap();
        frame.meta[INSTRUMENT_SLOT].certainty = 0.9;
        frame
    }

    #[test]
    fn math_engine_name() {
        let engine = MathEngine::new();
        assert_eq!(engine.name(), "math_engine");
    }

    #[test]
    fn math_engine_capability_vector_is_normalized() {
        let engine = MathEngine::new();
        let v = engine.capability_vector();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "capability vector should be unit norm, got {norm}"
        );
    }

    #[test]
    fn math_engine_addition() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_ADD, 100.0, 200.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        let vals = r.resolutions[0].unwrap();
        assert!((vals[0] - 300.0).abs() < 0.01);
        assert!((vals[1] - 1.0).abs() < 0.01); // valid flag
    }

    #[test]
    fn math_engine_subtraction() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_SUB, 500.0, 123.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - 377.0).abs() < 0.01);
    }

    #[test]
    fn math_engine_multiplication() {
        let engine = MathEngine::new();
        // Milestone test case: 847 x 392 = 332,024
        // (Note: PHASE-3.md says 331,824 but the correct product is 332,024)
        let frame = make_math_frame(OP_MUL, 847.0, 392.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!(
            (r.resolutions[0].unwrap()[0] - 332_024.0).abs() < 1.0,
            "847 * 392 should equal 332024, got {}",
            r.resolutions[0].unwrap()[0]
        );
    }

    #[test]
    fn math_engine_division() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_DIV, 100.0, 4.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - 25.0).abs() < 0.01);
    }

    #[test]
    fn math_engine_division_by_zero_errors() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_DIV, 100.0, 0.0);
        let result = engine.process(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn math_engine_power() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_POW, 2.0, 10.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - 1024.0).abs() < 0.01);
    }

    #[test]
    fn math_engine_sqrt() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_SQRT, 144.0, 0.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - 12.0).abs() < 0.01);
    }

    #[test]
    fn math_engine_sqrt_negative_errors() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_SQRT, -4.0, 0.0);
        let result = engine.process(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn math_engine_abs() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_ABS, -42.0, 0.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - 42.0).abs() < 0.01);
    }

    #[test]
    fn math_engine_neg() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_NEG, 42.0, 0.0);
        let result = engine.process(&frame).unwrap();
        assert!(result.activated);

        let r = result.frame.read_slot(RESULT_SLOT).unwrap();
        assert!((r.resolutions[0].unwrap()[0] - (-42.0)).abs() < 0.01);
    }

    #[test]
    fn math_engine_unknown_op_errors() {
        let engine = MathEngine::new();
        let frame = make_math_frame(99.0, 1.0, 2.0);
        let result = engine.process(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn math_engine_no_instrument_slot_passthrough() {
        let engine = MathEngine::new();
        let frame = TensorFrame::new();
        let result = engine.process(&frame).unwrap();
        assert!(!result.activated);
    }

    #[test]
    fn math_engine_sets_gamma_one() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_ADD, 1.0, 2.0);
        let result = engine.process(&frame).unwrap();

        assert_eq!(result.frame.meta[RESULT_SLOT].certainty, 1.0);
        assert_eq!(result.frame.meta[RESULT_SLOT].source, SlotSource::HardCore);
    }

    #[test]
    fn math_engine_marks_frame_verified() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_ADD, 1.0, 2.0);
        let result = engine.process(&frame).unwrap();

        assert!(result.frame.frame_meta.verified);
        assert!(result.frame.frame_meta.proof_length >= 1);
    }

    #[test]
    fn math_engine_global_certainty_is_min() {
        let engine = MathEngine::new();
        let mut frame = make_math_frame(OP_ADD, 1.0, 2.0);

        // Add another slot with low certainty
        let mut agent = SlotData::new(SlotRole::Agent);
        agent.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, agent).unwrap();
        frame.meta[0].certainty = 0.4;

        let result = engine.process(&frame).unwrap();

        // Global certainty should be min(0.4, 0.9, 1.0) = 0.4
        assert!(
            (result.frame.frame_meta.global_certainty - 0.4).abs() < 0.01,
            "global certainty should be 0.4 (min), got {}",
            result.frame.frame_meta.global_certainty
        );
    }

    #[test]
    fn math_engine_returns_in_under_1ms() {
        let engine = MathEngine::new();
        let frame = make_math_frame(OP_MUL, 847.0, 392.0);

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = engine.process(&frame).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 1000;
        assert!(
            per_call.as_micros() < 1000,
            "MathEngine should return in < 1ms, got {:?}",
            per_call
        );
    }
}
