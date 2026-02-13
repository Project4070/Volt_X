//! Code-specific attention bias for the RAR Attend phase.
//!
//! Defines a 16×16 additive attention bias matrix encoding structural
//! priors for how code concepts in different slot positions should
//! attend to each other. Applied as an additive term to pre-softmax
//! logits in [`SlotAttention::forward`](crate::attention::SlotAttention::forward).
//!
//! # Code Slot Mapping
//!
//! ```text
//! S0  (Agent):      Function name / class
//! S1  (Predicate):  Operation / method call
//! S2  (Patient):    Arguments / parameters
//! S3  (Location):   Return value / result
//! S4  (Time):       Execution order (loops, sequencing)
//! S5  (Manner):     Algorithm pattern (recursive, iterative)
//! S6  (Instrument): Control flow 1 (if/else)
//! S7  (Cause):      Control flow 2 (try/catch)
//! S8  (Result):     Control flow 3 (match/switch)
//! S9–S15 (Free):    Complex logic overflow
//! ```
//!
//! # Example
//!
//! ```
//! use volt_soft::code_attention::{code_attention_bias, new_code_attention};
//! use volt_core::MAX_SLOTS;
//!
//! let bias = code_attention_bias();
//! // Function ↔ Arguments is strong
//! assert_eq!(bias[0][2], 2.0);
//! assert_eq!(bias[2][0], 2.0); // symmetric
//!
//! // Free slots have no prior
//! assert_eq!(bias[10][11], 0.0);
//!
//! let attn = new_code_attention(42);
//! assert!(attn.attention_bias().is_some());
//! ```

use crate::attention::SlotAttention;
use volt_core::MAX_SLOTS;

/// Returns the 16×16 code attention bias matrix.
///
/// Encodes structural priors for how code concepts in different
/// slot positions should attend to each other. The matrix is
/// symmetric: `bias[i][j] == bias[j][i]`.
///
/// # Bias Values
///
/// | Pair | Bias | Rationale |
/// |------|------|-----------|
/// | S0↔S2 | +2.0 | Function always needs its arguments |
/// | S0↔S3 | +2.0 | Function always defines its return |
/// | S1↔S2 | +2.0 | Operations consume arguments |
/// | S0↔S1 | +1.5 | Function body is operations |
/// | S1↔S3 | +1.5 | Operations produce return values |
/// | S4↔S1 | +1.5 | Execution order sequences operations |
/// | S5↔S1 | +1.0 | Algorithm pattern governs operations |
/// | S6↔S1, S7↔S1, S8↔S1 | +1.0 | Control flow wraps operations |
/// | S4↔S6, S4↔S7, S4↔S8 | +1.0 | Execution order interacts with control flow |
/// | S6↔S7, S6↔S8, S7↔S8 | +0.5 | Control flow slots mildly related |
/// | Sᵢ↔Sᵢ (i=0..8) | +0.5 | Self-attention bonus for fixed roles |
/// | S9–S15 | 0.0 | No prior for free slots |
///
/// # Example
///
/// ```
/// use volt_soft::code_attention::code_attention_bias;
///
/// let bias = code_attention_bias();
/// assert_eq!(bias[0][2], 2.0); // Function ↔ Arguments
/// assert_eq!(bias[0][2], bias[2][0]); // symmetric
/// ```
pub fn code_attention_bias() -> [[f32; MAX_SLOTS]; MAX_SLOTS] {
    let mut bias = [[0.0f32; MAX_SLOTS]; MAX_SLOTS];

    // Helper: set symmetric bias
    let mut set = |i: usize, j: usize, val: f32| {
        bias[i][j] = val;
        bias[j][i] = val;
    };

    // Strong connections (+2.0)
    set(0, 2, 2.0); // Function ↔ Arguments
    set(0, 3, 2.0); // Function ↔ Return
    set(1, 2, 2.0); // Operation ↔ Arguments

    // Medium connections (+1.5)
    set(0, 1, 1.5); // Function ↔ Operation
    set(1, 3, 1.5); // Operation ↔ Return
    set(4, 1, 1.5); // ExecutionOrder ↔ Operation

    // Mild connections (+1.0)
    set(5, 1, 1.0); // AlgorithmPattern ↔ Operation
    set(6, 1, 1.0); // ControlFlow1 ↔ Operation
    set(7, 1, 1.0); // ControlFlow2 ↔ Operation
    set(8, 1, 1.0); // ControlFlow3 ↔ Operation
    set(4, 6, 1.0); // ExecutionOrder ↔ ControlFlow1
    set(4, 7, 1.0); // ExecutionOrder ↔ ControlFlow2
    set(4, 8, 1.0); // ExecutionOrder ↔ ControlFlow3

    // Weak connections (+0.5)
    set(6, 7, 0.5); // ControlFlow1 ↔ ControlFlow2
    set(6, 8, 0.5); // ControlFlow1 ↔ ControlFlow3
    set(7, 8, 0.5); // ControlFlow2 ↔ ControlFlow3

    // Self-attention bonus for fixed-role slots (+0.5)
    for (i, row) in bias.iter_mut().enumerate().take(9) {
        row[i] = 0.5;
    }

    bias
}

/// Creates a [`SlotAttention`] module with code-specific attention bias.
///
/// Uses Xavier/Glorot random initialization for Q/K/V projections
/// and applies the code attention bias from [`code_attention_bias`].
///
/// # Example
///
/// ```
/// use volt_soft::code_attention::new_code_attention;
/// use volt_core::{MAX_SLOTS, SLOT_DIM};
///
/// let attn = new_code_attention(42);
/// assert!(attn.attention_bias().is_some());
///
/// let mut states = [const { None }; MAX_SLOTS];
/// states[0] = Some([0.1; SLOT_DIM]);
/// let messages = attn.forward(&states).unwrap();
/// ```
pub fn new_code_attention(seed: u64) -> SlotAttention {
    SlotAttention::new_with_bias(seed, code_attention_bias())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_is_symmetric() {
        let bias = code_attention_bias();
        for i in 0..MAX_SLOTS {
            for j in 0..MAX_SLOTS {
                assert_eq!(
                    bias[i][j], bias[j][i],
                    "bias[{i}][{j}]={} != bias[{j}][{i}]={}",
                    bias[i][j], bias[j][i]
                );
            }
        }
    }

    #[test]
    fn strong_connections_are_2_0() {
        let bias = code_attention_bias();
        assert_eq!(bias[0][2], 2.0); // Function ↔ Arguments
        assert_eq!(bias[0][3], 2.0); // Function ↔ Return
        assert_eq!(bias[1][2], 2.0); // Operation ↔ Arguments
    }

    #[test]
    fn medium_connections_are_1_5() {
        let bias = code_attention_bias();
        assert_eq!(bias[0][1], 1.5); // Function ↔ Operation
        assert_eq!(bias[1][3], 1.5); // Operation ↔ Return
        assert_eq!(bias[4][1], 1.5); // ExecutionOrder ↔ Operation
    }

    #[test]
    fn free_slots_have_no_bias() {
        let bias = code_attention_bias();
        for i in 9..MAX_SLOTS {
            for j in 0..MAX_SLOTS {
                assert_eq!(
                    bias[i][j], 0.0,
                    "free slot S{i} should have no bias to S{j}"
                );
            }
        }
    }

    #[test]
    fn self_attention_bonus_for_fixed_roles() {
        let bias = code_attention_bias();
        for i in 0..9 {
            assert_eq!(
                bias[i][i], 0.5,
                "fixed-role slot S{i} should have self-bias 0.5"
            );
        }
        for i in 9..MAX_SLOTS {
            assert_eq!(
                bias[i][i], 0.0,
                "free slot S{i} should have no self-bias"
            );
        }
    }

    #[test]
    fn new_code_attention_has_bias() {
        let attn = new_code_attention(42);
        let stored = attn.attention_bias().unwrap();
        let expected = code_attention_bias();
        for i in 0..MAX_SLOTS {
            for j in 0..MAX_SLOTS {
                assert_eq!(stored[i][j], expected[i][j]);
            }
        }
    }
}
