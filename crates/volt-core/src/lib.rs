//! # volt-core
//!
//! The foundational crate for Volt X. Defines the core data structures
//! that all other crates depend on:
//!
//! - [`TensorFrame`] — the fundamental unit of thought: `[S=16 slots × R=4 resolutions × D=256 dims]`
//! - [`SlotData`] — multi-resolution embedding data for a single slot
//! - [`SlotRole`] — semantic role assignment for each slot
//! - [`SlotMeta`] — per-slot metadata (certainty, source, timestamp)
//! - [`FrameMeta`] — frame-level metadata (strand, discourse type, global certainty)
//! - [`VoltError`] — unified error type for the entire workspace
//!
//! ## Architecture Rules
//!
//! - `volt-core` may NOT import from any other `volt-*` crate.
//! - All cross-crate communication happens through [`TensorFrame`].
//! - No `async` code in this crate — pure synchronous logic.
//! - No `unwrap()` in library code — use `Result<T, VoltError>` everywhere.

pub mod error;
pub mod frame;
pub mod meta;
pub mod module_info;
pub mod slot;

pub use error::VoltError;
pub use frame::TensorFrame;
pub use meta::FrameMeta;
pub use module_info::{ModuleInfo, ModuleType};
pub use slot::{SlotData, SlotMeta, SlotRole};

/// Maximum number of slots in a TensorFrame.
///
/// 16 slots covers all semantic roles (Agent, Predicate, Patient, etc.)
/// with 7 free slots for domain-specific extensions.
///
/// # Example
///
/// ```
/// assert_eq!(volt_core::MAX_SLOTS, 16);
/// ```
pub const MAX_SLOTS: usize = 16;

/// Number of resolution levels in a TensorFrame.
///
/// - R₀: Discourse (coarsest — topic gist)
/// - R₁: Proposition (sentence-level)
/// - R₂: Phrase (detail-level)
/// - R₃: Token (finest — BPE subwords for output)
///
/// # Example
///
/// ```
/// assert_eq!(volt_core::NUM_RESOLUTIONS, 4);
/// ```
pub const NUM_RESOLUTIONS: usize = 4;

/// Dimensionality of each slot embedding vector.
///
/// 256 dims balances expressiveness with compute cost.
/// Total max Frame size: 16 × 4 × 256 × 4 bytes = 64KB.
///
/// # Example
///
/// ```
/// assert_eq!(volt_core::SLOT_DIM, 256);
/// ```
pub const SLOT_DIM: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_correct() {
        assert_eq!(MAX_SLOTS, 16);
        assert_eq!(NUM_RESOLUTIONS, 4);
        assert_eq!(SLOT_DIM, 256);
    }

    #[test]
    fn max_frame_size_is_64kb() {
        let max_bytes = MAX_SLOTS * NUM_RESOLUTIONS * SLOT_DIM * std::mem::size_of::<f32>();
        assert_eq!(max_bytes, 65536); // 64KB
    }

    #[test]
    fn default_frame_is_empty() {
        let frame = TensorFrame::default();
        assert!(frame.is_empty());
        assert_eq!(frame.active_slot_count(), 0);
    }
}
