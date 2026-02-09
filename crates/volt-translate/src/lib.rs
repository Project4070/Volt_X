//! # volt-translate
//!
//! Input/output translators for Volt X.
//!
//! Translators convert between external modalities and TensorFrames:
//! - **Forward Translator**: NL text -> TensorFrame (encode)
//! - **Reverse Translator**: TensorFrame -> NL text (decode)
//!
//! ## Current Implementation
//!
//! Milestone 1.3 provides a [`StubTranslator`] that uses heuristic
//! word-to-slot mapping with deterministic hash-based vectors.
//! No ML. Words are assigned to semantic role slots by position.
//!
//! ## Architecture Rules
//!
//! - Translators implement the [`Translator`] trait.
//! - Depends on `volt-core`, `volt-bus`, `volt-db`.

pub mod decode;
pub mod encode;
pub mod stub;

pub use stub::StubTranslator;
pub use volt_core;

use volt_core::{TensorFrame, VoltError};

/// Output of a forward translation (text -> frame).
///
/// Contains the resulting [`TensorFrame`] plus metadata about
/// how many tokens were processed and slots filled.
///
/// # Example
///
/// ```
/// use volt_translate::{StubTranslator, Translator};
///
/// let t = StubTranslator::new();
/// let output = t.encode("hello world").unwrap();
/// assert_eq!(output.token_count, 2);
/// assert_eq!(output.slots_filled, 2);
/// ```
#[derive(Debug, Clone)]
pub struct TranslateOutput {
    /// The resulting TensorFrame.
    pub frame: TensorFrame,
    /// Number of words/tokens processed from input.
    pub token_count: usize,
    /// Number of slots filled in the frame.
    pub slots_filled: usize,
}

/// Trait for translating between external modalities and TensorFrames.
///
/// Implementors convert raw input into TensorFrames (encode) and
/// TensorFrames back into human-readable output (decode).
///
/// # Example
///
/// ```
/// use volt_translate::{StubTranslator, Translator};
///
/// let t = StubTranslator::new();
/// let output = t.encode("cat sat mat").unwrap();
/// let text = t.decode(&output.frame).unwrap();
/// assert!(!text.is_empty());
/// ```
pub trait Translator {
    /// Encode raw text input into a TensorFrame.
    ///
    /// Returns a [`TranslateOutput`] containing the frame and metadata.
    /// Errors if input is empty, too large, or otherwise invalid.
    fn encode(&self, input: &str) -> Result<TranslateOutput, VoltError>;

    /// Decode a TensorFrame back into human-readable text.
    ///
    /// Returns a string representation of the frame contents.
    fn decode(&self, frame: &TensorFrame) -> Result<String, VoltError>;
}
