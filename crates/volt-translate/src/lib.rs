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
//! - Action cores implement the [`ActionCore`](action_core::ActionCore) trait.
//! - Depends on `volt-core`, `volt-bus`, `volt-db`.

pub mod action_core;
pub mod decode;
pub mod encode;
pub mod stub;

#[cfg(feature = "llm")]
pub mod llm;

#[cfg(feature = "code-training")]
pub mod code_encoder;
#[cfg(feature = "code-training")]
pub mod code_decoder;
#[cfg(feature = "code-training")]
pub mod learned;

pub use action_core::{ActionCore, ActionOutput, OutputModality, TextAction};
pub use stub::StubTranslator;
pub use volt_core;

#[cfg(feature = "llm")]
pub use llm::LlmTranslator;

#[cfg(feature = "code-training")]
pub use learned::LearnedTranslator;

use volt_core::{ModuleInfo, SlotRole, TensorFrame, VoltError};

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

    /// Decode each active slot individually.
    ///
    /// Returns a vec of `(slot_index, role, decoded_word)` tuples for
    /// every slot that has data. Used for debug output where a per-slot
    /// breakdown is needed.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::{StubTranslator, Translator};
    /// use volt_core::SlotRole;
    ///
    /// let t = StubTranslator::new();
    /// let output = t.encode("cat sat mat").unwrap();
    /// let slots = t.decode_slots(&output.frame).unwrap();
    /// assert_eq!(slots.len(), 3);
    /// assert_eq!(slots[0].1, SlotRole::Agent);
    /// ```
    fn decode_slots(
        &self,
        frame: &TensorFrame,
    ) -> Result<Vec<(usize, SlotRole, String)>, VoltError>;

    /// Optional metadata about this translator module.
    ///
    /// Returns `None` by default. Community modules should override
    /// this to provide introspectable metadata for the module registry.
    fn info(&self) -> Option<ModuleInfo> {
        None
    }
}
