//! Stub text translator for Milestone 1.3.
//!
//! Heuristic word-to-slot mapping:
//! - Word 0 -> S0 (Agent)
//! - Word 1 -> S1 (Predicate)
//! - Word 2 -> S2 (Patient)
//! - Words 3+ -> S3+ (Location, Time, Manner, ...)
//!
//! Each word is encoded as a deterministic 256-dim vector via hash.
//! The translator maintains a vocabulary for reverse translation.

use std::sync::RwLock;

use volt_core::meta::DiscourseType;
use volt_core::slot::SlotSource;
use volt_core::{SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

use crate::decode::{format_output, nearest_word, VocabEntry};
use crate::encode::{tokenize, word_to_vector, MAX_INPUT_BYTES};
use crate::{TranslateOutput, Translator};

/// Stub text translator using heuristic word-to-slot mapping.
///
/// Each word is encoded as a deterministic 256-dim vector via hash.
/// Words are assigned to slots by position (word 0 = Agent, word 1 =
/// Predicate, word 2 = Patient, etc.). The translator maintains a
/// vocabulary for reverse translation via nearest-neighbor lookup.
///
/// # Example
///
/// ```
/// use volt_translate::{StubTranslator, Translator};
///
/// let translator = StubTranslator::new();
/// let output = translator.encode("the cat sat").unwrap();
/// assert_eq!(output.slots_filled, 3);
///
/// let decoded = translator.decode(&output.frame).unwrap();
/// assert!(!decoded.is_empty());
/// ```
pub struct StubTranslator {
    /// Vocabulary for reverse lookup (word -> vector).
    vocab: RwLock<Vec<VocabEntry>>,
}

impl std::fmt::Debug for StubTranslator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vocab_len = self
            .vocab
            .read()
            .map(|v| v.len())
            .unwrap_or(0);
        f.debug_struct("StubTranslator")
            .field("vocab_size", &vocab_len)
            .finish()
    }
}

impl StubTranslator {
    /// Create a new stub translator with an empty vocabulary.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::StubTranslator;
    ///
    /// let t = StubTranslator::new();
    /// ```
    pub fn new() -> Self {
        Self {
            vocab: RwLock::new(Vec::new()),
        }
    }

    /// Map a word position index to a [`SlotRole`].
    fn index_to_role(index: usize) -> SlotRole {
        match index {
            0 => SlotRole::Agent,
            1 => SlotRole::Predicate,
            2 => SlotRole::Patient,
            3 => SlotRole::Location,
            4 => SlotRole::Time,
            5 => SlotRole::Manner,
            6 => SlotRole::Instrument,
            7 => SlotRole::Cause,
            8 => SlotRole::Result,
            i => SlotRole::Free((i - 9) as u8),
        }
    }

    /// Add a word to the vocabulary if not already present.
    fn add_to_vocab(&self, word: &str, vector: [f32; SLOT_DIM]) -> Result<(), VoltError> {
        let mut vocab = self.vocab.write().map_err(|e| VoltError::TranslateError {
            message: format!("failed to acquire vocab write lock: {e}"),
        })?;
        if !vocab.iter().any(|entry| entry.word == word) {
            vocab.push(VocabEntry {
                word: word.to_string(),
                vector,
            });
        }
        Ok(())
    }
}

impl Default for StubTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl Translator for StubTranslator {
    fn encode(&self, input: &str) -> Result<TranslateOutput, VoltError> {
        if input.len() > MAX_INPUT_BYTES {
            return Err(VoltError::TranslateError {
                message: format!(
                    "input too large: {} bytes (max {MAX_INPUT_BYTES})",
                    input.len(),
                ),
            });
        }

        let words = tokenize(input);
        if words.is_empty() {
            return Err(VoltError::TranslateError {
                message: "input text is empty or contains no words".to_string(),
            });
        }

        let mut frame = TensorFrame::new();
        let slots_to_fill = words.len().min(MAX_SLOTS);

        for (i, word) in words.iter().take(slots_to_fill).enumerate() {
            let vector = word_to_vector(word);
            let role = Self::index_to_role(i);

            // Write at R1 (proposition level)
            frame.write_at(i, 1, role, vector)?;

            // Set slot metadata
            frame.meta[i].certainty = 0.8;
            frame.meta[i].source = SlotSource::Translator;
            frame.meta[i].needs_verify = true;

            // Add to vocabulary for reverse translation
            self.add_to_vocab(word, vector)?;
        }

        // Set frame metadata
        frame.frame_meta.discourse_type = classify_discourse(input);
        frame.frame_meta.rar_iterations = 0;
        frame.frame_meta.global_certainty = 0.8;

        Ok(TranslateOutput {
            frame,
            token_count: words.len(),
            slots_filled: slots_to_fill,
        })
    }

    fn decode(&self, frame: &TensorFrame) -> Result<String, VoltError> {
        let slot_words = self.decode_slots(frame)?;
        Ok(format_output(&slot_words))
    }

    fn decode_slots(
        &self,
        frame: &TensorFrame,
    ) -> Result<Vec<(usize, SlotRole, String)>, VoltError> {
        let vocab = self.vocab.read().map_err(|e| VoltError::TranslateError {
            message: format!("failed to acquire vocab read lock: {e}"),
        })?;

        let mut slot_words: Vec<(usize, SlotRole, String)> = Vec::new();

        for i in 0..MAX_SLOTS {
            if let Some(slot_data) = &frame.slots[i] {
                // Try R1 first (proposition level, where encode writes),
                // then fall back through other resolutions
                let vector = slot_data.resolutions[1]
                    .as_ref()
                    .or(slot_data.resolutions[0].as_ref())
                    .or(slot_data.resolutions[2].as_ref())
                    .or(slot_data.resolutions[3].as_ref());

                if let Some(vec) = vector {
                    let word = nearest_word(vec, &vocab, 0.5)
                        .unwrap_or_else(|| format!("[slot{i}]"));
                    slot_words.push((i, slot_data.role, word));
                }
            }
        }

        Ok(slot_words)
    }
}

/// Classify the discourse type from input text punctuation.
fn classify_discourse(input: &str) -> DiscourseType {
    let trimmed = input.trim();
    if trimmed.ends_with('?') {
        DiscourseType::Query
    } else if trimmed.ends_with('!') {
        DiscourseType::Command
    } else {
        DiscourseType::Statement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_basic_sentence() {
        let t = StubTranslator::new();
        let output = t.encode("the cat sat").unwrap();
        assert_eq!(output.slots_filled, 3);
        assert_eq!(output.token_count, 3);

        let slot0 = output.frame.read_slot(0).unwrap();
        assert_eq!(slot0.role, SlotRole::Agent);

        let slot1 = output.frame.read_slot(1).unwrap();
        assert_eq!(slot1.role, SlotRole::Predicate);

        let slot2 = output.frame.read_slot(2).unwrap();
        assert_eq!(slot2.role, SlotRole::Patient);
    }

    #[test]
    fn encode_empty_errors() {
        let t = StubTranslator::new();
        assert!(t.encode("").is_err());
        assert!(t.encode("   ").is_err());
    }

    #[test]
    fn encode_huge_input_errors() {
        let t = StubTranslator::new();
        let huge = "a ".repeat(MAX_INPUT_BYTES);
        assert!(t.encode(&huge).is_err());
    }

    #[test]
    fn encode_sets_metadata() {
        let t = StubTranslator::new();
        let output = t.encode("hello world").unwrap();
        assert_eq!(output.frame.meta[0].certainty, 0.8);
        assert_eq!(output.frame.meta[0].source, SlotSource::Translator);
        assert!(output.frame.meta[0].needs_verify);
    }

    #[test]
    fn roundtrip_recovers_words() {
        let t = StubTranslator::new();
        let output = t.encode("cat sat mat").unwrap();
        let decoded = t.decode(&output.frame).unwrap();
        let lower = decoded.to_lowercase();
        assert!(lower.contains("cat"), "decoded: {decoded}");
        assert!(lower.contains("sat"), "decoded: {decoded}");
        assert!(lower.contains("mat"), "decoded: {decoded}");
    }

    #[test]
    fn decode_empty_frame() {
        let t = StubTranslator::new();
        let frame = TensorFrame::new();
        let decoded = t.decode(&frame).unwrap();
        assert_eq!(decoded, "[empty frame]");
    }

    #[test]
    fn classify_discourse_question() {
        assert_eq!(classify_discourse("what is this?"), DiscourseType::Query);
    }

    #[test]
    fn classify_discourse_command() {
        assert_eq!(classify_discourse("do it now!"), DiscourseType::Command);
    }

    #[test]
    fn classify_discourse_statement() {
        assert_eq!(classify_discourse("the sky is blue."), DiscourseType::Statement);
    }

    #[test]
    fn decode_slots_returns_per_slot_breakdown() {
        let t = StubTranslator::new();
        let output = t.encode("cat sat mat").unwrap();
        let slots = t.decode_slots(&output.frame).unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].0, 0);
        assert_eq!(slots[0].1, SlotRole::Agent);
        assert!(slots[0].2.contains("cat"), "expected 'cat', got '{}'", slots[0].2);
        assert_eq!(slots[1].0, 1);
        assert_eq!(slots[1].1, SlotRole::Predicate);
        assert_eq!(slots[2].0, 2);
        assert_eq!(slots[2].1, SlotRole::Patient);
    }

    #[test]
    fn decode_slots_empty_frame() {
        let t = StubTranslator::new();
        let frame = TensorFrame::new();
        let slots = t.decode_slots(&frame).unwrap();
        assert!(slots.is_empty());
    }

    #[test]
    fn index_to_role_mapping() {
        assert_eq!(StubTranslator::index_to_role(0), SlotRole::Agent);
        assert_eq!(StubTranslator::index_to_role(1), SlotRole::Predicate);
        assert_eq!(StubTranslator::index_to_role(2), SlotRole::Patient);
        assert_eq!(StubTranslator::index_to_role(9), SlotRole::Free(0));
        assert_eq!(StubTranslator::index_to_role(15), SlotRole::Free(6));
    }
}
