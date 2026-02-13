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

    /// Try to encode input as a math expression.
    /// Returns Some(output) if it's a math expression, None otherwise.
    fn try_encode_math(&self, input: &str) -> Result<Option<TranslateOutput>, VoltError> {
        let trimmed = input.trim();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        // Pattern: "number operator number" (e.g., "10 + 5")
        if parts.len() == 3 {
            let left = parts[0].parse::<f32>();
            let op = parts[1];
            let right = parts[2].parse::<f32>();

            if let (Ok(a), Ok(b)) = (left, right) {
                let op_code = match op {
                    "+" => Some(1.0_f32), // OP_ADD
                    "-" => Some(2.0_f32), // OP_SUB
                    "*" | "×" => Some(3.0_f32), // OP_MUL
                    "/" | "÷" => Some(4.0_f32), // OP_DIV
                    "^" | "**" => Some(5.0_f32), // OP_POW
                    _ => None,
                };

                if let Some(code) = op_code {
                    return Ok(Some(self.encode_math_operation(code, a, b)?));
                }
            }
        }

        Ok(None)
    }

    /// Encode a math operation into slot 6 (Instrument) format.
    ///
    /// Uses the math engine's capability vector as a "tag" in slot 1 (Predicate)
    /// to trigger routing, and encodes the actual operation in slot 6 (Instrument).
    fn encode_math_operation(&self, op_code: f32, left: f32, right: f32) -> Result<TranslateOutput, VoltError> {
        use volt_core::{SlotData, SlotMeta};

        let mut frame = TensorFrame::new();

        // CRITICAL: Tag with math capability vector in slot 1 (Predicate)
        // This triggers the Intent Router to route to math_engine.
        // The vector MUST match math_engine's build_capability_vector().
        let math_cap = Self::build_math_capability_vector();
        let mut predicate = SlotData::new(SlotRole::Predicate);
        predicate.write_resolution(0, math_cap);
        frame.slots[1] = Some(predicate);
        frame.meta[1] = SlotMeta {
            certainty: 0.9, // High certainty for exact match
            source: SlotSource::Translator,
            updated_at: 0,
            needs_verify: false,
        };

        // Encode operation data into slot 6 (Instrument)
        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = op_code;
        data[1] = left;
        data[2] = right;
        instrument.write_resolution(0, data);
        frame.slots[6] = Some(instrument);
        frame.meta[6] = SlotMeta {
            certainty: 1.0, // Math operations are certain
            source: SlotSource::Translator,
            updated_at: 0,
            needs_verify: false,
        };

        frame.frame_meta.discourse_type = DiscourseType::Query;
        frame.frame_meta.strand_id = 0;

        Ok(TranslateOutput {
            frame,
            token_count: 3,
            slots_filled: 2, // Both slot 1 and slot 6
        })
    }

    /// Build the same capability vector as MathEngine for routing.
    ///
    /// This MUST match the algorithm in volt-hard/src/math_engine.rs
    /// build_capability_vector() exactly, or routing will fail.
    fn build_math_capability_vector() -> [f32; SLOT_DIM] {
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
}

impl Clone for StubTranslator {
    fn clone(&self) -> Self {
        // Clone the vocabulary; if the lock is poisoned, start with empty vocab
        let vocab = self
            .vocab
            .read()
            .map(|v| v.clone())
            .unwrap_or_default();
        Self {
            vocab: RwLock::new(vocab),
        }
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

        // Check for math expressions first
        if let Some(output) = self.try_encode_math(input)? {
            return Ok(output);
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

            // Write at both R0 (discourse) and R1 (proposition) so the
            // frame is indexable by the HNSW gist extractor (which reads
            // R0) AND decodable by decode_slots (which prefers R1).
            frame.write_at(i, 0, role, vector)?;
            if let Some(slot) = &mut frame.slots[i] {
                slot.resolutions[1] = Some(vector);
            }

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
                // Special handling for slot 8 (Result) — decode numeric result
                if i == 8
                    && slot_data.role == SlotRole::Result
                    && let Some(vec) = slot_data.resolutions[0].as_ref()
                {
                    // Math engine writes result in dim[0], valid flag in dim[1]
                    let result_value = vec[0];
                    let valid_flag = vec[1];
                    if valid_flag > 0.5 {
                        let word = if result_value.fract().abs() < 0.0001 {
                            // Integer result
                            format!("{}", result_value as i64)
                        } else {
                            // Floating point result
                            format!("{:.4}", result_value)
                        };
                        slot_words.push((i, slot_data.role, word));
                        continue;
                    }
                }

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
