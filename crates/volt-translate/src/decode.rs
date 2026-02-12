//! Reverse decoding: TensorFrame to text.
//!
//! Provides vocabulary-based nearest-neighbor lookup and template-based
//! output formatting for the stub translator.

use volt_bus::similarity;
use volt_core::{SlotRole, SLOT_DIM};

/// A vocabulary entry mapping a word to its vector representation.
#[derive(Debug, Clone)]
pub struct VocabEntry {
    /// The original word string.
    pub word: String,
    /// The deterministic vector for this word.
    pub vector: [f32; SLOT_DIM],
}

/// Find the closest word in the vocabulary for a given slot vector.
///
/// Returns the word with the highest cosine similarity above `threshold`,
/// or `None` if no match exceeds the threshold.
///
/// # Example
///
/// ```
/// use volt_translate::decode::{VocabEntry, nearest_word};
/// use volt_translate::encode::word_to_vector;
///
/// let vocab = vec![
///     VocabEntry { word: "cat".into(), vector: word_to_vector("cat") },
///     VocabEntry { word: "dog".into(), vector: word_to_vector("dog") },
/// ];
/// let query = word_to_vector("cat");
/// let result = nearest_word(&query, &vocab, 0.5);
/// assert_eq!(result, Some("cat".to_string()));
/// ```
pub fn nearest_word(
    vector: &[f32; SLOT_DIM],
    vocabulary: &[VocabEntry],
    threshold: f32,
) -> Option<String> {
    let mut best_word: Option<&str> = None;
    let mut best_sim: f32 = threshold;

    for entry in vocabulary {
        let sim = similarity(vector, &entry.vector);
        if sim > best_sim {
            best_sim = sim;
            best_word = Some(&entry.word);
        }
    }

    best_word.map(|s| s.to_string())
}

/// Format decoded slot words into a human-readable sentence.
///
/// Special handling:
/// - If a Result slot (slot 8) is present, returns ONLY that value without a period.
///   This indicates a Hard Strand computation result (e.g., math engine).
/// - Otherwise, uses the pattern: "agent predicate patient." for the three core roles.
/// - Falls back to listing all words if the standard roles are missing.
///
/// # Example
///
/// ```
/// use volt_core::SlotRole;
/// use volt_translate::decode::format_output;
///
/// let words = vec![
///     (0, SlotRole::Agent, "cat".to_string()),
///     (1, SlotRole::Predicate, "sat".to_string()),
///     (2, SlotRole::Patient, "mat".to_string()),
/// ];
/// let text = format_output(&words);
/// assert_eq!(text, "cat sat mat.");
/// ```
pub fn format_output(slot_words: &[(usize, SlotRole, String)]) -> String {
    // Check if there's a Result slot (slot 8) — indicates Hard Strand output
    let result_slot = slot_words
        .iter()
        .find(|(slot_idx, role, _)| *slot_idx == 8 && *role == SlotRole::Result);

    if let Some((_, _, result_word)) = result_slot {
        // Hard Strand result: return just the computed value, no period
        return result_word.clone();
    }

    // Standard Agent/Predicate/Patient sentence construction
    let agent = slot_words
        .iter()
        .find(|(_, role, _)| *role == SlotRole::Agent);
    let predicate = slot_words
        .iter()
        .find(|(_, role, _)| *role == SlotRole::Predicate);
    let patient = slot_words
        .iter()
        .find(|(_, role, _)| *role == SlotRole::Patient);

    // Build the core sentence from Agent/Predicate/Patient
    let mut parts: Vec<&str> = Vec::new();
    if let Some((_, _, w)) = agent {
        parts.push(w);
    }
    if let Some((_, _, w)) = predicate {
        parts.push(w);
    }
    if let Some((_, _, w)) = patient {
        parts.push(w);
    }

    // Append any remaining words not in the core three roles
    for (_, role, word) in slot_words {
        if *role != SlotRole::Agent && *role != SlotRole::Predicate && *role != SlotRole::Patient {
            parts.push(word);
        }
    }

    if parts.is_empty() {
        "[empty frame]".to_string()
    } else {
        format!("{}.", parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::word_to_vector;

    #[test]
    fn nearest_word_finds_exact_match() {
        let vocab = vec![
            VocabEntry {
                word: "cat".into(),
                vector: word_to_vector("cat"),
            },
            VocabEntry {
                word: "dog".into(),
                vector: word_to_vector("dog"),
            },
        ];
        let query = word_to_vector("cat");
        assert_eq!(nearest_word(&query, &vocab, 0.5), Some("cat".to_string()));
    }

    #[test]
    fn nearest_word_returns_none_below_threshold() {
        let vocab = vec![VocabEntry {
            word: "cat".into(),
            vector: word_to_vector("cat"),
        }];
        let query = word_to_vector("dog");
        assert_eq!(nearest_word(&query, &vocab, 0.9), None);
    }

    #[test]
    fn nearest_word_empty_vocab() {
        let query = word_to_vector("cat");
        assert_eq!(nearest_word(&query, &[], 0.0), None);
    }

    #[test]
    fn format_output_agent_predicate_patient() {
        let words = vec![
            (0, SlotRole::Agent, "cat".into()),
            (1, SlotRole::Predicate, "sat".into()),
            (2, SlotRole::Patient, "mat".into()),
        ];
        assert_eq!(format_output(&words), "cat sat mat.");
    }

    #[test]
    fn format_output_with_extra_roles() {
        let words = vec![
            (0, SlotRole::Agent, "cat".into()),
            (1, SlotRole::Predicate, "sat".into()),
            (2, SlotRole::Patient, "mat".into()),
            (3, SlotRole::Location, "on".into()),
        ];
        assert_eq!(format_output(&words), "cat sat mat on.");
    }

    #[test]
    fn format_output_empty() {
        assert_eq!(format_output(&[]), "[empty frame]");
    }

    #[test]
    fn format_output_result_slot_only() {
        // When a Result slot (slot 8) is present, return only that value without period
        let words = vec![
            (1, SlotRole::Predicate, "[slot1]".into()),
            (6, SlotRole::Instrument, "[slot6]".into()),
            (8, SlotRole::Result, "15".into()),
        ];
        assert_eq!(format_output(&words), "15");
    }

    #[test]
    fn format_output_result_slot_with_decimal() {
        let words = vec![
            (8, SlotRole::Result, "3.1416".into()),
        ];
        assert_eq!(format_output(&words), "3.1416");
    }
}
