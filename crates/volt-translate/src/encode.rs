//! Forward encoding: text to word vectors to TensorFrame slots.
//!
//! Provides deterministic word-to-vector encoding via hash-based mixing.
//! The same word always produces the same 256-dim normalized vector.

use volt_core::SLOT_DIM;

/// Maximum number of input bytes allowed.
pub const MAX_INPUT_BYTES: usize = 10_000;

/// Generate a deterministic, normalized 256-dim vector from a word string.
///
/// Uses FNV-1a hashing to produce a seed, then a hash-mixing function
/// (consistent with volt-bus test vectors) to produce pseudo-random
/// components, then L2-normalizes to unit length.
///
/// # Example
///
/// ```
/// use volt_translate::encode::word_to_vector;
/// use volt_core::SLOT_DIM;
///
/// let v1 = word_to_vector("cat");
/// let v2 = word_to_vector("cat");
/// assert_eq!(v1, v2); // deterministic
///
/// let v3 = word_to_vector("dog");
/// // Different words produce different vectors
/// let sim: f32 = v1.iter().zip(v3.iter()).map(|(a, b)| a * b).sum();
/// assert!(sim.abs() < 0.3);
/// ```
pub fn word_to_vector(word: &str) -> [f32; SLOT_DIM] {
    let seed = hash_word(word);
    seed_to_vector(seed)
}

/// Hash a word string to a u64 seed using FNV-1a.
fn hash_word(word: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in word.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

/// Convert a u64 seed to a normalized 256-dim vector.
///
/// Uses the same hash-mixing algorithm as volt-bus for consistency.
fn seed_to_vector(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0f32; SLOT_DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut h = seed.wrapping_mul(0xd2b74407b1ce6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        *slot = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    // L2 normalize to unit length
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Split input text into lowercase words, filtering empty tokens.
///
/// # Example
///
/// ```
/// use volt_translate::encode::tokenize;
///
/// let words = tokenize("The Cat  sat");
/// assert_eq!(words, vec!["the", "cat", "sat"]);
/// ```
pub fn tokenize(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_to_vector_is_deterministic() {
        let v1 = word_to_vector("hello");
        let v2 = word_to_vector("hello");
        assert_eq!(v1, v2);
    }

    #[test]
    fn word_to_vector_is_normalized() {
        let v = word_to_vector("test");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[test]
    fn different_words_produce_different_vectors() {
        let v1 = word_to_vector("cat");
        let v2 = word_to_vector("dog");
        let sim: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(sim.abs() < 0.3, "sim = {} (should be near 0)", sim);
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        let words = tokenize("The  Cat SAT");
        assert_eq!(words, vec!["the", "cat", "sat"]);
    }

    #[test]
    fn tokenize_empty_string() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   ").is_empty());
    }
}
