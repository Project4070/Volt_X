//! Bloom filter for fast negative membership checks on LSM sorted runs.
//!
//! Uses double hashing from a `u64` key to avoid external dependencies.
//! Optimal bit count and hash count are computed from the expected number
//! of items and desired false positive rate.

use volt_core::VoltError;

/// A Bloom filter for fast negative membership checks.
///
/// Used on T2 sorted runs to skip runs that definitely don't contain
/// a given `frame_id`, avoiding unnecessary disk reads.
///
/// # Example
///
/// ```
/// use volt_db::bloom::BloomFilter;
///
/// let mut bloom = BloomFilter::new(1000, 0.01);
/// bloom.insert(42);
/// bloom.insert(99);
///
/// assert!(bloom.may_contain(42));
/// assert!(bloom.may_contain(99));
/// // May or may not contain 123 (possible false positive)
/// ```
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array stored as u64 words.
    bits: Vec<u64>,
    /// Total number of bits.
    num_bits: usize,
    /// Number of hash functions (k).
    num_hashes: u32,
}

impl BloomFilter {
    /// Creates a new Bloom filter sized for `expected_items` with the
    /// given `false_positive_rate`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::bloom::BloomFilter;
    ///
    /// let bloom = BloomFilter::new(10_000, 0.001);
    /// assert!(!bloom.may_contain(42));
    /// ```
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let expected = expected_items.max(1) as f64;
        let fp = false_positive_rate.clamp(1e-10, 0.999);

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let num_bits_f = -(expected * fp.ln()) / (2.0_f64.ln().powi(2));
        let num_bits = (num_bits_f as usize).max(64);

        // Optimal number of hashes: k = (m/n) * ln(2)
        let num_hashes_f = (num_bits as f64 / expected) * 2.0_f64.ln();
        let num_hashes = (num_hashes_f as u32).clamp(1, 30);

        // Round up to multiple of 64 bits
        let words = num_bits.div_ceil(64);
        let actual_bits = words * 64;

        Self {
            bits: vec![0u64; words],
            num_bits: actual_bits,
            num_hashes,
        }
    }

    /// Inserts a key into the Bloom filter.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::bloom::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::new(100, 0.01);
    /// bloom.insert(42);
    /// assert!(bloom.may_contain(42));
    /// ```
    pub fn insert(&mut self, key: u64) {
        let (h1, h2) = Self::hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            let word = idx / 64;
            let bit = idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Tests whether a key *may* be in the set.
    ///
    /// Returns `false` if the key is definitely not present.
    /// Returns `true` if the key might be present (possible false positive).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::bloom::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::new(100, 0.01);
    /// assert!(!bloom.may_contain(42));
    /// bloom.insert(42);
    /// assert!(bloom.may_contain(42));
    /// ```
    pub fn may_contain(&self, key: u64) -> bool {
        let (h1, h2) = Self::hash_pair(key);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            let word = idx / 64;
            let bit = idx % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Returns the number of bits in this filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Returns the number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Serializes the Bloom filter to bytes.
    ///
    /// Format: `[num_bits: u64][num_hashes: u32][padding: 4B][bit_data]`
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::bloom::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::new(100, 0.01);
    /// bloom.insert(42);
    /// let bytes = bloom.to_bytes();
    /// let restored = BloomFilter::from_bytes(&bytes).unwrap();
    /// assert!(restored.may_contain(42));
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.bits.len() * 8);
        buf.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        buf.extend_from_slice(&self.num_hashes.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // padding for alignment
        for &word in &self.bits {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    /// Deserializes a Bloom filter from bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the data is too short or invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VoltError> {
        if bytes.len() < 16 {
            return Err(VoltError::StorageError {
                message: "bloom filter data too short".to_string(),
            });
        }
        let num_bits = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let num_hashes = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        // bytes[12..16] is padding

        let words = num_bits.div_ceil(64);
        let expected_len = 16 + words * 8;
        if bytes.len() < expected_len {
            return Err(VoltError::StorageError {
                message: format!(
                    "bloom filter data too short: expected {expected_len}, got {}",
                    bytes.len()
                ),
            });
        }

        let mut bits = Vec::with_capacity(words);
        for i in 0..words {
            let offset = 16 + i * 8;
            let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            bits.push(word);
        }

        Ok(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Merges another Bloom filter into this one (union).
    ///
    /// Both filters must have the same parameters.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the filters have different sizes.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::bloom::BloomFilter;
    ///
    /// let mut a = BloomFilter::new(100, 0.01);
    /// a.insert(1);
    ///
    /// let mut b = BloomFilter::new(100, 0.01);
    /// b.insert(2);
    ///
    /// a.union(&b).unwrap();
    /// assert!(a.may_contain(1));
    /// assert!(a.may_contain(2));
    /// ```
    pub fn union(&mut self, other: &Self) -> Result<(), VoltError> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(VoltError::StorageError {
                message: "cannot merge bloom filters with different parameters".to_string(),
            });
        }
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        Ok(())
    }

    /// Double-hashing: produce two independent hash values from a u64 key.
    ///
    /// Uses splitmix64 (Murmur3-style finalizer) for excellent distribution
    /// of sequential keys. `h2` is forced odd to ensure good cycle coverage
    /// in the double-hashing scheme.
    fn hash_pair(key: u64) -> (u64, u64) {
        // splitmix64 finalizer for h1
        let mut h1 = key;
        h1 ^= h1 >> 30;
        h1 = h1.wrapping_mul(0xbf58476d1ce4e5b9);
        h1 ^= h1 >> 27;
        h1 = h1.wrapping_mul(0x94d049bb133111eb);
        h1 ^= h1 >> 31;

        // Independent hash for h2 (different seed)
        let mut h2 = key.wrapping_add(0x9e3779b97f4a7c15);
        h2 ^= h2 >> 30;
        h2 = h2.wrapping_mul(0xbf58476d1ce4e5b9);
        h2 ^= h2 >> 27;
        h2 = h2.wrapping_mul(0x94d049bb133111eb);
        h2 ^= h2 >> 31;

        // Force h2 odd so bit_index cycles cover the full range
        h2 |= 1;

        (h1, h2)
    }

    /// Compute the i-th bit index using double hashing.
    fn bit_index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined % self.num_bits as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut bloom = BloomFilter::new(1000, 0.01);
        let keys: Vec<u64> = (0..1000).collect();
        for &k in &keys {
            bloom.insert(k);
        }
        for &k in &keys {
            assert!(bloom.may_contain(k), "key {k} should be found");
        }
    }

    #[test]
    fn false_positive_rate() {
        let n = 10_000;
        let expected_fp = 0.01;
        let mut bloom = BloomFilter::new(n, expected_fp);

        // Insert keys 0..n
        for k in 0..n as u64 {
            bloom.insert(k);
        }

        // Test keys n..2n (none were inserted)
        let mut false_positives = 0;
        let test_count = 100_000;
        for k in n as u64..(n as u64 + test_count) {
            if bloom.may_contain(k) {
                false_positives += 1;
            }
        }

        let observed_fp = false_positives as f64 / test_count as f64;
        // Allow 3x the expected rate to account for randomness
        assert!(
            observed_fp < expected_fp * 3.0,
            "false positive rate {observed_fp:.4} too high (expected < {:.4})",
            expected_fp * 3.0
        );
    }

    #[test]
    fn empty_bloom_returns_false() {
        let bloom = BloomFilter::new(100, 0.01);
        for k in 0..100u64 {
            assert!(!bloom.may_contain(k));
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut bloom = BloomFilter::new(500, 0.001);
        for k in 0..500u64 {
            bloom.insert(k);
        }

        let bytes = bloom.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();

        assert_eq!(restored.num_bits(), bloom.num_bits());
        assert_eq!(restored.num_hashes(), bloom.num_hashes());

        // All inserted keys should still be found
        for k in 0..500u64 {
            assert!(restored.may_contain(k), "key {k} lost after roundtrip");
        }
    }

    #[test]
    fn from_bytes_too_short() {
        let result = BloomFilter::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn union_combines_filters() {
        let mut a = BloomFilter::new(100, 0.01);
        a.insert(1);
        a.insert(2);

        let mut b = BloomFilter::new(100, 0.01);
        b.insert(3);
        b.insert(4);

        a.union(&b).unwrap();
        assert!(a.may_contain(1));
        assert!(a.may_contain(2));
        assert!(a.may_contain(3));
        assert!(a.may_contain(4));
    }

    #[test]
    fn union_mismatched_sizes_fails() {
        let a = BloomFilter::new(100, 0.01);
        let b = BloomFilter::new(1000, 0.01);
        let mut a_mut = a;
        let result = a_mut.union(&b);
        assert!(result.is_err());
    }
}
